from __future__ import annotations

import argparse
import hashlib
import logging
import pickle
import re
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from pypdf import PdfReader
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import normalize

from pdf_speech_search.query_expansion import expand_query
from pdf_speech_search.settings import settings


INDEX_VERSION = 4
logging.getLogger("pypdf").setLevel(logging.ERROR)


@dataclass
class PageRecord:
    doc_id: str
    pdf_path: str
    pdf_name: str
    page: int
    text: str
    search_text: str


@dataclass
class PdfIndex:
    version: int
    built_at: float
    pdf_dir: str
    signature: list[tuple[str, int, int]]
    pages: list[PageRecord]
    word_vectorizer: Any
    word_matrix: Any
    char_vectorizer: Any
    char_matrix: Any
    svd: Any | None
    lsa_matrix: Any | None
    semantic_model_name: str
    semantic_texts: list[str]
    semantic_matrix: np.ndarray

    @property
    def doc_map(self) -> dict[str, str]:
        return {page.doc_id: page.pdf_path for page in self.pages}


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def make_doc_id(path: Path) -> str:
    stem = re.sub(r"[^a-zA-Z0-9]+", "-", path.stem).strip("-").lower()[:40]
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:10]
    return f"{stem}-{digest}"


def pdf_signature(pdf_paths: list[Path]) -> list[tuple[str, int, int]]:
    return [
        (str(path.resolve()), path.stat().st_size, int(path.stat().st_mtime))
        for path in sorted(pdf_paths)
    ]


def find_pdfs(pdf_dir: Path) -> list[Path]:
    if not pdf_dir.exists():
        raise FileNotFoundError(f"PDF directory does not exist: {pdf_dir}")
    return sorted(path for path in pdf_dir.glob("*.pdf") if path.is_file())


def extract_pdf_pages(pdf_path: Path) -> list[str]:
    reader = PdfReader(str(pdf_path))
    pages: list[str] = []
    for page in reader.pages:
        try:
            text = page.extract_text() or ""
        except Exception as exc:  # pragma: no cover - depends on individual PDF structure
            text = f"[text extraction failed: {exc}]"
        pages.append(normalize_text(text))
    return pages


def build_page_records(pdf_dir: Path) -> tuple[list[PageRecord], list[tuple[str, int, int]]]:
    pdf_paths = find_pdfs(pdf_dir)
    signature = pdf_signature(pdf_paths)
    records: list[PageRecord] = []

    for pdf_path in pdf_paths:
        doc_id = make_doc_id(pdf_path)
        page_texts = extract_pdf_pages(pdf_path)
        for idx, text in enumerate(page_texts):
            prev_text = page_texts[idx - 1] if idx > 0 else ""
            next_text = page_texts[idx + 1] if idx + 1 < len(page_texts) else ""
            search_parts = [
                f"Lecture file: {pdf_path.stem}. Page {idx + 1}.",
                text,
                "Previous slide context:",
                prev_text[:1800],
                "Next slide context:",
                next_text[:1800],
            ]
            records.append(
                PageRecord(
                    doc_id=doc_id,
                    pdf_path=str(pdf_path.resolve()),
                    pdf_name=pdf_path.name,
                    page=idx + 1,
                    text=text,
                    search_text=normalize_text("\n".join(part for part in search_parts if part)),
                )
            )

    return records, signature


@lru_cache(maxsize=4)
def get_embedding_model(model_name: str) -> Any:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, local_files_only=settings.model_local_files_only)


@lru_cache(maxsize=2)
def get_reranker_model(model_name: str) -> Any:
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name, local_files_only=settings.model_local_files_only)


def embedding_text(record: PageRecord) -> str:
    own_text = record.text.strip()
    if len(own_text) < 30:
        own_text = record.search_text
    return normalize_text(
        f"Lecture PDF: {record.pdf_name}\nSlide page: {record.page}\n{own_text}"
    )


def rerank_text(record: PageRecord) -> str:
    own_text = record.text.strip()
    if len(own_text) >= 60:
        return normalize_text(own_text)
    return normalize_text(record.search_text)


def encode_documents(texts: list[str], model_name: str) -> np.ndarray:
    model = get_embedding_model(model_name)
    embeddings = model.encode(
        texts,
        batch_size=settings.embedding_batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=len(texts) > 100,
    )
    return np.asarray(embeddings, dtype=np.float32)


def encode_query(query: str, model_name: str) -> np.ndarray:
    model = get_embedding_model(model_name)
    query_text = settings.semantic_query_prefix + expand_query(query)
    embedding = model.encode(
        [query_text],
        batch_size=1,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )[0]
    return np.asarray(embedding, dtype=np.float32)


def fit_index(records: list[PageRecord], pdf_dir: Path, signature: list[tuple[str, int, int]]) -> PdfIndex:
    if not records:
        raise ValueError(f"No PDF pages found in {pdf_dir}")

    corpus = [record.search_text or record.pdf_name for record in records]
    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 3),
        stop_words="english",
        min_df=1,
        max_df=0.92,
        sublinear_tf=True,
        strip_accents="unicode",
        max_features=90000,
    )
    word_matrix = normalize(word_vectorizer.fit_transform(corpus), copy=False)

    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
        max_df=0.95,
        sublinear_tf=True,
        max_features=50000,
    )
    char_matrix = normalize(char_vectorizer.fit_transform(corpus), copy=False)

    svd = None
    lsa_matrix = None
    max_components = min(word_matrix.shape[0] - 1, word_matrix.shape[1] - 1, 256)
    if max_components >= 2:
        svd = TruncatedSVD(n_components=max_components, random_state=7)
        lsa_matrix = normalize(svd.fit_transform(word_matrix), copy=False)

    semantic_texts = [embedding_text(record) for record in records]
    semantic_matrix = encode_documents(semantic_texts, settings.semantic_model)

    return PdfIndex(
        version=INDEX_VERSION,
        built_at=time.time(),
        pdf_dir=str(pdf_dir.resolve()),
        signature=signature,
        pages=records,
        word_vectorizer=word_vectorizer,
        word_matrix=word_matrix,
        char_vectorizer=char_vectorizer,
        char_matrix=char_matrix,
        svd=svd,
        lsa_matrix=lsa_matrix,
        semantic_model_name=settings.semantic_model,
        semantic_texts=semantic_texts,
        semantic_matrix=semantic_matrix,
    )


def build_index(pdf_dir: Path = settings.pdf_dir, index_path: Path = settings.index_path) -> PdfIndex:
    records, signature = build_page_records(pdf_dir)
    index = fit_index(records, pdf_dir, signature)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("wb") as handle:
        pickle.dump(index, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return index


def load_index(index_path: Path = settings.index_path) -> PdfIndex:
    with index_path.open("rb") as handle:
        index = pickle.load(handle)
    if not isinstance(index, PdfIndex) or index.version != INDEX_VERSION:
        raise ValueError("Index version mismatch")
    return index


def index_is_current(index: PdfIndex, pdf_dir: Path = settings.pdf_dir) -> bool:
    try:
        return index.signature == pdf_signature(find_pdfs(pdf_dir))
    except FileNotFoundError:
        return False


def load_or_build_index(
    pdf_dir: Path = settings.pdf_dir,
    index_path: Path = settings.index_path,
    force: bool = False,
) -> PdfIndex:
    if not force and index_path.exists():
        try:
            index = load_index(index_path)
            if index_is_current(index, pdf_dir):
                return index
        except Exception:
            pass
    return build_index(pdf_dir, index_path)


def token_set(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9][a-z0-9+\-_.]*", text.lower()))


def best_snippet(text: str, query: str, limit: int = 520) -> str:
    clean = normalize_text(text)
    if not clean:
        return "No extractable text on this page."

    query_terms = token_set(expand_query(query))
    lines = re.split(r"(?<=[.!?])\s+|\n+", clean)
    candidates = [line.strip() for line in lines if line.strip()]
    if not candidates:
        return clean[:limit]

    def sentence_score(sentence: str) -> tuple[int, int]:
        terms = token_set(sentence)
        overlap = len(query_terms & terms)
        acronym_hits = sum(1 for term in query_terms if len(term) <= 5 and term in sentence.lower())
        return overlap + acronym_hits * 2, len(sentence)

    best = max(candidates, key=sentence_score)
    if len(best) < 140 and len(candidates) > 1:
        idx = candidates.index(best)
        neighbors = candidates[max(0, idx - 1) : min(len(candidates), idx + 2)]
        best = " ".join(neighbors)

    if len(best) <= limit:
        return best
    return best[: limit - 1].rsplit(" ", 1)[0] + "..."


def minmax(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return values
    low = float(np.min(values))
    high = float(np.max(values))
    if high - low < 1e-9:
        return np.zeros_like(values, dtype=float)
    return (values - low) / (high - low)


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -30, 30)
    return 1.0 / (1.0 + np.exp(-clipped))


def search_index(index: PdfIndex, query: str, top_k: int = 5) -> list[dict[str, Any]]:
    query = query.strip()
    if not query:
        return []

    expanded = expand_query(query)
    word_query = normalize(index.word_vectorizer.transform([expanded]), copy=False)
    char_query = normalize(index.char_vectorizer.transform([expanded]), copy=False)
    semantic_query = encode_query(query, index.semantic_model_name)

    word_scores = linear_kernel(word_query, index.word_matrix).ravel()
    char_scores = linear_kernel(char_query, index.char_matrix).ravel()
    semantic_scores = index.semantic_matrix @ semantic_query
    if index.svd is not None and index.lsa_matrix is not None:
        lsa_query = normalize(index.svd.transform(word_query), copy=False)
        lsa_scores = linear_kernel(lsa_query, index.lsa_matrix).ravel()
    else:
        lsa_scores = word_scores

    query_terms = token_set(expanded)
    original_terms = token_set(query)
    original_acronyms = [term.lower() for term in re.findall(r"\b[A-Z][A-Z0-9]{1,}\b", query)]
    original_tokens = re.findall(r"[a-z0-9]+", query.lower())
    query_phrases = {
        " ".join(original_tokens[i : i + size])
        for size in (2, 3)
        for i in range(0, max(0, len(original_tokens) - size + 1))
    }
    boosts = np.zeros(len(index.pages), dtype=float)
    for i, page in enumerate(index.pages):
        page_terms = token_set(page.search_text)
        own_terms = token_set(page.text)
        context_overlap = len(query_terms & page_terms)
        own_overlap = len(original_terms & own_terms)
        context_coverage = context_overlap / max(len(query_terms), 1)
        own_coverage = own_overlap / max(len(original_terms), 1)

        own_text_lower = page.text.lower()
        exact_phrase = 0.05 if query.lower() in own_text_lower else 0.0
        phrase_hits = sum(1 for phrase in query_phrases if len(phrase) > 5 and phrase in own_text_lower)
        acronym_hit = any(acronym in own_terms for acronym in original_acronyms)
        acronym_score = 0.10 if acronym_hit else (-0.08 if original_acronyms else 0.0)

        boosts[i] = (
            min(0.08, context_coverage * 0.08)
            + min(0.20, own_coverage * 0.22)
            + min(0.16, phrase_hits * 0.045)
            + exact_phrase
            + acronym_score
        )

    lexical_scores = 0.45 * lsa_scores + 0.35 * word_scores + 0.20 * char_scores
    base_scores = 0.62 * semantic_scores + 0.26 * lexical_scores + boosts
    if len(base_scores) == 0:
        return []

    top_k = max(1, min(top_k, len(index.pages)))
    candidate_count = min(
        len(index.pages),
        max(top_k * 6, settings.rerank_candidates if settings.enable_reranker else top_k),
    )
    candidate_indices = np.argpartition(base_scores, -candidate_count)[-candidate_count:]

    if settings.enable_reranker and settings.reranker_model:
        try:
            reranker = get_reranker_model(settings.reranker_model)
            pairs = [(query, rerank_text(index.pages[int(idx)])) for idx in candidate_indices]
            rerank_raw = reranker.predict(
                pairs,
                batch_size=settings.rerank_batch_size,
                show_progress_bar=False,
            )
            rerank_scores = sigmoid(np.asarray(rerank_raw, dtype=float))
            semantic_candidate_scores = minmax(semantic_scores[candidate_indices])
            base_candidate_scores = minmax(base_scores[candidate_indices])
            candidate_scores = (
                0.70 * rerank_scores
                + 0.20 * semantic_candidate_scores
                + 0.10 * base_candidate_scores
            )
        except Exception:
            candidate_scores = base_scores[candidate_indices]
    else:
        candidate_scores = base_scores[candidate_indices]

    ordered = np.argsort(candidate_scores)[::-1][:top_k]
    ranked_indices = candidate_indices[ordered]
    ranked_scores = candidate_scores[ordered]

    results: list[dict[str, Any]] = []
    for idx, score in zip(ranked_indices, ranked_scores, strict=True):
        page = index.pages[int(idx)]
        results.append(
            {
                "doc_id": page.doc_id,
                "pdf_name": page.pdf_name,
                "page": page.page,
                "score": float(score),
                "snippet": best_snippet(page.text, query),
                "url": f"/pdf/{page.doc_id}#page={page.page}",
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or inspect the PDF search index.")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuilding the index.")
    parser.add_argument("--pdf-dir", default=str(settings.pdf_dir), help="Directory containing PDFs.")
    parser.add_argument("--index-path", default=str(settings.index_path), help="Index pickle path.")
    parser.add_argument("--query", default="", help="Optional query to test after loading/building.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of test results.")
    args = parser.parse_args()

    pdf_dir = Path(args.pdf_dir)
    index_path = Path(args.index_path)
    action = "Using current"
    if args.rebuild:
        index = build_index(pdf_dir, index_path)
        action = "Rebuilt"
    else:
        try:
            index = load_index(index_path)
            if not index_is_current(index, pdf_dir):
                index = build_index(pdf_dir, index_path)
                action = "Built"
        except Exception:
            index = build_index(pdf_dir, index_path)
            action = "Built"

    print(
        f"{action} index: {len(index.pages)} pages from {len(index.doc_map)} PDFs -> {index_path.resolve()}"
    )
    if args.query:
        for result in search_index(index, args.query, args.top_k):
            print(f"{result['score']:.3f} {result['pdf_name']} p.{result['page']}: {result['snippet']}")


if __name__ == "__main__":
    main()
