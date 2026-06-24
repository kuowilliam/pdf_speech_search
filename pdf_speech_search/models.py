from __future__ import annotations

import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser(description="Download local ASR/search models.")
    parser.add_argument("--skip-search", action="store_true", help="Do not warm embedding/reranker models.")
    parser.add_argument("--skip-asr", action="store_true", help="Do not warm the selected ASR model.")
    parser.add_argument("--asr-model", default=None, help="ASR model id to download and warm.")
    parser.add_argument("--semantic-model", default=None, help="Override SEMANTIC_MODEL for this run.")
    parser.add_argument("--reranker-model", default=None, help="Override RERANKER_MODEL for this run.")
    args = parser.parse_args()

    if args.asr_model:
        os.environ["ASR_MODEL_ID"] = args.asr_model
    if args.semantic_model:
        os.environ["SEMANTIC_MODEL"] = args.semantic_model
    if args.reranker_model:
        os.environ["RERANKER_MODEL"] = args.reranker_model
    os.environ.setdefault("MODEL_LOCAL_FILES_ONLY", "0")

    from pdf_speech_search.settings import settings

    if not args.skip_search:
        from pdf_speech_search.indexing import get_embedding_model, get_reranker_model

        print(f"Loading embedding model: {settings.semantic_model}")
        get_embedding_model(settings.semantic_model)
        if settings.enable_reranker and settings.reranker_model:
            print(f"Loading reranker model: {settings.reranker_model}")
            get_reranker_model(settings.reranker_model)

    if not args.skip_asr:
        from pdf_speech_search.asr_models import download_model, get_asr_model, model_installed

        spec = get_asr_model(os.getenv("ASR_MODEL_ID", settings.asr_model_id))
        if model_installed(spec):
            print(f"ASR model already downloaded: {spec.label} ({spec.model_name})")
        else:
            print(f"Downloading ASR model: {spec.label} ({spec.model_name})")
            download_model(spec)

    print("Required model files are present.")


if __name__ == "__main__":
    main()
