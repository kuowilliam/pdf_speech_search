from __future__ import annotations

import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and warm local ASR/search models.")
    parser.add_argument("--skip-search", action="store_true", help="Do not warm embedding/reranker models.")
    parser.add_argument("--skip-whisper", action="store_true", help="Do not warm the Whisper model.")
    parser.add_argument("--whisper-model", default=None, help="Override WHISPER_MODEL for this run.")
    parser.add_argument("--semantic-model", default=None, help="Override SEMANTIC_MODEL for this run.")
    parser.add_argument("--reranker-model", default=None, help="Override RERANKER_MODEL for this run.")
    args = parser.parse_args()

    if args.whisper_model:
        os.environ["WHISPER_MODEL"] = args.whisper_model
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

    if not args.skip_whisper:
        from faster_whisper import WhisperModel

        print(
            "Loading Whisper model: "
            f"{settings.whisper_model} ({settings.whisper_device}, {settings.whisper_compute_type})"
        )
        WhisperModel(
            settings.whisper_model,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
        )

    print("Models are downloaded and loadable.")


if __name__ == "__main__":
    main()
