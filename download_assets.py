#!/usr/bin/env python3
"""
download_assets.py — pre-download all model assets required by the backend.

Currently downloads:
  1. FastEmbed SPLADE sparse-embedding model (hybrid retrieval, sparse leg)
  2. Cross-encoder reranker model (post-RRF reranking)

Usage:
    python download_assets.py [options]

Options:
    --cache-dir PATH          FastEmbed cache directory
    --model MODEL             SPLADE model name
    --reranker-cache-dir PATH Reranker cache directory
    --reranker-model MODEL    Cross-encoder model name
    --skip-splade             Skip SPLADE download
    --skip-reranker           Skip reranker download

Defaults are read from environment variables (or .env) matching backend config:
    FASTEMBED_CACHE_DIR  (default: ./assets/fastembed)
    SPLADE_MODEL         (default: prithivida/Splade_PP_en_v1)
    RERANKER_CACHE_DIR   (default: ./assets/reranker)
    RERANKER_MODEL       (default: cross-encoder/ms-marco-MiniLM-L-12-v2)
"""

import argparse
import os
import sys
import time


def _load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader — no external dependency required."""
    if not os.path.isfile(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def download_splade(model_name: str, cache_dir: str) -> None:
    print(f"  model      : {model_name}")
    print(f"  cache_dir  : {os.path.abspath(cache_dir)}")

    try:
        from fastembed import SparseTextEmbedding
    except ImportError:
        print("\n[ERROR] fastembed is not installed.")
        print("        Run:  pip install fastembed")
        sys.exit(1)

    os.makedirs(cache_dir, exist_ok=True)

    print("\nDownloading / verifying model files …")
    t0 = time.time()
    SparseTextEmbedding(model_name=model_name, cache_dir=cache_dir)
    elapsed = time.time() - t0

    print(f"Done in {elapsed:.1f}s.\n")


def download_reranker(model_name: str, cache_dir: str) -> None:
    print(f"  model      : {model_name}")
    print(f"  cache_dir  : {os.path.abspath(cache_dir)}")

    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        print("\n[ERROR] sentence-transformers is not installed.")
        print("        Run:  pip install sentence-transformers")
        sys.exit(1)

    os.makedirs(cache_dir, exist_ok=True)

    print("\nDownloading / verifying model files …")
    t0 = time.time()
    CrossEncoder(model_name, cache_folder=cache_dir)
    elapsed = time.time() - t0

    print(f"Done in {elapsed:.1f}s.\n")


def main() -> None:
    _load_dotenv()

    default_splade_cache  = os.getenv("FASTEMBED_CACHE_DIR", "./assets/fastembed")
    default_splade_model  = os.getenv("SPLADE_MODEL", "prithivida/Splade_PP_en_v1")
    default_reranker_cache = os.getenv("RERANKER_CACHE_DIR", "./assets/reranker")
    default_reranker_model = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-12-v2")

    parser = argparse.ArgumentParser(description="Pre-download RAG-Web-UI model assets.")
    parser.add_argument("--cache-dir", default=default_splade_cache,
                        help=f"FastEmbed cache directory (default: {default_splade_cache})")
    parser.add_argument("--model", default=default_splade_model,
                        help=f"SPLADE model name (default: {default_splade_model})")
    parser.add_argument("--reranker-cache-dir", default=default_reranker_cache,
                        help=f"Reranker cache directory (default: {default_reranker_cache})")
    parser.add_argument("--reranker-model", default=default_reranker_model,
                        help=f"Cross-encoder model name (default: {default_reranker_model})")
    parser.add_argument("--skip-splade", action="store_true",
                        help="Skip SPLADE model download")
    parser.add_argument("--skip-reranker", action="store_true",
                        help="Skip reranker model download")
    args = parser.parse_args()

    print("=" * 60)
    print("RAG-Web-UI asset downloader")
    print("=" * 60)

    step = 1
    total = sum([not args.skip_splade, not args.skip_reranker])

    if not args.skip_splade:
        print(f"\n[{step}/{total}] SPLADE sparse-embedding model (FastEmbed)")
        download_splade(model_name=args.model, cache_dir=args.cache_dir)
        step += 1

    if not args.skip_reranker:
        print(f"[{step}/{total}] Cross-encoder reranker model (sentence-transformers)")
        download_reranker(model_name=args.reranker_model, cache_dir=args.reranker_cache_dir)
        step += 1

    print("All assets downloaded successfully.")
    print("You can now start the backend — no network access needed for model loading.")


if __name__ == "__main__":
    main()
