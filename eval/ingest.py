#!/usr/bin/env python3
"""
ingest.py — fetch SQuAD 2.0, create a KB, upload + process documents.

Writes ingest_state.json on success:
  {
    "kb_id": 42,
    "n_articles": 20,
    "questions": [ {"question": ..., "answers": [...], "title": ...}, ... ]
  }

Run once. Pass the output file to eval.py via --state.

Usage:
    python ingest.py \\
        --username eval_user --password eval_pass \\
        --articles 20 --questions 60 \\
        --state ingest_state.json
"""

import argparse
import json
import os
import re
import time

import requests
from tqdm import tqdm

from lib import RAGClient


# ── Dataset ────────────────────────────────────────────────────────────────────

def load_squad(n_articles: int, n_questions: int) -> tuple[dict[str, str], list[dict]]:
    from datasets import load_dataset
    ds = load_dataset("squad_v2", split="validation")

    articles: dict[str, list[str]] = {}  # title -> list of unique context paragraphs
    questions: list[dict] = []

    for row in ds:
        title = row["title"]
        context = row["context"]
        if title not in articles:
            if len(articles) >= n_articles:
                continue
            articles[title] = []
        if context not in articles[title]:
            articles[title].append(context)
        if len(questions) < n_questions:
            answers = row["answers"]["text"]
            if answers:
                questions.append({
                    "question": row["question"],
                    "answers":  answers,
                    "title":    title,
                })
        if len(articles) >= n_articles and len(questions) >= n_questions:
            break

    # Join all paragraphs into one document per article
    return {title: "\n\n".join(paragraphs) for title, paragraphs in articles.items()}, questions


# ── Ingest ─────────────────────────────────────────────────────────────────────

def ingest_articles(
    client: RAGClient,
    kb_id: int,
    articles: dict[str, str],
    poll_interval: int = 5,
    timeout: int = 300,
) -> None:
    upload_results = []
    print(f"  Uploading {len(articles)} articles...")
    for title, text in tqdm(articles.items(), desc="  upload"):
        safe_title = re.sub(r'[\\/:*?"<>|]+', "_", title).strip() or "untitled"
        resp = client.upload_text(kb_id, f"{safe_title}.txt", text)
        upload_results.extend(resp)

    print(f"  Triggering processing for {len(upload_results)} documents...")
    client.trigger_processing(kb_id, upload_results)

    print("  Waiting for ingest to complete...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            status = client.ingest_status(kb_id)
        except (requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout):
            time.sleep(poll_interval)
            continue
        print(f"    {status['completed']}/{status['total']} done, {status['failed']} failed", end="\r")
        if status["ready"]:
            print(f"\n  Ingest complete: {status['completed']} chunks indexed.")
            return
        if status["failed"] > 0:
            print(f"\n  Warning: {status['failed']} documents failed processing.")
            return
        time.sleep(poll_interval)
    raise TimeoutError(f"Ingest did not complete within {timeout}s")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest SQuAD articles into a RAG knowledge base")
    parser.add_argument("--base-url",  default=os.getenv("BASE_URL",  "http://localhost:8000/api"))
    parser.add_argument("--username",  default=os.getenv("USERNAME",  "eval_user"))
    parser.add_argument("--password",  default=os.getenv("PASSWORD",  "eval_pass"))
    parser.add_argument("--email",     default=os.getenv("EMAIL",     "eval@example.com"))
    parser.add_argument("--articles",  type=int, default=20)
    parser.add_argument("--questions", type=int, default=60)
    parser.add_argument("--state",     default="ingest_state.json", help="Output file for kb_id + questions")
    args = parser.parse_args()

    client = RAGClient(args.base_url)

    print("Authenticating...")
    client.register(args.username, args.password, args.email)
    client.login(args.username, args.password)
    print(f"  Logged in as {args.username}")

    print(f"Loading SQuAD 2.0 ({args.articles} articles, {args.questions} questions)...")
    articles, questions = load_squad(args.articles, args.questions)
    print(f"  Loaded {len(articles)} articles, {len(questions)} questions")

    ts = int(time.time())
    print("Creating knowledge base...")
    kb_id = client.create_kb(f"eval-squad-{ts}", "SQuAD 2.0 evaluation KB")
    print(f"  KB id={kb_id}")

    ingest_articles(client, kb_id, articles)

    state = {
        "kb_id":      kb_id,
        "n_articles": len(articles),
        "questions":  questions,
    }
    with open(args.state, "w") as f:
        json.dump(state, f, indent=2)
    print(f"State written to {args.state}")
    print(f"Run eval with: python eval.py --state {args.state}")


if __name__ == "__main__":
    main()
