#!/usr/bin/env python3
"""
RAG Evaluation Harness
======================
Fully external — communicates with the RAG app via HTTP only.
No imports from the RAG app codebase.

Pipeline
--------
1. Fetch SQuAD 2.0 dataset (or load from local cache)
2. Register / login to the RAG app
3. Create a fresh knowledge base
4. Upload article texts as plain-text documents
5. Wait for ingest to finish (poll /query/kb/{id}/ingest-status)
6. Run each question through POST /query
7. Score answers with token-F1 and exact-match (no LLM judge needed)
8. Write results to eval_results.json

Usage
-----
    pip install requests datasets tqdm
    python eval_harness.py \
        --base-url http://localhost:8000/api \
        --username eval_user \
        --password eval_pass \
        --articles 20 \
        --questions 60

All flags have defaults — just set BASE_URL / USERNAME / PASSWORD as env vars
or pass them on the command line.
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import requests
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger("eval")


# ── CLI args ───────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RAG Evaluation Harness")
    p.add_argument("--base-url",   default=os.getenv("RAG_BASE_URL", "http://localhost:8000/api"))
    p.add_argument("--username",   default=os.getenv("RAG_USERNAME", "admin"))
    p.add_argument("--password",   default=os.getenv("RAG_PASSWORD", "admin"))
    p.add_argument("--articles",   type=int, default=20, help="Number of SQuAD articles to ingest")
    p.add_argument("--questions",  type=int, default=60, help="Max questions to evaluate")
    p.add_argument("--output",     default="eval_results.json")
    p.add_argument("--no-cleanup", action="store_true", help="Keep the eval KB after run")
    p.add_argument("--dataset",    default="squad", choices=["squad", "squad_v2"],
                   help="HuggingFace dataset name")
    return p.parse_args()


# ── HTTP client ────────────────────────────────────────────────────────────────

class RAGClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.session  = requests.Session()
        self.session.headers["Content-Type"] = "application/json"

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def login(self, username: str, password: str) -> None:
        resp = self.session.post(
            self._url("/auth/token"),
            data={"username": username, "password": password},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        resp.raise_for_status()
        token = resp.json()["access_token"]
        self.session.headers["Authorization"] = f"Bearer {token}"
        log.info("Authenticated as %s", username)

    def register(self, username: str, password: str, email: str) -> None:
        resp = self.session.post(
            self._url("/auth/register"),
            json={"username": username, "password": password, "email": email},
        )
        if resp.status_code not in (200, 201, 400):
            resp.raise_for_status()
        # 400 = already exists — fine

    def create_kb(self, name: str, description: str = "") -> int:
        resp = self.session.post(
            self._url("/knowledge-base"),
            json={"name": name, "description": description},
        )
        resp.raise_for_status()
        kb_id = resp.json()["id"]
        log.info("Created KB id=%d  name=%r", kb_id, name)
        return kb_id

    def delete_kb(self, kb_id: int) -> None:
        resp = self.session.delete(self._url(f"/knowledge-base/{kb_id}"))
        resp.raise_for_status()
        log.info("Deleted KB id=%d", kb_id)

    def upload_text(self, kb_id: int, filename: str, content: str) -> list[dict]:
        """Upload a plain-text blob as a .txt file. Returns upload result list."""
        resp = self.session.post(
            self._url(f"/knowledge-base/{kb_id}/documents/upload"),
            files={"files": (filename, content.encode(), "text/plain")},
            headers={k: v for k, v in self.session.headers.items() if k != "Content-Type"},
        )
        resp.raise_for_status()
        return resp.json()

    def process_docs(self, kb_id: int, upload_results: list[dict]) -> list[dict]:
        resp = self.session.post(
            self._url(f"/knowledge-base/{kb_id}/documents/process"),
            json=upload_results,
        )
        resp.raise_for_status()
        return resp.json()["tasks"]

    def ingest_status(self, kb_id: int) -> dict:
        resp = self.session.get(self._url(f"/query/kb/{kb_id}/ingest-status"))
        resp.raise_for_status()
        return resp.json()

    def wait_for_ingest(self, kb_id: int, timeout: int = 600, poll: int = 5) -> None:
        """Block until KB is ready or timeout expires."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            status = self.ingest_status(kb_id)
            log.info(
                "Ingest status: total=%d completed=%d failed=%d pending=%d ready=%s",
                status["total"], status["completed"], status["failed"],
                status["pending"], status["ready"],
            )
            if status["ready"]:
                return
            if status["failed"] > 0:
                raise RuntimeError(
                    f"{status['failed']} document(s) failed to process. "
                    "Check the RAG app logs."
                )
            time.sleep(poll)
        raise TimeoutError(f"KB {kb_id} did not become ready within {timeout}s")

    def query(self, kb_id: int, question: str, generate_answer: bool = True) -> dict:
        resp = self.session.post(
            self._url("/query"),
            json={
                "question": question,
                "kb_ids": [kb_id],
                "generate_answer": generate_answer,
            },
        )
        resp.raise_for_status()
        return resp.json()


# ── Dataset loading ────────────────────────────────────────────────────────────

def load_squad(dataset_name: str, max_articles: int, max_questions: int) -> tuple[list[dict], list[dict]]:
    """
    Returns:
        articles  — list of {title, text}
        questions — list of {id, question, context_title, answers: [str]}
    """
    try:
        from datasets import load_dataset
    except ImportError:
        log.error("Run:  pip install datasets")
        sys.exit(1)

    log.info("Loading %s from HuggingFace (cached after first download)...", dataset_name)
    ds = load_dataset(dataset_name, split="validation")

    seen_titles: dict[str, str] = {}   # title -> text
    questions:   list[dict]     = []

    for row in ds:
        title = row["title"]
        if title not in seen_titles:
            if len(seen_titles) >= max_articles:
                continue
            seen_titles[title] = row["context"]

        if len(questions) >= max_questions:
            break

        # squad_v2 has unanswerable questions (answers.text == []) — skip them
        ans_texts = row["answers"]["text"]
        if not ans_texts:
            continue

        questions.append({
            "id":            row["id"],
            "question":      row["question"],
            "context_title": title,
            "answers":       ans_texts,
        })

    articles = [{"title": t, "text": c} for t, c in seen_titles.items()]
    log.info("Loaded %d articles, %d questions", len(articles), len(questions))
    return articles, questions


# ── Scoring ────────────────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9 ]", "", text)
    return " ".join(text.split())


def token_f1(prediction: str, references: list[str]) -> float:
    """Max token-F1 over all reference answers (SQuAD official metric)."""
    pred_tokens = Counter(_normalize(prediction).split())
    best = 0.0
    for ref in references:
        ref_tokens = Counter(_normalize(ref).split())
        common = sum((pred_tokens & ref_tokens).values())
        if common == 0:
            continue
        p = common / sum(pred_tokens.values())
        r = common / sum(ref_tokens.values())
        best = max(best, 2 * p * r / (p + r))
    return best


def exact_match(prediction: str, references: list[str]) -> bool:
    norm_pred = _normalize(prediction)
    return any(_normalize(ref) == norm_pred for ref in references)


# ── Result types ───────────────────────────────────────────────────────────────

@dataclass
class QuestionResult:
    question_id:   str
    question:      str
    ground_truth:  list[str]
    answer:        Optional[str]
    confidence:    str
    suggestion:    Optional[str]
    token_f1:      float
    exact_match:   bool
    num_contexts:  int
    latency_ms:    int
    failed_legs:   list[str]
    error:         Optional[str] = None


@dataclass
class EvalReport:
    dataset:       str
    timestamp:     str
    rag_base_url:  str
    num_articles:  int
    num_questions: int
    mean_f1:       float
    exact_match_pct: float
    confidence_dist: dict
    avg_latency_ms:  float
    avg_contexts:    float
    questions:     list[dict] = field(default_factory=list)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    client = RAGClient(args.base_url)

    # Register eval user (no-op if already exists), then login
    eval_email = f"{args.username}@eval.local"
    client.register(args.username, args.password, eval_email)
    client.login(args.username, args.password)

    # Load dataset
    articles, questions = load_squad(args.dataset, args.articles, args.questions)

    # Create a fresh KB for this eval run
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    kb_id  = client.create_kb(
        name=f"eval_{args.dataset}_{run_id}",
        description=f"Auto-created by eval harness — {args.dataset}",
    )

    try:
        # ── Ingest articles ────────────────────────────────────────────────────
        log.info("Uploading %d articles...", len(articles))
        all_uploads: list[dict] = []
        for article in tqdm(articles, desc="Uploading"):
            # Each article becomes a single .txt file named after its title
            safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", article["title"])[:60]
            filename  = f"{safe_name}.txt"
            uploads   = client.upload_text(kb_id, filename, article["text"])
            all_uploads.extend(u for u in uploads if not u.get("skip_processing"))

        if not all_uploads:
            log.warning("All files already existed in the KB — skipping process step")
        else:
            log.info("Triggering processing for %d uploads...", len(all_uploads))
            client.process_docs(kb_id, all_uploads)

        log.info("Waiting for ingest to complete...")
        client.wait_for_ingest(kb_id, timeout=600, poll=8)

        # ── Run evaluation queries ─────────────────────────────────────────────
        results: list[QuestionResult] = []

        for q in tqdm(questions, desc="Evaluating"):
            try:
                resp = client.query(kb_id, q["question"], generate_answer=True)
                answer       = resp.get("answer") or ""
                confidence   = resp.get("confidence", "none")
                suggestion   = resp.get("suggestion")
                latency_ms   = resp.get("latency_ms", 0)
                contexts     = resp.get("contexts", [])
                failed_legs  = resp.get("retrieval_info", {}).get("failed_legs", [])

                f1  = token_f1(answer, q["answers"])
                em  = exact_match(answer, q["answers"])

                results.append(QuestionResult(
                    question_id  = q["id"],
                    question     = q["question"],
                    ground_truth = q["answers"],
                    answer       = answer,
                    confidence   = confidence,
                    suggestion   = suggestion,
                    token_f1     = round(f1, 4),
                    exact_match  = em,
                    num_contexts = len(contexts),
                    latency_ms   = latency_ms,
                    failed_legs  = failed_legs,
                ))
            except Exception as exc:
                log.warning("Question %s failed: %s", q["id"], exc)
                results.append(QuestionResult(
                    question_id  = q["id"],
                    question     = q["question"],
                    ground_truth = q["answers"],
                    answer       = None,
                    confidence   = "none",
                    suggestion   = None,
                    token_f1     = 0.0,
                    exact_match  = False,
                    num_contexts = 0,
                    latency_ms   = 0,
                    failed_legs  = [],
                    error        = str(exc),
                ))

        # ── Compile report ─────────────────────────────────────────────────────
        ok = [r for r in results if r.error is None]
        conf_dist: dict[str, int] = {}
        for r in ok:
            conf_dist[r.confidence] = conf_dist.get(r.confidence, 0) + 1

        report = EvalReport(
            dataset          = args.dataset,
            timestamp        = run_id,
            rag_base_url     = args.base_url,
            num_articles     = len(articles),
            num_questions    = len(results),
            mean_f1          = round(sum(r.token_f1 for r in ok) / max(len(ok), 1), 4),
            exact_match_pct  = round(100 * sum(r.exact_match for r in ok) / max(len(ok), 1), 2),
            confidence_dist  = conf_dist,
            avg_latency_ms   = round(sum(r.latency_ms for r in ok) / max(len(ok), 1), 1),
            avg_contexts     = round(sum(r.num_contexts for r in ok) / max(len(ok), 1), 2),
            questions        = [asdict(r) for r in results],
        )

        # Write results
        out = Path(args.output)
        out.write_text(json.dumps(asdict(report), indent=2))

        # Print summary
        print("\n" + "=" * 60)
        print(f"  Dataset        : {report.dataset}")
        print(f"  Articles       : {report.num_articles}")
        print(f"  Questions      : {report.num_questions}  (errors: {len(results) - len(ok)})")
        print(f"  Mean token-F1  : {report.mean_f1:.4f}")
        print(f"  Exact match    : {report.exact_match_pct:.1f}%")
        print(f"  Avg latency    : {report.avg_latency_ms:.0f} ms")
        print(f"  Avg contexts   : {report.avg_contexts:.1f}")
        print(f"  Confidence     : {report.confidence_dist}")
        print(f"  Output         : {out.resolve()}")
        print("=" * 60)

    finally:
        if not args.no_cleanup:
            try:
                client.delete_kb(kb_id)
            except Exception as exc:
                log.warning("KB cleanup failed: %s", exc)


if __name__ == "__main__":
    main()
