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
6. For each RETRIEVAL_CONFIG, run every question through POST /query
7. Score with token-F1 and exact-match
8. Write per-config results + comparison table to eval_results.json

Retrieval configs
-----------------
The harness runs every question through multiple leg combinations so you can
see the effect of each source in isolation and in combination:

  exact_only      — keyword search only (baseline)
  dense_only      — dense vectors only
  sparse_only     — sparse vectors (SPLADE) only
  dense+sparse    — vector legs combined, no keyword
  dense+exact     — dense + keyword
  sparse+exact    — sparse + keyword
  all_3           — full hybrid (no graph)
  all_3+graph     — full hybrid + knowledge graph (opt-in, skipped if graph=False)

Each config sends different use_dense / use_sparse / use_exact / use_graph_rag
flags to POST /api/query. The same question set is reused across all configs —
no re-ingestion needed.

RRF behaviour per config
------------------------
With N legs enabled, RRF scores each chunk as:

  score = Σ  weight_leg / (60 + rank_leg)   for each leg where chunk appeared

A chunk absent from a disabled leg contributes 0 from that leg but can still
surface via the remaining legs. The weights come from the server's .env:

  HYBRID_DENSE_WEIGHT          default 0.5
  HYBRID_QDRANT_SPARSE_WEIGHT  default 0.3
  HYBRID_EXACT_WEIGHT          default 0.2

So with only dense enabled:  score = 0.5 / (60 + rank)  → pure cosine ranking
With dense+exact:            score = 0.5/(60+dr) + 0.2/(60+er)

Usage
-----
    pip install -r requirements.txt
    python eval_harness.py \\
        --base-url http://localhost:8000/api \\
        --username eval_user \\
        --password eval_pass \\
        --articles 20 \\
        --questions 60 \\
        --no-graph            # skip the all_3+graph config (default: skipped)
        --graph               # include graph config (requires Neo4j + GraphRAG)
        --generate-answers    # run LLM answer generation (slower, costs tokens)

All flags have defaults — set BASE_URL / USERNAME / PASSWORD as env vars or
pass them on the command line.
"""

import argparse
import json
import os
import re
import string
import sys
import time
from collections import Counter, defaultdict
from typing import Any

import requests
from tqdm import tqdm

# ── Retrieval configurations ───────────────────────────────────────────────────

RETRIEVAL_CONFIGS = [
    {
        "name":        "exact_only",
        "label":       "Keyword only (baseline)",
        "use_exact":   True,
        "use_dense":   False,
        "use_sparse":  False,
        "use_graph_rag": False,
    },
    {
        "name":        "dense_only",
        "label":       "Dense vectors only",
        "use_exact":   False,
        "use_dense":   True,
        "use_sparse":  False,
        "use_graph_rag": False,
    },
    {
        "name":        "sparse_only",
        "label":       "Sparse vectors only (SPLADE)",
        "use_exact":   False,
        "use_dense":   False,
        "use_sparse":  True,
        "use_graph_rag": False,
    },
    {
        "name":        "dense+sparse",
        "label":       "Dense + Sparse (no keyword)",
        "use_exact":   False,
        "use_dense":   True,
        "use_sparse":  True,
        "use_graph_rag": False,
    },
    {
        "name":        "dense+exact",
        "label":       "Dense + Keyword",
        "use_exact":   True,
        "use_dense":   True,
        "use_sparse":  False,
        "use_graph_rag": False,
    },
    {
        "name":        "sparse+exact",
        "label":       "Sparse + Keyword",
        "use_exact":   True,
        "use_dense":   False,
        "use_sparse":  True,
        "use_graph_rag": False,
    },
    {
        "name":        "all_3",
        "label":       "Full hybrid (dense + sparse + keyword)",
        "use_exact":   True,
        "use_dense":   True,
        "use_sparse":  True,
        "use_graph_rag": False,
    },
    {
        "name":        "all_3+graph",
        "label":       "Full hybrid + Knowledge Graph",
        "use_exact":   True,
        "use_dense":   True,
        "use_sparse":  True,
        "use_graph_rag": True,
        "graph_only":  True,   # skipped unless --graph flag is passed
    },
]

# ── Scoring ────────────────────────────────────────────────────────────────────

def _normalise(text: str) -> list[str]:
    """Lowercase, strip punctuation, tokenise."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.split()


def token_f1(prediction: str, ground_truths: list[str]) -> float:
    """Max token-F1 over all ground truth answers (official SQuAD metric)."""
    best = 0.0
    pred_tokens = Counter(_normalise(prediction))
    for gt in ground_truths:
        gt_tokens = Counter(_normalise(gt))
        common = sum((pred_tokens & gt_tokens).values())
        if common == 0:
            continue
        precision = common / sum(pred_tokens.values())
        recall    = common / sum(gt_tokens.values())
        f1 = 2 * precision * recall / (precision + recall)
        best = max(best, f1)
    return best


def exact_match(prediction: str, ground_truths: list[str]) -> float:
    """1.0 if normalised prediction matches any ground truth exactly."""
    pred = " ".join(_normalise(prediction))
    for gt in ground_truths:
        if pred == " ".join(_normalise(gt)):
            return 1.0
    return 0.0


def retrieval_hit(contexts: list[dict], ground_truths: list[str]) -> float:
    """1.0 if any ground truth answer appears (case-insensitive) in any chunk."""
    all_text = " ".join(c["content"].lower() for c in contexts)
    for gt in ground_truths:
        if " ".join(_normalise(gt)) in " ".join(_normalise(all_text)):
            return 1.0
    return 0.0


# ── HTTP client ────────────────────────────────────────────────────────────────

class RAGClient:
    def __init__(self, base_url: str, timeout: int = 60):
        self.base = base_url.rstrip("/")
        self.timeout = timeout
        self.token: str | None = None
        self.session = requests.Session()

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        return h

    def register(self, username: str, password: str, email: str) -> bool:
        r = self.session.post(
            f"{self.base}/auth/register",
            json={"username": username, "password": password, "email": email},
            timeout=self.timeout,
        )
        return r.status_code in (200, 201, 400)  # 400 = already exists

    def login(self, username: str, password: str) -> None:
        r = self.session.post(
            f"{self.base}/auth/token",
            data={"username": username, "password": password},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=self.timeout,
        )
        r.raise_for_status()
        self.token = r.json()["access_token"]

    def create_kb(self, name: str, description: str = "") -> int:
        r = self.session.post(
            f"{self.base}/knowledge-base",
            json={"name": name, "description": description},
            headers=self._headers(),
            timeout=self.timeout,
        )
        # r.raise_for_status()
        try:
            r.raise_for_status()
        except requests.exceptions.HTTPError:
            print(r.text)  # 🔥 THIS IS CRITICAL
            raise
        return r.json()["id"]

    def upload_text(self, kb_id: int, filename: str, content: str) -> list[dict]:
        r = self.session.post(
            f"{self.base}/knowledge-base/{kb_id}/documents/upload",
            files=[("files", (filename, content.encode(), "text/plain"))],
            headers={"Authorization": f"Bearer {self.token}"},
            timeout=self.timeout,
        )
        
        # r.raise_for_status()
        try:
            r.raise_for_status()
        except requests.exceptions.HTTPError:
            print(r.text)  # 🔥 THIS IS CRITICAL
            raise
        return r.json()

    def trigger_processing(self, kb_id: int, upload_results: list[dict]) -> dict:
        r = self.session.post(
            f"{self.base}/knowledge-base/{kb_id}/documents/process",
            json=upload_results,
            headers=self._headers(),
            timeout=self.timeout,
        )
        # r.raise_for_status()
        try:
            r.raise_for_status()
        except requests.exceptions.HTTPError:
            print(r.text)  # 🔥 THIS IS CRITICAL
            raise
        return r.json()

    def ingest_status(self, kb_id: int) -> dict:
        r = self.session.get(
            f"{self.base}/query/kb/{kb_id}/ingest-status",
            headers=self._headers(),
            timeout=self.timeout,
        )
        # r.raise_for_status()
        try:
            r.raise_for_status()
        except requests.exceptions.HTTPError:
            print(r.text)  # 🔥 THIS IS CRITICAL
            raise
        return r.json()

    def query(
        self,
        question: str,
        kb_ids: list[int],
        use_dense: bool = True,
        use_sparse: bool = True,
        use_exact: bool = True,
        use_graph_rag: bool = False,
        generate_answer: bool = False,
    ) -> dict:
        r = self.session.post(
            f"{self.base}/query",
            json={
                "question":       question,
                "kb_ids":         kb_ids,
                "use_dense":      use_dense,
                "use_sparse":     use_sparse,
                "use_exact":      use_exact,
                "use_graph_rag":  use_graph_rag,
                "generate_answer": generate_answer,
            },
            headers=self._headers(),
            timeout=self.timeout,
        )
        # r.raise_for_status()
        try:
            r.raise_for_status()
        except requests.exceptions.HTTPError:
            print(r.text)  # 🔥 THIS IS CRITICAL
            raise
        return r.json()

    def delete_kb(self, kb_id: int) -> None:
        self.session.delete(
            f"{self.base}/knowledge-base/{kb_id}",
            headers=self._headers(),
            timeout=self.timeout,
        )


# ── Dataset ────────────────────────────────────────────────────────────────────

def load_squad(n_articles: int, n_questions: int) -> tuple[dict[str, str], list[dict]]:
    """
    Returns:
      articles  — {title: text}
      questions — [{question, answers, title}]
    """
    from datasets import load_dataset
    ds = load_dataset("squad_v2", split="validation")

    articles: dict[str, str] = {}
    questions: list[dict] = []

    for row in ds:
        title = row["title"]
        if title not in articles:
            if len(articles) >= n_articles:
                continue
            articles[title] = row["context"]
        if len(questions) < n_questions:
            answers = row["answers"]["text"]
            if answers:   # skip unanswerable questions
                questions.append({
                    "question": row["question"],
                    "answers":  answers,
                    "title":    title,
                })
        if len(articles) >= n_articles and len(questions) >= n_questions:
            break

    return articles, questions


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
        # Keep filenames filesystem-safe; SQuAD titles can contain path separators.
        safe_title = re.sub(r'[\\/:*?"<>|]+', "_", title).strip() or "untitled"
        filename = f"{safe_title}.txt"
        resp = client.upload_text(kb_id, filename, text)
        upload_results.extend(resp)

    print(f"  Triggering processing for {len(upload_results)} documents...")
    client.trigger_processing(kb_id, upload_results)

    print("  Waiting for ingest to complete...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            status = client.ingest_status(kb_id)
        except (requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout):
            # Server may briefly drop connections during --reload or worker restart.
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


# ── Evaluation loop ────────────────────────────────────────────────────────────

def run_config(
    client: RAGClient,
    config: dict,
    questions: list[dict],
    kb_id: int,
    generate_answers: bool,
) -> dict:
    results = []
    latencies = []

    for q in tqdm(questions, desc=f"  {config['name']:<16}", leave=False):
        try:
            resp = client.query(
                question=q["question"],
                kb_ids=[kb_id],
                use_dense=config["use_dense"],
                use_sparse=config["use_sparse"],
                use_exact=config["use_exact"],
                use_graph_rag=config["use_graph_rag"],
                generate_answer=generate_answers,
            )
        except Exception as e:
            results.append({"error": str(e), "f1": 0.0, "em": 0.0, "hit": 0.0})
            continue

        answer     = resp.get("answer") or ""
        contexts   = resp.get("contexts", [])
        confidence = resp.get("confidence", "none")
        latency_ms = resp.get("latency_ms", 0)
        legs       = resp.get("retrieval_info", {}).get("legs", {})

        # When answer generation is off, score F1/EM against the retrieved
        # context text (oracle span scoring). This measures whether the answer
        # is present in the retrieved chunks — which is what retrieval eval
        # benchmarks. When generation is on, score against the LLM answer.
        if answer:
            score_text = answer
        else:
            score_text = " ".join(c["content"] for c in contexts)

        f1  = token_f1(score_text, q["answers"])    if score_text else 0.0
        em  = exact_match(score_text, q["answers"]) if score_text else 0.0
        hit = retrieval_hit(contexts, q["answers"])

        latencies.append(latency_ms)
        results.append({
            "question":   q["question"],
            "answers":    q["answers"],
            "prediction": answer,
            "f1":         round(f1,  4),
            "em":         round(em,  4),
            "hit":        round(hit, 4),
            "confidence": confidence,
            "latency_ms": latency_ms,
            "legs":       legs,
        })

    n = len(results)
    errors = sum(1 for r in results if "error" in r)
    return {
        "config":      config,
        "n_questions": n,
        "errors":      errors,
        "mean_f1":     round(sum(r["f1"]  for r in results) / n, 4) if n else 0,
        "mean_em":     round(sum(r["em"]  for r in results) / n, 4) if n else 0,
        "hit_rate":    round(sum(r["hit"] for r in results) / n, 4) if n else 0,
        "mean_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0,
        "details":     results,
    }


# ── Summary table ──────────────────────────────────────────────────────────────

def print_comparison_table(run_results: list[dict]) -> None:
    header = f"{'Config':<22} {'Label':<42} {'F1':>6} {'EM':>6} {'Hit%':>6} {'ms':>6}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for r in run_results:
        cfg = r["config"]
        print(
            f"{cfg['name']:<22} {cfg['label']:<42} "
            f"{r['mean_f1']:>6.3f} {r['mean_em']:>6.3f} "
            f"{r['hit_rate']:>6.3f} {r['mean_latency_ms']:>6.0f}"
        )
    print("=" * len(header))
    print()
    # Best config per metric
    best_f1  = max(run_results, key=lambda x: x["mean_f1"])
    best_hit = max(run_results, key=lambda x: x["hit_rate"])
    best_lat = min(run_results, key=lambda x: x["mean_latency_ms"])
    print(f"Best F1:      {best_f1['config']['name']}  ({best_f1['mean_f1']:.3f})")
    print(f"Best hit rate:{best_hit['config']['name']}  ({best_hit['hit_rate']:.3f})")
    print(f"Fastest:      {best_lat['config']['name']}  ({best_lat['mean_latency_ms']:.0f} ms)")
    print()


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="RAG evaluation harness")
    parser.add_argument("--base-url",  default=os.getenv("BASE_URL",  "http://localhost:8000/api"))
    parser.add_argument("--username",  default=os.getenv("USERNAME",  "eval_user"))
    parser.add_argument("--password",  default=os.getenv("PASSWORD",  "eval_pass"))
    parser.add_argument("--email",     default=os.getenv("EMAIL",     "eval@example.com"))
    parser.add_argument("--articles",  type=int, default=20,  help="Number of SQuAD articles to ingest")
    parser.add_argument("--questions", type=int, default=60,  help="Number of questions to evaluate")

    # ── Leg flags — run a single specific combination ───────────────────────
    # If none of these are passed, the harness runs ALL configs (benchmark sweep).
    # If any are passed, only the specified combination runs.
    parser.add_argument("--use-dense",  action="store_true", default=None,
                        help="Enable dense vector retrieval leg")
    parser.add_argument("--use-sparse", action="store_true", default=None,
                        help="Enable sparse vector (SPLADE) retrieval leg")
    parser.add_argument("--use-kg",     action="store_true", default=None,
                        help="Enable knowledge graph (Neo4j GraphRAG) retrieval leg")
    # Keyword search is always on — cannot be disabled (matches app behaviour).

    parser.add_argument("--graph",     action="store_true",
                        help="Include all_3+graph config in sweep (requires Neo4j). "
                             "Implied when --use-kg is passed.")
    parser.add_argument("--generate-answers", action="store_true", help="Run LLM answer generation (slower)")
    parser.add_argument("--output",    default="eval_results.json")
    parser.add_argument("--keep-kb",   action="store_true",   help="Don't delete the KB after eval")
    parser.add_argument("--kb-id",     type=int, default=None, help="Reuse existing KB (skip ingest)")
    args = parser.parse_args()

    client = RAGClient(args.base_url)

    # ── Auth ───────────────────────────────────────────────────────────────────
    print("Authenticating...")
    client.register(args.username, args.password, args.email)
    client.login(args.username, args.password)
    print(f"  Logged in as {args.username}")

    # ── Dataset ────────────────────────────────────────────────────────────────
    print(f"Loading SQuAD 2.0 ({args.articles} articles, {args.questions} questions)...")
    articles, questions = load_squad(args.articles, args.questions)
    print(f"  Loaded {len(articles)} articles, {len(questions)} questions")

    # ── Ingest (or reuse) ──────────────────────────────────────────────────────
    kb_id = args.kb_id
    if kb_id is None:
        ts = int(time.time())
        print("Creating knowledge base...")
        kb_id = client.create_kb(f"eval-squad-{ts}", "SQuAD 2.0 evaluation KB")
        print(f"  KB id={kb_id}")
        ingest_articles(client, kb_id, articles)
    else:
        print(f"Reusing existing KB id={kb_id}")

    # ── Determine configs to run ───────────────────────────────────────────────
    # Any leg flag passed → run exactly that one combination.
    # No leg flags passed → full benchmark sweep across all predefined configs.
    single_mode = any(v is True for v in [args.use_dense, args.use_sparse, args.use_kg])

    if single_mode:
        use_dense  = bool(args.use_dense)
        use_sparse = bool(args.use_sparse)
        use_kg     = bool(args.use_kg)
        parts = ["exact"]   # always on
        if use_dense:  parts.append("dense")
        if use_sparse: parts.append("sparse")
        if use_kg:     parts.append("graph")
        name  = "+".join(parts)
        label = "Custom: " + ", ".join(p.capitalize() for p in parts)
        configs = [{
            "name":          name,
            "label":         label,
            "use_exact":     True,
            "use_dense":     use_dense,
            "use_sparse":    use_sparse,
            "use_graph_rag": use_kg,
        }]
        print(f"Single-config mode: {name}")
    else:
        include_graph = args.graph
        configs = [c for c in RETRIEVAL_CONFIGS if not c.get("graph_only") or include_graph]
        print(f"Sweep mode: running {len(configs)} configs"
              + (" (including graph)" if include_graph else " (use --graph to include KG config)"))

    # ── Eval loop ──────────────────────────────────────────────────────────────
    run_results = []
    for cfg in configs:
        print(f"\nRunning config: {cfg['name']} — {cfg['label']}")
        result = run_config(client, cfg, questions, kb_id, args.generate_answers)
        run_results.append(result)
        print(
            f"  F1={result['mean_f1']:.3f}  EM={result['mean_em']:.3f}  "
            f"Hit={result['hit_rate']:.3f}  {result['mean_latency_ms']:.0f}ms avg"
        )

    # ── Results ────────────────────────────────────────────────────────────────
    print_comparison_table(run_results)

    output = {
        "timestamp":      time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "kb_id":          kb_id,
        "n_articles":     len(articles),
        "n_questions":    len(questions),
        "generate_answers": args.generate_answers,
        "configs_run":    [r["config"]["name"] for r in run_results],
        "summary": [
            {
                "config":       r["config"]["name"],
                "label":        r["config"]["label"],
                "mean_f1":      r["mean_f1"],
                "mean_em":      r["mean_em"],
                "hit_rate":     r["hit_rate"],
                "mean_latency_ms": r["mean_latency_ms"],
                "errors":       r["errors"],
            }
            for r in run_results
        ],
        "details": {r["config"]["name"]: r["details"] for r in run_results},
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results written to {args.output}")

    # ── Cleanup ────────────────────────────────────────────────────────────────
    if not args.keep_kb and args.kb_id is None:
        print(f"Deleting KB {kb_id}...")
        client.delete_kb(kb_id)


if __name__ == "__main__":
    main()
