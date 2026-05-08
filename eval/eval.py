#!/usr/bin/env python3
"""
eval.py — run retrieval evaluation against an already-ingested knowledge base.

Reads ingest_state.json (written by ingest.py) for kb_id and questions.
Runs each retrieval config, scores results, prints a comparison table,
and writes eval_results.json.

Usage:
    python eval.py \\
        --username eval_user --password eval_pass \\
        --state ingest_state.json

    # Single config instead of full sweep:
    python eval.py --state ingest_state.json --use-dense --use-sparse

    # Include graph config:
    python eval.py --state ingest_state.json --graph

    # With LLM answer generation (slower, costs tokens):
    python eval.py --state ingest_state.json --generate-answers
"""

import argparse
import json
import os
import time

from tqdm import tqdm

from lib import RAGClient, RETRIEVAL_CONFIGS, token_f1, exact_match, retrieval_hit


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
        # context text (oracle span scoring). When generation is on, score
        # against the LLM answer.
        score_text = answer if answer else " ".join(c["content"] for c in contexts)

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
    best_f1  = max(run_results, key=lambda x: x["mean_f1"])
    best_hit = max(run_results, key=lambda x: x["hit_rate"])
    best_lat = min(run_results, key=lambda x: x["mean_latency_ms"])
    print(f"Best F1:      {best_f1['config']['name']}  ({best_f1['mean_f1']:.3f})")
    print(f"Best hit rate:{best_hit['config']['name']}  ({best_hit['hit_rate']:.3f})")
    print(f"Fastest:      {best_lat['config']['name']}  ({best_lat['mean_latency_ms']:.0f} ms)")
    print()


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Run retrieval eval against an ingested KB")
    parser.add_argument("--base-url",  default=os.getenv("BASE_URL",  "http://localhost:8000/api"))
    parser.add_argument("--username",  default=os.getenv("USERNAME",  "eval_user"))
    parser.add_argument("--password",  default=os.getenv("PASSWORD",  "eval_pass"))
    parser.add_argument("--email",     default=os.getenv("EMAIL",     "eval@example.com"))
    parser.add_argument("--state",     default="ingest_state.json",
                        help="ingest_state.json written by ingest.py")
    parser.add_argument("--use-dense",  action="store_true", default=None)
    parser.add_argument("--use-sparse", action="store_true", default=None)
    parser.add_argument("--use-kg",     action="store_true", default=None)
    parser.add_argument("--graph",      action="store_true",
                        help="Include all_3+graph config in sweep")
    parser.add_argument("--generate-answers", action="store_true")
    parser.add_argument("--output",    default="eval_results.json")
    args = parser.parse_args()

    with open(args.state) as f:
        state = json.load(f)
    kb_id     = state["kb_id"]
    questions = state["questions"]
    print(f"Loaded state: KB id={kb_id}, {len(questions)} questions")

    client = RAGClient(args.base_url)
    print("Authenticating...")
    client.register(args.username, args.password, args.email)
    client.login(args.username, args.password)
    print(f"  Logged in as {args.username}")

    # Determine configs
    single_mode = any(v is True for v in [args.use_dense, args.use_sparse, args.use_kg])
    if single_mode:
        use_dense  = bool(args.use_dense)
        use_sparse = bool(args.use_sparse)
        use_kg     = bool(args.use_kg)
        parts = ["exact"]
        if use_dense:  parts.append("dense")
        if use_sparse: parts.append("sparse")
        if use_kg:     parts.append("graph")
        configs = [{
            "name":          "+".join(parts),
            "label":         "Custom: " + ", ".join(p.capitalize() for p in parts),
            "use_exact":     True,
            "use_dense":     use_dense,
            "use_sparse":    use_sparse,
            "use_graph_rag": use_kg,
        }]
        print(f"Single-config mode: {configs[0]['name']}")
    else:
        configs = [c for c in RETRIEVAL_CONFIGS if not c.get("graph_only") or args.graph]
        print(f"Sweep mode: {len(configs)} configs"
              + (" (including graph)" if args.graph else " (use --graph to include KG config)"))

    # Run
    run_results = []
    for cfg in configs:
        print(f"\nRunning config: {cfg['name']} — {cfg['label']}")
        result = run_config(client, cfg, questions, kb_id, args.generate_answers)
        run_results.append(result)
        print(
            f"  F1={result['mean_f1']:.3f}  EM={result['mean_em']:.3f}  "
            f"Hit={result['hit_rate']:.3f}  {result['mean_latency_ms']:.0f}ms avg"
        )

    print_comparison_table(run_results)

    output = {
        "timestamp":        time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "kb_id":            kb_id,
        "n_questions":      len(questions),
        "generate_answers": args.generate_answers,
        "configs_run":      [r["config"]["name"] for r in run_results],
        "summary": [
            {
                "config":          r["config"]["name"],
                "label":           r["config"]["label"],
                "mean_f1":         r["mean_f1"],
                "mean_em":         r["mean_em"],
                "hit_rate":        r["hit_rate"],
                "mean_latency_ms": r["mean_latency_ms"],
                "errors":          r["errors"],
            }
            for r in run_results
        ],
        "details": {r["config"]["name"]: r["details"] for r in run_results},
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
