"""
Shared utilities for the RAG eval suite.

Imported by both ingest.py and eval.py — no CLI logic here.
"""

import string
from collections import Counter

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

    def _raise(self, r: requests.Response) -> None:
        try:
            r.raise_for_status()
        except requests.exceptions.HTTPError:
            print(r.text)
            raise

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
        self._raise(r)
        return r.json()["id"]

    def upload_text(self, kb_id: int, filename: str, content: str) -> list[dict]:
        r = self.session.post(
            f"{self.base}/knowledge-base/{kb_id}/documents/upload",
            files=[("files", (filename, content.encode(), "text/plain"))],
            headers={"Authorization": f"Bearer {self.token}"},
            timeout=self.timeout,
        )
        self._raise(r)
        return r.json()

    def trigger_processing(self, kb_id: int, upload_results: list[dict]) -> dict:
        r = self.session.post(
            f"{self.base}/knowledge-base/{kb_id}/documents/process",
            json=upload_results,
            headers=self._headers(),
            timeout=self.timeout,
        )
        self._raise(r)
        return r.json()

    def ingest_status(self, kb_id: int) -> dict:
        r = self.session.get(
            f"{self.base}/query/kb/{kb_id}/ingest-status",
            headers=self._headers(),
            timeout=self.timeout,
        )
        self._raise(r)
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
        self._raise(r)
        return r.json()
