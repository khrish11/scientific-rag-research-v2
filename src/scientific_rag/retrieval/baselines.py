"""Retrieval baselines: BM25, dense, weighted hybrid, RRF (Phase 2).

Single-paper, paragraph-level retrieval. Frozen protocol:
- Corpus unit = original paragraph (paragraph-level representation).
- Retriever inputs: question text + paper paragraphs ONLY.
- No gold answers/evidence/labels anywhere in retrieval inputs.
- Deterministic: fixed seeds, stable tie-breaking by passage_id.
"""

import hashlib
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tiktoken

from scientific_rag.data.schemas import Paper
from scientific_rag.retrieval.evidence import PassageIdentifier

logger = logging.getLogger(__name__)

TOKENIZER_NAME = "cl100k_base"
TOKEN_RE = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    """Documented tokenization: lowercase word tokens (\\w+)."""
    return TOKEN_RE.findall(text.lower())


@dataclass
class Passage:
    """A retrievable paragraph passage."""

    passage_id: PassageIdentifier
    text: str
    section_name: str
    paragraph_index: int
    token_count: int = 0


def build_passages(paper: Paper, tokenizer: tiktoken.Encoding | None = None) -> list[Passage]:
    """Build paragraph-level passages for a paper."""
    enc = tokenizer or tiktoken.get_encoding(TOKENIZER_NAME)
    passages = []
    for section in paper.full_text.sections:
        for idx, para in enumerate(section.paragraphs):
            pid = PassageIdentifier(
                paper_id=paper.id,
                section_name=section.section_name or "",
                paragraph_index=idx,
                passage_type="paragraph",
            )
            passages.append(
                Passage(
                    passage_id=pid,
                    text=para,
                    section_name=section.section_name or "",
                    paragraph_index=idx,
                    token_count=len(enc.encode(para)),
                )
            )
    return passages


# ---------------------------------------------------------------------------
# BM25 (Okapi, implemented directly; deterministic)
# ---------------------------------------------------------------------------


class BM25Retriever:
    """Okapi BM25 over a fixed passage list. k1, b configurable."""

    def __init__(self, passages: list[Passage], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.passages = passages
        self._docs = [tokenize(p.text) for p in passages]
        self._tf = [Counter_local(d) for d in self._docs]
        self._dl = [len(d) for d in self._docs]
        self._avgdl = sum(self._dl) / max(len(self._dl), 1)
        n = len(passages)
        df: dict[str, int] = {}
        for tf in self._tf:
            for term in tf:
                df[term] = df.get(term, 0) + 1
        self._idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, k: int = 20) -> list[tuple[Passage, float]]:
        """Return top-k (passage, score); deterministic tie-break by passage_id."""
        q_terms = tokenize(query)
        scores = []
        for i, p in enumerate(self.passages):
            tf = self._tf[i]
            score = 0.0
            for t in q_terms:
                if t not in tf:
                    continue
                f = tf[t]
                denom = f + self.k1 * (1 - self.b + self.b * self._dl[i] / self._avgdl)
                score += self._idf.get(t, 0.0) * (f * (self.k1 + 1)) / denom
            if score > 0:
                scores.append((p, score))
        scores.sort(key=lambda x: (-x[1], str(x[0].passage_id)))
        return scores[:k]


def Counter_local(seq: list[str]) -> dict[str, int]:
    c: dict[str, int] = {}
    for x in seq:
        c[x] = c.get(x, 0) + 1
    return c


# ---------------------------------------------------------------------------
# Dense retrieval (sentence-transformers, cached, CPU)
# ---------------------------------------------------------------------------

_MODEL_CACHE: dict[str, Any] = {}
_MODEL_REVISIONS: dict[str, str] = {}


def resolve_device(device: str) -> str:
    """Resolve 'auto'|'cpu'|'cuda' to a concrete device string."""
    if device == "auto":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return device


def get_embedding_model(device: str = "cpu") -> Any:
    """Load the embedding model once per device per process."""
    device = resolve_device(device)
    if device not in _MODEL_CACHE:
        from sentence_transformers import SentenceTransformer

        _MODEL_CACHE[device] = SentenceTransformer(DenseRetriever.MODEL_NAME, device=device)
        card = getattr(_MODEL_CACHE[device], "_model_card_vars", {}) or {}
        _MODEL_REVISIONS[device] = card.get("revision", "unknown")
    return _MODEL_CACHE[device]


class DenseRetriever:
    """Cosine-similarity dense retrieval with all-MiniLM-L6-v2."""

    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(
        self,
        passages: list[Passage],
        cache_dir: Path | None = None,
        device: str = "auto",
        batch_size: int = 64,
    ):
        self.passages = passages
        self.device = resolve_device(device)
        self.batch_size = batch_size
        self.model = get_embedding_model(self.device)
        self.model_revision = _MODEL_REVISIONS.get(self.device, "unknown")
        cache_dir = cache_dir or Path("data/processed/embeddings")
        cache_dir.mkdir(parents=True, exist_ok=True)
        # cache key includes model, device, batch size and content hash so
        # device/batch changes invalidate automatically
        key = hashlib.sha256(
            (
                self.MODEL_NAME
                + "|"
                + self.device
                + "|"
                + str(self.batch_size)
                + "|"
                + "|".join(p.text for p in passages)
            ).encode()
        ).hexdigest()[:16]
        self.cache_path = cache_dir / f"{key}.npz"
        self.truncation_count = 0
        self._load_or_encode()

    def _load_or_encode(self) -> None:
        if self.cache_path.exists():
            z = np.load(self.cache_path)
            self.doc_emb = z["emb"]
            logger.info("Loaded cached embeddings %s", self.cache_path)
            return
        # model max_seq_length=128 wordpieces; detect truncation
        max_len = int(self.model.max_seq_length)
        texts = [p.text for p in self.passages]
        self.truncation_count = sum(1 for t in texts if len(tokenize(t)) > max_len)
        if self.truncation_count:
            logger.warning(
                "%d/%d passages exceed model context (%d tokens); "
                "sentence-transformers truncates silently - documented",
                self.truncation_count,
                len(texts),
                max_len,
            )
        self.doc_emb = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        np.savez_compressed(self.cache_path, emb=self.doc_emb)

    def search(self, query: str, k: int = 20) -> list[tuple[Passage, float]]:
        q_emb = self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
        sims = self.doc_emb @ q_emb
        order = np.argsort(-sims, kind="stable")
        out = []
        for i in order[:k]:
            out.append((self.passages[int(i)], float(sims[i])))
        # deterministic tie-break: sort by (-score, passage_id)
        out.sort(key=lambda x: (-x[1], str(x[0].passage_id)))
        return out


# ---------------------------------------------------------------------------
# Hybrid + RRF
# ---------------------------------------------------------------------------


def minmax(vals: list[float]) -> list[float]:
    """Min-max normalize; constant input -> all zeros."""
    if not vals:
        return []
    lo, hi = min(vals), max(vals)
    if hi == lo:
        return [0.0] * len(vals)
    return [(v - lo) / (hi - lo) for v in vals]


def hybrid_search(
    sparse_results: list[tuple[Passage, float]],
    dense_results: list[tuple[Passage, float]],
    alpha: float,
    k: int = 20,
    pool_size: int = 20,
) -> list[tuple[Passage, float]]:
    """hybrid = alpha*norm_dense + (1-alpha)*norm_sparse over union pool.

    Candidate pool = union of top-`pool_size` from each retriever.
    Missing candidates get score 0 for the absent retriever.
    """
    pool: dict[str, tuple[Passage, float, float]] = {}
    for p, s in sparse_results[:pool_size]:
        pool[str(p.passage_id)] = (p, s, 0.0)
    for p, s in dense_results[:pool_size]:
        key = str(p.passage_id)
        if key in pool:
            pool[key] = (pool[key][0], pool[key][1], s)
        else:
            pool[key] = (p, 0.0, s)

    keys = list(pool)
    sp = minmax([pool[k][1] for k in keys])
    dn = minmax([pool[k][2] for k in keys])
    scored = [(pool[k][0], alpha * dn[i] + (1 - alpha) * sp[i], k) for i, k in enumerate(keys)]
    scored.sort(key=lambda x: (-x[1], x[2]))
    return [(p, s) for p, s, _ in scored[:k]]


def rrf_search(
    ranked_lists: list[list[tuple[Passage, float]]],
    k: int = 20,
    rank_constant: int = 60,
) -> list[tuple[Passage, float]]:
    """Standard Reciprocal Rank Fusion: sum 1/(rank_constant + rank)."""
    scores: dict[str, tuple[Passage, float]] = {}
    for ranked in ranked_lists:
        for rank, (p, _) in enumerate(ranked):
            key = str(p.passage_id)
            add = 1.0 / (rank_constant + rank + 1)
            if key in scores:
                scores[key] = (scores[key][0], scores[key][1] + add)
            else:
                scores[key] = (p, add)
    out = sorted(scores.values(), key=lambda x: (-x[1], str(x[0].passage_id)))
    return out[:k]


# ---------------------------------------------------------------------------
# Context packing under token budget
# ---------------------------------------------------------------------------

SEPARATOR_TOKENS = 2  # "\n\n" via cl100k_base


def pack_budget(
    ranked: list[tuple[Passage, float]],
    budget_tokens: int,
    tokenizer: tiktoken.Encoding | None = None,  # noqa: ARG001 - reserved for serialized verification
) -> tuple[list[Passage], int]:
    """Greedy rank-order packing. Policy: skip a passage that does not fit
    the remaining budget and continue to the next (documented). Never exceed
    budget; never truncate-and-count-as-full."""
    selected: list[Passage] = []
    used = 0
    for p, _ in ranked:
        cost = p.token_count + (SEPARATOR_TOKENS if selected else 0)
        if used + cost <= budget_tokens:
            selected.append(p)
            used += cost
    return selected, used
