"""GPU benchmark: batch sizes, CPU-vs-GPU throughput, consistency."""

import json
import time
from pathlib import Path

import numpy as np
import tiktoken

from scientific_rag.data.dataset import load_dataset
from scientific_rag.retrieval.baselines import (
    DenseRetriever,
    get_embedding_model,
    resolve_device,
)

ENC = tiktoken.get_encoding("cl100k_base")
RESULTS = Path("results/gpu_benchmark.json")


def get_sample_texts(n=2000):
    ds = load_dataset("validation")
    texts = []
    for p in ds.papers:
        for s in p.full_text.sections:
            texts.extend(s.paragraphs)
        if len(texts) >= n:
            break
    return texts[:n]


def bench_encode(model, texts, batch_size, device):
    """Uncached encode; returns (seconds, passages/sec, peak_vram_bytes)."""
    import torch

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    emb = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    if device == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() if device == "cuda" else 0
    return dt, len(texts) / dt, peak, emb


def main():
    texts = get_sample_texts(2000)
    print(f"benchmarking on {len(texts)} real validation paragraphs")
    out = {
        "n_texts": len(texts),
        "model": DenseRetriever.MODEL_NAME,
        "device_available": resolve_device("auto"),
        "runs": {},
    }

    for dev in ["cpu", "cuda"]:
        if dev == "cuda":
            import torch

            if not torch.cuda.is_available():
                print("CUDA unavailable; skipping GPU")
                continue
        model = get_embedding_model(dev)
        # warmup
        model.encode(texts[:32], batch_size=32, normalize_embeddings=True, show_progress_bar=False)
        for bs in [16, 32, 64, 128] if dev == "cuda" else [64]:
            dt, tput, peak, emb = bench_encode(model, texts, bs, dev)
            key = f"{dev}_bs{bs}"
            out["runs"][key] = {
                "seconds": round(dt, 2),
                "passages_per_sec": round(tput, 1),
                "peak_vram_mb": round(peak / 1e6, 1) if peak else 0,
                "emb_shape": list(emb.shape),
            }
            print(key, out["runs"][key])
        # keep last emb for consistency check
        if dev == "cpu":
            emb_cpu = emb
        else:
            emb_gpu = emb

    # consistency: cosine sims CPU vs GPU on a query set
    queries = [
        "What is the main contribution of this paper?",
        "How does the model handle rare entities?",
        "What dataset was used for evaluation?",
    ]
    mc = get_embedding_model("cpu")
    mg = get_embedding_model("cuda")
    qc = mc.encode(queries, normalize_embeddings=True, convert_to_numpy=True)
    qg = mg.encode(queries, normalize_embeddings=True, convert_to_numpy=True)
    sims_c = emb_cpu @ qc.T
    sims_g = emb_gpu @ qg.T
    max_abs = float(np.max(np.abs(sims_c - sims_g)))
    rank_same = all(
        set(np.argsort(-sims_c[:, i])[:20]) == set(np.argsort(-sims_g[:, i])[:20])
        for i in range(len(queries))
    )
    out["consistency"] = {"max_abs_sim_diff": max_abs, "top20_overlap_all_queries": bool(rank_same)}
    print("consistency:", out["consistency"])

    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps(out, indent=2))
    print("saved", RESULTS)


if __name__ == "__main__":
    main()
