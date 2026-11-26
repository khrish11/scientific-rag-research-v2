"""Generate Phase 2.1 research figures from experiment outputs."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path("results")
FIGDIR = RESULTS / "figures"
FIGDIR.mkdir(exist_ok=True)


def load():
    agg = json.loads((RESULTS / "phase2_full_results.json").read_text())["aggregate"]
    manifest = json.loads((RESULTS / "tuning_manifest.json").read_text())
    rows = json.loads((RESULTS / "phase2_per_question.json").read_text())
    return agg, manifest, rows


def series(agg, method, mode_key):
    xs, cs, rs = [], [], []
    for k, v in agg.items():
        m, kk, md = k.split("|")
        if m == method and md == mode_key and kk.replace(".", "").isdigit():
            xs.append(float(kk))
            cs.append(v["complete"]["mean"])
            rs.append(v["r"]["mean"])
    o = np.argsort(xs)
    return [np.array(xs)[o], np.array(cs)[o], np.array(rs)[o]]


def fig_budget_recovery(agg):
    fig, ax = plt.subplots(figsize=(6, 4))
    for m, c in [("bm25", "C0"), ("dense", "C1"), ("hybrid", "C2"), ("rrf", "C3")]:
        x, cpl, _ = series(agg, m, "budget")
        ax.plot(x, cpl, "-o", color=c, label=m, ms=4)
    ax.set_xlabel("max context tokens")
    ax.set_ylabel("complete annotated-evidence recovery")
    ax.set_xscale("log", base=2)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGDIR / "recovery_vs_budget.png", dpi=150)
    fig.savefig(FIGDIR / "recovery_vs_budget.pdf")
    plt.close(fig)


def fig_recall_budget(agg):
    fig, ax = plt.subplots(figsize=(6, 4))
    for m, c in [("bm25", "C0"), ("dense", "C1"), ("hybrid", "C2"), ("rrf", "C3")]:
        x, _, r = series(agg, m, "budget")
        ax.plot(x, r, "-o", color=c, label=m, ms=4)
    ax.set_xlabel("max context tokens")
    ax.set_ylabel("evidence recall")
    ax.set_xscale("log", base=2)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGDIR / "recall_vs_budget.png", dpi=150)
    fig.savefig(FIGDIR / "recall_vs_budget.pdf")
    plt.close(fig)


def fig_alpha(manifest):
    xs = sorted(float(k) for k in manifest["alpha_scores"])
    ys = [manifest["alpha_scores"][str(x)] for x in xs]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, ys, "-o", color="C0")
    ax.axvline(manifest["alpha"], color="r", ls="--", label=f"selected {manifest['alpha']}")
    ax.set_xlabel("dense weight alpha")
    ax.set_ylabel("complete recovery @1024 (dev)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGDIR / "alpha_sensitivity.png", dpi=150)
    fig.savefig(FIGDIR / "alpha_sensitivity.pdf")
    plt.close(fig)


def fig_rrf(manifest):
    xs = sorted(manifest["rrf_scores"])
    ys = [manifest["rrf_scores"][k] for k in xs]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(xs, ys, "-o", color="C3")
    ax.set_xlabel("RRF rank constant k")
    ax.set_ylabel("complete recovery @1024 (dev)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGDIR / "rrf_sensitivity.png", dpi=150)
    fig.savefig(FIGDIR / "rrf_sensitivity.pdf")
    plt.close(fig)


def fig_latency(rows):
    lat = {}
    for r in rows:
        lat.setdefault(r["method"], []).append(r["latency_ms"])
    fig, ax = plt.subplots(figsize=(6, 4))
    ms = sorted(lat)
    ax.bar(ms, [np.mean(lat[m]) for m in ms], color="C0")
    ax.set_ylabel("mean online latency (ms)")
    ax.set_xlabel("method")
    fig.tight_layout()
    fig.savefig(FIGDIR / "latency.png", dpi=150)
    fig.savefig(FIGDIR / "latency.pdf")
    plt.close(fig)


def fig_oracle(rows):
    # oracle_possible rate vs actual complete per method at each budget
    fig, ax = plt.subplots(figsize=(7, 4))
    for m, c in [("bm25", "C0"), ("dense", "C1"), ("hybrid", "C2"), ("rrf", "C3")]:
        pts = {}
        for r in rows:
            if r["method"] == m and r.get("mode") == "budget" and "oracle_possible" in r:
                pts.setdefault(r["budget"], []).append(r)
        xs = sorted(pts)
        ys = [np.mean([x["complete"] for x in pts[b]]) for b in xs]
        op = [np.mean([x["oracle_possible"] for x in pts[b]]) for b in xs]
        ax.plot(xs, ys, "-o", color=c, label=m, ms=4)
        ax.plot(xs, op, "--", color=c, alpha=0.5)
    ax.set_xlabel("budget")
    ax.set_ylabel("complete recovery (solid) / oracle-possible (dashed)")
    ax.set_xscale("log", base=2)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGDIR / "oracle_vs_actual.png", dpi=150)
    fig.savefig(FIGDIR / "oracle_vs_actual.pdf")
    plt.close(fig)


def fig_by_setsize(rows):
    # group by n_rel buckets
    fig, ax = plt.subplots(figsize=(6, 4))
    for m, c in [("bm25", "C0"), ("dense", "C1"), ("rrf", "C3")]:
        buckets = {}
        for r in rows:
            if r["method"] == m and r.get("mode") == "budget" and r.get("budget") == 1024:
                b = min(r.get("n_rel", 1), 4)
                buckets.setdefault(b, []).append(r["complete"])
        xs = sorted(buckets)
        ys = [np.mean(buckets[x]) for x in xs]
        ax.plot(xs, ys, "-o", color=c, label=m, ms=4)
    ax.set_xlabel("gold evidence-set size (capped at 4)")
    ax.set_ylabel("complete recovery @1024")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGDIR / "by_evidence_setsize.png", dpi=150)
    fig.savefig(FIGDIR / "by_evidence_setsize.pdf")
    plt.close(fig)


def main():
    agg, manifest, rows = load()
    fig_budget_recovery(agg)
    fig_recall_budget(agg)
    fig_alpha(manifest)
    fig_rrf(manifest)
    fig_latency(rows)
    fig_oracle(rows)
    fig_by_setsize(rows)
    print("figures written to", FIGDIR)


if __name__ == "__main__":
    main()
