"""Phase 4 feasibility: eligibility audit + Models A/B + stopping eval.

Model A: logistic regression on (tokens, n_selected) only.
Model B: logistic regression on retrieval features (+ ablation w/o size).
Trained on train-dev trajectories (paper-disjoint, seed 42); evaluated on
validation trajectories (results/phase3_trajectories.json).
Gold labels offline only. Model C and faithful Adaptive-k NOT implemented
(reported incomplete).
"""

import json
import logging
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import tiktoken
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from scientific_rag.data.dataset import load_dataset
from scientific_rag.experiments.phase3_adaptive import (
    gold_sets_for,
    simulate,
)
from scientific_rag.retrieval.baselines import (
    BM25Retriever,
    DenseRetriever,
    build_passages,
    hybrid_search,
    rrf_search,
)
from scientific_rag.retrieval.evidence import is_non_textual_evidence

logging.basicConfig(level=logging.WARNING)
ENC = tiktoken.get_encoding("cl100k_base")
RESULTS = Path("results")
ALPHA, RRF_K, DEVICE = 0.9, 20, "cuda"

FEATURES_B = ["top_score", "margin", "score_decay", "rel_drop",
              "n_selected", "tokens"]


def build_trajectories(papers, tag):
    out = []
    for paper in papers:
        passages = build_passages(paper, ENC)
        if not passages:
            continue
        bm25 = BM25Retriever(passages)
        dense = DenseRetriever(passages, device=DEVICE, batch_size=32)
        for qa in paper.qas:
            gold = gold_sets_for(paper, passages, qa)
            if not gold:
                continue
            q = qa.question_text
            sp = bm25.search(q, k=len(passages))
            dn = dense.search(q, k=len(passages))
            hyb = hybrid_search(sp, dn, ALPHA, k=len(passages), pool_size=len(passages))
            rrf = rrf_search([sp, dn], k=len(passages), rank_constant=RRF_K)
            for mname, ranked in [("dense", dn), ("hybrid", hyb), ("rrf", rrf)]:
                for st in simulate(ranked, gold):
                    st.update({"paper": paper.id, "qid": qa.question_id,
                               "method": mname, "split": tag})
                    out.append(st)
    return out


def eligibility_audit():
    """Characterize excluded validation questions."""
    val = load_dataset("validation")
    reasons = Counter()
    eligible_types = Counter()
    for paper in val.papers:
        passages = build_passages(paper, ENC)
        for qa in paper.qas:
            n_ann = len(qa.answers)
            any_unans = any(a.answer.unanswerable for a in qa.answers)
            sets = gold_sets_for(paper, passages, qa)
            if sets:
                eligible_types[qa.answers[0].answer.answer_type] += 1
                continue
            # classify exclusion reason
            has_textual = False
            has_nontext_only = False
            for ann in qa.answers:
                ev = ann.answer.evidence + ann.answer.highlighted_evidence
                textual = [e for e in ev if not is_non_textual_evidence(e)]
                if textual:
                    has_textual = True
                elif ev:
                    has_nontext_only = True
            if any_unans and n_ann == sum(1 for a in qa.answers if a.answer.unanswerable):
                reasons["all_unanswerable"] += 1
            elif has_nontext_only and not has_textual:
                reasons["nontextual_only"] += 1
            elif has_textual:
                reasons["unmapped_textual"] += 1
            else:
                reasons["no_evidence"] += 1
    return {"exclusion_reasons": dict(reasons),
            "eligible_by_type": dict(eligible_types)}


def xy(rows, feats):
    X = np.array([[r[f] for f in feats] for r in rows], dtype=float)
    y = np.array([r["complete"] for r in rows], dtype=int)
    return X, y


def eval_model(name, model, X, y, group_papers):
    proba = model.predict_proba(X)[:, 1]
    res = {"auroc": float(roc_auc_score(y, proba)),
           "auprc": float(average_precision_score(y, proba)),
           "brier": float(brier_score_loss(y, proba)),
           "prevalence": float(np.mean(y)), "n": len(y)}
    # paper-cluster bootstrap on AUROC
    papers = np.array(group_papers)
    uniq = np.unique(papers)
    rng = np.random.default_rng(42)
    aucs = []
    for _ in range(2000):
        idx = rng.choice(uniq, size=len(uniq), replace=True)
        mask = np.isin(papers, idx)
        if len(np.unique(y[mask])) < 2:
            continue
        aucs.append(roc_auc_score(y[mask], proba[mask]))
    res["auroc_ci95"] = [float(np.percentile(aucs, 2.5)),
                         float(np.percentile(aucs, 97.5))] if aucs else None
    res["proba"] = proba.tolist()
    return res


def stop_eval(rows, proba_map, threshold, max_tokens=4096):
    """Stopping policy: stop at first step where P(complete) >= threshold.

    proba_map: (paper,qid,method,step)->prob. Returns per-question rows."""
    # group trajectory steps by question+method
    by_q = defaultdict(list)
    for r in rows:
        by_q[(r["paper"], r["qid"], r["method"])].append(r)
    out = []
    for (paper, qid, method), steps in by_q.items():
        steps.sort(key=lambda s: s["step"])
        chosen = None
        for st in steps:
            p = proba_map.get((paper, qid, method, st["step"]))
            if p is not None and p >= threshold:
                chosen = st
                break
        if chosen is None:
            chosen = steps[-1]  # budget/trajectory exhausted
        out.append({"paper": paper, "qid": qid, "method": method,
                    "complete": chosen["complete"], "tokens": chosen["tokens"],
                    "n_sel": chosen["n_selected"],
                    "stop_reason": "predicted" if chosen is not steps[-1] else "exhausted"})
    return out


def main():
    t0 = time.time()
    # 1. eligibility audit on validation
    elig = eligibility_audit()
    print("eligibility:", elig)

    # 2. train trajectories on dev partition (60 train papers)
    part = json.loads((RESULTS / "dev_partition.json").read_text())
    train = load_dataset("train")
    idset = set(part["paper_ids"])
    dev_papers = [p for p in train.papers if p.id in idset]
    print("building train trajectories on", len(dev_papers), "papers")
    train_traj = build_trajectories(dev_papers, "train_dev")
    print("train trajectory states:", len(train_traj))

    # 3. load validation trajectories
    val_traj = json.loads((RESULTS / "phase3_trajectories.json").read_text())
    print("val trajectory states:", len(val_traj))

    # 4. train Model A and B on dense trajectories only? Use all methods pooled
    #    but evaluate per method. Train on all methods pooled (documented).
    tr = [r for r in train_traj]
    va = [r for r in val_traj]

    models = {}
    for name, feats in [("A_size", ["tokens", "n_selected"]),
                        ("B_full", FEATURES_B),
                        ("B_nosize", ["top_score", "margin", "score_decay",
                                      "rel_drop"])]:
        Xtr, ytr = xy(tr, feats)
        m = LogisticRegression(max_iter=1000, C=1.0)
        m.fit(Xtr, ytr)
        models[name] = (m, feats)

    # standardize features using train stats
    # (LogisticRegression on raw scales is OK-ish; do quick standardization)
    results = {}
    for name, (m, feats) in models.items():
        Xva, yva = xy(va, feats)
        Xtr, ytr = xy(tr, feats)
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        m2 = LogisticRegression(max_iter=1000, C=1.0)
        m2.fit((Xtr - mu) / sd, ytr)
        pv = m2.predict_proba((Xva - mu) / sd)[:, 1]
        papers = [r["paper"] for r in va]
        res = {"auroc": float(roc_auc_score(yva, pv)),
               "auprc": float(average_precision_score(yva, pv)),
               "brier": float(brier_score_loss(yva, pv)),
               "prevalence": float(np.mean(yva)), "n": len(yva)}
        # bootstrap AUROC
        up = np.unique(papers); rng = np.random.default_rng(42); aucs = []
        yv = np.array(yva)
        for _ in range(2000):
            idx = rng.choice(up, size=len(up), replace=True)
            mask = np.isin(papers, idx)
            if len(np.unique(yv[mask])) > 1:
                aucs.append(roc_auc_score(yv[mask], pv[mask]))
        res["auroc_ci95"] = [float(np.percentile(aucs, 2.5)),
                             float(np.percentile(aucs, 97.5))]
        results[name] = res
        print(name, {k: v for k, v in res.items() if k != "proba"})

    # per-method AUROC for Model B
    per_method = {}
    mB, fB = models["B_full"]
    Xtr, ytr = xy(tr, fB); mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    mB2 = LogisticRegression(max_iter=1000).fit((Xtr - mu) / sd, ytr)
    for meth in ["dense", "hybrid", "rrf"]:
        sub = [i for i, r in enumerate(va) if r["method"] == meth]
        Xs = np.array([[va[i][f] for f in fB] for i in sub])
        ys = np.array([va[i]["complete"] for i in sub])
        ps = mB2.predict_proba((Xs - mu) / sd)[:, 1]
        per_method[meth] = {"auroc": float(roc_auc_score(ys, ps)),
                            "n": len(sub)}

    # 5. stopping comparison: thresholds from train (median prob of complete states)
    # pick threshold = train prevalence quantile; evaluate token/recovery trade-off
    XtrB, ytrB = xy(tr, fB)
    ptr = mB2.predict_proba((XtrB - mu) / sd)[:, 1]
    thr = float(np.quantile(ptr[ytrB == 1], 0.5))  # median prob of positives
    # build prob map for val
    pmap = {}
    pv_all = mB2.predict_proba((np.array([[r[f] for f in fB] for r in va]) - mu) / sd)[:, 1]
    for r, p in zip(va, pv_all):
        pmap[(r["paper"], r["qid"], r["method"], r["step"])] = float(p)
    stop_rows = stop_eval(va, pmap, thr)
    # aggregate by method
    stop_agg = defaultdict(lambda: {"complete": [], "tokens": []})
    for r in stop_rows:
        stop_agg[r["method"]]["complete"].append(r["complete"])
        stop_agg[r["method"]]["tokens"].append(r["tokens"])
    stop_summary = {m: {"complete": float(np.mean(v["complete"])),
                        "mean_tokens": float(np.mean(v["tokens"])),
                        "n": len(v["complete"])}
                    for m, v in stop_agg.items()}

    out = {"eligibility": elig, "n_train_states": len(tr),
           "n_val_states": len(va), "train_runtime_s": round(time.time() - t0, 1),
           "models": results, "per_method_B": per_method,
           "threshold_B": thr, "stopping_B": stop_summary,
           "incomplete": ["Model C semantic probe", "faithful Adaptive-k",
                          "matched-pair eval", "calibration metrics (ECE)",
                          "operational non-inferiority test"]}
    (RESULTS / "phase4_feasibility.json").write_text(json.dumps(out, indent=2))
    print("DONE", out["train_runtime_s"], "s")


if __name__ == "__main__":
    main()
