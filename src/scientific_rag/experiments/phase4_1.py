"""Phase 4.1: Model A stopping, matched-context eval, reconciliation."""

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

RESULTS = Path("results")


def xy(rows, feats):
    X = np.array([[r[f] for f in feats] for r in rows], float)
    y = np.array([r["complete"] for r in rows], int)
    return X, y


def main():
    t0 = time.time()
    val = json.loads((RESULTS / "phase3_trajectories.json").read_text())
    # rebuild train trajectories quickly via phase4 builder
    from scientific_rag.data.dataset import load_dataset
    from scientific_rag.experiments.phase4_feasibility import build_trajectories
    part = json.loads((RESULTS / "dev_partition.json").read_text())
    train = load_dataset("train")
    idset = set(part["paper_ids"])
    dev = [p for p in train.papers if p.id in idset]
    tr = build_trajectories(dev, "train_dev")
    print("train states", len(tr), "val states", len(val))

    # Model A fit on train
    featsA = ["tokens", "n_selected"]
    Xtr, ytr = xy(tr, featsA)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    mA = LogisticRegression(max_iter=1000).fit((Xtr - mu) / sd, ytr)
    Xv, yv = xy(val, featsA)
    pv = mA.predict_proba((Xv - mu) / sd)[:, 1]
    papers = [r["paper"] for r in val]
    res = {"modelA": {"auroc": float(roc_auc_score(yv, pv)),
                      "prevalence": float(np.mean(yv)), "n": len(yv)}}

    # Model A stopping: threshold = median prob of train positives
    ptr = mA.predict_proba((Xtr - mu) / sd)[:, 1]
    thr = float(np.quantile(ptr[ytr == 1], 0.5))
    # apply on val trajectories grouped by question+method
    pmap = {}
    for r, p in zip(val, pv):
        pmap[(r["paper"], r["qid"], r["method"], r["step"])] = float(p)
    by_q = defaultdict(list)
    for r in val:
        by_q[(r["paper"], r["qid"], r["method"])].append(r)
    stops = []
    for key, steps in by_q.items():
        steps.sort(key=lambda s: s["step"])
        pick = None
        for st in steps:
            if pmap[(st["paper"], st["qid"], st["method"], st["step"])] >= thr:
                pick = st
                break
        pick = pick or steps[-1]
        stops.append({"paper": pick["paper"], "qid": pick["qid"],
                      "method": pick["method"], "complete": pick["complete"],
                      "tokens": pick["tokens"], "n_sel": pick["n_selected"],
                      "premature": float(pick["complete"] == 0 and
                                         any(s["complete"] == 1 for s in steps[steps.index(pick)+1:]))})
    agg = defaultdict(lambda: {"c": [], "t": [], "prem": []})
    for s in stops:
        agg[s["method"]]["c"].append(s["complete"])
        agg[s["method"]]["t"].append(s["tokens"])
        agg[s["method"]]["prem"].append(s["premature"])
    res["modelA_stopping"] = {m: {"complete": float(np.mean(v["c"])),
                                  "mean_tokens": float(np.mean(v["t"])),
                                  "premature_rate": float(np.mean(v["prem"])),
                                  "n": len(v["c"])} for m, v in agg.items()}
    res["threshold_A"] = thr

    # matched-context pairs: within same question+method, pair complete vs
    # incomplete states with |dtokens|<=25 and same n_selected
    pairs = []
    for key, steps in by_q.items():
        comp = [s for s in steps if s["complete"] == 1]
        incomp = [s for s in steps if s["complete"] == 0]
        for c in comp:
            for i in incomp:
                if abs(c["tokens"] - i["tokens"]) <= 25 and \
                   c["n_selected"] == i["n_selected"]:
                    pairs.append({"paper": c["paper"], "qid": c["qid"],
                                  "method": c["method"],
                                  "comp_tokens": c["tokens"],
                                  "incomp_tokens": i["tokens"],
                                  "n_sel": c["n_selected"],
                                  "comp_probA": pmap[(c["paper"], c["qid"], c["method"], c["step"])],
                                  "incomp_probA": pmap[(i["paper"], i["qid"], i["method"], i["step"])]})
                    break
            if len(pairs) > 3000:
                break
    if pairs:
        diffs = [p["comp_probA"] - p["incomp_probA"] for p in pairs]
        res["matched_pairs"] = {"n": len(pairs),
                                "mean_prob_diff_A": float(np.mean(diffs)),
                                "frac_comp_higher": float(np.mean([d > 0 for d in diffs]))}
    else:
        res["matched_pairs"] = {"n": 0}

    res["runtime_s"] = round(time.time() - t0, 1)
    (RESULTS / "phase4_1_results.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "matched_pairs"}, indent=1)[:2000])
    print("matched:", res.get("matched_pairs"))


if __name__ == "__main__":
    main()
