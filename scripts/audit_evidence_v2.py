"""Phase 1.1 evidence-mapping audit using the corrected API."""

import json
from collections import Counter
from pathlib import Path

from scientific_rag.data.dataset import load_dataset
from scientific_rag.retrieval.evidence import (
    annotation_evidence_sets,
    map_paper_evidence,
)

RESULTS = Path("results")


def audit_split(split: str) -> dict:
    ds = load_dataset(split)
    agg = Counter()
    failures_sample = []
    ambiguous_sample = []
    for paper in ds.papers:
        res = map_paper_evidence(paper, use_normalization=True, use_fuzzy=False)
        agg["total"] += res.total_evidence_count
        agg["mapped_unique"] += res.mapped_unique
        agg["mapped_ambiguous"] += res.mapped_ambiguous
        agg["unmatched"] += res.unmatched
        agg["non_textual"] += res.non_textual
        agg["exact"] += res.exact_matches
        agg["normalized"] += res.normalized_matches
        # fuzzy diagnostic pass on a sample only (expensive)
        if paper.id in {p.id for p in ds.papers[:30]}:
            res_f = map_paper_evidence(paper, use_fuzzy=True)
            agg["fuzzy_rescued"] += res_f.mapped_unique - res.mapped_unique
        for f in res.failures[:2]:
            if len(failures_sample) < 30:
                failures_sample.append(
                    {
                        "paper_id": f.paper_id,
                        "qid": f.question_id,
                        "ann": f.annotation_id,
                        "evidence": f.evidence_text[:200],
                        "reason": f.reason,
                    }
                )
        for m in res.mappings:
            if m.is_ambiguous and len(ambiguous_sample) < 30:
                ambiguous_sample.append(
                    {
                        "paper_id": m.candidate_passages[0].paper_id
                        if m.candidate_passages
                        else "",
                        "evidence": m.evidence_text[:150],
                        "candidates": len(m.candidate_passages),
                    }
                )

    # annotation-level evidence-set stats + completeness on paragraph retrieval
    n_sets = 0
    n_empty_sets = 0
    n_multi_ev_questions = 0
    for paper in ds.papers[:100]:
        for qa in paper.qas:
            sets = annotation_evidence_sets(qa)
            if len(sets) > 1:
                n_multi_ev_questions += 1
            for s in sets:
                n_sets += 1
                if not s:
                    n_empty_sets += 1

    total = agg["total"] or 1
    return {
        "split": split,
        "total_evidence_strings": agg["total"],
        "mapped_unique": agg["mapped_unique"],
        "mapped_ambiguous": agg["mapped_ambiguous"],
        "unmatched": agg["unmatched"],
        "non_textual": agg["non_textual"],
        "exact": agg["exact"],
        "normalized": agg["normalized"],
        "fuzzy_rescued_on_sample": agg["fuzzy_rescued"],
        "unique_rate": round(agg["mapped_unique"] / total, 4),
        "ambiguous_rate": round(agg["mapped_ambiguous"] / total, 4),
        "unmatched_rate": round(agg["unmatched"] / total, 4),
        "non_textual_rate": round(agg["non_textual"] / total, 4),
        "evidence_sets_sampled": n_sets,
        "empty_evidence_sets": n_empty_sets,
        "multi_annotation_questions_sample": n_multi_ev_questions,
        "failures_sample": failures_sample,
        "ambiguous_sample": ambiguous_sample,
    }


def main():
    out = {}
    for split in ["train", "validation"]:
        print(f"auditing {split}...", flush=True)
        out[split] = audit_split(split)
        print(
            json.dumps({k: v for k, v in out[split].items() if not k.endswith("_sample")}, indent=2)
        )
    Path(RESULTS / "evidence_mapping_audit.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    print("saved results/evidence_mapping_audit.json")


if __name__ == "__main__":
    main()
