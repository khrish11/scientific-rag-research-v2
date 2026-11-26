"""Phase 1.1 audit: dataset verification, evidence mapping, chunking comparison."""

import json
import logging
from collections import Counter
from pathlib import Path

from scientific_rag.data.dataset import load_dataset
from scientific_rag.retrieval.chunking import Chunker, ChunkingConfig
from scientific_rag.retrieval.evidence import (
    map_evidence_to_passages,
    normalize_text,
)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
RESULTS_DIR = Path("results")


def split_counts():
    """Verify raw JSON counts for all splits."""
    counts = {}
    for name, fname in [
        ("train", "qasper-train-v0.3.json"),
        ("validation", "qasper-dev-v0.3.json"),
        ("test", "qasper-test-v0.3.json"),
    ]:
        with open(RAW_DIR / fname, encoding="utf-8") as f:
            d = json.load(f)
        nq = sum(len(p.get("qas", [])) for p in d.values())
        counts[name] = {"papers": len(d), "questions": nq}
    return counts


def answer_level_stats(dataset):
    """Compute question-level and annotation-level stats."""
    stats = {
        "num_papers": len(dataset.papers),
        "num_questions": 0,
        "num_answer_annotations": 0,
        "answer_type_question_level": Counter(),
        "answer_type_annotation_level": Counter(),
        "questions_multi_annotation": 0,
        "questions_all_unanswerable": 0,
        "questions_mixed_answerability": 0,
        "questions_any_unanswerable": 0,
        "questions_no_answers": 0,
        "evidence_text_count": 0,
        "evidence_highlighted_count": 0,
        "evidence_empty_annotations": 0,
        "questions_with_evidence": 0,
        "questions_multi_evidence": 0,
        "yes_no_false_count": 0,
        "yes_no_true_count": 0,
        "yes_no_str_count": 0,
        "malformed": Counter(),
    }

    for paper in dataset.papers:
        for qa in paper.qas:
            stats["num_questions"] += 1
            n_ans = len(qa.answers)
            stats["num_answer_annotations"] += n_ans

            if n_ans == 0:
                stats["questions_no_answers"] += 1
                stats["malformed"]["no_answers"] += 1
                continue

            if n_ans > 1:
                stats["questions_multi_annotation"] += 1

            # Question-level type: use first answer's type (documented rule)
            qtype = qa.answers[0].answer.answer_type
            stats["answer_type_question_level"][qtype] += 1

            # Annotation-level types
            unans_flags = []
            has_evidence = False
            q_evidence_count = 0
            for ann in qa.answers:
                ad = ann.answer
                stats["answer_type_annotation_level"][ad.answer_type] += 1
                unans_flags.append(ad.unanswerable)

                # yes/no handling
                if ad.yes_no is not None:
                    if ad.yes_no is True:
                        stats["yes_no_true_count"] += 1
                    elif ad.yes_no is False:
                        stats["yes_no_false_count"] += 1
                    else:
                        stats["yes_no_str_count"] += 1

                ev = len(ad.evidence)
                hl = len(ad.highlighted_evidence)
                stats["evidence_text_count"] += ev
                stats["evidence_highlighted_count"] += hl
                q_evidence_count += ev + hl
                if ev + hl == 0:
                    stats["evidence_empty_annotations"] += 1
                if ev + hl > 0:
                    has_evidence = True

            if all(unans_flags):
                stats["questions_all_unanswerable"] += 1
            if any(unans_flags):
                stats["questions_any_unanswerable"] += 1
            if any(unans_flags) and not all(unans_flags):
                stats["questions_mixed_answerability"] += 1
            if has_evidence:
                stats["questions_with_evidence"] += 1
            if q_evidence_count > 1:
                stats["questions_multi_evidence"] += 1

    # Convert counters
    for k in ["answer_type_question_level", "answer_type_annotation_level", "malformed"]:
        stats[k] = dict(stats[k])
    return stats


def evidence_mapping_audit(dataset, split_name):
    """Run evidence mapping with strict vs fuzzy modes."""
    results = {
        "split": split_name,
        "strict": {
            "exact": 0,
            "normalized": 0,
            "fuzzy": 0,
            "unmatched": 0,
            "total": 0,
            "ambiguous": 0,
        },
        "fuzzy_mode": {
            "exact": 0,
            "normalized": 0,
            "fuzzy": 0,
            "unmatched": 0,
            "total": 0,
            "ambiguous": 0,
        },
        "non_textual": 0,
        "empty_evidence": 0,
        "failures_sample": [],
        "success_sample": [],
    }

    for paper in dataset.papers:
        res_strict = map_evidence_to_passages(paper, use_normalization=True, use_fuzzy=False)
        res_fuzzy = map_evidence_to_passages(paper, use_normalization=True, use_fuzzy=True)

        for key, res in [("strict", res_strict), ("fuzzy_mode", res_fuzzy)]:
            results[key]["exact"] += res.exact_matches
            results[key]["normalized"] += res.normalized_matches
            results[key]["fuzzy"] += res.fuzzy_matches
            results[key]["unmatched"] += res.failed_mappings
            results[key]["total"] += res.total_evidence_count

        # Sample failures for inspection
        for f in res_strict.failures[:3]:
            if len(results["failures_sample"]) < 20:
                results["failures_sample"].append(
                    {
                        "paper_id": f.paper_id,
                        "evidence": f.evidence_text[:200],
                        "reason": f.reason,
                    }
                )
        for m in res_strict.mappings[:3]:
            if len(results["success_sample"]) < 20:
                results["success_sample"].append(
                    {
                        "paper_id": m.passage_id.paper_id,
                        "evidence": m.evidence_text[:150],
                        "match_type": m.match_type,
                        "section": m.passage_id.section_name,
                        "para": m.passage_id.paragraph_index,
                    }
                )

    # Count non-textual and empty evidence
    for paper in dataset.papers:
        for qa in paper.qas:
            for ann in qa.answers:
                ad = ann.answer
                for ev in ad.evidence + ad.highlighted_evidence:
                    ev_lower = ev.lower()
                    if any(
                        t in ev_lower
                        for t in ["figure", "table", "fig.", "tab.", "bibr", "tabref", "figref"]
                    ):
                        results["non_textual"] += 1
                if len(ad.evidence) + len(ad.highlighted_evidence) == 0:
                    results["empty_evidence"] += 1

    return results


def chunking_comparison(dataset, split_name):
    """Compare chunking configurations."""
    configs = [
        ("paragraph", ChunkingConfig(chunk_size=100000, chunk_overlap=0)),
        ("chunk128", ChunkingConfig(chunk_size=128, chunk_overlap=16)),
        ("chunk256", ChunkingConfig(chunk_size=256, chunk_overlap=32)),
        ("chunk512", ChunkingConfig(chunk_size=512, chunk_overlap=64)),
    ]

    comparison = {"split": split_name, "configs": {}}

    for name, cfg in configs:
        chunker = Chunker(cfg)
        total_chunks = 0
        total_tokens = 0
        token_lengths = []
        evidence_covered = 0
        evidence_total = 0
        evidence_partial = 0

        for paper in dataset.papers[:50]:  # Sample 50 papers for speed
            if name == "paragraph":
                result = chunker.chunk_paragraph_level(paper)
            else:
                result = chunker.chunk_paper(paper)

            total_chunks += result.total_chunks
            total_tokens += result.total_tokens
            token_lengths.extend([c.token_count for c in result.chunks])

            # Check evidence coverage: can evidence text be found in chunks?
            for qa in paper.qas:
                for ann in qa.answers:
                    for ev_text in ann.answer.evidence + ann.answer.highlighted_evidence:
                        evidence_total += 1
                        norm_ev = normalize_text(ev_text)
                        found_full = False
                        found_partial = False
                        for chunk in result.chunks:
                            norm_chunk = normalize_text(chunk.text)
                            if norm_ev == norm_chunk or norm_ev in norm_chunk:
                                found_full = True
                                break
                            elif any(len(norm_ev) > 20 and norm_ev[:50] in norm_chunk for _ in [0]):
                                found_partial = True
                        if found_full:
                            evidence_covered += 1
                        elif found_partial:
                            evidence_partial += 1

        if token_lengths:
            token_lengths.sort()
            comparison["configs"][name] = {
                "total_chunks": total_chunks,
                "total_tokens": total_tokens,
                "mean_chunk_tokens": sum(token_lengths) / len(token_lengths),
                "median_chunk_tokens": token_lengths[len(token_lengths) // 2],
                "max_chunk_tokens": max(token_lengths),
                "evidence_covered": evidence_covered,
                "evidence_partial": evidence_partial,
                "evidence_total": evidence_total,
                "coverage_rate": evidence_covered / evidence_total if evidence_total else 0,
            }

    return comparison


def check_split_isolation():
    """Verify paper-level isolation across splits."""
    splits = {}
    for name in ["train", "validation", "test"]:
        ds = load_dataset(name)
        splits[name] = {p.id for p in ds.papers}

    overlaps = {}
    for s1 in splits:
        for s2 in splits:
            if s1 < s2:
                ov = splits[s1] & splits[s2]
                if ov:
                    overlaps[f"{s1}-{s2}"] = len(ov)

    return {
        "train_size": len(splits["train"]),
        "validation_size": len(splits["validation"]),
        "test_size": len(splits["test"]),
        "overlaps": overlaps,
        "is_isolated": len(overlaps) == 0,
    }


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    # 1. Split counts
    counts = split_counts()
    print("=== SPLIT COUNTS ===")
    for k, v in counts.items():
        print(f"{k}: {v['papers']} papers, {v['questions']} questions")

    # 2. Load datasets and compute stats
    all_stats = {}
    for split in ["train", "validation", "test"]:
        print(f"\n=== LOADING {split.upper()} ===")
        ds = load_dataset(split)
        stats = answer_level_stats(ds)
        all_stats[split] = stats
        print(
            f"Papers: {stats['num_papers']}, Questions: {stats['num_questions']}, "
            f"Annotations: {stats['num_answer_annotations']}"
        )
        print(f"Q-level types: {stats['answer_type_question_level']}")
        print(f"A-level types: {stats['answer_type_annotation_level']}")
        print(
            f"Unanswerable: all={stats['questions_all_unanswerable']}, "
            f"any={stats['questions_any_unanswerable']}, "
            f"mixed={stats['questions_mixed_answerability']}"
        )
        print(
            f"yes/no: true={stats['yes_no_true_count']}, false={stats['yes_no_false_count']}, "
            f"str={stats['yes_no_str_count']}"
        )

    # 3. Evidence mapping on train + validation
    print("\n=== EVIDENCE MAPPING ===")
    mapping_results = {}
    for split in ["train", "validation"]:
        ds = load_dataset(split)
        res = evidence_mapping_audit(ds, split)
        mapping_results[split] = res
        print(f"{split} strict: {res['strict']}")
        print(f"{split} fuzzy:  {res['fuzzy_mode']}")

    # 4. Chunking comparison on validation (sample)
    print("\n=== CHUNKING COMPARISON (validation, 50-paper sample) ===")
    ds_val = load_dataset("validation")
    chunking = chunking_comparison(ds_val, "validation")
    for name, cfg in chunking["configs"].items():
        print(
            f"{name}: chunks={cfg['total_chunks']}, "
            f"mean_tokens={cfg['mean_chunk_tokens']:.1f}, "
            f"evidence_coverage={cfg['coverage_rate']:.3f}"
        )

    # 5. Split isolation
    print("\n=== SPLIT ISOLATION ===")
    iso = check_split_isolation()
    print(f"Isolated: {iso['is_isolated']}, overlaps: {iso['overlaps']}")

    # Save all results
    output = {
        "split_counts": counts,
        "dataset_stats": {k: dict(v.items()) for k, v in all_stats.items()},
        "evidence_mapping": mapping_results,
        "chunking_comparison": chunking,
        "split_isolation": iso,
    }

    with open(RESULTS_DIR / "phase_1_1_audit.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to {RESULTS_DIR / 'phase_1_1_audit.json'}")


if __name__ == "__main__":
    main()
