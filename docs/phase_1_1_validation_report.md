# Phase 1.1 Validation Report — Dataset Correctness Audit

Date: 2026-11-26. Branch: `feature/phase-1-dataset-preparation`.

## 1. Dataset split verification (raw JSON, locally computed)

| Split | File | Papers | Questions |
|---|---|---|---|
| train | qasper-train-v0.3.json | 888 | 2,593 |
| validation | qasper-dev-v0.3.json | 281 | 1,005 |
| test | qasper-test-v0.3.json | 416 | 1,451 |
| **total** | | **1,585** | **5,049** |

All counts match the expected official counts exactly.

Sources: `https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz`
(train+dev), `.../qasper-test-and-evaluator-v0.3.tgz` (test). License CC BY 4.0.
Local copies live in `data/raw/` (git-ignored).

## 2. Question-level vs annotation-level statistics

Aggregation rules (explicit):

- **Question-level answer type** = `answer_type` of the *first* answer annotation
  of each question (deterministic, documented; used for the headline distribution).
- **Annotation-level answer type** = `answer_type` counted once per answer
  annotation (multiple annotators → multiple counts).
- **any-unanswerable** = at least one annotation has `unanswerable=True`.
- **all-unanswerable** = every annotation has `unanswerable=True`.
- **mixed** = some but not all annotations unanswerable.

| Metric | train | validation | test |
|---|---|---|---|
| papers | 888 | 281 | 416 |
| questions | 2,593 | 1,005 | 1,451 |
| answer annotations | 2,675 | 1,764 | 3,554 |
| Q-level: span / free-form / yes-no / unanswerable | 1314/610/397/272 | 530/264/116/95 | 732/371/209/139 |
| A-level: span / free-form / yes-no / unanswerable | 1363/622/409/281 | 962/431/208/163 | 1817/878/493/366 |
| questions any-unanswerable | 278 | 135 | 243 |
| questions all-unanswerable | 271 | 60 | 79 |
| questions mixed answerability | 7 | 75 | 164 |
| questions multi-annotation | – | (see audit json) | – |
| yes/no True / False (annotation-level) | 225/184 | 105/103 | 281/212 |

### Unanswerable-count discrepancy — resolved

The earlier report showed `unanswerable=95` (answer-type distribution) vs
`num_unanswerable=135` on validation. These measure **different things**:

- `95` = questions whose **first** annotation is unanswerable (question-level
  type, first-answer rule).
- `135` = questions with **any** unanswerable annotation.
- Additionally: `60` questions are unanswerable across **all** annotations and
  `75` are **mixed** (annotators disagreed on answerability).

Both numbers are correct under their definitions; the docs now state the rule.

### yes/no correctness

`yes_no` in the raw JSON is a JSON boolean (`true`/`false`), not a string.
Phase 1.1 widened the schema to `str | bool | None` and `answer_text` maps
`True→"yes"`, `False→"no"`. Verified: validation has 105 `yes` and 103 `no`
annotation-level answers — `yes_no=False` is a valid answer, not missing data.

## 3. Evidence mapping audit (train + validation, strict mode)

Corrected implementation (`map_paper_evidence`): returns **all** candidate
passages per evidence string, flags ambiguity, preserves `annotation_id` /
`worker_id` / `question_id`, and flags non-textual evidence
(FLOAT SELECTED markers, figure/table references). Strict mode = exact +
normalized containment only; fuzzy is diagnostic-only.

Corrected-mapper results (`results/evidence_mapping_audit.json`, strict mode
= exact + normalized containment; fuzzy excluded):

| split | evidence strings | unique mapped | ambiguous | unmatched | non-textual |
|---|---|---|---|---|---|
| train | 7,788 | 6,506 (83.5%) | 34 (0.4%) | 1,248 (16.0%) | 821 (10.5%) |
| validation | 5,138 | 4,262 (83.0%) | 20 (0.4%) | 856 (16.7%) | 546 (10.6%) |

Match-type breakdown: train 4,064 exact / 2,476 normalized; validation
2,646 exact / 1,636 normalized. Non-textual = FLOAT SELECTED markers or
short figure/table pointers only (citation tokens inside excerpts excluded).

Known evidence-string properties measured on validation raw JSON:
- 5,138 evidence strings total in validation (evidence + highlighted_evidence).
- 506 contain `FLOAT SELECTED` markers; tightened non-textual flagging yields
  546/5,138 (10.6%) non-textual in validation, 821/7,788 (10.5%) in train.
- 0 empty evidence strings.
- Evidence strings are paragraph-length excerpts; `BIBREFn`/`TABREFn` tokens
  are citation markers inside running text, not non-textual evidence by
  themselves — only figure/table/FLOAT-SELECTED markers are flagged.

## 4. Fuzzy matching policy

- Strict mapping mode: exact + normalized only. Ambiguous matches (multiple
  candidate paragraphs) are recorded as ambiguous, never auto-resolved.
- Fuzzy containment is retained as `use_fuzzy=True` diagnostic mode only and
  is excluded from strict gold-evidence labels.
- Manual-review samples (exact / normalized / ambiguous / unmatched /
  non-textual) are exported in `results/evidence_mapping_audit.json`
  (`failures_sample`, `ambiguous_sample`). Automated checks are not described
  as human validation.

## 5. Alternative evidence sets

- `annotation_evidence_sets(qa)` returns one evidence set **per answer
  annotation**; alternative annotations are never merged.
- `evidence_set_complete(retrieved, ev_set)` requires every evidence string in
  the set to be covered; empty sets and unmapped strings are never complete.
- `question_has_complete_evidence_set` returns True iff retrieved text covers
  ≥1 complete annotated set (handles multi-set, partial, overlapping, dup,
  empty, unanswerable, mixed-answerability cases).
- Coverage of annotated evidence is explicitly **not** semantic sufficiency.

## 6. Chunking / provenance evaluation (validation, 50-paper sample)

| Representation | chunks | mean tokens | evidence coverage* |
|---|---|---|---|
| paragraph-level | 2,524 | 96.0 | 0.832 |
| 128-token (ov 16) | 3,345 | 76.3 | 0.592 |
| 256-token (ov 32) | 2,592 | 94.3 | 0.801 |
| 512-token (ov 64) | 2,525 | 95.9 | 0.831 |

\* fraction of evidence strings fully representable inside a single chunk
(normalized containment); partial-chunk hits tracked separately
(`evidence_partial` in `results/phase_1_1_audit.json`). Retrieving one partial
chunk of a gold paragraph does **not** count as full evidence recovery.

**Recommendation (evidence integrity, not retrieval performance):**
paragraph-level representation as the reference condition — simplest provenance
(1 chunk = 1 annotated paragraph), best evidence coverage, zero reconstruction
loss. 512-token chunking is nearly equivalent (0.831 vs 0.832) if smaller
contexts are needed later; 128-token chunking fragments 24% of evidence and is
not recommended. Final choice deferred until retrieval experiments; test split
was not used for this decision.

## 7. Test-split protection

- Paper-level isolation verified mechanically: train/validation/test paper-ID
  sets are pairwise disjoint (`split_isolation.is_isolated = true`).
- Test labels were not used for chunking selection, threshold tuning, or any
  model selection. Only schema/count verification was performed on test.
- Test-set performance evaluation remains deferred until the protocol is final.

## 8. Reproducibility manifest

- Python 3.13.5 (dev env); project requires >=3.11.
- Deps: pydantic 2.x, pydantic-settings, datasets, tiktoken, requests
  (see pyproject.toml; CI pins via pip install -e ".[dev]").
- Tokenizer: tiktoken `cl100k_base`.
- Dataset: QASPER v0.3.0, files `qasper-{train,dev,test}-v0.3.json`.
- Git commit: see branch history on `feature/phase-1-dataset-preparation`.
- Random seed: 42 (config default; preprocessing is deterministic — no RNG).
- Reports: `results/phase_1_1_audit.json`, `results/evidence_mapping_audit.json`,
  `results/validation_stats.{json,csv}`.

## 9. Corrections applied in this phase

1. `full_text` is a **list of sections** (not a dict) — parser rewritten.
2. Answer schema corrected to real QASPER shape (`answer` object with
   `unanswerable`, `extractive_spans`, `yes_no`, `free_form_answer`,
   `evidence`, `highlighted_evidence`; `annotation_id`, `worker_id`).
3. `yes_no` accepts JSON booleans; `False` is a valid answer.
4. `section_name` may be `None` → normalized to `""`.
5. Evidence mapper rewritten: ambiguity detection, annotation linkage,
   non-textual/FLOAT-SELECTED flagging, strict vs diagnostic fuzzy modes.
6. Added evidence-set completeness predicates (no merging of alternatives).
7. Question-level vs annotation-level stats separated; aggregation rules
   documented.
