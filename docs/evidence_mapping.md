# Evidence Mapping Documentation

## Overview

Evidence mapping connects human-annotated supporting evidence in the QASPER dataset to retrievable passages (paragraphs or chunks). This is critical for evaluating retrieval systems.

## Mapping Strategy

### Exact Match

Direct string comparison between evidence annotation and paragraph text.

**When it works**: When annotators copied text exactly from the paper.

**Limitations**: Fails when annotators made minor edits or used ellipses.

### Normalized Match

After text normalization (lowercase, whitespace normalization, punctuation removal).

**Normalization operations**:
1. Convert to lowercase
2. Normalize whitespace (collapse multiple spaces to single space)
3. Strip leading/trailing whitespace
4. Remove non-alphanumeric characters except punctuation (.,;:!?-)

**When it works**: Handles case differences and whitespace variations.

**Limitations**: May merge distinct passages that normalize to the same string.

### Fuzzy Match

Containment-based matching (evidence text in paragraph, or vice versa).

**When it works**: When evidence is a substring of a paragraph or vice versa.

**Limitations**: May produce false positives for common phrases.

## Implementation

### Passage Identifiers

Each passage has a stable identifier:

```
{paper_id}::{section_name}::{paragraph_index}
```

For chunks:
```
{paper_id}::{section_name}::{paragraph_index}::chunk{chunk_index}
```

### Mapping Result

The mapping result includes:
- Total evidence count
- Successful mappings (by match type)
- Failed mappings with reasons
- All successful mappings with provenance
- All failures for inspection

### Failure Reasons

Common failure reasons:
1. **Paragraph not found**: Section or paragraph index invalid
2. **No matching strategy succeeded**: Text mismatch after all strategies
3. **Ambiguous mapping**: Multiple paragraphs match equally well

## Usage

```python
from scientific_rag.data.dataset import load_dataset
from scientific_rag.retrieval.evidence import map_evidence_to_passages

dataset = load_dataset("validation")
paper = dataset.papers[0]

result = map_evidence_to_passages(
    paper,
    use_normalization=True,
    use_fuzzy=False,
)

print(f"Success rate: {result.successful_mappings / result.total_evidence_count:.2%}")
```

## Inspection

Inspect failed mappings:

```python
from scientific_rag.retrieval.evidence import inspect_failed_mappings

failures = inspect_failed_mappings(result, paper_id="paper1")
for failure in failures:
    print(f"Failed: {failure.evidence_text}")
    print(f"Reason: {failure.reason}")
```

## Handling Non-Textual Evidence

QASPER includes evidence that points to figures and tables. In text-only experiments:

1. Figure/table captions are treated as text passages
2. Evidence pointing to figures/tables is recorded but not used for retrieval
3. Failures for figure/table evidence are documented

## Multi-Evidence Questions

Some questions require evidence from multiple paragraphs. The mapping:
- Preserves all evidence passages for each question
- Does not merge alternative valid evidence sets
- Records the number of evidence passages per question

## Provenance

All mappings preserve:
- Source paper ID
- Section name
- Paragraph index
- Original evidence text
- Match type used
- Passage text

## Statistics

The mapping produces statistics on:
- Overall success rate
- Breakdown by match type (exact, normalized, fuzzy)
- Failure rate and reasons
- Questions requiring multiple evidence passages

## Known Limitations

1. Paragraph-level mapping (not sentence-level)
2. May not handle ellipses in evidence annotations
3. Fuzzy matching is conservative (containment only)
4. Does not resolve ambiguous mappings automatically
5. Non-textual evidence (figures/tables) is not fully supported


## Phase 1.1 corrections

- Mapping now returns **all** candidate passages per evidence string;
  ambiguity (>1 candidate) is recorded, never silently resolved.
- Each mapping preserves `annotation_id`, `worker_id`, `question_id`.
- Non-textual evidence is flagged: `FLOAT SELECTED` markers and short
  figure/table pointers. `BIBREFn`/`TABREFn` citation tokens inside paragraph
  excerpts are NOT treated as non-textual.
- Strict mode = exact + normalized containment only. Fuzzy containment is a
  diagnostic mode (`use_fuzzy=True`) and is excluded from gold labels.
- Alternative evidence sets are preserved per annotation (never merged).
  `evidence_set_complete` / `question_has_complete_evidence_set` implement the
  completeness predicate; empty or unmapped sets are never complete.
- Measured on real data (results/evidence_mapping_audit.json): ~83% of
  evidence strings map uniquely, ~0.4% ambiguous, ~16% unmatched,
  ~27% contain figure/table/FLOAT-SELECTED markers (before tightening).
