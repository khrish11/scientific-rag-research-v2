# QASPER Dataset Documentation

## Source Information

**Official Homepage**: https://allenai.org/data/qasper

**Primary Paper**:
- Title: "A Dataset of Information-Seeking Questions and Answers Anchored in Research Papers"
- Authors: Pradeep Dasigi, Kyle Lo, Iz Beltagy, Arman Cohan, Noah A. Smith, Matt Gardner
- Venue: NAACL 2021
- ACL Anthology ID: 2021.naacl-main.365
- arXiv: 2105.03011
- DOI: 10.18653/v1/2021.naacl-main.365

**Hugging Face Dataset**: https://huggingface.co/datasets/allenai/qasper

## License

**License**: CC BY 4.0 (Creative Commons Attribution 4.0 International)

**Redistribution Rights**: The CC BY 4.0 license permits redistribution with attribution. This repository will implement download scripts rather than redistributing the dataset files to maintain clear provenance.

## Dataset Version

**Current Version**: v0.3.0

**Version History**:
- v0.3.0: Current stable version with test split included
- Earlier versions: Training and development splits only

## Download URLs

**Training and Development**:
- URL: https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz
- Files: `qasper-train-v0.3.json`, `qasper-dev-v0.3.json`

**Test**:
- URL: https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-test-and-evaluator-v0.3.tgz
- Files: `qasper-test-v0.3.json`, evaluator script

## Dataset Splits

| Split | Papers | Questions |
|-------|--------|-----------|
| Train | 888 | ~2,800 |
| Validation | 281 | ~900 |
| Test | 416 | ~1,300 |
| **Total** | **1,585** | **5,049** |

*Note: Exact question counts per split will be computed from the downloaded data.*

## Dataset Schema

The dataset contains the following fields for each paper:

- `id`: Unique paper identifier
- `title`: Paper title
- `abstract`: Paper abstract
- `full_text`: Structured full text with sections
  - `section_name`: List of section names
  - `paragraphs`: List of paragraphs per section
- `qas`: Question-answer pairs
  - `question_id`: Unique question identifier
  - `question_text`: The question
  - `question_type`: Type of answer (e.g., "yes/no", "span", "free-form", "multiple-choice")
  - `answers`: List of annotated answers
    - `answer`: The answer text
    - `annotator`: Annotator identifier
    - `unanswerable`: Boolean flag if question is unanswerable
  - `evidence`: List of evidence annotations
    - `annotator`: Annotator identifier
    - `answer`: The answer this evidence supports
    - `evidence_list`: List of evidence passages
      - `section`: Section name
      - `paragraph`: Paragraph index within section
      - `text`: Exact text of evidence passage
- `figures_and_tables`: Non-textual content
  - `caption`: Caption text
  - `type`: "figure" or "table"

## Answer Types

QASPER includes the following answer types:
- **yes/no**: Binary yes/no questions
- **span**: Extractive span from text
- **free-form**: Free-text answers
- **multiple-choice**: Multiple choice questions

## Evidence Annotations

- Evidence is annotated at the paragraph level
- Multiple annotators may provide evidence for the same question
- Questions may require evidence from multiple paragraphs
- Some questions are marked as unanswerable from the paper

## Non-Textual Content

The dataset includes figures and tables with captions. In the initial text-only experiments:
- Figure and table captions will be treated as text passages
- Figure/table content (images, tabular data) will not be processed
- Evidence pointing to figures/tables will be recorded but not used for retrieval

## Research Constraints

- The dataset is limited to NLP papers (domain-specific)
- Questions are written by readers who only saw title and abstract
- Answers are provided by separate NLP practitioners
- Evidence annotations are paragraph-level, not sentence-level

## Citation

```bibtex
@inproceedings{dasigi-etal-2021-dataset,
    title = "A Dataset of Information-Seeking Questions and Answers Anchored in Research Papers",
    author = "Dasigi, Pradeep  and
      Lo, Kyle  and
      Beltagy, Iz  and
      Cohan, Arman  and
      Smith, Noah A.  and
      Gardner, Matt",
    booktitle = "Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jun,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.naacl-main.365/",
    doi = "10.18653/v1/2021.naacl-main.365",
    pages = "4599--4610",
}
```


## Phase 1.1 verified schema notes

- `full_text` is a **list** of `{section_name, paragraphs}` objects
  (section_name may be null).
- `qas[].question` (not `question_text`); `question_id` is a SHA-1 hex string.
- `answers[]` items have `answer` (object), `annotation_id`, `worker_id`.
- `answer` object: `unanswerable` (bool), `extractive_spans` (list),
  `yes_no` (JSON bool or null), `free_form_answer` (str), `evidence` and
  `highlighted_evidence` (lists of paragraph-excerpt strings, may contain
  FLOAT SELECTED markers and BIBREFn/TABREFn citation tokens).
- Verified counts: train 888/2593, validation 281/1005, test 416/1451
  (papers/questions). Question-level vs annotation-level statistics are
  distinguished in docs/phase_1_1_validation_report.md.
