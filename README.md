# Scientific RAG Research Infrastructure

Research infrastructure for: "Evidence-Sufficiency Stopping for Budgeted Hybrid Retrieval in Scientific Paper Question Answering."

## Overview

This repository provides a reproducible experimental environment for research on evidence-sufficiency estimation in budgeted hybrid retrieval systems for scientific literature question answering.

## Installation

```bash
# Clone the repository
git clone https://github.com/khrish11/scientific-rag-research.git
cd scientific-rag-research

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"
```

## Project Structure

```
scientific-rag-research/
├── configs/              # Configuration files
├── data/
│   ├── raw/             # Raw dataset files (git-ignored)
│   └── processed/       # Processed dataset files (git-ignored)
├── docs/                # Documentation
├── experiments/         # Experiment scripts and results
├── results/             # Experiment outputs (git-ignored)
├── scripts/             # Utility scripts
├── src/
│   └── scientific_rag/  # Source code
│       ├── analysis/    # Dataset analysis and statistics
│       ├── data/        # Dataset loading and schemas
│       ├── integrity/   # Research integrity safeguards
│       └── retrieval/   # Retrieval and evidence mapping
└── tests/               # Test suite
```

## Quick Start

### Download Dataset

```bash
python -c "from scientific_rag.data.dataset import download_dataset; download_dataset()"
```

### Compute Statistics

```bash
python -c "
from scientific_rag.data.dataset import load_dataset
from scientific_rag.analysis.stats import compute_statistics, export_statistics_json, export_statistics_csv
from pathlib import Path

dataset = load_dataset('validation')
stats = compute_statistics(dataset)
export_statistics_json(stats, Path('results/validation_stats.json'))
export_statistics_csv(stats, Path('results/validation_stats.csv'))
"
```

### Run Tests

```bash
pytest tests/ -v
```

### Linting

```bash
ruff check src/ tests/
ruff format src/ tests/
```

### Type Checking

```bash
mypy src/
```

## Dataset

This project uses the QASPER dataset:

- **Paper**: "A Dataset of Information-Seeking Questions and Answers Anchored in Research Papers" (NAACL 2021)
- **License**: CC BY 4.0
- **Version**: v0.3.0
- **Homepage**: https://allenai.org/data/qasper
- **Hugging Face**: https://huggingface.co/datasets/allenai/qasper

See [docs/dataset.md](docs/dataset.md) for detailed documentation.

## Development

### Running CI Locally

The CI pipeline runs:
- Ruff linting and formatting
- MyPy type checking
- Pytest with coverage

## Research Integrity

This project includes safeguards to prevent:
- Paper overlap across splits
- Gold label leakage into retrieval inputs
- Test data contamination in training
- Answer exposure in retrieval context

See [docs/experimental_protocol.md](docs/experimental_protocol.md) for details.

## Citation

If you use this infrastructure, please cite the QASPER dataset:

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

## License

MIT License

## Acknowledgments

This research uses the QASPER dataset from the Allen Institute for AI (AI2).
