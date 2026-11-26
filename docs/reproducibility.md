# Reproducibility Record

## Environment
- Python 3.13.5, `.venv\Scripts\python.exe`
- torch 2.10.0+cu126, CUDA 12.6, RTX 4060 Laptop (8GB), driver 566.07
- sentence-transformers 6.3.0, model sentence-transformers/all-MiniLM-L6-v2
  (Apache-2.0), FP32, batch_size=32 (GPU) / 64 (CPU)
- tiktoken cl100k_base; sklearn 1.9.1 (LogisticRegression)
- numpy 2.5.5; transformers 5.24.1

## Data
- QASPER v0.3 (CC BY 4.0), official S3 tarballs
- train 888 papers/2593 q; dev 281/1005; test 416/1451 (SEALED - unused)
- Eligibility: >=1 annotation, fully-mapped nonempty textual evidence
  (non-textual = FLOAT SELECTED/figure-table pointers excluded from req)
- Validation eligible: 812 questions / 275 papers; excluded 193

## Canonical protocol (Phase 5.1)
- Paragraph-level passages; paper known in advance (single-paper QA)
- Packing: greedy rank-order, token_count + 2*(n-1) ["\n\n" sep],
  skip-if-not-fit, never exceed budget, no truncation
- Budgets 256/512/1024/2048/4096; top-k 1/3/5/10/20
- Aggregation: question-micro; paper-cluster bootstrap 2000 reps seed 42
- Dev partition: 60 train papers, seed 42 (results/dev_partition.json)
- Frozen config: hybrid alpha=0.9, RRF k=20 (results/tuning_manifest.json)

## Reproduce
```
./.venv/Scripts/python.exe -m scientific_rag.experiments.phase5_canonical
./.venv/Scripts/python.exe -m scientific_rag.experiments.phase4_feasibility
./.venv/Scripts/python.exe -m scientific_rag.experiments.phase4_1
./.venv/Scripts/python.exe scripts/gpu_benchmark.py
pytest tests/ -q && ruff check src/ tests/ scripts/ && mypy src/
```

## Artifact provenance
| result | file | pipeline |
|---|---|---|
| phase2 baselines | results/phase2_full_results.json | CPU dense, phase2_full.py |
| phase3 trajectories | results/phase3_*.json | GPU dense, approx packing |
| phase4 models | results/phase4_feasibility.json | train-dev trajectories |
| phase4.1 stopping | results/phase4_1_results.json | Model A, GPU |
| phase5 canonical | results/phase5_canonical.json | unified packing, cpu+gpu dense |

Historical artifacts are NOT overwritten; canonical run is additive.
