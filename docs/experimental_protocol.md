# Experimental Protocol (Phase 2.1)

- Task: single-paper scientific QA (paper known in advance). Not open-domain.
- Representation: paragraph-level passages (default per Phase 1.1).
- Tuning: deterministic paper-disjoint dev partition of TRAIN split
  (60 papers, seed 42; ids in results/dev_partition.json).
- Selection: primary metric = complete annotated-evidence recovery @1024
  tokens on dev; ties -> larger alpha / k closest to 60. Frozen before
  validation eval (results/tuning_manifest.json).
- Validation split: exploratory baseline comparison (already informed
  preprocessing; NOT an untouched final test).
- Test split: sealed; unused for any decision.
- Budgets: 256/512/1024/2048/4096 tokens. Top-k: 1/3/5/10/20.
- Bootstrap: paired paper-cluster, 2000 replicates, seed 42, @1024 primary.
