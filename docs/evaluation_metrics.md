# Evaluation Metrics (Phase 2.1)

All metrics computed per question over retrieved passage-ID lists.
Gold = list of alternative annotated evidence sets (never merged).
`rel` = union of all gold passage IDs.

- Precision@k = |top-k ∩ rel| / k
- Recall@k = |top-k ∩ rel| / |rel|
- MRR = 1 / rank of first retrieved passage in rel (0 if none)
- nDCG@k = DCG(binary rel)/IDCG, IDCG = DCG of min(k,|rel|) ideal hits
- Complete annotated-evidence recovery = 1 iff ∃ gold set g ⊆ retrieved set
- Partial coverage = max_g |g ∩ retrieved| / |g|
- Budget mode: ranked list packed greedily under token budget; metrics use
  the selected set (k = number selected).
- Oracle-possible = ∃ gold set whose passages fit budget (size-ascending
  greedy packing estimate).

Eligibility: question eligible iff ≥1 annotation has ALL textual evidence
strings mapped to passages (non-textual evidence excluded from the textual
requirement; annotations with no textual evidence are ineligible).
