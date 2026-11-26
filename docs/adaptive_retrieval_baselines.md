# Adaptive Retrieval Baselines (Phase 3)

Policies evaluated OFFLINE from rank-order acquisition trajectories
(no re-retrieval; identical ranking, packing, serialization):
- fixed_k {1,3,5,10,20}, fixed budget {256..4096}
- score_threshold {0.3..0.7}, margin {0.05..0.3}, rel_drop {0.3,0.5,0.7}
- "Adaptive-k" faithful reproduction INFEASIBLE in scope -> score/margin/
  rel-drop stopping implemented as documented ADAPTATIONS, not the original
  algorithm. Original-paper verification pending (not fabricated).

Methodological note: stopping on an unchanged ranking cannot increase
complete-evidence recovery vs continuing; benefit = token savings at
equal recovery. Results: results/phase3_adaptive.json (812 eligible
validation questions, GPU dense bs=32).
