# Phase 2 Reconciliation

Cause of aggregate-vs-bootstrap mismatch: DIFFERENT ESTIMANDS.
- Aggregate table = question-micro mean over 812 eligible questions.
- Bootstrap = paper-macro mean over 275 papers (per-paper question means).

Recomputed @1024 complete recovery:
| method | micro | macro |
|---|---|---|
| bm25 | .5776 | .5678 |
| dense | .6441 | .6388 |
| hybrid | .6626 | .6527 |
| rrf | .6613 | .6462 |

Pairwise diffs micro vs macro:
rrf-dense +.0172/+.0073; hybrid-dense +.0185/+.0139;
rrf-hybrid -.0012/-.0065; bm25-dense -.0665/-.0710.
Bootstrap point estimates match the MACRO estimand; the aggregate table
reported MICRO. Both are valid; they must not be mixed. Win/loss counts
(27/12, 66/52) are question-level and consistent with small micro diffs.
