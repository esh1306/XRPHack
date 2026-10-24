# Prior work reused

## Decode & Detect AML investigator agent: the author's own earlier prototype
- Components reused from the author's own earlier prototype, *Decode & Detect AML investigator agent* (MIT, Copyright (c) 2026 Sahid). It is a separate codebase, written before this submission.
- Its original LICENSE file is kept unchanged at `prior_work/prototype_agent/LICENSE`.
- Files reused, copied unchanged into `prior_work/prototype_agent/app/`:
  - `data.py` loads one edge per payment.
  - `cluster_prop.py` provides `propagate_cluster_scores`, the multi-scale closed-cluster score propagation.
  - `features.py`, `motif.py`, `account_agg.py`, `model.py` and `train_e.py` provide the prototype's stage-1 ensemble and closed-cluster features.
- Jev/OpenAI/server parts not used. The prototype's Jev (TypeSafe API) client, OpenAI agent, FastAPI server and UI are excluded, so this code makes no network calls and needs no API keys.
- How it is used: `src/v9_prop.py` applies `propagate_cluster_scores` to the v7x no-head test scores (v9). With V9B=1, it first rank-blends the v7x no-head scores (0.7) with the prototype's stage-1 ensemble (0.3).
- These components come from that prototype. They were not developed during the hackathon day and were not taken from a paper.

## How it is used in the final submission (v9c / v9e)
- `prototype/build_design.py` and `prototype/run_prototype.py` build the prototype's design matrix and stage-1 ensemble scores from `data/`; `src/v10_post.py` blends them with our v7x no-head scores (`BLEND_OURS`, default 0.1) and applies `propagate_cluster_scores`.
- Our own additions on top: the whole v7x model (features, self-training) and the research pattern layer `src/patterns_research.py` (AMLworld, Egressy et al., Elliptic, FATF), which is not from the prototype.
