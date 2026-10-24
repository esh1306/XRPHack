# Decode & Detect: The XRPL AI Hack — Team Solo Validator

Team: **Solo Validator** — captain and sole member: Eshita Agarwal.

Scores every test transaction (0–1) for money-laundering suspicion on the synthetic XRPL ledger.

## Final submission (default `./run.sh`): v9e_w90, leaderboard 71.3
```sh
./run.sh data/test outputs/submission.csv                 # final pick v9e_w90 (BLEND_OURS=0.1)
BLEND_OURS=0.4 ./run.sh data/test outputs/v9c_w60.csv     # other sweep points: 0.2 (w80), 0.3 (w70), 0.4 (w60), 0.7 (v9b)
PATTERNS=0.2 ./run.sh data/test outputs/v11.csv           # + our research pattern layer (src/patterns_research.py)
FINAL=0 V7X=1 NOHEAD=1 ./run.sh ...                       # older variants (see sections below)
```
Steps: (1) our v7x no-head model (features `src/fe*.py`, `src/final.py`, self-training `src/pseudo.py`); (2) the stage-1 model of
Sahid's earlier prototype (`prior_work/prototype_agent`, MIT, see `PRIOR_WORK.md`; driver scripts in `prototype/`); (3) rank blend
BLEND_OURS·rank(ours) + (1−BLEND_OURS)·rank(prototype) and the prototype's closed-cluster score propagation (`src/v10_post.py`);
(4) optional pattern layer, top 5% kept fixed (`src/final_v11.py`); (5) `src/make_submission.py`.
Seeds fixed (42). CPU only, 8 cores / 16 GB RAM; about 15–25 min from scratch. The prototype step reads `data/train` and `data/test`.
**External services: none** in this pipeline (no LLM/API calls, no keys needed). `src/jev_*`/`src/tsfm_*` experiments are not shipped.
Team: **Solo Validator**, captain and sole member Eshita Agarwal.

## Exact command
```sh
# unzip the organisers' training package into data/ (data/train, data/validation, ...)
# unzip the test package into data/test/
./run.sh data/test outputs/submission.csv
```
Python 3.13; pinned deps in `requirements.txt`. Seed fixed at 42. CPU only (8 cores, 16 GB RAM); full run ≈ 5 min on an idle 8-core CPU. Models run as parallel single-thread processes (`NJOBS=1`). Logs are written to `logs/`.

## Method (v5 (default) = shift-robust v3b; v4 = v3b + level-3 stacker (STACK=1))
All features are computed per split, transductively, from that split's own 7-day transaction graph. No labels are used on test.
- `src/fe.py` (147 features). USD valuation via `market_rates.csv` (hour of payment); the first transfer leg is the amount paid; cross-currency, legs and paths; round amounts.
  Also sender/receiver aggregates, pair history, temporal-motif features (forward/receive of similar amounts in 5m/1h/6h/24h, bursts, fan-in/out)
  and graph features (PageRank, HITS, k-core, clustering, reciprocity, parallel paths, time-respecting 2-hop cycles/chains).
  Week-position features (t_rel, dow) are dropped because they are split-specific.
- `src/fe2.py` (39). Flow features: 3d/7d windows, running balances, same-asset in/out conservation, counterparty novelty, asset hops, amount z-scores.
- `src/fe3.py` (114). Reconstruction of candidate laundering episodes. Session features per (receiver, asset) inflow and (sender, asset) outflow at 5m/30m/2h gaps.
  It also builds temporal components (transactions sharing an account within 2m/15m/1h) and flow chains (inflow → next outflow within 8m/1h/4h), with size, duration, accounts, assets, value and conservation.
- `src/fe4.py` (v3, 89 features): in->out USD amount matching after fees/FX, greedy multi-hop chain following with amount decay,
  peel chains, out-of-character deviation from the account's own amount/hour/gap baseline (leave-one-out),
  counterparty-of-counterparty novelty, cohort synchrony (birth cohorts, synchronised amounts, 10-minute counterparty bursts),
  and episode-level aggregates over reconstructed episodes, broadcast back to each transaction.
- Robustness: adversarial validation (train vs practice split, and train vs test features; no labels used) showed that
  split-level graph statistics shift. Constant-per-split and giant-component (1-hour component) columns are dropped, and
  core, clustering and cross-currency rates are percentile-ranked within their own split. In v4, a train-vs-test KS check
  moved six more shifting columns into that list (cmp900_cross, acc_r_mean_r_out_after_total, s_out_med,
  acc_s_mean_s_conserve_2d, acc_s_mean_s_net_before, spread). No feature is derived from account-address strings.
- `src/fe5.py` / `src/fe6.py` (tried; not used in the model because grouped CV did not improve). fe5 is inspired by ExSTraQt
  (Tariq & Hassani 2026, arXiv:2604.02899): time-respecting collect/dispense flow and Louvain communities. fe6 is inspired by
  BlazingAML (arXiv:2604.12241): scatter-gather, gather-scatter and 3-cycle counts. It also takes ideas from TCEE
  (arXiv:2610.29667): w=8 neighbourhood windows and pair pooling. fe6 only feeds the explanatory `reason` column. These are
  our own implementations; no code was copied.
- `src/final.py`:
  - Stage 1 is a rank blend of LightGBM (3 seeds), XGBoost and CatBoost, using 5-fold StratifiedGroupKFold grouped by scenario with early stopping. Each model is refit on all of train.
  - Stage 2 is graph stacking: out-of-fold stage-1 scores are propagated to account, pair, temporal-neighbour, 2-hop and component level (leave-one-out mean/max). Test uses stage-1 test predictions. The stage-2 models are retrained on these.
  - The final score is a rank blend of stage 1 and stage 2, with the weight chosen on OOF AP.
- `src/v4.py` (v4) is a shallow LightGBM stacker trained with grouped CV on top of the v3 out-of-fold score. Its inputs are:
  - a medium/hard specialist LightGBM that uses f2, f3, f4 and f6. It is trained with easy positives removed and look-alike legitimate scenarios up-weighted 3x, over 2 seeds.
  - component/episode-smoothed versions of the v3 score: mean, max, size and within-component rank over the 2m/15m components and the 8m/1h/4h flow chains.

  Inputs that shift between train and test (KS 0.06–0.22) are percentile-ranked within their split. The final score is 0.5·v3 rank + 0.5·stacker rank. Extra ensemble members (`cache/members/<name>_oof.parquet`) can optionally be added with `MEMBERS=1`. A member is kept only if it raises two-fold-seed grouped CV AP by at least 0.1.
  We also tried account behaviour-change features (`src/fe7.py`: dormant-then-burst, history rank, dwell, in/out balance). They lowered grouped CV at both fold seeds, so they are not used.
- `src/make_submission.py` validates that the IDs match the test set exactly and writes rank-normalised scores in [0,1] with no ties, plus a short `reason` (the dominant pattern for the top 10%).

## Not written by us
- Libraries: pandas, numpy, pyarrow, scikit-learn, SciPy, LightGBM, XGBoost, CatBoost, NetworkX (see requirements.txt).
- Organiser files: data and `payments.py` reference logic (data not redistributed; `data/` excluded from the ZIP).
- Pre-trained models: none. External services/data: none.
- AI assistance: code built with an AI coding assistant (Grok Bot).

## Licence
MIT, see `LICENSE`.

## v5 note
`./run.sh` now ships the v3b blend directly (submission_v5). The level-3 stacker (`STACK=1 ./run.sh ...`, submission_v4)
scored below its own base in grouped CV (61.60/61.16 vs 61.82 AP), and an ablation showed the medium/hard specialist adds
nothing on the v3b base (comp-only 61.60/61.24 vs comp+spec 61.60/61.16), so it was dropped from the default path.

## v6 (default `./run.sh`): owner's pattern spec as feature groups + look-alike suppression
`src/fe8.py` (per split, no labels, no account IDs). End-to-end flow uses the sender->Destination graph; the transfer
graph (dominated by ~9 hub intermediaries) is used only for hop structure. USD uses hourly rates keyed by issuer:currency.
- **A row**: delivered/sent USD, hour sin/cos, log amount vs sender/recipient median, gap to sender's prev/next tx,
  sequence gap, same-second flag. Conversion/leg/path columns shift train->test (KS ~0.09) and are dropped.
- **B pair**: pair count, amount CV, span, inter-arrival median/CV, daily-cadence and recurring look-alike flags
  (5-7 repeats, daily, one currency, CV<10%), reverse pair, first-time pair, position in pair history.
- **C layering chains**: next/prev link counts at windows 60/300/900/3600 s x ratio bands [0.88,1.04], [0.90,1.00],
  [0.95,1.00]; forward/backward chain length, position, hop gap, value retained, returns-to-origin, ambiguous matches.
- **D gather bursts**: distinct senders into the recipient (trailing/centred 1 h and 6 h, max over 1 h windows), USD pooled,
  share of the recipient's inbound, forwarding ratio and delay after the window.
- **G**: IsolationForest anomaly rank fitted within each split.
- **Look-alike head** (`src/lookalike.py`): LightGBM trained only on positives vs legit look-alike-scenario rows; its rank
  is added with weight 0.2 (`src/v6.py`). v3 OOF error analysis: ~97% of top-ranked false positives are look-alikes.

## v7 — transductive self-training on the test week (`PSEUDO=1 bash run.sh`)
CV gains stopped transferring to the leaderboard, which points to a train→test week shift. v7 runs the v6 pipeline, then:
1. **Pseudo-labels from the v6 test scores.** The top 0.8% of test rows become pseudo-positives and the bottom 60% pseudo-negatives. A pseudo-negative is dropped if it shares a `cmp900` candidate cluster with a pseudo-positive, so a cluster's labelled rows never disagree.
2. **Retraining.** Stage 1 (LightGBM×3, XGBoost, CatBoost) and stage 2 (propagated scores; LightGBM×3, XGBoost) are retrained on train plus the pseudo rows, which get weight 0.5. Early stopping uses only real train labels.
3. **No self-scoring.** The pseudo set is split into 5 folds grouped by cluster. Each pseudo row is scored by the model that did not see its own label. Unlabelled test rows get the mean of the 5 models.
4. **Sanity CV.** Train OOF uses the same grouped folds, with all pseudo rows added to every training fold.
5. **Final blend.** The final score uses the same stage-1/stage-2 weighting and the same 0.2·rank look-alike head as v6.

`POS_FRAC=0.015` gives the v7b variant. Code: `src/pseudo.py`. One round only, fixed seeds.

## v6x probe (`V6X=1 bash run.sh`)
- Runs the v6 pipeline with FEATS=f2,f3,f4,f6 (BlazingAML scatter-gather, gather-scatter and 3-cycle counts from src/fe6.py) and FE10=1. FE10=1 merges cache/fe10/<split>_fe10.parquet (21 cols; build them with `python src/fe10_medium.py build`, which reads data/train and data/test).
- The v6 look-alike head (cache/lookalike.npz, built without the new features) is then blended at 0.2 via src/v6.py.
- The report is src/v6x_report.py.

## v7x (`V7X=1 bash run.sh`; `NOHEAD=1` for the no-head variant)
- v7 self-training (src/pseudo.py, same settings) on the v6x features, with pseudo-labels from v6x test scores.
- src/pseudo.py now also writes <out>_base.parquet, the no-head score.
- src/v7x_report.py prints the CV and Spearman comparisons.

## v7x2 (`V7X2=1 bash run.sh`)
- A second self-training round on the v6x features, with the v7x no-head scores as the teacher. Outputs the no-head score.
- Use `POS_FRAC=0.012 V7X2_TAG=_pf012` for the wider-positive variant.

## v9 / v9b: closed-cluster score propagation (Sahid's earlier prototype, see PRIOR_WORK.md)
- `V9=1 ./run.sh` takes the v7x no-head scores (`cache/pred_test_v7x_base.parquet`, rebuilt with V7X=1 NOHEAD=1 if missing). It then applies the prototype's multi-scale closed-cluster score propagation: each payment's final score is 0.35 × its own rank + 0.65 × the weighted mean/max/q75 of base scores over its closed first-time-payment cluster in 5m–4h windows.
- `V9B=1 ./run.sh` does the same, but first blends 0.7 × rank(v7x no-head) with 0.3 × rank(prototype stage-1: 5-seed HistGradientBoosting + ExtraTrees + LogReg on flow/motif/account-aggregate/closed-cluster domain features).
- Our CV (StratifiedGroupKFold(5) by scenario_id, seed 42), AP all/easy/medium/hard:
  - v7x no-head: 63.79 / 84.58 / 9.73 / 0.74
  - v9: 64.89 / 85.54 / 12.65 / 0.76
  - v9b: 65.64 / 85.41 / 13.99 / 0.76
  - prototype pipeline alone: 61.72 / 79.03 / 11.68 / 0.71
- Propagation on older OOFs: v6 fold-seed 7 60.37→61.42; v6 fold-seed 2026 60.33→61.12; v6x 62.91→63.79.

## v10 / v13 Final (`V13=1 ./run.sh`, Leaderboard **71.6**): Super-Ensemble + Hard-Relay & Medium-Cluster Detectors (`src/fe11_hard_med.py`, `src/v13_final.py`)
- See **`LEARNING_GUIDE_TO_80_AP.md`** for a complete deep-dive into the domain mechanics, dataset anatomy, model architecture, ablation lessons, and the roadmap to 80+ AP.
- Adds `src/fe11_hard_med.py` (145 features):
  - **Hard relay detectors**: 2-hop `distributed_relay` bridge identification (`pair_n` in {3,4} with >=20h episode spacing), `[1.5h, 3.3h]` pre-funding and post-payout episode lag windows, 1-hop `aggregate_relay` `[3.6h, 5.8h]` hold-time lag bands, forward vs reverse refill asymmetry (`strict_fwd`), and per-episode USD conservation (`min_r`).
  - **Medium closed-cluster detectors**: 2-hop closed novel-edge clusters extended across `15m, 45m, 90m, 3h, 5h, 8h`, pure-source to pure-sink end-to-end USD conservation (`mc_e2e_ratio`, `mc_e2e_err`), stack/bipartite/cycle topology indicators, and inter-step timing regularity (`mc_dt_cv`).
- Also wires in `prior_work/prototype_agent/app/{cluster,relay,funding,fingerprint}.py` (74 features), with within-split percentile rank normalization for any column with train-vs-test KS > 0.06.
- `src/v13_final.py` (**71.6 LB**) blends `v7x no-head` (0.08) + `Prototype Stage-1` (0.82) + `3-seed Xs_983 LightGBM specialist` (0.10), applies multi-scale closed-cluster score propagation (`propagate_cluster_scores`), locks the top 4% (`Easy + top Medium`), and re-ranks below the top 4% with the 3-seed `Xs_983` LightGBM specialist (0.15) + verified distributed-relay episode propagation (0.10).
