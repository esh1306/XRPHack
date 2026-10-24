# Submission log

| Time (UK) | Version | CV score (PR-AUC / ROC-AUC) | Leaderboard score | Notes |
|---|---|---|---|---|
| 11:22 | v1 (practice, validation/ only — NOT the test set) | OOF AP 50.37 (5-fold StratifiedGroupKFold by scenario; easy 66.96 / medium 4.03 / hard 0.67) | — not uploaded | LightGBM stage-1, 147 features (amounts, account aggregates, temporal motifs, graph). outputs/practice_validation_v1.csv |
| 11:36 | v2 (practice, validation/ only — NOT the test set) | OOF AP 61.78 (5-fold grouped by scenario; easy 83.64 / medium 4.64 / hard 0.77) | — not uploaded | fe+fe2+fe3 (300 features), stage-1 LGB×3+XGB+Cat rank blend, stage-2 graph stacking, 0.5/0.5 blend. outputs/practice_validation_v2.csv |
| 12:23 | v3 robust (TEST set) | OOF AP 60.35 (easy 80.49 / medium 5.45 / hard 0.77); v2 OOF 61.78 included split-shifted features | — not uploaded | f+f2+f3+f4, shift-robust (giant-component/split-constant dropped, graph stats percentile-ranked), stage1+stage2 50/50. outputs/submission_v3.csv (40,101 rows, with reason) |

- 2026-10-24 12:30 London: submission_v3.csv uploaded by Sahid; leaderboard AP 65.1 (reported: 92% accuracy-type stat, ~50% easy caught, 0 medium/hard). CV 60.35.

## 2026-10-24 12:31 BST — submission_v1 (v2 frozen model, real test set)
- Model: v2 (code_solo_validator_v2.zip), run in isolated /workspace/xrpl-hack-v2run; CV AP 61.78 (grouped OOF, stage-1 weight 0.5)
- File: outputs/submission_v1_v2model.csv (40,101 rows, all hashes match test, 40,101 unique scores, rank-scaled 0–1)
- sha256: 07dabc6ecf8c48bcce479e866cf6d7e8cdecc1529989ac9657fca45111f5c91d
- Log: logs/submission_v1_run.log (runtime 35m06s)
- Leaderboard score: pending
- 2026-10-24 12:38 London: submission_v1_v2model.csv (v2, CV 61.78) uploaded by Sahid; leaderboard about 1 point lower than v3 (~64.1, Sahid reported "dropped by a point and 1%"). Confirms shift-robust v3 generalises better despite lower CV.

## 2026-10-24 13:03 BST — submission_v5 (v3b alone, no stacker)
- File: outputs/submission_v5.csv (40,101 rows, unique, no NaN, 3.29 MiB, reason column), sha256 2740ca6612740d6137182b228f341fb11bcd8c7eb4a6fd9ba06e91e6e187f593
- Code: outputs/code_solo_validator_v5.zip; reproduce with `./run.sh data/test outputs/submission_v5.csv` (default STACK=0).
- Model: v3 pipeline with extended SHIFT_NORM (six extra train-vs-test-shifting columns rank-pct within split), final W=0.3.
- Grouped CV OOF AP (all/easy/medium/hard): v3b 61.82 / 82.75 / 5.29 / 0.81.
- Why not the stacker: v4 (v3b+stacker) 61.60/81.89/5.90/0.81 (fold seed 42), 61.16/81.51/5.88/0.79 (seed 7); specialist ablation
  comp-only 61.60/61.24 vs comp+spec 61.60/61.16 -> specialist does not help; stacker trades ~0.9 easy AP for ~0.6 medium AP.
- Spearman vs v3: 0.983, vs v4: 0.967. Verified run.sh STACK=1 reproduces submission_v4 byte-identically (sha ea31bac7...).

## 2026-10-24 12:56 BST — submission_v4 (shift-robust v3 base + level-3 stacker, real test set)
- Changes vs v3:
  - Six more features that shift between train and test (KS > 0.08: cmp900_cross, acc_r_mean_r_out_after_total, s_out_med, acc_s_mean_s_conserve_2d, acc_s_mean_s_net_before, spread) are now percentile-ranked within their own split.
  - Added the src/v4.py stacker: a medium/hard specialist plus component-smoothed base scores, with the shifting stacker inputs percentile-ranked. No external members are used.
  - No account-ID-derived features; addresses are only used as graph node keys.
- Grouped OOF AP (all/easy/medium/hard):
  - v3: 60.35/80.49/5.45/0.77
  - shift-robust base (v3b): 61.82/82.75/5.29/0.81, stage-1 weight 0.3
  - v4: 61.60/81.89/5.90/0.81 at fold seed 42 and 61.16/81.51/5.88/0.79 at fold seed 7
- The stacker gain is not stable. On the v3 base it added +1.35 (61.70/82.09/6.38/0.73). On the more robust v3b base it changed AP by −0.2 overall and added +0.6 on medium. Treat v4 as "v3 made shift-robust + medium lift", not as a big CV jump.
- Alternative file: outputs/submission_v4a_v3base.csv (v3 base + stacker, OOF 61.70 / 61.27; it uses the less shift-robust base).
- Members variant (MEMBERS=1 with the GNN member chosen by CV, OOF 61.64/61.67) is cache/pred_test_v4m.parquet. It is not shipped, because run.sh can't reproduce the GNN member.
- Tried and not kept:
  - plain specialist fusion (add/band/max: all ≤ v3)
  - raw component smoothing (≤ +0.3, and it shifts)
  - fe7 behaviour-change features (single-LightGBM CV −1.75 at fold seed 42 and −2.95 at fold seed 7)
  - fingerprint-scan features (noise, per that worker)
- File: outputs/submission_v4.csv (40,101 rows, IDs match test, unique, scores in [0,1], 3.29 MiB, reason column). Spearman vs v3 is 0.951.
- sha256: ea31bac7cf35d15671c6eef1b829a9e729c2e76c8c0f37764c790d0caad6a9da
- Code: outputs/code_solo_validator_v4.zip. Logs: logs/final_v3b.log, logs/v4b.log
- Leaderboard score: pending (I have not uploaded it)
- 2026-10-24 13:05 London: Sahid reports v4 (submission_v4.csv, CV 61.6) leaderboard 64.7 (below v3 65.1). v5 not uploaded. Next: v6 from Sahid spec (chains/gather/recurring-decoy/pair/IsoForest groups, 3-seed OOF ablation).
- 2026-10-24 13:17 London: submission_v3gnn.csv = 0.9 v3 + 0.1 Multi-GNN (gnn_s3) rank blend; OOF fold-ranked 61.51 vs v3 61.17, pooled 60.64 vs 60.35; code outputs/code_solo_validator_v3gnn.zip. Leaderboard: pending.

- 13:20 v3gnn (submission_v3gnn.csv): leaderboard 64.8 (below v3 65.1). Blend members dropped. Focus v6 only.

## v6 — 2026-10-24 13:24 London (submission_v6.csv)
- v3 base (SHIFT_EXT=0) + fe8 feature groups A (row/amount/time), B (pair cadence/recurring), C (layering chains, e2e sender→destination graph), D (gather bursts), G (IsolationForest rank). Conversion/path/leg-count features that shift between train and test (KS ~0.09) are dropped.
- Look-alike head: LightGBM trained only on positives vs look-alike scenario rows. Final score = rank(base) + 0.2·rank(head), rescaled to [0,1].
- Stage-1 ablation (single LightGBM, mean over fold seeds 42/7/2026): base 56.88±0.09 → A–D,G 59.35±0.64 (+2.47, which is more than 1 sd). Adding the head gives +0.44±0.04.
- Full pipeline at fold seed 42 (all/easy/medium/hard): v3 60.35/80.49/5.45/0.77; v6 61.09/81.92/5.52/0.69; v6+head 61.79/82.94/6.80/0.64.
- sha256: 83ad0b1f7932e3d973270896af0ea5d813b166a64de0337a2d2dc46aa0fc7aac. Spearman vs v3 0.964.
- Code: outputs/code_solo_validator_v6.zip. Leaderboard: pending (I have not uploaded it).

- 13:57 London: Sahid reports v6 (submission_v6.csv) leaderboard 65.1 (ties v3 65.1, "92%"). CV gain +1.4 did not carry over.

- 13:57 v6: leaderboard 65.1, tied with v3. 3-seed full-pipeline OOF (all/easy/medium/hard): v3 60.33/80.62/5.76/0.87 (sd 0.34); v6 base 60.60/81.05/6.07/0.86 (sd 0.43); v6 + head 0.2 (shipped) 60.99/81.76/6.50/0.77 (sd 0.72).
- ~14:00 v6b NOT built: nothing passed the 3-seed check (mean gain > 1 sd) on top of v6. Stage-1 LGB, mean over fold seeds 42/7/2026: v6 59.36/79.83/5.19/0.68. Gains vs v6:
  - +8 fe8 episode columns: −1.34
  - fe9 H1 local bursts 30m–6h: −1.13
  - H2 one-off pairs: −0.51
  - H3 account-week context: −1.23
  - H4 fee/afternoon/recurrence: −0.80
  - all H: −1.05
  - all H + fe8: −1.22
  - full fe8 (384 columns): −2.3
  Inside the look-alike head instead (full pipeline, 3 seeds, head weight 0.1/0.2): every variant was 60.85–61.02, against v6's 60.99. Medium stayed between 6.1 and 6.4, below v6's 6.50.
- Code for these tries: src/fe9.py, exp9.py, exp10.py, v6b_report.py. The F9_GROUPS and USE_E8 environment switches in pipeline.py are off by default, so v6 reproduces unchanged. Logs: logs/exp9_fe8ep.log, logs/exp10_v6b.log, logs/lookalike_v6b*.log.

## v7: 2026-10-24 14:15 London (submission_v7.csv), transductive self-training (`PSEUDO=1 bash run.sh`)
- Pseudo-labels from the v6 test scores: top 0.8% → 321 pseudo-positives; bottom 60% → 21,755 pseudo-negatives. 2,305 negatives that shared a cmp900 cluster with a positive were dropped. Pseudo rows get weight 0.5.
- Stage 1 and stage 2 are retrained on train plus the pseudo rows. Each pseudo row is scored by a model that never saw its label (5-fold over the pseudo set, grouped by cluster). Look-alike head weight 0.2 as in v6. One round, seed 42.
- Sanity check, train OOF at fold seed 42 with pseudo rows in every training fold (all/easy/medium/hard):
  - v7 base (W=0.5): 61.45/81.44/8.84/0.78
  - v7 + head: 61.21/81.26/8.36/0.70
  - v6 base on the same folds: 61.09/81.92/5.52/0.69
  - v6 + head on the same folds: 61.79/82.94/6.80/0.64
  - Stage-1 blend: 60.37 vs 59.61; stage-2 blend: 60.98 vs 58.26.
- sha256: 7ef8501b0bc0efbc7813420acc5b8160eb79259d3196989cc55387a684e7b95c. Spearman 0.983 vs v6. Top 400: 19 rows changed vs v6.
- Code: outputs/code_solo_validator_v7.zip. Log: logs/pseudo_v7.log. Leaderboard: pending (I have not uploaded it).

## v6x — "everything additive" probe (outputs/submission_v6x.csv)
- What changed vs v6: the shipped v6 pipeline is unchanged and nothing is removed. Added: the BlazingAML fe6 features (FEATS=f2,f3,f4,f6) and all 21 fe10 medium-motif columns (FE10=1, from cache/fe10, built by src/fe10_medium.py). The look-alike head is the v6 head (cache/lookalike.npz), blended at 0.2.
- Reproduce: `V6X=1 bash run.sh` (flags are in run.sh), using outputs/code_solo_validator_v6x.zip.
- Fold-seed-42 full-pipeline OOF (all/easy/medium/hard):
  - v6 base 61.09/81.92/5.52/0.69; v6+head 61.79/82.94/6.80/0.64.
  - v6x base 62.91/83.62/7.35/0.75; v6x+head 62.58/83.43/7.51/0.70 (this is the shipped one).
  - 1 seed only; the v6 3-seed sd is about 0.4–0.7.
- Test agreement with other submissions:
  - Spearman vs v6 0.9776 (top-400 changed 25, top-1250 changed 47).
  - Spearman vs v3 0.9558 (39 / 101).
  - Spearman vs v7 0.9758 (33 / 47).
- Validation: 40,101 unique hashes, exactly matching v6's hash set; scores 2.5e-05..1.0; no NaN.
- File: sha256 2c25fef3491a0d67f869fd325ebe91f6e0ab95b7f3b4b81aebfe8e883119b59b.

- 14:38 London: Sahid reports v7 (submission_v7.csv) leaderboard 65.2 (new best, +0.1 vs v3/v6 65.1).

## Leaderboard sync (14:43 UK, read from site)
- 13:36 UTC (14:36 UK): submission_v6x.csv + code_solo_validator_v6x.zip -> 65.2 (P@1% 91%, R 29%, AUC 0.91)
- 13:37 UTC (14:37 UK): code_solo_validator_v7.zip (CSV presumably submission_v7.csv) -> 65.3 (best)
- 13:43 UTC (14:43 UK): submission_v5.csv + code_solo_validator_v5.zip, run "v5" -> 65.0 (P 92%, AUC 0.91)
- Tier "found" is top-1% only (401 rows < 800 easy positives), so medium/hard 0% is structural for every run.

## v7x — v7 self-training on top of v6x (outputs/submission_v7x.csv)
- What it is: src/pseudo.py with the same settings as v7 (top 0.8% pseudo-positive, bottom 60% pseudo-negative, negatives sharing a cmp900 cluster with a positive dropped, pseudo weight 0.5, test scored by cmp900-grouped 5-fold over the pseudo rows). It runs on the v6x feature set (FEATS=f2,f3,f4,f6, FE10=1), with pseudo-labels taken from v6x test scores. The v6 look-alike head is blended at 0.2.
- Pseudo set: 321 positives, 21,473 negatives (2,587 dropped for sharing a cluster).
- Reproduce: `V7X=1 bash run.sh` (builds v6x first if cache/pred_test_v6xfinal.parquet is missing). Add `NOHEAD=1` for the no-head variant.
- Fold-seed-42 train OOF (all/easy/medium/hard; pseudo rows are in every training fold):
  - v7x base 63.79/84.58/9.73/0.74; v7x+head 63.36/84.27/10.38/0.69 (this is the shipped one).
  - v6x base 62.91/83.62/7.35/0.75; v6x+head 62.58/83.43/7.51/0.70.
  - v7 base 61.45/81.44/8.84/0.78; v7+head 61.21/81.26/8.36/0.70.
- Test agreement (Spearman / top-400 changed / top-1250 changed):
  - vs v6x 0.9854 / 18 / 28.
  - vs v7 0.9820 / 26 / 48.
  - vs v6 0.9699 / 30 / 48.
- Validation: 40,101 unique hashes, same set as v6; scores 2.5e-05..1.0; no NaN.
- File: sha256 169210e1d0d68e546860b68ffd087c04723b5f01b5d8d111d752edc367f3b6c9.
- No-head variant (prepared, NOT shipped): cache/submission_v7x_nohead.csv, sha256 450183fa….
  - Spearman vs v6x 0.9787 (top-400 changed 25, top-1250 changed 107); vs v7 0.9663 (36 / 122).

## v9b -> v9c -> v9e -> v13 Final (Leaderboard 69.0 -> 70.5 -> 71.3 -> **71.6**)
- `v9b` (0.7 v7x_nohead + 0.3 Prototype Stage-1 -> closed-cluster propagation): **69.0 LB**
- `v9c_w60` (0.4 v7x_nohead + 0.6 Prototype Stage-1 -> closed-cluster propagation): **70.5 LB**
- `v9e_w90` (0.1 v7x_nohead + 0.9 Prototype Stage-1 -> closed-cluster propagation): **71.3 LB**
- `v10` (4-suite blend including CatBoost on 983-col matrix, unprotected): **71.1 LB** (CatBoost's 2.49% Medium AP on the 983-col matrix diluted LightGBM's 14.24% Medium AP).
- **`v13_protected_hard_med` (`src/v13_final.py`, `outputs/submission_v13_protected_hard_med.csv`)**: **71.6 LB (FINAL BEST)**
  - Stage 1: `0.08 * rank(v7x_nohead) + 0.82 * rank(Prototype_S1) + 0.10 * rank(3-seed Xs_983 LightGBM specialist with FE11 Hard-Relay & Medium-Cluster features)`
  - Stage 2: Multi-scale closed-cluster score propagation (`propagate_cluster_scores`)
  - Stage 3: Top-4% protected re-ranking (`0.75 * base + 0.15 * Xs_983_LGB + 0.10 * propagate_relay_episodes` below top 4%).
  - See `LEARNING_GUIDE_TO_80_AP.md` for full analysis and roadmap to 80+ AP.
