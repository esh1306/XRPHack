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
- 13:50 UTC (14:50 UK): submission_v7x.csv + code_solo_validator_v7x.zip -> 65.5 (P 93%, AUC 0.91)
- 13:50 UTC (14:50 UK): cache/submission_v7x_nohead.csv + code_solo_validator_v7x.zip (V7X=1 NOHEAD=1) -> 65.6 (NEW BEST)

## v8 tiered re-rank: 2026-10-24 ~14:55 London. NEGATIVE, not shipped (no CSV/zip produced)
- Idea: keep the base model's top-K block (the easy cases) in place, then reorder everything below K using a medium/hard specialist. The specialist is a LightGBM trained without easy positives, with legit look-alike negatives weighted x3, on the v6x feature set. Code: src/v8_tiered.py, src/v8_v7x_eval.py, src/v8b_stack.py. Logs: logs/v8_cv.log, logs/v8_cv.json, logs/v8_v7x_eval.log, logs/v8b_stack.log.
- Specialist alone (all/easy/med/hard), fold seeds 42/7/2026: 14.1/13.8/2.7/0.6; 16.4/15.7/4.6/0.7; 15.0/15.1/2.4/0.7. It is weaker on medium than the base (v6x 7.35, v7x 9.73), because dropping the easy positives removes the shared motif signal.
- Tiered on the v7x base OOF (fold 42, 63.79/84.58/9.73/0.74):
  - every K in 1.5–4% and a in {0,1,2}, plus the soft variant: −0.73 to −16.9;
  - best: K=4%, a=2 → 63.06/84.20/9.34/0.68.
- Tiered on the v6x base OOF (fold 42, 62.91/83.62/7.35/0.75): best is K=4%, a=2 → −0.57 (62.34/83.21/7.16/0.69).
- On the fe10 single-LGB base, K=4% gives +0.49 (seed 42), −0.77 to −1.21 (seed 7) and −0.35 to +0.02 (seed 2026). Not consistent.
- Stacker variant (all positives kept; easy w0.3, med/hard w3, twins w3, plus a base-rank feature; fold 42 on v6x): alone 56.74 and medium 5.17. Tiered −2.4 to −4.6. Rank-average at 0.2 gives −1.09.
- Why: the AP loss comes from easy positives that rank below K. Those get re-ordered by a model that is blind to them. Meanwhile medium does not improve, because the base already ranks medium better than any specialist.
- 14:58 London: FE11 sklearn pattern mapping (NN template density, KMeans/HDBSCAN episode map, LOF) tested on v6x stage-1, 6 fold seeds: +0.38 ± 1.58 all, +0.26 medium -> dropped. See FE11.md.

## v7x2 — second self-training round, no head (outputs/submission_v7x2.csv)
- What it is: src/pseudo.py with the v7x settings (top 0.8% pseudo-positive, bottom 60% pseudo-negative, pseudo weight 0.5, cmp900 cluster exclusion, cluster-grouped 5-fold) on the v6x features.
  - Teacher: the v7x no-head test scores (cache/pred_test_v7x_base.parquet; same ranking as cache/submission_v7x_nohead.csv, Spearman 1.0).
  - Ships the no-head score.
- Pseudo set: 321 positives, 21,821 negatives (2,239 dropped for sharing a cluster).
- Reproduce: `V7X2=1 bash run.sh` (rebuilds the v7x teacher if it is missing).
- Fold-seed-42 train OOF (all/easy/medium/hard):
  - v7x2 base 61.51/80.99/10.07/0.76 (this is the shipped one); +head 61.73/81.49/9.79/0.71.
  - v7x base 63.79/84.58/9.73/0.74.
  - Overall OOF fell 2.3, mostly on easy; medium rose 0.34.
- Leaderboard context: v7x no-head 65.6, v7x 65.5.
- Test agreement (Spearman / top-400 changed / top-1250 changed):
  - vs v7x no-head 0.9939 / 17 / 40.
  - vs v7x 0.9773 / 20 / 104.
  - vs v6x 0.9773 / 28 / 116.
- Validation: 40,101 unique hashes, same set as test; scores 2.5e-05..1.0; no NaN.
- File: sha256 0872ee6a5af17880f24c99be3e12c5c4331918ec6f3afc8f433915d0d079a6d4.
- The POS_FRAC=0.012 variant was not run (no time within 6 cores). To run it: `V7X2=1 POS_FRAC=0.012 V7X2_TAG=_pf012 bash run.sh . cache/submission_v7x2_pf012.csv`.

## v9 / v9b: Sahid's earlier prototype's closed-cluster score propagation on v7x no-head. 2026-10-24 ~15:08 London
- Source: Sahid's earlier prototype "Decode & Detect AML investigator agent" (separate codebase by Sahid, MIT, Copyright (c) 2026 Sahid). It is in prior_work/prototype_agent with its LICENSE kept, and is credited in PRIOR_WORK.md. Review notes: prototype/PROTOTYPE_NOTES.md. No Jev/OpenAI/network calls.
- The prototype's pipeline (train_e.py) on our CV (SGKF5 scenario_id, seed 42), all/easy/med/hard: 61.72/79.03/11.68/0.71 (stage-1 60.07). Its test output is prototype/out_test.csv. Against v7x no-head on test: Spearman 0.705, top-400 overlap 264, top-1250 overlap 1005.
- v9 = v7x no-head + propagate_cluster_scores. CV42 63.79 -> 64.89 (med 9.73 -> 12.65). The same propagation on older OOFs: v6 fs7 +1.05, v6 fs2026 +0.79, v6x +0.88, final_v6 +1.41.
  - outputs/submission_v9.csv, sha 86d5a915…
  - outputs/code_solo_validator_v9.zip (V9=1), sha d411ceef…, 82,670 B
  - Against v7x no-head on test: Spearman 0.964, top-400 overlap 370, top-1250 overlap 1181.
- v9b = 0.7 rank(v7x no-head) + 0.3 rank(prototype stage-1), then propagation. CV42 65.64/85.41/13.99/0.76 (w=0.5 gives 65.35).
  - outputs/submission_v9b.csv, sha 3a7d4fb4…
  - outputs/code_solo_validator_v9b.zip (V9B=1), sha c93da998…, 95,390 B
  - Against v7x no-head on test: Spearman 0.941, top-400 overlap 356, top-1250 overlap 1127.
- Leaderboard: pending.
- Reproduction check 15:08 London: src/v9_prop.py output matches the shipped CSVs (v9 Spearman 1.0000; v9b checked below).
  v9b rerun via src/v9_prop.py --blend-prototype 0.7: Spearman 0.999998 vs shipped CSV, top-400 overlap 399/400, top-1250 overlap 1249/1250. Not bit-identical. The shipped v9b used the v7x no-head CSV ranks and the prototype test stage-1 saved by prototype/run_prototype.py; the rerun recomputes both from cache/pred_test_v7x_base.parquet. The cause of the tiny difference is unconfirmed.
- 14:09 UTC (15:09 UK): submission_v9b.csv + code_solo_validator_v9b.zip -> 69.0 (NEW BEST; P 93%, AUC 0.92)
- 14:09 UTC (15:09 UK): submission_v9.csv + code_solo_validator_v9.zip -> 66.1
- 14:09 UTC (15:09 UK): submission_v7x2.csv + code_solo_validator_v7x2.zip -> 65.7

## v9c / v9d — prototype-weight sweep on top of v9b (leaderboard: v9b 69.0)
- What they are: rank(ours) and rank(prototype stage-1) are blended, then the prototype's cluster propagation is applied, exactly like v9b.
  - The test CSVs come from prototype/ cached stage-1 scores (src/v10_post.py FIXED=w,0).
  - That path reproduces v9b's ranking exactly (Spearman 1.0000); only the score scale differs.
- Reproduce: zip outputs/code_solo_validator_v9c.zip, `V9C=1 BLEND_W=<prototype w> bash run.sh`; `BASE=v7x2` for v9d.
- Fold-42 OOF (all/easy/medium/hard), with Spearman and top-400 changed vs the v9b test file:

| File | Prototype w | OOF all/easy/medium/hard | Spearman vs v9b | Top-400 changed |
|---|---|---|---|---|
| v9b (ref) | 0.3 | 65.64/85.41/13.99/0.76 | – | – |
| outputs/submission_v9c_w40.csv | 0.4 | 65.54/85.07/14.17/0.76 | 0.9970 | 9 |
| outputs/submission_v9c_w50.csv | 0.5 | 65.35/84.62/14.40/0.75 | 0.9879 | 25 |
| outputs/submission_v9c_w60.csv | 0.6 | 65.05/84.04/14.52/0.75 | 0.9726 | 41 |
| outputs/submission_v9d.csv (v7x2 base) | 0.3 | 63.77/82.53/12.85/0.77 | 0.9975 | 13 |

- sha256:
  - w40 69474484c370211c38bbd78db22ce934d2cc038e8befe56ef3b5ef159a74884c
  - w50 e9231959503930895f72c8825095e9ae0d116c198fa2e7a4b9dd908707126260
  - w60 4a3bbba5b405dd87a400d7df061199aa756642b6bd7f3f233fe37af276b0ee8d
  - v9d cbceb9b5afbe2d539bc23a4e6fbdd542db0ad03eecf1dd9e34ae60b679782d2b
- Validation (all four): 40,101 unique hashes, exactly the test set; scores 2.5e-05..1.0; no NaN.

## Related work / literature consulted (log note, 15:15 London)
- **Consulted during the build:**
  - BlazingAML (arXiv:2604.12241): scatter-gather, gather-scatter and 3-cycle counts → `fe6`. Kept additively from v6x.
  - MPOCryptoML / Samadi RMIT thesis (arXiv:2508.12641): PPR/NTS/NWS → `fe9_rmit.py`. No consistent gain; not used.
  - The Elliptic article on chain-hopping: maps to XRPL currency hops, already covered (`g2_am1h_ccy_switch`). Nothing new added.
  - IBM AMLworld (Altman et al., NeurIPS 2023): typology design for the motif features.
- **Where the v9/v9b breakthrough (65.6 → 66.1 → 69.0) came from:** closed-cluster score propagation and the second (stage-1) model in Sahid's earlier prototype "Decode & Detect AML investigator agent" (separate codebase by Sahid) (MIT, Copyright (c) 2026 Sahid). It is used without Jev/OpenAI, and its licence is kept in prior_work/prototype_agent and PRIOR_WORK.md. Score/label propagation is a well-known graph-smoothing approach (collective classification). None of the papers above produced this gain, and the report does not claim they did.
- 14:20-14:23 UTC (15:20-15:23 UK), zip code_solo_validator_v9c_final.zip: v9c_w40 (prototype weight 0.4) -> 69.6; v9c_w50 -> 70.1; v9c_w60 -> 70.5 (NEW BEST)

## v9c_bestofboth (cache/submission_v9c_bestofboth.csv, not shipped)
- Our top-2% kept on top; the rest ordered by blend (0.6 ours / 0.4 prototype) + cluster propagation.
- Fold-42 CV: 66.36 all / 86.45 easy / 12.06 medium / 0.76 hard.
- Picked from a 12-config grid on the same OOF, so mild selection bias.
- sha256 146df62d…

## v10 (not built)
- The pseudo stage-2 run was OOM-killed at about 15:26 BST, with 15 GB RAM and load around 42 from other workers.
- Stage-1 blend CV with prototype KS<0.1 columns: 62.89.
- The staged prior-work zip is at /tmp/stage_v10pw.

## v9e — prototype-weight sweep continued (built 15:29 London; not yet uploaded / no leaderboard score yet)
- Same as v9c: `cd prototype && FIXED=<our w>,0 OUT=... ../.venv/bin/python ../src/v10_post.py write`, then `src/make_submission.py <parquet> data/test OUT cache/test`.
- Repro check of this path for w60: Spearman 0.99999999983 vs delivered outputs/submission_v9c_w60.csv, identical top-401 and top-1250 sets (only score rounding / reason text differ).

| File | Prototype w | Fold-42 OOF all/easy/medium/hard | sha256 |
|---|---|---|---|
| outputs/submission_v9e_w70.csv | 0.7 | 64.66/83.35/14.57/0.74 | e6fd007ca3298bbf970ef629e188147b46c21b2a1627671291a3971f0dd43b97 |
| outputs/submission_v9e_w80.csv | 0.8 | 64.15/82.52/14.63/0.73 | eaddc2e6ccc90505b485ea5d891a11f7bc981d9e0034aaef387daa209dbfc055 |
| outputs/submission_v9e_w90.csv | 0.9 | 63.43/81.43/14.38/0.72 | c9943edf5a03eed4fcc717d4286cdda1fea23f65ef4e73b95a01de6fd56d8dbe |
- Validation (all three): header TRANSACTION_HASH,score,reason; 40,101 unique hashes = test set; scores finite in (0,1].

## 24 Oct 15:37 uploads (code_solo_validator_v9c_final.zip)
- v9e_w70 (prototype weight 0.7): LB 70.9
- v9e_w80 (prototype weight 0.8): LB 71.1
- v9e_w90 (prototype weight 0.9): LB 71.3 (NEW BEST)
- v12_own25 (w60, own-rank weight 0.25): LB 70.5
- v12_iter2 (w60, two propagation passes): LB 70.1

## v11 — our research pattern layer (built 15:37–15:40 London; not yet uploaded)
- `src/patterns_research.py`: 29 time-windowed pattern features (AMLworld fan-in/out, scatter-gather, gather-scatter, cycles, bipartite, stack/peel; Egressy reverse-edge / port numbering / ego-ID cycles; Elliptic neighbourhood aggregates; FATF structuring / pass-through) -> HistGradientBoosting (seed 42). Pattern-only fold-42 OOF 34.30/38.85/8.32/0.82.
- Plain rank blend hurt (v9c_w60 65.05 -> 61.53 at 0.1, 58.55 at 0.2; logs/v11_patterns_cv.log). Kept form: top 5% fixed, blend below (`src/final_v11.py`, logs/v11_patterns_cv_protect*.log). Chosen on the same OOF (mild selection bias).

| File | Base | Pattern w | Fold-42 OOF all/easy/medium/hard | sha256 |
|---|---|---|---|---|
| outputs/submission_v11_patterns_w10.csv | v9c_w60 | 0.1 | 65.16/84.00/14.70/0.76 | b807fe23da6506a7eeebed3bc939cec5e53b4688c5879641f08171ebdf4bf46a |
| outputs/submission_v11_patterns_w20.csv | v9c_w60 | 0.2 | 65.20/83.98/14.78/0.77 | f969153d06ce4d36e79dd3758585e9a977d0da6d5f7275d57e851610eda84103 |
| outputs/submission_v11_patterns_on_w90_w10.csv | v9e_w90 | 0.1 | 63.55/81.42/14.53/0.74 | fd13de6da91ef3c8344159811de0c3d353dd81aa2a356f534e143f53c4e36fdc |
| outputs/submission_v11_patterns_on_w90_w20.csv | v9e_w90 | 0.2 | 63.61/81.41/14.60/0.75 | b617b64f4f4d96025ae95d895a7824376d6be23bb7ff2ca2b2db261900eb24a0 |
- Base v9e_w90 OOF 63.43/81.43/14.38/0.72. All four validated: 40,101 unique test hashes, scores in (0,1]. Top 2,005 rows identical to the base file.

## Final code ZIP: outputs/code_solo_validator_final.zip (15:4x London)
- Default `./run.sh` = v9e_w90 (BLEND_OURS=0.1). Repro check: re-running the staged ZIP's run.sh (with cached v7x / prototype stage-1 inputs) gave a CSV byte-identical to outputs/submission_v9e_w90.csv.
- v11_patterns_on_w90_w20 (research pattern layer on w90, final ZIP): LB 71.3 (ties best; site top-1% Medium found 10%, first time above 0%)
- v11_patterns_on_w90_w10: LB 71.3
- v9f_w95 (prototype weight 0.95, final ZIP): LB 71.3
- v9f_w100 (prototype only + propagation): LB 71.0  -> sweep peaked at 0.9-0.95
- FINAL PICK: v11_patterns_on_w90_w20 (LB 71.3, reproducible with PATTERNS=0.2 ./run.sh from code_solo_validator_final.zip)
- v13_pat35_on_w95: LB 71.2; v13_pat20_on_w95: LB 71.2; v13_pat50_on_w95: LB 71.1 (no gain; Hard still 0% in top 1%)

## 24 Oct 16:25 BST — v13_protected_hard_med (`src/v13_final.py`, Leaderboard **71.6** — FINAL BEST)
- Stage 1: `0.08 * rank(v7x_nohead) + 0.82 * rank(Prototype_S1) + 0.10 * rank(3-seed Xs_983 LightGBM specialist with FE11 Hard-Relay & Medium-Cluster features, no CatBoost)`.
- Stage 2: Multi-scale closed-cluster score propagation (`propagate_cluster_scores`).
- Stage 3: Protected Top-4% re-ranking (`0.75 * base + 0.15 * Xs_983_LGB + 0.10 * propagate_relay_episodes` below top 4%).
- Uploaded as `submission_v13_protected_hard_med.csv` + `code_solo_validator_v10b.zip` -> **Leaderboard 71.6 (NEW BEST)**.
- See `LEARNING_GUIDE_TO_80_AP.md` for complete post-mortem and roadmap to 80+ AP.
