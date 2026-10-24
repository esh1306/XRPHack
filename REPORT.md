# Decode & Detect: The XRPL AI Hack — Method report

**Team Solo Validator** · captain and sole member: **Eshita Agarwal** · 24 October 2026 · Licence: MIT (`LICENSE`)

All numbers below come from the repository's logs and caches (`SUBMISSIONS.md`, `cache/*_cv.json`, `logs/`, `FINGERPRINTS.md`, `JEV.md`, `TSFM.md`, `MEMBERS.md`). AP = 100 × average precision. "easy/medium/hard" AP = that tier's positives plus all negatives.

## 1. Team and submissions

All leaderboard figures are from `SUBMISSIONS.md` (reported by the captain or read from the leaderboard). OOF figures are 5-fold grouped CV at fold seed 42 unless stated otherwise.

| Version | What changed | OOF AP all / easy / med / hard | Leaderboard |
|---|---|---|---|
| v2 | + flow & episode features (`fe2`, `fe3`), 5-model blend, stage-2 propagation | 61.78 / 83.64 / 4.64 / 0.77 | ~64.1 |
| v3 | + in→out chains, deviation (`fe4`); shift-robust feature handling | 60.35 / 80.49 / 5.45 / 0.77 | 65.1 |
| v4 | v3b + level-3 medium/hard specialist stacker | 61.60 / 81.89 / 5.90 / 0.81 | 64.7 |
| v5 | v3b alone, stage-1 weight 0.3 | 61.82 / 82.75 / 5.29 / 0.81 | 65.0 |
| v3gnn | 0.9 v3 + 0.1 Multi-GNN rank blend | pooled 60.64 | 64.8 |
| v6 | v3 + fe8 groups A,B,C,D,G + look-alike head (0.2) | 61.79 / 82.94 / 6.80 / 0.64 | 65.1 |
| v6x | v6 + BlazingAML-style `fe6` + 21 `fe10` medium-motif columns | 62.58 / 83.43 / 7.51 / 0.70 | 65.2 |
| v7 | v6 + transductive self-training (pseudo-labels) | 61.21 / 81.26 / 8.36 / 0.70 | 65.3 |
| v7x | v7 self-training on the v6x feature set, + head | 63.36 / 84.27 / 10.38 / 0.69 | 65.5 |
| v7x no-head | v7x without the look-alike head (`V7X=1 NOHEAD=1`) | 63.79 / 84.58 / 9.73 / 0.74 | 65.6 |
| v7x2 | second pseudo-labelling round (teacher = v7x no-head), no head | 61.51 / 80.99 / 10.07 / 0.76 | 65.7 |
| v9 | v7x no-head + closed-cluster score propagation (Sahid's earlier prototype, see §7) | 64.89 / – / 12.65 / – | 66.1 |
| v9b | 0.7 rank(v7x no-head) + 0.3 rank(prototype stage-1 model), then propagation | 65.64 / 85.41 / 13.99 / 0.76 | 69.0 |
| v9c_w40 / w50 / w60 | prototype stage-1 weight 0.4 / 0.5 / 0.6 | 65.54 / 65.35 / 65.05 (all) | 69.6 / 70.1 / 70.5 |
| v9e_w70 / w80 | prototype stage-1 weight 0.7 / 0.8 | 64.66 / 64.15 (all) | 70.9 / 71.1 |
| **v9e_w90** | prototype stage-1 weight 0.9 (0.1 ours), then propagation | **63.43 / 81.43 / 14.38 / 0.72** | **71.3 — best** |
| v11 (on w90) | + our research pattern layer below the fixed top 5% (PATTERNS 0.1 / 0.2) | 63.55 / 63.61 (all) | not yet scored |

**Final pick: v9e_w90** (`outputs/submission_v9e_w90.csv`, leaderboard 71.3; default `./run.sh`). Note that fold-42 OOF *falls* as the prototype weight rises while the leaderboard rises: CV did not track the leaderboard here. v1 (practice split only, 50.37 OOF) was never uploaded. v7/v7x OOF numbers include the pseudo-labelled test rows in every training fold. File hashes are in `SUBMISSIONS.md`.

## 2. Data and validation

**Data.** The organisers provide `transactions` (one row per hash; all are successful `Payment`s), `transfers` (1–3 executed legs per transaction; `direct_payment` or `path_payment`), hourly `market_rates` and `blocks`. Destination, Amount and SendMax are parsed from the `EXTRA_TRANSACTION_FIELDS` JSON. The train split has **40,131** labelled transactions (1 to 7 Sep 2026; 1,250 positives = 3.11%: 800 easy, 225 medium, 225 hard). The test split has **40,101** transactions from a later week (20 to 26 Sep). Wallets and hashes are disjoint between splits, so **no test account appears in train**. The labels also include `family`, `difficulty`, `scenario_id` and `cohort_id`. These are used only for grouping folds and reporting, never as features.

**Cross-validation.** We use 5-fold `StratifiedGroupKFold(shuffle=True, random_state=42)` grouped by `scenario_id`. Background rows form their own groups. This keeps every planted scenario entirely in one fold. All stage-2 and label-derived inputs are computed out-of-fold. Graph features are built transductively from each split's own 7-day graph, without labels, so test features never see train labels.

**Adversarial validation.** A train-vs-test classifier separates the weeks with **AUC 0.997** (`logs/fp_adversarial.log`). Almost all of that comes from account-address characters, because the two splits use different character sets. **No feature is derived from account IDs**; addresses are used only as graph node keys. Split-level graph statistics also shift between weeks. So we drop split-constant and giant-component columns, and percentile-rank core, clustering and cross-currency rates *within their own split*. In v3b/v4, a train-vs-test KS check moved six more columns into that ranked set (KS > 0.08). The same robustness explains the leaderboard order: v3 beat v2 on the leaderboard even though v2 has the higher CV.

## 3. Method

**Features (`src/fe*.py`, computed per split).**

- **Row and amount.** USD value uses the hourly reference rate for the payment's hour, keyed by **currency + issuer** (the same currency from two issuers is a different asset). The first executed leg is the actual cost; SendMax is only a ceiling. Also: cross-currency flag, number of legs and paths, round-amount indicators.
- **Account and pair context (`fe.py`, `fe2.py`).** Sender/receiver aggregates, pair history, 3-day and 7-day flow windows, running balances, same-asset in/out conservation, counterparty novelty, amount z-scores.
- **Temporal motifs and graph (`fe.py`).** Forwarding or receiving similar amounts within 5m/1h/6h/24h, bursts, fan-in/out, PageRank, HITS, k-core, clustering, reciprocity, parallel paths, and time-respecting 2-hop cycles and chains. These reimplement the IBM AMLworld / GFP typology idea; no GFP code is used.
- **Episode reconstruction (`fe3.py`).** Inflow and outflow sessions per (account, asset) at 5m/30m/2h gaps. Temporal components (transactions sharing an account within 2m/15m/1h) and flow chains (inflow → next outflow within 8m/1h/4h), with size, duration, accounts, assets, value and conservation.
- **In→out amount-matched chains and deviation (`fe4.py`).** USD in→out matching with tolerance for fees and FX spread, greedy multi-hop chain following with amount decay, peel chains. Also *out-of-character* deviation from the account's own leave-one-out amount, hour and gap baseline, cohort synchrony, and episode-level aggregates broadcast back to each transaction.

**Models (`src/final.py`).**

- **Stage 1** is a rank blend of LightGBM (3 seeds), XGBoost and CatBoost, with early stopping on the grouped folds. Each model is then refit on all of train.
- **Stage 2** is collective classification. Out-of-fold stage-1 scores are propagated to the account, pair, temporal-neighbour, 2-hop and component level (leave-one-out mean and max), and the same model family is retrained on them.
- **Final score.** Stage 1 and stage 2 are rank-blended, with the weight chosen on OOF AP (0.5 in v3, 0.3 in v3b). `make_submission.py` checks the IDs, writes rank-normalised scores in [0,1] with no ties, and adds a short `reason`.

## 4. Results

**Feature ablation** (single LightGBM, seed 42, fold seed 42; `logs/report_ap_from_cache.log`, `TSFM.md`):

| Feature set | AP all | easy | medium | hard |
|---|---|---|---|---|
| `fe` only (146 cols; 400 rounds, `TSFM.md`) | 50.96 | 68.97 | 3.01 | 0.67 |
| `fe+fe2+fe3` (episodes) | 57.72 | 78.17 | 7.10 | 0.58 |
| + `fe4` (chains, deviation) | **59.73** | 80.68 | 5.42 | 0.60 |
| + `fe4`, fold seed 7 | 60.92 | 81.08 | 6.16 | 0.80 |

**Pipeline stages** (`cache/pred_test_v3_cv.json`, `cache/pred_test_v3b_cv.json`). v3: stage-1 blend 58.85 → stage-2 blend 58.67 → final (w = 0.5) **60.35**. v3b: stage-1 58.44 → stage-2 60.29 → final (w = 0.3) **61.82**. Individual stage-1 models range from 55.6 to 58.9.

**Leaderboard vs CV.** Leaderboard order v3 (65.1) > v4 (64.7) > v2 (~64.1) is the reverse of CV order (v2 61.78 > v4 61.60 > v3 60.35). Changing the fold seed alone moves single-model AP by 1.2 (59.73 vs 60.92), and changing the LightGBM seed moves it by up to 1.8 (`FINGERPRINTS.md`). **CV differences of about 1 AP are within noise**, so we now accept a change only if it beats multi-seed spread.

## 5. Early experiments (v1–v5, same grouped CV)

ExSTraQt-style flow/Louvain features (`fe5`, −1.8); BlazingAML motif counts (`fe6`, ±0, kept additively in v6x); frozen Chronos-2 embeddings (TCEE-style, sign inconsistent); JEV API probabilities (+1.41, but a shuffled control gave +1.12, so not used); generator-fingerprint scan (no artifact); behaviour-change features (`fe7`, −1.8/−3.0); a medium/hard specialist stacker (v4, LB 64.7 < 65.1); GARG-AML, a Multi-GNN-style GNN and TabICLv2 members (`MEMBERS.md`; unproven, not shipped). Details: `SUBMISSIONS.md`, `TSFM.md`, `JEV.md`, `FINGERPRINTS.md`.

## 6. Findings and limitations

- **Medium and hard sit near the floor.** The base rate of a tier subset is about 0.58%. Medium reaches 5–7 AP, roughly 10× random but low in absolute terms. Hard stays at 0.6–0.8 AP, which is random. The organisers state that **25 of the 225 hard cases are labelled independently of observable features** by design. In the relay families, each cohort hides about one real scenario among about 99 look-alike decoy scenarios (`MEMBERS.md`), and no member we tried separates them.
- **Look-alike legitimate scenarios dominate the false positives.** In v3 OOF (`logs/report_fp_analysis.log`), 62 of the 63 negatives in the top 400 and 1,083 of the 1,134 negatives in the top 2,000 belong to legitimate planted look-alike scenarios. Within the top 2,000, aggregate_relay (317) and distributed_relay (266) are the largest families.
- **CV–leaderboard gap and AP noise.** The leaderboard is about 4–5 points above CV, and the two rank versions differently. With 1,250 positives (about 450 medium + hard), AP differences under about 1.5 are not reliable.
- **Week shift.** The test week includes "batch variations absent from training" (README), such as SendMax/conversion mix with KS ≈ 0.095. We rely on within-split percentile ranking and drop ID-like features, but some residual shift is likely.
- **Synthetic-only.** Typology thinking transfers, but the thresholds and every learned weight are fitted to a simulator. Nothing here is validated on the real XRP Ledger.

## 7. Final results (v6 → v9e, v11)

**What worked.**

- **fe8 feature groups on v3 (v6).** A (row/amount/time), B (pair cadence/recurring), C (layering chains), D (gather bursts), G (Isolation Forest rank). Stage-1 single LightGBM, mean over fold seeds 42/7/2026: 56.88±0.09 → 59.35±0.64 (+2.47). 3-seed full pipeline: v3 60.33 → v6 + head 60.99 (sd 0.72). Leaderboard tied v3 (65.1).
- **BlazingAML-style motifs (`fe6`) + `fe10` medium-motif columns (v6x).** `fe10` adds stretched-motif detectors for medium cases: one-off pairs between non-hub accounts with up to 3 h of tolerance. Paired over 6 fold seeds, `fe10` gives +0.59±1.89 overall and 6.3 fewer look-alike false positives in the top 400 (single model; noisy). Full pipeline at fold seed 42: base 61.09 → 62.91, medium 5.52 → 7.35. Leaderboard 65.2.
- **Transductive self-training (v7, v7x).** The top 0.8% of test scores become pseudo-positives (321) and the bottom 60% pseudo-negatives (~21.5k), with weight 0.5. Pseudo-negatives that share a cmp900 cluster with a pseudo-positive are dropped, and each pseudo row is scored out-of-fold. Largest medium gain before v9: v6x base medium 7.35 → v7x base 9.73, overall 62.91 → 63.79. Leaderboard 65.6 (no head).

- **Closed-cluster score propagation + second model (v9, v9b): the breakthrough (65.6 → 69.0).** Both come from Sahid's earlier prototype, *Decode & Detect AML investigator agent*: a separate codebase by Sahid (MIT, Copyright (c) 2026 Sahid), written before this submission. It is used **without** its Jev/OpenAI components (no network calls). Its licence is kept in `prior_work/prototype_agent/` and credited in `PRIOR_WORK.md`.
  - **v9:** smooths our v7x no-head scores over closed clusters of first-time payments (5 min–4 h windows). Fold-42 OOF 63.79 → 64.89, medium 9.73 → 12.65. The same propagation also adds +0.79 to +1.41 on older OOFs (v6, v6x). Leaderboard 66.1.
  - **v9b:** blends 0.7 rank(ours) + 0.3 rank(the prototype's stage-1 model), then propagates. OOF 65.64 / 85.41 / 13.99 / 0.76, against 65.35 at weight 0.5. The prototype model alone scores 61.72 / 79.03 / 11.68 / 0.71 on our CV, and its Spearman with ours on test is 0.705, so the two models are complementary. Leaderboard **69.0**.

- **Prototype-weight sweep (v9c, v9e).** Raising the prototype stage-1 weight from 0.3 to 0.9 raised the leaderboard monotonically (69.0 → 69.6 → 70.1 → 70.5 → 70.9 → 71.1 → **71.3**) while fold-42 OOF fell (65.64 → 63.43). We followed the leaderboard for the final pick and report the CV disagreement openly.
- **Our research pattern layer (v11, `src/patterns_research.py`).** 29 cheap, time-windowed per-transaction pattern features, each tied to a source: AMLworld typologies (Altman et al., NeurIPS 2023 D&B: fan-out, fan-in, scatter-gather, gather-scatter, 2-/3-cycles, bipartite reach, stack/peel chains); Egressy et al. (AAAI 2024: reverse-edge activity, port numbering, ego-ID cycles); Elliptic (Weber et al., KDD 2019 workshop: neighbourhood aggregates); FATF virtual-asset red flags (structuring just below thresholds, rapid pass-through). A HistGradientBoosting model on these alone gets 34.30 / 38.85 / 8.32 / 0.82 OOF. Rank-blended over the whole list it **hurt** (v9c_w60 65.05 → 61.53 at 0.1). Keeping the top 5% fixed and blending only below it helped slightly: on v9c_w60 65.05 → 65.16 / 65.20 (0.1 / 0.2), on v9e_w90 63.43 → 63.55 / 63.61, medium +0.15–0.26, hard +0.01–0.03. Weights and the 5% block were chosen on the same OOF, so expect mild selection bias.

**What failed** (same grouped CV).

| Idea | Result |
|---|---|
| More fe8/fe9 groups on top of v6 (episodes, local bursts, one-off pairs, account-week context, fee/afternoon habits) | every group −0.5 to −2.3 (3-seed stage-1); none added |
| RMIT thesis / MPOCryptoML (arXiv 2508.12641): personalised PageRank, NTS, NWS (`fe9_rmit.py`) | ref 58.69 vs +PTW 58.81 mean over 3 seeds; signs inconsistent per seed; not used |
| Medium specialist linear blend (`spec10` at weight 0.1/0.2) | base 58.02 → 53.92 / 49.82; medium +0.7 but look-alike FPs in the top 400 rose from 67 to 77–89 |
| v8 tiered re-rank (keep base top-K, reorder below K by a specialist trained without easy positives) | on v7x base: −0.73 (best, K = 4%) to −16.9; specialist alone medium 2.4–4.6 < base 9.73 |
| Second self-training round (v7x2) | OOF 63.79 → 61.51 (easy −3.6, medium +0.34); leaderboard 65.7 (+0.1) |
| Look-alike head on v6x/v7x | v7x + head 65.5 < no-head 65.6 on the leaderboard; dropped |
| Multi-GNN, TabICLv2, GARG-AML members | see §5; not in the final files |

**Findings.**

- **The leaderboard's top 1% is entirely easy cases.** It contains 401 rows, against 800 easy positives in test. Every run shows 91–93% precision there (AUC 0.91), so the site's medium/hard "found" counts are structurally 0. All remaining AP gain sits below that block.
- **Medium and hard cases are hidden among look-alike legitimate "twin" scenarios.** In OOF, 62 of the 63 false positives in the top 400 come from planted look-alike scenarios. A model trained without easy positives loses the shared motif signal, so it is worse on medium than the full model.
- **Week shift caps the CV → leaderboard transfer.** CV gains of +1 to +3 moved the leaderboard by only 0.1–0.5. v7x no-head (65.6) is +0.5 over v3 (65.1); only v9b's propagation and second model broke through (69.0).

**Limitations.** These are single-fold-seed numbers for v6x, v7 and v7x, where the 3-seed sd is about 0.4–0.7. Pseudo-labels are only as good as the base ranking, and round-2 self-training (v7x2) lowered OOF by 2.3. Hard cases (0.6–0.8 AP) remain essentially unsolved. Everything is fitted to a simulator.

**Final submission.** `outputs/submission_v9e_w90.csv` + `outputs/code_solo_validator_final.zip` (default `./run.sh`; `BLEND_OURS` and `PATTERNS` select the other variants), leaderboard **71.3**.

**Related work / literature consulted.** AMLworld, Egressy et al., Elliptic and FATF red flags feed our pattern layer (above). BlazingAML (arXiv:2604.12241; motif counts, `fe6`), MPOCryptoML (arXiv:2508.12641; PPR/NTS/NWS, no gain), the Elliptic article on chain-hopping (maps to XRPL currency hops, already covered by our features), IBM AMLworld (typology design). Score/label propagation over graph clusters is a well-known graph-smoothing approach (collective classification). We do **not** attribute the v9 gain to any of these papers: it came from Sahid's earlier prototype described above.

## 8. Reproducibility and disclosures

**Reproduce.** Run `./run.sh data/test outputs/submission.csv`. It expects the organisers' `data/train/` (with `labels.csv`) and the test files. It uses Python 3.13 with pinned `requirements.txt` (lightgbm 4.7.0, xgboost 3.4.1, catboost 1.2.10, scikit-learn 1.9.1, pandas 3.1.0, numpy 2.5.4, networkx 3.7, scipy 1.18.1, pyarrow 26.0.0, joblib 1.6.0). It runs on CPU only (8 cores, 16 GB RAM) with all seeds fixed (42; LightGBM seeds 42/1/2). The last logged run took about 6 minutes with cached train features (`logs/run_20261024_125634.log`); building the train features as well adds a few minutes. `STACK=1` reproduces v4. The default path reproduces the final pick (v9e_w90): v7x no-head base, the prototype stage-1 model, rank blend `BLEND_OURS=0.1`, cluster propagation, then `make_submission.py`. Re-running it gives the same ranking as the uploaded CSV (see `SUBMISSIONS.md`). `PATTERNS=0.1/0.2` adds the v11 pattern layer. The prototype step reads `data/train` and `data/test` directly. Logs are written to `logs/`.

**Disclosures.**

- The code was built with an AI coding assistant (**Grok Bot**). **No LLM or external API is used inside the model or `run.sh`.**
- **Pseudo-labelling (v7/v7x):** the final model trains on its own high- and low-confidence scores for the **unlabelled test transactions** (transductive self-training). No test labels are used.
- **BlazingAML-inspired features:** `fe6.py` reimplements the scatter-gather, gather-scatter and 3-cycle motif counts described in *BlazingAML* (arXiv:2604.12241). No BlazingAML code is used.
- **JEV (TypeSafe AI)** was tested through its API. Only aggregated per-account summaries from the train and validation splits were sent: no hashes, labels, raw transactions or test data. It is not used in the final model.
- **Chronos-2** (`amazon/chronos-2`, Apache-2.0) was tested as a frozen encoder and is not used.
- TabICLv2 (`tabicl==2.2.0`, pretrained checkpoint) and a PyTorch GNN were tested as ensemble members (`requirements-members.txt`). They are not in the shipped files.
- **Prior work reused:** cluster propagation and the second model (v9/v9b) are from Sahid's earlier prototype, *Decode & Detect AML investigator agent* (separate codebase, MIT, Copyright (c) 2026 Sahid). They were not developed during the hackathon. Its licence is preserved in `prior_work/prototype_agent/`, and its Jev/OpenAI parts are not used.
- **No external data is used.** The organisers' data and `payments.py` are not redistributed. The libraries are listed in `requirements.txt`.

**References.** Weber et al., *Anti-Money Laundering in Bitcoin: Experimenting with GCNs for Financial Forensics*, KDD 2019 workshop. Egressy et al., AAAI 2024. FATF, *Red Flag Indicators of ML/TF Associated with Virtual Assets*, 2020. Altman et al., *Realistic Synthetic Financial Transactions for AML Models* (IBM AMLworld), NeurIPS 2023 D&B. Egressy et al., *Provably Powerful GNNs for Directed Multigraphs* (Multi-GNN), arXiv:2306.11586. Deprez et al., *GARG-AML against Smurfing*, arXiv:2506.04292. Tariq & Hassani, *ExSTraQt*, arXiv:2604.02899. *BlazingAML*, arXiv:2604.12241. Samadi (RMIT thesis) / *MPOCryptoML*, arXiv:2508.12641. Savvidi et al., *TCEE*, ICAIF'26, arXiv:2610.29667. Ansari et al., *Chronos-2*, Amazon, 2025.
