#!/usr/bin/env bash
# Usage: ./run.sh [TEST_DIR] [OUT_CSV]
# Reproduces the predictions CSV from the organisers' data. TEST_DIR defaults to data/test.
# Expects data/train/ (with labels.csv) and TEST_DIR/{transactions,transfers,blocks}.parquet + market_rates.csv.
set -euo pipefail
cd "$(dirname "$0")"
TEST_DIR="${1:-data/test}"; OUT="${2:-outputs/submission.csv}"
PY=.venv/bin/python
export NJOBS="${NJOBS:-1}"  # one thread per model process; models run in parallel
[ -x "$PY" ] || { python3 -m venv .venv && .venv/bin/pip install -q -r requirements.txt; }
mkdir -p cache outputs logs "$(dirname "$OUT")"
exec > >(tee -a "logs/run_$(date +%Y%m%d_%H%M%S).log") 2>&1
# ---------------------------------------------------------------------------------------------------------------
# FINAL (default since v9c; FINAL=0 gives the older variants below). Reproduces the chosen leaderboard CSV:
#   1. our v7x no-head scores (v6x features + transductive self-training) -> cache/pred_test_v7x_base.parquet
#   2. Sahid's earlier prototype stage-1 ensemble (prior_work/prototype_agent, MIT) -> prototype/cache/test_stage1.npy
#   3. rank blend BLEND_OURS*rank(ours) + (1-BLEND_OURS)*rank(prototype stage-1), then the prototype's closed-cluster
#      score propagation (src/v10_post.py)
#   4. optional: our research pattern layer (src/patterns_research.py) rank-blended at weight PATTERNS (src/final_v11.py)
#   5. src/make_submission.py: ID check, rank-normalised scores in [0,1], reason column
# Variants: BLEND_OURS=0.1 (default, v9e_w90, leaderboard 71.3); 0.2 -> v9e_w80 (71.1); 0.3 -> v9e_w70 (70.9); 0.4 -> v9c_w60 (70.5); 0.5 -> v9c_w50 (70.1); 0.6 -> v9c_w40 (69.6);
#           0.7 -> v9b (69.0). PATTERNS=0 (default); PATTERNS=0.1 / 0.2 -> v11_patterns_on_w90_w10 / w20 (top 5% kept fixed).
if [ "${FINAL:-1}" = "1" ]; then
  BLEND_OURS="${BLEND_OURS:-0.1}"; PATTERNS="${PATTERNS:-0}"
  [ -f cache/pred_test_v7x_base.parquet ] || FINAL=0 POS_FRAC=0.008 V7X=1 NOHEAD=1 bash "$0" "$TEST_DIR" cache/submission_v7x_nohead_rebuild.csv
  mkdir -p prototype/cache
  (cd prototype && for s in train test; do [ -f cache/D_$s.parquet ] || ../$PY build_design.py $s; done)
  (cd prototype && { [ -f cache/test_stage1.npy ] && [ -f cache/oof42_stage1.npy ] || ../$PY run_prototype.py; })
  (cd prototype && FIXED=$BLEND_OURS,0 OUT=../cache/final_raw.csv ../$PY ../src/v10_post.py write)
  if [ "$PATTERNS" != "0" ]; then
    [ -f cache/pat_test.parquet ] || $PY src/patterns_research.py test cache/pat_test.parquet
    $PY src/final_v11.py cache/final_raw.csv cache/final_pred.parquet cache/pat_test.parquet "$PATTERNS"
  else
    $PY src/final_v11.py cache/final_raw.csv cache/final_pred.parquet
  fi
  $PY src/make_submission.py cache/final_pred.parquet "$TEST_DIR" "$OUT" cache/test
  exit 0
fi
# ---------------------------------------------------------------------------------------------------------------
if [ ! -f cache/train_f6.parquet ] || [ "${REBUILD_TRAIN:-0}" = 1 ]; then
  $PY src/fe.py  data/train cache/train_f.parquet
  $PY src/fe2.py data/train cache/train_f2.parquet
  $PY src/fe3.py data/train cache/train_f3.parquet
  $PY src/fe4.py data/train cache/train_f4.parquet
  $PY src/fe5.py data/train cache/train_f5.parquet
  $PY src/fe6.py data/train cache/train_f6.parquet
fi
[ -f cache/train_f8.parquet ] && [ "${REBUILD_TRAIN:-0}" != 1 ] || $PY src/fe8.py data/train cache/train_f8.parquet
$PY src/fe.py  "$TEST_DIR" cache/test_f.parquet
$PY src/fe2.py "$TEST_DIR" cache/test_f2.parquet
$PY src/fe3.py "$TEST_DIR" cache/test_f3.parquet
$PY src/fe4.py "$TEST_DIR" cache/test_f4.parquet
$PY src/fe5.py "$TEST_DIR" cache/test_f5.parquet
$PY src/fe6.py "$TEST_DIR" cache/test_f6.parquet
$PY src/fe8.py "$TEST_DIR" cache/test_f8.parquet
if [ "${V10:-0}" = "1" ]; then
  # v10: v7x no-head pipeline (v6x features + pseudo-labelling) + the prototype's c_/m_/acs_/acr_ features (KS<0.1),
  # then 0.7 ours + 0.3 prototype stage-1 blend and prototype cluster-score propagation (prior_work/prototype_agent, MIT, see PRIOR_WORK.md).
  [ -f cache/pred_test_v6xfinal.parquet ] || FINAL=0 V10=0 V6X=1 bash "$0" "$TEST_DIR" cache/submission_v6x_rebuild.csv
  mkdir -p prototype/cache cache/f11fr
  (cd prototype && for s in train test; do [ -f cache/D_$s.parquet ] || ../$PY build_design.py $s; done)
  $PY src/prototype_feats.py
  SHIFT_EXT=0 F8_GROUPS=A,B,C,D,G FEATS=f2,f3,f4,f6 FE10=1 PROTOTYPE_FR=1 POS_FRAC=0.008 $PY src/pseudo.py cache/pred_test_v6xfinal.parquet cache/lookalike.npz cache/pred_test_v10.parquet
  (cd prototype && { [ -f cache/test_stage1.npy ] && [ -f cache/oof42_stage1.npy ] || ../$PY run_prototype.py; })
  (cd prototype && OURS_OOF=../cache/pred_test_v10_oof.parquet OURS_TEST=../cache/pred_test_v10_base.parquet FIXED=${V10_FIXED:-0.7,0} OUT=../$OUT ../$PY ../src/v10_post.py write)
  exit 0
fi
if [ "${V7X2:-0}" = "1" ]; then
  # v7x2: second self-training round on the v6x features; teacher = v7x no-head test scores (cache/pred_test_v7x_base.parquet),
  # same pseudo.py settings (top 0.8% pos, bottom 60% neg, weight 0.5, cmp900 cluster exclusion). Ships the no-head score.
  [ -f cache/pred_test_v7x_base.parquet ] || FINAL=0 POS_FRAC=0.008 V7X2=0 V7X=1 NOHEAD=1 bash "$0" "$TEST_DIR" cache/submission_v7x_nohead_rebuild.csv
  SHIFT_EXT=0 F8_GROUPS=A,B,C,D,G FEATS=f2,f3,f4,f6 FE10=1 POS_FRAC=${POS_FRAC:-0.008} $PY src/pseudo.py cache/pred_test_v7x_base.parquet cache/lookalike.npz cache/pred_test_v7x2${V7X2_TAG:-}.parquet
  $PY src/make_submission.py cache/pred_test_v7x2${V7X2_TAG:-}_base.parquet "$TEST_DIR" "$OUT" cache/test
  exit 0
fi
if [ "${V7X:-0}" = "1" ]; then
  # v7x: v7 self-training (src/pseudo.py: top 0.8% pseudo-pos, bottom 60% pseudo-neg, weight 0.5, cmp900-grouped 5-fold)
  # on top of the v6x pipeline; pseudo-labels from v6x test scores. Builds v6x first if needed.
  [ -f cache/pred_test_v6xfinal.parquet ] || FINAL=0 V7X=0 V6X=1 bash "$0" "$TEST_DIR" cache/submission_v6x_rebuild.csv
  SHIFT_EXT=0 F8_GROUPS=A,B,C,D,G FEATS=f2,f3,f4,f6 FE10=1 POS_FRAC=${POS_FRAC:-0.008} $PY src/pseudo.py cache/pred_test_v6xfinal.parquet cache/lookalike.npz cache/pred_test_v7x.parquet
  if [ "${NOHEAD:-0}" = "1" ]; then P7X=cache/pred_test_v7x_base.parquet; else P7X=cache/pred_test_v7x.parquet; fi
  $PY src/make_submission.py "$P7X" "$TEST_DIR" "$OUT" cache/test
  exit 0
fi
if [ "${V6X:-0}" = "1" ]; then
  # v6x probe ("everything additive"): v6 exactly + BlazingAML fe6 features (FEATS f6) + 21 fe10 look-alike columns
  # (cache/fe10/<split>_fe10.parquet, FE10=1); look-alike head kept exactly as in v6.
  mkdir -p cache/fe10; [ -f cache/fe10/test_fe10.parquet ] && [ "${REBUILD_FE10:-0}" != 1 ] || $PY src/fe10_medium.py build
  SHIFT_EXT=0 F8_GROUPS=A,B,C,D,G FEATS=f2,f3,f4,f6 FE10=1 OOF_OUT=cache/oof_v6x.parquet $PY src/final.py cache/test "$TEST_DIR" cache/pred_test_v6x.parquet
  [ -f cache/lookalike.npz ] || FEATS=f2,f3,f4 FE10=0 $PY src/lookalike.py cache/lookalike.npz 42 7 2026 cache/test
  $PY src/v6.py cache/pred_test_v6x.parquet cache/lookalike.npz cache/pred_test_v6xfinal.parquet
  $PY src/make_submission.py cache/pred_test_v6xfinal.parquet "$TEST_DIR" "$OUT" cache/test
  exit 0
fi
if [ "${V6:-1}" = "1" ]; then
  # v6 (default): v3 pipeline + fe8 groups A,B,C,D,G (owner's layering-chain / gather / pair-cadence / row / IsolationForest)
  # + look-alike head (positives vs legit look-alike scenarios) blended at 0.2 rank weight.
  SHIFT_EXT=0 F8_GROUPS=A,B,C,D,G $PY src/final.py cache/test "$TEST_DIR" cache/pred_test_v6.parquet
  $PY src/lookalike.py cache/lookalike.npz 42 7 2026 cache/test
  $PY src/v6.py cache/pred_test_v6.parquet cache/lookalike.npz cache/pred_test_v6final.parquet
  if [ "${PSEUDO:-0}" = "1" ]; then
    # v7: transductive self-training on the test week (pseudo-labels from the v6 scores; POS_FRAC=0.008 default,
    # POS_FRAC=0.015 gives v7b). pseudo.py reads data/test + cache/test.
    SHIFT_EXT=0 F8_GROUPS=A,B,C,D,G POS_FRAC=${POS_FRAC:-0.008} $PY src/pseudo.py cache/pred_test_v6final.parquet cache/lookalike.npz cache/pred_test_v7.parquet
    $PY src/make_submission.py cache/pred_test_v7.parquet "$TEST_DIR" "$OUT" cache/test
    exit 0
  fi
  $PY src/make_submission.py cache/pred_test_v6final.parquet "$TEST_DIR" "$OUT" cache/test
  exit 0
fi
$PY src/final.py cache/test "$TEST_DIR" cache/pred_test.parquet
# v4: level-3 stacker (medium/hard specialist + percentile-ranked component-smoothed v3 scores)
# v5 default (STACK=0): ship the shift-robust v3b blend directly (stacker did not beat it in grouped CV).
# STACK=1 reproduces submission_v4 (level-3 stacker).
if [ "${STACK:-0}" = "1" ]; then
  $PY src/v4.py cache/train cache/test cache/pred_test.parquet cache/pred_test_v4.parquet
  $PY src/make_submission.py cache/pred_test_v4.parquet "$TEST_DIR" "$OUT" cache/test
else
  $PY src/make_submission.py cache/pred_test.parquet "$TEST_DIR" "$OUT" cache/test
fi
