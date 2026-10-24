#!/usr/bin/env bash
# Usage: ./run.sh [TEST_DIR] [OUT_CSV]   (v9: V9=1 ./run.sh ; v9b: V9B=1 ./run.sh ; v7x no-head: V7X=1 NOHEAD=1)
# Weight variants: BLEND_OURS=0.6|0.5|0.4 V9B=1 ./run.sh  (v9c_w40/w50/w60 = prototype weight 0.4/0.5/0.6)
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
if [ "${V9:-0}" = "1" ] || [ "${V9B:-0}" = "1" ]; then
  # v9: v7x no-head base + closed-cluster score propagation (prior_work/prototype_agent, MIT, see PRIOR_WORK.md).
  # v9b (V9B=1): 0.7*rank(v7x no-head) + 0.3*rank(prototype stage-1 ensemble), then propagation.
  [ -f cache/pred_test_v7x_base.parquet ] || V9=0 V9B=0 V7X=1 NOHEAD=1 bash "$0" "$TEST_DIR" cache/submission_v7x_nohead_rebuild.csv
  if [ "${V9B:-0}" = "1" ]; then
    $PY src/v9_prop.py cache/pred_test_v7x_base.parquet "$TEST_DIR" cache/pred_test_v9b.parquet --blend-prototype ${BLEND_OURS:-0.7}
    $PY src/make_submission.py cache/pred_test_v9b.parquet "$TEST_DIR" "$OUT"
  else
    $PY src/v9_prop.py cache/pred_test_v7x_base.parquet "$TEST_DIR" cache/pred_test_v9.parquet
    $PY src/make_submission.py cache/pred_test_v9.parquet "$TEST_DIR" "$OUT"
  fi
  exit 0
fi
if [ "${V7X:-0}" = "1" ]; then
  # v7x: v7 self-training (src/pseudo.py: top 0.8% pseudo-pos, bottom 60% pseudo-neg, weight 0.5, cmp900-grouped 5-fold)
  # on top of the v6x pipeline; pseudo-labels from v6x test scores. Builds v6x first if needed.
  [ -f cache/pred_test_v6xfinal.parquet ] || V7X=0 V6X=1 bash "$0" "$TEST_DIR" cache/submission_v6x_rebuild.csv
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
