"""Second view: everything pooled (raw OOF -> global rank), no fold-ranking."""
from common import *
tr, te, y, folds, cols = train_setup()
L = lambda n: rank01(pd.read_parquet(f"{MEM}/{n}_oof.parquet").set_index("TRANSACTION_HASH").loc[tr.TRANSACTION_HASH, "oof"].to_numpy())
s1 = np.load(f"{ROOT}/cache/pred_test_v3_s1_oofs.npy")
noG = np.mean([rank01(s1[i]) for i in range(4)], 0)
G = np.mean([L(n) for n in ["garg_v3feat_lgb", "garg_v3feat_lgb1", "garg_v3feat_lgb2", "garg_v3feat_xgb42"]], 0)
v3 = rank01(v3_oof(tr.TRANSACTION_HASH)); gn = L("gnn_s3")  # gnn_s3 OOF is already fold-ranked per seed (stored that way)
gn1 = L("gnn")
print("pooled: v3", report(tr, y, v3), "| noGARG 4-model mean", report(tr, y, noG), "| GARG 4-model mean", report(tr, y, G))
for wv, wg, wn in [(1,0,0),(0.7,0.3,0),(0.5,0.5,0),(0.3,0.6,0.1),(0.1,0.8,0.1),(0.8,0,0.2),(0.9,0,0.1),(0.6,0.3,0.1),(0.5,0.4,0.1)]:
    print("v3/GARGens/gnn_s3", wv, wg, wn, report(tr, y, wv * v3 + wg * G + wn * gn), "| with gnn(raw pooled)", report(tr, y, wv * v3 + wg * G + wn * gn1))
