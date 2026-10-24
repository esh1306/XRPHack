"""Apply v8 tiered re-rank (fold-42 specialist OOF) on the v7x base OOF."""
import sys, numpy as np, pandas as pd; sys.path.insert(0, "src")
from pipeline import AP, rank01
h = pd.read_parquet("cache/train_f.parquet", columns=["TRANSACTION_HASH"]).TRANSACTION_HASH
lab = pd.read_csv("data/train/labels.csv").set_index("TRANSACTION_HASH").loc[h]
y = lab.is_suspicious.to_numpy(); diff = lab.difficulty.to_numpy(); spec = np.load("cache/v8_spec_oof_42.npy")
base = pd.read_parquet("cache/pred_test_v7x_oof.parquet").set_index("TRANSACTION_HASH").oof.loc[h].to_numpy()
def met(s): return np.round([100*AP(y,s)]+[100*AP(y[(diff==d)|(y==0)],s[(diff==d)|(y==0)]) for d in ("easy","medium","hard")],2).tolist()
rb = rank01(base); b0 = met(base); print("v7x baseline", b0)
for K in (0.015,0.02,0.025,0.03,0.04):
    for a in (0,1,2):
        v = met(np.where(rb > 1-K, 2+rb, rank01(rank01(spec)*rb**a)*.999)); print("K",K,"a",a,v,"d_all",round(v[0]-b0[0],2))
    v = met(np.where(rb > 1-K, 2+rb, rank01(.5*rb+.5*rank01(spec))*.999)); print("K",K,"soft",v,"d_all",round(v[0]-b0[0],2))
