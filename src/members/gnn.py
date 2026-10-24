"""Multi-GNN-style edge classifier (Egressy et al., arXiv 2306.11586), plain PyTorch, CPU, full-batch.
Accounts = nodes, transactions = directed edges with engineered (v3) features as edge attributes.
Adaptations from Multi-GNN: reverse message passing (separate in/out aggregators), port numbering
(edge's rank among the node's distinct out/in counterparties, in first-seen time order), GINe-style
edge-aware messages with edge updates. Ego IDs need per-edge subgraph sampling; with full-batch training
on a ~700-node graph we omit them (the readout already concatenates src/dst embeddings with the edge).
Train: message passing over the whole train graph, loss on train-fold edges only (labels never used as input).
Test: average of the 5 fold models applied to the test graph. Seeds fixed (42)."""
import sys, time, json, numpy as np, pandas as pd, torch, torch.nn as nn
from common import *
torch.set_num_threads(2); torch.manual_seed(SEED); np.random.seed(SEED)
torch.use_deterministic_algorithms(True)

def graph(split_dir, hashes, X):
    m = meta_of(split_dir, hashes)
    acc = pd.Index(pd.unique(np.concatenate([m.ACCOUNT, m.DEST])))
    S = acc.get_indexer(m.ACCOUNT); R = acc.get_indexer(m.DEST); T = m.TIMESTAMP.to_numpy().astype(float)
    d = pd.DataFrame({"S": S, "R": R, "T": T})
    # port numbering: order in which the counterparty first appears for this node
    first_out = d.groupby(["S", "R"]).T.transform("min"); first_in = d.groupby(["R", "S"]).T.transform("min")
    d["po"] = pd.DataFrame({"S": S, "f": first_out}).groupby("S").f.rank(method="dense")
    d["pi"] = pd.DataFrame({"R": R, "f": first_in}).groupby("R").f.rank(method="dense")
    n = len(acc)
    deg = np.zeros((n, 4))
    np.add.at(deg[:, 0], S, 1); np.add.at(deg[:, 1], R, 1)
    deg[:, 2] = pd.Series(R).groupby(S).nunique().reindex(range(n)).fillna(0).to_numpy()
    deg[:, 3] = pd.Series(S).groupby(R).nunique().reindex(range(n)).fillna(0).to_numpy()
    xn = np.log1p(deg)
    port = np.log1p(d[["po", "pi"]].to_numpy())
    trank = (pd.Series(T).rank(pct=True).to_numpy())[:, None]
    E = np.hstack([X, port, trank]).astype(np.float32)
    return dict(S=torch.tensor(S), R=torch.tensor(R), xn=torch.tensor(xn, dtype=torch.float32), E=torch.tensor(E), n=n)

class GINeLayer(nn.Module):
    def __init__(s, h):
        super().__init__()
        s.ein, s.eout = nn.Linear(h, h), nn.Linear(h, h)
        s.mlp = nn.Sequential(nn.Linear(3 * h, h), nn.ReLU(), nn.Linear(h, h))
        s.emlp = nn.Sequential(nn.Linear(3 * h, h), nn.ReLU(), nn.Linear(h, h))
        s.nn_, s.ne = nn.LayerNorm(h), nn.LayerNorm(h)
    def forward(s, h, e, S, R):
        n = h.shape[0]
        m_in = torch.relu(h[S] + s.ein(e)); m_out = torch.relu(h[R] + s.eout(e))  # forward / reverse messages
        agg_in = torch.zeros_like(h).index_add_(0, R, m_in); agg_out = torch.zeros_like(h).index_add_(0, S, m_out)
        h2 = s.nn_(h + s.mlp(torch.cat([h, agg_in, agg_out], 1)))
        e2 = s.ne(e + s.emlp(torch.cat([h2[S], h2[R], e], 1)))
        return h2, e2

class Net(nn.Module):
    def __init__(s, dn, de, h=64, L=2, p=0.1):
        super().__init__()
        s.nin, s.ein = nn.Linear(dn, h), nn.Sequential(nn.Linear(de, h), nn.ReLU(), nn.Linear(h, h))
        s.layers = nn.ModuleList([GINeLayer(h) for _ in range(L)]); s.drop = nn.Dropout(p)
        s.out = nn.Sequential(nn.Linear(3 * h, h), nn.ReLU(), nn.Dropout(p), nn.Linear(h, 1))
    def forward(s, g):
        h, e = s.nin(g["xn"]), s.drop(s.ein(g["E"]))
        for l in s.layers: h, e = l(h, e, g["S"], g["R"])
        return s.out(torch.cat([h[g["S"]], h[g["R"]], e], 1)).squeeze(1)

def run(L=2, h=64, epochs=300, patience=40, lr=3e-3, tag="gnn", nseeds=1):
    t0 = time.time()
    tr, te, y, folds, cols = train_setup()
    Xa, Xb = tr[cols].copy(), te[cols].copy()
    # within-split percentile scaling (robust to the week shift), NaN -> 0.5 plus missing flag share
    Xa = Xa.rank(pct=True).fillna(0.5).to_numpy() - 0.5; Xb = Xb.rank(pct=True).fillna(0.5).to_numpy() - 0.5
    gtr = graph(f"{ROOT}/data/train", tr.TRANSACTION_HASH, Xa); gte = graph(f"{ROOT}/data/test", te.TRANSACTION_HASH, Xb)
    yt = torch.tensor(y, dtype=torch.float32); oof = np.zeros(len(y)); tep = np.zeros(len(te)); its = []
    posw = torch.tensor((y == 0).sum() / (y == 1).sum() / 4)  # partial class weighting
    for k, (a, b), sd in [(k, f, sd) for sd in range(nseeds) for k, f in enumerate(folds)]:
        torch.manual_seed(SEED + k + 100 * sd)
        net = Net(gtr["xn"].shape[1], gtr["E"].shape[1], h, L); opt = torch.optim.AdamW(net.parameters(), lr, weight_decay=1e-4)
        ai, bi = torch.tensor(a), torch.tensor(b); best, bst, bad = -1, None, 0
        for ep in range(epochs):
            net.train(); opt.zero_grad()
            z = net(gtr); loss = nn.functional.binary_cross_entropy_with_logits(z[ai], yt[ai], pos_weight=posw)
            loss.backward(); opt.step()
            net.eval()
            with torch.no_grad(): zv = net(gtr)[bi].numpy()
            ap = AP(y[b], zv)
            if ap > best: best, bst, bad, be = ap, {kk: v.clone() for kk, v in net.state_dict().items()}, 0, ep
            else: bad += 1
            if bad >= patience: break
        net.load_state_dict(bst); net.eval()
        with torch.no_grad(): oof[b] += rank01(net(gtr)[bi].numpy()) / nseeds; tep += rank01(net(gte).numpy()) / (5 * nseeds)  # fold-ranked OOF
        its.append(be); print(tag, "fold", k, round(100 * best, 2), "best_ep", be, f"{time.time()-t0:.0f}s", flush=True)
    save(tag, tr.TRANSACTION_HASH, oof, te.TRANSACTION_HASH, tep)
    res = blend_report(tr, y, oof, tag); res["best_epochs"] = its
    json.dump(res, open(f"{MEM}/{tag}_results.json", "w"), indent=1)

if __name__ == "__main__":
    a = dict(L=int(sys.argv[1]) if len(sys.argv) > 1 else 2, h=int(sys.argv[2]) if len(sys.argv) > 2 else 64,
             nseeds=int(sys.argv[4]) if len(sys.argv) > 4 else 1)
    run(**a, tag=sys.argv[3] if len(sys.argv) > 3 else "gnn")
