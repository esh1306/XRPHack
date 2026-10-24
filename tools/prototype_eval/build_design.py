import sys, time, pandas as pd
sys.path.insert(0, "agent")
from app.train_e import design
name = sys.argv[1]
t0=time.time()
sp, D = design(f"../data/{name}", name)
D.to_parquet(f"cache/D_{name}.parquet")
sp.edges.to_parquet(f"cache/edges_{name}.parquet")
print(name, D.shape, time.time()-t0, flush=True)
