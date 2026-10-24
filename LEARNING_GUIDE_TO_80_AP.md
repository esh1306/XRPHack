# Mastering On-Chain AML Detection: From 50 AP to 71.6 LB — and How to Reach 80+ AP

**Project:** Decode & Detect — The XRPL AI Hack (Team Solo Validator)  
**Progression Achieved:** `50.37` (CV v1) $\rightarrow$ `65.1` (LB v3) $\rightarrow$ `65.6` (LB v7x) $\rightarrow$ `69.0` (LB v9b) $\rightarrow$ `70.5` (LB v9c) $\rightarrow$ `71.3` (LB v9e) $\rightarrow$ **`71.6` (LB v13 Final)**

This guide distills everything we built, reverse-engineered, and discovered during the hackathon into a structured study guide. It covers:
1. **Real-World Use Case**: How XRPL mechanics work in practice and how real crypto-forensics teams (Ripple Compliance, Chainalysis, TRM Labs, Elliptic) detect money laundering.
2. **Dataset Anatomy & Why 71.6 Was Hard**: The exact structure of the 40,131 transactions, the 13 scenario families, and the math of Average Precision (PR-AUC).
3. **Architecture of What We Built (`v1` $\rightarrow$ `v13`)**: Every feature layer, model ensemble, self-training loop, and graph propagation pass — what worked, what failed, and why.
4. **The Roadmap from 71.6 to 80+ AP**: The exact 3 algorithmic bottlenecks holding back the remaining ~8.5 AP points, and the concrete mathematical/ML solutions to crack them.
5. **Study Curriculum & Hands-On Exercises**: Papers, algorithms, and step-by-step experiments you can run directly on this repo.

---

## 1. Real-World Use Case: AML on the XRP Ledger (XRPL)

### How Money Moves on XRPL (and Why It Is Unique for AML)
Unlike Bitcoin (UTXO graph) or Ethereum (account + smart-contract ERC-20 calls), the **XRP Ledger (XRPL)** has protocol-native primitives for issued currencies and atomic multi-hop foreign exchange:
1. **Native XRP vs. Issued Currencies (IOUs)**:
   - Native `XRP` is held directly on accounts (measured in *drops*, $1\text{ XRP} = 10^6\text{ drops}$) and is burned as the transaction fee (`SOURCE_FEE`, typically 10–15 drops) on **every** transaction.
   - Fiat/stablecoin tokens (`USD`, `EUR`, `GBP`) are **Issued Currencies (IOUs)** represented as `issuer_address:currency_code`. **Crucial domain rule**: `rIssuerA:USD` and `rIssuerB:USD` are different assets with different liquidity pools and slightly different reference exchange rates.
2. **Path Payments & Auto-Bridging (`direct_payment` vs. `path_payment`)**:
   - A single XRPL `Payment` transaction (`TRANSACTION_HASH`) can execute **1, 2, or 3 transfer legs** (`TRANSFER_INDEX = 0, 1, 2`) atomically across liquidity providers/hubs:
     - **1 leg (`direct_payment`)**: Same asset (`XRP -> XRP` or `rIssuer1:USD -> rIssuer1:USD`).
     - **2 legs (`path_payment`, direct FX conversion)**: Sender pays `rIssuer1:EUR` to a market-maker hub (Leg 0), and the market-maker delivers `rIssuer2:USD` to the Destination (Leg 1).
     - **3 legs (`path_payment`, XRP-bridged FX conversion)**: Sender pays `rIssuer1:GBP` $\rightarrow$ Hub 1 sends `XRP` $\rightarrow$ Hub 2 delivers `rIssuer2:USD` to Destination.
3. **`DELIVERED_AMOUNT` vs. `SendMax` vs. First Transfer Leg**:
   - In XRPL, `SendMax` (inside `EXTRA_TRANSACTION_FIELDS`) is only the **maximum ceiling** the sender is willing to spend, **not** what was actually debited.
   - To measure **value conservation** (whether an intermediary account passed on 95% of what it received), you must compare the **delivered USD** of the incoming payment (`DELIVERED_AMOUNT`) against the **first executed transfer leg (`TRANSFER_INDEX == 0`) USD** of the outgoing payment (`paid_usd` / `src_usd`).

### Where This System Is Used in the Real World
In production financial crime compliance (VASPs, exchanges like Revolut/Coinbase/Bitstamp, institutional DeFi, and blockchain intelligence firms):
- **Exchange Deposit/Withdrawal Screening (KYT — Know Your Transaction)**: When funds hit an exchange deposit address, compliance engines score the incoming subgraph in real time or near-real time to decide whether to freeze the deposit, file a Suspicious Activity Report (SAR), or clear it.
- **Triage Under Strict Analyst Budgets (`P@1%` / PR-AUC)**: A compliance team cannot manually review 40,000 transactions per week. They review the top 0.5%–1% (~400 transactions). If legitimate corporate treasury, payroll, or market-making ("look-alike twins") flood the top 1%, analysts waste all their time on false positives.
- **FATF Red Flag Typology Detection**: Regulators require explainable typology attribution (`reason` column): *multi-hop relay chain*, *scatter-gather (smurfing)*, *gather-scatter*, *cyclic wash trading*, or *peel chains*.

---

## 2. Anatomy of the Dataset & Why Average Precision Is Ruthless

### Exact Breakdown of the 40,131 Training Transactions
Every split covers **7 days** of activity across **700 accounts** (~110 transactions per account per week):

| Category | Families | Difficulty | Rows | Scenarios | `pair_n == 1` (One-Off Pair) | Typical Span |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Background** | `background` | `normal` (0) | `18,000` | — | `27.6%` | Full week |
| **Motif Positives (Easy)** | `bipartite`, `chain`, `cycle`, `fan_in`, `fan_out`, `fan_out_in`, `multi_stage`, `random`, `stack` | `easy` (1) | **`800`** | `105` | **`96.8%`** | **1.5 – 20 min** (med `5.3m`) |
| **Motif Positives (Medium)** | Same 9 motif families | `medium` (1) | **`225`** | `27` | **`95.1%`** | **40 – 490 min** (med `110m`) |
| **Motif Decoy Twins (Neg)** | Same 9 motif families | `normal` (0) | **`1,081`** | `140` | **`96.3%`** | **10 – 300 min** (med `61.7m`) |
| **Relay Positives (Hard)** | `aggregate_relay` (102), `distributed_relay` (98) | `hard` (1) | **`200`** | `12` | `36.9%` | **5.5 days** (3–4 episodes) |
| **Relay Decoy Twins (Neg)** | `aggregate_relay` (10,098), `distributed_relay` (9,702) | `normal` (0) | **`19,800`** | `1,188` | `36.0%` | **5.5 days** (3–4 episodes) |
| **Ambiguous Positives (Hard)** | `ambiguous` | `hard` (1) | **`25`** | `25` | `16.0%` | Single payment (unobservable) |

### The Math of Average Precision (`100 × AP`)
$$\text{AP} = \sum_{k=1}^{N} P(k) \cdot \Delta R(k) = \frac{1}{1250} \sum_{i=1}^{1250} \frac{i}{\text{rank}(p_i)}$$
where $\text{rank}(p_i)$ is the 1-indexed position of the $i$-th positive transaction in your sorted list.
- **Why the Top 2,000 Ranks Decide 90% of Your Score**:
  - Moving a positive from rank `30,000` to rank `15,000` changes its precision contribution from `i / 30,000` to `i / 15,000` (a tiny $\sim 0.02$ change).
  - Pushing 300 **Motif Normal Twins** (`is_suspicious = 0`) out of ranks `200..1,200` and replacing them with **Easy/Medium Positives** increases $i / \text{rank}(p_i)$ for **hundreds of positives simultaneously**, jumping AP by **+5 to +10 points**!
  - Conversely, if you blend an uncalibrated Hard or Medium specialist into the top 4% (`ranks 1..1,600`), any false positives it pushes into the top 800 will immediately **destroy Easy precision** and drop your score by 1–4 points (which is why unprotected blends dropped from `71.3` to `71.1`, while protecting the top 4% jumped to **`71.6`**!).

---

## 3. What We Built: Architecture of the `71.6` Pipeline

Our final pipeline (`src/v13_final.py`) combines **three distinct feature/model paradigms** plus **two graph score-propagation operators**:

```
Raw XRPL Tables (transactions.parquet, transfers.parquet, market_rates.csv)
 │
 ├──► 1. Solo Validator Pipeline A (v7x no-head: src/fe*.py -> src/final.py -> src/pseudo.py)
 │       • 450+ features: fe (intrinsic/graph), fe2 (3d/7d flows), fe3 (sessions/components),
 │         fe4 (amount-matched chains/peel/OOC), fe6 (BlazingAML motifs), fe8 (layering/gather/IsoForest),
 │         fe10 (3h stretched medium motifs)
 │       • Shift robustness: drop giant components (cmp3600) + within-split rank-pct for shifted features
 │       • Stage 1 (LGB×3 + XGB + CatBoost) -> Stage 2 Graph Stacking -> Transductive Self-Training on Test
 │
 ├──► 2. Prototype Pipeline B (prior_work/prototype_agent/app/train_e.py)
 │       • 257 domain features: AccountIndex fast window queries (features.py), closed novel-edge
 │         motifs (motif.py), whole-week sender/receiver aggregates (account_agg.py), and
 │         multi-scale closed-cluster domain features (cluster_prop.py: pass-through ratio, dt_cv)
 │       • Stage 1 Ensemble: 5-seed HistGradientBoosting (0.8) + ExtraTrees (0.1) + Quantile-LogReg (0.1)
 │
 └──► 3. Super-Matrix Specialist Pipeline C (src/fe11_hard_med.py + src/v13_final.py)
         • 983 combined features: Pipeline A + Pipeline B + cluster/relay/funding/fingerprint
           + FE11 (2-hop closed clusters up to 8h, source->sink e2e conservation, Hard-Relay lag bands)
         • 3-seed LightGBM specialist (no CatBoost, avoiding CatBoost's low recall on sparse medium motifs)
 │
 ▼
Stage-1 Rank Blend: 0.82 * rank(Proto_S1) + 0.08 * rank(v7x_nohead) + 0.10 * rank(Xs_983_LGB)
 │
 ▼
Stage-2 Multi-Scale Closed-Cluster Score Propagation (propagate_cluster_scores)
 • Over 5m, 15m, 30m, 1h, 2h, 4h closed 2-hop novel-edge clusters:
   0.35 * rank(s1) + 0.65 * rank(weighted mean / max / q75 of cluster scores)
 │
 ▼
Stage-3 Protected Top-4% Re-Ranking + Verified Relay Episode Propagation
 • Top 4% (ranks 1..1,604 = Easy + top Medium) locked in exact cluster-propagated order
 • Below top 4%: 0.75 * base + 0.15 * rank(Xs_983_LGB) + 0.10 * rank(propagate_relay_episodes)
```

### Key Lessons from the Ablation History (`SUBMISSIONS.md`)
1. **Distribution Shift Between Weeks Is Real (`v2` 64.1 LB vs. `v3` 65.1 LB)**:
   - Even in synthetic data, global graph density, FX conversion mix (`SendMax` share shifted by KS `0.095`), and giant 1-hour connected components (`cmp3600`) drift from week to week.
   - **Rule**: Always run **Adversarial Validation (`src/adv.py`)** and a **2-sample Kolmogorov–Smirnov (`ks_2samp`) check** between `train` and `test` features. Drop giant components and apply within-split percentile ranking (`df[c] = df[c].rank(pct=True)`) to any feature with $\text{KS} > 0.06$.
2. **Why Closed-Cluster Score Propagation Was the Biggest Breakthrough (`65.6` $\rightarrow$ `71.3` LB)**:
   - A per-transaction GBDT scores each edge independently. In a 10-edge `fan_out_in` or `multi_stage` scenario, 6 edges might get high scores (`0.95`) while 4 entry/exit legs look like ordinary payments (`0.40`).
   - By constructing the **closed 2-hop subgraph of novel (`pair_n == 1`) edges** within `5m, 15m, 30m, 1h, 2h, 4h` around each payment and taking `mean`, `q75`, and `max`, all 10 legs of the scenario inherit the scenario's collective score, while isolated false-positive spikes on single edges get averaged down.
3. **Why `Prototype Stage-1` Generalized Better to Test Than `v7x` (`BLEND_OURS=0.1` $\rightarrow$ `71.3` LB)**:
   - `Prototype` (`HistGradientBoosting` with `min_samples_leaf=40, l2_regularization=5.0` on compact novel-motif + closed-cluster features) had lower train CV (`61.75`) than `v7x` (`63.79`), but scored **5+ points higher on the test leaderboard** because its features (`m_closed_*`, `c_pt_err_*`, `c_dt_cv_*`) are invariant to week-level volume/FX shifts.

---

## 4. How to Reach 80+ AP: The 3 Remaining Bottlenecks & How to Crack Them

We ran an oracle simulation on the training set during our analysis:
- **Current OOF AP**: `~62.6` (`71.6` LB)
- **If we cleanly separate the Positives from the ~600 Motif Normal Twins inside the Top 2,500**: OOF AP jumps to **`80.26`** (which corresponds to **`~83–85` on the Leaderboard**!).
- **If we also lift 100 of the 200 Hard Relay positives into the Top 1,000**: OOF AP jumps to **`87.40`**.

Here are the **exact 3 technical bottlenecks** and how to solve each one to cross **80+ AP**:

---

### Bottleneck 1: Separating Motif Positives (`Easy` + `Medium`) from Motif Normal Twins (`1,081` Decoys)
**The Problem:**  
In `data/train`, there are `1,025` Motif Positives (`800` Easy + `225` Medium) and `1,081` Motif Normal Twins across the exact same 9 graph topologies (`bipartite`, `chain`, `cycle`, `fan_in`, `fan_out`, `fan_out_in`, `multi_stage`, `random`, `stack`). Both have `pair_n == 1` (`96%`) and `6–12` edges. In our `71.6` model, almost **100% of the false positives in the Top 1,500** are Motif Normal Twins!

**Why Our Current Features Only Partially Separate Them:**  
In `cluster_prop.py`, the local cluster is built by an ego-centric 1-to-2 hop neighborhood in fixed time windows (`5m, 15m, 30m, 1h, 2h, 4h`).
- For a **4-hop `chain` or `multi_stage` scenario spanning 90 minutes**, a `15m` window only sees 2 edges, while a `2h` window without 2-hop/3-hopamount-matched path following pulls in unrelated novel edges from the same accounts (`purity drops from 97% to 68%`).

**How to Solve It (Worth +5 to +8 AP):**
1. **Exact Candidate Scenario Graph Extraction (Amount-Compatible Temporal Connected Components)**:
   - Instead of fixed ego-windows, build a **directed line graph (transaction-to-transaction graph)** where edge $e_1 = (u \to v, t_1, V_1)$ connects to $e_2 = (v' \to w, t_2, V_2)$ if they share an non-hub endpoint ($\{u,v\} \cap \{v',w\} \neq \emptyset$), both are novel (`pair_n == 1`), $|t_2 - t_1| \le \Delta t_{\max}$, and their USD amounts belong to a coherent flow conservation equations.
2. **Whole-Subgraph Conservation & Cycle/Flow Solver Features**:
   - Once each candidate novel subgraph $G_c = (V_c, E_c)$ is extracted as a single object:
     - **Pure-Source to Pure-Sink Conservation**: $\left| \sum_{e \in \text{Out}(\text{Sources})} \text{paid\_usd}(e) - \sum_{e \in \text{In}(\text{Sinks})} \text{deliv\_usd}(e) \right| / \sum \text{paid\_usd}$. (On `data/train`, this error is **`0.019`** on Medium Positives vs **`0.074`** on Normal Twins!).
     - **Node-by-Node Flow Conservation Residual**: For every internal node $m \in V_c$, compute $\max_{m} | \text{InUSD}(m) - \text{OutUSD}(m) | / \text{InUSD}(m)$ and check **temporal causality** ($\min t_{\text{out}}(m) \ge \min t_{\text{in}}(m)$, i.e., no intermediary sends money before receiving it in the subgraph).
     - **Inter-Arrival Pacing Regularity ($\text{CV}(\Delta t)$)**: Laundering bots execute steps on a tighter relative clock ($\text{CV}(\Delta t) = 0.13\text{–}0.38$) than organic normal twins ($\text{CV}(\Delta t) = 0.46\text{–}0.55$).
3. **Subgraph-Level Classifier (Set Transformer / DeepSets / Subgraph GBDT)**:
   - Train a model **at the candidate-scenario level** (one row per extracted novel component $G_c$, predicting whether $G_c$ is a positive scenario vs. a normal decoy twin), and broadcast the scenario probability back to all edges $e \in E_c$.

---

### Bottleneck 2: Cracking `aggregate_relay` and `distributed_relay` (The 200 Observable Hard Cases)
**The Problem:**  
There are 12 cohorts (`6 aggregate_relay`, `6 distributed_relay`), each containing **1 positive scenario and 99 negative decoy scenarios** (`20,000` total transactions). Every scenario consists of **3 to 4 episodes** spaced `~35–75 hours` apart across the 7-day week.
- **Why Standard Graph Features Fail (`0.72` AP)**: The 700 accounts are reused! A single relay hub `M` participates in **5 to 8 different relay scenarios simultaneously** during the week. When you compute account-level or 6h-window aggregates on `M`, all 8 scenarios get mixed together.

**What We Discovered About the Generator (Verified on `data/train`):**
1. **50% of Decoys Are Reversed in Time (`strict_fwd`)**:
   - In **12 out of 12** positive scenarios (`100%`), inside every episode, all incoming payments $S \to M$ finish **before** the outgoing payments $M \to D$ begin (with a hold lag of **`3.96h to 5.67h`** in `aggregate_relay`, and **`1.68h–3.13h` + `1.83h–2.84h`** in `distributed_relay`).
   - In **exactly 50 out of 99 decoys** per cohort, the hub $M$ pays $D$ **first** and gets refilled by $S$ `4–5 hours` **later**!
2. **The `distributed_relay` Bridge Signature**:
   - In `distributed_relay` ($S \to M_1 \to M_2 \to D$), the middle link $(M_1, M_2)$ is a **repeat directed pair with `pair_n` $\in \{3, 4\}$ and minimum gap $\ge 20\text{h}$**.
3. **Episode-Level Non-Overdraft & Conservation (`ep_ratio_min`)**:
   - Among the 50 forward-ordered scenarios in each cohort, positive scenarios conserve USD in **every single episode** ($\text{OutUSD}_e / \text{InUSD}_e \in [0.60, 1.30]$, cumulative running balance never negative, total weekly ratio $\in [0.75, 0.97]$).
   - In negative decoys, random episode amounts cause huge single-episode mismatches ($\min_e \text{OutUSD}_e / \text{InUSD}_e$ has median `0.26`).

**How to Solve It Without Scenario Labels (Worth +3 to +5 AP):**
- **Bipartite Episode Matching / De-Multiplexing Algorithm**:
  1. For each hub account $M$, cluster its incoming transactions into 1-hour **In-Bursts** $I_1, I_2, \dots, I_k$ and its outgoing transactions into 1-hour **Out-Bursts** $O_1, O_2, \dots, O_m$.
  2. Build a **bipartite graph** between In-Bursts $I_a$ at time $t(I_a)$ and Out-Bursts $O_b$ at time $t(O_b)$ where:
     $$t(O_b) - t(I_a) \in [3.6\text{h}, 5.8\text{h}] \quad \text{and} \quad \frac{\text{USD}(O_b)}{\text{USD}(I_a)} \in [0.60, 1.25]$$
  3. Solve the **Minimum-Cost Bipartite Matching** (`scipy.optimize.linear_sum_assignment`) to pair each In-Burst $I_a$ with its true corresponding Out-Burst $O_b$ (minimizing $|\log(\text{USD}(O_b) / 0.90\,\text{USD}(I_a))|$).
  4. Group matched `(In-Burst, Out-Burst)` episodes across the week that share recurring `(S, M)` or `(M, D)` counterparties (`pair_n in {2, 3, 4}`). If a hub has a sequence of 3–4 matched forward episodes separated by `~30–75h` where **every** episode matches within `[0.65, 1.20]` and has no reverse-refill match, boost all transactions in those matched bursts!

---

### Bottleneck 3: End-to-End Edge-Aware Graph Neural Networks (Line-Graph / Multi-GNN)
**The Problem:**  
Tree models (`LightGBM`, `HistGBM`, `CatBoost`, `XGBoost`) require hand-crafted multi-hop features (`fe4`, `fe8`, `cluster_prop`). Even with 983 features, trees cannot natively learn arbitrary 3-hop or 4-hop message passing with continuous edge timestamps and USD conservation.

**How to Solve It (Worth +2 to +3 AP in Ensemble with GBDT):**
- **Directed Multigraph GNN with Reverse Edges, Port Numbering, and Ego-IDs (`Multi-GNN`, Egressy et al., AAAI 2024)**:
  1. **Line-Graph Transformation ($L(G)$)**: Make each **payment transaction** a node in $L(G)$, and connect transaction $e_1 = (u \to v, t_1)$ to $e_2 = (v \to w, t_2)$ if $0 \le t_2 - t_1 \le 6\text{h}$, with edge attributes $(\Delta t = t_2 - t_1, \text{ratio} = \text{paid\_usd}(e_2)/\text{deliv\_usd}(e_1), \text{is\_novel})$.
  2. Run a 3-layer **Relational / Edge-Featured Graph Attention Network (GATv2 / PNA / TransformerConv)** on $L(G)$ trained with a **Smooth AP / Pairwise Ranking Loss** (optimizing PR-AUC directly rather than binary cross-entropy).

---

## 5. Study Curriculum: Topics & Papers to Master

If you want to be able to walk into any financial crime / graph ML benchmark or production engineering role and hit **80%+ AP**, master these 5 areas in order:

### Topic 1: Temporal Graph Motifs & Financial Typologies
- **Core Concepts**: Temporal motifs ($\delta$-temporal cycles, scatter-gather, gather-scatter, peel chains), time-respecting paths, flow conservation, and ego-network closure.
- **Must-Read Papers**:
  1. **IBM AMLworld**: Altman et al., *"Realistic Synthetic Financial Transactions for Anti-Money Laundering Models"*, NeurIPS 2023 D&B. *(Defines the exact 8 canonical laundering topologies used in this hackathon).*
  2. **Multi-GNN / GFP**: Egressy et al., *"Provably Powerful Graph Neural Networks for Directed Multigraphs"*, AAAI 2024 (`arXiv:2306.11586`). *(Shows why standard GNNs fail to count cycles/scatter-gather and how Reverse Edges + Port Numbering + Ego-IDs make GNNs provably expressive for AML).*
  3. **BlazingAML**: *"BlazingAML: Fast Motif Counting for Anti-Money Laundering"* (`arXiv:2604.12241`). *(Implemented in `src/fe6.py`).*

### Topic 2: Collective Classification & Graph Score Propagation
- **Core Concepts**: Label propagation, Personalized PageRank (PPR), belief propagation, multi-scale neighborhood quantile pooling (`mean`, `q75`, `max` over closed subgraphs), and Stage-2 stacked generalization on graphs.
- **Code in This Repo to Study**:
  - `prior_work/prototype_agent/app/cluster_prop.py` (`propagate_cluster_scores` and `cluster_domain_features`)
  - `src/pipeline.py` (`propagate` function for leave-one-out account, pair, and temporal-component smoothing)

### Topic 3: Covariate Shift, Adversarial Validation & Transductive Self-Training
- **Core Concepts**: Why time-series and temporal-graph splits shift; Adversarial Validation (`ROC-AUC` of `train vs. test`); Kolmogorov–Smirnov (`ks_2samp`) feature filtering; within-split quantile/rank normalization; out-of-fold cluster-grouped pseudo-labeling.
- **Code in This Repo to Study**:
  - `src/adv.py` (Adversarial validation script)
  - `src/pseudo.py` (Cluster-consistent transductive self-training with 5-fold out-of-fold pseudo-scoring)

### Topic 4: Optimizing Ranking Metrics (Average Precision / Top-$K$ Precision)
- **Core Concepts**: Why binary log-loss (`binary_logloss`) misallocates model capacity to the bottom 95% of background transactions; **Protected Top-$K$ Cascading** (locking high-confidence easy detections at the top while re-ranking lower-confidence regions with specialized models); Rank-Averaging vs. Probability-Averaging across heterogeneous model families.
- **Code in This Repo to Study**:
  - `src/v13_final.py` and `src/final_v11.py` (Protected Top-$K$ rank blending)

### Topic 5: Combinatorial Flow De-Multiplexing (Network Flows & Bipartite Matching)
- **Core Concepts**: Minimum-cost maximum flow, Hungarian algorithm (`linear_sum_assignment`) for matching incoming deposits to outgoing withdrawals across mixer/relay hubs, and temporal FIFO/LIFO taint tracking.

---

## 6. Hands-On Practice Roadmap Using This Repo

You can run these three self-contained experiments in this repo to practice pushing CV from `62.6` toward `75–80+`:

1. **Exercise 1 — Build the Candidate-Scenario Subgraph Classifier (Target: Medium AP `13.7` $\rightarrow$ `30+`)**:
   - Open `prior_work/prototype_agent/app/cluster.py` and `src/fe11_hard_med.py`.
   - Extract connected components of `pair_n == 1` edges with `link = 1800s (30m)` and `3600s (1h)`.
   - Create a DataFrame where **each row is one connected component** (with features: `n_edges`, `n_accts`, `n_srcs`, `n_snks`, `n_mids`, `e2e_conservation_err`, `mid_conservation_err`, `causal_order_fraction`, `dt_cv`, `mean_stage1_score`, `max_stage1_score`).
   - Train a `LightGBM` / `HistGBM` classifier on the components (`is_suspicious_component`) using `StratifiedGroupKFold(5)` and multiply the component probability with the transaction score.

2. **Exercise 2 — Build the Hungarian Bipartite Burst Matcher for Hard Relays (Target: Hard AP `0.80` $\rightarrow$ `10+`)**:
   - On `data/train`, group each account's inflows and outflows into 1-hour bursts.
   - Use `scipy.optimize.linear_sum_assignment` to match each account's In-Bursts to Out-Bursts occurring `3.6h–5.8h` later.
   - Measure what fraction of the 6 positive `aggregate_relay` scenarios are recovered in the top 20 matched hubs!

3. **Exercise 3 — Reproduce the Full Pipeline End-to-End**:
   ```bash
   ./run.sh data/test outputs/submission_v13.csv
   ```
   Inspect `src/v13_final.py`, `src/fe11_hard_med.py`, and `prior_work/prototype_agent/app/cluster_prop.py` side by side with this guide.
