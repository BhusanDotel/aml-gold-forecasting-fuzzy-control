# 7078CEM Advanced Machine Learning — Implementation Guide

**Title:** Probabilistic Machine Learning Approaches for Gold Market Forecasting and Financial Text Mining  
**Module:** 7078CEM Advanced Machine Learning, March Intake 2024  
**Dataset:** XAUUSD M15 (100,000 bars of Gold vs USD, 15-minute OHLC + Volume)

---

## Directory Structure

```
AML/
├── XAUUSD_M15.csv              Raw dataset (tab-separated, 100 k rows)
├── IMPLEMENTATION.md           This file
├── assets/                     All generated graphs and plots
│   ├── task1_gp_regression.png
│   ├── task1_gp_classification.png
│   ├── task1_bayesian_network.png
│   ├── task1_lda_text_mining.png
│   ├── task1_lda_topic_words.png
│   ├── task2_flc_membership_functions.png
│   ├── task2_flc_control_surface.png
│   ├── task2_flc_scenario.png
│   ├── task2_ga_optimisation.png
│   └── task2_cec2005_benchmark.png
└── ImplementationCode/
    ├── Task1/
    │   ├── data_preprocessing.py
    │   ├── gp_regression_classification.py
    │   ├── bayesian_network.py
    │   ├── lda_text_mining.py
    │   └── run_task1.py
    └── Task2/
        ├── flc_design.py
        ├── ga_optimization.py
        ├── cec2005_benchmark.py
        └── run_task2.py
```

---

## How to Run

```bash
# Task 1 — from ImplementationCode/Task1/
python3 run_task1.py

# Task 2 — from ImplementationCode/Task2/
python3 run_task2.py
```

Both runners print results to stdout and save all plots to `assets/`.

### Dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn scipy \
            pgmpy scikit-fuzzy networkx nltk
```

---

## Task 1 — Machine Learning for Gold Market Forecasting

### Overview

Three probabilistic ML methods applied to the XAUUSD M15 dataset to predict gold price movements and mine financial text for latent topics.

---

### 1.1 Data Preprocessing (`data_preprocessing.py`)

**What it does:** Loads the raw OHLC CSV (100 k rows), computes 10 technical indicators, and prepares two data views — a feature matrix for supervised learning and a discretised table for Bayesian network modelling.

**Key computed features:**

| Feature | Description |
|---------|-------------|
| RSI | Relative Strength Index over 14 bars |
| SMA_Ratio | Close / SMA-14 − 1 (momentum signal) |
| HL_Range | (High − Low) / Close (relative bar range) |
| Volume_Norm | Z-score of volume over 14-bar window |
| ATR | Average True Range over 14 bars |
| MACD | EMA-12 − EMA-26 |
| Session | Asian / European / American (from bar hour) |

**Target variables:**
- Regression: next-bar log-return (`log(Close_t+1 / Close_t)`)
- Classification: sign of next-bar return (1 = up, 0 = down)

---

### 1.2 Gaussian Process Regression (`gp_regression_classification.py`)

**Task requirement:** ≥ 4 input variables, 1 output variable.

**Inputs (5):** RSI, SMA_Ratio, HL_Range, Volume_Norm, ATR  
**Output:** Next-bar log-return

**How it works:**

A Gaussian Process Regressor places a prior over functions and uses a Matérn ν=2.5 kernel (smooth but not infinitely differentiable — appropriate for financial data). The kernel is:

```
k(x, x') = C(1.0) × Matérn(ν=2.5) + WhiteKernel(noise)
```

After fitting on 300 training samples (subsampled for tractability), the model returns both a **predicted mean** and a **predictive standard deviation** for each test point. The ±2σ interval forms a 95% credible interval.

**Why 300 samples?** Full GPR scales O(n³) in training time. 300 samples gives a tractable fit while capturing the key covariance structure.

**Outputs produced:**
- Prediction vs actual with 95% CI ribbon
- Scatter plot of predicted vs actual returns
- Distribution of predictive uncertainty (σ)
- Residual plot

---

### 1.3 Gaussian Process Classification (`gp_regression_classification.py`)

**Task requirement:** Classification via threshold on output variable.

**How the threshold works:** The continuous log-return target is thresholded at 0:
- `return > 0` → Class 1 (price goes up)
- `return ≤ 0` → Class 0 (price goes down)

A `GaussianProcessClassifier` with an RBF kernel is then trained on the binary labels using the Laplace approximation for the posterior.

**Output:** Posterior predictive probability P(Up | features). The decision boundary is 0.5.

---

### 1.4 Bayesian Network (`bayesian_network.py`)

**Task requirement:** ≥ 8 random variables.

**Variables (10 total):**

| Variable | Type | States | Meaning |
|----------|------|--------|---------|
| Open_cat | Discrete | 0/1/2 | Open price below / at / above rolling mean |
| High_cat | Discrete | 0/1/2 | High price level |
| Low_cat | Discrete | 0/1/2 | Low price level |
| Close_cat | Discrete | 0/1/2 | Close price level |
| Volume_cat | Discrete | 0/1/2 | Volume low / mid / high |
| RSI_cat | Discrete | 0/1/2 | Oversold / Neutral / Overbought |
| Trend_cat | Discrete | 0/1 | Close below / above SMA-14 |
| Session | Discrete | 0/1/2 | Asian / European / American |
| Volatility_cat | Discrete | 0/1 | Low / High ATR |
| Direction | Discrete | 0/1 | Next bar Down / Up |

**Graph structure (domain-knowledge DAG, 14 edges):**

```
Session → RSI_cat, Volume_cat, Volatility_cat
Open_cat → High_cat, Low_cat, Close_cat
High_cat, Low_cat → Close_cat
Close_cat → RSI_cat, Trend_cat
Trend_cat, RSI_cat, Volume_cat, Volatility_cat → Direction
```

**Fitting:** Bayesian parameter estimation (Dirichlet pseudo-counts) via `pgmpy`'s `DiscreteBayesianEstimator`.

**Inference:** Variable Elimination to compute exact marginal P(Direction | evidence).

---

### 1.5 Latent Dirichlet Allocation (`lda_text_mining.py`)

**Task requirement:** Topic modelling — no restriction on application.

**Corpus:** 300 synthetic financial news documents (60 per ground-truth topic) covering gold market themes.

**5 Topics discovered:**

| Topic | Theme | Key Words |
|-------|-------|-----------|
| T1 | Geopolitical Risk / Safe-Haven | conflict, tensions, safe-haven, bullion, crisis |
| T2 | Federal Reserve / Monetary Policy | rate, Fed, hike, cut, FOMC, yields |
| T3 | Inflation Hedge / Macroeconomics | inflation, CPI, hedge, real yields, debasement |
| T4 | Technical Analysis / Price Levels | RSI, SMA, breakout, resistance, support, Fibonacci |
| T5 | Supply, Mining & Physical Demand | mine, production, jewellery, central bank, ETF |

**How LDA works:**

Each document is modelled as a mixture of K topics. Each topic is a distribution over vocabulary words. The generative process:
1. For document d, draw topic proportions θ_d ~ Dirichlet(α)
2. For each word position, draw topic z_dn ~ Multinomial(θ_d)
3. Draw word w_dn ~ Multinomial(φ_{z_dn})

The model is fit using online variational Bayes (sklearn `LatentDirichletAllocation`).

**Outputs produced:**
- Topic–word weight heatmap
- Document–topic mixture bar chart (first 60 docs)
- t-SNE embedding of document topic distributions (coloured by ground-truth topic)
- Per-topic top-10 word horizontal bar charts

---

## Task 2 — Evolutionary and Fuzzy Systems

### Overview

A Fuzzy Logic Controller for an intelligent assistive care flat, optimised by a Genetic Algorithm, plus a CEC'2005 benchmark comparison of GA vs PSO.

---

### 2.1 Fuzzy Logic Controller (`flc_design.py`)

**Model:** Mamdani Fuzzy Inference System  
**Defuzzification:** Centroid (Centre of Gravity)

#### Controller 1 — HVAC Temperature Control

**Inputs:**

| Input | Range | Membership Functions |
|-------|-------|---------------------|
| Temperature | 0 – 40 °C | cold, cool, comfortable, warm, hot |
| Humidity | 0 – 100 % | dry, normal, humid |
| Time of Day | 0 – 24 h | night, morning, afternoon, evening, late_night |

**Output:** `hvac_power` (−100 to +100), negative = cooling, positive = heating

**17 rules** encode behaviours like:
- Hot + Humid → Strong Cooling
- Cold + Normal humidity → Strong Heating
- Night + Cool → Mild Heating (supports resident sleep comfort)
- Afternoon + Warm → Mild Cooling (peak heat period)

#### Controller 2 — Lighting Control

**Inputs:**

| Input | Range | Membership Functions |
|-------|-------|---------------------|
| Ambient Light | 0 – 1000 lux | dark, dim, moderate, bright |
| Time of Day | 0 – 24 h | night, morning, afternoon, evening, late_night |

**Output:** `light_level` (0 – 100 %)

**14 rules** encode behaviours like:
- Dark + Night → Dim light (not to disturb sleep)
- Dark + Afternoon → Full light (compensate for no natural light)
- Bright → Off (natural light sufficient)

**Plots generated:**
1. All membership functions (inputs + outputs)
2. Control surface plots (2D contour)
3. 24-hour operational scenario showing FLC response to a typical daily cycle

---

### 2.2 Genetic Algorithm — FLC Optimisation (`ga_optimization.py`)

**Task requirement:** Design a GA to adjust membership function parameters.

#### Chromosome Encoding (26 real-valued genes, all in [0, 1])

| Genes | Decoded to | Count |
|-------|-----------|-------|
| 0–4 | Temperature MF centres (denorm → [0, 40] °C) | 5 |
| 5–9 | Temperature MF half-widths (→ [1.5, 9.5] °C) | 5 |
| 10–12 | Humidity MF centres (→ [0, 100] %) | 3 |
| 13–15 | Humidity MF half-widths (→ [5, 35] %) | 3 |
| 16–20 | HVAC output MF centres (→ [−100, 100]) | 5 |
| 21–25 | HVAC output MF half-widths (→ [10, 50]) | 5 |

#### Fitness Function

RMSE between FLC output and a 400-point expert-rule dataset:

```
Fitness(c) = sqrt( mean( (FLC(temp_i, hum_i; c) − target_i)^2 ) )
```

#### Genetic Operators

| Parameter | Value |
|-----------|-------|
| Population size | 40 |
| Generations | 60 |
| Crossover | Uniform (p=0.85) |
| Mutation | Gaussian noise (p=0.08 per gene, σ=0.05) |
| Selection | Tournament (k=4) |
| Elitism | 1 best individual |

**Improvement observed:** ~64% RMSE reduction (initial ~48 → optimised ~18).

#### Sugeno (TSK) Alternative

In a Mamdani model, output MFs are fuzzy sets defuzzified by centroid. For a **Sugeno model**, output MFs become crisp linear functions:

```
y_k = a0_k + a1_k × temperature + a2_k × humidity
```

GA chromosome change: remove 10 HVAC MF genes, add 5 × 3 = 15 Sugeno consequent coefficients → total 31 genes. Defuzzification is a weighted average (exact, no numerical integration), making fitness evaluation faster and GA convergence faster.

---

### 2.3 CEC'2005 Benchmark (`cec2005_benchmark.py`)

**Task requirement:** Compare ≥ 2 optimisation techniques on 2 CEC'2005 functions, D=2 and D=10, 15 runs each.

#### Functions

**F1 — Shifted Sphere** (unimodal, separable):
```
f(x) = sum((x - o)^2) − 450      global minimum = −450
```

**F6 — Shifted Rotated Ackley's** (multimodal, non-separable):
```
f(x) = −20·exp(−0.2·√(||M(x-o)||²/D)) − exp(Σcos(2π·M(x-o)_i)/D) + 20 + e − 140
global minimum = −140
```

where `o` is the shifted optimum and `M` is a rotation matrix.

#### Algorithms

**GA** — Real-valued with BLX-α crossover (α=0.5), tournament selection (k=3), Gaussian mutation (1/D per gene).

**PSO** — Standard global-best PSO. Parameters: w=0.729 (inertia), c1=c2=1.494 (cognitive + social).

#### Results Summary (mean error over 15 runs)

| Function | D | GA Mean Error | PSO Mean Error |
|----------|---|--------------|----------------|
| F1 Sphere | 2 | ≈ 0.0002 | ≈ 0.0000 |
| F1 Sphere | 10 | ≈ 1.14 | ≈ 228 |
| F6 Ackley | 2 | ≈ 0.013 | ≈ 0.0000 |
| F6 Ackley | 10 | ≈ 0.85 | ≈ 1.84 |

**Observations:**
- For D=2 both algorithms perform well; PSO slightly edges GA on smooth unimodal functions.
- For D=10, GA generalises better on both functions. PSO suffers from premature convergence on F1 at higher dimensions (large variance across runs).
- F6 (multimodal, rotated) is harder for both: GA maintains lower mean error at D=10.

**Plots produced:**
- Convergence curves (log scale, mean ± all runs) for each (function, D) combination
- Box plots of final error distributions comparing GA vs PSO

---

## Key Design Decisions

1. **GP subsetting to 300 samples:** Full GPR is O(n³); 300 samples fits within seconds while preserving covariance structure. Production deployment would use sparse GPR (inducing points).

2. **Manual BN structure vs. learned:** Domain-knowledge DAG avoids the combinatorial structure search and ensures interpretable causal ordering (Session → market behaviour → direction).

3. **Synthetic LDA corpus:** No gold-market news corpus was provided in the task. A 300-document corpus with 5 thematic groups was generated to demonstrate the full LDA pipeline. A real corpus (e.g. Reuters financial news, DBpedia) can replace `build_corpus()` in `lda_text_mining.py`.

4. **Mamdani vs Sugeno FLC:** Mamdani was chosen for interpretability (output MFs are linguistic sets, easy to justify to care workers). The GA optimisation section describes the exact chromosome change needed for Sugeno.

5. **CEC'2005 max evaluations:** Set to 10,000 × D per run (standard CEC budget), with population sizes scaled to 10 × D to balance exploration and convergence speed.
