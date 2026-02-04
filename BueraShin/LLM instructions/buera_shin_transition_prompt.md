# Prompt for another LLM: Generate code for Buera–Shin transition path with idiosyncratic distortions

You are an expert quantitative macroeconomist **and** a senior Python engineer.  
Generate **working, runnable Python code** that computes:

1) A **pre-reform stationary equilibrium** with financial frictions **λ = 1.35** and **idiosyncratic output wedges** (distortions).  
2) A **post-reform stationary equilibrium** with the **same λ = 1.35** but **no idiosyncratic distortions** (τ ≡ 0).  
3) A **perfect-foresight transition path** from the pre-reform stationary distribution to the post-reform steady state after an **unexpected, once-and-for-all reform at t = 0** that removes τ forever (λ fixed).

Your code should be clear, modular, and reasonably fast (vectorization and/or Numba is encouraged but not mandatory).  
Do **not** assume any existing user code; produce a complete self-contained implementation.

---

## 0) Deliverables

Produce a single Python script (or a small module + main script) that:

- Solves **stationary GE** with:
  - **distortions ON** (pre-reform)
  - **distortions OFF** (post-reform)
- Solves a **transition** using a shooting / time-path-iteration algorithm.
- Prints a results table and saves plots of key aggregates over time.
- Provides a CLI entry point like:

```bash
python transition_buera_shin.py --T 250 --na 600 --ne 40 --method tpi
```

Save outputs to an `outputs/` folder (plots + CSV).

---

## 1) Model: states, shocks, and wedges

### 1.1 Preferences and savings
- Agents have CRRA utility u(c) = c^(1-σ)/(1-σ), σ > 0.
- Discount factor β.
- Assets/wealth a ≥ A_MIN with upper bound A_MAX (large).

### 1.2 Occupational choice (worker vs entrepreneur)
At each date, individual with (a,e,τ) chooses between:
- Worker: earns wage w (no wedge on wages).
- Entrepreneur: chooses static inputs (k,ℓ) subject to collateral constraint k ≤ λ a.

### 1.3 Production technology (span of control)
Let ν denote the entrepreneur share parameter (so span = 1 − ν).  
Given ability e and inputs k,ℓ:
- Output: y = e * (k^α * ℓ^(1-α))^(1-ν)

Static profits **without distortions**:
- π = y − wℓ − (r+δ)k, subject to k ≤ λa.

### 1.4 Idiosyncratic distortions (output wedges)
Pre-reform, entrepreneur i faces an idiosyncratic wedge τ_{y,i} that scales output in the profit function:

- π = (1 − τ_y) * e * (k^α * ℓ^(1-α))^(1-ν) − wℓ − (r+δ)k,  subject to k ≤ λa.

Key interpretation: **τ_y can be positive (tax) or negative (subsidy)**.

#### Two-point distribution and correlation with ability
τ_y takes only two values:
- τ_plus (≥ 0)
- τ_minus (≤ 0)

Conditional on ability e, probability of being “taxed”:
- Pr(τ_y = τ_plus | e) = 1 − exp(−q * e)

Thus higher e => higher chance of τ_plus when q > 0.

Use these benchmark parameters (make them CLI overridable):
- λ = 1.35
- τ_plus = 0.57   (if you prefer 0.50, allow user to set)
- τ_minus = −0.15
- q = 1.55

### 1.5 Ability process and joint (e,τ) persistence
Ability e follows a “persistence with redraw” law:
- With prob ψ: e' = e
- With prob 1−ψ: redraw e' from a Pareto distribution discretized on a grid.

**Important:** τ is redrawn **exactly when** e is redrawn.  
So:
- With prob ψ: (e',τ') = (e,τ)
- With prob 1−ψ: draw e' from Pareto grid; then draw τ' from Pr(τ' = τ_plus | e') above.

---

## 2) What must be computed

### 2.1 Two stationary equilibria
Implement:

- `solve_stationary(distortions: bool) -> dict`
  - Returns equilibrium prices (w,r), policy functions, stationary distribution, aggregates.

Pre-reform equilibrium: distortions=True, λ=1.35  
Post-reform equilibrium: distortions=False, λ=1.35

Stationary equilibrium conditions:
- Labor market clears: L_d(w,r) = L_s(w,r) where labor supply comes from workers (mass of workers = 1 − entrepreneur_share).
- Capital market clears: K_d(w,r) = A_s(w,r) where A_s is aggregate assets (household savings) (or implement a zero net supply for “external finance” as in your accounting; be explicit).

### 2.2 Transition path after reform
At t=0:
- Distortions are removed forever: τ ≡ 0 for all agents, all future periods.
- Financial friction λ stays at 1.35.
- Initial distribution μ_0(a,e) equals the **marginal** of the pre-reform stationary distribution over τ:
  - μ_0(a,e) = μ_pre(a,e,τ_plus) + μ_pre(a,e,τ_minus)

Compute a perfect-foresight path of length T (e.g., 250):
- Sequences {w_t, r_t} for t=0..T-1
- Distributions μ_t(a,e)
- Aggregates (Y_t, K_t, L_t, TFP_t, entrepreneur share, external finance/GDP)

Terminal condition:
- For t ≥ T, economy is at the post-reform steady state (w_post, r_post) and policy g_post.

---

## 3) Numerical representation choices (be explicit in code)

### 3.1 Grids
- Asset grid `a_grid` with na points, preferably log-spaced between A_MIN and A_MAX.
- Ability grid `e_grid` with ne points. Use Pareto discretization (e.g., the M-grid method used in the literature). Also store stationary redraw probabilities `p_e`.

### 3.2 Transition matrices
Construct:
- `P_e`: ne×ne matrix reflecting persistence ψ with redraw:
  - P_e[i,i] += ψ
  - P_e[i,j] += (1−ψ)*p_e[j] for all j

For distortions ON (stationary pre-reform), use two τ states:
- τ_state in {plus, minus} (index 0/1).
- Construct a joint transition for (e,τ):
  - With prob ψ: stay at (e_i, τ_s)
  - With prob 1−ψ: move to (e_j, τ_s') where:
    - Pr(τ'=plus|e_j)=1−exp(−q*e_j)
    - Pr(τ'=minus|e_j)=exp(−q*e_j)

You may implement distribution updates without forming the full giant matrix by using forward-iteration “push mass” logic.

---

## 4) Core building blocks to implement

### 4.1 Static entrepreneur problem
Write a function:

- `entrepreneur_static(a, e, tau, w, r, params) -> (profit, k_star, l_star, y)`

Solve for k*, ℓ* given the wedge and collateral constraint k ≤ λ a.
You can:
- use closed-form FOCs + min with constraint (preferred if you already know the algebra), OR
- do a fast numerical optimization (but keep it stable).

**Key tip:** wedge enters as scaling of e: `e_eff = (1 - tau) * e`.  
Then use the same FOC formulas as no-wedge case with e replaced by e_eff.

Entrepreneur chooses occupation if profit > w (consistent with your earlier implementation).

### 4.2 Household savings policy (time iteration / Euler inversion)
Implement policies for next-period assets a' = g(a,e[,tau], t).

There are two regimes:

#### Post-reform and in-transition (τ ≡ 0):
State is (a,e). Policy is g_t(a,e).

#### Pre-reform stationary with distortions:
State is (a,e,τ). To avoid a 3D approximation, implement **two policies**:
- g_plus(a,e) when τ=τ_plus
- g_minus(a,e) when τ=τ_minus

Euler equation:
- u'(c_t) = β(1+r_t) * E_t[u'(c_{t+1})]
with c_t = income_t − a' and income depends on occupation and (if distortions ON) current τ.

Enforce feasibility:
- c_t > 0
- a' ∈ [A_MIN, A_MAX]
- and a' ≤ income_t − tiny_eps

For the expectation, use:
- persistence branch with prob ψ (same e, same τ when distortions ON)
- redraw branch with prob 1−ψ using p_e and (if distortions ON) Pr(τ'|e').

### 4.3 Stationary distribution computation
Given a stationary policy (or pair of policies for τ states), compute the stationary distribution via:
- power iteration until convergence, with normalization each step.

Represent:
- no distortions: μ(a,e) shape (na,ne)
- distortions: μ_plus(a,e), μ_minus(a,e) each shape (na,ne)

Distribution update step:
- For each current (a_i,e_j[,τ_s]) with mass μ:
  1) compute a' = g_s(a_i,e_j)
  2) split mass to neighboring asset grid points via linear interpolation weights
  3) split across e' (and τ') using transition law.

Stop when max abs difference < tol (e.g. 1e-10).

### 4.4 Aggregates
Given μ and prices:
Compute:
- entrepreneur share (mass with profit > w)
- aggregate capital demand K = ∫ k*(state) dμ
- aggregate labor demand L = ∫ ℓ*(state) dμ
- output Y = ∫ y(state) dμ
- aggregate assets A = ∫ a dμ
- external finance = ∫ max(0, k* − a) dμ
- TFP measure consistent with span-of-control aggregator (document your formula)

---

## 5) General equilibrium solver (stationary)

Implement a robust GE loop:
- guesses (w,r)
- solve policy (with/without distortions)
- compute stationary distribution
- compute residuals:
  - ED_L = L_d − L_s  (L_s = 1 − entrepreneur_share)
  - ED_K = K_d − A   (or K − A)
- update prices with damping or use a root finder (hybrid is fine).
Return when |ED_L|+|ED_K| < tol.

---

## 6) Transition path algorithm (shooting / time-path iteration)

Implement **Time Path Iteration (TPI)** with multiple shooting flavor.

### 6.1 Unknown price path
Let x = {w_t, r_t}_{t=0..T-1}.  
Initialize with an exponential interpolation from pre to post:

- w_t^0 = w_post + (w_pre − w_post) * exp(−κ t)
- r_t^0 = r_post + (r_pre − r_post) * exp(−κ t)

Choose κ ~ 0.03–0.10 (CLI param).

### 6.2 Backward step: compute policy functions along the path
Given {w_t,r_t} and terminal policy g_post at t=T:

For t = T−1 down to 0:
- solve for g_t(a,e) from Euler equation using continuation g_{t+1} and prices (w_{t+1}, r_{t+1}).
- This is a deterministic perfect-foresight dynamic programming problem with stochastic e' (but τ=0).

Store policies g_t on the grid (na×ne).

### 6.3 Forward step: update distribution and compute aggregates
Start from μ_0(a,e) (marginal from pre-reform stationary with distortions).

For each t=0..T−1:
- compute aggregates at t using μ_t and prices (w_t,r_t)
- update μ_{t+1} from μ_t using policy g_t and ability transition P_e.

### 6.4 Residuals and price updating (the “shooting” part)
Compute per-period market clearing residuals:
- ED_L[t] = L_d[t] − (1 − entrepreneur_share[t])
- ED_K[t] = K[t] − A[t]

Update the price path with damping:
- w_t <- w_t * (1 + η_w * ED_L[t])   (or w_t + η_w*ED_L[t])
- r_t <- r_t + η_r * ED_K[t]

Then smooth the updated path slightly (optional but often stabilizing):
- w <- (1−s)*w + s*moving_average(w, window=5)
- same for r

Finally damp the whole step:
- x <- (1−θ)*x_old + θ*x_new

Iterate until max_t (|ED_L[t]| + |ED_K[t]|) < tol_transition.

### 6.5 Alternative: Broyden (optional)
Optionally implement Broyden for faster convergence, but only if stable.  
If you do, still keep damping/line search.

---

## 7) Output requirements

Save:
- `outputs/stationary_pre.json`, `outputs/stationary_post.json` (prices + key moments)
- `outputs/transition.csv` with columns:
  - t, w, r, Y, K, L, A, TFP, ext_fin, ext_fin_Y, entrepreneur_share
- Plots:
  - w_t, r_t, Y_t, TFP_t, K/Y, ext_fin/Y, entrepreneur share
  - normalize Y and TFP by post-reform steady state (or by pre-reform—be explicit)

---

## 8) Sanity checks (must include in code)

1) Stationary equilibria should satisfy market clearing to tolerance.  
2) Transition should converge: last ~20 periods aggregates close to post steady state.  
3) Removing distortions should (typically) weakly raise TFP and output (qualitative check).

Print a short summary with:
- pre vs post: Y, TFP, K/Y, ext_fin/Y, r, w, entrepreneur share
- transition: Y_T / Y_post, TFP_T / TFP_post at final period.

---

## 9) Implementation notes and constraints

- Use only standard Python scientific stack: numpy, scipy, numba (optional), matplotlib.
- Keep runtime reasonable; avoid O(na^2 * ne^2 * T) operations.
- Do not create giant dense matrices of size (na*ne)×(na*ne) unless you use sparse.
- Enforce feasibility: consumption positive, a' bounded, and never exceed cash-on-hand.

---

## 10) Suggested file structure (recommended)

- `transition_buera_shin.py` (main entry)
- optional: `model_blocks.py` with functions:
  - grids, transitions
  - entrepreneur static
  - policy solver (stationary, transition backward)
  - distribution forward
  - GE solver
  - TPI solver

---

## 11) Parameters (defaults)

Use these as defaults but allow CLI override:

- σ = 1.5
- β = 0.904
- α = 0.33
- ν = 0.21
- δ = 0.06
- ψ = 0.894
- Pareto tail for e: η = 4.15
- λ = 1.35
- τ_plus = 0.57
- τ_minus = −0.15
- q = 1.55
- asset bounds: A_MIN = 1e-6, A_MAX = 4000
- grids: na = 600, ne = 40
- transition horizon: T = 250

Document these in the script header.

---

## 12) What to return in the final answer (as the LLM)

Return:
1) The complete Python code (one file is fine).
2) Brief instructions to run it.
3) Notes on where outputs are saved.

**Do not** provide placeholder “TODO” logic for the core steps (policy, distribution, GE, TPI).  
The code must run end-to-end.
