# Instructions for Claude Opus: Buera & Shin (2010) solver using Chebyshev projection for the asset policy

You are Claude Opus. Write **Python code** that solves the Buera & Shin (2010) “Financial Frictions and the Persistence of History” stationary equilibrium and **replicates Figure 2-style plots** (GDP/TFP vs External Finance/GDP and interest rate vs External Finance/GDP), but **replace grid-search VFI** for savings with a **projection method using Chebyshev polynomials to approximate the asset policy**.

This document specifies: (i) the economic problem, (ii) the numerical approach (Euler-equation iteration + Chebyshev projection), (iii) general-equilibrium loop, (iv) simulation/distribution, and (v) plotting targets.

---

## 0) Deliverables

Write code that:

1. **Takes a list of collateral constraints** `lambda_values = [np.inf, 2.0, 1.75, ... , 1.0]`.
2. For each `λ`, finds stationary equilibrium prices `(w, r)` such that labor and capital markets clear (within tolerance).
3. Uses **Chebyshev projection** to compute the **asset policy** `a'(a, z)` (or equivalently savings policy) *instead of* value function iteration over a grid.
4. Computes stationary aggregates from a **large simulation** (Appendix B.1 style): `K, L, Y, A, extfin, share_entre`.
5. Produces plots:
   - Main “Figure 2” panel: **GDP and TFP normalized to λ=∞** vs **External Finance/GDP**.
   - Interest rate vs External Finance/GDP.
   - Diagnostics (optional but recommended): ExtFin/GDP vs λ, GDP/TFP vs λ, wage vs ExtFin/GDP, entrepreneur share vs ExtFin/GDP.
6. Saves plots to disk and prints a summary table like the baseline script.

**Implementation language:** Python 3.  
**Allowed libs:** `numpy`, `scipy` (optimize), `numba` optional, `matplotlib`.  
**No external data required.**

---

## 1) Model (stationary environment)

### 1.1 Preferences
CRRA utility:
- `u(c) = (c^(1-σ) - 1)/(1-σ)` if `σ != 1`, else `log(c)`
- marginal utility: `u'(c) = c^(-σ)`

Parameters used in the baseline replication:
- `SIGMA = 1.5`
- `BETA = 0.904`

### 1.2 Technology (span-of-control entrepreneurship)
Production at the establishment level (same as baseline code):
- `span = 1 - NU` with `NU = 0.21`
- output: `y = z * (k^α * l^(1-α))^span`
- rental cost: `(r + δ)k`, wage cost: `w l`
- `ALPHA = 0.33`, `DELTA = 0.06`

Entrepreneur chooses `(k, l)` subject to collateral:
- `k <= λ a` where `a` is the agent’s wealth/assets at the start of period.

Compute unconstrained `k1` from FOCs (as in baseline) and set:
- `k* = min(k1, λ a)`
Then compute `l*`, output, profit:
- `π(a,z; w,r,λ) = y - w l* - (r+δ) k*`

**Occupational choice:**
- Worker income: `w`
- Entrepreneur income: `π`
Agent chooses entrepreneur if `π > w` (ties can go either way).

### 1.3 State process for ability z
Ability is discrete with probabilities `prob_z` over grid `z_grid` (40 points, can reuse baseline hard-coded discretization).

Transition (as in baseline):
- With prob `ψ` (`PSI = 0.894`): `z' = z`
- With prob `1-ψ`: `z' ~ prob_z` iid draw

So:
`E[g(z') | z] = ψ g(z) + (1-ψ) * Σ_j prob_z[j] g(z_j)`

### 1.4 Budget / asset accumulation
Savings is risk-free with gross return `1+r`.
Define “income” including asset return **after occupational choice**:
- If worker: `income(a,z) = w + (1+r) a`
- If entrepreneur: `income(a,z) = π(a,z) + (1+r) a`

Assets next period satisfy:
- `a' ∈ [a_min, a_max]`
- `c = income(a,z) - a'` with `c > 0`

### 1.5 Euler equation (policy characterization)
For interior solutions:
`u'(c) = β (1+r) E[ u'(c') | z ]`

Where:
- `c = income(a,z) - a'(a,z)`
- `c' = income(a', z') - a'(a', z')` (policy evaluated at next state)

The expectation uses the transition above:
`E[u'(c') | z] = ψ u'(c'(a',z)) + (1-ψ) * Σ_j prob_z[j] u'(c'(a', z_j))`

---

## 2) Numerical strategy: Chebyshev projection for `a'(a,z)`

### 2.1 What to approximate
Approximate the **asset policy function** `a'(a,z)` using Chebyshev polynomials in **a** for each discrete ability state `z`.

Recommended structure:
- For each `z_j`, approximate `a'(a, z_j)` on `a ∈ [a_min, a_max]` using a Chebyshev expansion:
  `a'(a,z_j) ≈ Σ_{m=0}^{P-1} γ_{m,j} T_m( x(a) )`
  where `x(a)` maps `[a_min, a_max] → [-1,1]`.

You will store coefficients as `gamma[:, j]` with shape `(P, n_z)`.

### 2.2 Domain transform
Assets may span several orders of magnitude (baseline used `a_max=4000`). Chebyshev on a wide linear domain can be stiff. Use one of:

**Option A (simplest): linear mapping**
- `x = 2*(a - a_min)/(a_max - a_min) - 1`

**Option B (recommended): log-mapping**
- Work in `ã = log(a + a_shift)` to compress scale.
- Choose `a_shift = 1.0` or `1e-6` to avoid log(0).
- Map `ã ∈ [log(a_min+a_shift), log(a_max+a_shift)]` to `[-1,1]`.
- Policy still returns `a'` in levels.

State clearly in code which mapping you use.

### 2.3 Collocation nodes
Pick `N` Chebyshev nodes for the asset dimension (`N >= P`, usually `N=P`):
- `x_i = cos( (2i-1)π / (2N) ), i=1..N`
- Map `x_i` to `a_i` (or to `ã_i` then to `a_i`)

You will enforce Euler residuals at `(a_i, z_j)` for all i,j.

### 2.4 Solving for the policy: Euler-equation iteration + projection (recommended)
Do **not** solve a giant nonlinear system for all coefficients at once. Instead implement **time iteration / policy function iteration**:

**Outer fixed-point on the policy:**
1. Given current coefficients `gamma_old` defining `a'_old(a,z)`:
2. For each collocation node `(a_i, z_j)`, compute a new optimal `a'_new(i,j)` by solving the **Euler equation in 1D** for `a'`:
   - Define residual function:
     `F(a') = u'( income(a_i,z_j) - a' ) - β(1+r) * E[ u'( income(a',z') - a'_old(a',z') ) | z_j ]`
   - Find `a'` such that `F(a') = 0` on feasible interval.
3. After computing `a'_new(i,j)` for all i,j, fit Chebyshev coefficients `gamma_new[:, j]` for each `z_j` by interpolating the node values.
4. Dampen update: `gamma = (1-η)*gamma_old + η*gamma_new` with `η ∈ (0,1]` (e.g. 0.3–0.8).
5. Iterate until `max |a'_new - a'_old|` (or coefficient diff) is below tolerance.

This approach is robust and scales.

### 2.5 Details for the 1D Euler root solve at a node
At node `(a,z)`:

- Compute `income = max( w, π(a,z) ) + (1+r)*a` (implement exactly).
- Feasible `a'` bounds:
  - `lo = a_min`
  - `hi = min(a_max, income - c_min)` with `c_min = 1e-10`
- Define `c(a') = income - a'` (must be >0)

Compute expectation term:
- For any candidate `a'`, you need `c'(a', z_k)` for each `z_k`:
  - Evaluate `a'' = a'_old(a', z_k)` from Chebyshev approximation (clamp into `[a_min, a_max]`).
  - Compute `income_next = max(w, π(a', z_k)) + (1+r)*a'`
  - Compute `c_next = income_next - a''` (clip at `c_min`)
  - `mu_next = c_next^(-σ)`
- Then:
  `E_mu = ψ * mu_next(z_j) + (1-ψ) * Σ_k prob_z[k] * mu_next(z_k)`

Euler residual:
- `F(a') = (income - a')^(-σ) - β*(1+r)*E_mu`

**Bracketing and corners:**
- Evaluate `F(lo)` and `F(hi)`:
  - If `F(lo) < 0`, the RHS dominates even at minimal saving ⇒ optimal is at **lower bound** `a'=lo`.
  - If `F(hi) > 0`, LHS dominates even at maximal saving ⇒ optimal is at **upper bound** `a'=hi`.
  - Otherwise bracket root and use `brentq`.
- Always guard against invalid `π`, negative consumption, NaNs.

**Speed:** precompute and reuse objects as much as possible (see Section 6).

### 2.6 Handling occupational kinks
Because `income(a,z)` uses `max(w, π(a,z))`, the residual can have kinks (non-smooth). Root-finding still works because the Euler equation in this setting is usually monotone in `a'` (but be defensive).

Implementation tips:
- Do not rely on derivative-based solvers.
- Use `brentq` with careful brackets and fallback to bounds when no sign change.

---

## 3) Static entrepreneur block: implement once, reuse everywhere

Write a function that returns `(profit, k*, l*, output)` given `(a,z,w,r,λ)` exactly like the baseline replication.

Also provide a **vectorized precompute** for all `(a_nodes, z_grid)` at current `(w,r,λ)`:
- `profit[a_i, z_j]`, `is_entrep[a_i, z_j]`, and `income[a_i, z_j]`

But note: during Euler root finding you will evaluate `π(a', z_k)` at many `a'` values that are **not exactly** on nodes. You need a fast way to get profits/income at arbitrary `a`:

Choose one approach:

**Approach 1 (recommended): compute entrepreneur solution “on the fly”** inside the residual using the closed-form formulas (it’s not that expensive if vectorized for all z_k per a').

**Approach 2 (fastest at scale): precompute on a fine asset grid and interpolate** profits/income as a function of `a` for each `z`.
- If you do this, be explicit about interpolation (linear) and clamping.

Either is acceptable; start with on-the-fly + vectorization over z for clarity.

---

## 4) Stationary distribution & aggregates (simulation approach)

Keep the baseline simulation philosophy:

1. Large population, long burn-in: e.g. `N=200k–400k`, `T=300–500`.
2. Fixed RNG seed for reproducibility.
3. Ability updates:
   - With prob `(1-ψ)` redraw from `prob_z`, else keep same index.
4. Asset update:
   - `a_{t+1} = a'(a_t, z_t)` evaluated by Chebyshev approximation (and interpolation, if necessary).
   - Clamp result into `[a_min, a_max]`.

After burn-in, compute aggregates from cross-section:

For each agent:
- Solve entrepreneur static block (same formulas) and determine entrepreneur status.
- If entrepreneur: add `k*`, `l*`, `y` and external finance `max(0, k* - a)`
- Track average assets `A`.

Return:
- `K`, `L`, `Y`, `A`, `extfin`, `share_entre`
and derived:
- `ext_fin_to_gdp = extfin / Y`
- `K_Y = K / Y`
- `TFP` definition consistent with your target (if matching baseline, use α and 1-α, not hard-coded 1/3,2/3)

---

## 5) General equilibrium loop over (w, r)

For each `λ`:

1. Initialize `(w,r)` (warm start from previous λ).
2. Repeat until market clearing:
   - Given `(w,r)`, solve for policy `a'(a,z)` via Chebyshev Euler iteration (Section 2).
   - Simulate stationary distribution (Section 4).
   - Compute excess demands:
     - `exc_K = K - A`
     - `exc_L = L - (1 - share_entre)`
   - Update prices with damping:
     - `w <- w * (1 + step_w * exc_L)` (clamp to positive)
     - `r <- r + step_r * exc_K` (clamp bounds, e.g. [-0.25, 0.12])
     - Use additional damping to prevent oscillations.
3. Save equilibrium results for plotting.

**Critical:** Warm-start the **policy coefficients** `gamma` from the previous λ to accelerate.

---

## 6) Performance and numerical stability guidance

### 6.1 Precompute Chebyshev basis
- Precompute Chebyshev basis matrix `T` at the collocation nodes once: shape `(N, P)`.
- When evaluating the policy at arbitrary `a`, compute `x(a)` and evaluate `T(x)` via recursion; for speed, consider a fast scalar recursion.

### 6.2 Vectorize residual evaluation over z
When root-finding at a node, the expectation term needs `mu_next` for **all** z_k. Compute this in a vectorized way:
- For a candidate `a'`, compute arrays over z:
  - `profit_next[z]`, `income_next[z]`, `a''_old[z]`, `c_next[z]`, `mu_next[z]`
- Then compute expectation using `ψ` and `prob_z`.

### 6.3 Damping everywhere
Use damping in:
- Policy coefficient updates (`gamma`)
- GE price updates (`w`, `r`)
- If root-finding sometimes fails, blend with previous node value.

### 6.4 Feasibility guards
- Always ensure `c = income - a' >= c_min`.
- If `income` is very low, force `a'=a_min`.
- Clamp `a'` and `a''` to bounds.

### 6.5 Diagnostics
Print periodically:
- max Euler residual at collocation nodes
- max change in `a'(a_i,z_j)`
- GE excess demands

Stop when:
- Euler residuals are small enough for the policy iteration, AND
- GE markets clear within tolerance.

---

## 7) Code organization and required functions

You may write a single file, but modular is better. At minimum implement:

### Chebyshev utilities
- `cheb_nodes(N) -> x in [-1,1]`
- `map_to_cheb(a) -> x`
- `map_from_cheb(x) -> a`
- `cheb_basis(x, P) -> array length P`
- `eval_policy(a, gamma_z) -> a'` for a single z
- `eval_policy_all_z(a, gamma) -> vector a'' over z` (recommended)

### Entrepreneur block
- `entrepreneur_static(a, z, w, r, lam, params) -> profit, kstar, lstar, output`
- Optionally: vectorized version over `z_grid`

### Policy solver (projection)
- `solve_policy_cheb(w, r, lam, gamma_init, grids, params) -> gamma_opt, diagnostics`
  - implements Euler-equation iteration at collocation nodes + Chebyshev refit

### Simulation
- `simulate_stationary(gamma, w, r, lam, shocks, grids, params) -> aggregates`

### GE loop
- `find_equilibrium(lam, w_init, r_init, gamma_init, ...) -> result dict`
  - returns `w,r,Y,K,L,A,TFP,extfin,share_entre,K_Y,ext_fin_to_gdp,gamma_opt`

### Plotting
- replicate the baseline plot structure and save PNGs.

---

## 8) Validation checks (must include)

Before running the full λ grid, run a quick check at one λ (e.g. λ=2):
1. Plot `a'(a,z)` for a few z types (low/median/high) to ensure monotonicity.
2. Check Euler residuals across nodes are small.
3. Check consumption is positive for all simulated agents.
4. Confirm the GE loop decreases `|exc_K|+|exc_L|`.

Optional cross-check:
- Compare your projection-solver equilibrium outputs against the baseline grid-VFI version for one λ, allowing small numerical differences.

---

## 9) Figure targets

Match the baseline replication outputs:

- For each λ, compute:
  - `ext_fin_to_gdp = extfin / Y`
  - `Y_normalized = Y / Y(λ=∞)`
  - `TFP_normalized = TFP / TFP(λ=∞)`
  - `interest_rate = r`
- Plot:
  1. `Y_normalized` and `TFP_normalized` vs `ext_fin_to_gdp`
  2. `r` vs `ext_fin_to_gdp`
- Save under `plots/` directory.

Use clear titles and labels similar to the baseline code.

---

## 10) Implementation notes you should follow

- Use the **paper’s z-grid and probabilities** (hard-code them exactly as in the baseline replication).
- Use a sensible asset domain; start with the baseline `[a_min=1e-6, a_max=4000]`.
- Choose Chebyshev order `P` and nodes `N` such that the policy is accurate:
  - Start with `P=N=25` or `P=N=35` and tune.
- Keep the simulation method for the stationary distribution (do not build a full transition matrix).
- Warm start both prices and policy coefficients when iterating over λ.

---

## 11) Pseudocode sketch (policy solve)

```
function solve_policy_cheb(w, r, lam, gamma_init):
    gamma = gamma_init
    for it in range(max_policy_iter):
        # 1) evaluate policy at nodes: a_old[i,j] = a'(a_i,z_j)
        a_old = policy_at_nodes(gamma)

        # 2) node-by-node Euler solve for new a'
        for j in z_states:
            for i in nodes:
                income = compute_income(a_i, z_j, w, r, lam)
                bounds = [a_min, min(a_max, income-c_min)]
                a_new[i,j] = solve_root_or_corner(F(a'), bounds)

        # 3) fit Chebyshev coefficients separately for each z_j
        for j:
            gamma_new[:,j] = fit_cheb_coeffs(a_nodes, a_new[:,j])

        # 4) damp update and check convergence
        gamma = (1-eta)*gamma + eta*gamma_new
        if max_abs(a_new - a_old) < tol: break

    return gamma
```

---

## 12) Output formatting (summary table)

Print a summary table over λ with columns:
- Lambda, ExtFin/GDP, GDP (normalized), TFP (normalized), r, w, K/Y, %Entrepreneurs

And print key findings like:
- Perfect credit output
- Autarky output loss
- Interest rate range

---

**End of instructions.**
