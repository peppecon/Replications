"""
Buera & Shin (2010) Transition Dynamics — VERSION 4 (SIMULATION) — FINAL RUNNABLE
--------------------------------------------------------------------------------
What this script does (end-to-end):
1) Solve post-reform steady state (no distortions) by simulation-based market clearing
2) Solve pre-reform steady state (with tau+ / tau-) by simulation-based market clearing
3) Compute transition path from pre -> post using a correct Simulation-based TPI:
   - Backward pass: time-dependent Bellman steps using continuation V_{t+1}
   - Forward pass: Monte Carlo distribution simulation with common random numbers
   - Price-path updates per date using excess demands
4) Plot normalized transition paths

Key FIXES vs many "v4" drafts:
- Correct timing: compute aggregates using (a_t,z_t) at time t BEFORE applying policy to get (a_{t+1},z_{t+1})
- True TPI updates: update {w_t,r_t} for each t using ED(t), with terminal anchored at post-SS
- Backward pass uses continuation V_{t+1} (no stationary fixed point inside the transition)

Run:
python buera_shin_v4_final.py --T 125 --na 501 --output_dir outputs
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
import os
import argparse
import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# Model Parameters (Standard Paper Calibration)
# =============================================================================

SIGMA = 1.5           # Risk aversion (CRRA)
BETA = 0.904          # Discount factor
ALPHA = 0.33          # Capital share
NU = 0.21             # Entrepreneur share (Span of control = 0.79)
DELTA = 0.06          # Depreciation
ETA = 4.15            # Pareto tail
PSI = 0.894           # Persistence

# Financial friction
LAMBDA = 1.35         # Collateral constraint

# Distortion parameters (Pre-reform)
TAU_PLUS = 0.57       # Tax rate (positive wedge)
TAU_MINUS = -0.15     # Subsidy rate (negative wedge)
Q_DIST = 1.55         # Correlation: Pr(tau=tau_plus|e) = 1 - exp(-q*e)

# Simulation Parameters
N_AGENTS = 350000      # Monte Carlo agents
T_SIM_SS = 500         # burn-in for stationary
DAMPING_FACTOR = 0.8   # damping in steady-state price updates


# =============================================================================
# Grid Construction
# =============================================================================

def create_ability_grid_paper(eta):
    """Paper's exact 40-point ability discretization"""
    n_z = 40
    M_values = np.zeros(n_z)
    M_values[:38] = np.linspace(0.633, 0.998, 38)
    M_values[38] = 0.999
    M_values[39] = 0.9995
    z_grid = (1 - M_values) ** (-1 / eta)

    prob_z = np.zeros(n_z)
    prob_z[0] = M_values[0] / M_values[-1]
    for j in range(1, n_z):
        prob_z[j] = (M_values[j] - M_values[j - 1]) / M_values[-1]
    prob_z = prob_z / prob_z.sum()
    return z_grid, prob_z


def create_asset_grid(n_a, a_min, a_max):
    """Asset grid with curvature scaling (Power 2)"""
    a_grid = a_min + (a_max - a_min) * np.linspace(0, 1, n_a) ** 2
    a_grid[0] = max(a_grid[0], 1e-6)
    return a_grid


def compute_tau_probs(z_grid, q):
    """Probability of tau_plus given ability"""
    return 1 - np.exp(-q * z_grid)


# =============================================================================
# Entrepreneur Problem (Numba Accelerated)
# =============================================================================

@njit(cache=False)
def solve_entrepreneur_single(a, z, w, r, lam, delta, alpha, upsilon):
    """Static profit maximization for a single state (no tau)"""
    rental = max(r + delta, 1e-8)
    wage = max(w, 1e-8)
    span = 1.0 - upsilon

    aux1 = (1.0 / rental) * alpha * span * z
    aux2 = (1.0 / wage) * (1.0 - alpha) * span * z
    exp1 = 1.0 - (1.0 - alpha) * span
    exp2 = (1.0 - alpha) * span

    k1 = (aux1 ** exp1 * aux2 ** exp2) ** (1.0 / upsilon)
    kstar = min(k1, lam * a)
    lstar = (aux2 * (kstar ** (alpha * span))) ** (1.0 / exp1)

    output = z * ((kstar ** alpha) * (lstar ** (1.0 - alpha))) ** span
    profit = output - wage * lstar - rental * kstar
    return profit, kstar, lstar, output


@njit(cache=False)
def solve_entrepreneur_with_tau(a, z, tau, w, r, lam, delta, alpha, upsilon):
    """
    Static profit maximization WITH distortion tau.
    Distortion applies to output: profit = (1-tau)*y - wl - (r+delta)*k
    Equivalent to z_eff=(1-tau)*z inside FOCs and output.
    """
    z_eff = (1.0 - tau) * z
    if z_eff <= 0.0:
        return -1e10, 0.0, 0.0, 0.0

    rental = max(r + delta, 1e-8)
    wage = max(w, 1e-8)
    span = 1.0 - upsilon

    aux1 = (1.0 / rental) * alpha * span * z_eff
    aux2 = (1.0 / wage) * (1.0 - alpha) * span * z_eff
    exp1 = 1.0 - (1.0 - alpha) * span
    exp2 = (1.0 - alpha) * span

    k1 = (aux1 ** exp1 * aux2 ** exp2) ** (1.0 / upsilon)
    kstar = min(k1, lam * a)
    lstar = (aux2 * (kstar ** (alpha * span))) ** (1.0 / exp1)

    output = z_eff * ((kstar ** alpha) * (lstar ** (1.0 - alpha))) ** span
    profit = output - wage * lstar - rental * kstar
    return profit, kstar, lstar, output


@njit(cache=False, parallel=True)
def precompute_entrepreneur_all(a_grid, z_grid, w, r, lam, delta, alpha, upsilon):
    """Precompute incomes without tau: income = max(w, profit) + (1+r)a"""
    n_a, n_z = len(a_grid), len(z_grid)
    income_g = np.zeros((n_a, n_z))

    for i_z in prange(n_z):
        z = z_grid[i_z]
        for i_a in range(n_a):
            p, _, _, _ = solve_entrepreneur_single(a_grid[i_a], z, w, r, lam, delta, alpha, upsilon)
            if p > w:
                income_g[i_a, i_z] = p + (1.0 + r) * a_grid[i_a]
            else:
                income_g[i_a, i_z] = w + (1.0 + r) * a_grid[i_a]
    return income_g


@njit(cache=False, parallel=True)
def precompute_entrepreneur_with_tau(a_grid, z_grid, tau, w, r, lam, delta, alpha, upsilon):
    """Precompute incomes with tau fixed: income = max(w, profit_tau) + (1+r)a"""
    n_a, n_z = len(a_grid), len(z_grid)
    income_g = np.zeros((n_a, n_z))

    for i_z in prange(n_z):
        z = z_grid[i_z]
        for i_a in range(n_a):
            p, _, _, _ = solve_entrepreneur_with_tau(a_grid[i_a], z, tau, w, r, lam, delta, alpha, upsilon)
            if p > w:
                income_g[i_a, i_z] = p + (1.0 + r) * a_grid[i_a]
            else:
                income_g[i_a, i_z] = w + (1.0 + r) * a_grid[i_a]
    return income_g


# =============================================================================
# Value Function + Policies (Stationary)
# =============================================================================

@njit(cache=False)
def utility(c, sigma):
    if c <= 1e-10:
        return -1e10
    if abs(sigma - 1.0) < 1e-10:
        return np.log(c)
    return (c ** (1.0 - sigma) - 1.0) / (1.0 - sigma)


@njit(cache=False)
def find_optimal_savings(income, a_grid, EV_row, beta, sigma, start_idx):
    n_a = len(a_grid)
    best_val = -1e15
    best_idx = start_idx
    for i_ap in range(start_idx, n_a):
        c = income - a_grid[i_ap]
        if c <= 1e-10:
            break
        val = utility(c, sigma) + beta * EV_row[i_ap]
        if val > best_val:
            best_val = val
            best_idx = i_ap
    return best_val, best_idx


@njit(cache=False, parallel=True)
def bellman_operator(V, a_grid, z_grid, prob_z, income_grid, beta, sigma, psi):
    """
    Stationary Bellman (given V on RHS). Used for steady state only.
    """
    n_a, n_z = len(a_grid), len(z_grid)
    V_new = np.zeros((n_a, n_z))
    pol_idx = np.zeros((n_a, n_z), dtype=np.int64)

    # mean over z of V(a,z)
    V_mean = np.zeros(n_a)
    for i_a in range(n_a):
        s = 0.0
        for i_z in range(n_z):
            s += prob_z[i_z] * V[i_a, i_z]
        V_mean[i_a] = s

    for i_z in prange(n_z):
        start = 0
        EV_row = psi * V[:, i_z] + (1.0 - psi) * V_mean
        for i_a in range(n_a):
            v, idx = find_optimal_savings(income_grid[i_a, i_z], a_grid, EV_row, beta, sigma, start)
            V_new[i_a, i_z] = v
            pol_idx[i_a, i_z] = idx
            start = idx
    return V_new, pol_idx


@njit(cache=False, parallel=True)
def howard_acceleration(V, pol_idx, a_grid, z_grid, prob_z, income_grid, beta, sigma, psi, n_howard=15):
    n_a, n_z = len(a_grid), len(z_grid)
    for _ in range(n_howard):
        V_mean = np.zeros(n_a)
        for i_a in range(n_a):
            s = 0.0
            for i_z in range(n_z):
                s += prob_z[i_z] * V[i_a, i_z]
            V_mean[i_a] = s

        V_next = np.zeros((n_a, n_z))
        for i_z in prange(n_z):
            for i_a in range(n_a):
                ap = pol_idx[i_a, i_z]
                c = income_grid[i_a, i_z] - a_grid[ap]
                V_next[i_a, i_z] = utility(c, sigma) + beta * (psi * V[ap, i_z] + (1.0 - psi) * V_mean[ap])
        V = V_next
    return V


def solve_value_function_stationary(a_grid, z_grid, prob_z, income_grid, beta, sigma, psi,
                                   V_init=None, tol=1e-5, max_iter=500, n_howard=15):
    n_a, n_z = len(a_grid), len(z_grid)
    if V_init is None:
        V = np.zeros((n_a, n_z))
        for i_a in range(n_a):
            c0 = max(income_grid[i_a, 0] - a_grid[0], 1e-2)
            V[i_a, :] = utility(c0, sigma) / (1.0 - beta)
    else:
        V = V_init.copy()

    for _ in range(max_iter):
        V_new, pol_idx = bellman_operator(V, a_grid, z_grid, prob_z, income_grid, beta, sigma, psi)
        diff = np.max(np.abs(V_new - V))
        if diff < tol:
            return V_new, pol_idx
        if n_howard > 0 and diff > 10.0 * tol:
            V = howard_acceleration(V_new, pol_idx, a_grid, z_grid, prob_z, income_grid, beta, sigma, psi, n_howard)
        else:
            V = V_new
    return V, pol_idx


# =============================================================================
# Coupled VFI (Steady-state with tau+ / tau-)
# =============================================================================

@njit(cache=False, parallel=True)
def coupled_bellman_operator(V_p, V_m, a_grid, z_grid, prob_z, prob_tau_plus,
                            inc_p, inc_m, beta, sigma, psi):
    n_a, n_z = len(a_grid), len(z_grid)
    V_p_new = np.zeros((n_a, n_z))
    V_m_new = np.zeros((n_a, n_z))
    pol_p = np.zeros((n_a, n_z), dtype=np.int64)
    pol_m = np.zeros((n_a, n_z), dtype=np.int64)

    # EV(a') = sum_{z'} prob_z(z') * [ p_tau(z') V_p(a',z') + (1-p_tau(z')) V_m(a',z') ]
    EV = np.zeros(n_a)
    for i_a in range(n_a):
        s = 0.0
        for i_z in range(n_z):
            s += prob_z[i_z] * (prob_tau_plus[i_z] * V_p[i_a, i_z] + (1.0 - prob_tau_plus[i_z]) * V_m[i_a, i_z])
        EV[i_a] = s

    for i_z in prange(n_z):
        start_p = 0
        start_m = 0
        EV_row_p = psi * V_p[:, i_z] + (1.0 - psi) * EV
        EV_row_m = psi * V_m[:, i_z] + (1.0 - psi) * EV
        for i_a in range(n_a):
            vp, ip = find_optimal_savings(inc_p[i_a, i_z], a_grid, EV_row_p, beta, sigma, start_p)
            vm, im = find_optimal_savings(inc_m[i_a, i_z], a_grid, EV_row_m, beta, sigma, start_m)
            V_p_new[i_a, i_z] = vp
            V_m_new[i_a, i_z] = vm
            pol_p[i_a, i_z] = ip
            pol_m[i_a, i_z] = im
            start_p = ip
            start_m = im

    return V_p_new, V_m_new, pol_p, pol_m


@njit(cache=False, parallel=True)
def coupled_howard(V_p, V_m, pol_p, pol_m, a_grid, z_grid, prob_z, prob_tau_plus,
                   inc_p, inc_m, beta, sigma, psi, n_howard=15):
    n_a, n_z = len(a_grid), len(z_grid)
    for _ in range(n_howard):
        EV = np.zeros(n_a)
        for i_a in range(n_a):
            s = 0.0
            for i_z in range(n_z):
                s += prob_z[i_z] * (prob_tau_plus[i_z] * V_p[i_a, i_z] + (1.0 - prob_tau_plus[i_z]) * V_m[i_a, i_z])
            EV[i_a] = s

        Vp_next = np.zeros((n_a, n_z))
        Vm_next = np.zeros((n_a, n_z))
        for i_z in prange(n_z):
            for i_a in range(n_a):
                ap = pol_p[i_a, i_z]
                am = pol_m[i_a, i_z]
                cp = inc_p[i_a, i_z] - a_grid[ap]
                cm = inc_m[i_a, i_z] - a_grid[am]
                Vp_next[i_a, i_z] = utility(cp, sigma) + beta * (psi * V_p[ap, i_z] + (1.0 - psi) * EV[ap])
                Vm_next[i_a, i_z] = utility(cm, sigma) + beta * (psi * V_m[am, i_z] + (1.0 - psi) * EV[am])
        V_p, V_m = Vp_next, Vm_next
    return V_p, V_m


def solve_value_function_coupled_stationary(a_grid, z_grid, prob_z, prob_tau_plus,
                                            inc_p, inc_m, beta, sigma, psi,
                                            Vp_init=None, Vm_init=None,
                                            tol=1e-5, max_iter=500, n_howard=15):
    n_a, n_z = len(a_grid), len(z_grid)
    if Vp_init is None:
        V_p = np.zeros((n_a, n_z))
        V_m = np.zeros((n_a, n_z))
        for i_a in range(n_a):
            cp0 = max(inc_p[i_a, 0] - a_grid[0], 1e-2)
            cm0 = max(inc_m[i_a, 0] - a_grid[0], 1e-2)
            V_p[i_a, :] = utility(cp0, sigma) / (1.0 - beta)
            V_m[i_a, :] = utility(cm0, sigma) / (1.0 - beta)
    else:
        V_p = Vp_init.copy()
        V_m = Vm_init.copy()

    for _ in range(max_iter):
        Vp_new, Vm_new, pol_p, pol_m = coupled_bellman_operator(
            V_p, V_m, a_grid, z_grid, prob_z, prob_tau_plus, inc_p, inc_m, beta, sigma, psi
        )
        diff = max(np.max(np.abs(Vp_new - V_p)), np.max(np.abs(Vm_new - V_m)))
        if diff < tol:
            return Vp_new, Vm_new, pol_p, pol_m
        if n_howard > 0 and diff > 10.0 * tol:
            V_p, V_m = coupled_howard(Vp_new, Vm_new, pol_p, pol_m, a_grid, z_grid, prob_z,
                                      prob_tau_plus, inc_p, inc_m, beta, sigma, psi, n_howard)
        else:
            V_p, V_m = Vp_new, Vm_new

    return V_p, V_m, pol_p, pol_m


# =============================================================================
# Shocks + Simulation Steps
# =============================================================================

def generate_shocks(n_agents, t_steps, psi, prob_z, seed=42):
    rng = np.random.default_rng(seed)
    resets = (rng.random((t_steps, n_agents)) > psi).astype(np.uint8)
    cdf_z = np.cumsum(prob_z)
    shocks = np.searchsorted(cdf_z, rng.random((t_steps, n_agents))).astype(np.uint8)
    return resets, shocks


def generate_shocks_with_tau(n_agents, t_steps, psi, prob_z, prob_tau_plus, seed=42):
    rng = np.random.default_rng(seed)
    resets = (rng.random((t_steps, n_agents)) > psi).astype(np.uint8)
    cdf_z = np.cumsum(prob_z)
    shocks = np.searchsorted(cdf_z, rng.random((t_steps, n_agents))).astype(np.uint8)

    # tau shock only when reset, depends on new z draw
    tau_shocks = np.zeros((t_steps, n_agents), dtype=np.uint8)
    for t in range(t_steps):
        probs = prob_tau_plus[shocks[t]]
        tau_shocks[t] = (rng.random(n_agents) < probs).astype(np.uint8)
    return resets, shocks, tau_shocks


@njit(cache=False)
def get_interp_weights(x, x_grid):
    n = len(x_grid)
    if x <= x_grid[0]:
        return 0, 0, 1.0
    if x >= x_grid[-1]:
        return n - 2, n - 1, 0.0
    low, high = 0, n - 1
    while high - low > 1:
        mid = (low + high) // 2
        if x_grid[mid] > x:
            high = mid
        else:
            low = mid
    w_low = (x_grid[high] - x) / (x_grid[high] - x_grid[low])
    return low, high, w_low


@njit(cache=False, parallel=True)
def simulate_step(a_curr, z_idx_curr, policy_a_vals, a_grid, reset_shocks, ability_shocks):
    """
    Transition for z:
      with prob psi: z_{t+1}=z_t
      with prob 1-psi: z_{t+1}=new draw
    Decision at time t uses (a_t, z_t) => policy indexed by z_idx_curr.
    """
    n = len(a_curr)
    a_next = np.zeros(n)
    z_idx_next = np.zeros(n, dtype=np.int64)

    for i in prange(n):
        z_idx_next[i] = ability_shocks[i] if reset_shocks[i] else z_idx_curr[i]

        il, ih, wl = get_interp_weights(a_curr[i], a_grid)
        # policy depends on current z (before reset)
        zc = z_idx_curr[i]
        a_next[i] = wl * policy_a_vals[il, zc] + (1.0 - wl) * policy_a_vals[ih, zc]

    return a_next, z_idx_next


@njit(cache=False, parallel=True)
def simulate_step_coupled(a_curr, z_idx_curr, tau_curr,
                         pol_p_vals, pol_m_vals, a_grid,
                         resets, shocks, tau_shocks, tau_p, tau_m):
    """
    Pre-reform: tau is drawn only on reset (entry), depends on new z draw externally.
    Decision at time t uses current tau_t and z_t.
    """
    n = len(a_curr)
    a_next = np.zeros(n)
    z_next = np.zeros(n, dtype=np.int64)
    tau_next = np.zeros(n)

    for i in prange(n):
        if resets[i]:
            z_next[i] = shocks[i]
            tau_next[i] = tau_p if tau_shocks[i] else tau_m
        else:
            z_next[i] = z_idx_curr[i]
            tau_next[i] = tau_curr[i]

        il, ih, wl = get_interp_weights(a_curr[i], a_grid)
        zc = z_idx_curr[i]
        if tau_curr[i] == tau_p:
            a_next[i] = wl * pol_p_vals[il, zc] + (1.0 - wl) * pol_p_vals[ih, zc]
        else:
            a_next[i] = wl * pol_m_vals[il, zc] + (1.0 - wl) * pol_m_vals[ih, zc]

    return a_next, z_next, tau_next


# =============================================================================
# Aggregates + Excess Demand (IMPORTANT timing helper for TPI)
# =============================================================================

@njit(cache=False, parallel=True)
def compute_ED_and_aggregates_from_population(a_pop, z_idx_pop, z_grid,
                                             w, r, lam, delta, alpha, upsilon):
    """
    Given current population (a_t,z_t) and prices (w_t,r_t),
    compute aggregates and relative excess demands:
      ED_K = (K - A)/A
      ED_L = (L - (1-s_e)) / (1-s_e)
    """
    n = len(a_pop)
    K = 0.0
    L = 0.0
    Y = 0.0
    A = 0.0
    extfin = 0.0
    s_e = 0.0

    wage = max(w, 1e-8)
    for i in prange(n):
        a = a_pop[i]
        z = z_grid[z_idx_pop[i]]
        A += a
        p, k, l, o = solve_entrepreneur_with_tau(a, z, 0.0, w, r, lam, delta, alpha, upsilon)  # tau=0 on transition
        if p > wage:
            s_e += 1.0
            K += k
            L += l
            Y += o
            if k > a:
                extfin += (k - a)

    K /= n
    L /= n
    Y /= n
    A /= n
    extfin /= n
    s_e /= n

    Ls = max(1.0 - s_e, 1e-6)
    ED_K = (K - A) / max(A, 1e-6)
    ED_L = (L - Ls) / max(Ls, 1e-3)

    span = 1.0 - upsilon
    TFP = Y / max(((K ** alpha) * (Ls ** (1.0 - alpha))) ** span, 1e-8)

    return ED_L, ED_K, Y, K, L, A, TFP, extfin, s_e


# =============================================================================
# Stationary Equilibrium Finders (Simulation-based clearing)
# =============================================================================

def find_equilibrium_nodist(a_grid, z_grid, prob_z, params,
                           w_init=0.8, r_init=-0.04,
                           max_iter=100, tol=1e-3, verbose=True):
    """Stationary equilibrium WITHOUT distortions"""
    delta, alpha, upsilon, lam, beta, sigma, psi = params
    w, r = w_init, r_init
    V = None

    resets, shocks = generate_shocks(N_AGENTS, T_SIM_SS, psi, prob_z, seed=42)

    # price update steps
    w_step, r_step = 0.3, 0.05
    excL_prev, excK_prev = 0.0, 0.0
    w_prev, r_prev = w, r

    best_err = np.inf
    best = None

    for it in range(max_iter):
        inc = precompute_entrepreneur_all(a_grid, z_grid, w, r, lam, delta, alpha, upsilon)
        V, pol_idx = solve_value_function_stationary(a_grid, z_grid, prob_z, inc, beta, sigma, psi, V_init=V)

        # simulate distribution
        rng = np.random.default_rng(123)
        a_pop = np.ones(N_AGENTS) * a_grid[0]
        z_pop = rng.integers(0, len(z_grid), N_AGENTS, dtype=np.int64)

        pol_vals = a_grid[pol_idx]
        for t in range(T_SIM_SS):
            a_pop, z_pop = simulate_step(a_pop, z_pop, pol_vals, a_grid, resets[t], shocks[t])

        # compute aggregates (levels) for clearing
        # reuse your entrepreneur solver tau=0
        ED_L, ED_K, Y, K, L, A, TFP, extfin, s_e = compute_ED_and_aggregates_from_population(
            a_pop, z_pop, z_grid, w, r, lam, delta, alpha, upsilon
        )

        # convert relative ED back to level-style for your original update logic
        # but we’ll keep relative error measure:
        total_err = abs(ED_L) + abs(ED_K)

        if verbose:
            print(f"[Post SS {it+1:03d}] w={w:.6f} r={r:.6f} | ED_L={ED_L:+.4e} ED_K={ED_K:+.4e}")

        if total_err < best_err:
            best_err = total_err
            best = dict(w=w, r=r, Y=Y, K=K, L=L, A=A, TFP=TFP,
                        extfin=extfin, s_e=s_e, V=V.copy(), pol_idx=pol_idx.copy(),
                        a_sim=a_pop.copy(), z_idx_sim=z_pop.copy())

        if abs(ED_L) < tol and abs(ED_K) < tol:
            if verbose:
                print("Converged (excess demands)!")
            break

        price_change = abs(w - w_prev) / max(w, 1e-6) + abs(r - r_prev) / max(abs(r), 1e-6)
        if it > 5 and price_change < 1e-6 and (w_step < 0.01 or r_step < 0.005):
            if verbose:
                print("Converged (prices stabilized / noise floor)!")
            break

        # adaptive step shrink on sign flip
        if it > 0:
            if np.sign(ED_L) != np.sign(excL_prev):
                w_step *= 0.5
            if np.sign(ED_K) != np.sign(excK_prev):
                r_step *= 0.5

        # update prices (relative ED)
        w_target = w * (1.0 + w_step * ED_L)
        r_target = r + r_step * ED_K

        # bounds
        w_target = np.clip(w_target, 0.01, 2.5)
        r_cap = (1.0 / beta) - 1.0 - 1e-6
        r_target = np.clip(r_target, -0.25, r_cap)

        # damping
        w_prev, r_prev = w, r
        w = DAMPING_FACTOR * w + (1.0 - DAMPING_FACTOR) * w_target
        r = DAMPING_FACTOR * r + (1.0 - DAMPING_FACTOR) * r_target

        excL_prev, excK_prev = ED_L, ED_K

    return best


def find_equilibrium_with_dist(a_grid, z_grid, prob_z, prob_tau_plus, params,
                              tau_p, tau_m,
                              w_init=0.55, r_init=-0.04,
                              max_iter=100, tol=1e-3, verbose=True):
    """Stationary equilibrium WITH distortions (tau drawn on reset)"""
    delta, alpha, upsilon, lam, beta, sigma, psi = params
    w, r = w_init, r_init
    Vp, Vm = None, None

    resets, shocks, tau_shocks = generate_shocks_with_tau(N_AGENTS, T_SIM_SS, psi, prob_z, prob_tau_plus, seed=42)

    w_step, r_step = 0.3, 0.05
    excL_prev, excK_prev = 0.0, 0.0
    w_prev, r_prev = w, r

    best_err = np.inf
    best = None

    for it in range(max_iter):
        inc_p = precompute_entrepreneur_with_tau(a_grid, z_grid, tau_p, w, r, lam, delta, alpha, upsilon)
        inc_m = precompute_entrepreneur_with_tau(a_grid, z_grid, tau_m, w, r, lam, delta, alpha, upsilon)

        Vp, Vm, pol_p, pol_m = solve_value_function_coupled_stationary(
            a_grid, z_grid, prob_z, prob_tau_plus, inc_p, inc_m, beta, sigma, psi,
            Vp_init=Vp, Vm_init=Vm, n_howard=15
        )

        # simulate distribution
        rng = np.random.default_rng(123)
        a_pop = np.ones(N_AGENTS) * a_grid[0]
        z_pop = rng.integers(0, len(z_grid), N_AGENTS, dtype=np.int64)
        # initial tau given initial z
        tau_pop = np.empty(N_AGENTS)
        u = rng.random(N_AGENTS)
        for i in range(N_AGENTS):
            tau_pop[i] = tau_p if u[i] < prob_tau_plus[z_pop[i]] else tau_m

        pol_p_vals = a_grid[pol_p]
        pol_m_vals = a_grid[pol_m]

        for t in range(T_SIM_SS):
            a_pop, z_pop, tau_pop = simulate_step_coupled(
                a_pop, z_pop, tau_pop, pol_p_vals, pol_m_vals, a_grid,
                resets[t], shocks[t], tau_shocks[t], tau_p, tau_m
            )

        # compute aggregates under distortions for clearing:
        # We compute K,L,Y by actually applying tau per agent.
        # For market clearing, the relevant EDs are still K vs A and L vs workers.
        n = len(a_pop)
        K = 0.0
        L = 0.0
        Y = 0.0
        A = 0.0
        extfin = 0.0
        s_e = 0.0
        wage = max(w, 1e-8)
        for i in range(n):
            a = a_pop[i]
            z = z_grid[z_pop[i]]
            tau = tau_pop[i]
            A += a
            p, k, l, o = solve_entrepreneur_with_tau(a, z, tau, w, r, lam, delta, alpha, upsilon)
            if p > wage:
                s_e += 1.0
                K += k
                L += l
                Y += o
                if k > a:
                    extfin += (k - a)
        K /= n
        L /= n
        Y /= n
        A /= n
        extfin /= n
        s_e /= n
        Ls = max(1.0 - s_e, 1e-6)

        ED_K = (K - A) / max(A, 1e-6)
        ED_L = (L - Ls) / max(Ls, 1e-3)

        span = 1.0 - upsilon
        TFP = Y / max(((K ** alpha) * (Ls ** (1.0 - alpha))) ** span, 1e-8)

        total_err = abs(ED_L) + abs(ED_K)

        if verbose:
            print(f"[Pre SS  {it+1:03d}] w={w:.6f} r={r:.6f} | ED_L={ED_L:+.4e} ED_K={ED_K:+.4e}")

        if total_err < best_err:
            best_err = total_err
            best = dict(w=w, r=r, Y=Y, K=K, L=L, A=A, TFP=TFP,
                        extfin=extfin, s_e=s_e, V_p=Vp.copy(), V_m=Vm.copy(),
                        pol_p=pol_p.copy(), pol_m=pol_m.copy(),
                        a_sim=a_pop.copy(), z_idx_sim=z_pop.copy(), tau_sim=tau_pop.copy())

        if abs(ED_L) < tol and abs(ED_K) < tol:
            if verbose:
                print("Converged (excess demands)!")
            break

        price_change = abs(w - w_prev) / max(w, 1e-6) + abs(r - r_prev) / max(abs(r), 1e-6)
        if it > 5 and price_change < 1e-6 and (w_step < 0.01 or r_step < 0.005):
            if verbose:
                print("Converged (prices stabilized / noise floor)!")
            break

        # collapse guard
        if L < 1e-6 or Y < 1e-10:
            w *= 0.8
            if verbose:
                print("[WARN] collapse detected; reducing w and continuing")
            continue

        if it > 0:
            if np.sign(ED_L) != np.sign(excL_prev):
                w_step *= 0.5
            if np.sign(ED_K) != np.sign(excK_prev):
                r_step *= 0.5

        w_target = w * (1.0 + w_step * ED_L)
        r_target = r + r_step * ED_K

        w_target = np.clip(w_target, 0.01, 2.5)
        r_cap = (1.0 / beta) - 1.0 - 1e-6
        r_target = np.clip(r_target, -0.25, r_cap)

        w_prev, r_prev = w, r
        w = DAMPING_FACTOR * w + (1.0 - DAMPING_FACTOR) * w_target
        r = DAMPING_FACTOR * r + (1.0 - DAMPING_FACTOR) * r_target

        excL_prev, excK_prev = ED_L, ED_K

    return best


# =============================================================================
# TRANSITION — Correct Simulation-based TPI
# =============================================================================

@njit(cache=False, parallel=True)
def bellman_time_step_from_continuation(V_tp1, a_grid, z_grid, prob_z, income_t, beta, sigma, psi):
    """
    Computes (V_t, pol_t) given continuation V_{t+1} and income grid at time t.
    """
    n_a, n_z = len(a_grid), len(z_grid)
    V_t = np.zeros((n_a, n_z))
    pol = np.zeros((n_a, n_z), dtype=np.int64)

    # V_mean(a') = sum_z prob_z(z) * V_{t+1}(a',z)
    V_mean = np.zeros(n_a)
    for i_a in range(n_a):
        s = 0.0
        for i_z in range(n_z):
            s += prob_z[i_z] * V_tp1[i_a, i_z]
        V_mean[i_a] = s

    for i_z in prange(n_z):
        EV_row = psi * V_tp1[:, i_z] + (1.0 - psi) * V_mean
        start = 0
        for i_a in range(n_a):
            v, idx = find_optimal_savings(income_t[i_a, i_z], a_grid, EV_row, beta, sigma, start)
            V_t[i_a, i_z] = v
            pol[i_a, i_z] = idx
            start = idx

    return V_t, pol


def solve_transition_tpi_simulation(pre_eq, post_eq, params, a_grid, z_grid, prob_z,
                                    T=125,
                                    max_tpi_iter=50,
                                    tol=1e-3,
                                    damping=0.7,
                                    eta_w=0.20,
                                    eta_r=0.02,
                                    seed=12345):
    """
    Correct simulation-based TPI:
    - Guess paths (w_t,r_t) anchored to post-SS
    - Backward pass: compute policies pol_t using continuation V_{t+1}
    - Forward pass: simulate distribution and compute ED(t) using beginning-of-period states
    - Update price paths per date using ED(t), keep terminal anchored
    """
    delta, alpha, upsilon, lam, beta, sigma, psi = params

    # initial micro state = pre-reform stationary distribution
    a0 = pre_eq["a_sim"].copy()
    z0 = pre_eq["z_idx_sim"].copy()

    # terminal continuation = post-reform steady state value function
    V_T = post_eq["V"].copy()
    pol_T = post_eq["pol_idx"].copy()

    # initial guess price paths (exponential bridge)
    w0, r0 = pre_eq["w"], pre_eq["r"]
    wT, rT = post_eq["w"], post_eq["r"]

    tgrid = np.arange(T)
    kappa = 0.05
    w_path = wT + (w0 - wT) * np.exp(-kappa * tgrid)
    r_path = rT + (r0 - rT) * np.exp(-kappa * tgrid)
    w_path[-1] = wT
    r_path[-1] = rT

    # common random numbers shocks
    rng = np.random.default_rng(seed)
    resets = (rng.random((T, len(a0))) > psi).astype(np.uint8)
    cdf_z = np.cumsum(prob_z)
    shocks = np.searchsorted(cdf_z, rng.random((T, len(a0)))).astype(np.uint8)

    # storage
    Y = np.zeros(T)
    K = np.zeros(T)
    L = np.zeros(T)
    A = np.zeros(T)
    TFP = np.zeros(T)
    extfin = np.zeros(T)
    s_e = np.zeros(T)
    ED_L = np.zeros(T)
    ED_K = np.zeros(T)

    for it in range(max_tpi_iter):
        # ---------------------------
        # Backward pass: policies[t]
        # ---------------------------
        policies = [None] * T
        policies[T - 1] = pol_T  # anchor terminal policy

        V_next = V_T.copy()
        for t in range(T - 2, -1, -1):
            income_t = precompute_entrepreneur_all(a_grid, z_grid, w_path[t], r_path[t], lam, delta, alpha, upsilon)
            V_t, pol_t = bellman_time_step_from_continuation(
                V_next, a_grid, z_grid, prob_z, income_t, beta, sigma, psi
            )
            policies[t] = pol_t
            V_next = V_t

        # ---------------------------
        # Forward pass: simulate dist
        # ---------------------------
        a_pop = a0.copy()
        z_pop = z0.copy()

        for t in range(T):
            # (1) compute aggregates at time t using (a_t,z_t) and prices (w_t,r_t)  <-- timing fix
            edl, edk, y, k, l, aavg, tfp, ef, se = compute_ED_and_aggregates_from_population(
                a_pop, z_pop, z_grid, w_path[t], r_path[t], lam, delta, alpha, upsilon
            )
            ED_L[t] = edl
            ED_K[t] = edk
            Y[t] = y
            K[t] = k
            L[t] = l
            A[t] = aavg
            TFP[t] = tfp
            extfin[t] = ef
            s_e[t] = se

            # (2) apply policy to get next state
            if t < T - 1:
                pol_vals = a_grid[policies[t]]
                a_pop, z_pop = simulate_step(a_pop, z_pop, pol_vals, a_grid, resets[t], shocks[t])

        maxED = max(np.max(np.abs(ED_L)), np.max(np.abs(ED_K)))
        print(f"[TPI {it+1:02d}] max|ED_L|={np.max(np.abs(ED_L)):.4e}  max|ED_K|={np.max(np.abs(ED_K)):.4e}")
        if maxED < tol:
            break

        # ---------------------------
        # Update price paths per date
        # ---------------------------
        w_target = w_path * (1.0 + eta_w * ED_L)
        r_target = r_path + eta_r * ED_K

        w_target = np.clip(w_target, 1e-4, 5.0)
        r_cap = (1.0 / beta) - 1.0 - 1e-6
        r_target = np.clip(r_target, -0.5, r_cap)

        # anchor terminal
        w_target[-1] = wT
        r_target[-1] = rT

        w_path = damping * w_path + (1.0 - damping) * w_target
        r_path = damping * r_path + (1.0 - damping) * r_target

    return dict(t=np.arange(T), w=w_path, r=r_path,
                Y=Y, K=K, L=L, A=A, TFP=TFP,
                ED_L=ED_L, ED_K=ED_K,
                extfin=extfin, s_e=s_e)


# =============================================================================
# Plotting
# =============================================================================

def plot_transition(pre_eq, post_eq, trans, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)

    T = len(trans["t"])
    t = trans["t"]

    # Normalize positive variables by pre-SS
    norm_Y = trans["Y"] / max(pre_eq["Y"], 1e-12)
    norm_K = trans["K"] / max(pre_eq["K"], 1e-12)
    norm_TFP = trans["TFP"] / max(pre_eq["TFP"], 1e-12)
    norm_w = trans["w"] / max(pre_eq["w"], 1e-12)

    # r: plot level (normalizing by negative number is confusing)
    path_r = trans["r"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].plot(t, norm_Y)
    axes[0, 0].set_title("Output (Y) / pre-SS")

    axes[0, 1].plot(t, norm_TFP)
    axes[0, 1].set_title("TFP / pre-SS")

    axes[0, 2].plot(t, norm_K)
    axes[0, 2].set_title("Capital (K) / pre-SS")

    axes[1, 0].plot(t, norm_w)
    axes[1, 0].set_title("Wage (w) / pre-SS")

    axes[1, 1].plot(t, path_r)
    axes[1, 1].set_title("Interest rate (r) level")

    axes[1, 2].plot(t, np.abs(trans["ED_L"]), label="|ED_L|")
    axes[1, 2].plot(t, np.abs(trans["ED_K"]), label="|ED_K|")
    axes[1, 2].set_title("Excess demand (abs)")
    axes[1, 2].legend()

    for ax in axes.flatten():
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("t")

    plt.tight_layout()
    outpath = os.path.join(output_dir, "transition_v4_final.png")
    plt.savefig(outpath, dpi=200)
    print(f"Saved plot: {outpath}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Buera-Shin V4 Final (Simulation TPI)")
    parser.add_argument("--T", type=int, default=125)
    parser.add_argument("--na", type=int, default=501)
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    z_grid, prob_z = create_ability_grid_paper(ETA)
    a_grid = create_asset_grid(args.na, 1e-6, 4000.0)
    prob_tau_plus = compute_tau_probs(z_grid, Q_DIST)

    params = (DELTA, ALPHA, NU, LAMBDA, BETA, SIGMA, PSI)

    print("Step 1: Post-reform steady state (no distortions)")
    post_eq = find_equilibrium_nodist(a_grid, z_grid, prob_z, params, w_init=0.8, r_init=-0.04, verbose=True)
    post_eq["V"] = post_eq["V"]
    post_eq["pol_idx"] = post_eq["pol_idx"]

    print("\nStep 2: Pre-reform steady state (with distortions)")
    pre_eq = find_equilibrium_with_dist(a_grid, z_grid, prob_z, prob_tau_plus, params, TAU_PLUS, TAU_MINUS, w_init=0.55, verbose=True)

    if pre_eq is None:
        print("Pre-reform failed.")
        return

    print("\nStep 3: Transition path (correct simulation-based TPI)")
    trans = solve_transition_tpi_simulation(
        pre_eq=dict(w=pre_eq["w"], r=pre_eq["r"], Y=pre_eq["Y"], K=pre_eq["K"], TFP=pre_eq["TFP"],
                    a_sim=pre_eq["a_sim"], z_idx_sim=pre_eq["z_idx_sim"]),
        post_eq=dict(w=post_eq["w"], r=post_eq["r"], Y=post_eq["Y"], K=post_eq["K"], TFP=post_eq["TFP"],
                     V=post_eq["V"], pol_idx=post_eq["pol_idx"]),
        params=params, a_grid=a_grid, z_grid=z_grid, prob_z=prob_z, T=args.T
    )

    plot_transition(pre_eq, post_eq, trans, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
