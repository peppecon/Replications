
"""
Buera & Shin (2010) — Appendix Algorithm B.2
Fast and stable Monte Carlo + market-clearing (computed in Numba)

Key improvements vs naive MC implementation:
1) NO tail anchoring. We solve for prices on t=0..T-1 and impose terminal condition at t=T.
2) Per-period market clearing uses a *binned distribution* on the (a,z) grid (and (a,z,tau) pre-reform).
   This makes wage/interest root-finds fast + much less noisy, eliminating ED spikes from MC jitter.
3) Robust bracketing for per-period wage and interest rate (scan + bisection).
4) Convergence criteria require small sequence updates AND small market-clearing gaps.

Economy:
- Agents choose occupation each period: entrepreneur if profit(a,z;w,r) > w (assets return cancels).
- Ability follows "reset" process: with prob PSI keep z, else draw z' from prob_z.
- Pre-reform only: distortion tau ∈ {tau_plus, tau_minus} drawn conditional on z when ability resets.

What this script does:
A) Solve post-reform stationary steady state (tau=0) using B.2-style nested loops.
B) Solve pre-reform stationary steady state (distortions) to obtain initial distribution at t=0.
C) Solve the transition path after reform at t=0 using Algorithm B.2.

Plot:
- Final figure shows t=-4..+20 (shock at 0).
- All positive variables normalized by pre steady state.
- Interest rate plotted as (1+r)/(1+r_pre) for a clean normalization even if r_pre < 0.

Run:
  python transition_howard.py --T 125 --na 1200 --amax 300 --N 350000 --out outputs

Tip (debug faster):
  python ... --N 50000 --T 60 --na 401
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
import warnings
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# Calibration (paper)
# =============================================================================
SIGMA = 1.5
BETA  = 0.904
ALPHA = 0.33
NU    = 0.21          # entrepreneur share; span-of-control = 1-NU
DELTA = 0.06
ETA   = 4.15
PSI   = 0.894

LAMBDA = 1.35         # collateral constraint multiplier

# Pre-reform distortions (only for initial steady state / distribution)
TAU_PLUS  = 0.57
TAU_MINUS = -0.15
Q_DIST    = 1.55

# =============================================================================
# Algorithm B.2 numerical controls
# =============================================================================
# Relaxation parameters (η_w, η_r in the Appendix)
ETA_W = 0.35
ETA_R = 0.20

# Convergence tolerances (sequence sup norms)
TOL_W_SEQ = 2e-5
TOL_R_SEQ = 2e-5

# How hard we enforce market clearing at the end of the algorithm
TOL_ED_L = 2e-4
TOL_ED_K = 2e-4

MAX_W_INNER = 30    # inner iterations on wage path given r path
MAX_OUTER   = 30    # outer iterations on interest rate path

# Per-period bisection parameters
N_SCAN       = 28    # scan points for robust bracketing
N_SCAN_NC    = 100   # finer scan for neoclassical (labor market can be discontinuous)
MAX_BISECT_IT = 28   # bisection steps once bracket found

# Price bounds
W_MIN, W_MAX = 0.02, 8.0
# Important: r + delta is the rental cost in the firm problem. Ensure rental > 0.
R_MIN = -DELTA + 1e-6
R_MAX = (1.0 / BETA) - 1.0 - 1e-6  # ≈ 0.106, appropriate for frictions case
# For neoclassical (λ→∞), interest rates can be MUCH higher due to unconstrained capital demand
R_MAX_NC = 0.50  # Allow up to 50% interest rate for neoclassical debugging

# =============================================================================
# Value Iteration Controls
# =============================================================================
# Tolerance for value function convergence in stationary solvers
TOL_VF = 1e-6
# Maximum iterations for value function iteration (post-reform / no distortion)
MAX_ITER_VF_POST = 800
# Maximum iterations for value function iteration (pre-reform / with distortion)
MAX_ITER_VF_PRE = 900
# Number of Howard policy improvement steps per Bellman iteration
N_HOWARD = 25
# Multiplier for tolerance to decide when to apply Howard improvement
HOWARD_TOL_MULT = 20

# =============================================================================
# Steady State Solver Controls
# =============================================================================
# Post-reform steady state
SS_POST_MAX_OUTER = 50    # outer iterations on interest rate
SS_POST_MAX_INNER = 50    # inner iterations on wage given r
SS_POST_TOL_W = 5e-5      # wage convergence in inner loop
SS_POST_TOL_R = 1e-5      # interest rate convergence in outer loop
SS_POST_TOL_ED_L = 2e-4   # labor market clearing tolerance for post SS
SS_POST_TOL_ED_K = 2e-4   # capital market clearing tolerance for post SS

# Pre-reform steady state (with distortions)
SS_PRE_MAX_OUTER = 50     # outer iterations on interest rate
SS_PRE_MAX_INNER = 50     # inner iterations on wage given r
SS_PRE_TOL_W = 5e-5       # wage convergence in inner loop
SS_PRE_TOL_R = 1e-5       # interest rate convergence in outer loop
SS_PRE_TOL_ED_L = 3e-4    # labor market clearing tolerance for pre SS
SS_PRE_TOL_ED_K = 3e-4    # capital market clearing tolerance for pre SS

# Neoclassical (λ→∞) relaxed tolerances
# In neoclassical case, occupational choice is discrete and can prevent exact labor clearing
SS_NC_TOL_ED_L = 2e-3     # 2% labor market gap acceptable for neoclassical
SS_NC_TOL_ED_K = 5e-4     # capital still clears well

# =============================================================================
# Grid Parameters
# =============================================================================
# Ability grid
Z_GRID_SIZE = 300         # number of ability grid points (default)
ZMAX_TARGET = 6.5         # target maximum ability level
Z_UMAX = 0.995            # fallback probability mass cutoff for ability grid

# Asset grid
A_GRID_POWER = 1.5        # power for non-uniform asset grid spacing
A_MIN_FLOOR = 1e-10       # minimum asset level (floor)

# =============================================================================
# Small Value Thresholds (numerical stability)
# =============================================================================
# Price floors / tolerance for detecting near-zero values
EPS_WAGE = 1e-12          # floor for wage in various computations
EPS_RENTAL = 1e-12        # floor for rental rate (r + delta)
EPS_CONSUMPTION = 1e-12   # floor for consumption in utility
EPS_Z_EFF = 1e-14         # floor for effective productivity
EPS_SIGMA_LOG = 1e-10     # tolerance for sigma near log utility
EPS_ROOT_CONVERGED = 1e-10         # tolerance for root solver convergence check
EPS_BISECT_TOL = 1e-10    # tolerance for bisection solver
EPS_WEIGHT = 1e-12        # floor for distribution weights
EPS_INIT_C = 1e-3         # floor for initial consumption guess in value iteration
EPS_TFP_DENOM = 1e-12     # floor for TFP denominator
EPS_TOP_MASS_WARN = 1e-3  # threshold for warning about mass at top of asset grid

# Neoclassical benchmark
LAM_NEOCLASSICAL = 1e6    # very large λ removes collateral constraint

# =============================================================================
# Grids (paper's ability grid)
# =============================================================================
def create_ability_grid_paper(
    eta: float,
    n_z: int = None,  # Will use Z_GRID_SIZE if None
    zmax_target: float = 6.5,   # set None if you want to pick umax directly
    umax: float = 0.995,  # e.g. 0.995
):
    """
    Pareto ability grid using equal-probability bins on u in [0, umax].
    z(u) = (1-u)^(-1/eta)

    If zmax_target is given, we set umax so that the top quantile equals zmax_target:
        umax = 1 - zmax_target^(-eta)
    """
    if n_z is None:
        n_z = Z_GRID_SIZE
        
    if umax is None:
        if zmax_target is None:
            raise ValueError("Provide either umax or zmax_target.")
        umax = 1.0 - zmax_target ** (-eta)

    # equal-probability bins on [0, umax]
    edges = np.linspace(0.0, umax, n_z + 1)
    mids  = 0.5 * (edges[:-1] + edges[1:])

    z_grid = (1.0 - mids) ** (-1.0 / eta)

    # probabilities = bin lengths, renormalized (sum to 1)
    prob_z = (edges[1:] - edges[:-1]) / umax
    prob_z = prob_z / prob_z.sum()

    return z_grid.astype(np.float64), prob_z.astype(np.float64)

def create_asset_grid(n_a: int, a_min: float, a_max: float, power: float = A_GRID_POWER):
    x = np.linspace(0.0, 1.0, n_a) ** power
    a = a_min + (a_max - a_min) * x
    a[0] = max(a[0], A_MIN_FLOOR)
    return a.astype(np.float64)

def compute_tau_probs(z_grid: np.ndarray, q: float):
    return (1.0 - np.exp(-q * z_grid)).astype(np.float64)

# =============================================================================
# Firm problem (static): span-of-control technology
# y = z * (k^alpha * l^(1-alpha))^(1-nu)
# =============================================================================
@njit(cache=False)
def solve_firm_no_tau(a, z, w, r, lam, delta, alpha, nu):
    """
    Returns: profit, k*, l*, output
    Constraint: k <= lam * a
    """
    rental = r + delta
    if rental <= 1e-12:
        rental = 1e-12
    wage = w
    if wage <= 1e-12:
        wage = 1e-12

    span = 1.0 - nu  # returns to scale in (k,l)

    # closed-form unconstrained k (from FOCs) then apply collateral constraint
    aux1 = (alpha * span * z) / rental
    aux2 = ((1.0 - alpha) * span * z) / wage

    exp1 = 1.0 - (1.0 - alpha) * span
    exp2 = (1.0 - alpha) * span

    k_uncon = (aux1 ** exp1 * aux2 ** exp2) ** (1.0 / nu)
    k = k_uncon
    k_cap = lam * a
    if k > k_cap:
        k = k_cap

    # labor from FOC given k
    l = (aux2 * (k ** (alpha * span))) ** (1.0 / exp1)

    y = z * ((k ** alpha) * (l ** (1.0 - alpha))) ** span
    profit = y - wage * l - rental * k
    return profit, k, l, y

@njit(cache=False)
def solve_firm_with_tau(a, z, tau, w, r, lam, delta, alpha, nu):
    """
    Distortion as output wedge: (1-tau)*y - w l - (r+delta)k
    Equivalent to productivity z_eff = (1-tau)*z.
    """
    z_eff = (1.0 - tau) * z
    if z_eff <= 1e-14:
        return -1e18, 0.0, 0.0, 0.0
    return solve_firm_no_tau(a, z_eff, w, r, lam, delta, alpha, nu)

# =============================================================================
# Preferences and Bellman pieces
# =============================================================================
@njit(cache=False)
def utility(c, sigma):
    if c <= 1e-12:
        return -1e18
    if abs(sigma - 1.0) < 1e-10:
        return np.log(c)
    return (c ** (1.0 - sigma) - 1.0) / (1.0 - sigma)

@njit(cache=False)
def find_optimal_savings(income, a_grid, EV_row, beta, sigma, start_idx):
    n_a = len(a_grid)
    best_val = -1e19
    best_idx = start_idx
    for ip in range(start_idx, n_a):
        c = income - a_grid[ip]
        if c <= 1e-12:
            break
        v = utility(c, sigma) + beta * EV_row[ip]
        if v > best_val:
            best_val = v
            best_idx = ip
    return best_val, best_idx

@njit(cache=False, parallel=True)
def bellman_one_step(V_next, a_grid, z_grid, prob_z, income_t, beta, sigma, psi):
    """
    Finite-horizon one-step operator:
      V_t(a,z) = max_{a'} u(income_t(a,z) - a') + beta * E[ V_{t+1}(a', z') | z ]
    Ability transition:
      z' = z with prob psi
      z' ~ prob_z with prob (1-psi)
    """
    n_a = len(a_grid)
    n_z = len(z_grid)

    V_t = np.empty((n_a, n_z), dtype=np.float64)
    pol_idx = np.empty((n_a, n_z), dtype=np.int32)

    # Vbar(a') = sum_{z'} prob_z(z') * V_next(a', z')
    Vbar = np.zeros(n_a, dtype=np.float64)
    for ia in range(n_a):
        s = 0.0
        for iz in range(n_z):
            s += prob_z[iz] * V_next[ia, iz]
        Vbar[ia] = s

    for iz in prange(n_z):
        EV_row = psi * V_next[:, iz] + (1.0 - psi) * Vbar
        start = 0
        for ia in range(n_a):
            v, idx = find_optimal_savings(income_t[ia, iz], a_grid, EV_row, beta, sigma, start)
            V_t[ia, iz] = v
            pol_idx[ia, iz] = idx
            start = idx

    return V_t, pol_idx

@njit(cache=False)
def howard_update(V, pol_idx, a_grid, z_grid, prob_z, income, beta, sigma, psi, n_howard):
    n_a = len(a_grid)
    n_z = len(z_grid)
    for _ in range(n_howard):
        # Vbar
        Vbar = np.zeros(n_a, dtype=np.float64)
        for ia in range(n_a):
            s = 0.0
            for iz in range(n_z):
                s += prob_z[iz] * V[ia, iz]
            Vbar[ia] = s
        V_new = np.empty((n_a, n_z), dtype=np.float64)
        for iz in range(n_z):
            for ia in range(n_a):
                ip = pol_idx[ia, iz]
                c = income[ia, iz] - a_grid[ip]
                cont = psi * V[ip, iz] + (1.0 - psi) * Vbar[ip]
                V_new[ia, iz] = utility(c, sigma) + beta * cont
        V = V_new
    return V

def solve_stationary_value(a_grid, z_grid, prob_z, income, V_init=None, 
                           tol=TOL_VF, max_iter=MAX_ITER_VF_POST, n_howard=N_HOWARD):
    n_a, n_z = len(a_grid), len(z_grid)
    if V_init is None:
        V = np.zeros((n_a, n_z), dtype=np.float64)
        for ia in range(n_a):
            c0 = max(income[ia, 0] - a_grid[0], EPS_INIT_C)
            V[ia, :] = utility(c0, SIGMA) / (1.0 - BETA)
    else:
        V = V_init.copy()

    pol = np.zeros((n_a, n_z), dtype=np.int32)

    for it in range(max_iter):
        V_new, pol_new = bellman_one_step(V, a_grid, z_grid, prob_z, income, BETA, SIGMA, PSI)
        diff = np.max(np.abs(V_new - V))
        pol = pol_new
        # Howard if not too close
        if n_howard > 0 and diff > HOWARD_TOL_MULT * tol:
            V = howard_update(V_new, pol, a_grid, z_grid, prob_z, income, BETA, SIGMA, PSI, n_howard)
        else:
            V = V_new
        if diff < tol:
            break
    return V, pol

# =============================================================================
# Coupled stationary value iteration (pre-reform distortions)
# =============================================================================
@njit(cache=False, parallel=True)
def coupled_bellman_one_step(Vp, Vm, a_grid, z_grid, prob_z, prob_tau_plus, inc_p, inc_m, beta, sigma, psi):
    """
    Bellman operator for two tau states (tau_plus, tau_minus) with persistence driven by ability persistence:
      with prob psi: (z,tau) stays the same
      with prob 1-psi: new z' ~ prob_z, and tau' drawn conditional on z' with prob_tau_plus(z')
    """
    n_a = len(a_grid)
    n_z = len(z_grid)

    Vp_new = np.empty((n_a, n_z), dtype=np.float64)
    Vm_new = np.empty((n_a, n_z), dtype=np.float64)
    polp = np.empty((n_a, n_z), dtype=np.int32)
    polm = np.empty((n_a, n_z), dtype=np.int32)

    # EV_uncond(a') = sum_z' prob_z(z') [p(z')Vp(a',z') + (1-p(z'))Vm(a',z')]
    EV = np.zeros(n_a, dtype=np.float64)
    for ia in range(n_a):
        s = 0.0
        for iz in range(n_z):
            pz = prob_tau_plus[iz]
            s += prob_z[iz] * (pz * Vp[ia, iz] + (1.0 - pz) * Vm[ia, iz])
        EV[ia] = s

    for iz in prange(n_z):
        EVp_row = psi * Vp[:, iz] + (1.0 - psi) * EV
        EVm_row = psi * Vm[:, iz] + (1.0 - psi) * EV
        sp = 0
        sm = 0
        for ia in range(n_a):
            vp, ip = find_optimal_savings(inc_p[ia, iz], a_grid, EVp_row, beta, sigma, sp)
            vm, im = find_optimal_savings(inc_m[ia, iz], a_grid, EVm_row, beta, sigma, sm)
            Vp_new[ia, iz] = vp
            Vm_new[ia, iz] = vm
            polp[ia, iz] = ip
            polm[ia, iz] = im
            sp = ip
            sm = im

    return Vp_new, Vm_new, polp, polm

@njit(cache=False)
def coupled_howard_update(Vp, Vm, polp, polm, a_grid, z_grid, prob_z, prob_tau_plus, inc_p, inc_m, beta, sigma, psi, n_howard):
    n_a = len(a_grid)
    n_z = len(z_grid)
    for _ in range(n_howard):
        EV = np.zeros(n_a, dtype=np.float64)
        for ia in range(n_a):
            s = 0.0
            for iz in range(n_z):
                pz = prob_tau_plus[iz]
                s += prob_z[iz] * (pz * Vp[ia, iz] + (1.0 - pz) * Vm[ia, iz])
            EV[ia] = s

        Vp_new = np.empty((n_a, n_z), dtype=np.float64)
        Vm_new = np.empty((n_a, n_z), dtype=np.float64)
        for iz in range(n_z):
            for ia in range(n_a):
                ipp = polp[ia, iz]
                ipm = polm[ia, iz]
                cp = inc_p[ia, iz] - a_grid[ipp]
                cm = inc_m[ia, iz] - a_grid[ipm]
                cont_p = psi * Vp[ipp, iz] + (1.0 - psi) * EV[ipp]
                cont_m = psi * Vm[ipm, iz] + (1.0 - psi) * EV[ipm]
                Vp_new[ia, iz] = utility(cp, sigma) + beta * cont_p
                Vm_new[ia, iz] = utility(cm, sigma) + beta * cont_m
        Vp = Vp_new
        Vm = Vm_new
    return Vp, Vm

def solve_stationary_value_coupled(a_grid, z_grid, prob_z, prob_tau_plus, inc_p, inc_m,
                                  Vp_init=None, Vm_init=None,
                                  tol=TOL_VF, max_iter=MAX_ITER_VF_PRE, n_howard=N_HOWARD):
    n_a, n_z = len(a_grid), len(z_grid)
    if Vp_init is None:
        Vp = np.zeros((n_a, n_z), dtype=np.float64)
        Vm = np.zeros((n_a, n_z), dtype=np.float64)
        for ia in range(n_a):
            cp0 = max(inc_p[ia, 0] - a_grid[0], EPS_INIT_C)
            cm0 = max(inc_m[ia, 0] - a_grid[0], EPS_INIT_C)
            Vp[ia, :] = utility(cp0, SIGMA) / (1.0 - BETA)
            Vm[ia, :] = utility(cm0, SIGMA) / (1.0 - BETA)
    else:
        Vp = Vp_init.copy()
        Vm = Vm_init.copy()

    polp = np.zeros((n_a, n_z), dtype=np.int32)
    polm = np.zeros((n_a, n_z), dtype=np.int32)

    for it in range(max_iter):
        Vp_new, Vm_new, polp_new, polm_new = coupled_bellman_one_step(
            Vp, Vm, a_grid, z_grid, prob_z, prob_tau_plus,
            inc_p, inc_m, BETA, SIGMA, PSI
        )
        diff = max(np.max(np.abs(Vp_new - Vp)), np.max(np.abs(Vm_new - Vm)))
        polp, polm = polp_new, polm_new

        if n_howard > 0 and diff > HOWARD_TOL_MULT * tol:
            Vp, Vm = coupled_howard_update(Vp_new, Vm_new, polp, polm,
                                           a_grid, z_grid, prob_z, prob_tau_plus,
                                           inc_p, inc_m, BETA, SIGMA, PSI, n_howard)
        else:
            Vp, Vm = Vp_new, Vm_new

        if diff < tol:
            break

    return Vp, Vm, polp, polm

# =============================================================================
# Income precomputation (given prices)
# =============================================================================
@njit(cache=False, parallel=True)
def precompute_income_no_dist(a_grid, z_grid, w, r, lam, delta, alpha, nu):
    n_a = len(a_grid)
    n_z = len(z_grid)
    inc = np.empty((n_a, n_z), dtype=np.float64)
    wage = w if w > 1e-12 else 1e-12
    for iz in prange(n_z):
        z = z_grid[iz]
        for ia in range(n_a):
            a = a_grid[ia]
            p, _, _, _ = solve_firm_no_tau(a, z, w, r, lam, delta, alpha, nu)
            if p > wage:
                inc[ia, iz] = p + (1.0 + r) * a
            else:
                inc[ia, iz] = wage + (1.0 + r) * a
    return inc

@njit(cache=False, parallel=True)
def precompute_income_with_tau(a_grid, z_grid, tau, w, r, lam, delta, alpha, nu):
    n_a = len(a_grid)
    n_z = len(z_grid)
    inc = np.empty((n_a, n_z), dtype=np.float64)
    wage = w if w > 1e-12 else 1e-12
    for iz in prange(n_z):
        z = z_grid[iz]
        for ia in range(n_a):
            a = a_grid[ia]
            p, _, _, _ = solve_firm_with_tau(a, z, tau, w, r, lam, delta, alpha, nu)
            if p > wage:
                inc[ia, iz] = p + (1.0 + r) * a
            else:
                inc[ia, iz] = wage + (1.0 + r) * a
    return inc

# =============================================================================
# Shocks
# =============================================================================
def generate_shocks(n_agents, t_steps, psi, prob_z, seed=1234):
    rng = np.random.default_rng(seed)
    resets = (rng.random((t_steps, n_agents)) > psi).astype(np.uint8)
    cdf = np.cumsum(prob_z)
    shocks = np.searchsorted(cdf, rng.random((t_steps, n_agents))).astype(np.uint16)
    return resets, shocks

def generate_shocks_with_tau(n_agents, t_steps, psi, prob_z, prob_tau_plus, seed=1234):
    resets, shocks = generate_shocks(n_agents, t_steps, psi, prob_z, seed=seed)
    rng = np.random.default_rng(seed + 111)
    tau_shocks = np.empty((t_steps, n_agents), dtype=np.uint8)
    for t in range(t_steps):
        z_new = shocks[t]
        probs = prob_tau_plus[z_new]
        tau_shocks[t] = (rng.random(n_agents) < probs).astype(np.uint8)
    return resets, shocks, tau_shocks

# =============================================================================
# Simulation helpers
# =============================================================================
@njit(cache=False)
def get_interp_weights(x, grid):
    n = len(grid)
    if x <= grid[0]:
        return 0, 0, 1.0
    if x >= grid[-1]:
        return n - 2, n - 1, 0.0
    lo = 0
    hi = n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if grid[mid] > x:
            hi = mid
        else:
            lo = mid
    w_lo = (grid[hi] - x) / (grid[hi] - grid[lo])
    return lo, hi, w_lo

@njit(cache=False, parallel=True)
def simulate_step_nodist(a_curr, z_curr, pol_a_vals, a_grid, reset_t, shock_t):
    """
    Decision uses current (a,z). Next-period z updates via reset process.
    pol_a_vals is policy in asset *levels* (not indices), shape (n_a,n_z).
    """
    n = len(a_curr)
    a_next = np.empty(n, dtype=np.float64)
    z_next = np.empty(n, dtype=np.uint16)
    for i in prange(n):
        zc = z_curr[i]
        z_next[i] = shock_t[i] if reset_t[i] else zc
        il, ih, wl = get_interp_weights(a_curr[i], a_grid)
        a_next[i] = wl * pol_a_vals[il, zc] + (1.0 - wl) * pol_a_vals[ih, zc]
    return a_next, z_next

@njit(cache=False, parallel=True)
def simulate_step_dist(a_curr, z_curr, tau_state, polp_a_vals, polm_a_vals,
                       a_grid, reset_t, shock_t, tau_shock_t):
    """
    tau_state: 1 if tau_plus, 0 if tau_minus.
    tau updates only when reset.
    """
    n = len(a_curr)
    a_next = np.empty(n, dtype=np.float64)
    z_next = np.empty(n, dtype=np.uint16)
    tau_next = np.empty(n, dtype=np.uint8)

    for i in prange(n):
        zc = z_curr[i]
        tc = tau_state[i]
        z_next[i] = shock_t[i] if reset_t[i] else zc
        tau_next[i] = tau_shock_t[i] if reset_t[i] else tc

        il, ih, wl = get_interp_weights(a_curr[i], a_grid)
        if tc == 1:
            a_next[i] = wl * polp_a_vals[il, zc] + (1.0 - wl) * polp_a_vals[ih, zc]
        else:
            a_next[i] = wl * polm_a_vals[il, zc] + (1.0 - wl) * polm_a_vals[ih, zc]

    return a_next, z_next, tau_next

@njit(cache=False)
def fill_mu_az(a_vec, z_vec, a_grid, mu_out):
    """
    Build histogram μ(a,z) on the asset grid using linear weights.
    mu_out is overwritten and normalized to sum to 1.
    Returns mean assets A = E[a] (computed from sample).
    """
    n_a = mu_out.shape[0]
    n_z = mu_out.shape[1]
    # reset
    for ia in range(n_a):
        for iz in range(n_z):
            mu_out[ia, iz] = 0.0

    sA = 0.0
    n = len(a_vec)
    for i in range(n):
        a = a_vec[i]
        sA += a
        iz = z_vec[i]
        il, ih, wl = get_interp_weights(a, a_grid)
        mu_out[il, iz] += wl
        mu_out[ih, iz] += (1.0 - wl)

    inv = 1.0 / n
    for ia in range(n_a):
        for iz in range(n_z):
            mu_out[ia, iz] *= inv

    return sA * inv

@njit(cache=False)
def fill_mu_aztau(a_vec, z_vec, tau_state, a_grid, mu_p_out, mu_m_out):
    """
    Build μ_plus(a,z) and μ_minus(a,z), normalized so μ_plus + μ_minus sums to 1.
    Returns mean assets A.
    """
    n_a = mu_p_out.shape[0]
    n_z = mu_p_out.shape[1]
    for ia in range(n_a):
        for iz in range(n_z):
            mu_p_out[ia, iz] = 0.0
            mu_m_out[ia, iz] = 0.0

    sA = 0.0
    n = len(a_vec)
    for i in range(n):
        a = a_vec[i]
        sA += a
        iz = z_vec[i]
        il, ih, wl = get_interp_weights(a, a_grid)
        if tau_state[i] == 1:
            mu_p_out[il, iz] += wl
            mu_p_out[ih, iz] += (1.0 - wl)
        else:
            mu_m_out[il, iz] += wl
            mu_m_out[ih, iz] += (1.0 - wl)

    inv = 1.0 / n
    for ia in range(n_a):
        for iz in range(n_z):
            mu_p_out[ia, iz] *= inv
            mu_m_out[ia, iz] *= inv

    return sA * inv

# =============================================================================
# Market clearing on binned distributions
# =============================================================================
@njit(cache=False)
def labor_excess_mu_nodist(mu, a_grid, z_grid, w, r, lam, delta, alpha, nu):
    """
    ED_L = Ld - Ls, where Ls = 1 - s_e.
    """
    n_a = len(a_grid)
    n_z = len(z_grid)
    wage = w if w > 1e-12 else 1e-12
    Ld = 0.0
    s_e = 0.0
    for ia in range(n_a):
        a = a_grid[ia]
        for iz in range(n_z):
            wt = mu[ia, iz]
            if wt <= 0.0:
                continue
            z = z_grid[iz]
            p, _, l, _ = solve_firm_no_tau(a, z, w, r, lam, delta, alpha, nu)
            if p > wage:
                Ld += wt * l
                s_e += wt
    Ls = 1.0 - s_e
    return Ld - Ls

@njit(cache=False)
def capital_excess_mu_nodist(mu, A, a_grid, z_grid, w, r, lam, delta, alpha, nu):
    """
    ED_K = Kd - A, where A is fixed capital supply (mean assets from simulation).
    """
    n_a = len(a_grid)
    n_z = len(z_grid)
    wage = w if w > 1e-12 else 1e-12
    Kd = 0.0
    for ia in range(n_a):
        a = a_grid[ia]
        for iz in range(n_z):
            wt = mu[ia, iz]
            if wt <= 0.0:
                continue
            z = z_grid[iz]
            p, k, _, _ = solve_firm_no_tau(a, z, w, r, lam, delta, alpha, nu)
            if p > wage:
                Kd += wt * k
    return Kd - A

@njit(cache=False)
def labor_excess_mu_dist(mu_p, mu_m, a_grid, z_grid, w, r, lam, delta, alpha, nu, tau_p, tau_m):
    wage = w if w > 1e-12 else 1e-12
    n_a = len(a_grid)
    n_z = len(z_grid)
    Ld = 0.0
    s_e = 0.0

    for ia in range(n_a):
        a = a_grid[ia]
        for iz in range(n_z):
            z = z_grid[iz]
            wt_p = mu_p[ia, iz]
            if wt_p > 0.0:
                p, _, l, _ = solve_firm_with_tau(a, z, tau_p, w, r, lam, delta, alpha, nu)
                if p > wage:
                    Ld += wt_p * l
                    s_e += wt_p
            wt_m = mu_m[ia, iz]
            if wt_m > 0.0:
                p, _, l, _ = solve_firm_with_tau(a, z, tau_m, w, r, lam, delta, alpha, nu)
                if p > wage:
                    Ld += wt_m * l
                    s_e += wt_m

    Ls = 1.0 - s_e
    return Ld - Ls

@njit(cache=False)
def capital_excess_mu_dist(mu_p, mu_m, A, a_grid, z_grid, w, r, lam, delta, alpha, nu, tau_p, tau_m):
    wage = w if w > 1e-12 else 1e-12
    n_a = len(a_grid)
    n_z = len(z_grid)
    Kd = 0.0

    for ia in range(n_a):
        a = a_grid[ia]
        for iz in range(n_z):
            z = z_grid[iz]
            wt_p = mu_p[ia, iz]
            if wt_p > 0.0:
                p, k, _, _ = solve_firm_with_tau(a, z, tau_p, w, r, lam, delta, alpha, nu)
                if p > wage:
                    Kd += wt_p * k
            wt_m = mu_m[ia, iz]
            if wt_m > 0.0:
                p, k, _, _ = solve_firm_with_tau(a, z, tau_m, w, r, lam, delta, alpha, nu)
                if p > wage:
                    Kd += wt_m * k

    return Kd - A

@njit(cache=False)
def aggregates_mu_nodist(mu, A, a_grid, z_grid, w, r, lam, delta, alpha, nu):
    """
    Returns K, L, Y, s_e (all per capita; mu sums to 1).
    A is mean assets from simulation (used to report ED_K too).
    """
    wage = w if w > 1e-12 else 1e-12
    n_a = len(a_grid)
    n_z = len(z_grid)
    K = 0.0
    L = 0.0
    Y = 0.0
    s_e = 0.0
    for ia in range(n_a):
        a = a_grid[ia]
        for iz in range(n_z):
            wt = mu[ia, iz]
            if wt <= 0.0:
                continue
            z = z_grid[iz]
            p, k, l, y = solve_firm_no_tau(a, z, w, r, lam, delta, alpha, nu)
            if p > wage:
                K += wt * k
                L += wt * l
                Y += wt * y
                s_e += wt
    return K, L, Y, s_e

@njit(cache=False)
def aggregates_mu_dist(mu_p, mu_m, A, a_grid, z_grid, w, r, lam, delta, alpha, nu, tau_p, tau_m):
    wage = w if w > 1e-12 else 1e-12
    n_a = len(a_grid)
    n_z = len(z_grid)
    K = 0.0
    L = 0.0
    Y = 0.0
    s_e = 0.0
    for ia in range(n_a):
        a = a_grid[ia]
        for iz in range(n_z):
            z = z_grid[iz]
            wt_p = mu_p[ia, iz]
            if wt_p > 0.0:
                p, k, l, y = solve_firm_with_tau(a, z, tau_p, w, r, lam, delta, alpha, nu)
                if p > wage:
                    K += wt_p * k
                    L += wt_p * l
                    Y += wt_p * y
                    s_e += wt_p
            wt_m = mu_m[ia, iz]
            if wt_m > 0.0:
                p, k, l, y = solve_firm_with_tau(a, z, tau_m, w, r, lam, delta, alpha, nu)
                if p > wage:
                    K += wt_m * k
                    L += wt_m * l
                    Y += wt_m * y
                    s_e += wt_m
    return K, L, Y, s_e

@njit(cache=False)
def avg_entrepreneurial_ability_mu(mu, a_grid, z_grid, w, r, lam, delta, alpha, nu):
    """
    Computes the average entrepreneurial ability among active entrepreneurs.
    Returns: (avg_z_entre, share_entre)
    """
    wage = w if w > 1e-12 else 1e-12
    n_a = len(a_grid)
    n_z = len(z_grid)
    sum_z = 0.0
    sum_wt = 0.0
    for ia in range(n_a):
        a = a_grid[ia]
        for iz in range(n_z):
            wt = mu[ia, iz]
            if wt <= 0.0:
                continue
            z = z_grid[iz]
            p, _, _, _ = solve_firm_no_tau(a, z, w, r, lam, delta, alpha, nu)
            if p > wage:
                sum_z += wt * z
                sum_wt += wt
    if sum_wt > 1e-12:
        return sum_z / sum_wt, sum_wt
    return 0.0, 0.0

@njit(cache=False)
def wealth_share_top5_ability_mu(mu, a_grid, z_grid, top5_threshold_idx):
    """
    Computes the wealth share held by agents in the top 5% of the ability distribution.
    top5_threshold_idx: index such that agents with iz >= top5_threshold_idx are in top 5%.
    """
    n_a = len(a_grid)
    n_z = len(z_grid)
    total_wealth = 0.0
    top5_wealth = 0.0
    for ia in range(n_a):
        a = a_grid[ia]
        for iz in range(n_z):
            wt = mu[ia, iz]
            if wt <= 0.0:
                continue
            total_wealth += wt * a
            if iz >= top5_threshold_idx:
                top5_wealth += wt * a
    if total_wealth > 1e-12:
        return top5_wealth / total_wealth
    return 0.0

@njit(cache=False)
def avg_entrepreneurial_ability_mu_dist(mu_p, mu_m, a_grid, z_grid, w, r, lam, delta, alpha, nu, tau_p, tau_m):
    """
    Computes average entrepreneurial ability among active entrepreneurs in distorted economy.
    """
    wage = w if w > 1e-12 else 1e-12
    n_a = len(a_grid)
    n_z = len(z_grid)
    sum_z = 0.0
    sum_wt = 0.0
    for ia in range(n_a):
        a = a_grid[ia]
        for iz in range(n_z):
            z = z_grid[iz]
            wt_p = mu_p[ia, iz]
            if wt_p > 0.0:
                p, _, _, _ = solve_firm_with_tau(a, z, tau_p, w, r, lam, delta, alpha, nu)
                if p > wage:
                    sum_z += wt_p * z
                    sum_wt += wt_p
            wt_m = mu_m[ia, iz]
            if wt_m > 0.0:
                p, _, _, _ = solve_firm_with_tau(a, z, tau_m, w, r, lam, delta, alpha, nu)
                if p > wage:
                    sum_z += wt_m * z
                    sum_wt += wt_m
    if sum_wt > 1e-12:
        return sum_z / sum_wt, sum_wt
    return 0.0, 0.0

@njit(cache=False)
def wealth_share_top5_ability_mu_dist(mu_p, mu_m, a_grid, z_grid, top5_threshold_idx):
    """
    Computes wealth share held by agents in top 5% of ability in distorted economy.
    """
    n_a = len(a_grid)
    n_z = len(z_grid)
    total_wealth = 0.0
    top5_wealth = 0.0
    for ia in range(n_a):
        a = a_grid[ia]
        for iz in range(n_z):
            wt = mu_p[ia, iz] + mu_m[ia, iz]
            if wt <= 0.0:
                continue
            total_wealth += wt * a
            if iz >= top5_threshold_idx:
                top5_wealth += wt * a
    if total_wealth > 1e-12:
        return top5_wealth / total_wealth
    return 0.0

# =============================================================================
# Robust root solves (Python wrappers calling fast Numba ED functions)
# =============================================================================
def _bisect_root(func, lo, hi, max_it=MAX_BISECT_IT, tol_x=EPS_BISECT_TOL):
    f_lo = func(lo)
    f_hi = func(hi)
    if f_lo == 0.0:
        return lo
    if f_hi == 0.0:
        return hi
    # assumes sign change
    for _ in range(max_it):
        mid = 0.5 * (lo + hi)
        f_mid = func(mid)
        if abs(hi - lo) < tol_x:
            return mid
        if f_lo * f_mid <= 0.0:
            hi = mid
            f_hi = f_mid
        else:
            lo = mid
            f_lo = f_mid
    return 0.5 * (lo + hi)

def solve_w_clear_mu_nodist(mu_t, a_grid, z_grid, r_t, w_guess, lam=None):
    """Solve for wage that clears labor market. Uses lam if provided, else LAMBDA.
    
    For neoclassical (large lam), uses finer scan grid since occupational choice
    creates discontinuities in excess demand.
    """
    if lam is None:
        lam = LAMBDA
    w_guess = float(np.clip(w_guess, W_MIN, W_MAX))
    
    # Use finer grid for neoclassical (occupational choice is more sensitive)
    is_neoclassical = (lam is not None and lam > 10.0)
    n_scan = N_SCAN_NC if is_neoclassical else N_SCAN

    def f(w):
        return float(labor_excess_mu_nodist(mu_t, a_grid, z_grid, w, r_t, lam, DELTA, ALPHA, NU))

    # quick check
    f0 = f(w_guess)
    if abs(f0) < EPS_ROOT_CONVERGED:
        return w_guess

    # local bracket (wider for neoclassical)
    bracket_mult = 0.8 if is_neoclassical else 0.6
    lo = max(W_MIN, bracket_mult * w_guess)
    hi = min(W_MAX, (2.0 - bracket_mult) * w_guess)
    f_lo = f(lo)
    f_hi = f(hi)
    if f_lo * f_hi < 0.0:
        return _bisect_root(f, lo, hi)

    # global scan with fine grid
    grid = np.linspace(W_MIN, W_MAX, n_scan)
    vals = np.array([f(x) for x in grid], dtype=np.float64)

    # find sign changes
    sign = np.sign(vals)
    idx = np.where(sign[:-1] * sign[1:] < 0.0)[0]
    if idx.size > 0:
        # choose interval closest to w_guess
        centers = 0.5 * (grid[idx] + grid[idx+1])
        j = int(np.argmin(np.abs(centers - w_guess)))
        lo, hi = float(grid[idx[j]]), float(grid[idx[j]+1])
        return _bisect_root(f, lo, hi)

    # no sign change: return minimizer of |f|
    j = int(np.argmin(np.abs(vals)))
    return float(grid[j])

def _get_r_max(lam):
    """Return appropriate R_MAX based on lambda. Use wider bounds for neoclassical (high lambda)."""
    if lam is not None and lam > 10.0:
        return R_MAX_NC
    return R_MAX

def solve_r_clear_mu_nodist(mu_t, A_t, a_grid, z_grid, w_t, r_guess, lam=None):
    """Solve for interest rate that clears capital market. Uses lam if provided, else LAMBDA."""
    if lam is None:
        lam = LAMBDA
    r_max = _get_r_max(lam)
    r_guess = float(np.clip(r_guess, R_MIN, r_max))

    def f(r):
        return float(capital_excess_mu_nodist(mu_t, A_t, a_grid, z_grid, w_t, r, lam, DELTA, ALPHA, NU))

    f0 = f(r_guess)
    if abs(f0) < EPS_ROOT_CONVERGED:
        return r_guess

    lo = max(R_MIN, r_guess - 0.04)
    hi = min(r_max, r_guess + 0.04)
    f_lo = f(lo)
    f_hi = f(hi)
    if f_lo * f_hi < 0.0:
        return _bisect_root(f, lo, hi)

    grid = np.linspace(R_MIN, r_max, N_SCAN)
    vals = np.array([f(x) for x in grid], dtype=np.float64)

    sign = np.sign(vals)
    idx = np.where(sign[:-1] * sign[1:] < 0.0)[0]
    if idx.size > 0:
        centers = 0.5 * (grid[idx] + grid[idx+1])
        j = int(np.argmin(np.abs(centers - r_guess)))
        lo, hi = float(grid[idx[j]]), float(grid[idx[j]+1])
        return _bisect_root(f, lo, hi)

    j = int(np.argmin(np.abs(vals)))
    return float(grid[j])

def solve_w_clear_mu_dist(mu_p_t, mu_m_t, a_grid, z_grid, r_t, w_guess, lam=None):
    """Solve for wage that clears labor market with distortions. Uses lam if provided, else LAMBDA."""
    if lam is None:
        lam = LAMBDA
    w_guess = float(np.clip(w_guess, W_MIN, W_MAX))

    def f(w):
        return float(labor_excess_mu_dist(mu_p_t, mu_m_t, a_grid, z_grid, w, r_t,
                                         lam, DELTA, ALPHA, NU, TAU_PLUS, TAU_MINUS))

    f0 = f(w_guess)
    if abs(f0) < EPS_ROOT_CONVERGED:
        return w_guess

    lo = max(W_MIN, 0.6 * w_guess)
    hi = min(W_MAX, 1.6 * w_guess)
    f_lo = f(lo)
    f_hi = f(hi)
    if f_lo * f_hi < 0.0:
        return _bisect_root(f, lo, hi)

    grid = np.linspace(W_MIN, W_MAX, N_SCAN)
    vals = np.array([f(x) for x in grid], dtype=np.float64)
    sign = np.sign(vals)
    idx = np.where(sign[:-1] * sign[1:] < 0.0)[0]
    if idx.size > 0:
        centers = 0.5 * (grid[idx] + grid[idx+1])
        j = int(np.argmin(np.abs(centers - w_guess)))
        lo, hi = float(grid[idx[j]]), float(grid[idx[j]+1])
        return _bisect_root(f, lo, hi)
    j = int(np.argmin(np.abs(vals)))
    return float(grid[j])

def solve_r_clear_mu_dist(mu_p_t, mu_m_t, A_t, a_grid, z_grid, w_t, r_guess, lam=None):
    """Solve for interest rate that clears capital market with distortions. Uses lam if provided, else LAMBDA."""
    if lam is None:
        lam = LAMBDA
    r_max = _get_r_max(lam)
    r_guess = float(np.clip(r_guess, R_MIN, r_max))

    def f(r):
        return float(capital_excess_mu_dist(mu_p_t, mu_m_t, A_t, a_grid, z_grid, w_t, r,
                                            lam, DELTA, ALPHA, NU, TAU_PLUS, TAU_MINUS))

    f0 = f(r_guess)
    if abs(f0) < EPS_ROOT_CONVERGED:
        return r_guess

    lo = max(R_MIN, r_guess - 0.04)
    hi = min(r_max, r_guess + 0.04)
    f_lo = f(lo)
    f_hi = f(hi)
    if f_lo * f_hi < 0.0:
        return _bisect_root(f, lo, hi)

    grid = np.linspace(R_MIN, r_max, N_SCAN)
    vals = np.array([f(x) for x in grid], dtype=np.float64)
    sign = np.sign(vals)
    idx = np.where(sign[:-1] * sign[1:] < 0.0)[0]
    if idx.size > 0:
        centers = 0.5 * (grid[idx] + grid[idx+1])
        j = int(np.argmin(np.abs(centers - r_guess)))
        lo, hi = float(grid[idx[j]]), float(grid[idx[j]+1])
        return _bisect_root(f, lo, hi)
    j = int(np.argmin(np.abs(vals)))
    return float(grid[j])

# =============================================================================
# Steady state solvers (B.2-style nested loops)
# =============================================================================
def steady_state_post(a_grid, z_grid, prob_z, N, burn, seed=42, verbose=True, lam=None):
    """
    Post-reform stationary equilibrium (tau=0) using nested loops:
      - inner loop solves w given r using labor clearing varpi
      - outer loop updates r given w using capital clearing iota

    lam: collateral constraint parameter. If None, uses LAMBDA (1.35).
         Set to large value (e.g., 1e6) for neoclassical (no collateral constraint).
    """
    if lam is None:
        lam = LAMBDA

    resets, shocks = generate_shocks(N, burn, PSI, prob_z, seed=seed)

    # initial guesses
    w = 0.8
    r = -0.04

    V = None
    pol = None

    mu = np.empty((len(a_grid), len(z_grid)), dtype=np.float64)

    # to check grid truncation
    top_mass = 0.0

    for outer in range(SS_POST_MAX_OUTER):
        r_old = r

        # ---------- inner wage loop ----------
        for inner in range(SS_POST_MAX_INNER):
            w_old = w

            inc = precompute_income_no_dist(a_grid, z_grid, w, r, lam, DELTA, ALPHA, NU)
            V, pol = solve_stationary_value(a_grid, z_grid, prob_z, inc, V_init=V)

            pol_a = a_grid[pol]  # levels

            # simulate burn-in
            rng = np.random.default_rng(1000)  # common seed across iterations
            a = np.ones(N, dtype=np.float64) * a_grid[0]
            # draw initial z from stationary marginal prob_z (faster burn-in)
            cdf_z = np.cumsum(prob_z)
            z = np.searchsorted(cdf_z, rng.random(N)).astype(np.uint16)
            for t in range(burn):
                a, z = simulate_step_nodist(a, z, pol_a, a_grid, resets[t], shocks[t])

            # build μ(a,z)
            A = fill_mu_az(a, z, a_grid, mu)

            # diagnostic: mass at top bin (proxy for truncation)
            top_mass = float(mu[-1, :].sum())

            # labor clearing wage
            w_clear = solve_w_clear_mu_nodist(mu, a_grid, z_grid, r, w, lam=lam)
            w = ETA_W * w_clear + (1.0 - ETA_W) * w
            w = float(np.clip(w, W_MIN, W_MAX))

            if verbose and (inner < 2 or inner % 5 == 0):
                edL = float(labor_excess_mu_nodist(mu, a_grid, z_grid, w, r, lam, DELTA, ALPHA, NU))
                print(f"[POST SS | outer {outer+1:02d} wage {inner+1:02d}] w={w:.6f}  |Δw|={abs(w-w_old):.2e}  ED_L={edL:.3e}")

            if abs(w - w_old) < SS_POST_TOL_W:
                break

        # ---------- r update (capital clearing) ----------
        r_clear = solve_r_clear_mu_nodist(mu, A, a_grid, z_grid, w, r, lam=lam)
        r = ETA_R * r_clear + (1.0 - ETA_R) * r
        r = float(np.clip(r, R_MIN, _get_r_max(lam)))

        edL = float(labor_excess_mu_nodist(mu, a_grid, z_grid, w, r, lam, DELTA, ALPHA, NU))
        edK = float(capital_excess_mu_nodist(mu, A, a_grid, z_grid, w, r, lam, DELTA, ALPHA, NU))
        if verbose:
            print(f"[POST SS | outer {outer+1:02d}] r={r:.6f}  |Δr|={abs(r-r_old):.2e}  ED_L={edL:.3e}  ED_K={edK:.3e}\n")

        # Use relaxed tolerances for neoclassical (discrete occ choice can prevent exact clearing)
        is_neoclassical = (lam is not None and lam > 10.0)
        tol_edL = SS_NC_TOL_ED_L if is_neoclassical else SS_POST_TOL_ED_L
        tol_edK = SS_NC_TOL_ED_K if is_neoclassical else SS_POST_TOL_ED_K
        
        if abs(r - r_old) < SS_POST_TOL_R and abs(edL) < tol_edL and abs(edK) < tol_edK:
            break

    # aggregates at SS
    K, L, Y, s_e = aggregates_mu_nodist(mu, A, a_grid, z_grid, w, r, lam, DELTA, ALPHA, NU)
    span = 1.0 - NU
    Ls = 1.0 - s_e
    TFP = Y / max(((K ** ALPHA) * (Ls ** (1.0 - ALPHA))) ** span, EPS_TFP_DENOM)

    # Micro metrics
    n_z = len(z_grid)
    top5_idx = int(0.95 * n_z)
    avg_z, _ = avg_entrepreneurial_ability_mu(mu, a_grid, z_grid, w, r, lam, DELTA, ALPHA, NU)
    ws_top5 = wealth_share_top5_ability_mu(mu, a_grid, z_grid, top5_idx)

    if verbose and top_mass > EPS_TOP_MASS_WARN:
        print(f"WARNING: mass at top asset bin is {top_mass:.3e}. Consider increasing --amax or --na.\n")

    return dict(w=w, r=r, V=V, pol=pol, mu=mu.copy(), A=A, Y=Y, K=K, L=L, s_e=s_e, TFP=TFP,
                avg_z_entre=avg_z, wealth_share_top5=ws_top5, lam=lam)

def steady_state_pre(a_grid, z_grid, prob_z, prob_tau_plus, N, burn, seed=77, verbose=True, lam=None):
    """
    Pre-reform stationary equilibrium with distortions (tau_plus/tau_minus).
    Uses the same nested-loop structure as post SS.
    Returns also a sample cross-section (a,z) to initialize transition at t=0.

    lam: collateral constraint parameter. If None, uses LAMBDA (1.35).
         Set to large value (e.g., 1e6) for neoclassical (no collateral constraint).
    """
    if lam is None:
        lam = LAMBDA

    resets, shocks, tau_shocks = generate_shocks_with_tau(N, burn, PSI, prob_z, prob_tau_plus, seed=seed)

    # initial guesses close to post SS but lower wage
    w = 0.55
    r = -0.05

    Vp = None
    Vm = None
    polp = None
    polm = None

    mu_p = np.empty((len(a_grid), len(z_grid)), dtype=np.float64)
    mu_m = np.empty((len(a_grid), len(z_grid)), dtype=np.float64)

    for outer in range(SS_PRE_MAX_OUTER):
        r_old = r

        # ---------- inner wage loop ----------
        for inner in range(SS_PRE_MAX_INNER):
            w_old = w

            inc_p = precompute_income_with_tau(a_grid, z_grid, TAU_PLUS,  w, r, lam, DELTA, ALPHA, NU)
            inc_m = precompute_income_with_tau(a_grid, z_grid, TAU_MINUS, w, r, lam, DELTA, ALPHA, NU)

            Vp, Vm, polp, polm = solve_stationary_value_coupled(
                a_grid, z_grid, prob_z, prob_tau_plus, inc_p, inc_m,
                Vp_init=Vp, Vm_init=Vm
            )

            polp_a = a_grid[polp]
            polm_a = a_grid[polm]

            # simulate burn-in with tau state
            rng = np.random.default_rng(2000)  # common seed across iterations
            a = np.ones(N, dtype=np.float64) * a_grid[0]
            # draw initial z from stationary marginal prob_z (faster burn-in)
            cdf_z = np.cumsum(prob_z)
            z = np.searchsorted(cdf_z, rng.random(N)).astype(np.uint16)
            u = rng.random(N)
            tau_state = (u < prob_tau_plus[z]).astype(np.uint8)

            for t in range(burn):
                a, z, tau_state = simulate_step_dist(a, z, tau_state, polp_a, polm_a,
                                                     a_grid, resets[t], shocks[t], tau_shocks[t])

            A = fill_mu_aztau(a, z, tau_state, a_grid, mu_p, mu_m)

            w_clear = solve_w_clear_mu_dist(mu_p, mu_m, a_grid, z_grid, r, w, lam=lam)
            w = ETA_W * w_clear + (1.0 - ETA_W) * w
            w = float(np.clip(w, W_MIN, W_MAX))

            if verbose and (inner < 2 or inner % 6 == 0):
                edL = float(labor_excess_mu_dist(mu_p, mu_m, a_grid, z_grid, w, r,
                                                lam, DELTA, ALPHA, NU, TAU_PLUS, TAU_MINUS))
                print(f"[PRE  SS | outer {outer+1:02d} wage {inner+1:02d}] w={w:.6f}  |Δw|={abs(w-w_old):.2e}  ED_L={edL:.3e}")

            if abs(w - w_old) < SS_PRE_TOL_W:
                break

        r_clear = solve_r_clear_mu_dist(mu_p, mu_m, A, a_grid, z_grid, w, r, lam=lam)
        r = ETA_R * r_clear + (1.0 - ETA_R) * r
        r = float(np.clip(r, R_MIN, _get_r_max(lam)))

        edL = float(labor_excess_mu_dist(mu_p, mu_m, a_grid, z_grid, w, r,
                                        lam, DELTA, ALPHA, NU, TAU_PLUS, TAU_MINUS))
        edK = float(capital_excess_mu_dist(mu_p, mu_m, A, a_grid, z_grid, w, r,
                                          lam, DELTA, ALPHA, NU, TAU_PLUS, TAU_MINUS))
        if verbose:
            print(f"[PRE  SS | outer {outer+1:02d}] r={r:.6f}  |Δr|={abs(r-r_old):.2e}  ED_L={edL:.3e}  ED_K={edK:.3e}\n")

        # Use relaxed tolerances for neoclassical
        is_neoclassical = (lam is not None and lam > 10.0)
        tol_edL = SS_NC_TOL_ED_L if is_neoclassical else SS_PRE_TOL_ED_L
        tol_edK = SS_NC_TOL_ED_K if is_neoclassical else SS_PRE_TOL_ED_K
        
        if abs(r - r_old) < SS_PRE_TOL_R and abs(edL) < tol_edL and abs(edK) < tol_edK:
            break

    # aggregates at SS
    K, L, Y, s_e = aggregates_mu_dist(mu_p, mu_m, A, a_grid, z_grid, w, r,
                                     lam, DELTA, ALPHA, NU, TAU_PLUS, TAU_MINUS)
    span = 1.0 - NU
    Ls = 1.0 - s_e
    TFP = Y / max(((K ** ALPHA) * (Ls ** (1.0 - ALPHA))) ** span, EPS_TFP_DENOM)

    # Micro metrics
    n_z = len(z_grid)
    top5_idx = int(0.95 * n_z)
    avg_z, _ = avg_entrepreneurial_ability_mu_dist(mu_p, mu_m, a_grid, z_grid, w, r,
                                                    lam, DELTA, ALPHA, NU, TAU_PLUS, TAU_MINUS)
    ws_top5 = wealth_share_top5_ability_mu_dist(mu_p, mu_m, a_grid, z_grid, top5_idx)

    # Also return a sample cross-section for transition initial condition (a,z)
    # Use the last simulated (a,z) from the last iteration:
    return dict(w=w, r=r, a=a.copy(), z=z.copy(),
                Vp=Vp, Vm=Vm, polp=polp, polm=polm,
                mu_p=mu_p.copy(), mu_m=mu_m.copy(), A=A,
                Y=Y, K=K, L=L, s_e=s_e, TFP=TFP,
                avg_z_entre=avg_z, wealth_share_top5=ws_top5, lam=lam)

# =============================================================================
# Transition path solver (Algorithm B.2)
# =============================================================================
def solve_transition_B2(pre_eq, post_eq, a_grid, z_grid, prob_z, T, N, seed=2026, verbose=True, lam=None):
    """
    Implements Appendix Algorithm B.2:
      outer loop on r_path
        inner loop on w_path given r_path:
          backward induction -> policies
          forward simulation -> μ_t
          per-period labor clearing -> varpi_t
          relax update w_path
        per-period capital clearing -> iota_t
        relax update r_path
    Terminal condition: w_T = w_post, r_T = r_post, V_T = V_post.

    lam: collateral constraint parameter. If None, uses LAMBDA. Set to large value (e.g., 1e6) for neoclassical.
    """
    if lam is None:
        lam = LAMBDA
    n_a, n_z = len(a_grid), len(z_grid)

    # Index for top 5% ability (95th percentile)
    top5_idx = int(0.95 * n_z)

    # fixed shocks (common random numbers)
    resets, shocks = generate_shocks(N, T, PSI, prob_z, seed=seed)

    # initial distribution at t=0
    a0 = pre_eq["a"].astype(np.float64, copy=False)
    z0 = pre_eq["z"].astype(np.uint16, copy=False)

    w_pre, r_pre = pre_eq["w"], pre_eq["r"]
    w_post, r_post = post_eq["w"], post_eq["r"]

    # price paths with terminal condition at t=T
    w_path = np.linspace(w_pre, w_post, T + 1).astype(np.float64)
    r_path = np.linspace(r_pre, r_post, T + 1).astype(np.float64)
    w_path[T] = w_post
    r_path[T] = r_post

    V_T = post_eq["V"].astype(np.float64, copy=False)

    # storage (filled in last simulation of each wage-iteration)
    mu_path = np.empty((T, n_a, n_z), dtype=np.float64)
    A_path = np.empty(T, dtype=np.float64)

    for outer in range(MAX_OUTER):
        r_old = r_path.copy()

        # =========================
        # Inner loop on wage path
        # =========================
        for inner in range(MAX_W_INNER):
            w_old = w_path.copy()

            # backward induction: store policy indices for t=0..T-1
            pol_idx_path = np.empty((T, n_a, n_z), dtype=np.int32)
            V_next = V_T.copy()

            for tt in range(T-1, -1, -1):
                inc_t = precompute_income_no_dist(a_grid, z_grid, float(w_path[tt]), float(r_path[tt]),
                                                  lam, DELTA, ALPHA, NU)
                V_t, pol_t = bellman_one_step(V_next, a_grid, z_grid, prob_z, inc_t,
                                              BETA, SIGMA, PSI)
                pol_idx_path[tt] = pol_t
                V_next = V_t

            # Convert policies to levels once
            pol_a_path = a_grid[pol_idx_path]  # shape (T,n_a,n_z)

            # forward simulate and build μ_t (no storing massive cross-sections)
            a = a0.copy()
            z = z0.copy()

            mu_tmp = np.empty((n_a, n_z), dtype=np.float64)

            for tt in range(T):
                # histogram of current distribution
                A_path[tt] = fill_mu_az(a, z, a_grid, mu_tmp)
                mu_path[tt, :, :] = mu_tmp  # copy 20k floats
                # step forward
                a, z = simulate_step_nodist(a, z, pol_a_path[tt], a_grid, resets[tt], shocks[tt])

            # per-period labor-clearing wage varpi_t (for t=0..T-1)
            w_clear = np.empty(T, dtype=np.float64)
            # warm start sequentially (w is smooth)
            wg = float(w_path[0])
            for tt in range(T):
                wg = solve_w_clear_mu_nodist(mu_path[tt], a_grid, z_grid, float(r_path[tt]), wg, lam=lam)
                w_clear[tt] = wg

            # relax update (do not touch terminal)
            w_path[:T] = ETA_W * w_clear + (1.0 - ETA_W) * w_path[:T]
            w_path[:T] = np.clip(w_path[:T], W_MIN, W_MAX)
            w_path[T] = w_post

            w_diff = float(np.max(np.abs(w_path - w_old)))
            if verbose and (inner < 2 or inner % 3 == 0):
                print(f"[outer {outer+1:02d} | wage {inner+1:02d}] max|Δw|={w_diff:.3e}")
            if w_diff < TOL_W_SEQ:
                break

        # =========================
        # Capital clearing sequence iota_t (t=0..T-1)
        # =========================
        r_clear = np.empty(T, dtype=np.float64)
        rg = float(r_path[0])
        for tt in range(T):
            rg = solve_r_clear_mu_nodist(mu_path[tt], float(A_path[tt]), a_grid, z_grid, float(w_path[tt]), rg, lam=lam)
            r_clear[tt] = rg

        # relax update
        r_path[:T] = ETA_R * r_clear + (1.0 - ETA_R) * r_path[:T]
        r_path[:T] = np.clip(r_path[:T], R_MIN, _get_r_max(lam))
        r_path[T] = r_post

        r_diff = float(np.max(np.abs(r_path - r_old)))

        # market-clearing gaps on the *fixed μ_t* from the last simulation
        # (these go to zero when the fixed point is reached)
        w_gap = float(np.max(np.abs(w_clear - w_path[:T])))
        r_gap = float(np.max(np.abs(r_clear - r_path[:T])))

        if verbose:
            print(f"[outer {outer+1:02d}] max|Δr|={r_diff:.3e}  max|gap_w|={w_gap:.3e}  max|gap_r|={r_gap:.3e}\n")

        if r_diff < TOL_R_SEQ and w_gap < 5 * TOL_W_SEQ and r_gap < 5 * TOL_R_SEQ:
            # extra safeguard: check residual ED using current prices with the *same μ_t*
            # (should already be tiny at this point)
            max_edL = 0.0
            max_edK = 0.0
            for tt in range(T):
                edL = abs(float(labor_excess_mu_nodist(mu_path[tt], a_grid, z_grid, float(w_path[tt]), float(r_path[tt]),
                                                     lam, DELTA, ALPHA, NU)))
                edK = abs(float(capital_excess_mu_nodist(mu_path[tt], float(A_path[tt]), a_grid, z_grid,
                                                       float(w_path[tt]), float(r_path[tt]),
                                                       lam, DELTA, ALPHA, NU)))
                if edL > max_edL:
                    max_edL = edL
                if edK > max_edK:
                    max_edK = edK
            if verbose:
                print(f"  residual (fixed-μ) max|ED_L|={max_edL:.3e} max|ED_K|={max_edK:.3e}\n")
            if max_edL < TOL_ED_L and max_edK < TOL_ED_K:
                if verbose:
                    print(f"CONVERGED outer loop after {outer+1} iterations.\n")
                break

    # =========================
    # Final evaluation: recompute policies & simulate once more using final sequences
    # =========================
    pol_idx_path = np.empty((T, n_a, n_z), dtype=np.int32)
    V_next = V_T.copy()
    for tt in range(T-1, -1, -1):
        inc_t = precompute_income_no_dist(a_grid, z_grid, float(w_path[tt]), float(r_path[tt]),
                                          lam, DELTA, ALPHA, NU)
        V_t, pol_t = bellman_one_step(V_next, a_grid, z_grid, prob_z, inc_t, BETA, SIGMA, PSI)
        pol_idx_path[tt] = pol_t
        V_next = V_t
    pol_a_path = a_grid[pol_idx_path]

    a = a0.copy()
    z = z0.copy()

    mu_tmp = np.empty((n_a, n_z), dtype=np.float64)
    mu_path_final = np.empty((T, n_a, n_z), dtype=np.float64)
    A_path_final = np.empty(T, dtype=np.float64)

    for tt in range(T):
        A_path_final[tt] = fill_mu_az(a, z, a_grid, mu_tmp)
        mu_path_final[tt] = mu_tmp
        a, z = simulate_step_nodist(a, z, pol_a_path[tt], a_grid, resets[tt], shocks[tt])

    # aggregates and ED
    Y = np.empty(T, dtype=np.float64)
    K = np.empty(T, dtype=np.float64)
    L = np.empty(T, dtype=np.float64)
    s_e = np.empty(T, dtype=np.float64)
    TFP = np.empty(T, dtype=np.float64)
    ED_L = np.empty(T, dtype=np.float64)
    ED_K = np.empty(T, dtype=np.float64)
    avg_z_entre = np.empty(T, dtype=np.float64)
    wealth_share_top5 = np.empty(T, dtype=np.float64)

    span = 1.0 - NU
    for tt in range(T):
        Kt, Lt, Yt, se = aggregates_mu_nodist(mu_path_final[tt], float(A_path_final[tt]),
                                             a_grid, z_grid, float(w_path[tt]), float(r_path[tt]),
                                             lam, DELTA, ALPHA, NU)
        Y[tt] = Yt
        K[tt] = Kt
        L[tt] = Lt
        s_e[tt] = se
        Ls = 1.0 - se
        TFP[tt] = Yt / max(((Kt ** ALPHA) * (Ls ** (1.0 - ALPHA))) ** span, EPS_TFP_DENOM)
        ED_L[tt] = Lt - Ls
        ED_K[tt] = Kt - A_path_final[tt]

        # New metrics: average entrepreneurial ability and wealth share of top 5% ability
        avg_z, _ = avg_entrepreneurial_ability_mu(mu_path_final[tt], a_grid, z_grid,
                                                   float(w_path[tt]), float(r_path[tt]),
                                                   lam, DELTA, ALPHA, NU)
        avg_z_entre[tt] = avg_z
        wealth_share_top5[tt] = wealth_share_top5_ability_mu(mu_path_final[tt], a_grid, z_grid, top5_idx)

    return dict(t=np.arange(T+1), w=w_path, r=r_path,
                Y=Y, K=K, L=L, A=A_path_final, s_e=s_e, TFP=TFP,
                ED_L=ED_L, ED_K=ED_K,
                avg_z_entre=avg_z_entre, wealth_share_top5=wealth_share_top5)

# =============================================================================
# Data Saving
# =============================================================================
def plot_asset_grid_distribution(a_grid, outdir):
    """
    Plots the distribution of asset grid points to visualize the power spacing.
    """
    os.makedirs(outdir, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. Index vs Value (Shows the curvature/power function)
    ax1.plot(np.arange(len(a_grid)), a_grid, 'b.-', markersize=3)
    ax1.set_title("Grid Spacing: Index vs Value")
    ax1.set_xlabel("Grid Index i")
    ax1.set_ylabel("Asset Value a[i]")
    ax1.grid(True, alpha=0.3)
    
    # 2. Histogram of point density (log scale to see small bins)
    # or just plotting the points on a line
    ax2.scatter(a_grid, np.zeros_like(a_grid), alpha=0.5, s=10, marker='|')
    ax2.set_title("Grid Point Locations (Density)")
    ax2.set_xlabel("Asset Value a")
    ax2.set_yticks([])
    ax2.set_xlim(0, 50) # Zoom in on the bottom end where density is high
    ax2.text(0.95, 0.95, "Zoomed to a < 50", transform=ax2.transAxes, ha='right', va='top')
    
    plt.tight_layout()
    path = os.path.join(outdir, "asset_grid_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved grid distribution plot: {path}")

def save_steady_states(pre_eq, post_eq, a_grid, z_grid, outdir, filename="steady_states.npz"):
    """
    Saves steady state results (for policy plotting).

    filename: output filename (default "steady_states.npz", use "steady_states_neoclassical.npz" for λ=∞)
    """
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, filename)

    # Extract policies (unpack dictionaries if needed or save whole dict)
    # We save specific arrays needed for plotting
    np.savez_compressed(path,
                        # Grids
                        a_grid=a_grid,
                        z_grid=z_grid,
                        # Post-Reform
                        post_w=post_eq['w'],
                        post_r=post_eq['r'],
                        post_pol=post_eq['pol'], # indices
                        post_mu=post_eq['mu'],
                        # Pre-Reform
                        pre_w=pre_eq['w'],
                        pre_r=pre_eq['r'],
                        pre_polp=pre_eq['polp'],
                        pre_polm=pre_eq['polm'],
                        pre_mu_p=pre_eq['mu_p'],
                        pre_mu_m=pre_eq['mu_m'],
                        # Lambda used
                        lam=post_eq.get('lam', LAMBDA),
                        )
    print(f"Saved steady state results to: {path}")

def save_transition_path(trans, pre_eq, post_eq, outdir, filename="transition_path.npz"):
    """
    Saves transition path simulation results.
    """
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, filename)

    np.savez_compressed(path,
                        t=trans['t'],
                        Y=trans['Y'],
                        K=trans['K'],
                        L=trans['L'],
                        TFP=trans['TFP'],
                        w=trans['w'],
                        r=trans['r'],
                        ED_L=trans['ED_L'],
                        ED_K=trans['ED_K'],
                        # New metrics
                        avg_z_entre=trans.get('avg_z_entre', np.array([])),
                        wealth_share_top5=trans.get('wealth_share_top5', np.array([])),
                        # Save pre-SS levels for normalization
                        pre_Y=pre_eq['Y'],
                        pre_K=pre_eq['K'],
                        pre_L=pre_eq['L'],
                        pre_TFP=pre_eq['TFP'],
                        pre_w=pre_eq['w'],
                        pre_r=pre_eq['r'],
                        pre_avg_z_entre=pre_eq.get('avg_z_entre', 0.0),
                        pre_wealth_share_top5=pre_eq.get('wealth_share_top5', 0.0),
                        # Save post-SS levels for reference lines in plots
                        post_Y=post_eq['Y'],
                        post_K=post_eq['K'],
                        post_L=post_eq['L'],
                        post_TFP=post_eq['TFP'],
                        post_w=post_eq['w'],
                        post_r=post_eq['r'],
                        post_avg_z_entre=post_eq.get('avg_z_entre', 0.0),
                        post_wealth_share_top5=post_eq.get('wealth_share_top5', 0.0),
                        # Lambda used for this transition
                        lam=post_eq.get('lam', LAMBDA),
                        )
    print(f"Saved transition path results to: {path}")

# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=125, help="Horizon length (paper uses 125)")
    parser.add_argument("--N", type=int, default=350000, help="Number of simulated agents")
    parser.add_argument("--burn", type=int, default=500, help="Burn-in length for steady states")
    parser.add_argument("--na", type=int, default=1201, help="Asset grid points")
    parser.add_argument("--amax", type=float, default=300.0, help="Asset grid max")
    parser.add_argument("--out", type=str, default="outputs_B2_fast", help="Output directory")
    parser.add_argument("--quiet", action="store_true", help="Reduce prints")
    
    # Economy selection flags
    parser.add_argument("--frictions", action="store_true", 
                        help="Run ONLY the benchmark economy with financial frictions (λ=1.35)")
    parser.add_argument("--neoclassical", action="store_true", 
                        help="Run ONLY the neoclassical economy (λ=∞, no collateral constraints)")
    parser.add_argument("--lam-nc", type=float, default=None,
                        help="Override neoclassical λ value (default: 1e6). Try smaller values like 100 or 1000 for debugging.")
    
    args = parser.parse_args()

    verbose = not args.quiet
    os.makedirs(args.out, exist_ok=True)
    
    # Determine which economies to run
    run_frictions = True
    run_neoclassical = True
    
    if args.frictions and args.neoclassical:
        print("Both --frictions and --neoclassical specified, running both economies.")
    elif args.frictions:
        run_frictions = True
        run_neoclassical = False
        print("Running ONLY benchmark economy with financial frictions (λ=1.35)")
    elif args.neoclassical:
        run_frictions = False
        run_neoclassical = True
        print("Running ONLY neoclassical economy (λ=∞)")
    
    # Override neoclassical λ if specified
    lam_nc = args.lam_nc if args.lam_nc is not None else LAM_NEOCLASSICAL
    if args.lam_nc is not None:
        print(f"Using custom neoclassical λ = {lam_nc:.2e}")

    print("Building grids...")
    z_grid, prob_z = create_ability_grid_paper(ETA, n_z=Z_GRID_SIZE, zmax_target=ZMAX_TARGET)
    a_grid = create_asset_grid(args.na, A_MIN_FLOOR, args.amax, power=A_GRID_POWER)
    prob_tau_plus = compute_tau_probs(z_grid, Q_DIST)

    # Plot grid immediately
    plot_asset_grid_distribution(a_grid, args.out)

    # =========================================================================
    # BENCHMARK ECONOMY (λ = 1.35, with collateral constraints)
    # =========================================================================
    if run_frictions:
        print(f"\n{'='*70}")
        print(f"BENCHMARK ECONOMY (λ = {LAMBDA})")
        print(f"{'='*70}")

        print(f"\nSTEP 1a: Post-reform steady state (no distortions, λ={LAMBDA})")
        post = steady_state_post(a_grid, z_grid, prob_z, N=args.N, burn=args.burn, verbose=verbose, lam=LAMBDA)
        print(f"Post SS (λ={LAMBDA}): w={post['w']:.6f}, r={post['r']:.6f}, ED check: |K-A|={abs(post['K']-post['A']):.3e}\n")

        print(f"STEP 1b: Pre-reform steady state (distortions + λ={LAMBDA}) — initial distribution at t=0")
        pre = steady_state_pre(a_grid, z_grid, prob_z, prob_tau_plus, N=args.N, burn=args.burn, verbose=verbose, lam=LAMBDA)
        print(f"Pre  SS (λ={LAMBDA}): w={pre['w']:.6f}, r={pre['r']:.6f}, ED check approx: |K-A|={abs(pre['K']-pre['A']):.3e}\n")

        # Save steady states immediately as requested
        save_steady_states(pre, post, a_grid, z_grid, args.out)

        print(f"STEP 1c: Transition path (reform at t=0, λ={LAMBDA})")
        trans = solve_transition_B2(pre, post, a_grid, z_grid, prob_z, T=args.T, N=args.N, verbose=verbose, lam=LAMBDA)

        # final diagnostics
        max_edL = float(np.max(np.abs(trans["ED_L"])))
        max_edK = float(np.max(np.abs(trans["ED_K"])))
        print(f"Transition diagnostics (λ={LAMBDA}): max|ED_L|={max_edL:.3e}, max|ED_K|={max_edK:.3e}")

        # Save transition results
        save_transition_path(trans, pre, post, args.out, filename="transition_path.npz")

    # =========================================================================
    # NEOCLASSICAL ECONOMY (λ = ∞, no collateral constraints)
    # =========================================================================
    if run_neoclassical:
        print(f"\n{'='*70}")
        print(f"NEOCLASSICAL ECONOMY (λ = {lam_nc:.0e})")
        print(f"{'='*70}")

        print(f"\nSTEP 2a: Neoclassical post-reform steady state (no distortions, λ={lam_nc:.0e})...")
        post_nc = steady_state_post(a_grid, z_grid, prob_z, N=args.N, burn=args.burn, verbose=verbose, lam=lam_nc)
        print(f"Post SS (λ={lam_nc:.0e}): w={post_nc['w']:.6f}, r={post_nc['r']:.6f}, ED check: |K-A|={abs(post_nc['K']-post_nc['A']):.3e}\n")

        print(f"STEP 2b: Neoclassical pre-reform steady state (distortions + λ={lam_nc:.0e}) — initial distribution at t=0")
        pre_nc = steady_state_pre(a_grid, z_grid, prob_z, prob_tau_plus, N=args.N, burn=args.burn, verbose=verbose, lam=lam_nc)
        print(f"Pre  SS (λ={lam_nc:.0e}): w={pre_nc['w']:.6f}, r={pre_nc['r']:.6f}, ED check approx: |K-A|={abs(pre_nc['K']-pre_nc['A']):.3e}\n")

        # Save neoclassical steady states
        save_steady_states(pre_nc, post_nc, a_grid, z_grid, args.out, filename="steady_states_neoclassical.npz")

        print(f"STEP 2c: Neoclassical transition (reform at t=0, λ={lam_nc:.0e})")
        trans_nc = solve_transition_B2(pre_nc, post_nc, a_grid, z_grid, prob_z,
                                        T=args.T, N=args.N, verbose=verbose, lam=lam_nc)

        max_edL_nc = float(np.max(np.abs(trans_nc["ED_L"])))
        max_edK_nc = float(np.max(np.abs(trans_nc["ED_K"])))
        print(f"Neoclassical diagnostics (λ={lam_nc:.0e}): max|ED_L|={max_edL_nc:.3e}, max|ED_K|={max_edK_nc:.3e}")

        # Save neoclassical transition (use pre_nc for proper normalization)
        save_transition_path(trans_nc, pre_nc, post_nc, args.out, filename="transition_neoclassical.npz")

    print("\nAll results saved to:", args.out)

if __name__ == "__main__":
    main()

