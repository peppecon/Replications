
"""
Buera & Shin (2010) — Simulation-based transition dynamics (Appendix Algorithm B.2)

This script implements:

1) Post-reform stationary steady state (no distortions)
2) Pre-reform stationary steady state (with idiosyncratic output wedges, correlated with ability)
3) Transition path after reform using Appendix Algorithm B.2:
   - Outer loop on interest-rate sequence {r_t}
   - Inner loop on wage sequence {w_t} conditional on {r_t}
   - Backward induction for policy functions, forward Monte Carlo simulation
   - Per-period static market clearing (labor -> varpi_t, capital -> iota_t) holding simulated assets fixed
   - Relaxation updates with (eta_w, eta_r)

Notes:
- The economy after reform (t>=0) has NO distortions, but the initial distribution at t=0
  comes from the distorted pre-reform steady state.
- Uses common random numbers (fixed shocks) during the transition iterations to reduce noise and
  produce smooth paths as in the paper.

Run:
  python buera_shin_transition_B2.py --T 125 --na 501 --N 350000

If you want a fast debug run:
  python buera_shin_transition_B2.py --T 40 --na 201 --N 40000 --Tss 200

Author: ChatGPT (GPT-5.2 Pro)
"""

import os
import time
import argparse
import warnings
import numpy as np
import matplotlib.pyplot as plt
# --- Compatibility patch: coverage>=7.5 removed some attributes numba expects ---
try:
    import coverage  # type: ignore
    if hasattr(coverage, 'types'):
        if (not hasattr(coverage.types, 'Tracer')) and hasattr(coverage.types, 'TracerCore'):
            coverage.types.Tracer = coverage.types.TracerCore  # type: ignore
        if not hasattr(coverage.types, 'TShouldTraceFn'):
            coverage.types.TShouldTraceFn = object  # type: ignore
        if not hasattr(coverage.types, 'TShouldStartContextFn'):
            coverage.types.TShouldStartContextFn = object  # type: ignore
except Exception:
    pass
from numba import njit, prange, set_num_threads

warnings.filterwarnings("ignore")

# =============================================================================
# Calibration (paper baseline)
# =============================================================================
SIGMA = 1.5
BETA  = 0.904
ALPHA = 0.33
NU    = 0.21           # "upsilon" in our notation
DELTA = 0.06
ETA_PARETO = 4.15
PSI   = 0.894

# Financial friction
LAMBDA = 1.35

# Pre-reform distortions (only for initial distribution)
TAU_PLUS  = 0.57
TAU_MINUS = -0.15
Q_DIST    = 1.55       # P(tau=plus|z) = 1-exp(-q*z)

# =============================================================================
# Helper: grids
# =============================================================================
def create_ability_grid_paper(eta: float):
    """
    Paper's 40-point discretization.
    Returns z_grid (float64), prob_z (float64)
    """
    n_z = 200
    M = np.zeros(n_z)
    M[:198] = np.linspace(0.633, 0.998, 198)
    M[198]  = 0.999
    M[199]  = 0.9995
    z_grid = (1.0 - M) ** (-1.0 / eta)

    prob_z = np.zeros(n_z)
    prob_z[0] = M[0] / M[-1]
    for j in range(1, n_z):
        prob_z[j] = (M[j] - M[j-1]) / M[-1]
    prob_z = prob_z / prob_z.sum()
    return z_grid.astype(np.float64), prob_z.astype(np.float64)

def create_asset_grid(n_a: int, a_min: float, a_max: float, curvature: float = 2.0):
    """
    Curved asset grid: a = a_min + (a_max-a_min) * u^curvature.
    """
    u = np.linspace(0.0, 1.0, n_a)
    a_grid = a_min + (a_max - a_min) * (u ** curvature)
    a_grid[0] = max(a_grid[0], 1e-10)
    return a_grid.astype(np.float64)

def compute_tau_prob_plus(z_grid: np.ndarray, q: float):
    return (1.0 - np.exp(-q * z_grid)).astype(np.float64)

# =============================================================================
# Entrepreneur static problem
# =============================================================================
@njit(cache=True, fastmath=True)
def solve_entrepreneur_no_dist(a, z, w, r, lam, delta, alpha, upsilon):
    """
    Static choice of (k,l) with collateral k <= lam*a, no distortions.
    Technology: y = z * (k^alpha * l^(1-alpha))^(1-upsilon)
    Profit: pi = y - w l - (r+delta) k
    """
    rental = r + delta
    if rental <= 1e-12:
        rental = 1e-12
    if w <= 1e-12:
        w = 1e-12

    span = 1.0 - upsilon

    aux1 = (alpha * span * z) / rental
    aux2 = ((1.0 - alpha) * span * z) / w

    exp1 = 1.0 - (1.0 - alpha) * span
    exp2 = (1.0 - alpha) * span

    # unconstrained k*
    k_unc = ( (aux1 ** exp1) * (aux2 ** exp2) ) ** (1.0 / upsilon)
    k = k_unc
    k_cap = lam * a
    if k > k_cap:
        k = k_cap

    l = (aux2 * (k ** (alpha * span))) ** (1.0 / exp1)
    y = z * ((k ** alpha) * (l ** (1.0 - alpha))) ** span
    pi = y - w * l - rental * k
    return pi, k, l, y

@njit(cache=True, fastmath=True)
def solve_entrepreneur_with_tau(a, z, tau, w, r, lam, delta, alpha, upsilon):
    """
    Distortion tau applies to output:
      pi = (1-tau)*y - w l - (r+delta) k
    Equivalent to z_eff = (1-tau)*z.
    """
    z_eff = (1.0 - tau) * z
    if z_eff <= 0.0:
        return -1e18, 0.0, 0.0, 0.0

    rental = r + delta
    if rental <= 1e-12:
        rental = 1e-12
    if w <= 1e-12:
        w = 1e-12

    span = 1.0 - upsilon

    aux1 = (alpha * span * z_eff) / rental
    aux2 = ((1.0 - alpha) * span * z_eff) / w

    exp1 = 1.0 - (1.0 - alpha) * span
    exp2 = (1.0 - alpha) * span

    k_unc = ( (aux1 ** exp1) * (aux2 ** exp2) ) ** (1.0 / upsilon)
    k = k_unc
    k_cap = lam * a
    if k > k_cap:
        k = k_cap

    l = (aux2 * (k ** (alpha * span))) ** (1.0 / exp1)
    y_eff = z_eff * ((k ** alpha) * (l ** (1.0 - alpha))) ** span
    pi = y_eff - w * l - rental * k
    return pi, k, l, y_eff

# =============================================================================
# Income grids for DP (Numba)
# =============================================================================
@njit(cache=True, parallel=True, fastmath=True)
def precompute_income_no_dist(a_grid, z_grid, w, r, lam, delta, alpha, upsilon):
    """
    income(a,z) = max{w, pi(a,z;w,r)} + (1+r)*a
    """
    n_a = a_grid.shape[0]
    n_z = z_grid.shape[0]
    income = np.empty((n_a, n_z), dtype=np.float64)
    w0 = w
    if w0 <= 1e-12:
        w0 = 1e-12

    for iz in prange(n_z):
        z = z_grid[iz]
        for ia in range(n_a):
            a = a_grid[ia]
            pi, _, _, _ = solve_entrepreneur_no_dist(a, z, w, r, lam, delta, alpha, upsilon)
            if pi > w0:
                income[ia, iz] = pi + (1.0 + r) * a
            else:
                income[ia, iz] = w0 + (1.0 + r) * a
    return income

@njit(cache=True, parallel=True, fastmath=True)
def precompute_income_with_tau(a_grid, z_grid, tau, w, r, lam, delta, alpha, upsilon):
    """
    income_tau(a,z) for a fixed tau state.
    """
    n_a = a_grid.shape[0]
    n_z = z_grid.shape[0]
    income = np.empty((n_a, n_z), dtype=np.float64)
    w0 = w
    if w0 <= 1e-12:
        w0 = 1e-12

    for iz in prange(n_z):
        z = z_grid[iz]
        for ia in range(n_a):
            a = a_grid[ia]
            pi, _, _, _ = solve_entrepreneur_with_tau(a, z, tau, w, r, lam, delta, alpha, upsilon)
            if pi > w0:
                income[ia, iz] = pi + (1.0 + r) * a
            else:
                income[ia, iz] = w0 + (1.0 + r) * a
    return income

# =============================================================================
# Utility + monotone savings choice
# =============================================================================
@njit(cache=True, fastmath=True)
def utility(c, sigma):
    if c <= 1e-12:
        return -1e18
    if abs(sigma - 1.0) < 1e-10:
        return np.log(c)
    return (c ** (1.0 - sigma) - 1.0) / (1.0 - sigma)

@njit(cache=True, fastmath=True)
def find_optimal_a_prime(income, a_grid, EV_row, beta, sigma, start_idx):
    """
    Monotone search: start_idx is the previous best choice index for the same z,
    exploiting policy monotonicity in assets.
    """
    n_a = a_grid.shape[0]
    best_val = -1e30
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

# =============================================================================
# Stationary Bellman operator (no distortions) + Howard evaluation
# =============================================================================
@njit(cache=True, parallel=True, fastmath=True)
def bellman_stationary_no_dist(V, a_grid, prob_z, income, beta, sigma, psi):
    n_a, n_z = V.shape
    V_new = np.empty((n_a, n_z), dtype=np.float64)
    pol = np.empty((n_a, n_z), dtype=np.int32)

    # Vbar(a') = sum_z prob_z(z) V(a',z)
    Vbar = np.zeros(n_a, dtype=np.float64)
    for ia in range(n_a):
        s = 0.0
        for iz in range(n_z):
            s += prob_z[iz] * V[ia, iz]
        Vbar[ia] = s

    for iz in prange(n_z):
        EV_row = psi * V[:, iz] + (1.0 - psi) * Vbar
        start = 0
        for ia in range(n_a):
            val, idx = find_optimal_a_prime(income[ia, iz], a_grid, EV_row, beta, sigma, start)
            V_new[ia, iz] = val
            pol[ia, iz] = idx
            start = idx
    return V_new, pol

@njit(cache=True, parallel=True, fastmath=True)
def howard_eval_no_dist(V, pol, a_grid, prob_z, income, beta, sigma, psi, n_iter):
    n_a, n_z = V.shape
    for _ in range(n_iter):
        # recompute Vbar under current V
        Vbar = np.zeros(n_a, dtype=np.float64)
        for ia in range(n_a):
            s = 0.0
            for iz in range(n_z):
                s += prob_z[iz] * V[ia, iz]
            Vbar[ia] = s

        V_next = np.empty_like(V)
        for iz in prange(n_z):
            for ia in range(n_a):
                ip = pol[ia, iz]
                c = income[ia, iz] - a_grid[ip]
                V_next[ia, iz] = utility(c, sigma) + beta * (psi * V[ip, iz] + (1.0 - psi) * Vbar[ip])
        V = V_next
    return V

def solve_stationary_value_no_dist(a_grid, z_grid, prob_z, income, beta, sigma, psi,
                                  V_init=None, tol=1e-6, max_iter=500, howard_iter=20):
    n_a = a_grid.shape[0]
    n_z = z_grid.shape[0]
    if V_init is None:
        V = np.zeros((n_a, n_z), dtype=np.float64)
        # crude initialization: consume all net of minimum saving
        for ia in range(n_a):
            c0 = max(income[ia, 0] - a_grid[0], 1e-3)
            V[ia, :] = utility(c0, sigma) / (1.0 - beta)
    else:
        V = V_init.copy()

    pol = np.zeros((n_a, n_z), dtype=np.int32)

    for it in range(max_iter):
        V_new, pol_new = bellman_stationary_no_dist(V, a_grid, prob_z, income, beta, sigma, psi)
        diff = np.max(np.abs(V_new - V))
        pol = pol_new
        if diff < tol:
            V = V_new
            break
        # Howard when far from convergence
        if howard_iter > 0 and diff > 10.0 * tol:
            V = howard_eval_no_dist(V_new, pol, a_grid, prob_z, income, beta, sigma, psi, howard_iter)
        else:
            V = V_new
    return V, pol

# =============================================================================
# Stationary Bellman operator (distortions) + Howard
# =============================================================================
@njit(cache=True, parallel=True, fastmath=True)
def bellman_stationary_dist(Vp, Vm, a_grid, prob_z, prob_tau_plus, inc_p, inc_m,
                            beta, sigma, psi):
    """
    Vp(a,z): value if current tau=plus
    Vm(a,z): value if current tau=minus
    """
    n_a, n_z = Vp.shape
    Vp_new = np.empty((n_a, n_z), dtype=np.float64)
    Vm_new = np.empty((n_a, n_z), dtype=np.float64)
    pol_p = np.empty((n_a, n_z), dtype=np.int32)
    pol_m = np.empty((n_a, n_z), dtype=np.int32)

    # EV_redraw(a') = sum_{z'} prob_z(z') [p_plus(z') Vp(a',z') + (1-p_plus(z')) Vm(a',z')]
    EV_redraw = np.zeros(n_a, dtype=np.float64)
    for ia in range(n_a):
        s = 0.0
        for iz in range(n_z):
            pz = prob_z[iz]
            pp = prob_tau_plus[iz]
            s += pz * (pp * Vp[ia, iz] + (1.0 - pp) * Vm[ia, iz])
        EV_redraw[ia] = s

    for iz in prange(n_z):
        EVp_row = psi * Vp[:, iz] + (1.0 - psi) * EV_redraw
        EVm_row = psi * Vm[:, iz] + (1.0 - psi) * EV_redraw
        sp = 0
        sm = 0
        for ia in range(n_a):
            vp, ip = find_optimal_a_prime(inc_p[ia, iz], a_grid, EVp_row, beta, sigma, sp)
            vm, im = find_optimal_a_prime(inc_m[ia, iz], a_grid, EVm_row, beta, sigma, sm)
            Vp_new[ia, iz] = vp
            Vm_new[ia, iz] = vm
            pol_p[ia, iz] = ip
            pol_m[ia, iz] = im
            sp = ip
            sm = im

    return Vp_new, Vm_new, pol_p, pol_m

@njit(cache=True, parallel=True, fastmath=True)
def howard_eval_dist(Vp, Vm, pol_p, pol_m, a_grid, prob_z, prob_tau_plus, inc_p, inc_m,
                     beta, sigma, psi, n_iter):
    n_a, n_z = Vp.shape
    for _ in range(n_iter):
        EV_redraw = np.zeros(n_a, dtype=np.float64)
        for ia in range(n_a):
            s = 0.0
            for iz in range(n_z):
                pz = prob_z[iz]
                pp = prob_tau_plus[iz]
                s += pz * (pp * Vp[ia, iz] + (1.0 - pp) * Vm[ia, iz])
            EV_redraw[ia] = s

        Vp_next = np.empty_like(Vp)
        Vm_next = np.empty_like(Vm)
        for iz in prange(n_z):
            for ia in range(n_a):
                ip = pol_p[ia, iz]
                im = pol_m[ia, iz]
                cp = inc_p[ia, iz] - a_grid[ip]
                cm = inc_m[ia, iz] - a_grid[im]
                Vp_next[ia, iz] = utility(cp, sigma) + beta * (psi * Vp[ip, iz] + (1.0 - psi) * EV_redraw[ip])
                Vm_next[ia, iz] = utility(cm, sigma) + beta * (psi * Vm[im, iz] + (1.0 - psi) * EV_redraw[im])
        Vp = Vp_next
        Vm = Vm_next
    return Vp, Vm

def solve_stationary_value_dist(a_grid, z_grid, prob_z, prob_tau_plus, inc_p, inc_m,
                                beta, sigma, psi,
                                Vp_init=None, Vm_init=None,
                                tol=1e-6, max_iter=500, howard_iter=20):
    n_a = a_grid.shape[0]
    n_z = z_grid.shape[0]
    if Vp_init is None:
        Vp = np.zeros((n_a, n_z), dtype=np.float64)
        Vm = np.zeros((n_a, n_z), dtype=np.float64)
        for ia in range(n_a):
            cp = max(inc_p[ia, 0] - a_grid[0], 1e-3)
            cm = max(inc_m[ia, 0] - a_grid[0], 1e-3)
            Vp[ia, :] = utility(cp, sigma) / (1.0 - beta)
            Vm[ia, :] = utility(cm, sigma) / (1.0 - beta)
    else:
        Vp = Vp_init.copy()
        Vm = Vm_init.copy()

    pol_p = np.zeros((n_a, n_z), dtype=np.int32)
    pol_m = np.zeros((n_a, n_z), dtype=np.int32)

    for it in range(max_iter):
        Vp_new, Vm_new, pol_p_new, pol_m_new = bellman_stationary_dist(
            Vp, Vm, a_grid, prob_z, prob_tau_plus, inc_p, inc_m, beta, sigma, psi
        )
        diff = max(np.max(np.abs(Vp_new - Vp)), np.max(np.abs(Vm_new - Vm)))
        pol_p = pol_p_new
        pol_m = pol_m_new
        if diff < tol:
            Vp, Vm = Vp_new, Vm_new
            break
        if howard_iter > 0 and diff > 10.0 * tol:
            Vp, Vm = howard_eval_dist(Vp_new, Vm_new, pol_p, pol_m,
                                      a_grid, prob_z, prob_tau_plus, inc_p, inc_m,
                                      beta, sigma, psi, howard_iter)
        else:
            Vp, Vm = Vp_new, Vm_new

    return Vp, Vm, pol_p, pol_m

# =============================================================================
# Finite-horizon one-step Bellman (backward induction for transition)
# =============================================================================
@njit(cache=True, parallel=True, fastmath=True)
def bellman_one_step(V_next, a_grid, prob_z, income_t, beta, sigma, psi):
    """
    Compute (V_t, pol_t) given continuation value V_{t+1}=V_next.
    This is NOT a fixed point; it is one maximization.
    """
    n_a, n_z = V_next.shape
    V_t = np.empty((n_a, n_z), dtype=np.float64)
    pol = np.empty((n_a, n_z), dtype=np.int32)

    # Vbar_next(a') = sum_z prob_z(z) V_next(a',z)
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
            val, idx = find_optimal_a_prime(income_t[ia, iz], a_grid, EV_row, beta, sigma, start)
            V_t[ia, iz] = val
            pol[ia, iz] = idx
            start = idx

    return V_t, pol

# =============================================================================
# Simulation helpers
# =============================================================================
@njit(cache=True, fastmath=True)
def get_interp_weights(x, grid):
    """
    Return (i_lo, i_hi, w_lo) such that:
      x in [grid[i_lo], grid[i_hi]] and
      x ≈ w_lo*grid[i_lo] + (1-w_lo)*grid[i_hi]
    """
    n = grid.shape[0]
    if x <= grid[0]:
        return 0, 0, 1.0
    if x >= grid[n-1]:
        return n-2, n-1, 0.0
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

@njit(cache=True, parallel=True, fastmath=True)
def simulate_step_no_dist(a_curr, z_curr, pol_a_vals, a_grid, reset_t, shock_t):
    """
    One simulation step (no distortions).
    Decisions use current z_curr; shocks determine z_next.
    """
    n = a_curr.shape[0]
    a_next = np.empty(n, dtype=np.float64)
    z_next = np.empty(n, dtype=np.uint8)

    for i in prange(n):
        zc = z_curr[i]
        il, ih, wl = get_interp_weights(a_curr[i], a_grid)
        a_next[i] = wl * pol_a_vals[il, zc] + (1.0 - wl) * pol_a_vals[ih, zc]
        z_next[i] = shock_t[i] if reset_t[i] else zc

    return a_next, z_next

@njit(cache=True, parallel=True, fastmath=True)
def simulate_step_with_tau(a_curr, z_curr, tau_curr,
                           pol_p_vals, pol_m_vals,
                           a_grid, reset_t, shock_t, tau_shock_t):
    """
    One simulation step in distorted economy with tau state.
    tau_curr is uint8: 1 => tau_plus, 0 => tau_minus
    tau_shock_t is uint8 for those who reset.
    """
    n = a_curr.shape[0]
    a_next = np.empty(n, dtype=np.float64)
    z_next = np.empty(n, dtype=np.uint8)
    tau_next = np.empty(n, dtype=np.uint8)

    for i in prange(n):
        zc = z_curr[i]
        tc = tau_curr[i]
        il, ih, wl = get_interp_weights(a_curr[i], a_grid)

        if tc == 1:
            a_next[i] = wl * pol_p_vals[il, zc] + (1.0 - wl) * pol_p_vals[ih, zc]
        else:
            a_next[i] = wl * pol_m_vals[il, zc] + (1.0 - wl) * pol_m_vals[ih, zc]

        if reset_t[i]:
            z_next[i] = shock_t[i]
            tau_next[i] = tau_shock_t[i]
        else:
            z_next[i] = zc
            tau_next[i] = tc

    return a_next, z_next, tau_next

# =============================================================================
# Market clearing: excess demand functions (holding a,z fixed)
# =============================================================================
@njit(cache=True, parallel=True, fastmath=True)
def labor_excess_no_dist(a_vec, z_vec, z_grid, w, r, lam, delta, alpha, upsilon):
    """
    ED_L(w) = Ld(w) - Ls(w), where Ls = 1 - share_entrepreneurs(w)
    """
    n = a_vec.shape[0]
    w0 = w
    if w0 <= 1e-12:
        w0 = 1e-12

    L = 0.0
    s_e = 0.0
    for i in prange(n):
        a = a_vec[i]
        z = z_grid[z_vec[i]]
        pi, _, l, _ = solve_entrepreneur_no_dist(a, z, w, r, lam, delta, alpha, upsilon)
        if pi > w0:
            L += l
            s_e += 1.0

    L /= n
    s_e /= n
    return L - (1.0 - s_e)

@njit(cache=True, parallel=True, fastmath=True)
def capital_excess_no_dist(a_vec, z_vec, z_grid, w, r, lam, delta, alpha, upsilon):
    """
    ED_K(r) = Kd(r) - A, where A = mean assets
    """
    n = a_vec.shape[0]
    w0 = w
    if w0 <= 1e-12:
        w0 = 1e-12

    K = 0.0
    A = 0.0
    for i in prange(n):
        a = a_vec[i]
        z = z_grid[z_vec[i]]
        A += a
        pi, k, _, _ = solve_entrepreneur_no_dist(a, z, w, r, lam, delta, alpha, upsilon)
        if pi > w0:
            K += k

    K /= n
    A /= n
    return K - A

@njit(cache=True, parallel=True, fastmath=True)
def labor_excess_dist(a_vec, z_vec, tau_vec, z_grid, w, r, lam, delta, alpha, upsilon, tau_plus, tau_minus):
    n = a_vec.shape[0]
    w0 = w
    if w0 <= 1e-12:
        w0 = 1e-12

    L = 0.0
    s_e = 0.0
    for i in prange(n):
        a = a_vec[i]
        z = z_grid[z_vec[i]]
        tau = tau_plus if tau_vec[i] == 1 else tau_minus
        pi, _, l, _ = solve_entrepreneur_with_tau(a, z, tau, w, r, lam, delta, alpha, upsilon)
        if pi > w0:
            L += l
            s_e += 1.0

    L /= n
    s_e /= n
    return L - (1.0 - s_e)

@njit(cache=True, parallel=True, fastmath=True)
def capital_excess_dist(a_vec, z_vec, tau_vec, z_grid, w, r, lam, delta, alpha, upsilon, tau_plus, tau_minus):
    n = a_vec.shape[0]
    w0 = w
    if w0 <= 1e-12:
        w0 = 1e-12

    K = 0.0
    A = 0.0
    for i in prange(n):
        a = a_vec[i]
        z = z_grid[z_vec[i]]
        A += a
        tau = tau_plus if tau_vec[i] == 1 else tau_minus
        pi, k, _, _ = solve_entrepreneur_with_tau(a, z, tau, w, r, lam, delta, alpha, upsilon)
        if pi > w0:
            K += k

    K /= n
    A /= n
    return K - A

# =============================================================================
# Aggregates for plotting
# =============================================================================
@njit(cache=True, parallel=True, fastmath=True)
def aggregates_no_dist(a_vec, z_vec, z_grid, w, r, lam, delta, alpha, upsilon):
    n = a_vec.shape[0]
    w0 = w
    if w0 <= 1e-12:
        w0 = 1e-12

    K = 0.0
    L = 0.0
    Y = 0.0
    A = 0.0
    s_e = 0.0

    for i in prange(n):
        a = a_vec[i]
        z = z_grid[z_vec[i]]
        A += a
        pi, k, l, y = solve_entrepreneur_no_dist(a, z, w, r, lam, delta, alpha, upsilon)
        if pi > w0:
            K += k
            L += l
            Y += y
            s_e += 1.0

    K /= n
    L /= n
    Y /= n
    A /= n
    s_e /= n
    return K, L, Y, A, s_e

@njit(cache=True, parallel=True, fastmath=True)
def aggregates_dist(a_vec, z_vec, tau_vec, z_grid, w, r, lam, delta, alpha, upsilon, tau_plus, tau_minus):
    n = a_vec.shape[0]
    w0 = w
    if w0 <= 1e-12:
        w0 = 1e-12

    K = 0.0
    L = 0.0
    Y = 0.0
    A = 0.0
    s_e = 0.0

    for i in prange(n):
        a = a_vec[i]
        z = z_grid[z_vec[i]]
        A += a
        tau = tau_plus if tau_vec[i] == 1 else tau_minus
        pi, k, l, y = solve_entrepreneur_with_tau(a, z, tau, w, r, lam, delta, alpha, upsilon)
        if pi > w0:
            K += k
            L += l
            Y += y
            s_e += 1.0

    K /= n
    L /= n
    Y /= n
    A /= n
    s_e /= n
    return K, L, Y, A, s_e

# =============================================================================
# Shock generation (panel) with low peak memory
# =============================================================================
def generate_panel_shocks(T, N, psi, prob_z, seed=123, with_tau=False, prob_tau_plus=None):
    """
    Generate arrays:
      resets[t,i] in {0,1}, uint8
      z_shocks[t,i] in {0..n_z-1}, uint8
      tau_shocks[t,i] in {0,1}, uint8   (only if with_tau)

    Generation is done step-by-step to avoid allocating giant float matrices.
    """
    rng = np.random.default_rng(seed)
    cdf = np.cumsum(prob_z)

    resets = np.empty((T, N), dtype=np.uint8)
    z_shocks = np.empty((T, N), dtype=np.uint8)

    if with_tau:
        if prob_tau_plus is None:
            raise ValueError("prob_tau_plus must be provided when with_tau=True")
        tau_shocks = np.empty((T, N), dtype=np.uint8)
    else:
        tau_shocks = None

    for t in range(T):
        u = rng.random(N)
        resets[t, :] = (u > psi).astype(np.uint8)

        u2 = rng.random(N)
        z_shocks[t, :] = np.searchsorted(cdf, u2).astype(np.uint8)

        if with_tau:
            # tau shock only matters for those who reset; still pre-generate for all
            u3 = rng.random(N)
            probs = prob_tau_plus[z_shocks[t, :]]
            tau_shocks[t, :] = (u3 < probs).astype(np.uint8)

    return resets, z_shocks, tau_shocks

# =============================================================================
# Static per-period market clearing solvers (bisection + warm start)
# =============================================================================
def bisect_w_labor_clearing(a_t, z_t, z_grid, r_t,
                            w_guess, w_min, w_max,
                            lam, delta, alpha, upsilon,
                            max_it=14, expand_it=10):
    """
    Find w that solves ED_L(w)=0 holding (a_t,z_t) fixed.
    """
    lo = max(w_min, 0.7 * w_guess)
    hi = min(w_max, 1.3 * w_guess)

    f_lo = labor_excess_no_dist(a_t, z_t, z_grid, lo, r_t, lam, delta, alpha, upsilon)
    f_hi = labor_excess_no_dist(a_t, z_t, z_grid, hi, r_t, lam, delta, alpha, upsilon)

    k = 0
    while f_lo * f_hi > 0.0 and k < expand_it:
        lo = max(w_min, lo * 0.7)
        hi = min(w_max, hi * 1.3)
        f_lo = labor_excess_no_dist(a_t, z_t, z_grid, lo, r_t, lam, delta, alpha, upsilon)
        f_hi = labor_excess_no_dist(a_t, z_t, z_grid, hi, r_t, lam, delta, alpha, upsilon)
        k += 1

    if f_lo * f_hi > 0.0:
        # fallback: damped correction
        f = labor_excess_no_dist(a_t, z_t, z_grid, w_guess, r_t, lam, delta, alpha, upsilon)
        w_new = w_guess * (1.0 + 0.25 * np.clip(f, -0.5, 0.5))
        return float(np.clip(w_new, w_min, w_max))

    for _ in range(max_it):
        mid = 0.5 * (lo + hi)
        f_mid = labor_excess_no_dist(a_t, z_t, z_grid, mid, r_t, lam, delta, alpha, upsilon)
        if f_lo * f_mid <= 0.0:
            hi = mid
            f_hi = f_mid
        else:
            lo = mid
            f_lo = f_mid
    return float(0.5 * (lo + hi))

def bisect_r_capital_clearing(a_t, z_t, z_grid, w_t,
                              r_guess, r_min, r_max,
                              lam, delta, alpha, upsilon,
                              max_it=14, expand_it=10):
    """
    Find r that solves ED_K(r)=0 holding (a_t,z_t) fixed.
    """
    lo = max(r_min, r_guess - 0.06)
    hi = min(r_max, r_guess + 0.06)

    f_lo = capital_excess_no_dist(a_t, z_t, z_grid, w_t, lo, lam, delta, alpha, upsilon)
    f_hi = capital_excess_no_dist(a_t, z_t, z_grid, w_t, hi, lam, delta, alpha, upsilon)

    k = 0
    while f_lo * f_hi > 0.0 and k < expand_it:
        lo = max(r_min, lo - 0.05)
        hi = min(r_max, hi + 0.05)
        f_lo = capital_excess_no_dist(a_t, z_t, z_grid, w_t, lo, lam, delta, alpha, upsilon)
        f_hi = capital_excess_no_dist(a_t, z_t, z_grid, w_t, hi, lam, delta, alpha, upsilon)
        k += 1

    if f_lo * f_hi > 0.0:
        f = capital_excess_no_dist(a_t, z_t, z_grid, w_t, r_guess, lam, delta, alpha, upsilon)
        r_new = r_guess + 0.20 * np.clip(f, -0.25, 0.25)
        return float(np.clip(r_new, r_min, r_max))

    for _ in range(max_it):
        mid = 0.5 * (lo + hi)
        f_mid = capital_excess_no_dist(a_t, z_t, z_grid, w_t, mid, lam, delta, alpha, upsilon)
        if f_lo * f_mid <= 0.0:
            hi = mid
            f_hi = f_mid
        else:
            lo = mid
            f_lo = f_mid
    return float(0.5 * (lo + hi))

def bisect_w_labor_clearing_dist(a_t, z_t, tau_t, z_grid, r_t,
                                 w_guess, w_min, w_max,
                                 lam, delta, alpha, upsilon,
                                 tau_plus, tau_minus,
                                 max_it=14, expand_it=10):
    lo = max(w_min, 0.7 * w_guess)
    hi = min(w_max, 1.3 * w_guess)

    f_lo = labor_excess_dist(a_t, z_t, tau_t, z_grid, lo, r_t, lam, delta, alpha, upsilon, tau_plus, tau_minus)
    f_hi = labor_excess_dist(a_t, z_t, tau_t, z_grid, hi, r_t, lam, delta, alpha, upsilon, tau_plus, tau_minus)

    k = 0
    while f_lo * f_hi > 0.0 and k < expand_it:
        lo = max(w_min, lo * 0.7)
        hi = min(w_max, hi * 1.3)
        f_lo = labor_excess_dist(a_t, z_t, tau_t, z_grid, lo, r_t, lam, delta, alpha, upsilon, tau_plus, tau_minus)
        f_hi = labor_excess_dist(a_t, z_t, tau_t, z_grid, hi, r_t, lam, delta, alpha, upsilon, tau_plus, tau_minus)
        k += 1

    if f_lo * f_hi > 0.0:
        f = labor_excess_dist(a_t, z_t, tau_t, z_grid, w_guess, r_t, lam, delta, alpha, upsilon, tau_plus, tau_minus)
        w_new = w_guess * (1.0 + 0.25 * np.clip(f, -0.5, 0.5))
        return float(np.clip(w_new, w_min, w_max))

    for _ in range(max_it):
        mid = 0.5 * (lo + hi)
        f_mid = labor_excess_dist(a_t, z_t, tau_t, z_grid, mid, r_t, lam, delta, alpha, upsilon, tau_plus, tau_minus)
        if f_lo * f_mid <= 0.0:
            hi = mid
            f_hi = f_mid
        else:
            lo = mid
            f_lo = f_mid
    return float(0.5 * (lo + hi))

def bisect_r_capital_clearing_dist(a_t, z_t, tau_t, z_grid, w_t,
                                   r_guess, r_min, r_max,
                                   lam, delta, alpha, upsilon,
                                   tau_plus, tau_minus,
                                   max_it=14, expand_it=10):
    lo = max(r_min, r_guess - 0.06)
    hi = min(r_max, r_guess + 0.06)

    f_lo = capital_excess_dist(a_t, z_t, tau_t, z_grid, w_t, lo, lam, delta, alpha, upsilon, tau_plus, tau_minus)
    f_hi = capital_excess_dist(a_t, z_t, tau_t, z_grid, w_t, hi, lam, delta, alpha, upsilon, tau_plus, tau_minus)

    k = 0
    while f_lo * f_hi > 0.0 and k < expand_it:
        lo = max(r_min, lo - 0.05)
        hi = min(r_max, hi + 0.05)
        f_lo = capital_excess_dist(a_t, z_t, tau_t, z_grid, w_t, lo, lam, delta, alpha, upsilon, tau_plus, tau_minus)
        f_hi = capital_excess_dist(a_t, z_t, tau_t, z_grid, w_t, hi, lam, delta, alpha, upsilon, tau_plus, tau_minus)
        k += 1

    if f_lo * f_hi > 0.0:
        f = capital_excess_dist(a_t, z_t, tau_t, z_grid, w_t, r_guess, lam, delta, alpha, upsilon, tau_plus, tau_minus)
        r_new = r_guess + 0.20 * np.clip(f, -0.25, 0.25)
        return float(np.clip(r_new, r_min, r_max))

    for _ in range(max_it):
        mid = 0.5 * (lo + hi)
        f_mid = capital_excess_dist(a_t, z_t, tau_t, z_grid, w_t, mid, lam, delta, alpha, upsilon, tau_plus, tau_minus)
        if f_lo * f_mid <= 0.0:
            hi = mid
            f_hi = f_mid
        else:
            lo = mid
            f_lo = f_mid
    return float(0.5 * (lo + hi))

# =============================================================================
# Tail anchoring (stabilizes the last periods)
# =============================================================================
def anchor_tail(path, target, tail_len=20, decay=0.35):
    T = len(path)
    if tail_len <= 0:
        path[-1] = target
        return path
    k = min(tail_len, T)
    start = T - k
    for j in range(k):
        path[start + j] = target + (path[start + j] - target) * np.exp(-decay * j)
    path[-1] = target
    return path

# =============================================================================
# Stationary equilibria (B.2-style nested price iteration)
# =============================================================================
def solve_post_steady_state(a_grid, z_grid, prob_z,
                           N, Tss,
                           w_init=0.8, r_init=-0.04,
                           eta_w=0.35, eta_r=0.20,
                           tol_w=5e-4, tol_r=5e-4,
                           max_outer=40, max_inner=40,
                           w_min=0.02, w_max=6.0,
                           r_min=-0.25, r_max=None,
                           seed_shocks=42,
                           verbose=True):
    """
    Post-reform steady state (no distortions).
    Uses nested iteration: inner clears wage given r, outer updates r.
    """
    if r_max is None:
        r_max = 1.0 / BETA - 1.0 - 1e-6

    # Common random numbers for burn-in simulation in SS search:
    resets, z_shocks, _ = generate_panel_shocks(Tss, N, PSI, prob_z, seed=seed_shocks, with_tau=False)

    w = float(w_init)
    r = float(r_init)

    V = None
    best = None
    best_err = 1e18

    # initial agent states for simulation (fixed seed each iteration to reduce noise)
    rng_init = np.random.default_rng(777)
    z0_fixed = rng_init.integers(0, len(z_grid), size=N, dtype=np.uint8)

    for outer in range(1, max_outer + 1):
        r_old = r

        # -------- inner loop on wages --------
        for inner in range(1, max_inner + 1):
            w_old = w

            income = precompute_income_no_dist(a_grid, z_grid, w, r, LAMBDA, DELTA, ALPHA, NU)
            V, pol = solve_stationary_value_no_dist(a_grid, z_grid, prob_z, income,
                                                   BETA, SIGMA, PSI,
                                                   V_init=V, tol=1e-6, max_iter=400, howard_iter=25)

            # simulate burn-in to approximate stationary distribution
            a = np.full(N, a_grid[0], dtype=np.float64)
            z = z0_fixed.copy()
            pol_vals = a_grid[pol]  # (n_a,n_z)

            for t in range(Tss):
                a, z = simulate_step_no_dist(a, z, pol_vals, a_grid, resets[t], z_shocks[t])

            # labor-clearing wage holding (a,z) fixed
            w_clear = bisect_w_labor_clearing(a, z, z_grid, r, w, w_min, w_max,
                                              LAMBDA, DELTA, ALPHA, NU, max_it=14)

            w = eta_w * w_clear + (1.0 - eta_w) * w
            w = float(np.clip(w, w_min, w_max))

            if verbose and (inner <= 2 or inner % 5 == 0):
                ED_L = labor_excess_no_dist(a, z, z_grid, w, r, LAMBDA, DELTA, ALPHA, NU)
                print(f"[POST SS | outer {outer:02d} wage {inner:02d}] w={w:.6f}  |Δw|={abs(w-w_old):.2e}  ED_L={ED_L:.3e}")

            if abs(w - w_old) < tol_w:
                break

        # -------- update r to clear capital --------
        r_clear = bisect_r_capital_clearing(a, z, z_grid, w, r, r_min, r_max,
                                            LAMBDA, DELTA, ALPHA, NU, max_it=14)
        r = eta_r * r_clear + (1.0 - eta_r) * r
        r = float(np.clip(r, r_min, r_max))

        ED_L = labor_excess_no_dist(a, z, z_grid, w, r, LAMBDA, DELTA, ALPHA, NU)
        ED_K = capital_excess_no_dist(a, z, z_grid, w, r, LAMBDA, DELTA, ALPHA, NU)
        err = abs(ED_L) + abs(ED_K)

        if verbose:
            print(f"[POST SS | outer {outer:02d}] r={r:.6f}  |Δr|={abs(r-r_old):.2e}  ED_L={ED_L:.3e}  ED_K={ED_K:.3e}\n")

        if err < best_err:
            K, L, Y, A, s_e = aggregates_no_dist(a, z, z_grid, w, r, LAMBDA, DELTA, ALPHA, NU)
            best_err = err
            best = dict(w=w, r=r, V=V.copy(), pol=pol.copy(),
                        a=a.copy(), z=z.copy(),
                        K=K, L=L, Y=Y, A=A, s_e=s_e)

        if abs(r - r_old) < tol_r and abs(ED_L) < 5e-3 and abs(ED_K) < 5e-3:
            if verbose:
                print("Post-reform steady state converged.\n")
            break

    return best

def solve_pre_steady_state(a_grid, z_grid, prob_z, prob_tau_plus,
                          N, Tss,
                          w_init, r_init,
                          eta_w=0.35, eta_r=0.20,
                          tol_w=5e-4, tol_r=5e-4,
                          max_outer=50, max_inner=40,
                          w_min=0.02, w_max=6.0,
                          r_min=-0.25, r_max=None,
                          seed_shocks=123,
                          verbose=True):
    """
    Pre-reform steady state (with distortions) solved with the same nested logic.
    State includes tau in {plus, minus}.
    """
    if r_max is None:
        r_max = 1.0 / BETA - 1.0 - 1e-6

    resets, z_shocks, tau_shocks = generate_panel_shocks(
        Tss, N, PSI, prob_z, seed=seed_shocks, with_tau=True, prob_tau_plus=prob_tau_plus
    )

    w = float(w_init)
    r = float(r_init)

    Vp = None
    Vm = None
    best = None
    best_err = 1e18

    rng_init = np.random.default_rng(999)
    z0_fixed = rng_init.integers(0, len(z_grid), size=N, dtype=np.uint8)
    u0 = rng_init.random(N)
    tau0_fixed = (u0 < prob_tau_plus[z0_fixed]).astype(np.uint8)  # 1=plus,0=minus

    for outer in range(1, max_outer + 1):
        r_old = r

        # -------- inner loop on wages --------
        for inner in range(1, max_inner + 1):
            w_old = w

            inc_p = precompute_income_with_tau(a_grid, z_grid, TAU_PLUS,  w, r, LAMBDA, DELTA, ALPHA, NU)
            inc_m = precompute_income_with_tau(a_grid, z_grid, TAU_MINUS, w, r, LAMBDA, DELTA, ALPHA, NU)

            Vp, Vm, pol_p, pol_m = solve_stationary_value_dist(
                a_grid, z_grid, prob_z, prob_tau_plus, inc_p, inc_m,
                BETA, SIGMA, PSI,
                Vp_init=Vp, Vm_init=Vm, tol=1e-6, max_iter=400, howard_iter=25
            )

            # simulate burn-in (a,z,tau)
            a = np.full(N, a_grid[0], dtype=np.float64)
            z = z0_fixed.copy()
            tau = tau0_fixed.copy()
            pol_p_vals = a_grid[pol_p]
            pol_m_vals = a_grid[pol_m]

            for t in range(Tss):
                a, z, tau = simulate_step_with_tau(a, z, tau, pol_p_vals, pol_m_vals,
                                                   a_grid, resets[t], z_shocks[t], tau_shocks[t])

            w_clear = bisect_w_labor_clearing_dist(a, z, tau, z_grid, r, w, w_min, w_max,
                                                   LAMBDA, DELTA, ALPHA, NU,
                                                   TAU_PLUS, TAU_MINUS,
                                                   max_it=14)

            w = eta_w * w_clear + (1.0 - eta_w) * w
            w = float(np.clip(w, w_min, w_max))

            if verbose and (inner <= 2 or inner % 5 == 0):
                ED_L = labor_excess_dist(a, z, tau, z_grid, w, r, LAMBDA, DELTA, ALPHA, NU, TAU_PLUS, TAU_MINUS)
                print(f"[PRE  SS | outer {outer:02d} wage {inner:02d}] w={w:.6f}  |Δw|={abs(w-w_old):.2e}  ED_L={ED_L:.3e}")

            if abs(w - w_old) < tol_w:
                break

        # -------- update r --------
        r_clear = bisect_r_capital_clearing_dist(a, z, tau, z_grid, w, r, r_min, r_max,
                                                 LAMBDA, DELTA, ALPHA, NU,
                                                 TAU_PLUS, TAU_MINUS,
                                                 max_it=14)
        r = eta_r * r_clear + (1.0 - eta_r) * r
        r = float(np.clip(r, r_min, r_max))

        ED_L = labor_excess_dist(a, z, tau, z_grid, w, r, LAMBDA, DELTA, ALPHA, NU, TAU_PLUS, TAU_MINUS)
        ED_K = capital_excess_dist(a, z, tau, z_grid, w, r, LAMBDA, DELTA, ALPHA, NU, TAU_PLUS, TAU_MINUS)
        err = abs(ED_L) + abs(ED_K)

        if verbose:
            print(f"[PRE  SS | outer {outer:02d}] r={r:.6f}  |Δr|={abs(r-r_old):.2e}  ED_L={ED_L:.3e}  ED_K={ED_K:.3e}\n")

        if err < best_err:
            K, L, Y, A, s_e = aggregates_dist(a, z, tau, z_grid, w, r, LAMBDA, DELTA, ALPHA, NU, TAU_PLUS, TAU_MINUS)
            best_err = err
            best = dict(w=w, r=r, a=a.copy(), z=z.copy(), tau=tau.copy(),
                        K=K, L=L, Y=Y, A=A, s_e=s_e)

        if abs(r - r_old) < tol_r and abs(ED_L) < 5e-3 and abs(ED_K) < 5e-3:
            if verbose:
                print("Pre-reform steady state converged.\n")
            break

    return best

# =============================================================================
# Algorithm B.2 transition
# =============================================================================
def solve_transition_B2(pre_ss, post_ss,
                        a_grid, z_grid, prob_z,
                        N, T,
                        eta_w=0.35, eta_r=0.20,
                        tol_w_seq=2e-4, tol_r_seq=2e-4,
                        max_w_inner=30, max_outer=30,
                        w_min=0.02, w_max=6.0,
                        r_min=-0.25, r_max=None,
                        tail_len=1, tail_decay=0.35,
                        seed_shocks=2026,
                        verbose=True):
    """
    Appendix Algorithm B.2 (Simulation-based transition).
    Distortions removed from t>=0, so we use no-distortion entrepreneur problem in the transition.
    """
    if r_max is None:
        r_max = 1.0 / BETA - 1.0 - 1e-6

    # Initial cross section at t=0 comes from the pre-reform SS distribution.
    a0 = pre_ss["a"].astype(np.float64, copy=True)
    z0 = pre_ss["z"].astype(np.uint8, copy=True)

    # Terminal value function = post-reform stationary value function
    V_T = post_ss["V"].astype(np.float64, copy=True)
    w_post, r_post = float(post_ss["w"]), float(post_ss["r"])
    w_pre,  r_pre  = float(pre_ss["w"]),  float(pre_ss["r"])

    # Initial guesses for sequences
    w_path = np.linspace(w_pre, w_post, T).astype(np.float64)
    r_path = np.linspace(r_pre, r_post, T).astype(np.float64)

    # Common random numbers for the transition (fixed across iterations)
    resets, z_shocks, _ = generate_panel_shocks(T, N, PSI, prob_z, seed=seed_shocks, with_tau=False)

    if verbose:
        print("\n" + "=" * 78)
        print("ALGORITHM B.2 — Transition (Monte Carlo + backward induction)")
        print("=" * 78)
        print(f"T={T}, N={N}, eta_w={eta_w:.2f}, eta_r={eta_r:.2f}")
        print(f"Pre SS:  w={w_pre:.4f}, r={r_pre:.4f} | Post SS: w={w_post:.4f}, r={r_post:.4f}")
        print("=" * 78 + "\n")

    # Storage (optional) for diagnostics
    last_aT = None
    last_zT = None

    for outer in range(1, max_outer + 1):
        r_old_path = r_path.copy()

        # ------------------------------
        # Inner loop: converge wage path given r_path
        # ------------------------------
        for inner in range(1, max_w_inner + 1):
            w_old_path = w_path.copy()

            # Backward induction (store policies only)
            pol_idx_path = [None] * T
            V_next = V_T

            for tt in range(T - 1, -1, -1):
                income_t = precompute_income_no_dist(a_grid, z_grid, w_path[tt], r_path[tt],
                                                     LAMBDA, DELTA, ALPHA, NU)
                V_t, pol_t = bellman_one_step(V_next, a_grid, prob_z, income_t, BETA, SIGMA, PSI)
                pol_idx_path[tt] = pol_t
                V_next = V_t

            # Forward simulate, but compute w_clear_t on the fly to avoid storing full history
            a = a0.copy()
            z = z0.copy()
            w_clear = np.empty(T, dtype=np.float64)

            # Use a "smooth warm start" across t: start guess at w_path[0] then use previous solution.
            w_guess_t = float(w_path[0])

            for tt in range(T):
                # compute labor-clearing wage holding (a,z) fixed and given r_t
                w_guess_t = float(w_path[tt]) if tt == 0 else w_guess_t
                w_clear_tt = bisect_w_labor_clearing(a, z, z_grid, r_path[tt],
                                                     w_guess_t, w_min, w_max,
                                                     LAMBDA, DELTA, ALPHA, NU,
                                                     max_it=14)
                w_clear[tt] = w_clear_tt
                w_guess_t = w_clear_tt

                # simulate to next period under current policy (computed with guessed w_path, not w_clear)
                pol_vals = a_grid[pol_idx_path[tt]]
                a, z = simulate_step_no_dist(a, z, pol_vals, a_grid, resets[tt], z_shocks[tt])

            last_aT = a
            last_zT = z

            # Relax-update wage sequence
            w_path = eta_w * w_clear + (1.0 - eta_w) * w_path
            w_path = np.clip(w_path, w_min, w_max)
            w_path = anchor_tail(w_path, w_post, tail_len=tail_len, decay=tail_decay)

            w_diff = float(np.max(np.abs(w_path - w_old_path)))
            if verbose and (inner <= 2 or inner % 3 == 0):
                print(f"[outer {outer:02d} | wage {inner:02d}] max|Δw|={w_diff:.3e}")

            if w_diff < tol_w_seq:
                break

        # ------------------------------
        # Step 3 in paper: after wage converges, construct iota_t clearing capital markets
        # Holding simulated (a_t,z_t) fixed — we must re-simulate once with the *converged* wage path.
        # ------------------------------
        # Backward induction once more (under converged w_path and current r_path)
        pol_idx_path = [None] * T
        V_next = V_T
        for tt in range(T - 1, -1, -1):
            income_t = precompute_income_no_dist(a_grid, z_grid, w_path[tt], r_path[tt],
                                                 LAMBDA, DELTA, ALPHA, NU)
            V_t, pol_t = bellman_one_step(V_next, a_grid, prob_z, income_t, BETA, SIGMA, PSI)
            pol_idx_path[tt] = pol_t
            V_next = V_t

        # simulate again, and compute r_clear_t on the fly
        a = a0.copy()
        z = z0.copy()
        r_clear = np.empty(T, dtype=np.float64)
        r_guess_t = float(r_path[0])

        for tt in range(T):
            r_guess_t = float(r_path[tt]) if tt == 0 else r_guess_t
            r_clear_tt = bisect_r_capital_clearing(a, z, z_grid, w_path[tt],
                                                   r_guess_t, r_min, r_max,
                                                   LAMBDA, DELTA, ALPHA, NU,
                                                   max_it=14)
            r_clear[tt] = r_clear_tt
            r_guess_t = r_clear_tt

            pol_vals = a_grid[pol_idx_path[tt]]
            a, z = simulate_step_no_dist(a, z, pol_vals, a_grid, resets[tt], z_shocks[tt])

        last_aT = a
        last_zT = z

        # Relax-update r sequence
        r_path = eta_r * r_clear + (1.0 - eta_r) * r_path
        r_path = np.clip(r_path, r_min, r_max)
        r_path = anchor_tail(r_path, r_post, tail_len=tail_len, decay=tail_decay)

        r_diff = float(np.max(np.abs(r_path - r_old_path)))
        if verbose:
            print(f"[outer {outer:02d}] max|Δr|={r_diff:.3e}\n")

        if r_diff < tol_r_seq:
            if verbose:
                print(f"Converged outer loop after {outer} iterations.\n")
            break

    # ============================
    # Final: compute aggregates along converged sequences
    # (do one last backward+forward under final sequences)
    # ============================
    pol_idx_path = [None] * T
    V_next = V_T
    for tt in range(T - 1, -1, -1):
        income_t = precompute_income_no_dist(a_grid, z_grid, w_path[tt], r_path[tt],
                                             LAMBDA, DELTA, ALPHA, NU)
        V_t, pol_t = bellman_one_step(V_next, a_grid, prob_z, income_t, BETA, SIGMA, PSI)
        pol_idx_path[tt] = pol_t
        V_next = V_t

    Y = np.empty(T, dtype=np.float64)
    K = np.empty(T, dtype=np.float64)
    A = np.empty(T, dtype=np.float64)
    L = np.empty(T, dtype=np.float64)
    s_e = np.empty(T, dtype=np.float64)
    ED_L = np.empty(T, dtype=np.float64)
    ED_K = np.empty(T, dtype=np.float64)
    TFP = np.empty(T, dtype=np.float64)

    a = a0.copy()
    z = z0.copy()
    span = 1.0 - NU

    for tt in range(T):
        Kt, Lt, Yt, At, se_t = aggregates_no_dist(a, z, z_grid, w_path[tt], r_path[tt],
                                                  LAMBDA, DELTA, ALPHA, NU)
        K[tt] = Kt
        L[tt] = Lt
        Y[tt] = Yt
        A[tt] = At
        s_e[tt] = se_t
        ED_L[tt] = Lt - (1.0 - se_t)
        ED_K[tt] = Kt - At

        Ls = 1.0 - se_t
        denom = ((Kt ** ALPHA) * (Ls ** (1.0 - ALPHA))) ** span
        TFP[tt] = Yt / max(denom, 1e-12)

        pol_vals = a_grid[pol_idx_path[tt]]
        a, z = simulate_step_no_dist(a, z, pol_vals, a_grid, resets[tt], z_shocks[tt])

    return dict(
        t=np.arange(T),
        w=w_path,
        r=r_path,
        Y=Y, K=K, A=A, L=L, s_e=s_e,
        ED_L=ED_L, ED_K=ED_K,
        TFP=TFP
    )

# =============================================================================
# Plotting
# =============================================================================
def plot_transition(pre_ss, post_ss, trans, outdir):
    os.makedirs(outdir, exist_ok=True)

    t = trans["t"]
    Y = trans["Y"] / pre_ss["Y"]
    K = trans["K"] / pre_ss["K"]
    w = trans["w"] / pre_ss["w"]
    r = trans["r"]  # levels (can be negative)
    TFP = trans["TFP"] / (pre_ss["Y"] / max(((pre_ss["K"] ** ALPHA) * ((1.0 - pre_ss["s_e"]) ** (1.0 - ALPHA))) ** (1.0 - NU), 1e-12))

    ED_L = np.abs(trans["ED_L"])
    ED_K = np.abs(trans["ED_K"])

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    axes[0,0].plot(t, Y)
    axes[0,0].set_title("Output (Y) / pre-SS")
    axes[0,0].grid(True, alpha=0.3)

    axes[0,1].plot(t, TFP)
    axes[0,1].set_title("TFP / pre-SS")
    axes[0,1].grid(True, alpha=0.3)

    axes[0,2].plot(t, K)
    axes[0,2].set_title("Capital (K) / pre-SS")
    axes[0,2].grid(True, alpha=0.3)

    axes[1,0].plot(t, w)
    axes[1,0].set_title("Wage (w) / pre-SS")
    axes[1,0].grid(True, alpha=0.3)

    axes[1,1].plot(t, r)
    axes[1,1].set_title("Interest rate (r) level")
    axes[1,1].grid(True, alpha=0.3)

    axes[1,2].plot(t, ED_L, label="|ED_L|")
    axes[1,2].plot(t, ED_K, label="|ED_K|")
    axes[1,2].set_title("Excess demand (abs)")
    axes[1,2].legend()
    axes[1,2].grid(True, alpha=0.3)

    for ax in axes.flat:
        ax.set_xlabel("t")

    plt.tight_layout()
    outpath = os.path.join(outdir, "transition_B2_sim.png")
    plt.savefig(outpath, dpi=200)
    plt.close()
    print(f"Saved plot: {outpath}")

# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Buera & Shin (2010) Appendix B.2 transition (Numba)")
    parser.add_argument("--T", type=int, default=125, help="Transition horizon (paper uses 125)")
    parser.add_argument("--na", type=int, default=2001, help="Asset grid size")
    parser.add_argument("--amax", type=float, default=4000.0, help="Max asset")
    parser.add_argument("--N", type=int, default=350000, help="Number of simulated agents")
    parser.add_argument("--Tss", type=int, default=500, help="Burn-in length for SS simulation")
    parser.add_argument("--outdir", type=str, default="outputs_B2", help="Output folder")
    parser.add_argument("--threads", type=int, default=0, help="Numba threads (0=default)")
    args = parser.parse_args()

    if args.threads and args.threads > 0:
        set_num_threads(args.threads)

    os.makedirs(args.outdir, exist_ok=True)

    print("Building grids...")
    z_grid, prob_z = create_ability_grid_paper(ETA_PARETO)
    a_grid = create_asset_grid(args.na, 1e-6, args.amax, curvature=2.0)
    prob_tau_plus = compute_tau_prob_plus(z_grid, Q_DIST)

    # Bounds
    w_min, w_max = 0.02, 6.0
    r_min = -0.25
    r_max = 1.0 / BETA - 1.0 - 1e-6

    # -------------------------------------------------------------------------
    # 1) Post-reform steady state
    # -------------------------------------------------------------------------
    print("\nSTEP 1: Post-reform steady state (no distortions)")
    t0 = time.time()
    post_ss = solve_post_steady_state(
        a_grid, z_grid, prob_z,
        N=args.N, Tss=args.Tss,
        w_init=0.8, r_init=-0.04,
        eta_w=0.35, eta_r=0.20,
        tol_w=5e-4, tol_r=5e-4,
        max_outer=35, max_inner=35,
        w_min=w_min, w_max=w_max,
        r_min=r_min, r_max=r_max,
        seed_shocks=42,
        verbose=True
    )
    print(f"Post SS done in {time.time()-t0:.1f}s: w={post_ss['w']:.6f}, r={post_ss['r']:.6f}")

    # -------------------------------------------------------------------------
    # 2) Pre-reform steady state
    # -------------------------------------------------------------------------
    print("\nSTEP 2: Pre-reform steady state (with distortions) — initial distribution")
    t0 = time.time()
    pre_ss = solve_pre_steady_state(
        a_grid, z_grid, prob_z, prob_tau_plus,
        N=args.N, Tss=args.Tss,
        w_init=post_ss["w"], r_init=post_ss["r"],
        eta_w=0.35, eta_r=0.20,
        tol_w=5e-4, tol_r=5e-4,
        max_outer=45, max_inner=35,
        w_min=w_min, w_max=w_max,
        r_min=r_min, r_max=r_max,
        seed_shocks=123,
        verbose=True
    )
    print(f"Pre SS done in {time.time()-t0:.1f}s: w={pre_ss['w']:.6f}, r={pre_ss['r']:.6f}")

    # -------------------------------------------------------------------------
    # 3) Transition via B.2
    # -------------------------------------------------------------------------
    print("\nSTEP 3: Transition path via Appendix Algorithm B.2 (reform at t=0)")
    t0 = time.time()
    trans = solve_transition_B2(
        pre_ss, post_ss,
        a_grid, z_grid, prob_z,
        N=args.N, T=args.T,
        eta_w=0.35, eta_r=0.20,
        tol_w_seq=2e-4, tol_r_seq=2e-4,
        max_w_inner=25, max_outer=25,
        w_min=w_min, w_max=w_max,
        r_min=r_min, r_max=r_max,
        tail_len=1, tail_decay=0.35,
        seed_shocks=2026,
        verbose=True
    )
    print(f"Transition solved in {time.time()-t0:.1f}s")

    # -------------------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------------------
    plot_transition(pre_ss, post_ss, trans, args.outdir)

    # Tip: if you still see end-of-horizon wiggles, try tail_len=20 in solve_transition_B2().

if __name__ == "__main__":
    main()