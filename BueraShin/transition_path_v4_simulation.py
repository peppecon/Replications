"""
Buera & Shin (2010) Transition Dynamics - VERSION 4 (SIMULATION)
Combines Howard Policy Iteration (backward pass) with Monte Carlo Simulation (forward pass).
Matches paper's exact ability discretization and recent plotting enhancements.
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
import time
import os
import json
import argparse
import warnings

warnings.filterwarnings('ignore')

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
N_AGENTS = 350000      # Number of agents for Monte Carlo
T_SIM_SS = 500         # Burn-in for stationary equilibrium
DAMPING_FACTOR = 0.8   # Price damping parameter (0=no damping, 1=no update)

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
    z_grid = (1 - M_values) ** (-1/eta)

    prob_z = np.zeros(n_z)
    prob_z[0] = M_values[0] / M_values[-1]
    for j in range(1, n_z):
        prob_z[j] = (M_values[j] - M_values[j-1]) / M_values[-1]
    prob_z = prob_z / prob_z.sum()

    return z_grid, prob_z

def create_asset_grid(n_a, a_min, a_max):
    """Asset grid with curvature scaling (Power 2 for robustness)"""
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
    """Static profit maximization for a single state"""
    rental = max(r + delta, 1e-8)
    wage = max(w, 1e-8)
    span = 1 - upsilon
    
    aux1 = (1/rental) * alpha * span * z
    aux2 = (1/wage) * (1-alpha) * span * z
    exp1 = 1 - (1-alpha) * span
    exp2 = (1-alpha) * span
    k1 = (aux1 ** exp1 * aux2 ** exp2) ** (1/upsilon)
    
    kstar = min(k1, lam * a)
    lstar = ((aux2 * (kstar ** (alpha * span))) ** (1/exp1))
    
    output = z * ((kstar ** alpha) * (lstar ** (1-alpha))) ** span
    profit = output - wage * lstar - rental * kstar
    return profit, kstar, lstar, output

@njit(cache=False)
def solve_entrepreneur_with_tau(a, z, tau, w, r, lam, delta, alpha, upsilon):
    """Static profit maximization WITH distortion tau.

    Distortion applies to output: profit = (1-tau)*y - wl - (r+delta)*k
    This is equivalent to solving with z_eff = (1-tau)*z
    """
    # Effective productivity after tax
    z_eff = (1.0 - tau) * z
    if z_eff <= 0:
        return -1e10, 0.0, 0.0, 0.0

    rental = max(r + delta, 1e-8)
    wage = max(w, 1e-8)
    span = 1 - upsilon

    # Use z_eff in FOCs (both capital and labor)
    aux1 = (1/rental) * alpha * span * z_eff
    aux2 = (1/wage) * (1-alpha) * span * z_eff
    exp1 = 1 - (1-alpha) * span
    exp2 = (1-alpha) * span
    k1 = (aux1 ** exp1 * aux2 ** exp2) ** (1/upsilon)

    kstar = min(k1, lam * a)
    lstar = ((aux2 * (kstar ** (alpha * span))) ** (1/exp1))

    # Output uses z_eff: y_eff = (1-tau)*z*(k^a*l^(1-a))^span
    output = z_eff * ((kstar ** alpha) * (lstar ** (1-alpha))) ** span
    # Profit = y_eff - costs = (1-tau)*z*(...) - wl - (r+delta)*k
    profit = output - wage * lstar - rental * kstar
    return profit, kstar, lstar, output

@njit(cache=False, parallel=True)
def precompute_entrepreneur_all(a_grid, z_grid, w, r, lam, delta, alpha, upsilon):
    """Vectorized precomputation for VFI"""
    n_a, n_z = len(a_grid), len(z_grid)
    profit_g = np.zeros((n_a, n_z))
    kstar_g = np.zeros((n_a, n_z))
    lstar_g = np.zeros((n_a, n_z))
    output_g = np.zeros((n_a, n_z))
    is_entrep_g = np.zeros((n_a, n_z))
    income_g = np.zeros((n_a, n_z))

    for i_z in prange(n_z):
        z = z_grid[i_z]
        for i_a in range(n_a):
            p, k, l, o = solve_entrepreneur_single(a_grid[i_a], z, w, r, lam, delta, alpha, upsilon)
            profit_g[i_a, i_z] = p
            kstar_g[i_a, i_z] = k
            lstar_g[i_a, i_z] = l
            output_g[i_a, i_z] = o
            if p > w:
                is_entrep_g[i_a, i_z] = 1.0
                income_g[i_a, i_z] = p + (1 + r) * a_grid[i_a]
            else:
                is_entrep_g[i_a, i_z] = 0.0
                income_g[i_a, i_z] = w + (1 + r) * a_grid[i_a]
    return profit_g, kstar_g, lstar_g, output_g, is_entrep_g, income_g

@njit(cache=False, parallel=True)
def precompute_entrepreneur_with_tau(a_grid, z_grid, tau, w, r, lam, delta, alpha, upsilon):
    """Vectorized precomputation WITH distortion"""
    n_a, n_z = len(a_grid), len(z_grid)
    profit_g = np.zeros((n_a, n_z))
    kstar_g = np.zeros((n_a, n_z))
    lstar_g = np.zeros((n_a, n_z))
    output_g = np.zeros((n_a, n_z))
    is_entrep_g = np.zeros((n_a, n_z))
    income_g = np.zeros((n_a, n_z))

    for i_z in prange(n_z):
        z = z_grid[i_z]
        for i_a in range(n_a):
            p, k, l, o = solve_entrepreneur_with_tau(a_grid[i_a], z, tau, w, r, lam, delta, alpha, upsilon)
            profit_g[i_a, i_z] = p
            kstar_g[i_a, i_z] = k
            lstar_g[i_a, i_z] = l
            output_g[i_a, i_z] = o
            if p > w:
                is_entrep_g[i_a, i_z] = 1.0
                income_g[i_a, i_z] = p + (1 + r) * a_grid[i_a]
            else:
                is_entrep_g[i_a, i_z] = 0.0
                income_g[i_a, i_z] = w + (1 + r) * a_grid[i_a]
    return profit_g, kstar_g, lstar_g, output_g, is_entrep_g, income_g

# =============================================================================
# Value Function Iteration (Optimized from V3)
# =============================================================================

@njit(cache=False)
def utility(c, sigma):
    if c <= 1e-10: return -1e10
    if abs(sigma - 1.0) < 1e-6: return np.log(c)
    return (c**(1-sigma) - 1) / (1-sigma)

@njit(cache=False)
def find_optimal_savings(income, a_grid, EV_row, beta, sigma, start_idx):
    n_a = len(a_grid)
    best_val = -1e15
    best_idx = start_idx
    for i_a_prime in range(start_idx, n_a):
        c = income - a_grid[i_a_prime]
        if c <= 1e-10: break
        val = utility(c, sigma) + beta * EV_row[i_a_prime]
        if val > best_val:
            best_val, best_idx = val, i_a_prime
    return best_val, best_idx

@njit(cache=False, parallel=True)
def bellman_operator(V, a_grid, z_grid, prob_z, income_grid, beta, sigma, psi):
    n_a, n_z = len(a_grid), len(z_grid)
    V_new = np.zeros((n_a, n_z))
    policy_a_idx = np.zeros((n_a, n_z), dtype=np.int64)
    
    # Expected value
    V_mean = np.zeros(n_a)
    for i_a in range(n_a):
        for i_z in range(n_z):
            V_mean[i_a] += prob_z[i_z] * V[i_a, i_z]
            
    for i_z in prange(n_z):
        start_idx = 0
        for i_a in range(n_a):
            income = income_grid[i_a, i_z]
            EV_row = psi * V[:, i_z] + (1 - psi) * V_mean
            v, idx = find_optimal_savings(income, a_grid, EV_row, beta, sigma, start_idx)
            V_new[i_a, i_z], policy_a_idx[i_a, i_z] = v, idx
            start_idx = idx
    return V_new, policy_a_idx

@njit(cache=False, parallel=True)
def howard_acceleration(V, policy_a_idx, a_grid, z_grid, prob_z, income_grid, beta, sigma, psi, n_howard=15):
    n_a, n_z = len(a_grid), len(z_grid)
    for _ in range(n_howard):
        V_mean = np.zeros(n_a)
        for i_a in range(n_a):
            for i_z in range(n_z):
                V_mean[i_a] += prob_z[i_z] * V[i_a, i_z]
        V_next = np.zeros((n_a, n_z))
        for i_z in prange(n_z):
            for i_a in range(n_a):
                idx_prime = policy_a_idx[i_a, i_z]
                c = income_grid[i_a, i_z] - a_grid[idx_prime]
                V_next[i_a, i_z] = utility(c, sigma) + beta * (psi * V[idx_prime, i_z] + (1 - psi) * V_mean[idx_prime])
        V = V_next
    return V

def solve_value_function(a_grid, z_grid, prob_z, income_grid, beta, sigma, psi, V_init=None, tol=1e-5, max_iter=500, n_howard=15):
    V = V_init.copy() if V_init is not None else np.zeros((len(a_grid), len(z_grid)))
    if V_init is None:
        for i_a in range(len(a_grid)):
            c_guess = max(income_grid[i_a, 0] - a_grid[0], 0.01)
            V[i_a, :] = utility(c_guess, sigma) / (1 - beta)
            
    for iteration in range(max_iter):
        V_new, policy_idx = bellman_operator(V, a_grid, z_grid, prob_z, income_grid, beta, sigma, psi)
        diff = np.max(np.abs(V_new - V))
        if diff < tol: return V_new, policy_idx
        if n_howard > 0 and diff > tol*10:
            V = howard_acceleration(V_new, policy_idx, a_grid, z_grid, prob_z, income_grid, beta, sigma, psi, n_howard)
        else:
            V = V_new
    return V, policy_idx

def solve_value_function_coupled(a_grid, z_grid, prob_z, prob_tau_plus, inc_p, inc_m, beta, sigma, psi, V_p_init=None, V_m_init=None, tol=1e-5, max_iter=500, n_howard=15):
    V_p = V_p_init.copy() if V_p_init is not None else np.zeros((len(a_grid), len(z_grid)))
    V_m = V_m_init.copy() if V_m_init is not None else np.zeros((len(a_grid), len(z_grid)))
    
    if V_p_init is None:
        for i_a in range(len(a_grid)):
            c_p_guess = max(inc_p[i_a, 0] - a_grid[0], 0.01)
            c_m_guess = max(inc_m[i_a, 0] - a_grid[0], 0.01)
            V_p[i_a, :] = utility(c_p_guess, sigma) / (1 - beta)
            V_m[i_a, :] = utility(c_m_guess, sigma) / (1 - beta)

    for iteration in range(max_iter):
        V_p_new, V_m_new, pol_p, pol_m = coupled_bellman_operator(V_p, V_m, a_grid, z_grid, prob_z, prob_tau_plus, inc_p, inc_m, beta, sigma, psi)
        diff = max(np.max(np.abs(V_p_new - V_p)), np.max(np.abs(V_m_new - V_m)))
        if diff < tol: return V_p_new, V_m_new, pol_p, pol_m
        if n_howard > 0 and diff > tol*10:
            V_p, V_m = coupled_howard_acceleration(V_p_new, V_m_new, pol_p, pol_m, a_grid, z_grid, prob_z, prob_tau_plus, inc_p, inc_m, beta, sigma, psi, n_howard)
        else:
            V_p, V_m = V_p_new, V_m_new
    return V_p, V_m, pol_p, pol_m

# =============================================================================
# Coupled VFI (For Pre-Reform Distortions)
# =============================================================================

@njit(cache=False, parallel=True)
def coupled_bellman_operator(V_p, V_m, a_grid, z_grid, prob_z, prob_tau_plus, inc_p, inc_m, beta, sigma, psi):
    n_a, n_z = len(a_grid), len(z_grid)
    V_p_new, V_m_new = np.zeros((n_a, n_z)), np.zeros((n_a, n_z))
    pol_p_idx, pol_m_idx = np.zeros((n_a, n_z), dtype=np.int64), np.zeros((n_a, n_z), dtype=np.int64)
    
    # Joint Expected value across tau states
    EV = np.zeros(n_a)
    for i_a in range(n_a):
        for i_z in range(n_z):
            # E[V | a', z', tau'] = prob_z * (prob_tau_plus*V_p + (1-prob_tau_plus)*V_m)
            EV[i_a] += prob_z[i_z] * (prob_tau_plus[i_z]*V_p[i_a, i_z] + (1-prob_tau_plus[i_z])*V_m[i_a, i_z])
            
    for i_z in prange(n_z):
        start_p, start_m = 0, 0
        EV_row_p = psi * V_p[:, i_z] + (1 - psi) * EV
        EV_row_m = psi * V_m[:, i_z] + (1 - psi) * EV
        for i_a in range(n_a):
            v_p, idx_p = find_optimal_savings(inc_p[i_a, i_z], a_grid, EV_row_p, beta, sigma, start_p)
            V_p_new[i_a, i_z], pol_p_idx[i_a, i_z] = v_p, idx_p
            start_p = idx_p
            
            v_m, idx_m = find_optimal_savings(inc_m[i_a, i_z], a_grid, EV_row_m, beta, sigma, start_m)
            V_m_new[i_a, i_z], pol_m_idx[i_a, i_z] = v_m, idx_m
            start_m = idx_m
            
    return V_p_new, V_m_new, pol_p_idx, pol_m_idx

@njit(cache=False, parallel=True)
def coupled_howard_acceleration(V_p, V_m, pol_p_idx, pol_m_idx, a_grid, z_grid, prob_z, prob_tau_plus, inc_p, inc_m, beta, sigma, psi, n_howard=15):
    n_a, n_z = len(a_grid), len(z_grid)
    for _ in range(n_howard):
        EV = np.zeros(n_a)
        for i_a in range(n_a):
            for i_z in range(n_z):
                EV[i_a] += prob_z[i_z] * (prob_tau_plus[i_z]*V_p[i_a, i_z] + (1-prob_tau_plus[i_z])*V_m[i_a, i_z])
        V_p_next, V_m_next = np.zeros((n_a, n_z)), np.zeros((n_a, n_z))
        for i_z in prange(n_z):
            for i_a in range(n_a):
                idx_p, idx_m = pol_p_idx[i_a, i_z], pol_m_idx[i_a, i_z]
                c_p = inc_p[i_a, i_z] - a_grid[idx_p]
                V_p_next[i_a, i_z] = utility(c_p, sigma) + beta * (psi * V_p[idx_p, i_z] + (1-psi) * EV[idx_p])
                c_m = inc_m[i_a, i_z] - a_grid[idx_m]
                V_m_next[i_a, i_z] = utility(c_m, sigma) + beta * (psi * V_m[idx_m, i_z] + (1-psi) * EV[idx_m])
        V_p, V_m = V_p_next, V_m_next
    return V_p, V_m

# =============================================================================
# Simulation Routines (Simplified from V4)
# =============================================================================

@njit(cache=False)
def get_interp_weights(x, x_grid):
    n = len(x_grid)
    if x <= x_grid[0]: return 0, 0, 1.0
    if x >= x_grid[-1]: return n-2, n-1, 0.0
    low, high = 0, n-1
    while high - low > 1:
        mid = (low + high) // 2
        if x_grid[mid] > x: high = mid
        else: low = mid
    return low, high, (x_grid[high] - x)/(x_grid[high] - x_grid[low])

@njit(cache=False, parallel=True)
def simulate_step(a_curr, z_idx_curr, policy_a_vals, a_grid, reset_shocks, ability_shocks):
    n = len(a_curr)
    a_next = np.zeros(n)
    z_idx_next = np.zeros(n, dtype=np.int64)
    for i in prange(n):
        z_idx_next[i] = ability_shocks[i] if reset_shocks[i] else z_idx_curr[i]
        i_l, i_h, w_l = get_interp_weights(a_curr[i], a_grid)
        a_next[i] = w_l * policy_a_vals[i_l, z_idx_curr[i]] + (1-w_l) * policy_a_vals[i_h, z_idx_curr[i]]
    return a_next, z_idx_next

@njit(cache=False, parallel=True)
def simulate_step_coupled(a_curr, z_idx_curr, tau_curr, pol_p_vals, pol_m_vals, a_grid, resets, shocks, tau_shocks, tau_p, tau_m):
    n = len(a_curr)
    a_next = np.zeros(n)
    z_idx_next = np.zeros(n, dtype=np.int64)
    tau_next = np.zeros(n)
    for i in prange(n):
        if resets[i]:
            z_idx_next[i] = shocks[i]
            tau_next[i] = tau_p if tau_shocks[i] else tau_m
        else:
            z_idx_next[i] = z_idx_curr[i]
            tau_next[i] = tau_curr[i]
            
        i_l, i_h, w_l = get_interp_weights(a_curr[i], a_grid)
        if tau_curr[i] == tau_p:
            a_next[i] = w_l * pol_p_vals[i_l, z_idx_curr[i]] + (1-w_l) * pol_p_vals[i_h, z_idx_curr[i]]
        else:
            a_next[i] = w_l * pol_m_vals[i_l, z_idx_curr[i]] + (1-w_l) * pol_m_vals[i_h, z_idx_curr[i]]
    return a_next, z_idx_next, tau_next

@njit(cache=False, parallel=True)
def compute_sim_aggregates(a_sim, z_idx_sim, z_grid, w, r, lam, delta, alpha, upsilon, tau=0.0):
    n = len(a_sim)
    K, L, Y, A, extfin, s_e = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    rental = max(r + delta, 1e-8)
    wage = max(w, 1e-8)
    for i in prange(n):
        a = a_sim[i]
        z = z_grid[z_idx_sim[i]]
        A += a
        p, k, l, o = solve_entrepreneur_with_tau(a, z, tau, w, r, lam, delta, alpha, upsilon)
        if p > wage:
            K += k
            L += l
            Y += o
            extfin += max(0.0, k - a)
            s_e += 1.0
    return K/n, L/n, Y/n, A/n, extfin/n, s_e/n

def compute_sim_aggregates_extended(a_sim, z_idx_sim, z_grid, w, r, lam, delta, alpha, upsilon, tau=0.0):
    """Compute aggregates plus avg entrepreneur z and wealth concentration."""
    n = len(a_sim)
    K, L, Y, A, extfin, s_e = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    sum_z_entrep = 0.0
    wage = max(w, 1e-8)

    for i in range(n):
        a = a_sim[i]
        z = z_grid[z_idx_sim[i]]
        A += a
        p, k, l, o = solve_entrepreneur_with_tau(a, z, tau, w, r, lam, delta, alpha, upsilon)
        if p > wage:
            K += k
            L += l
            Y += o
            extfin += max(0.0, k - a)
            s_e += 1.0
            sum_z_entrep += z

    # Wealth concentration: top 5% share
    sorted_a = np.sort(a_sim)[::-1]
    top5_idx = max(1, int(0.05 * n))
    wealth_top5 = np.sum(sorted_a[:top5_idx]) / max(np.sum(a_sim), 1e-10)

    avg_z_entrep = sum_z_entrep / max(s_e, 1.0)

    return K/n, L/n, Y/n, A/n, extfin/n, s_e/n, avg_z_entrep, wealth_top5

def generate_shocks(n_agents, t_steps, psi, prob_z):
    np.random.seed(42)
    resets = (np.random.rand(t_steps, n_agents) > psi).astype(np.uint8)
    cdf_z = np.cumsum(prob_z)
    shocks = np.searchsorted(cdf_z, np.random.rand(t_steps, n_agents)).astype(np.uint8)
    return resets, shocks

def generate_shocks_with_tau(n_agents, t_steps, psi, prob_z, prob_tau_plus):
    resets, shocks = generate_shocks(n_agents, t_steps, psi, prob_z)
    # Pre-generate tax state transitions for resets
    tau_shocks = np.zeros((t_steps, n_agents), dtype=np.uint8)
    for t in range(t_steps):
        # Probability depends on the specific ability shock chosen for that agent
        probs = prob_tau_plus[shocks[t]]
        tau_shocks[t] = (np.random.rand(n_agents) < probs).astype(np.uint8)
    return resets, shocks, tau_shocks

@njit(cache=False, parallel=True)
def compute_sim_aggregates_with_dist(a_sim, z_idx_sim, tau_sim, z_grid, w, r, lam, delta, alpha, upsilon):
    n = len(a_sim)
    K, L, Y, A, extfin, s_e = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    wage = max(w, 1e-8)
    for i in prange(n):
        a = a_sim[i]
        z = z_grid[z_idx_sim[i]]
        tau = tau_sim[i]
        A += a
        p, k, l, o = solve_entrepreneur_with_tau(a, z, tau, w, r, lam, delta, alpha, upsilon)
        if p > wage:
            K += k
            L += l
            Y += o
            extfin += max(0.0, k - a)
            s_e += 1.0
    return K/n, L/n, Y/n, A/n, extfin/n, s_e/n

def find_equilibrium_nodist(a_grid, z_grid, prob_z, params, w_init=0.8, r_init=-0.04,
                            max_iter=100, tol=1e-3, verbose=True):
    """Find stationary equilibrium WITHOUT distortions using simultaneous price updates."""
    delta, alpha, upsilon, lam, beta, sigma, psi = params
    w, r = w_init, r_init
    V = None
    resets, shocks = generate_shocks(N_AGENTS, T_SIM_SS, PSI, prob_z)

    # Price adjustment parameters
    w_step, r_step = 0.3, 0.05
    exc_L_prev, exc_K_prev = 0.0, 0.0
    w_prev, r_prev = w, r
    best_error = np.inf
    best_result = None

    for iteration in range(max_iter):
        # 1. Pre-compute entrepreneur solutions
        _, _, _, _, _, inc = precompute_entrepreneur_all(a_grid, z_grid, w, r, lam, delta, alpha, upsilon)

        # 2. Solve value function
        V, pol_idx = solve_value_function(a_grid, z_grid, prob_z, inc, beta, sigma, psi, V_init=V)

        # 3. Simulate to find distribution
        np.random.seed(123)
        a_curr = np.ones(N_AGENTS) * a_grid[0]
        z_idx_curr = np.random.randint(0, len(z_grid), N_AGENTS)
        pol_vals = a_grid[pol_idx]
        for t in range(T_SIM_SS):
            a_curr, z_idx_curr = simulate_step(a_curr, z_idx_curr, pol_vals, a_grid, resets[t], shocks[t])

        # 4. Compute aggregates (extended version for additional stats)
        K, L, Y, A, extfin, s_e, avg_z_e, wealth_top5 = compute_sim_aggregates_extended(
            a_curr, z_idx_curr, z_grid, w, r, lam, delta, alpha, upsilon)

        exc_K = K - A
        exc_L = L - (1 - s_e)
        total_error = abs(exc_L) + abs(exc_K)

        if verbose:
            print(f"  [Post SS {iteration+1:3d}] w={w:.6f} r={r:.6f} | ExcL={exc_L:.4f} ExcK={exc_K:.4f}")

        # Track best result
        if total_error < best_error:
            best_error = total_error
            best_result = {'w': w, 'r': r, 'Y': Y, 'K': K, 'L': L, 'A': A,
                          'extfin': extfin, 's_e': s_e, 'V': V.copy(), 'pol_idx': pol_idx.copy(),
                          'a_sim': a_curr.copy(), 'z_idx_sim': z_idx_curr.copy(),
                          'avg_z_entrep': avg_z_e, 'wealth_top5_share': wealth_top5}

        # Convergence check 1: excess demands small
        if abs(exc_L) < tol and abs(exc_K) < tol:
            if verbose: print(f"  Converged (excess demands)!")
            break

        # Convergence check 2: prices stabilized (simulation noise floor reached)
        price_change = abs(w - w_prev)/max(w, 1e-6) + abs(r - r_prev)/max(abs(r), 1e-6)
        if iteration > 5 and price_change < 1e-6 and (w_step < 0.01 or r_step < 0.005):
            if verbose: print(f"  Converged (prices stabilized, simulation noise floor)!")
            break

        # Adaptive damping (reduce step if sign flips)
        if iteration > 0:
            if exc_L * exc_L_prev < 0: w_step *= 0.5
            if exc_K * exc_K_prev < 0: r_step *= 0.5

        # Simultaneous price updates
        w_new = w * (1 + w_step * exc_L)
        r_new = r + r_step * exc_K

        # Bounds
        w_new = max(0.01, min(2.5, w_new))
        r_new = max(-0.25, min(1/beta - 1 - 1e-6, r_new))

        # Damped update
        damping = DAMPING_FACTOR
        w_prev, r_prev = w, r
        w = damping * w + (1 - damping) * w_new
        r = damping * r + (1 - damping) * r_new

        exc_L_prev, exc_K_prev = exc_L, exc_K

    # Use best result
    res = best_result
    TFP = res['Y'] / max(((res['K']**alpha) * ((1-res['s_e'])**(1-alpha)))**(1-upsilon), 1e-8)

    return {'w': res['w'], 'r': res['r'], 'Y': res['Y'], 'K': res['K'], 'L': res['L'], 'A': res['A'],
            'TFP': TFP, 'share_entre': res['s_e'], 'extfin_Y': res['extfin']/res['Y'] if res['Y'] > 0 else 0,
            'V': res['V'], 'pol_idx': res['pol_idx'], 'a_sim': res['a_sim'], 'z_idx_sim': res['z_idx_sim']}

def find_equilibrium_with_dist(a_grid, z_grid, prob_z, prob_tau_plus, params, tau_p, tau_m,
                                w_init=0.55, r_init=-0.04, max_iter=100, tol=1e-3, verbose=True):
    """Find stationary equilibrium WITH distortions using simultaneous price updates."""
    delta, alpha, upsilon, lam, beta, sigma, psi = params
    w, r = w_init, r_init
    V_p, V_m = None, None
    resets, shocks, tau_shocks = generate_shocks_with_tau(N_AGENTS, T_SIM_SS, PSI, prob_z, prob_tau_plus)

    # Price adjustment parameters
    w_step, r_step = 0.3, 0.05
    exc_L_prev, exc_K_prev = 0.0, 0.0
    w_prev, r_prev = w, r
    best_error = np.inf
    best_result = None

    for iteration in range(max_iter):
        # 1. Pre-compute entrepreneur solutions for both tau states
        _, _, _, _, _, inc_p = precompute_entrepreneur_with_tau(a_grid, z_grid, tau_p, w, r, lam, delta, alpha, upsilon)
        _, _, _, _, _, inc_m = precompute_entrepreneur_with_tau(a_grid, z_grid, tau_m, w, r, lam, delta, alpha, upsilon)

        # 2. Solve coupled value function
        V_p, V_m, pol_p_idx, pol_m_idx = solve_value_function_coupled(
            a_grid, z_grid, prob_z, prob_tau_plus, inc_p, inc_m, beta, sigma, psi,
            V_p_init=V_p, V_m_init=V_m, n_howard=15
        )

        # 3. Simulate to find distribution
        np.random.seed(123)
        a_curr = np.ones(N_AGENTS) * a_grid[0]
        z_idx_curr = np.random.randint(0, len(z_grid), N_AGENTS)
        tau_curr = np.array([tau_p if np.random.rand() < prob_tau_plus[z_idx_curr[i]] else tau_m for i in range(N_AGENTS)])
        pol_p_vals, pol_m_vals = a_grid[pol_p_idx], a_grid[pol_m_idx]
        for t in range(T_SIM_SS):
            a_curr, z_idx_curr, tau_curr = simulate_step_coupled(
                a_curr, z_idx_curr, tau_curr, pol_p_vals, pol_m_vals, a_grid,
                resets[t], shocks[t], tau_shocks[t], tau_p, tau_m
            )

        # 4. Compute aggregates
        K, L, Y, A, extfin, s_e = compute_sim_aggregates_with_dist(
            a_curr, z_idx_curr, tau_curr, z_grid, w, r, lam, delta, alpha, upsilon
        )

        exc_K = K - A
        exc_L = L - (1 - s_e)
        total_error = abs(exc_L) + abs(exc_K)

        if verbose:
            print(f"  [Pre SS {iteration+1:3d}] w={w:.6f} r={r:.6f} | ExcL={exc_L:.4f} ExcK={exc_K:.4f}")

        # Track best result
        if total_error < best_error:
            best_error = total_error
            best_result = {'w': w, 'r': r, 'Y': Y, 'K': K, 'L': L, 'A': A,
                          'extfin': extfin, 's_e': s_e, 'V_p': V_p.copy(), 'V_m': V_m.copy(),
                          'a_sim': a_curr.copy(), 'z_idx_sim': z_idx_curr.copy(), 'tau_sim': tau_curr.copy()}

        # Convergence check 1: excess demands small
        if abs(exc_L) < tol and abs(exc_K) < tol:
            if verbose: print(f"  Converged (excess demands)!")
            break

        # Convergence check 2: prices stabilized (simulation noise floor reached)
        price_change = abs(w - w_prev)/max(w, 1e-6) + abs(r - r_prev)/max(abs(r), 1e-6)
        if iteration > 5 and price_change < 1e-6 and (w_step < 0.01 or r_step < 0.005):
            if verbose: print(f"  Converged (prices stabilized, simulation noise floor)!")
            break

        # Handle collapse case
        if L < 1e-4 or Y < 1e-6:
            w *= 0.8
            if verbose: print(f"  [WARNING] Collapse detected, reducing w")
            continue

        # Adaptive damping (reduce step if sign flips)
        if iteration > 0:
            if exc_L * exc_L_prev < 0: w_step *= 0.5
            if exc_K * exc_K_prev < 0: r_step *= 0.5

        # Simultaneous price updates
        w_new = w * (1 + w_step * exc_L)
        r_new = r + r_step * exc_K

        # Bounds
        w_new = max(0.01, min(2.5, w_new))
        r_new = max(-0.25, min(1/beta - 1 - 1e-6, r_new))

        # Damped update
        damping = DAMPING_FACTOR
        w_prev, r_prev = w, r
        w = damping * w + (1 - damping) * w_new
        r = damping * r + (1 - damping) * r_new

        exc_L_prev, exc_K_prev = exc_L, exc_K

    # Use best result
    res = best_result
    TFP = res['Y'] / max(((res['K']**alpha) * ((1-res['s_e'])**(1-alpha)))**(1-upsilon), 1e-8)

    return {'w': res['w'], 'r': res['r'], 'Y': res['Y'], 'K': res['K'], 'L': res['L'], 'A': res['A'],
            'TFP': TFP, 'share_entre': res['s_e'], 'extfin_Y': res['extfin']/res['Y'] if res['Y'] > 0 else 0,
            'V_p': res['V_p'], 'V_m': res['V_m'], 'a_sim': res['a_sim'], 'z_idx_sim': res['z_idx_sim'], 'tau_sim': res['tau_sim']}

def solve_transition(pre_eq, post_eq, params, a_grid, z_grid, prob_z, T=50):
    """Solve transition path using simulation."""
    delta, alpha, upsilon, lam, beta, sigma, psi = params
    w_path = np.linspace(pre_eq['w'], post_eq['w'], T)
    r_path = np.linspace(pre_eq['r'], post_eq['r'], T)
    
    resets, shocks = generate_shocks(N_AGENTS, T, PSI, prob_z)
    
    # Adaptive damping state
    w_step_path = 0.1 * np.ones(T)
    r_step_path = 0.01 * np.ones(T)
    last_ED_L = np.zeros(T)
    last_ED_K = np.zeros(T)
    
    for it in range(50):
        # 1. Backward Pass
        V_path = [None] * T
        V_path[T-1] = post_eq['V']
        pol_idx_path = [None] * T
        
        for t in range(T-2, -1, -1):
            _, _, _, _, _, inc = precompute_entrepreneur_all(a_grid, z_grid, w_path[t], r_path[t], lam, delta, alpha, upsilon)
            V_path[t], pol_idx_path[t] = bellman_operator(V_path[t+1], a_grid, z_grid, prob_z, inc, beta, sigma, psi)
            
        # 2. Forward Pass (Simulation)
        a_curr, z_idx_curr = pre_eq['a_sim'], pre_eq['z_idx_sim']
        
        ED_L, ED_K = np.zeros(T), np.zeros(T)
        Y_path, K_path, TFP_path = [], [], []
        
        for t in range(T):
            pol_vals = a_grid[pol_idx_path[t]] if pol_idx_path[t] is not None else a_grid[post_eq['pol_idx']]
            a_curr, z_idx_curr = simulate_step(a_curr, z_idx_curr, pol_vals, a_grid, resets[t], shocks[t])
            
            K, L, Y, A, extfin, s_e = compute_sim_aggregates(a_curr, z_idx_curr, z_grid, w_path[t], r_path[t], lam, delta, alpha, upsilon)
            ED_L[t] = L - (1-s_e)
            ED_K[t] = K - A
            Y_path.append(Y); K_path.append(K); TFP_path.append(Y/max(((K**alpha) * ((1-s_e)**(1-alpha)))**(1-upsilon), 1e-8))
            
        max_ED = max(np.max(np.abs(ED_L)), np.max(np.abs(ED_K)))
        print(f"  [TPI {it:2d}] max|ED_L|={np.max(np.abs(ED_L)):.4f} max|ED_K|={np.max(np.abs(ED_K)):.4f}")
        if max_ED < 1e-3: break
        
        # Adaptive damping for whole path (per-period t)
        if it > 0:
            for t in range(T):
                if np.sign(ED_L[t]) != np.sign(last_ED_L[t]): w_step_path[t] *= 0.5
                if np.sign(ED_K[t]) != np.sign(last_ED_K[t]): r_step_path[t] *= 0.5
                
                # Enforce lower bounds to prevent stalling
                w_step_path[t] = max(w_step_path[t], 1e-4)
                r_step_path[t] = max(r_step_path[t], 1e-5)
        
        last_ED_L[:] = ED_L[:]
        last_ED_K[:] = ED_K[:]
        
        w_path_target = w_path + w_step_path * ED_L
        r_path_target = r_path + r_step_path * ED_K
        
        # Guards on target
        w_path_target = np.maximum(w_path_target, 1e-4)
        r_path_target = np.maximum(np.minimum(r_path_target, 1/beta - 1 - 1e-6), -0.5)
        
        # Damped Update
        damping = DAMPING_FACTOR
        w_path = damping * w_path + (1 - damping) * w_path_target
        r_path = damping * r_path + (1 - damping) * r_path_target
        
    return {'t': np.arange(T), 'w': w_path, 'r': r_path, 'Y': np.array(Y_path), 'K': np.array(K_path), 'TFP': np.array(TFP_path)}

def plot_transition(pre_eq, post_eq, trans, output_dir='outputs'):
    os.makedirs(output_dir, exist_ok=True)
    
    # Premium Styling
    plt.rcParams.update({
        'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12,
        'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 10,
        'figure.titlesize': 16, 'grid.alpha': 0.3, 'lines.linewidth': 2.5
    })

    # Prepare V4 Data (Simulation)
    T = len(trans['t'])
    t_plot = trans['t']
    
    # Normalization by Pre-Reform Steady State
    norm_Y = trans['Y'] / pre_eq['Y']
    norm_TFP = trans['TFP'] / pre_eq['TFP']
    norm_K = trans['K'] / pre_eq['K']
    norm_w = trans['w'] / pre_eq['w']
    # For interest rate, we can plot difference r_t - r_pre or just raw levels. 
    # Usually papers plot raw levels or diff. User asked for "normalized by pre-ss value", but r can be negative.
    # Div by negative is confusing. Let's keep r as levels for clarity unless strictly forced. 
    # Actually user said "all variables ... normalized", let's assume levels for rates is standard unless specified.
    # Re-reading prompt: "normalized by their pre-steady-state value".
    # If r_pre is -0.04, then r_t/-0.04 flips sign. 
    # Let's stick to levels for r (or 1+r gross return). 
    # Paper plots typically show levels for r, and relative for Y, K, TFP, w.
    # I will stick to relative for positive vars, and levels for r to avoid sign flip confusion.
    # Update: User specifically said "All variables". 
    # Let's do (1+r_t)/(1+r_pre) for interest rate to be safe? Or just levels.
    # I'll stick to levels for r (path_r) but label it clearly, as dividing by small negative number is bad.
    
    path_r = trans['r'] 

    # Load V3 Data (Howard PI Comparison)
    v3_data = None
    v3_path = os.path.join(output_dir, 'transition_v3.csv')
    if os.path.exists(v3_path):
        try:
            import csv
            v3_t, v3_Y, v3_K, v3_TFP, v3_w, v3_r = [], [], [], [], [], []
            with open(v3_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if float(row['t']) < T:
                        v3_t.append(float(row['t']))
                        v3_Y.append(float(row['Y']))
                        v3_K.append(float(row['K']))
                        v3_TFP.append(float(row['TFP']))
                        v3_w.append(float(row['w']))
                        v3_r.append(float(row['r']))
            
            # Normalize V3 data by V4 pre_eq to be comparable relative to the SAME baseline
            # (Assuming V3 and V4 start from similar pre-SS)
            if len(v3_Y) > 0:
                v3_data = {
                    't': np.array(v3_t),
                    'Y': np.array(v3_Y) / pre_eq['Y'],
                    'K': np.array(v3_K) / pre_eq['K'],
                    'TFP': np.array(v3_TFP) / pre_eq['TFP'],
                    'w': np.array(v3_w) / pre_eq['w'],
                    'r': np.array(v3_r)
                }
                print(f"Loaded V3 comparison data ({len(v3_t)} periods)")
        except Exception as e:
            print(f"Failed to load V3 comparison: {e}")

    # Plotting
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Output
    ax = axes[0,0]
    ax.plot(t_plot, norm_Y, 'b-', label='Sim (V4)')
    if v3_data: ax.plot(v3_data['t'], v3_data['Y'], 'r--', label='Howard (V3)')
    ax.set_title('Output (Y)'); ax.legend()
    
    # 2. TFP
    ax = axes[0,1]
    ax.plot(t_plot, norm_TFP, 'b-')
    if v3_data: ax.plot(v3_data['t'], v3_data['TFP'], 'r--')
    ax.set_title('TFP')
    
    # 3. Capital
    ax = axes[0,2]
    ax.plot(t_plot, norm_K, 'b-')
    if v3_data: ax.plot(v3_data['t'], v3_data['K'], 'r--')
    ax.set_title('Capital (K)')

    # 4. Wage
    ax = axes[1,0]
    ax.plot(t_plot, norm_w, 'b-')
    if v3_data: ax.plot(v3_data['t'], v3_data['w'], 'r--')
    ax.set_title('Wage (w)')

    # 5. Interest Rate
    ax = axes[1,1]
    ax.plot(t_plot, path_r, 'b-')
    if v3_data: ax.plot(v3_data['t'], v3_data['r'], 'r--')
    ax.set_title('Interest Rate (r)')

    # 6. Convergence Metrics (ED)
    # ax = axes[1,2] ... unused

    for ax in axes.flatten():
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Period')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'transition_sim_v4.png'), dpi=200)
    print(f"Plot saved to {os.path.join(output_dir, 'transition_sim_v4.png')}")

def main():
    parser = argparse.ArgumentParser(description="Buera-Shin Simulation-Based Transition")
    parser.add_argument("--T", type=int, default=125, help="Transition periods")
    parser.add_argument("--na", type=int, default=501, help="Asset grid points")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    z_grid, prob_z = create_ability_grid_paper(ETA)
    a_grid = create_asset_grid(args.na, 1e-6, 4000)
    prob_tau_plus = compute_tau_probs(z_grid, Q_DIST)
    params = (DELTA, ALPHA, NU, LAMBDA, BETA, SIGMA, PSI)
    
    print("Step 1: Post-reform (Simulation)")
    post_eq = find_equilibrium_nodist(a_grid, z_grid, prob_z, params)
    
    print("Step 2: Pre-reform (Simulation)")
    pre_eq = find_equilibrium_with_dist(a_grid, z_grid, prob_z, prob_tau_plus, params, TAU_PLUS, TAU_MINUS, w_init=0.55)
    
    if 'Y' not in pre_eq:
        print("Pre-reform failed. Stopping.")
        return

    print("Step 3: Transition (Simulation)")
    trans = solve_transition(pre_eq, post_eq, params, a_grid, z_grid, prob_z, T=args.T)
    
    plot_transition(pre_eq, post_eq, trans, args.output_dir)

if __name__ == "__main__":
    main()
