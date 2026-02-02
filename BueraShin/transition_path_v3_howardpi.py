"""
Buera & Shin (2010) Transition Dynamics with Idiosyncratic Distortions
VERSION: Howard Policy Iteration (based on buera_shin_v3_howard_pi.py)

This script builds on the working v3 Howard PI code and adds:
1. Distortions mode for pre-reform stationary equilibrium
2. Transition path computation via Time Path Iteration (TPI)

Computes:
1. Post-reform stationary equilibrium: lambda=1.35 without distortions (tau=0)
2. Pre-reform stationary equilibrium: lambda=1.35 with idiosyncratic output wedges
3. Perfect-foresight transition path

Usage:
    python transition_path_v3_howardpi.py --T 250
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numba import njit, prange
from scipy import sparse
import os
import json
import argparse
import time
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# Model Parameters (from v3 + distortion parameters)
# =============================================================================
SIGMA = 1.5      # Risk aversion
BETA = 0.904     # Discount factor
ALPHA = 0.33     # Capital share
NU = 0.21        # Entrepreneur share (Span of control = 0.79)
DELTA = 0.06     # Depreciation
ETA = 4.15       # Pareto tail
PSI = 0.894      # Persistence

# Financial friction
LAMBDA = 1.35    # Collateral constraint

# Distortion parameters
TAU_PLUS = 0.57      # Tax rate (positive wedge)
TAU_MINUS = -0.15    # Subsidy rate (negative wedge)
Q_DIST = 1.55        # Correlation: Pr(tau=tau_plus|e) = 1 - exp(-q*e)

# Grid parameters
N_A = 501
A_MIN = 1e-6
A_MAX = 4000

# =============================================================================
# Grid Construction (from v3)
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
    """Asset grid with curvature scaling"""
    a_grid = a_min + (a_max - a_min) * np.linspace(0, 1, n_a) ** 2
    a_grid[0] = max(a_grid[0], 1e-6)
    return a_grid

def compute_tau_probs(z_grid, q):
    """Probability of tau_plus given ability"""
    return 1 - np.exp(-q * z_grid)

# =============================================================================
# Entrepreneur Problem (with distortion support)
# =============================================================================

@njit(cache=True)
def solve_entrepreneur_single(a, z, w, r, lam, delta, alpha, upsilon):
    """Original entrepreneur problem (no distortion)"""
    rental = max(r + delta, 1e-8)
    wage = max(w, 1e-8)

    span = 1 - upsilon
    aux1 = (1/rental) * alpha * span * z
    aux2 = (1/wage) * (1-alpha) * span * z
    exp1 = 1 - (1-alpha) * span
    exp2 = (1-alpha) * span
    k1 = (aux1 ** exp1 * aux2 ** exp2) ** (1/upsilon)

    kstar = min(k1, lam * a)

    inside_lab = (1/wage) * (1-alpha) * span * z * (kstar ** (alpha * span))
    lstar = inside_lab ** (1/exp1)

    output = z * ((kstar ** alpha) * (lstar ** (1-alpha))) ** span
    profit = output - wage * lstar - rental * kstar

    return profit, kstar, lstar, output

@njit(cache=True)
def solve_entrepreneur_with_tau(a, z, tau, w, r, lam, delta, alpha, upsilon):
    """Entrepreneur problem WITH distortion tau"""
    z_eff = (1.0 - tau) * z
    if z_eff <= 0:
        return -1e10, 0.0, 0.0, 0.0

    rental = max(r + delta, 1e-8)
    wage = max(w, 1e-8)

    span = 1 - upsilon
    aux1 = (1/rental) * alpha * span * z_eff
    aux2 = (1/wage) * (1-alpha) * span * z_eff
    exp1 = 1 - (1-alpha) * span
    exp2 = (1-alpha) * span
    k1 = (aux1 ** exp1 * aux2 ** exp2) ** (1/upsilon)

    kstar = min(k1, lam * a)

    inside_lab = (1/wage) * (1-alpha) * span * z_eff * (kstar ** (alpha * span))
    lstar = inside_lab ** (1/exp1)

    output = z_eff * ((kstar ** alpha) * (lstar ** (1-alpha))) ** span
    profit = output - wage * lstar - rental * kstar

    return profit, kstar, lstar, output

@njit(cache=True, parallel=True)
def precompute_entrepreneur_all(a_grid, z_grid, w, r, lam, delta, alpha, upsilon):
    """Pre-compute entrepreneur solutions for ALL (a,z) pairs - NO distortion"""
    n_a = len(a_grid)
    n_z = len(z_grid)

    profit_grid = np.zeros((n_a, n_z))
    kstar_grid = np.zeros((n_a, n_z))
    lstar_grid = np.zeros((n_a, n_z))
    output_grid = np.zeros((n_a, n_z))
    is_entrep_grid = np.zeros((n_a, n_z))
    income_grid = np.zeros((n_a, n_z))

    for i_z in prange(n_z):
        z = z_grid[i_z]
        for i_a in range(n_a):
            a = a_grid[i_a]
            profit, kstar, lstar, output = solve_entrepreneur_single(
                a, z, w, r, lam, delta, alpha, upsilon
            )

            profit_grid[i_a, i_z] = profit
            kstar_grid[i_a, i_z] = kstar
            lstar_grid[i_a, i_z] = lstar
            output_grid[i_a, i_z] = output

            if profit > w:
                is_entrep_grid[i_a, i_z] = 1.0
                income_grid[i_a, i_z] = profit + (1 + r) * a
            else:
                is_entrep_grid[i_a, i_z] = 0.0
                income_grid[i_a, i_z] = w + (1 + r) * a

    return profit_grid, kstar_grid, lstar_grid, output_grid, is_entrep_grid, income_grid

@njit(cache=True, parallel=True)
def precompute_entrepreneur_with_tau(a_grid, z_grid, tau, w, r, lam, delta, alpha, upsilon):
    """Pre-compute entrepreneur solutions WITH distortion tau"""
    n_a = len(a_grid)
    n_z = len(z_grid)

    profit_grid = np.zeros((n_a, n_z))
    kstar_grid = np.zeros((n_a, n_z))
    lstar_grid = np.zeros((n_a, n_z))
    output_grid = np.zeros((n_a, n_z))
    is_entrep_grid = np.zeros((n_a, n_z))
    income_grid = np.zeros((n_a, n_z))

    for i_z in prange(n_z):
        z = z_grid[i_z]
        for i_a in range(n_a):
            a = a_grid[i_a]
            profit, kstar, lstar, output = solve_entrepreneur_with_tau(
                a, z, tau, w, r, lam, delta, alpha, upsilon
            )

            profit_grid[i_a, i_z] = profit
            kstar_grid[i_a, i_z] = kstar
            lstar_grid[i_a, i_z] = lstar
            output_grid[i_a, i_z] = output

            if profit > w:
                is_entrep_grid[i_a, i_z] = 1.0
                income_grid[i_a, i_z] = profit + (1 + r) * a
            else:
                is_entrep_grid[i_a, i_z] = 0.0
                income_grid[i_a, i_z] = w + (1 + r) * a

    return profit_grid, kstar_grid, lstar_grid, output_grid, is_entrep_grid, income_grid

# =============================================================================
# Value Function Iteration (from v3)
# =============================================================================

@njit(cache=True)
def utility(c, sigma):
    """CRRA utility"""
    if c <= 1e-10:
        return -1e10
    if abs(sigma - 1.0) < 1e-6:
        return np.log(c)
    return (c ** (1 - sigma) - 1) / (1 - sigma)

@njit(cache=True)
def find_optimal_savings_monotone(income, a_grid, EV_row, beta, sigma, start_idx):
    """Binary-like search exploiting monotonicity"""
    n_a = len(a_grid)
    best_val = -1e15
    best_idx = start_idx

    for i_a_prime in range(start_idx, n_a):
        c = income - a_grid[i_a_prime]
        if c <= 1e-10:
            break
        val = utility(c, sigma) + beta * EV_row[i_a_prime]
        if val > best_val:
            best_val = val
            best_idx = i_a_prime

    return best_val, best_idx

@njit(cache=True, parallel=True)
def bellman_operator_fast(V, a_grid, z_grid, prob_z, income_grid, beta, sigma, psi):
    """Fast Bellman operator with pre-computed income"""
    n_a = len(a_grid)
    n_z = len(z_grid)

    V_new = np.zeros((n_a, n_z))
    policy_a_idx = np.zeros((n_a, n_z), dtype=np.int64)

    # Pre-compute expected values
    EV = np.zeros((n_a, n_z))
    V_mean = np.zeros(n_a)
    for i_a in range(n_a):
        for i_z in range(n_z):
            V_mean[i_a] += prob_z[i_z] * V[i_a, i_z]

    for i_a in range(n_a):
        for i_z in range(n_z):
            EV[i_a, i_z] = psi * V[i_a, i_z] + (1 - psi) * V_mean[i_a]

    for i_z in prange(n_z):
        start_idx = 0

        for i_a in range(n_a):
            income = income_grid[i_a, i_z]

            best_val, best_idx = find_optimal_savings_monotone(
                income, a_grid, EV[:, i_z], beta, sigma, start_idx
            )

            V_new[i_a, i_z] = best_val
            policy_a_idx[i_a, i_z] = best_idx
            start_idx = best_idx

    return V_new, policy_a_idx

@njit(cache=True, parallel=True)
def howard_acceleration(V, policy_a_idx, a_grid, z_grid, prob_z,
                        income_grid, beta, sigma, psi, n_howard=10):
    """Howard's Policy Iteration Acceleration"""
    n_a = len(a_grid)
    n_z = len(z_grid)

    for _ in range(n_howard):
        EV = np.zeros((n_a, n_z))
        V_mean = np.zeros(n_a)
        for i_a in range(n_a):
            for i_z in range(n_z):
                V_mean[i_a] += prob_z[i_z] * V[i_a, i_z]

        for i_a in range(n_a):
            for i_z in range(n_z):
                EV[i_a, i_z] = psi * V[i_a, i_z] + (1 - psi) * V_mean[i_a]

        V_new = np.zeros((n_a, n_z))

        for i_z in prange(n_z):
            for i_a in range(n_a):
                income = income_grid[i_a, i_z]
                i_a_prime = policy_a_idx[i_a, i_z]
                c = income - a_grid[i_a_prime]
                V_new[i_a, i_z] = utility(c, sigma) + beta * EV[i_a_prime, i_z]

        V = V_new

    return V

def solve_value_function_fast(a_grid, z_grid, prob_z, income_grid,
                              beta, sigma, psi, V_init=None,
                              tol=1e-5, max_iter=500, n_howard=15):
    """VFI with Howard acceleration and warm start"""
    n_a, n_z = len(a_grid), len(z_grid)

    if V_init is not None:
        V = V_init.copy()
    else:
        V = np.zeros((n_a, n_z))
        for i_a in range(n_a):
            c_guess = max(income_grid[i_a, 0] - a_grid[0], 0.01)
            V[i_a, :] = utility(c_guess, sigma) / (1 - beta)

    for iteration in range(max_iter):
        V_new, policy_a_idx = bellman_operator_fast(
            V, a_grid, z_grid, prob_z, income_grid, beta, sigma, psi
        )

        diff = np.max(np.abs(V_new - V))

        if diff < tol:
            return V_new, policy_a_idx

        if n_howard > 0 and diff > tol * 10:
            V = howard_acceleration(V_new, policy_a_idx, a_grid, z_grid, prob_z,
                                   income_grid, beta, sigma, psi, n_howard)
        else:
            V = V_new

    return V, policy_a_idx

# =============================================================================
# Stationary Distribution (from v3)
# =============================================================================

@njit(cache=True)
def build_transition_sparse_data(policy_a_idx, n_a, n_z, prob_z, psi):
    """Build sparse transition matrix data"""
    nnz = n_a * n_z * n_z
    rows = np.zeros(nnz, dtype=np.int64)
    cols = np.zeros(nnz, dtype=np.int64)
    data = np.zeros(nnz)

    idx = 0
    for i_a in range(n_a):
        for i_z in range(n_z):
            s = i_a * n_z + i_z
            i_a_prime = policy_a_idx[i_a, i_z]

            for i_z_prime in range(n_z):
                if i_z_prime == i_z:
                    p = psi + (1 - psi) * prob_z[i_z_prime]
                else:
                    p = (1 - psi) * prob_z[i_z_prime]

                if p > 1e-14:
                    s_prime = i_a_prime * n_z + i_z_prime
                    rows[idx] = s_prime
                    cols[idx] = s
                    data[idx] = p
                    idx += 1

    return rows[:idx], cols[:idx], data[:idx]

def compute_stationary_distribution(policy_a_idx, a_grid, z_grid, prob_z, psi):
    """Compute stationary distribution"""
    n_a, n_z = len(a_grid), len(z_grid)
    n_states = n_a * n_z

    rows, cols, data = build_transition_sparse_data(
        policy_a_idx, n_a, n_z, prob_z, psi
    )

    Q = sparse.csr_matrix((data, (rows, cols)), shape=(n_states, n_states))

    mu = np.ones(n_states) / n_states
    for _ in range(1000):
        mu_new = Q @ mu
        mu_new /= mu_new.sum()
        if np.max(np.abs(mu_new - mu)) < 1e-14:
            break
        mu = mu_new

    return mu_new.reshape((n_a, n_z))

# =============================================================================
# Stationary Distribution WITH Distortions
# =============================================================================

@njit(cache=True)
def build_transition_sparse_data_with_dist(policy_plus_idx, policy_minus_idx,
                                            n_a, n_z, prob_z, prob_tau_plus, psi):
    """Build sparse transition for distortion case"""
    # Each of the n_a*n_z*2 states has:
    # 1 persistence entry + 2*n_z redraw entries (to tau_plus and tau_minus)
    nnz = n_a * n_z * 2 * (1 + 2 * n_z)
    rows = np.zeros(nnz, dtype=np.int64)
    cols = np.zeros(nnz, dtype=np.int64)
    data = np.zeros(nnz)

    idx = 0
    for i_a in range(n_a):
        for i_z in range(n_z):
            for i_tau in range(2):  # 0=plus, 1=minus
                s = i_a * n_z * 2 + i_z * 2 + i_tau

                if i_tau == 0:
                    i_a_prime = policy_plus_idx[i_a, i_z]
                else:
                    i_a_prime = policy_minus_idx[i_a, i_z]

                # Persistence: stay at (z, tau) with prob psi
                s_next = i_a_prime * n_z * 2 + i_z * 2 + i_tau
                rows[idx] = s_next
                cols[idx] = s
                data[idx] = psi
                idx += 1

                # Redraw: with prob 1-psi
                for i_z_prime in range(n_z):
                    p_z = prob_z[i_z_prime]
                    p_tau_plus = prob_tau_plus[i_z_prime]

                    # To tau_plus
                    s_next = i_a_prime * n_z * 2 + i_z_prime * 2 + 0
                    rows[idx] = s_next
                    cols[idx] = s
                    data[idx] = (1 - psi) * p_z * p_tau_plus
                    idx += 1

                    # To tau_minus
                    s_next = i_a_prime * n_z * 2 + i_z_prime * 2 + 1
                    rows[idx] = s_next
                    cols[idx] = s
                    data[idx] = (1 - psi) * p_z * (1 - p_tau_plus)
                    idx += 1

    return rows[:idx], cols[:idx], data[:idx]

def compute_stationary_with_dist(policy_plus_idx, policy_minus_idx,
                                  a_grid, z_grid, prob_z, prob_tau_plus, psi):
    """Compute stationary distribution WITH distortions"""
    n_a, n_z = len(a_grid), len(z_grid)
    n_states = n_a * n_z * 2

    rows, cols, data = build_transition_sparse_data_with_dist(
        policy_plus_idx, policy_minus_idx, n_a, n_z, prob_z, prob_tau_plus, psi
    )

    Q = sparse.csr_matrix((data, (rows, cols)), shape=(n_states, n_states))

    mu = np.ones(n_states) / n_states
    for _ in range(1000):
        mu_new = Q @ mu
        mu_new /= mu_new.sum()
        if np.max(np.abs(mu_new - mu)) < 1e-14:
            break
        mu = mu_new

    mu_full = mu_new.reshape((n_a, n_z, 2))
    return mu_full[:, :, 0], mu_full[:, :, 1]

# =============================================================================
# Aggregate Computation
# =============================================================================

def compute_aggregates(dist, a_grid, z_grid, is_entrep_grid, kstar_grid, lstar_grid, output_grid):
    """Compute aggregates (no distortions)"""
    K = np.sum(dist * kstar_grid * is_entrep_grid)
    L = np.sum(dist * lstar_grid * is_entrep_grid)
    Y = np.sum(dist * output_grid * is_entrep_grid)

    a_broadcast = a_grid[:, np.newaxis]
    extfin_grid = np.maximum(0, kstar_grid - a_broadcast) * is_entrep_grid
    extfin = np.sum(dist * extfin_grid)

    A = np.sum(dist * a_broadcast)
    share_entre = np.sum(dist * is_entrep_grid)

    return {'K': K, 'L': L, 'Y': Y, 'A': A, 'extfin': extfin, 'share_entre': share_entre}

def compute_aggregates_with_dist(mu_plus, mu_minus, a_grid, z_grid,
                                  is_plus, k_plus, l_plus, out_plus,
                                  is_minus, k_minus, l_minus, out_minus):
    """Compute aggregates WITH distortions"""
    a_broadcast = a_grid[:, np.newaxis]

    K = np.sum(mu_plus * k_plus * is_plus) + np.sum(mu_minus * k_minus * is_minus)
    L = np.sum(mu_plus * l_plus * is_plus) + np.sum(mu_minus * l_minus * is_minus)
    Y = np.sum(mu_plus * out_plus * is_plus) + np.sum(mu_minus * out_minus * is_minus)

    extfin_plus = np.maximum(0, k_plus - a_broadcast) * is_plus
    extfin_minus = np.maximum(0, k_minus - a_broadcast) * is_minus
    extfin = np.sum(mu_plus * extfin_plus) + np.sum(mu_minus * extfin_minus)

    A = np.sum(mu_plus * a_broadcast) + np.sum(mu_minus * a_broadcast)
    share_entre = np.sum(mu_plus * is_plus) + np.sum(mu_minus * is_minus)

    return {'K': K, 'L': L, 'Y': Y, 'A': A, 'extfin': extfin, 'share_entre': share_entre}

# =============================================================================
# GE Solver - No Distortions
# =============================================================================

def find_equilibrium_nodist(a_grid, z_grid, prob_z, params,
                            w_init=0.80, r_init=-0.04, V_init=None,
                            max_iter=100, tol=1e-3, verbose=True):
    """Find stationary equilibrium WITHOUT distortions"""
    delta, alpha, upsilon, lam, beta, sigma, psi = params

    w, r = w_init, r_init
    V_current = V_init
    w_step, r_step = 0.3, 0.05
    exc_L_prev, exc_K_prev = 0.0, 0.0

    best_error = np.inf
    best_result = None

    for iteration in range(max_iter):
        (profit_grid, kstar_grid, lstar_grid, output_grid,
         is_entrep_grid, income_grid) = precompute_entrepreneur_all(
            a_grid, z_grid, w, r, lam, delta, alpha, upsilon
        )

        V_current, policy_a_idx = solve_value_function_fast(
            a_grid, z_grid, prob_z, income_grid,
            beta, sigma, psi, V_init=V_current
        )

        dist = compute_stationary_distribution(
            policy_a_idx, a_grid, z_grid, prob_z, psi
        )

        agg = compute_aggregates(
            dist, a_grid, z_grid, is_entrep_grid,
            kstar_grid, lstar_grid, output_grid
        )

        exc_K = agg['K'] - agg['A']
        exc_L = agg['L'] - (1 - agg['share_entre'])

        if verbose:
            print(f"  [{iteration+1}] w={w:.4f}, r={r:.4f} | K={agg['K']:.2f}, A={agg['A']:.2f} | "
                  f"Ld={agg['L']:.2f}, Ls={1-agg['share_entre']:.2f}")

        total_error = abs(exc_L) + abs(exc_K)

        if total_error < best_error:
            best_error = total_error
            best_result = {
                'w': w, 'r': r, 'agg': agg.copy(),
                'dist': dist.copy(), 'policy': policy_a_idx.copy(),
                'V': V_current.copy(),
                'is_entrep': is_entrep_grid.copy(),
                'kstar': kstar_grid.copy(),
                'lstar': lstar_grid.copy(),
                'output': output_grid.copy(),
            }

        if abs(exc_L) < tol and abs(exc_K) < tol:
            break

        if iteration > 0:
            if exc_L * exc_L_prev < 0:
                w_step *= 0.5
            if exc_K * exc_K_prev < 0:
                r_step *= 0.5

        w_new = w * (1 + w_step * exc_L)
        r_new = r + r_step * exc_K

        w_new = max(0.01, min(2.5, w_new))
        r_new = max(-0.25, min(0.12, r_new))

        damping = 0.5
        w = damping * w + (1 - damping) * w_new
        r = damping * r + (1 - damping) * r_new

        exc_L_prev, exc_K_prev = exc_L, exc_K

    agg = best_result['agg']
    span = 1 - upsilon
    L_s = 1 - agg['share_entre']
    TFP = agg['Y'] / max(((agg['K'] ** alpha) * (L_s ** (1-alpha))) ** span, 1e-8)

    return {
        'w': best_result['w'], 'r': best_result['r'],
        'Y': agg['Y'], 'K': agg['K'], 'L': agg['L'], 'A': agg['A'],
        'TFP': TFP, 'extfin': agg['extfin'],
        'ExtFin_Y': agg['extfin'] / max(agg['Y'], 1e-8),
        'share_entre': agg['share_entre'],
        'dist': best_result['dist'],
        'policy': best_result['policy'],
        'V': best_result['V'],
        'a_grid': a_grid, 'z_grid': z_grid, 'prob_z': prob_z,
    }

# =============================================================================
# GE Solver - WITH Distortions
# =============================================================================

def find_equilibrium_with_dist(a_grid, z_grid, prob_z, prob_tau_plus, params,
                                tau_plus, tau_minus,
                                w_init=0.80, r_init=-0.04,
                                V_plus_init=None, V_minus_init=None,
                                max_iter=100, tol=1e-3, verbose=True):
    """Find stationary equilibrium WITH distortions"""
    delta, alpha, upsilon, lam, beta, sigma, psi = params

    w, r = w_init, r_init
    V_plus, V_minus = V_plus_init, V_minus_init
    w_step, r_step = 0.3, 0.05
    exc_L_prev, exc_K_prev = 0.0, 0.0

    best_error = np.inf
    best_result = None

    for iteration in range(max_iter):
        # Precompute for both tau states
        (_, k_plus, l_plus, out_plus, is_plus, inc_plus) = precompute_entrepreneur_with_tau(
            a_grid, z_grid, tau_plus, w, r, lam, delta, alpha, upsilon
        )
        (_, k_minus, l_minus, out_minus, is_minus, inc_minus) = precompute_entrepreneur_with_tau(
            a_grid, z_grid, tau_minus, w, r, lam, delta, alpha, upsilon
        )

        # Solve VFI for both tau states
        V_plus, policy_plus = solve_value_function_fast(
            a_grid, z_grid, prob_z, inc_plus, beta, sigma, psi, V_init=V_plus
        )
        V_minus, policy_minus = solve_value_function_fast(
            a_grid, z_grid, prob_z, inc_minus, beta, sigma, psi, V_init=V_minus
        )

        # Compute distribution
        mu_plus, mu_minus = compute_stationary_with_dist(
            policy_plus, policy_minus, a_grid, z_grid, prob_z, prob_tau_plus, psi
        )

        # Aggregates
        agg = compute_aggregates_with_dist(
            mu_plus, mu_minus, a_grid, z_grid,
            is_plus, k_plus, l_plus, out_plus,
            is_minus, k_minus, l_minus, out_minus
        )

        exc_K = agg['K'] - agg['A']
        exc_L = agg['L'] - (1 - agg['share_entre'])

        if verbose:
            print(f"  [{iteration+1}] w={w:.4f}, r={r:.4f} | K={agg['K']:.2f}, A={agg['A']:.2f} | "
                  f"Ld={agg['L']:.2f}, Ls={1-agg['share_entre']:.2f}")

        total_error = abs(exc_L) + abs(exc_K)

        if total_error < best_error:
            best_error = total_error
            best_result = {
                'w': w, 'r': r, 'agg': agg.copy(),
                'mu_plus': mu_plus.copy(), 'mu_minus': mu_minus.copy(),
                'policy_plus': policy_plus.copy(), 'policy_minus': policy_minus.copy(),
                'V_plus': V_plus.copy(), 'V_minus': V_minus.copy(),
            }

        if abs(exc_L) < tol and abs(exc_K) < tol:
            break

        if iteration > 0:
            if exc_L * exc_L_prev < 0:
                w_step *= 0.5
            if exc_K * exc_K_prev < 0:
                r_step *= 0.5

        w_new = w * (1 + w_step * exc_L)
        r_new = r + r_step * exc_K

        w_new = max(0.01, min(2.5, w_new))
        r_new = max(-0.25, min(0.12, r_new))

        damping = 0.5
        w = damping * w + (1 - damping) * w_new
        r = damping * r + (1 - damping) * r_new

        exc_L_prev, exc_K_prev = exc_L, exc_K

    agg = best_result['agg']
    span = 1 - upsilon
    L_s = 1 - agg['share_entre']
    TFP = agg['Y'] / max(((agg['K'] ** alpha) * (L_s ** (1-alpha))) ** span, 1e-8)

    return {
        'w': best_result['w'], 'r': best_result['r'],
        'Y': agg['Y'], 'K': agg['K'], 'L': agg['L'], 'A': agg['A'],
        'TFP': TFP, 'extfin': agg['extfin'],
        'ExtFin_Y': agg['extfin'] / max(agg['Y'], 1e-8),
        'share_entre': agg['share_entre'],
        'mu_plus': best_result['mu_plus'],
        'mu_minus': best_result['mu_minus'],
        'policy_plus': best_result['policy_plus'],
        'policy_minus': best_result['policy_minus'],
        'V_plus': best_result['V_plus'],
        'V_minus': best_result['V_minus'],
        'a_grid': a_grid, 'z_grid': z_grid, 'prob_z': prob_z,
        'prob_tau_plus': prob_tau_plus,
    }

# =============================================================================
# Transition Path - Backward VFI
# =============================================================================

def solve_vfi_transition_at_t(a_grid, z_grid, prob_z, income_grid,
                               beta, sigma, psi, w_tp1, r_tp1,
                               income_tp1, V_tp1, tol=1e-5, max_iter=200):
    """
    Solve VFI at time t given prices at t and continuation value at t+1.
    Key: Expected value uses V_{t+1}, not V_t.
    """
    n_a, n_z = len(a_grid), len(z_grid)

    # Compute EV using V_{t+1}
    V_mean_tp1 = np.zeros(n_a)
    for i_a in range(n_a):
        for i_z in range(n_z):
            V_mean_tp1[i_a] += prob_z[i_z] * V_tp1[i_a, i_z]

    EV_tp1 = np.zeros((n_a, n_z))
    for i_a in range(n_a):
        for i_z in range(n_z):
            EV_tp1[i_a, i_z] = psi * V_tp1[i_a, i_z] + (1 - psi) * V_mean_tp1[i_a]

    # Solve for V_t and policy_t
    V = V_tp1.copy()
    policy_a_idx = np.zeros((n_a, n_z), dtype=np.int64)

    for iteration in range(max_iter):
        V_new = np.zeros((n_a, n_z))

        for i_z in range(n_z):
            start_idx = 0
            for i_a in range(n_a):
                income = income_grid[i_a, i_z]

                best_val, best_idx = find_optimal_savings_monotone(
                    income, a_grid, EV_tp1[:, i_z], beta, sigma, start_idx
                )

                V_new[i_a, i_z] = best_val
                policy_a_idx[i_a, i_z] = best_idx
                start_idx = best_idx

        diff = np.max(np.abs(V_new - V))
        V = V_new

        if diff < tol:
            break

    return V, policy_a_idx

# =============================================================================
# Transition Path - Forward Distribution Update
# =============================================================================

def update_distribution_forward(dist_t, policy_t, a_grid, z_grid, prob_z, psi):
    """Update distribution from t to t+1"""
    n_a, n_z = len(a_grid), len(z_grid)
    dist_next = np.zeros((n_a, n_z))

    for i_a in range(n_a):
        for i_z in range(n_z):
            mass = dist_t[i_a, i_z]
            if mass < 1e-14:
                continue

            i_a_prime = policy_t[i_a, i_z]

            for i_z_prime in range(n_z):
                if i_z_prime == i_z:
                    p = psi + (1 - psi) * prob_z[i_z_prime]
                else:
                    p = (1 - psi) * prob_z[i_z_prime]

                if p > 1e-14:
                    dist_next[i_a_prime, i_z_prime] += mass * p

    return dist_next / dist_next.sum()

# =============================================================================
# Time Path Iteration (TPI)
# =============================================================================

def solve_transition(pre_eq, post_eq, params, T=250, kappa=0.05,
                     eta_w=0.3, eta_r=0.02, theta=0.5, tol=5e-3, max_iter=50):
    """Solve transition path via TPI"""
    delta, alpha, upsilon, lam, beta, sigma, psi = params

    a_grid = post_eq['a_grid']
    z_grid = post_eq['z_grid']
    prob_z = post_eq['prob_z']
    n_a, n_z = len(a_grid), len(z_grid)

    w_pre, r_pre = pre_eq['w'], pre_eq['r']
    w_post, r_post = post_eq['w'], post_eq['r']

    # Initial distribution: marginal over tau
    mu_0 = pre_eq['mu_plus'] + pre_eq['mu_minus']
    mu_0 /= mu_0.sum()

    # Terminal
    V_post = post_eq['V']
    policy_post = post_eq['policy']

    # Initialize price paths
    t_arr = np.arange(T)
    w_path = w_post + (w_pre - w_post) * np.exp(-kappa * t_arr)
    r_path = r_post + (r_pre - r_post) * np.exp(-kappa * t_arr)

    print(f"\n{'='*70}")
    print("TRANSITION PATH ITERATION (Howard PI)")
    print(f"{'='*70}")
    print(f"T = {T}, Initial: w_pre={w_pre:.4f}, r_pre={r_pre:.4f}")
    print(f"Target: w_post={w_post:.4f}, r_post={r_post:.4f}")
    print(f"{'='*70}\n")

    # Storage
    Y_path = np.zeros(T)
    K_path = np.zeros(T)
    L_path = np.zeros(T)
    A_path = np.zeros(T)
    TFP_path = np.zeros(T)
    ExtFin_path = np.zeros(T)
    Entre_path = np.zeros(T)
    ED_L_path = np.zeros(T)
    ED_K_path = np.zeros(T)

    for tpi_iter in range(max_iter):
        # =================================================================
        # BACKWARD: Solve policies
        # =================================================================
        policies = [None] * T
        V_list = [None] * T

        # Pre-compute income grids for all t
        income_list = []
        kstar_list = []
        lstar_list = []
        output_list = []
        is_entrep_list = []

        for t in range(T):
            (_, kstar, lstar, output, is_entrep, income) = precompute_entrepreneur_all(
                a_grid, z_grid, w_path[t], r_path[t], lam, delta, alpha, upsilon
            )
            income_list.append(income)
            kstar_list.append(kstar)
            lstar_list.append(lstar)
            output_list.append(output)
            is_entrep_list.append(is_entrep)

        # Terminal
        policies[T-1] = policy_post.copy()
        V_list[T-1] = V_post.copy()

        # Backward iteration
        for t in range(T-2, -1, -1):
            V_tp1 = V_list[t+1]
            income_tp1 = income_list[t+1] if t+1 < T else income_list[-1]

            V_t, policy_t = solve_vfi_transition_at_t(
                a_grid, z_grid, prob_z, income_list[t],
                beta, sigma, psi, w_path[t+1], r_path[t+1],
                income_tp1, V_tp1
            )

            policies[t] = policy_t
            V_list[t] = V_t

        # =================================================================
        # FORWARD: Update distributions and compute aggregates
        # =================================================================
        dist_t = mu_0.copy()

        for t in range(T):
            agg = compute_aggregates(
                dist_t, a_grid, z_grid, is_entrep_list[t],
                kstar_list[t], lstar_list[t], output_list[t]
            )

            K_path[t] = agg['K']
            L_path[t] = agg['L']
            Y_path[t] = agg['Y']
            A_path[t] = agg['A']
            ExtFin_path[t] = agg['extfin']
            Entre_path[t] = agg['share_entre']

            span = 1 - upsilon
            L_s = 1 - agg['share_entre']
            TFP_path[t] = agg['Y'] / max(((agg['K'] ** alpha) * (L_s ** (1-alpha))) ** span, 1e-8)

            ED_L_path[t] = (agg['L'] - L_s) / max(L_s, 0.05)
            ED_K_path[t] = (agg['K'] - agg['A']) / max(agg['A'], 1.0)

            if t < T - 1:
                dist_t = update_distribution_forward(dist_t, policies[t], a_grid, z_grid, prob_z, psi)

        # =================================================================
        # CHECK CONVERGENCE
        # =================================================================
        max_ED = max(np.max(np.abs(ED_L_path)), np.max(np.abs(ED_K_path)))

        if tpi_iter % 5 == 0 or tpi_iter < 3:
            print(f"[TPI iter {tpi_iter+1:3d}] max|ED_L|={np.max(np.abs(ED_L_path)):.5f}, "
                  f"max|ED_K|={np.max(np.abs(ED_K_path)):.5f}")

        if max_ED < tol:
            print(f"\n[CONVERGED] after {tpi_iter+1} iterations")
            break

        # =================================================================
        # UPDATE PRICES
        # =================================================================
        w_path_new = w_path * (1 + eta_w * ED_L_path)
        r_path_new = r_path + eta_r * ED_K_path

        # Smooth
        window = 5
        for t in range(window, T - window):
            w_path_new[t] = 0.7 * w_path_new[t] + 0.3 * np.mean(w_path_new[t-window:t+window])
            r_path_new[t] = 0.7 * r_path_new[t] + 0.3 * np.mean(r_path_new[t-window:t+window])

        # Enforce terminal
        decay_len = min(20, T//10)
        w_path_new[-decay_len:] = w_post + (w_path_new[-decay_len] - w_post) * np.exp(-0.3 * np.arange(decay_len))
        r_path_new[-decay_len:] = r_post + (r_path_new[-decay_len] - r_post) * np.exp(-0.3 * np.arange(decay_len))

        # Damped update
        w_path = (1 - theta) * w_path + theta * w_path_new
        r_path = (1 - theta) * r_path + theta * r_path_new

    return {
        't': np.arange(T),
        'w': w_path, 'r': r_path,
        'Y': Y_path, 'K': K_path, 'L': L_path, 'A': A_path,
        'TFP': TFP_path, 'ExtFin': ExtFin_path,
        'ExtFin_Y': ExtFin_path / np.maximum(Y_path, 1e-8),
        'Entre_share': Entre_path,
        'ED_L': ED_L_path, 'ED_K': ED_K_path,
    }

# =============================================================================
# Output and Plotting
# =============================================================================

def save_results(pre_eq, post_eq, trans, output_dir='outputs'):
    os.makedirs(output_dir, exist_ok=True)

    for name, eq in [('stationary_pre', pre_eq), ('stationary_post', post_eq)]:
        summary = {k: float(v) if isinstance(v, (int, float, np.floating)) else None
                   for k, v in eq.items() if not isinstance(v, np.ndarray)}
        with open(os.path.join(output_dir, f'{name}_v3.json'), 'w') as f:
            json.dump(summary, f, indent=2)

    import csv
    with open(os.path.join(output_dir, 'transition_v3.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['t', 'w', 'r', 'Y', 'K', 'L', 'A', 'TFP', 'ext_fin', 'ext_fin_Y', 'entrepreneur_share'])
        for i in range(len(trans['t'])):
            writer.writerow([trans['t'][i], trans['w'][i], trans['r'][i], trans['Y'][i],
                           trans['K'][i], trans['L'][i], trans['A'][i], trans['TFP'][i],
                           trans['ExtFin'][i], trans['ExtFin_Y'][i], trans['Entre_share'][i]])

    print(f"\nResults saved to {output_dir}/")

def plot_transition(pre_eq, post_eq, trans, output_dir='outputs'):
    os.makedirs(output_dir, exist_ok=True)

    t = trans['t']
    Y_post, TFP_post = post_eq['Y'], post_eq['TFP']

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0,0].plot(t, trans['Y']/Y_post, 'b-', lw=2)
    axes[0,0].axhline(1.0, color='r', ls='--', alpha=0.7)
    axes[0,0].axhline(pre_eq['Y']/Y_post, color='g', ls=':', alpha=0.7)
    axes[0,0].set_xlabel('Period'); axes[0,0].set_ylabel('Y / Y_post')
    axes[0,0].set_title('Output'); axes[0,0].grid(True, alpha=0.3)

    axes[0,1].plot(t, trans['TFP']/TFP_post, 'b-', lw=2)
    axes[0,1].axhline(1.0, color='r', ls='--', alpha=0.7)
    axes[0,1].axhline(pre_eq['TFP']/TFP_post, color='g', ls=':', alpha=0.7)
    axes[0,1].set_xlabel('Period'); axes[0,1].set_ylabel('TFP / TFP_post')
    axes[0,1].set_title('TFP'); axes[0,1].grid(True, alpha=0.3)

    axes[0,2].plot(t, trans['r'], 'b-', lw=2)
    axes[0,2].axhline(post_eq['r'], color='r', ls='--', alpha=0.7)
    axes[0,2].axhline(pre_eq['r'], color='g', ls=':', alpha=0.7)
    axes[0,2].set_xlabel('Period'); axes[0,2].set_ylabel('r')
    axes[0,2].set_title('Interest Rate'); axes[0,2].grid(True, alpha=0.3)

    axes[1,0].plot(t, trans['w'], 'b-', lw=2)
    axes[1,0].axhline(post_eq['w'], color='r', ls='--', alpha=0.7)
    axes[1,0].axhline(pre_eq['w'], color='g', ls=':', alpha=0.7)
    axes[1,0].set_xlabel('Period'); axes[1,0].set_ylabel('w')
    axes[1,0].set_title('Wage'); axes[1,0].grid(True, alpha=0.3)

    axes[1,1].plot(t, trans['ExtFin_Y'], 'b-', lw=2)
    axes[1,1].axhline(post_eq['ExtFin_Y'], color='r', ls='--', alpha=0.7)
    axes[1,1].axhline(pre_eq['ExtFin_Y'], color='g', ls=':', alpha=0.7)
    axes[1,1].set_xlabel('Period'); axes[1,1].set_ylabel('ExtFin/Y')
    axes[1,1].set_title('External Finance'); axes[1,1].grid(True, alpha=0.3)

    axes[1,2].plot(t, trans['Entre_share'], 'b-', lw=2)
    axes[1,2].axhline(post_eq['share_entre'], color='r', ls='--', alpha=0.7)
    axes[1,2].axhline(pre_eq['share_entre'], color='g', ls=':', alpha=0.7)
    axes[1,2].set_xlabel('Period'); axes[1,2].set_ylabel('Share')
    axes[1,2].set_title('Entrepreneur Share'); axes[1,2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'transition_dynamics_v3.png'), dpi=150)
    plt.close()
    print(f"Plots saved to {output_dir}/transition_dynamics_v3.png")

def print_summary(pre_eq, post_eq, trans):
    print("\n" + "="*70)
    print("RESULTS SUMMARY (Howard PI Version)")
    print("="*70)
    print(f"\n{'Variable':<20} {'Pre-Reform':>15} {'Post-Reform':>15} {'Change %':>12}")
    print("-"*62)

    for name, key in [('Output (Y)', 'Y'), ('TFP', 'TFP'), ('Capital (K)', 'K'),
                      ('Assets (A)', 'A'), ('Wage (w)', 'w'), ('Interest Rate (r)', 'r'),
                      ('Ext.Fin/GDP', 'ExtFin_Y'), ('Entre Share', 'share_entre')]:
        pre_v, post_v = pre_eq[key], post_eq[key]
        chg = (post_v - pre_v) / abs(pre_v) * 100 if pre_v != 0 else 0
        print(f"{name:<20} {pre_v:>15.4f} {post_v:>15.4f} {chg:>11.2f}%")

    print("\nSanity Checks:")
    print(f"  TFP change: {'[PASS] increases' if post_eq['TFP'] > pre_eq['TFP'] else '[WARN] decreases'}")
    print(f"  Y change: {'[PASS] increases' if post_eq['Y'] > pre_eq['Y'] else '[WARN] decreases'}")
    print("="*70)

# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Buera & Shin Transition (Howard PI)')
    parser.add_argument('--T', type=int, default=250, help='Transition horizon')
    parser.add_argument('--na', type=int, default=N_A, help='Asset grid points')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory')
    args = parser.parse_args()

    print("="*70)
    print("BUERA & SHIN (2010) TRANSITION DYNAMICS")
    print("Howard Policy Iteration Version (based on v3)")
    print("="*70)
    print(f"\nParameters: lambda={LAMBDA}, tau_plus={TAU_PLUS}, tau_minus={TAU_MINUS}, q={Q_DIST}")
    print(f"Grid: na={args.na}, Transition horizon T={args.T}")
    print("="*70)

    t_start = time.time()

    # Setup grids
    z_grid, prob_z = create_ability_grid_paper(ETA)
    a_grid = create_asset_grid(args.na, A_MIN, A_MAX)
    prob_tau_plus = compute_tau_probs(z_grid, Q_DIST)

    params = (DELTA, ALPHA, NU, LAMBDA, BETA, SIGMA, PSI)

    # JIT warmup
    print("\nWarming up JIT...")
    _ = solve_entrepreneur_single(1.0, 1.5, 1.0, 0.04, LAMBDA, DELTA, ALPHA, NU)
    _ = solve_entrepreneur_with_tau(1.0, 1.5, 0.5, 1.0, 0.04, LAMBDA, DELTA, ALPHA, NU)
    print("JIT warmup done.")

    # Step 1: Post-reform (no distortions)
    print("\n" + "="*70)
    print("STEP 1: Post-reform Stationary Equilibrium (NO distortions)")
    print("="*70)
    post_eq = find_equilibrium_nodist(a_grid, z_grid, prob_z, params,
                                       w_init=0.80, r_init=-0.04)
    print(f"\nPost-reform: w={post_eq['w']:.4f}, r={post_eq['r']:.4f}, Y={post_eq['Y']:.4f}, TFP={post_eq['TFP']:.4f}")

    # Step 2: Pre-reform (with distortions)
    print("\n" + "="*70)
    print("STEP 2: Pre-reform Stationary Equilibrium (WITH distortions)")
    print("="*70)
    pre_eq = find_equilibrium_with_dist(a_grid, z_grid, prob_z, prob_tau_plus, params,
                                         TAU_PLUS, TAU_MINUS,
                                         w_init=post_eq['w'], r_init=post_eq['r'])
    print(f"\nPre-reform: w={pre_eq['w']:.4f}, r={pre_eq['r']:.4f}, Y={pre_eq['Y']:.4f}, TFP={pre_eq['TFP']:.4f}")

    # Step 3: Transition
    print("\n" + "="*70)
    print("STEP 3: Transition Dynamics (TPI)")
    print("="*70)
    trans = solve_transition(pre_eq, post_eq, params, T=args.T)

    print(f"\nTotal time: {time.time() - t_start:.1f}s")

    # Save and plot
    save_results(pre_eq, post_eq, trans, args.output_dir)
    plot_transition(pre_eq, post_eq, trans, args.output_dir)
    print_summary(pre_eq, post_eq, trans)

    return pre_eq, post_eq, trans

if __name__ == '__main__':
    main()
