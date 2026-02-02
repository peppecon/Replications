"""
Buera & Shin (2010) Transition Dynamics with Idiosyncratic Distortions
======================================================================

This script builds on buera_shin_v6_chebyshev.py (which works perfectly for stationary equilibrium)
and adds:
1. Distortions mode for pre-reform stationary equilibrium
2. Transition path computation via Time Path Iteration (TPI)

Computes:
1. Pre-reform stationary equilibrium: lambda=1.35 with idiosyncratic output wedges
2. Post-reform stationary equilibrium: lambda=1.35 without distortions (tau=0)
3. Perfect-foresight transition path

Usage:
    python transition_buera_shin.py --T 250
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numba import njit, prange
from scipy import sparse
from scipy.sparse.linalg import eigs
import os
import json
import argparse
import warnings
import time

# Import from the working v6 code
from library.functions_library import Chebyshev_Nodes, Chebyshev_Polynomials_Recursion_mv

warnings.filterwarnings('ignore')

# =============================================================================
# Model Parameters (from v6 + distortion parameters)
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

# Global Grid Bounds (from v6)
A_MIN, A_MAX = 1e-6, 4000.0
A_SHIFT = 1.0
Z_MIN, Z_MAX = (1 - 0.633)**(-1/4.15), (1 - 0.9995)**(-1/4.15)
N_CHEBY_A = 40
N_CHEBY_Z = 40

# =============================================================================
# Bivariate Spectral Tools (from v6)
# =============================================================================

@njit(cache=True)
def bivariate_eval(a, z, coeffs, a_min, a_max, z_min, z_max):
    na, nz = N_CHEBY_A, N_CHEBY_Z
    la = np.log(a + A_SHIFT)
    lmin, lmax = np.log(a_min + A_SHIFT), np.log(a_max + A_SHIFT)
    xa = max(-1.0, min(1.0, 2.0 * (la - lmin) / (lmax - lmin) - 1.0))
    xz = max(-1.0, min(1.0, 2.0 * (z - z_min) / (z_max - z_min) - 1.0))

    Ta = np.zeros(na); Ta[0] = 1.0
    if na > 1: Ta[1] = xa
    for i in range(2, na): Ta[i] = 2.0 * xa * Ta[i-1] - Ta[i-2]
    Tz = np.zeros(nz); Tz[0] = 1.0
    if nz > 1: Tz[1] = xz
    for j in range(2, nz): Tz[j] = 2.0 * xz * Tz[j-1] - Tz[j-2]

    val = 0.0
    C = coeffs.reshape((na, nz))
    for i in range(na):
        row_sum = 0.0
        for j in range(nz): row_sum += C[i, j] * Tz[j]
        val += Ta[i] * row_sum

    return max(a_min, min(val, a_max))

def bivariate_eval_matrix(a_vec, z_vec, coeffs, a_min, a_max, z_min, z_max):
    na, nz = N_CHEBY_A, N_CHEBY_Z
    lmin, lmax = np.log(a_min + A_SHIFT), np.log(a_max + A_SHIFT)
    xa_vec = 2.0 * (np.log(a_vec + A_SHIFT) - lmin) / (lmax - lmin) - 1.0
    xz_vec = 2.0 * (z_vec - z_min) / (z_max - z_min) - 1.0

    Ta = Chebyshev_Polynomials_Recursion_mv(xa_vec, na).T
    Tz = Chebyshev_Polynomials_Recursion_mv(xz_vec, nz).T

    C = coeffs.reshape((na, nz))
    res = Ta @ C @ Tz.T
    return np.clip(res, a_min, a_max)

def generate_bivariate_nodes_matrix(a_min, a_max, z_min, z_max):
    cheb_a = Chebyshev_Nodes(N_CHEBY_A).ravel()
    lmin, lmax = np.log(a_min + A_SHIFT), np.log(a_max + A_SHIFT)
    nodes_a = np.exp((lmin + lmax)/2.0 + (lmax - lmin)/2.0 * cheb_a) - A_SHIFT
    cheb_z = Chebyshev_Nodes(N_CHEBY_Z).ravel()
    nodes_z = (z_min + z_max)/2.0 + (z_max - z_min)/2.0 * cheb_z
    Ta = Chebyshev_Polynomials_Recursion_mv(cheb_a, N_CHEBY_A)
    Tz = Chebyshev_Polynomials_Recursion_mv(cheb_z, N_CHEBY_Z)
    T_full = np.zeros((N_CHEBY_A * N_CHEBY_Z, N_CHEBY_A * N_CHEBY_Z))
    idx = 0
    for ia in range(N_CHEBY_A):
        for iz in range(N_CHEBY_Z):
            row = np.zeros(N_CHEBY_A * N_CHEBY_Z)
            j_idx = 0
            for ja in range(N_CHEBY_A):
                for jz in range(N_CHEBY_Z):
                    row[j_idx] = Ta[ja, ia] * Tz[jz, iz]
                    j_idx += 1
            T_full[idx, :] = row
            idx += 1
    return nodes_a, nodes_z, T_full

# =============================================================================
# Entrepreneur Logic (with distortion support)
# =============================================================================

@njit(cache=True)
def solve_entrepreneur_single(a, z, w, r, lam, delta, alpha, nu):
    """Original entrepreneur problem (no distortion)"""
    rental = max(r + delta, 1e-8)
    wage = max(w, 1e-8)
    span = 1 - nu
    exp_k = 1 - (1-alpha)*span
    exp_l = (1-alpha)*span
    k1 = (((alpha*span*z/rental)**exp_k) * (((1-alpha)*span*z/wage)**exp_l)) ** (1/nu)
    kstar = min(k1, 1e8)
    kstar = min(kstar, lam * a)
    denom = max((z * (kstar**(alpha*span)) * (1-alpha) * span), 1e-12)
    lstar = ((wage / denom)) ** (1 / ((1-alpha)*span - 1))
    lstar = min(lstar, 1e8)
    output = z * ((kstar**alpha) * (lstar**(1-alpha))) ** span
    profit = output - wage * lstar - rental * kstar
    return profit, kstar, lstar, output

@njit(cache=True)
def solve_entrepreneur_with_tau(a, z, tau, w, r, lam, delta, alpha, nu):
    """
    Entrepreneur problem WITH distortion tau.
    profit = (1-tau)*e*(k^alpha * l^(1-alpha))^(1-nu) - w*l - (r+delta)*k

    Key: use effective ability z_eff = (1-tau)*z
    """
    z_eff = (1.0 - tau) * z
    if z_eff <= 0:
        return -1e10, 0.0, 0.0, 0.0

    rental = max(r + delta, 1e-8)
    wage = max(w, 1e-8)
    span = 1 - nu
    exp_k = 1 - (1-alpha)*span
    exp_l = (1-alpha)*span

    k1 = (((alpha*span*z_eff/rental)**exp_k) * (((1-alpha)*span*z_eff/wage)**exp_l)) ** (1/nu)
    kstar = min(k1, 1e8)
    kstar = min(kstar, lam * a)
    denom = max((z_eff * (kstar**(alpha*span)) * (1-alpha) * span), 1e-12)
    lstar = ((wage / denom)) ** (1 / ((1-alpha)*span - 1))
    lstar = min(lstar, 1e8)
    output = z_eff * ((kstar**alpha) * (lstar**(1-alpha))) ** span
    profit = output - wage * lstar - rental * kstar
    return profit, kstar, lstar, output

# =============================================================================
# Policy Solver - No Distortions (from v6)
# =============================================================================

@njit(cache=True, parallel=True)
def solve_policy_bivariate_update(coeffs, nodes_a, nodes_z, T_az_inv,
                                 beta, sigma, psi, w, r, lam, delta, alpha, nu,
                                 a_min, a_max, z_min, z_max,
                                 quad_z, quad_w):
    nz_c = N_CHEBY_Z
    nt = len(nodes_a) * nz_c
    target_vals = np.zeros(nt)
    for i_n in prange(nt):
        ia, iz = i_n // nz_c, i_n % nz_c
        a, z = nodes_a[ia], nodes_z[iz]
        p, _, _, _ = solve_entrepreneur_single(a, z, w, r, lam, delta, alpha, nu)
        inc = max(p, w) + (1 + r) * a
        aprime = bivariate_eval(a, z, coeffs, a_min, a_max, z_min, z_max)
        aprime = max(a_min, min(aprime, inc - 1e-6))
        p_p, _, _, _ = solve_entrepreneur_single(aprime, z, w, r, lam, delta, alpha, nu)
        inc_p = max(p_p, w) + (1 + r) * aprime
        app = bivariate_eval(aprime, z, coeffs, a_min, a_max, z_min, z_max)
        mu_pers = max(inc_p - app, 1e-9) ** (-sigma)
        e_mu_res = 0.0
        for k in range(len(quad_z)):
            pk, _, _, _ = solve_entrepreneur_single(aprime, quad_z[k], w, r, lam, delta, alpha, nu)
            ick = max(pk, w) + (1 + r) * aprime
            akk = bivariate_eval(aprime, quad_z[k], coeffs, a_min, a_max, z_min, z_max)
            e_mu_res += quad_w[k] * max(ick - akk, 1e-9) ** (-sigma)
        expected_mu = psi * mu_pers + (1 - psi) * e_mu_res
        c_target = (beta * (1 + r) * expected_mu) ** (-1.0/sigma)
        target_vals[i_n] = max(a_min, min(inc - c_target, a_max))
    return T_az_inv @ target_vals

def solve_policy_spectral(params, w, r, coeffs_init=None):
    """Solve for policy function WITHOUT distortions (from v6)"""
    delta, alpha, nu, lam, beta, sigma, psi = params
    nodes_a, nodes_z, T_full = generate_bivariate_nodes_matrix(A_MIN, A_MAX, Z_MIN, Z_MAX)
    T_inv = np.linalg.inv(T_full)
    M_vals = np.concatenate([np.linspace(0.633, 0.998, 38), [0.999, 0.9995]])
    z_quad = (1 - M_vals)**(-1/ETA)
    z_w = np.zeros(40)
    z_w[0] = M_vals[0]
    z_w[1:] = M_vals[1:] - M_vals[:-1]
    z_w /= z_w.sum()
    if coeffs_init is None:
        flat = np.zeros(len(nodes_a) * len(nodes_z))
        for i in range(len(flat)):
            ia, iz = i // len(nodes_z), i % len(nodes_z)
            p, _, _, _ = solve_entrepreneur_single(nodes_a[ia], nodes_z[iz], w, r, lam, delta, alpha, nu)
            flat[i] = 0.8 * (max(p, w) + (1+r)*nodes_a[ia])
        coeffs = T_inv @ flat
    else:
        coeffs = coeffs_init
    for it in range(150):
        c_new = solve_policy_bivariate_update(coeffs, nodes_a, nodes_z, T_inv,
                                            beta, sigma, psi, w, r, lam, delta, alpha, nu,
                                            A_MIN, A_MAX, Z_MIN, Z_MAX, z_quad, z_w)
        diff = np.max(np.abs(c_new - coeffs))
        coeffs = 0.8 * coeffs + 0.2 * c_new
        if diff < 1e-7: break
    return coeffs

# =============================================================================
# Policy Solver - WITH Distortions
# =============================================================================

@njit(cache=True, parallel=True)
def solve_policy_bivariate_update_with_dist(coeffs_plus, coeffs_minus,
                                            nodes_a, nodes_z, T_az_inv,
                                            beta, sigma, psi, w, r, lam, delta, alpha, nu,
                                            a_min, a_max, z_min, z_max,
                                            quad_z, quad_w, prob_tau_plus_arr,
                                            tau_plus, tau_minus, is_plus):
    """
    Policy update for distortion case.
    is_plus: True for tau_plus policy, False for tau_minus policy
    """
    tau_curr = tau_plus if is_plus else tau_minus
    coeffs_curr = coeffs_plus if is_plus else coeffs_minus

    nz_c = N_CHEBY_Z
    nt = len(nodes_a) * nz_c
    target_vals = np.zeros(nt)

    for i_n in prange(nt):
        ia, iz = i_n // nz_c, i_n % nz_c
        a, z = nodes_a[ia], nodes_z[iz]

        # Current income with tau
        p, _, _, _ = solve_entrepreneur_with_tau(a, z, tau_curr, w, r, lam, delta, alpha, nu)
        inc = max(p, w) + (1 + r) * a

        # Current policy guess
        aprime = bivariate_eval(a, z, coeffs_curr, a_min, a_max, z_min, z_max)
        aprime = max(a_min, min(aprime, inc - 1e-6))

        # Expected marginal utility
        # Persistence branch: stay at (z, tau) with prob psi
        p_p, _, _, _ = solve_entrepreneur_with_tau(aprime, z, tau_curr, w, r, lam, delta, alpha, nu)
        inc_p = max(p_p, w) + (1 + r) * aprime
        app = bivariate_eval(aprime, z, coeffs_curr, a_min, a_max, z_min, z_max)
        mu_pers = max(inc_p - app, 1e-9) ** (-sigma)

        # Redraw branch: with prob 1-psi, draw new (z', tau')
        e_mu_redraw = 0.0
        for k in range(len(quad_z)):
            z_k = quad_z[k]
            p_tau_plus_k = prob_tau_plus_arr[k]

            # Case: tau' = tau_plus
            pk_plus, _, _, _ = solve_entrepreneur_with_tau(aprime, z_k, tau_plus, w, r, lam, delta, alpha, nu)
            ick_plus = max(pk_plus, w) + (1 + r) * aprime
            akk_plus = bivariate_eval(aprime, z_k, coeffs_plus, a_min, a_max, z_min, z_max)
            mu_plus = max(ick_plus - akk_plus, 1e-9) ** (-sigma)

            # Case: tau' = tau_minus
            pk_minus, _, _, _ = solve_entrepreneur_with_tau(aprime, z_k, tau_minus, w, r, lam, delta, alpha, nu)
            ick_minus = max(pk_minus, w) + (1 + r) * aprime
            akk_minus = bivariate_eval(aprime, z_k, coeffs_minus, a_min, a_max, z_min, z_max)
            mu_minus = max(ick_minus - akk_minus, 1e-9) ** (-sigma)

            e_mu_redraw += quad_w[k] * (p_tau_plus_k * mu_plus + (1 - p_tau_plus_k) * mu_minus)

        expected_mu = psi * mu_pers + (1 - psi) * e_mu_redraw
        c_target = (beta * (1 + r) * expected_mu) ** (-1.0/sigma)
        target_vals[i_n] = max(a_min, min(inc - c_target, a_max))

    return T_az_inv @ target_vals

def solve_policy_spectral_with_dist(params, w, r, tau_plus, tau_minus, q_dist,
                                     coeffs_plus_init=None, coeffs_minus_init=None):
    """Solve for policy functions WITH distortions"""
    delta, alpha, nu, lam, beta, sigma, psi = params
    nodes_a, nodes_z, T_full = generate_bivariate_nodes_matrix(A_MIN, A_MAX, Z_MIN, Z_MAX)
    T_inv = np.linalg.inv(T_full)

    M_vals = np.concatenate([np.linspace(0.633, 0.998, 38), [0.999, 0.9995]])
    z_quad = (1 - M_vals)**(-1/ETA)
    z_w = np.zeros(40)
    z_w[0] = M_vals[0]
    z_w[1:] = M_vals[1:] - M_vals[:-1]
    z_w /= z_w.sum()

    # Probability of tau_plus for each z in quadrature
    prob_tau_plus_arr = 1 - np.exp(-q_dist * z_quad)

    # Initialize coefficients
    if coeffs_plus_init is None:
        flat_plus = np.zeros(len(nodes_a) * len(nodes_z))
        flat_minus = np.zeros(len(nodes_a) * len(nodes_z))
        for i in range(len(flat_plus)):
            ia, iz = i // len(nodes_z), i % len(nodes_z)
            p_plus, _, _, _ = solve_entrepreneur_with_tau(nodes_a[ia], nodes_z[iz], tau_plus, w, r, lam, delta, alpha, nu)
            p_minus, _, _, _ = solve_entrepreneur_with_tau(nodes_a[ia], nodes_z[iz], tau_minus, w, r, lam, delta, alpha, nu)
            flat_plus[i] = 0.8 * (max(p_plus, w) + (1+r)*nodes_a[ia])
            flat_minus[i] = 0.8 * (max(p_minus, w) + (1+r)*nodes_a[ia])
        coeffs_plus = T_inv @ flat_plus
        coeffs_minus = T_inv @ flat_minus
    else:
        coeffs_plus = coeffs_plus_init
        coeffs_minus = coeffs_minus_init

    for it in range(150):
        # Update tau_plus policy
        c_new_plus = solve_policy_bivariate_update_with_dist(
            coeffs_plus, coeffs_minus, nodes_a, nodes_z, T_inv,
            beta, sigma, psi, w, r, lam, delta, alpha, nu,
            A_MIN, A_MAX, Z_MIN, Z_MAX, z_quad, z_w, prob_tau_plus_arr,
            tau_plus, tau_minus, True
        )

        # Update tau_minus policy
        c_new_minus = solve_policy_bivariate_update_with_dist(
            coeffs_plus, coeffs_minus, nodes_a, nodes_z, T_inv,
            beta, sigma, psi, w, r, lam, delta, alpha, nu,
            A_MIN, A_MAX, Z_MIN, Z_MAX, z_quad, z_w, prob_tau_plus_arr,
            tau_plus, tau_minus, False
        )

        diff = max(np.max(np.abs(c_new_plus - coeffs_plus)),
                   np.max(np.abs(c_new_minus - coeffs_minus)))

        coeffs_plus = 0.8 * coeffs_plus + 0.2 * c_new_plus
        coeffs_minus = 0.8 * coeffs_minus + 0.2 * c_new_minus

        if diff < 1e-7:
            break

    return coeffs_plus, coeffs_minus

# =============================================================================
# Stationary Distribution (from v6, extended for distortions)
# =============================================================================

@njit(cache=True)
def get_interpolation_weights(x, grid):
    n = len(grid)
    if x <= grid[0]: return 0, 1.0, 0.0
    if x >= grid[-1]: return n-2, 0.0, 1.0
    idx = 0
    while idx < n-2 and grid[idx+1] < x: idx += 1
    w2 = (x - grid[idx]) / (grid[idx+1] - grid[idx])
    return idx, 1.0 - w2, w2

def compute_stationary_analytical(coeffs, a_grid, z_grid, prob_z, psi, mu_init=None):
    """Stationary distribution WITHOUT distortions (from v6)"""
    na, nz = len(a_grid), len(z_grid)
    n_states = na * nz

    A_prime_mat = bivariate_eval_matrix(a_grid, z_grid, coeffs, A_MIN, A_MAX, Z_MIN, Z_MAX)

    rows, cols, data = [], [], []
    for i_a in range(na):
        for i_z in range(nz):
            s = i_a * nz + i_z
            aprime = A_prime_mat[i_a, i_z]
            ia_low, w1, w2 = get_interpolation_weights(aprime, a_grid)
            for i_zp in range(nz):
                p_z = (psi + (1-psi)*prob_z[i_zp]) if i_zp == i_z else (1-psi)*prob_z[i_zp]
                if p_z > 1e-12:
                    if w1 > 1e-9:
                        rows.append(ia_low * nz + i_zp); cols.append(s); data.append(p_z * w1)
                    if w2 > 1e-9:
                        rows.append((ia_low+1) * nz + i_zp); cols.append(s); data.append(p_z * w2)
    Q = sparse.csr_matrix((data, (rows, cols)), shape=(n_states, n_states))

    if mu_init is not None:
        mu = mu_init
        for _ in range(50):
            mu_new = Q @ mu
            if np.max(np.abs(mu_new - mu)) < 1e-9: break
            mu = mu_new
        return (mu / mu.sum()).reshape((na, nz))

    try:
        vals, vecs = eigs(Q, k=1, which='LM')
        mu = np.abs(vecs[:, 0].real)
    except:
        mu = np.ones(n_states) / n_states
        for _ in range(500):
            mu_new = Q @ mu
            if np.max(np.abs(mu_new - mu)) < 1e-10: break
            mu = mu_new
    return (mu / mu.sum()).reshape((na, nz))

def compute_stationary_with_dist(coeffs_plus, coeffs_minus, a_grid, z_grid, prob_z,
                                  prob_tau_plus, psi, mu_init=None):
    """Stationary distribution WITH distortions"""
    na, nz = len(a_grid), len(z_grid)
    n_states = na * nz * 2  # 2 tau states

    A_prime_plus = bivariate_eval_matrix(a_grid, z_grid, coeffs_plus, A_MIN, A_MAX, Z_MIN, Z_MAX)
    A_prime_minus = bivariate_eval_matrix(a_grid, z_grid, coeffs_minus, A_MIN, A_MAX, Z_MIN, Z_MAX)

    rows, cols, data = [], [], []

    for i_a in range(na):
        for i_z in range(nz):
            for i_tau in range(2):  # 0=plus, 1=minus
                s = i_a * nz * 2 + i_z * 2 + i_tau

                aprime = A_prime_plus[i_a, i_z] if i_tau == 0 else A_prime_minus[i_a, i_z]
                ia_low, w1, w2 = get_interpolation_weights(aprime, a_grid)

                # Persistence: stay at (z, tau) with prob psi
                if w1 > 1e-9:
                    s_next = ia_low * nz * 2 + i_z * 2 + i_tau
                    rows.append(s_next); cols.append(s); data.append(psi * w1)
                if w2 > 1e-9 and ia_low + 1 < na:
                    s_next = (ia_low + 1) * nz * 2 + i_z * 2 + i_tau
                    rows.append(s_next); cols.append(s); data.append(psi * w2)

                # Redraw: with prob 1-psi, draw new (z', tau')
                for i_zp in range(nz):
                    p_zp = prob_z[i_zp]
                    if p_zp < 1e-12:
                        continue
                    p_tau_plus_new = prob_tau_plus[i_zp]

                    # Transition to tau_plus
                    if p_tau_plus_new > 1e-12:
                        if w1 > 1e-9:
                            s_next = ia_low * nz * 2 + i_zp * 2 + 0
                            rows.append(s_next); cols.append(s); data.append((1-psi) * p_zp * p_tau_plus_new * w1)
                        if w2 > 1e-9 and ia_low + 1 < na:
                            s_next = (ia_low + 1) * nz * 2 + i_zp * 2 + 0
                            rows.append(s_next); cols.append(s); data.append((1-psi) * p_zp * p_tau_plus_new * w2)

                    # Transition to tau_minus
                    p_tau_minus_new = 1 - p_tau_plus_new
                    if p_tau_minus_new > 1e-12:
                        if w1 > 1e-9:
                            s_next = ia_low * nz * 2 + i_zp * 2 + 1
                            rows.append(s_next); cols.append(s); data.append((1-psi) * p_zp * p_tau_minus_new * w1)
                        if w2 > 1e-9 and ia_low + 1 < na:
                            s_next = (ia_low + 1) * nz * 2 + i_zp * 2 + 1
                            rows.append(s_next); cols.append(s); data.append((1-psi) * p_zp * p_tau_minus_new * w2)

    Q = sparse.csr_matrix((data, (rows, cols)), shape=(n_states, n_states))

    if mu_init is not None:
        mu = mu_init
    else:
        mu = np.ones(n_states) / n_states

    for _ in range(500):
        mu_new = Q @ mu
        mu_new /= mu_new.sum()
        if np.max(np.abs(mu_new - mu)) < 1e-10:
            break
        mu = mu_new

    mu_full = mu.reshape((na, nz, 2))
    return mu_full[:, :, 0], mu_full[:, :, 1]  # mu_plus, mu_minus

# =============================================================================
# Aggregate Computation
# =============================================================================

@njit(cache=True)
def get_aggregates_analytical(dist, a_h, z_h, w, r, lam, delta, alpha, nu):
    """Aggregates WITHOUT distortions (from v6)"""
    na, nz = len(a_h), len(z_h)
    K, L, Y, En, A, Ext = 0., 0., 0., 0., 0., 0.
    for ia in range(na):
        a = a_h[ia]
        for iz in range(nz):
            wgt = dist[ia, iz]
            A += a * wgt
            p, ks, ld, out = solve_entrepreneur_single(a, z_h[iz], w, r, lam, delta, alpha, nu)
            if p > w:
                K += ks*wgt; L += ld*wgt; Y += out*wgt; En += wgt
                Ext += max(0.0, ks - a)*wgt
    return K, L, Y, En, A, Ext

@njit(cache=True)
def get_aggregates_with_dist(mu_plus, mu_minus, a_h, z_h, w, r, lam, delta, alpha, nu, tau_plus, tau_minus):
    """Aggregates WITH distortions"""
    na, nz = len(a_h), len(z_h)
    K, L, Y, En, A, Ext = 0., 0., 0., 0., 0., 0.

    for ia in range(na):
        a = a_h[ia]
        for iz in range(nz):
            z = z_h[iz]

            # tau_plus state
            wgt_p = mu_plus[ia, iz]
            A += a * wgt_p
            p_p, ks_p, ld_p, out_p = solve_entrepreneur_with_tau(a, z, tau_plus, w, r, lam, delta, alpha, nu)
            if p_p > w:
                K += ks_p * wgt_p; L += ld_p * wgt_p; Y += out_p * wgt_p; En += wgt_p
                Ext += max(0.0, ks_p - a) * wgt_p

            # tau_minus state
            wgt_m = mu_minus[ia, iz]
            A += a * wgt_m
            p_m, ks_m, ld_m, out_m = solve_entrepreneur_with_tau(a, z, tau_minus, w, r, lam, delta, alpha, nu)
            if p_m > w:
                K += ks_m * wgt_m; L += ld_m * wgt_m; Y += out_m * wgt_m; En += wgt_m
                Ext += max(0.0, ks_m - a) * wgt_m

    return K, L, Y, En, A, Ext

# =============================================================================
# GE Solver (from v6, extended for distortions)
# =============================================================================

def find_equilibrium(params, distortions=False, w_init=0.30, r_init=-0.02,
                     coeffs_init=None, coeffs_plus_init=None, coeffs_minus_init=None,
                     tau_plus=TAU_PLUS, tau_minus=TAU_MINUS, q_dist=Q_DIST):
    """
    Find general equilibrium.

    Args:
        params: (delta, alpha, nu, lam, beta, sigma, psi)
        distortions: whether to include tau wedges
    """
    delta, alpha, nu, lam, beta, sigma, psi = params
    w, r = w_init, r_init

    # Build grids
    na_h, nz_h = 600, 40
    a_h = np.exp(np.linspace(np.log(A_MIN+A_SHIFT), np.log(A_MAX+A_SHIFT), na_h)) - A_SHIFT
    M_v = np.concatenate([np.linspace(0.633, 0.998, 38), [0.999, 0.9995]])
    z_h = (1 - M_v)**(-1/ETA)
    pr_z = np.zeros(nz_h)
    pr_z[0] = M_v[0]
    pr_z[1:] = M_v[1:] - M_v[:-1]
    pr_z /= pr_z.sum()

    # Probability of tau_plus for each z
    prob_tau_plus = 1 - np.exp(-q_dist * z_h)

    # Initialize coefficients
    if distortions:
        coeffs_plus = coeffs_plus_init
        coeffs_minus = coeffs_minus_init
    else:
        coeffs = coeffs_init

    dist = None
    w_step, r_step = 0.1, 0.03
    exc_L_p, exc_K_p = 0.0, 0.0
    wh, rh = [], []

    for it in range(120):
        if distortions:
            coeffs_plus, coeffs_minus = solve_policy_spectral_with_dist(
                params, w, r, tau_plus, tau_minus, q_dist,
                coeffs_plus, coeffs_minus
            )
            mu_plus, mu_minus = compute_stationary_with_dist(
                coeffs_plus, coeffs_minus, a_h, z_h, pr_z, prob_tau_plus, psi
            )
            K_agg, L_d, Y_agg, share_en, A_agg, extfin = get_aggregates_with_dist(
                mu_plus, mu_minus, a_h, z_h, w, r, lam, delta, alpha, nu, tau_plus, tau_minus
            )
        else:
            coeffs = solve_policy_spectral(params, w, r, coeffs)
            dist = compute_stationary_analytical(coeffs, a_h, z_h, pr_z, psi,
                                                  mu_init=dist.ravel() if dist is not None else None)
            K_agg, L_d, Y_agg, share_en, A_agg, extfin = get_aggregates_analytical(
                dist, a_h, z_h, w, r, lam, delta, alpha, nu
            )

        if K_agg < 1e-4:
            w *= 0.98
            print(f"  [{it+1}] !!! COLLAPSE - w reset")
            continue

        ws = 1.0 - share_en
        eL, eK = (L_d - ws)/max(ws, 0.05), (K_agg - A_agg)/max(A_agg, 1.0)
        print(f"  [{it+1}] w={w:.6f}, r={r:.6f} | K={K_agg:.6f}, A={A_agg:.6f} | Ld={L_d:.6f}, Ls={ws:.6f}")

        if abs(eL) + abs(eK) < 2e-3:
            break

        if it > 0:
            if eL * exc_L_p < 0: w_step *= 0.5
            else: w_step = min(w_step * 1.05, 0.2)
            if eK * exc_K_p < 0: r_step *= 0.5
            else: r_step = min(r_step * 1.05, 0.05)

        dw, dr = max(-0.05, min(0.05, w_step*eL)), max(-0.01, min(0.01, r_step*eK))
        w_r, r_r = w * (1.0 + dw), r + dr
        wh.append(w); rh.append(r)

        if len(wh) > 3:
            w, r = 0.5*w_r + 0.5*np.median(wh[-4:]), 0.5*r_r + 0.5*np.median(rh[-4:])
        else:
            w, r = w_r, r_r

        exc_L_p, exc_K_p = eL, eK

    # Compute TFP
    span = 1 - nu
    L_s = 1.0 - share_en
    TFP = Y_agg / max(((K_agg ** alpha) * (L_s ** (1-alpha))) ** span, 1e-8)

    result = {
        'w': w, 'r': r, 'Y': Y_agg, 'K': K_agg, 'L': L_d, 'A': A_agg,
        'extfin': extfin, 'share_entre': share_en, 'TFP': TFP,
        'ExtFin_Y': extfin / max(Y_agg, 1e-8),
        'a_grid': a_h, 'z_grid': z_h, 'prob_z': pr_z,
    }

    if distortions:
        result['coeffs_plus'] = coeffs_plus
        result['coeffs_minus'] = coeffs_minus
        result['mu_plus'] = mu_plus
        result['mu_minus'] = mu_minus
        result['prob_tau_plus'] = prob_tau_plus
    else:
        result['coeffs'] = coeffs
        result['dist'] = dist

    return result

# =============================================================================
# Transition Path - Backward Policy Iteration
# =============================================================================

@njit(cache=True, parallel=True)
def solve_policy_transition_update(coeffs_curr, coeffs_next, nodes_a, nodes_z, T_az_inv,
                                    beta, sigma, psi, w_t, r_t, w_tp1, r_tp1,
                                    lam, delta, alpha, nu,
                                    a_min, a_max, z_min, z_max,
                                    quad_z, quad_w):
    """
    One step of backward policy iteration for transition (no distortions).
    Given policy at t+1 and prices at t and t+1, solve for policy at t.
    """
    nz_c = N_CHEBY_Z
    nt = len(nodes_a) * nz_c
    target_vals = np.zeros(nt)

    for i_n in prange(nt):
        ia, iz = i_n // nz_c, i_n % nz_c
        a, z = nodes_a[ia], nodes_z[iz]

        # Income at t
        p, _, _, _ = solve_entrepreneur_single(a, z, w_t, r_t, lam, delta, alpha, nu)
        inc = max(p, w_t) + (1 + r_t) * a

        # Current guess for a'
        aprime = bivariate_eval(a, z, coeffs_curr, a_min, a_max, z_min, z_max)
        aprime = max(a_min, min(aprime, inc - 1e-6))

        # Expected marginal utility at t+1 using next period's policy
        # Persistence branch
        p_p, _, _, _ = solve_entrepreneur_single(aprime, z, w_tp1, r_tp1, lam, delta, alpha, nu)
        inc_p = max(p_p, w_tp1) + (1 + r_tp1) * aprime
        app = bivariate_eval(aprime, z, coeffs_next, a_min, a_max, z_min, z_max)
        mu_pers = max(inc_p - app, 1e-9) ** (-sigma)

        # Redraw branch
        e_mu_redraw = 0.0
        for k in range(len(quad_z)):
            pk, _, _, _ = solve_entrepreneur_single(aprime, quad_z[k], w_tp1, r_tp1, lam, delta, alpha, nu)
            ick = max(pk, w_tp1) + (1 + r_tp1) * aprime
            akk = bivariate_eval(aprime, quad_z[k], coeffs_next, a_min, a_max, z_min, z_max)
            e_mu_redraw += quad_w[k] * max(ick - akk, 1e-9) ** (-sigma)

        expected_mu = psi * mu_pers + (1 - psi) * e_mu_redraw
        c_target = (beta * (1 + r_t) * expected_mu) ** (-1.0/sigma)
        target_vals[i_n] = max(a_min, min(inc - c_target, a_max))

    return T_az_inv @ target_vals

def solve_policy_transition_at_t(coeffs_next, w_t, r_t, w_tp1, r_tp1, params, coeffs_init=None):
    """Solve for policy at time t given policy at t+1"""
    delta, alpha, nu, lam, beta, sigma, psi = params
    nodes_a, nodes_z, T_full = generate_bivariate_nodes_matrix(A_MIN, A_MAX, Z_MIN, Z_MAX)
    T_inv = np.linalg.inv(T_full)

    M_vals = np.concatenate([np.linspace(0.633, 0.998, 38), [0.999, 0.9995]])
    z_quad = (1 - M_vals)**(-1/ETA)
    z_w = np.zeros(40)
    z_w[0] = M_vals[0]
    z_w[1:] = M_vals[1:] - M_vals[:-1]
    z_w /= z_w.sum()

    coeffs = coeffs_init if coeffs_init is not None else coeffs_next.copy()

    for it in range(100):
        c_new = solve_policy_transition_update(
            coeffs, coeffs_next, nodes_a, nodes_z, T_inv,
            beta, sigma, psi, w_t, r_t, w_tp1, r_tp1,
            lam, delta, alpha, nu,
            A_MIN, A_MAX, Z_MIN, Z_MAX, z_quad, z_w
        )
        diff = np.max(np.abs(c_new - coeffs))
        coeffs = 0.7 * coeffs + 0.3 * c_new
        if diff < 1e-7:
            break

    return coeffs

# =============================================================================
# Transition Path - Forward Distribution Update
# =============================================================================

def update_distribution_forward(dist_t, coeffs_t, a_grid, z_grid, prob_z, psi):
    """Update distribution from t to t+1"""
    na, nz = len(a_grid), len(z_grid)

    A_prime_mat = bivariate_eval_matrix(a_grid, z_grid, coeffs_t, A_MIN, A_MAX, Z_MIN, Z_MAX)

    dist_next = np.zeros((na, nz))

    for i_a in range(na):
        for i_z in range(nz):
            mass = dist_t[i_a, i_z]
            if mass < 1e-14:
                continue

            aprime = A_prime_mat[i_a, i_z]
            ia_low, w1, w2 = get_interpolation_weights(aprime, a_grid)

            for i_zp in range(nz):
                p_z = (psi + (1-psi)*prob_z[i_zp]) if i_zp == i_z else (1-psi)*prob_z[i_zp]
                if p_z > 1e-12:
                    if w1 > 1e-9:
                        dist_next[ia_low, i_zp] += mass * p_z * w1
                    if w2 > 1e-9 and ia_low + 1 < na:
                        dist_next[ia_low + 1, i_zp] += mass * p_z * w2

    return dist_next / dist_next.sum()

# =============================================================================
# Time Path Iteration (TPI)
# =============================================================================

def solve_transition(pre_eq, post_eq, params, T=250, kappa=0.05,
                     eta_w=0.3, eta_r=0.02, theta=0.5, tol=5e-3, max_iter=100):
    """
    Solve transition path from pre-reform to post-reform steady state.
    """
    delta, alpha, nu, lam, beta, sigma, psi = params

    # Grids
    a_grid = post_eq['a_grid']
    z_grid = post_eq['z_grid']
    prob_z = post_eq['prob_z']

    # Prices
    w_pre, r_pre = pre_eq['w'], pre_eq['r']
    w_post, r_post = post_eq['w'], post_eq['r']

    # Initial distribution: marginal over tau from pre-reform
    mu_0 = pre_eq['mu_plus'] + pre_eq['mu_minus']
    mu_0 /= mu_0.sum()

    # Terminal policy
    coeffs_post = post_eq['coeffs']

    # Initialize price paths
    t_arr = np.arange(T)
    w_path = w_post + (w_pre - w_post) * np.exp(-kappa * t_arr)
    r_path = r_post + (r_pre - r_post) * np.exp(-kappa * t_arr)

    print(f"\n{'='*70}")
    print("TRANSITION PATH ITERATION")
    print(f"{'='*70}")
    print(f"T = {T}, Initial: w_pre={w_pre:.6f}, r_pre={r_pre:.6f}")
    print(f"Target: w_post={w_post:.6f}, r_post={r_post:.6f}")
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

        # Terminal condition
        for t in range(T-1, -1, -1):
            if t == T-1:
                policies[t] = coeffs_post.copy()
            else:
                w_tp1 = w_path[t+1] if t+1 < T else w_post
                r_tp1 = r_path[t+1] if t+1 < T else r_post
                coeffs_next = policies[t+1] if t+1 < T else coeffs_post

                policies[t] = solve_policy_transition_at_t(
                    coeffs_next, w_path[t], r_path[t], w_tp1, r_tp1,
                    params, coeffs_init=policies[t+1] if t+1 < T else coeffs_post
                )

        # =================================================================
        # FORWARD: Update distributions and compute aggregates
        # =================================================================
        dist_t = mu_0.copy()

        for t in range(T):
            w_t, r_t = w_path[t], r_path[t]

            # Aggregates at t
            K, L, Y, En, A, Ext = get_aggregates_analytical(
                dist_t, a_grid, z_grid, w_t, r_t, lam, delta, alpha, nu
            )

            K_path[t] = K
            L_path[t] = L
            Y_path[t] = Y
            A_path[t] = A
            ExtFin_path[t] = Ext
            Entre_path[t] = En

            # TFP
            span = 1 - nu
            L_s = 1.0 - En
            TFP_path[t] = Y / max(((K ** alpha) * (L_s ** (1-alpha))) ** span, 1e-8)

            # Market clearing residuals
            ED_L_path[t] = (L - L_s) / max(L_s, 0.05)
            ED_K_path[t] = (K - A) / max(A, 1.0)

            # Update distribution
            if t < T - 1:
                dist_t = update_distribution_forward(dist_t, policies[t], a_grid, z_grid, prob_z, psi)

        # =================================================================
        # CHECK CONVERGENCE
        # =================================================================
        max_ED = max(np.max(np.abs(ED_L_path)), np.max(np.abs(ED_K_path)))

        if tpi_iter % 5 == 0 or tpi_iter < 3:
            print(f"[TPI iter {tpi_iter+1:3d}] max|ED_L|={np.max(np.abs(ED_L_path)):.6f}, "
                  f"max|ED_K|={np.max(np.abs(ED_K_path)):.6f}")

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

    # Save equilibria
    for name, eq in [('stationary_pre', pre_eq), ('stationary_post', post_eq)]:
        summary = {k: float(v) if isinstance(v, (int, float, np.floating)) else None
                   for k, v in eq.items() if not isinstance(v, np.ndarray)}
        with open(os.path.join(output_dir, f'{name}.json'), 'w') as f:
            json.dump(summary, f, indent=2)

    # Save transition CSV
    import csv
    with open(os.path.join(output_dir, 'transition.csv'), 'w', newline='') as f:
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

    # Output
    axes[0,0].plot(t, trans['Y']/Y_post, 'b-', lw=2)
    axes[0,0].axhline(1.0, color='r', ls='--', alpha=0.7)
    axes[0,0].axhline(pre_eq['Y']/Y_post, color='g', ls=':', alpha=0.7)
    axes[0,0].set_xlabel('Period'); axes[0,0].set_ylabel('Y / Y_post')
    axes[0,0].set_title('Output'); axes[0,0].grid(True, alpha=0.3)

    # TFP
    axes[0,1].plot(t, trans['TFP']/TFP_post, 'b-', lw=2)
    axes[0,1].axhline(1.0, color='r', ls='--', alpha=0.7)
    axes[0,1].axhline(pre_eq['TFP']/TFP_post, color='g', ls=':', alpha=0.7)
    axes[0,1].set_xlabel('Period'); axes[0,1].set_ylabel('TFP / TFP_post')
    axes[0,1].set_title('TFP'); axes[0,1].grid(True, alpha=0.3)

    # Interest Rate
    axes[0,2].plot(t, trans['r'], 'b-', lw=2)
    axes[0,2].axhline(post_eq['r'], color='r', ls='--', alpha=0.7)
    axes[0,2].axhline(pre_eq['r'], color='g', ls=':', alpha=0.7)
    axes[0,2].set_xlabel('Period'); axes[0,2].set_ylabel('r')
    axes[0,2].set_title('Interest Rate'); axes[0,2].grid(True, alpha=0.3)

    # Wage
    axes[1,0].plot(t, trans['w'], 'b-', lw=2)
    axes[1,0].axhline(post_eq['w'], color='r', ls='--', alpha=0.7)
    axes[1,0].axhline(pre_eq['w'], color='g', ls=':', alpha=0.7)
    axes[1,0].set_xlabel('Period'); axes[1,0].set_ylabel('w')
    axes[1,0].set_title('Wage'); axes[1,0].grid(True, alpha=0.3)

    # External Finance
    axes[1,1].plot(t, trans['ExtFin_Y'], 'b-', lw=2)
    axes[1,1].axhline(post_eq['ExtFin_Y'], color='r', ls='--', alpha=0.7)
    axes[1,1].axhline(pre_eq['ExtFin_Y'], color='g', ls=':', alpha=0.7)
    axes[1,1].set_xlabel('Period'); axes[1,1].set_ylabel('ExtFin/Y')
    axes[1,1].set_title('External Finance'); axes[1,1].grid(True, alpha=0.3)

    # Entrepreneur Share
    axes[1,2].plot(t, trans['Entre_share'], 'b-', lw=2)
    axes[1,2].axhline(post_eq['share_entre'], color='r', ls='--', alpha=0.7)
    axes[1,2].axhline(pre_eq['share_entre'], color='g', ls=':', alpha=0.7)
    axes[1,2].set_xlabel('Period'); axes[1,2].set_ylabel('Share')
    axes[1,2].set_title('Entrepreneur Share'); axes[1,2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'transition_dynamics.png'), dpi=150)
    plt.close()
    print(f"Plots saved to {output_dir}/transition_dynamics.png")

def print_summary(pre_eq, post_eq, trans):
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
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
    parser = argparse.ArgumentParser(description='Buera & Shin Transition Dynamics')
    parser.add_argument('--T', type=int, default=250, help='Transition horizon')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory')
    args = parser.parse_args()

    print("="*70)
    print("BUERA & SHIN (2010) TRANSITION DYNAMICS")
    print("With Idiosyncratic Output Distortions")
    print("="*70)
    print(f"\nParameters: lambda={LAMBDA}, tau_plus={TAU_PLUS}, tau_minus={TAU_MINUS}, q={Q_DIST}")
    print(f"Transition horizon T={args.T}")
    print("="*70)

    t_start = time.time()

    params = (DELTA, ALPHA, NU, LAMBDA, BETA, SIGMA, PSI)

    # Step 1: Post-reform equilibrium (NO distortions) - solve this first as it converges faster
    print("\n" + "="*70)
    print("STEP 1: Post-reform Stationary Equilibrium (NO distortions)")
    print("="*70)
    post_eq = find_equilibrium(params, distortions=False, w_init=0.80, r_init=-0.04)
    print(f"\nPost-reform: w={post_eq['w']:.4f}, r={post_eq['r']:.4f}, Y={post_eq['Y']:.4f}, TFP={post_eq['TFP']:.4f}")

    # Step 2: Pre-reform equilibrium (WITH distortions) - use post-reform values as initial guess
    print("\n" + "="*70)
    print("STEP 2: Pre-reform Stationary Equilibrium (WITH distortions)")
    print("="*70)
    pre_eq = find_equilibrium(params, distortions=True, w_init=post_eq['w'], r_init=post_eq['r'])
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
