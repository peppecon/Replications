"""
Replication of Buera & Shin (2010) - JPE
VERSION 5: BIVARIATE SPECTRAL COLLOCATION

Features:
1. Endogenous Chebyshev nodes for both wealth (a) and ability (z)
2. Bivariate Tensor-Product basis T(a, z)
3. Parallelized simulation and aggregation (numba.prange)
4. Additive Walrasian price updates with adaptive damping
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numba import njit, prange
import time
import warnings
import os
import sys

# Import user library
from library.functions_library import *

warnings.filterwarnings('ignore')

# =============================================================================
# Model Parameters (Paper Table 1)
# =============================================================================
SIGMA = 1.5      # Risk aversion
BETA = 0.904     # Discount factor
ALPHA = 0.33     # Capital share
NU = 0.21        # Entrepreneur share (Span of control = 1 - NU = 0.79)
DELTA = 0.06     # Depreciation
ETA = 4.15       # Pareto tail
PSI = 0.894      # Persistence

# Global Grid Bounds
A_MIN, A_MAX = 1e-6, 500.0
Z_MIN, Z_MAX = 0.22, 2.0  # Endogenous range covering discretization artifacts
N_CHEBY_A = 20
N_CHEBY_Z = 20

# Simulation Parameters
N_AGENTS = 100000
T_SIM = 200
N_AGENTS_FINAL = 350000
T_SIM_FINAL = 500

# =============================================================================
# Optimized Bivariate spectral tools
# =============================================================================

@njit(cache=True)
def bivariate_eval(a, z, coeffs, a_min, a_max, z_min, z_max):
    """
    Allocation-free bivariate evaluation using direct recurrence.
    coeffs is size (N_A * N_Z) 
    """
    na, nz = N_CHEBY_A, N_CHEBY_Z
    
    # Map to [-1, 1]
    xa = 2.0 * a / (a_max - a_min) - (a_max + a_min) / (a_max - a_min)
    xz = 2.0 * z / (z_max - z_min) - (z_max + z_min) / (z_max - z_min)
    
    # Clip
    xa = max(-1.0, min(1.0, xa))
    xz = max(-1.0, min(1.0, xz))
    
    val, idx = 0.0, 0
    ta_m2, ta_m1 = 1.0, xa
    for ia in range(na):
        if ia == 0: t_a = 1.0
        elif ia == 1: t_a = xa
        else:
            t_a = 2.0 * xa * ta_m1 - ta_m2
            ta_m2, ta_m1 = ta_m1, t_a
        
        tz_m2, tz_m1 = 1.0, xz
        for iz in range(nz):
            if iz == 0: t_z = 1.0
            elif iz == 1: t_z = xz
            else:
                t_z = 2.0 * xz * tz_m1 - tz_m2
                tz_m2, tz_m1 = tz_m1, t_z
            val += coeffs[idx] * t_a * t_z
            idx += 1
    return val

def generate_bivariate_nodes_matrix(a_min, a_max, z_min, z_max):
    """Generate 2D nodes and the T matrix for coefficients fit"""
    nodes_a_cheb = Chebyshev_Nodes(N_CHEBY_A).ravel()
    nodes_z_cheb = Chebyshev_Nodes(N_CHEBY_Z).ravel()
    
    nodes_a = Change_Variable_Fromcheb(a_min, a_max, nodes_a_cheb)
    nodes_z = Change_Variable_Fromcheb(z_min, z_max, nodes_z_cheb)
    
    # Construct T matrix (na*nz, na*nz)
    Ta = Chebyshev_Polynomials_Recursion_mv(nodes_a_cheb, N_CHEBY_A)
    Tz = Chebyshev_Polynomials_Recursion_mv(nodes_z_cheb, N_CHEBY_Z)
    
    # T = Ta.T kron Tz.T
    T_full = np.zeros((N_CHEBY_A * N_CHEBY_Z, N_CHEBY_A * N_CHEBY_Z))
    idx = 0
    for ia in range(N_CHEBY_A):
        for iz in range(N_CHEBY_Z):
            row = np.zeros(N_CHEBY_A * N_CHEBY_Z)
            # Basis eval at node (ia, iz)
            j_idx = 0
            for ja in range(N_CHEBY_A):
                for jz in range(N_CHEBY_Z):
                    row[j_idx] = Ta[ja, ia] * Tz[jz, iz]
                    j_idx += 1
            T_full[idx, :] = row
            idx += 1
            
    return nodes_a, nodes_z, T_full

# =============================================================================
# Entrepreneur Logic
# =============================================================================

@njit(cache=True)
def solve_entrepreneur_single(a, z, w, r, lam, delta, alpha, upsilon):
    rental = max(r + delta, 1e-8)
    wage = max(w, 1e-8)
    span = 1 - upsilon
    
    aux1 = (1/rental) * alpha * span * z
    aux2 = (1/wage) * (1-alpha) * span * z
    k_unconstr = ( (aux1 ** (1-(1-alpha)*span)) * (aux2 ** ((1-alpha)*span)) ) ** (1/upsilon)
    
    kstar = min(k_unconstr, lam * a)
    inside_lab = (1/wage) * (1-alpha) * span * z * (kstar ** (alpha * span))
    lstar = inside_lab ** (1/(1-(1-alpha)*span))
    
    output = z * ((kstar ** alpha) * (lstar ** (1-alpha))) ** span
    profit = output - wage * lstar - rental * kstar
    
    return profit, kstar, lstar, output

# =============================================================================
# Policy Iteration
# =============================================================================

@njit(cache=True, parallel=True)
def solve_policy_bivariate_update(coeffs, nodes_a, nodes_z, T_az_inv, 
                                 beta, sigma, psi, w, r, lam, delta, alpha, upsilon, 
                                 a_min, a_max, z_min, z_max, 
                                 quad_z, quad_w):
    nz = N_CHEBY_Z
    n_total = len(nodes_a) * nz
    target_vals = np.zeros(n_total)
    
    for i_n in prange(n_total):
        ia, iz = i_n // nz, i_n % nz
        a, z = nodes_a[ia], nodes_z[iz]
        
        profit, _, _, _ = solve_entrepreneur_single(a, z, w, r, lam, delta, alpha, upsilon)
        inc = max(profit, w) + (1 + r) * a
        
        # Current policy guess
        aprime = bivariate_eval(a, z, coeffs, a_min, a_max, z_min, z_max)
        aprime = max(a_min, min(aprime, inc - 1e-6))
        
        # E[mu'|z] = psi * mu(a', z) + (1-psi) * integral mu(a', z')
        p_p, _, _, _ = solve_entrepreneur_single(aprime, z, w, r, lam, delta, alpha, upsilon)
        inc_p = max(p_p, w) + (1 + r) * aprime
        app = bivariate_eval(aprime, z, coeffs, a_min, a_max, z_min, z_max)
        mu_persistent = max(inc_p - app, 1e-9) ** (-sigma)
        
        e_mu_reset = 0.0
        for k in range(len(quad_z)):
            pk, _, _, _ = solve_entrepreneur_single(aprime, quad_z[k], w, r, lam, delta, alpha, upsilon)
            ick = max(pk, w) + (1 + r) * aprime
            akk = bivariate_eval(aprime, quad_z[k], coeffs, a_min, a_max, z_min, z_max)
            e_mu_reset += quad_w[k] * max(ick - akk, 1e-9) ** (-sigma)
            
        expected_mu = psi * mu_persistent + (1 - psi) * e_mu_reset
        c_target = (beta * (1 + r) * expected_mu) ** (-1.0/sigma)
        
        target_vals[i_n] = max(a_min, min(inc - c_target, a_max))
        
    return T_az_inv @ target_vals

def solve_policy_spectral(params, w, r, coeffs_init=None):
    delta, alpha, upsilon, lam, beta, sigma, psi = params
    nodes_a, nodes_z, T_full = generate_bivariate_nodes_matrix(A_MIN, A_MAX, Z_MIN, Z_MAX)
    T_inv = np.linalg.inv(T_full)
    
    # Pareto quadrature
    z_quad = np.exp(np.linspace(np.log(Z_MIN), np.log(Z_MAX), 40))
    weights = z_quad ** (-(ETA + 1))
    z_weights = weights / weights.sum()
    
    if coeffs_init is None:
        flat = np.zeros(len(nodes_a) * len(nodes_z))
        for i in range(len(flat)):
            ia, iz = i // len(nodes_z), i % len(nodes_z)
            p, _, _, _ = solve_entrepreneur_single(nodes_a[ia], nodes_z[iz], w, r, lam, delta, alpha, upsilon)
            flat[i] = 0.8 * (max(p, w) + (1+r)*nodes_a[ia])
        coeffs = T_inv @ flat
    else:
        coeffs = coeffs_init
        
    for it in range(150):
        c_new = solve_policy_bivariate_update(coeffs, nodes_a, nodes_z, T_inv, 
                                            beta, sigma, psi, w, r, lam, delta, alpha, upsilon,
                                            A_MIN, A_MAX, Z_MIN, Z_MAX, z_quad, z_weights)
        diff = np.max(np.abs(c_new - coeffs))
        coeffs = 0.2 * coeffs + 0.8 * c_new
        if diff < 1e-6: break
    return coeffs

# =============================================================================
# Parallel Simulation
# =============================================================================

@njit(cache=True, parallel=True)
def simulation_step_parallel(a_curr, z_curr, coeffs, reset_shocks, shocks_raw):
    n = len(a_curr)
    a_next = np.zeros(n)
    z_next = np.zeros(n)
    nu = 2.0
    
    for i in prange(n):
        if reset_shocks[i]:
            u = max(1e-10, shocks_raw[i] / 255.0)
            z_next[i] = max(Z_MIN, min(Z_MIN * (u**(-1/nu)), Z_MAX))
        else:
            z_next[i] = z_curr[i]
            
        val = bivariate_eval(a_curr[i], z_next[i], coeffs, A_MIN, A_MAX, Z_MIN, Z_MAX)
        a_next[i] = max(A_MIN, min(val, A_MAX))
    return a_next, z_next

@njit(cache=True, parallel=True)
def get_aggregates_parallel(a_vec, z_vec, w, r, lam, delta, alpha, upsilon):
    n = len(a_vec)
    K, L_d, Y, En, A, E = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    for i in prange(n):
        prof, ks, ld, out = solve_entrepreneur_single(a_vec[i], z_vec[i], w, r, lam, delta, alpha, upsilon)
        A += a_vec[i]
        if prof > w:
            K += ks; L_d += ld; Y += out; En += 1.0; E += max(0.0, ks - a_vec[i])
    return K, L_d, Y, En, A, E

# =============================================================================
# GE Solver
# =============================================================================

def find_equilibrium(params, fixed_shocks, w_init, r_init, coeffs_init=None):
    delta, alpha, upsilon, lam, beta, sigma, psi = params
    w, r = w_init, r_init
    coeffs = coeffs_init
    w_step, r_step = 0.2, 0.05
    exc_L_p, exc_K_p = 0.0, 0.0
    
    for it in range(80):
        coeffs = solve_policy_spectral(params, w, r, coeffs_init=coeffs)
        
        # Simulate
        res_sh, raw_sh = fixed_shocks
        t_sim, n_agents = res_sh.shape
        a, z = np.full(n_agents, 1e-2), np.full(n_agents, Z_MIN)
        
        for t in range(t_sim // 2):
            a, z = simulation_step_parallel(a, z, coeffs, res_sh[t], raw_sh[t])
            
        K_s, L_s, Y_s, En_s, A_s, E_s = 0., 0., 0., 0., 0., 0.
        n_p = t_sim - (t_sim // 2)
        for t in range(t_sim // 2, t_sim):
            a, z = simulation_step_parallel(a, z, coeffs, res_sh[t], raw_sh[t])
            k, l, y, en, aa, ee = get_aggregates_parallel(a, z, w, r, lam, delta, alpha, upsilon)
            K_s += k; L_s += l; Y_s += y; En_s += en; A_s += aa; E_s += ee
            
        K_agg, L_d, Y_agg, share_en, A_agg, extfin = [x/(n_agents*n_p) for x in [K_s, L_s, Y_s, En_s, A_s, E_s]]
        
        exc_L = (1.0 - share_en) - L_d
        exc_K = (A_agg - K_agg) / max(K_agg, 1e-6)
        
        print(f"  [{it+1}] w={w:.4f}, r={r:.4f}, ExcL={exc_L:.4f}, ExcK={exc_K:.4f}")
        if abs(exc_L) + abs(exc_K) < 1e-3: break
        
        if it > 0:
            w_step = w_step * 0.6 if exc_L * exc_L_p < 0 else min(w_step * 1.1, 0.5)
            r_step = r_step * 0.6 if exc_K * exc_K_p < 0 else min(r_step * 1.1, 0.1)
        
        w = max(0.01, w - w_step * exc_L)
        r = max(-delta + 0.001, min(r - r_step * exc_K, 0.12))
        exc_L_p, exc_K_p = exc_L, exc_K
        
    return {'w':w, 'r':r, 'Y':Y_agg, 'K':K_agg, 'L':L_d, 'A':A_agg, 'extfin':extfin, 'share_entre':share_en, 'coeffs':coeffs}

# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("\n1. Pre-generating Shocks...")
    np.random.seed(42)
    s_res = (np.random.rand(T_SIM, N_AGENTS) > PSI).astype(np.uint8)
    s_raw = np.random.randint(0, 256, (T_SIM, N_AGENTS), dtype=np.uint8)
    
    s_res_f = (np.random.rand(T_SIM_FINAL, N_AGENTS_FINAL) > PSI).astype(np.uint8)
    s_raw_f = np.random.randint(0, 256, (T_SIM_FINAL, N_AGENTS_FINAL), dtype=np.uint8)
    
    print("\n2. Discovery...")
    lambdas = [np.inf, 2.0, 1.5, 1.0]
    results = []
    w_i, r_i, c_i = 1.73, 0.045, None
    
    for lam in lambdas:
        print(f"\n--- Lambda = {lam} ---")
        p = (DELTA, ALPHA, NU, lam, BETA, SIGMA, PSI)
        res = find_equilibrium(p, (s_res, s_raw), w_i, r_i, coeffs_init=c_i)
        res['lambda'] = lam
        results.append(res)
        w_i, r_i, c_i = res['w'], res['r'], res['coeffs']
    
    print("\n3. Summary")
    for r in results:
        print(f"L={r['lambda']:>5}: Y={r['Y']:.4f}, r={r['r']:.4f}, w={r['w']:.4f}")
