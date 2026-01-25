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
import os
import sys
import argparse
from scipy import sparse
from scipy.sparse.linalg import eigs

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
A_MIN, A_MAX = 1e-6, 4000.0
A_SHIFT = 1.0  
# Paper's discretization begins at M=0.633 (e ~ 1.27)
Z_MIN, Z_MAX = 1.25, 6.5  
N_CHEBY_A = 24
N_CHEBY_Z = 40  # Higher density to smooth the occupational threshold

# Simulation Parameters
N_AGENTS = 100000
T_SIM = 500       # Increased for fewer 'jumps'
N_AGENTS_FINAL = 350000
T_SIM_FINAL = 750

# =============================================================================
# Optimized Bivariate spectral tools
# =============================================================================

@njit(cache=True)
def bivariate_eval(a, z, coeffs, a_min, a_max, z_min, z_max):
    """
    Allocation-free bivariate evaluation using direct recurrence.
    Uses log-mapping for asset dimension.
    """
    na, nz = N_CHEBY_A, N_CHEBY_Z
    
    # 1. Map a to x_a using log-transformation
    la, lmin, lmax = np.log(a + A_SHIFT), np.log(a_min + A_SHIFT), np.log(a_max + A_SHIFT)
    xa = 2.0 * (la - lmin) / (lmax - lmin) - 1.0
    
    # 2. Map z to x_z using linear transformation
    xz = 2.0 * (z - z_min) / (z_max - z_min) - 1.0
    
    # Clip to [-1, 1]
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
    # Clamping the output allows spectral interpolation while avoiding crazy extrapolation values
    return max(a_min, min(val, a_max))

def generate_bivariate_nodes_matrix(a_min, a_max, z_min, z_max):
    """Generate 2D nodes and the T matrix for coefficients fit"""
    # Asset nodes: Chebyshev in log-space
    cheb_a = Chebyshev_Nodes(N_CHEBY_A).ravel()
    lmin, lmax = np.log(a_min + A_SHIFT), np.log(a_max + A_SHIFT)
    l_nodes = (lmin + lmax)/2.0 + (lmax - lmin)/2.0 * cheb_a
    nodes_a = np.exp(l_nodes) - A_SHIFT
    
    # Ability nodes: Chebyshev in linear space
    cheb_z = Chebyshev_Nodes(N_CHEBY_Z).ravel()
    nodes_z = (z_min + z_max)/2.0 + (z_max - z_min)/2.0 * cheb_z
    
    # Pre-cache bases at nodes
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
# Entrepreneur Logic
# =============================================================================

@njit(cache=True)
def solve_entrepreneur_single(a, z, w, r, lam, delta, alpha, upsilon):
    rental = max(r + delta, 1e-8)
    wage = max(w, 1e-8)
    
    # Paper Production: Y = z * (k^alpha * l^(1-alpha))^span
    # where span = 1 - NU = 0.79
    span = 1 - NU
    
    # 1. Unconstrained: l = [ z * (k^alpha)^span * (1-alpha) * span / wage ] ^ (1 / (1 - (1-alpha)*span))
    # Optimal k1 = [ (alpha*span*z/rental) ^ (1-(1-alpha)span) * ((1-alpha)span*z/wage) ^ ((1-alpha)span) ] ^ (1/(1-span))
    
    # We use exponents matching the Buera-Shin paper derivation
    # denominator = 1 - alpha*span - (1-alpha)*span = 1 - span = NU
    
    exp_k = 1 - (1-alpha)*span
    exp_l = (1-alpha)*span
    
    k_unconstr = ( ( (alpha*span*z/rental)**exp_k ) * ( ((1-alpha)*span*z/wage)**exp_l ) ) ** (1/NU)
    
    kstar = min(k_unconstr, 1e8) # Safety cap
    kstar = min(kstar, lam * a)
    
    # 2. Optimal l given kstar:
    # dY/dl = wage => span * z * kstar^(alpha*span) * l^((1-alpha)*span - 1) * (1-alpha) = wage
    # Safety: ensure wage and z are positive
    denom = max( (z * (kstar**(alpha*span)) * (1-alpha) * span), 1e-12)
    lstar = ( (wage / denom) ) ** (1 / ((1-alpha)*span - 1))
    lstar = min(lstar, 1e8)
    
    output = z * ( (kstar**alpha) * (lstar**(1-alpha)) ) ** span
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
        # Damping: 0.7 old, 0.3 new for maximum stability
        coeffs = 0.7 * coeffs + 0.3 * c_new
        if diff < 1e-7: break
    return coeffs

# =============================================================================
# Parallel Simulation
# =============================================================================

@njit(cache=True, parallel=True)
def simulation_step_parallel(a_curr, z_curr, coeffs, reset_shocks, shocks_raw, eta):
    n = len(a_curr)
    a_next = np.zeros(n)
    z_next = np.zeros(n)
    
    for i in prange(n):
        if reset_shocks[i]:
            u = max(1e-10, shocks_raw[i] / 255.0)
            # Pareto Inverse CDF: z_min * (u)^(-1/eta) where u is U(0,1)
            # This follows paper's z >= 1.0 support
            z_next[i] = max(Z_MIN, min(Z_MIN * (u**(-1.0/eta)), Z_MAX))
        else:
            z_next[i] = z_curr[i]
            
        val = bivariate_eval(a_curr[i], z_next[i], coeffs, A_MIN, A_MAX, Z_MIN, Z_MAX)
        a_next[i] = max(A_MIN, min(val, A_MAX))
    return a_next, z_next

@njit(cache=True)
def get_interpolation_weights(x, grid):
    """Linear interpolation weights for Young's method."""
    n = len(grid)
    if x <= grid[0]: return 0, 1.0, 0
    if x >= grid[-1]: return n-2, 0.0, 1.0
    
    # Simple search
    idx = 0
    while idx < n-2 and grid[idx+1] < x:
        idx += 1
    
    w2 = (x - grid[idx]) / (grid[idx+1] - grid[idx])
    w1 = 1.0 - w2
    return idx, w1, w2

@njit(cache=True, parallel=True)
def build_transition_matrix_spectral(coeffs, a_grid, z_grid, prob_z, psi, 
                                     a_min, a_max, z_min, z_max):
    """
    Build sparse transition matrix using Young's method.
    Q[s', s] = P(s' | s)
    """
    na, nz = len(a_grid), len(z_grid)
    n_states = na * nz
    
    # We return components for a CSR matrix: data, indices, indptr
    # Max entries per column: 1 (psi) + 40 ((1-psi)*nz)
    # Actually each transition (a,z) -> (a',z') has 2 possible a' states
    # So max 2 * nz transitions per state.
    
    # Parallel construction of rows, cols, data
    # (Using a flat list in JIT is tricky, so we approximate or use fixed size)
    data = np.zeros(n_states * nz * 2)
    rows = np.zeros(n_states * nz * 2, dtype=np.int32)
    cols = np.zeros(n_states * nz * 2, dtype=np.int32)
    
    count = 0
    # For each state s = (i_a, i_z)
    for i_a in range(na):
        for i_z in range(nz):
            s = i_a * nz + i_z
            a_val, z_val = a_grid[i_a], z_grid[i_z]
            
            # Policy at current state
            aprime = bivariate_eval(a_val, z_val, coeffs, a_min, a_max, z_min, z_max)
            
            # Young's weights for a'
            ia_low, w1, w2 = get_interpolation_weights(aprime, a_grid)
            
            # For each possible next-period ability z'
            for i_zp in range(nz):
                # Prob(z' | z)
                p_z = (psi + (1-psi)*prob_z[i_zp]) if i_zp == i_z else (1-psi)*prob_z[i_zp]
                
                if p_z > 1e-10:
                    # Transition to (ia_low, i_zp)
                    s_low = ia_low * nz + i_zp
                    # Row s_low, Col s
                    # Wait, usually Q[next, curr]
                    # We'll use Q @ mu = mu where mu is column vector
                    if w1 > 1e-10:
                        # (Self-note: this needs to be atomic or collected safely if parallel)
                        # For simplicity, we'll do this sequentially for the matrix build
                        pass

    # Re-implementing non-parallel version for stability in CSR build
    return None

def compute_stationary_analytical(coeffs, a_grid, z_grid, prob_z, psi):
    na, nz = len(a_grid), len(z_grid)
    n_states = na * nz
    
    rows, cols, data = [], [], []
    
    for i_a in range(na):
        for i_z in range(nz):
            s = i_a * nz + i_z
            a_val, z_val = a_grid[i_a], z_grid[i_z]
            aprime = bivariate_eval(a_val, z_val, coeffs, A_MIN, A_MAX, Z_MIN, Z_MAX)
            ia_low, w1, w2 = get_interpolation_weights(aprime, a_grid)
            
            for i_zp in range(nz):
                p_z = (psi + (1-psi)*prob_z[i_zp]) if i_zp == i_z else (1-psi)*prob_z[i_zp]
                if p_z > 1e-12:
                    if w1 > 1e-9:
                        rows.append(ia_low * nz + i_zp)
                        cols.append(s)
                        data.append(p_z * w1)
                    if w2 > 1e-9:
                        rows.append((ia_low+1) * nz + i_zp)
                        cols.append(s)
                        data.append(p_z * w2)
                        
    Q = sparse.csr_matrix((data, (rows, cols)), shape=(n_states, n_states))
    
    # Solve Q @ mu = mu
    try:
        vals, vecs = eigs(Q, k=1, which='LM')
        mu = np.abs(vecs[:, 0].real)
        mu /= mu.sum()
    except:
        # Power iteration fallback
        mu = np.ones(n_states) / n_states
        for _ in range(500):
            mu_new = Q @ mu
            if np.max(np.abs(mu_new - mu)) < 1e-10: break
            mu = mu_new
        mu /= mu.sum()
        
    return mu.reshape((na, nz))

# =============================================================================
# GE Solver
# =============================================================================

def find_equilibrium(params, method='simulation', fixed_shocks=None, w_init=1.5, r_init=0.045, coeffs_init=None):
    delta, alpha, upsilon, lam, beta, sigma, psi = params
    w, r = w_init, r_init
    coeffs = coeffs_init
    w_step, r_step = 0.1, 0.03
    exc_L_p, exc_K_p = 0.0, 0.0
    
    # Analytical Grids
    na_hist, nz_hist = 600, 40
    a_hist = np.linspace(A_MIN, A_MAX, na_hist)
    # Standard Pareto discretization for z
    z_hist, prob_z = np.exp(np.linspace(np.log(Z_MIN), np.log(Z_MAX), nz_hist)), np.zeros(nz_hist)
    # (Simple mid-point weights for discrete z)
    z_hist = nodes_z_paper = (1 - np.linspace(0.633, 0.9995, 40))**(-1/ETA)
    prob_z = np.zeros(40)
    prob_z[0] = 0.633
    prob_z[1:38] = (np.linspace(0.633, 0.998, 38)[1:] - np.linspace(0.633, 0.998, 38)[:-1])
    prob_z[38] = 0.999 - 0.998
    prob_z[39] = 0.9995 - 0.999
    prob_z /= prob_z.sum()

    # Simulation Setup
    if method == 'simulation':
        res_sh, raw_sh = fixed_shocks
        t_sim, n_agents = res_sh.shape
        a_vec, z_vec = np.full(n_agents, 0.5), np.full(n_agents, Z_MIN)
    
    w_hist, r_hist = [], []
        coeffs = solve_policy_spectral(params, w, r, coeffs_init=coeffs)
        
        # Simulate and aggregate
        k_sum, l_sum, y_sum, en_sum, aa_sum, e_sum = 0., 0., 0., 0., 0., 0.
        n_sample = 0
        
        for t in range(t_sim):
            a, z = simulation_step_parallel(a, z, coeffs, res_sh[t], raw_sh[t], ETA)
            if t >= t_sim - 100:
                k_a, l_a, y_a, en_a, aa_a, e_a = get_aggregates_parallel(a, z, w, r, lam, delta, alpha, upsilon)
                k_sum+=k_a; l_sum+=l_a; y_sum+=y_a; en_sum+=en_a; aa_sum+=aa_a; e_sum+=e_a
                n_sample += 1
                
        K_agg, L_d, Y_agg, share_en, A_agg, extfin = [x/(n_agents * n_sample) for x in [k_sum, l_sum, y_sum, en_sum, aa_sum, e_sum]]
        
        # Singularity Recovery: if economy collapsed, kick it back to life to allow discovery
        if K_agg < 1e-4:
            a = np.full(n_agents, 0.5)
            # Force wage down slightly if at extinction to find the border
            w = w * 0.98
            print(f"  [{it+1}] !!! ECONOMY COLLAPSED (K=0) - Resetting wealth and forcing wage down...")
            continue

        # Market Errors
        workers_supply = 1.0 - share_en
        denom_L = max(workers_supply, 0.05)
        denom_A = max(A_agg, 1.0)
        
        err_L = (L_d - workers_supply) / denom_L
        err_K = (K_agg - A_agg) / denom_A
        
        print(f"  [{it+1}] w={w:.4f}, r={r:.4f} | K={K_agg:.2f}, A={A_agg:.2f} | Ld={L_d:.2f}, Ls={workers_supply:.2f}")
        
        if abs(err_L) + abs(err_K) < 2e-3: break
        
        # Exponential damping on sign flip
        if it > 0:
            if err_L * exc_L_p < 0: w_step *= 0.5
            else: w_step = min(w_step * 1.05, 0.2)
            
            if err_K * exc_K_p < 0: r_step *= 0.5
            else: r_step = min(r_step * 1.05, 0.05)
        
        # Combined damping
        # Limit price step to 5% for lambda < inf
        dw = max(-0.05, min(0.05, w_step * err_L))
        dr = max(-0.01, min(0.01, r_step * err_K))
        
        w_raw = w * (1.0 + dw)
        r_raw = r + dr
        
        # Rolling Median smoothing
        w_hist.append(w)
        r_hist.append(r)
        
        if len(w_hist) > 3:
            w = 0.5 * w_raw + 0.5 * np.median(w_hist[-4:])
            r = 0.5 * r_raw + 0.5 * np.median(r_hist[-4:])
        else:
            w, r = w_raw, r_raw
            
        exc_L_p, exc_K_p = err_L, err_K
        
    return {'w':w, 'r':r, 'Y':Y_agg, 'K':K_agg, 'L':L_d, 'A':A_agg, 'extfin':extfin, 'share_entre':share_en, 'coeffs':coeffs}

# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("BUERA-SHIN (2010): BIVARIATE SPECTRAL COLLOCATION SOLVER (V5)")
    print("=" * 80)
    
    print("\n[CONFIG: MODEL PARAMETERS]")
    print(f"  BETA={BETA:.3f}, SIGMA={SIGMA:.1f}, ALPHA={ALPHA:.2f}, NU={NU:.2f}")
    print(f"  DELTA={DELTA:.2f}, ETA={ETA:.2f}, PSI={PSI:.3f}")
    
    print("\n[CONFIG: APPROXIMATION DETAILS]")
    print(f"  Asset State a:  [{A_MIN}, {A_MAX}] (Nodes={N_CHEBY_A})")
    print(f"  Ability State z: [{Z_MIN:.4f}, {Z_MAX:.1f}] (Nodes={N_CHEBY_Z})")
    print(f"  Total Collocation Points: {N_CHEBY_A * N_CHEBY_Z}")
    
    print("\n[CONFIG: SIMULATION SCALE]")
    print(f"  Searching: {N_AGENTS:,} agents over {T_SIM} periods")
    print(f"  Final Refining: {N_AGENTS_FINAL:,} agents over {T_SIM_FINAL} periods")
    
    print("\n" + "-" * 80)
    print("1. Pre-generating Shocks...")
    np.random.seed(42)
    s_res = (np.random.rand(T_SIM, N_AGENTS) > PSI).astype(np.uint8)
    s_raw = np.random.randint(0, 256, (T_SIM, N_AGENTS), dtype=np.uint8)
    
    s_res_f = (np.random.rand(T_SIM_FINAL, N_AGENTS_FINAL) > PSI).astype(np.uint8)
    s_raw_f = np.random.randint(0, 256, (T_SIM_FINAL, N_AGENTS_FINAL), dtype=np.uint8)
    
    print("\n2. Discovery...")
    lambdas = [np.inf, 2.0, 1.5, 1.0]
    results = []
    # Using the scale of previous successful runs
    w_i, r_i, c_i = 1.5, 0.0450, None
    
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
