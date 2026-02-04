"""
Replication of Buera & Shin (2010) - JPE
VERSION 6: PERFORMANCE SUPERCHARGED SPECTRAL COLLOCATION

Performance Innovations:
1. Matrix-Matrix Chebyshev evaluation (BLAS-optimized)
2. Stationary distribution warm-start (Power Iteration)
3. Vectorized integration and early-exit policy solving
4. Adaptive Walrasian clearing (stability maintained)
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
import warnings

# Import user library (recursion/nodes)
from shared_library.functions import *

warnings.filterwarnings('ignore')

# =============================================================================
# Model Parameters (Paper Table 1)
# =============================================================================
SIGMA = 1.5      # Risk aversion
BETA = 0.904     # Discount factor
ALPHA = 0.33     # Capital share
NU = 0.21        # Entrepreneur share (Span of control = 0.79)
DELTA = 0.06     # Depreciation
ETA = 4.15       # Pareto tail
PSI = 0.894      # Persistence

# Global Grid Bounds
A_MIN, A_MAX = 1e-6, 4000.0
A_SHIFT = 1.0  
# Paper's discretization (v3 bounds): M=0.633 (e ~ 1.2675) and 0.9995 (e ~ 6.2164)
# Z_MIN, Z_MAX = (1 - 0.633)**(-1/4.15), (1 - 0.9995)**(-1/4.15)
Z_MIN, Z_MAX = (1 - 0.001)**(-1/4.15), (1 - 0.9995)**(-1/4.15)
N_CHEBY_A = 30
N_CHEBY_Z = 20  

# Simulation Parameters
N_AGENTS = 100000
T_SIM = 500

# =============================================================================
# Optimized Bivariate spectral tools
# =============================================================================

@njit(cache=True)
def bivariate_eval(a, z, coeffs, a_min, a_max, z_min, z_max):
    na, nz = N_CHEBY_A, N_CHEBY_Z
    la, lmin, lmax = np.log(a + A_SHIFT), np.log(a_min + A_SHIFT), np.log(a_max + A_SHIFT)
    xa = max(-1.0, min(1.0, 2.0 * (la - lmin) / (lmax - lmin) - 1.0))
    xz = max(-1.0, min(1.0, 2.0 * (z - z_min) / (z_max - z_min) - 1.0))
    
    # 1D Chebyshev components
    Ta = np.zeros(na); Ta[0] = 1.0
    if na > 1: Ta[1] = xa
    for i in range(2, na): Ta[i] = 2.0 * xa * Ta[i-1] - Ta[i-2]
    Tz = np.zeros(nz); Tz[0] = 1.0
    if nz > 1: Tz[1] = xz
    for j in range(2, nz): Tz[j] = 2.0 * xz * Tz[j-1] - Tz[j-2]
    
    # Outer product contraction: val = sum_ij C_ij * Ta_i * Tz_j
    val = 0.0
    C = coeffs.reshape((na, nz))
    for i in range(na):
        row_sum = 0.0
        for j in range(nz): row_sum += C[i, j] * Tz[j]
        val += Ta[i] * row_sum
        
    return max(a_min, min(val, a_max))

def bivariate_eval_matrix(a_vec, z_vec, coeffs, a_min, a_max, z_min, z_max):
    """Batch evaluation for grids: Y = Ta @ C @ Tz.T"""
    na, nz = N_CHEBY_A, N_CHEBY_Z
    lmin, lmax = np.log(a_min + A_SHIFT), np.log(a_max + A_SHIFT)
    xa_vec = 2.0 * (np.log(a_vec + A_SHIFT) - lmin) / (lmax - lmin) - 1.0
    xz_vec = 2.0 * (z_vec - z_min) / (z_max - z_min) - 1.0
    
    # Linear projection matrices
    Ta = Chebyshev_Polynomials_Recursion_mv(xa_vec, na).T # (len(a_vec), na)
    Tz = Chebyshev_Polynomials_Recursion_mv(xz_vec, nz).T # (len(z_vec), nz)
    
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
# Entrepreneur Logic
# =============================================================================

@njit(cache=True)
def solve_entrepreneur_single(a, z, w, r, lam, delta, alpha, upsilon):
    rental = max(r + delta, 1e-8)
    wage = max(w, 1e-8)
    span = 1 - upsilon
    exp_k = 1 - (1-alpha)*span
    exp_l = (1-alpha)*span
    k1 = ( ( (alpha*span*z/rental)**exp_k ) * ( ((1-alpha)*span*z/wage)**exp_l ) ) ** (1/NU)
    kstar = min(k1, 1e8)
    kstar = min(kstar, lam * a)
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
    nz_c = N_CHEBY_Z
    nt = len(nodes_a) * nz_c
    target_vals = np.zeros(nt)
    for i_n in prange(nt):
        ia, iz = i_n // nz_c, i_n % nz_c
        a, z = nodes_a[ia], nodes_z[iz]
        p, _, _, _ = solve_entrepreneur_single(a, z, w, r, lam, delta, alpha, upsilon)
        inc = max(p, w) + (1 + r) * a
        aprime = bivariate_eval(a, z, coeffs, a_min, a_max, z_min, z_max)
        aprime = max(a_min, min(aprime, inc - 1e-6))
        p_p, _, _, _ = solve_entrepreneur_single(aprime, z, w, r, lam, delta, alpha, upsilon)
        inc_p = max(p_p, w) + (1 + r) * aprime
        app = bivariate_eval(aprime, z, coeffs, a_min, a_max, z_min, z_max)
        mu_pers = max(inc_p - app, 1e-9) ** (-sigma)
        e_mu_res = 0.0
        for k in range(len(quad_z)):
            pk, _, _, _ = solve_entrepreneur_single(aprime, quad_z[k], w, r, lam, delta, alpha, upsilon)
            ick = max(pk, w) + (1 + r) * aprime
            akk = bivariate_eval(aprime, quad_z[k], coeffs, a_min, a_max, z_min, z_max)
            e_mu_res += quad_w[k] * max(ick - akk, 1e-9) ** (-sigma)
        expected_mu = psi * mu_pers + (1 - psi) * e_mu_res
        c_target = (beta * (1 + r) * expected_mu) ** (-1.0/sigma)
        target_vals[i_n] = max(a_min, min(inc - c_target, a_max))
    return T_az_inv @ target_vals

def solve_policy_spectral(params, w, r, coeffs_init=None):
    delta, alpha, upsilon, lam, beta, sigma, psi = params
    nodes_a, nodes_z, T_full = generate_bivariate_nodes_matrix(A_MIN, A_MAX, Z_MIN, Z_MAX)
    T_inv = np.linalg.inv(T_full)
    M_vals = np.concatenate([np.linspace(0.0001, 0.998, 38), [0.999, 0.9995]])
    z_quad = (1 - M_vals)**(-1/ETA)
    z_w = np.zeros(40)
    z_w[0] = M_vals[0]
    z_w[1:] = M_vals[1:] - M_vals[:-1]
    z_w /= z_w.sum()
    if coeffs_init is None:
        flat = np.zeros(len(nodes_a) * len(nodes_z))
        for i in range(len(flat)):
            ia, iz = i // len(nodes_z), i % len(nodes_z)
            p, _, _, _ = solve_entrepreneur_single(nodes_a[ia], nodes_z[iz], w, r, lam, delta, alpha, upsilon)
            flat[i] = 0.8 * (max(p, w) + (1+r)*nodes_a[ia])
        coeffs = T_inv @ flat
    else: coeffs = coeffs_init
    for it in range(150):
        c_new = solve_policy_bivariate_update(coeffs, nodes_a, nodes_z, T_inv, 
                                            beta, sigma, psi, w, r, lam, delta, alpha, upsilon,
                                            A_MIN, A_MAX, Z_MIN, Z_MAX, z_quad, z_w)
        diff = np.max(np.abs(c_new - coeffs))
        coeffs = 0.8 * coeffs + 0.2 * c_new
        if diff < 1e-7: break
    return coeffs

# =============================================================================
# Stationary Distribution Logic
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
    na, nz = len(a_grid), len(z_grid)
    n_states = na * nz
    
    # Batch evaluate policies for speed
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
    
    # Warm-start Power Iteration (Much faster than eigs)
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

@njit(cache=True, parallel=True)
def simulation_step_parallel(a_curr, z_curr, coeffs, reset_shocks, shocks_raw, eta):
    n = len(a_curr)
    a_next, z_next = np.zeros(n), np.zeros(n)
    for i in prange(n):
        if reset_shocks[i]:
            u = max(1e-10, shocks_raw[i] / 255.0)
            z_next[i] = max(Z_MIN, min(Z_MIN * (u**(-1.0/eta)), Z_MAX))
        else: z_next[i] = z_curr[i]
        a_next[i] = bivariate_eval(a_curr[i], z_next[i], coeffs, A_MIN, A_MAX, Z_MIN, Z_MAX)
    return a_next, z_next

@njit(cache=True, parallel=True)
def get_aggregates_parallel(a_vec, z_vec, w, r, lam, delta, alpha, upsilon):
    n = len(a_vec)
    K, L, Y, En, A, E = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    for i in prange(n):
        p, ks, ld, out = solve_entrepreneur_single(a_vec[i], z_vec[i], w, r, lam, delta, alpha, upsilon)
        A += a_vec[i]
        if p > w:
            K += ks; L += ld; Y += out; En += 1.0; E += max(0.0, ks - a_vec[i])
    return K, L, Y, En, A, E

# =============================================================================
# GE Solver
# =============================================================================

@njit(cache=True)
def get_aggregates_analytical(dist, a_h, z_h, w, r, lam, delta, alpha, upsilon):
    na, nz = len(a_h), len(z_h)
    K, L, Y, En, A, Ext = 0., 0., 0., 0., 0., 0.
    for ia in range(na):
        a = a_h[ia]
        for iz in range(nz):
            wgt = dist[ia, iz]
            A += a * wgt
            p, ks, ld, out = solve_entrepreneur_single(a, z_h[iz], w, r, lam, delta, alpha, upsilon)
            if p > w:
                K += ks*wgt; L += ld*wgt; Y += out*wgt; En += wgt
                Ext += max(0.0, ks - a)*wgt
    return K, L, Y, En, A, Ext

def find_equilibrium(params, method='analytical', fixed_shocks=None, w_init=1.5, r_init=0.045, coeffs_init=None):
    delta, alpha, upsilon, lam, beta, sigma, psi = params
    w, r = w_init, r_init
    coeffs = coeffs_init
    dist = None
    w_step, r_step = 0.1, 0.03
    exc_L_p, exc_K_p = 0.0, 0.0
    na_h, nz_h = 600, 40
    a_h = np.exp(np.linspace(np.log(A_MIN+A_SHIFT), np.log(A_MAX+A_SHIFT), na_h)) - A_SHIFT
    M_v = np.concatenate([np.linspace(0.0001, 0.998, 38), [0.999, 0.9995]])
    z_h = (1 - M_v)**(-1/ETA)
    pr_z = np.zeros(nz_h); pr_z[0] = M_v[0]; pr_z[1:] = M_v[1:] - M_v[:-1]; pr_z /= pr_z.sum()
    if method == 'simulation':
        rs, rws = fixed_shocks; t_sim, n_a = rs.shape
        av, zv = np.full(n_a, 0.5), np.full(n_a, Z_MIN)
    wh, rh = [], []
    for it in range(120):
        coeffs = solve_policy_spectral(params, w, r, coeffs_init=coeffs)
        if method == 'simulation':
            k_s, l_s, y_s, en_s, aa_s, e_s = 0., 0., 0., 0., 0., 0.
            ns = 0
            for t in range(t_sim):
                av, zv = simulation_step_parallel(av, zv, coeffs, rs[t], rws[t], ETA)
                if t >= t_sim - 100:
                    ka, la, ya, ena, aaa, ea = get_aggregates_parallel(av, zv, w, r, lam, delta, alpha, upsilon)
                    k_s+=ka; l_s+=la; ya_s+=ya; en_s+=ena; aa_s+=aaa; e_s+=ea; ns += 1
            K_agg, L_d, Y_agg, share_en, A_agg, extfin = [x/(n_a * ns) for x in [k_s, l_s, y_s, en_s, aa_s, e_s]]
        else:
            dist = compute_stationary_analytical(coeffs, a_h, z_h, pr_z, psi, mu_init=dist.ravel() if dist is not None else None)
            K_agg, L_d, Y_agg, share_en, A_agg, extfin = get_aggregates_analytical(dist, a_h, z_h, w, r, lam, delta, alpha, upsilon)
        if K_agg < 1e-4:
            if method == 'simulation': av = np.full(n_a, 0.5)
            w *= 0.98; print(f"  [{it+1}] !!! COLLAPSE - w reset"); continue
        ws = 1.0 - share_en
        eL, eK = (L_d - ws)/max(ws, 0.05), (K_agg - A_agg)/max(A_agg, 1.0)
        print(f"  [{it+1}] w={w:.8f}, r={r:.8f} | K={K_agg:.8f}, A={A_agg:.8f} | Ld={L_d:.8f}, Ls={ws:.8f}")
        if abs(eL) + abs(eK) < 2e-4: break
        if it > 0:
            if eL * exc_L_p < 0: w_step *= 0.5
            else: w_step = min(w_step * 1.05, 0.2)
            if eK * exc_K_p < 0: r_step *= 0.5
            else: r_step = min(r_step * 1.05, 0.05)
        dw, dr = max(-0.05, min(0.05, w_step*eL)), max(-0.01, min(0.01, r_step*eK))
        w_r, r_r = w * (1.0 + dw), r + dr; wh.append(w); rh.append(r)
        if len(wh) > 3: w, r = 0.5*w_r + 0.5*np.median(wh[-4:]), 0.5*r_r + 0.5*np.median(rh[-4:])
        else: w, r = w_r, r_r
        exc_L_p, exc_K_p = eL, eK
    return {'w':w, 'r':r, 'Y':Y_agg,'K':K_agg,'L':L_d,'A':A_agg,'extfin':extfin,'share_entre':share_en,'coeffs':coeffs}

# =============================================================================
# Execution
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="analytical", choices=["analytical", "simulation"])
    args = parser.parse_args()
    print("=" * 80); print(f"BUERA-SHIN (2010): BIVARIATE SPECTRAL COLLOCATION SOLVER (V5) - {args.method.upper()}"); print("=" * 80)
    print(f"\n[CONFIG: APPROXIMATION]\n  Assets: [{A_MIN}, {A_MAX}] ({N_CHEBY_A})\n  Ability: [{Z_MIN}, {Z_MAX}] ({N_CHEBY_Z})")
    sh = None
    if args.method == "simulation":
        print(f"  Scale: {N_AGENTS} agents over {T_SIM} periods\n\n1. Pre-generating Shocks...")
        np.random.seed(42); s_r = (np.random.rand(T_SIM, N_AGENTS) > PSI).astype(np.uint8)
        s_w = np.random.randint(0, 256, (T_SIM, N_AGENTS), dtype=np.uint8); sh = (s_r, s_w)
    lambdas = [np.inf, 2.0, 1.5, 1.35, 1.0]; results = []; w_i, r_i, c_i = 1.5, 0.0450, None
    for lam in lambdas:
        print(f"\n--- Lambda = {lam} ---")
        p = (DELTA, ALPHA, NU, lam, BETA, SIGMA, PSI)
        res = find_equilibrium(p, method=args.method, fixed_shocks=sh, w_init=w_i, r_init=r_i, coeffs_init=c_i)
        res['lambda'] = lam; results.append(res); w_i, r_i, c_i = res['w'], res['r'], res['coeffs']
    print("\n3. Results Summary")
    print(f"{'Lambda':>8} {'ExtFin/Y':>10} {'GDP':>10} {'r':>10} {'w':>10} {'TFP':>10}")
    print("-" * 65)
    
    span = 1 - NU
    plot_data = {'ext_fin':[], 'gdp':[], 'r':[], 'tfp':[]}
    
    for r in results:
        lam_s = "inf" if r['lambda'] == np.inf else f"{r['lambda']:.2f}"
        L_s = 1.0 - r['share_entre']
        curr_tfp = r['Y'] / max(( (r['K']**ALPHA * L_s**(1-ALPHA))**span ), 1e-8)
        ext_to_y = r['extfin'] / max(r['Y'], 1e-8)
        
        print(f"{lam_s:>8} {ext_to_y:>10.4f} {r['Y']:>10.4f} {r['r']:>10.4f} {r['w']:>10.4f} {curr_tfp:>10.4f}")
        
        plot_data['ext_fin'].append(ext_to_y)
        plot_data['gdp'].append(r['Y'])
        plot_data['r'].append(r['r'])
        plot_data['tfp'].append(curr_tfp)

    # Normalized Plotting (Relative to Perfect Credit)
    os.makedirs("plots", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    Y_perfect = plot_data['gdp'][0]
    TFP_perfect = plot_data['tfp'][0]
    Y_norm = [y / Y_perfect for y in plot_data['gdp']]
    TFP_norm = [t / TFP_perfect for t in plot_data['tfp']]
    
    # Left: GDP and TFP (Relative)
    axes[0].plot(plot_data['ext_fin'], Y_norm, 'b-o', label='GDP', lw=2, markersize=10)
    axes[0].plot(plot_data['ext_fin'], TFP_norm, 'r--s', label='TFP', lw=2, markersize=10)
    axes[0].set_xlabel('External Finance to GDP', fontsize=14)
    axes[0].set_ylabel('Relative to Perfect Credit (λ=∞)', fontsize=14)
    axes[0].set_title('GDP and TFP vs Financial Development', fontsize=16)
    axes[0].legend(fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0.35, 1.1])
    
    # Right: Interest Rate
    axes[1].plot(plot_data['ext_fin'], plot_data['r'], 'g-^', lw=2, markersize=10)
    axes[1].set_xlabel('External Finance to GDP', fontsize=14)
    axes[1].set_ylabel('Interest Rate', fontsize=14)
    axes[1].set_title('Equilibrium Interest Rate', fontsize=16)
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color='k', linestyle='-', lw=0.5)
    
    plt.tight_layout()
    plt.savefig("plots/figure2_replication_v6.png")
    print(f"\n[SUCCESS] Figure 2 saved to plots/figure2_replication_v6.png")
