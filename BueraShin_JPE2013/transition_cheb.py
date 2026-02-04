"""
Buera & Shin (2010) Transition Dynamics - Version 2 (Spectral + Nested Solver)
==============================================================================

This script combines:
1. Spectral Chebyshev Methods (from v1/v6) for high-precision policy approx.
2. Nested Price Clearing Algorithm (from Appendix B.2) for GE and Transition.
3. Pareto distribution covering the full population (z_min=1.0).

Key Algorithm (B.2 Transition):
- Outer loop: Iterates over the path of interest rates {r_t}.
- Inner loop: Iterates over the path of wages {w_t}.
- Per-period bisections: Use fixed-point distribution logic to clear markets.
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

# Import from the working library (make sure it exists in path)
from library.functions_library import Chebyshev_Nodes, Chebyshev_Polynomials_Recursion_mv

warnings.filterwarnings('ignore')

# =============================================================================
# Model Parameters
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

# Global Grid Bounds (Spectral)
A_MIN, A_MAX = 1e-6, 200.0
A_SHIFT = 1.0
Z_MIN, Z_MAX = 1.0, (1 - 0.9995)**(-1/4.15)
N_CHEBY_A = 15
N_CHEBY_Z = 15
MAX_ITER_POLICY = 100  # Spectral policy solver max iterations

# =============================================================================
# Bivariate Spectral Tools
# =============================================================================

@njit(cache=False)
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

def plot_policy_2d(coeffs, iteration, name="post", outdir="plots", coeffs_alt=None):
    if not os.path.exists(outdir): os.makedirs(outdir, exist_ok=True)
    limit_a = 200.0; nodes_a, nodes_z, _ = generate_bivariate_nodes_matrix(A_MIN, A_MAX, Z_MIN, Z_MAX)
    a_dense = np.linspace(A_MIN, limit_a, 500); idx_max = len(nodes_z)-1; idx_med = len(nodes_z)//2
    z_max = nodes_z[idx_max]; z_med = nodes_z[idx_med]; fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot([0, limit_a], [0, limit_a], 'k--', alpha=0.3, label='45-degree')
    def plot_line(c, z, label, color, style, marker_color):
        pol_dense = np.array([bivariate_eval(aa, z, c, A_MIN, A_MAX, Z_MIN, Z_MAX) for aa in a_dense])
        ax.plot(a_dense, pol_dense, color=color, linestyle=style, linewidth=2, label=label)
        pol_nodes = np.array([bivariate_eval(na, z, c, A_MIN, A_MAX, Z_MIN, Z_MAX) for na in nodes_a])
        mask = nodes_a <= limit_a; ax.scatter(nodes_a[mask], pol_nodes[mask], color=marker_color, s=25, alpha=0.6, edgecolors='k', zorder=5)
    if coeffs_alt is not None:
        plot_line(coeffs, z_max, f"Max z: Distorted Type", '#D62728', '-', '#D62728')
        plot_line(coeffs_alt, z_max, f"Max z: Subsidized Type", '#2CA02C', '-', '#2CA02C')
        plot_line(coeffs, z_med, f"Med z: Distorted", '#1F77B4', '--', '#1F77B4')
    else:
        plot_line(coeffs, z_max, f"Max z ({z_max:.2f})", '#D62728', '-', '#D62728')
        plot_line(coeffs, z_med, f"Med z ({z_med:.2f})", '#1F77B4', '--', '#1F77B4')
    ax.set_title(f"Spectral Policy: {name} (Iter {iteration})"); ax.set_xlabel("a"); ax.set_ylabel("a'"); ax.legend(); ax.grid(True, alpha=0.2)
    save_path = os.path.join(outdir, f"policy_2d_{name}.png"); plt.savefig(save_path, dpi=120); plt.close()

def plot_grid_comparison(outdir="plots"):
    if not os.path.exists(outdir): os.makedirs(outdir, exist_ok=True)
    nodes_a, _, _ = generate_bivariate_nodes_matrix(A_MIN, A_MAX, Z_MIN, Z_MAX); na_h = 600
    a_h = np.exp(np.linspace(np.log(A_MIN+A_SHIFT), np.log(A_MAX+A_SHIFT), na_h)) - A_SHIFT
    zoom = min(100.0, A_MAX); plt.style.use('seaborn-v0_8-whitegrid'); fig, ax = plt.subplots(figsize=(12, 5))
    ax.vlines(a_h[a_h <= zoom], 0, 0.4, color='#457B9D', alpha=0.3, linewidth=0.5, label='Mass Grid')
    ax.vlines(nodes_a[nodes_a <= zoom], 0, 1.0, color='#E63946', alpha=0.9, linewidth=2.5, label='Chebyshev Nodes')
    ax.set_title("Multi-Grid Architecture", fontsize=15, fontweight='bold'); ax.set_yticks([]); ax.set_xlim(-1, zoom + 1); ax.legend()
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "asset_grid_comparison.png"), dpi=200); plt.close()

def plot_policy_comparison(pre_eq, post_eq, output_dir='plots'):
    if not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True)
    z_grid = pre_eq['z_grid']; limit_a = 200.0; a_dense = np.linspace(A_MIN, limit_a, 500)
    indices = [0, len(z_grid)//2, len(z_grid)-1]
    z_vals = [z_grid[i] for i in indices]; colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    labels = [f'Low z ({z_vals[0]:.2f})', f'Med z ({z_vals[1]:.2f})', f'High z ({z_vals[2]:.2f})']
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    ax = axes[0] # Pre-reform
    for i, idx in enumerate(indices):
        z_val = z_grid[idx]
        pol_p = np.array([bivariate_eval(aa, z_val, pre_eq['coeffs_plus'], A_MIN, A_MAX, Z_MIN, Z_MAX) for aa in a_dense])
        ax.plot(a_dense, pol_p, color=colors[i], label=labels[i])
    ax.plot(a_dense, a_dense, 'k--', alpha=0.3, label='45-degree')
    ax.set_title("Pre-Reform (Distorted $\\tau^+$ Type)"); ax.set_xlabel("a"); ax.set_ylabel("a'"); ax.set_xlim(0, limit_a); ax.set_ylim(0, limit_a); ax.legend(frameon=False); ax.grid(True, alpha=0.2)
    ax = axes[1] # Post-reform
    for i, idx in enumerate(indices):
        z_val = z_grid[idx]
        pol = np.array([bivariate_eval(aa, z_val, post_eq['coeffs'], A_MIN, A_MAX, Z_MIN, Z_MAX) for aa in a_dense])
        ax.plot(a_dense, pol, color=colors[i], label=labels[i])
    ax.plot(a_dense, a_dense, 'k--', alpha=0.3)
    ax.set_title("Post-Reform (No Distortion)"); ax.set_xlabel("a"); ax.set_xlim(0, limit_a); ax.legend(frameon=False); ax.grid(True, alpha=0.2)
    plt.tight_layout(); save_path = os.path.join(output_dir, 'policy_comparison_v2.png')
    plt.savefig(save_path, dpi=150); plt.close(); print(f"Policy plot saved to {save_path}")

def plot_diagnostics(diagnostics, outdir="plots"):
    """Plot detailed numerical convergence diagnostics."""
    if not os.path.exists(outdir): os.makedirs(outdir, exist_ok=True)
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    
    # 1. Policy Residuals (History across all solver calls)
    ax = axes[0, 0]
    for key, vals in diagnostics['policy_errs'].items():
        if len(vals) > 0: ax.semilogy(vals, label=f"Pol: {key}", alpha=0.7)
    ax.set_title("Policy Solver Convergence (Steps)"); ax.set_xlabel("Iteration Step"); ax.set_ylabel("max|diff|")
    ax.legend(fontsize=7, ncol=2); ax.grid(True, which='both', alpha=0.3)
    
    # 2. Stationary Distribution Convergence
    ax = axes[0, 1]
    for key, vals in diagnostics['dist_errs'].items():
        if len(vals) > 0: ax.semilogy(vals, label=key)
    ax.set_title("Stationary Distribution Convergence"); ax.set_xlabel("Power Iteration Step"); ax.set_ylabel("max|mu_new - mu|")
    ax.legend(fontsize=8, ncol=2); ax.grid(True, which='both', alpha=0.3)
    
    # 3. GE Labor Market (Steady State)
    ax = axes[1, 0]
    for key, vals in diagnostics['edL_history'].items():
        if len(vals) > 0: ax.semilogy(np.abs(vals), label=key)
    ax.set_title("GE Labor Market Clearing (SS)"); ax.set_xlabel("Wage Bisection Step"); ax.set_ylabel("|ED_L|")
    ax.legend(fontsize=8, ncol=2); ax.grid(True, which='both', alpha=0.3)
    
    # 4. GE Capital Market (Steady State)
    ax = axes[1, 1]
    for key, vals in diagnostics['edK_history'].items():
        if len(vals) > 0: ax.semilogy(np.abs(vals), marker='o', label=key)
    ax.set_title("GE Capital Market Clearing (SS)"); ax.set_xlabel("Interest Rate Bisection Step"); ax.set_ylabel("|ED_K|")
    ax.legend(fontsize=8); ax.grid(True, which='both', alpha=0.3)
    
    # 5. TPI Sequence Convergence
    ax = axes[2, 0]
    if len(diagnostics['tpi_errs_L']) > 0: ax.semilogy(diagnostics['tpi_errs_L'], label='max |ED_L| (path)')
    if len(diagnostics['tpi_errs_K']) > 0: ax.semilogy(diagnostics['tpi_errs_K'], label='max |ED_K| (path)')
    ax.set_title("TPI Market Clearing Convergence"); ax.set_xlabel("Outer iteration"); ax.set_ylabel("Max Resid")
    ax.legend(); ax.grid(True, which='both', alpha=0.3)

    # 6. Price Paths (Guess/Final comparison if possible, or just blank)
    ax = axes[2, 1]
    ax.text(0.5, 0.5, "Diagnostic Panel\nFully Detailed", ha='center', va='center', fontsize=14)
    ax.set_axis_off()

    plt.tight_layout(); save_path = os.path.join(outdir, "convergence_diagnostics.png")
    plt.savefig(save_path, dpi=150); plt.close()
    print(f"Numerical diagnostics saved to {save_path}")

def plot_current_policy_status(coeffs, label, outdir="plots", filename="current_policy_snapshot.png"):
    """Plot a snapshot of the current policy function with a low-asset zoom panel (Overwrites)."""
    if not os.path.exists(outdir): os.makedirs(outdir, exist_ok=True)
    
    nodes_a, nodes_z_full, _ = generate_bivariate_nodes_matrix(A_MIN, A_MAX, Z_MIN, Z_MAX)
    z_high = nodes_z_full[-1]; z_med = nodes_z_full[len(nodes_z_full)//2]
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    
    # limits for global and zoom
    limits = [50.0, 5.0]
    titles = ["Global Status (0-50)", "Zoom low-a (0-5)"]
    
    for ax, lim, title in zip(axes, limits, titles):
        a_dense = np.linspace(A_MIN, lim, 200)
        ax.plot([0, lim], [0, lim], 'k--', alpha=0.3, label='45-degree')
        
        # Policy curves
        p_high = np.array([bivariate_eval(aa, z_high, coeffs, A_MIN, A_MAX, Z_MIN, Z_MAX) for aa in a_dense])
        p_med = np.array([bivariate_eval(aa, z_med, coeffs, A_MIN, A_MAX, Z_MIN, Z_MAX) for aa in a_dense])
        ax.plot(a_dense, p_high, 'r-', label=f'High z ({z_high:.2f})')
        ax.plot(a_dense, p_med, 'b-', label=f'Med z ({z_med:.2f})')
        
        # Scatter nodes
        nodes_a_plot = nodes_a[nodes_a <= lim]
        if len(nodes_a_plot) > 0:
            dots_high = np.array([bivariate_eval(na, z_high, coeffs, A_MIN, A_MAX, Z_MIN, Z_MAX) for na in nodes_a_plot])
            dots_med = np.array([bivariate_eval(na, z_med, coeffs, A_MIN, A_MAX, Z_MIN, Z_MAX) for na in nodes_a_plot])
            ax.scatter(nodes_a_plot, dots_high, s=100, color='r', edgecolors='k', zorder=5, label='Nodes (High z)')
            ax.scatter(nodes_a_plot, dots_med, s=100, color='b', edgecolors='k', zorder=5, label='Nodes (Med z)')
        
        ax.set_title(f"{title}: {label}"); ax.set_xlabel("a"); ax.set_ylabel("a'"); ax.grid(True, alpha=0.2); ax.legend(fontsize=9)
        ax.set_xlim(-0.1, lim); ax.set_ylim(-0.1, lim)

    plt.tight_layout(); save_path = os.path.join(outdir, filename)
    plt.savefig(save_path, dpi=120); plt.close()

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
# Entrepreneur Logic
# =============================================================================

@njit(cache=False)
def solve_entrepreneur_single(a, z, w, r, lam, delta, alpha, nu):
    rental = max(r + delta, 1e-8)
    wage = max(w, 1e-8)
    span = 1 - nu
    exp_k = 1 - (1-alpha)*span
    exp_l = (1-alpha)*span
    k1 = (((alpha*span*z/rental)**exp_k) * (((1-alpha)*span*z/wage)**exp_l)) ** (1/nu)
    kstar = min(k1, lam * a)
    denom = max((z * (kstar**(alpha*span)) * (1-alpha) * span), 1e-12)
    lstar = ((wage / denom)) ** (1 / ((1-alpha)*span - 1))
    output = z * ((kstar**alpha) * (lstar**(1-alpha))) ** span
    profit = output - wage * lstar - rental * kstar
    return profit, kstar, lstar, output

@njit(cache=False)
def solve_entrepreneur_with_tau(a, z, tau, w, r, lam, delta, alpha, nu):
    z_eff = (1.0 - tau) * z
    if z_eff <= 0: return -1e10, 0.0, 0.0, 0.0
    rental = max(r + delta, 1e-8)
    wage = max(w, 1e-8)
    span = 1 - nu
    exp_k = 1 - (1-alpha)*span
    exp_l = (1-alpha)*span
    k1 = (((alpha*span*z_eff/rental)**exp_k) * (((1-alpha)*span*z_eff/wage)**exp_l)) ** (1/nu)
    kstar = min(k1, lam * a)
    denom = max((z_eff * (kstar**(alpha*span)) * (1-alpha) * span), 1e-12)
    lstar = ((wage / denom)) ** (1 / ((1-alpha)*span - 1))
    output = z_eff * ((kstar**alpha) * (lstar**(1-alpha))) ** span
    profit = output - wage * lstar - rental * kstar
    return profit, kstar, lstar, output

# =============================================================================
# Policy Solvers
# =============================================================================

@njit(cache=False, parallel=True)
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

@njit(cache=False, parallel=True)
def solve_policy_bivariate_update_with_dist(coeffs_plus, coeffs_minus,
                                            nodes_a, nodes_z, T_az_inv,
                                            beta, sigma, psi, w, r, lam, delta, alpha, nu,
                                            a_min, a_max, z_min, z_max,
                                            quad_z, quad_w, prob_tau_plus_arr,
                                            tau_plus, tau_minus, is_plus):
    tau_curr = tau_plus if is_plus else tau_minus
    coeffs_curr = coeffs_plus if is_plus else coeffs_minus
    nz_c = N_CHEBY_Z
    nt = len(nodes_a) * nz_c
    target_vals = np.zeros(nt)
    for i_n in prange(nt):
        ia, iz = i_n // nz_c, i_n % nz_c
        a, z = nodes_a[ia], nodes_z[iz]
        p, _, _, _ = solve_entrepreneur_with_tau(a, z, tau_curr, w, r, lam, delta, alpha, nu)
        inc = max(p, w) + (1 + r) * a
        aprime = bivariate_eval(a, z, coeffs_curr, a_min, a_max, z_min, z_max)
        aprime = max(a_min, min(aprime, inc - 1e-6))
        p_p, _, _, _ = solve_entrepreneur_with_tau(aprime, z, tau_curr, w, r, lam, delta, alpha, nu)
        inc_p = max(p_p, w) + (1 + r) * aprime
        app = bivariate_eval(aprime, z, coeffs_curr, a_min, a_max, z_min, z_max)
        mu_pers = max(inc_p - app, 1e-9) ** (-sigma)
        e_mu_redraw = 0.0
        for k in range(len(quad_z)):
            zk, ptp_k = quad_z[k], prob_tau_plus_arr[k]
            # tau_plus mu
            p_plus, _, _, _ = solve_entrepreneur_with_tau(aprime, zk, tau_plus, w, r, lam, delta, alpha, nu)
            inc_plus = max(p_plus, w) + (1 + r) * aprime
            a_plus = bivariate_eval(aprime, zk, coeffs_plus, a_min, a_max, z_min, z_max)
            mu_plus = max(inc_plus - a_plus, 1e-9) ** (-sigma)
            # tau_minus mu
            p_minus, _, _, _ = solve_entrepreneur_with_tau(aprime, zk, tau_minus, w, r, lam, delta, alpha, nu)
            inc_minus = max(p_minus, w) + (1 + r) * aprime
            a_minus = bivariate_eval(aprime, zk, coeffs_minus, a_min, a_max, z_min, z_max)
            mu_minus = max(inc_minus - a_minus, 1e-9) ** (-sigma)
            e_mu_redraw += quad_w[k] * (ptp_k * mu_plus + (1 - ptp_k) * mu_minus)
        expected_mu = psi * mu_pers + (1 - psi) * e_mu_redraw
        c_target = (beta * (1 + r) * expected_mu) ** (-1.0/sigma)
        target_vals[i_n] = max(a_min, min(inc - c_target, a_max))
    return T_az_inv @ target_vals

# =============================================================================
# Distribution Update
# =============================================================================

@njit(cache=False)
def get_interpolation_weights(x, grid):
    n = len(grid)
    if x <= grid[0]: return 0, 1.0, 0.0
    if x >= grid[-1]: return n-2, 0.0, 1.0
    idx = 0
    while idx < n-2 and grid[idx+1] < x: idx += 1
    w2 = (x - grid[idx]) / (grid[idx+1] - grid[idx])
    return idx, 1.0 - w2, w2

@njit(cache=False)
def update_dist_matrix_nodist(A_prime_mat, dist_t, na, nz, prob_z, psi, a_grid):
    dist_next = np.zeros((na, nz))
    for ia in range(na):
        for iz in range(nz):
            mass = dist_t[ia, iz]
            if mass < 1e-14: continue
            aprime = A_prime_mat[ia, iz]
            ia_low, w1, w2 = get_interpolation_weights(aprime, a_grid)
            for izp in range(nz):
                p_z = (psi + (1-psi)*prob_z[izp]) if izp == iz else (1-psi)*prob_z[izp]
                if w1 > 1e-9: dist_next[ia_low, izp] += mass * p_z * w1
                if w2 > 1e-9: dist_next[ia_low+1, izp] += mass * p_z * w2
    return dist_next / dist_next.sum()

# =============================================================================
# Nested Price Clearing Utils (Appendix B.2)
# =============================================================================

@njit(cache=False)
def labor_excess_mu_nodist(w, r, mu, a_h, z_h, params):
    delta, alpha, nu, lam = params[0], params[1], params[2], params[3]
    na, nz = len(a_h), len(z_h)
    L_d = 0.0; En_share = 0.0
    for ia in range(na):
        for iz in range(nz):
            wgt = mu[ia, iz]
            p, _, ld, _ = solve_entrepreneur_single(a_h[ia], z_h[iz], w, r, lam, delta, alpha, nu)
            if p > w:
                L_d += ld * wgt
                En_share += wgt
    return L_d - (1.0 - En_share)

@njit(cache=False)
def capital_excess_mu_nodist(r, w, mu, a_h, z_h, params):
    delta, alpha, nu, lam = params[0], params[1], params[2], params[3]
    na, nz = len(a_h), len(z_h)
    K_d, A_s = 0.0, 0.0
    for ia in range(na):
        for iz in range(nz):
            wgt = mu[ia, iz]
            A_s += a_h[ia] * wgt
            p, ks, _, _ = solve_entrepreneur_single(a_h[ia], z_h[iz], w, r, lam, delta, alpha, nu)
            if p > w: K_d += ks * wgt
    return K_d - A_s

@njit(cache=False)
def labor_excess_with_dist(w, r, mu_p, mu_m, a_h, z_h, params, tp, tm):
    delta, alpha, nu, lam = params[0], params[1], params[2], params[3]
    na, nz = len(a_h), len(z_h)
    L_d, En_share = 0.0, 0.0
    # Plus state
    for ia in range(na):
        for iz in range(nz):
            wgt = mu_p[ia, iz]
            pp, _, ld, _ = solve_entrepreneur_with_tau(a_h[ia], z_h[iz], tp, w, r, lam, delta, alpha, nu)
            if pp > w: 
                L_d += ld*wgt; En_share += wgt
    # Minus state
    for ia in range(na):
        for iz in range(nz):
            wgt = mu_m[ia, iz]
            pm, _, ld, _ = solve_entrepreneur_with_tau(a_h[ia], z_h[iz], tm, w, r, lam, delta, alpha, nu)
            if pm > w: 
                L_d += ld*wgt; En_share += wgt
    return L_d - (1.0 - En_share)

@njit(cache=False)
def capital_excess_with_dist(r, w, mu_p, mu_m, a_h, z_h, params, tp, tm):
    delta, alpha, nu, lam = params[0], params[1], params[2], params[3]
    na, nz = len(a_h), len(z_h)
    K_d, A_s = 0.0, 0.0
    for ia in range(na):
        for iz in range(nz):
            A_s += a_h[ia] * (mu_p[ia, iz] + mu_m[ia, iz])
            pp, ks_p, _, _ = solve_entrepreneur_with_tau(a_h[ia], z_h[iz], tp, w, r, lam, delta, alpha, nu)
            if pp > w: K_d += ks_p * mu_p[ia, iz]
            pm, ks_m, _, _ = solve_entrepreneur_with_tau(a_h[ia], z_h[iz], tm, w, r, lam, delta, alpha, nu)
            if pm > w: K_d += ks_m * mu_m[ia, iz]
    return K_d - A_s

# Bisections
def solve_w_clear_mu(r, mu, a_h, z_h, params, dists=False, mu_m=None):
    w_min, w_max = 0.01, 2.5
    for _ in range(30):
        w = (w_min + w_max) / 2
        if dists:
            ed = labor_excess_with_dist(w, r, mu, mu_m, a_h, z_h, params, TAU_PLUS, TAU_MINUS)
        else:
            ed = labor_excess_mu_nodist(w, r, mu, a_h, z_h, params)
        if ed > 0: w_min = w
        else: w_max = w
        if abs(ed) < 1e-6: break
    return w

# =============================================================================
# Steady State Solver (Nested w in r)
# =============================================================================

@njit(cache=False)
def get_aggregates_analytical(dist, a_h, z_h, w, r, lam, delta, alpha, nu):
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

@njit(cache=False)
def get_aggregates_with_dist(mu_p, mu_m, a_h, z_h, w, r, lam, d, al, nu, tp, tm):
    na, nz = len(a_h), len(z_h)
    K, L, Y, En, A, Ext = 0., 0., 0., 0., 0., 0.
    for ia in range(na):
        a = a_h[ia]
        for iz in range(nz):
            z = z_h[iz]
            # tp
            wp = mu_p[ia, iz]; A += a * wp
            p_p, k_p, l_p, o_p = solve_entrepreneur_with_tau(a, z, tp, w, r, lam, d, al, nu)
            if p_p > w:
                K += k_p*wp; L += l_p*wp; Y += o_p*wp; En += wp; Ext += max(0., k_p - a)*wp
            # tm
            wm = mu_m[ia, iz]; A += a * wm
            p_m, k_m, l_m, o_m = solve_entrepreneur_with_tau(a, z, tm, w, r, lam, d, al, nu)
            if p_m > w:
                K += k_m*wm; L += l_m*wm; Y += o_m*wm; En += wm; Ext += max(0., k_m - a)*wm
    return K, L, Y, En, A, Ext

def compute_stationary_analytical(coeffs, a_grid, z_grid, prob_z, psi, mu_init=None, max_iter=200, diag_out=None, key="post"):
    na, nz = len(a_grid), len(z_grid); n_states = na * nz
    A_prime_mat = bivariate_eval_matrix(a_grid, z_grid, coeffs, A_MIN, A_MAX, Z_MIN, Z_MAX)
    rows, cols, data = [], [], []
    for ia in range(na):
        for iz in range(nz):
            s = ia * nz + iz; aprime = A_prime_mat[ia, iz]
            ia_low, w1, w2 = get_interpolation_weights(aprime, a_grid)
            for izp in range(nz):
                pz = (psi + (1-psi)*prob_z[izp]) if izp == iz else (1-psi)*prob_z[izp]
                if w1 > 1e-9: rows.append(ia_low * nz + izp); cols.append(s); data.append(pz * w1)
                if w2 > 1e-9: rows.append((ia_low+1) * nz + izp); cols.append(s); data.append(pz * w2)
    Q = sparse.csr_matrix((data, (rows, cols)), shape=(n_states, n_states))
    if mu_init is None: mu = np.ones(n_states) / n_states
    else: mu = mu_init.ravel()
    for i in range(max_iter):
        m_new = Q @ mu; m_new /= m_new.sum()
        diff = np.max(np.abs(m_new - mu))
        if diag_out is not None: diag_out['dist_errs'][key].append(diff)
        if i % 10 == 0: print(f"    [Dist analytical {key}] it={i}, diff={diff:.2e}")
        if diff < 1e-10: break
        mu = m_new
    return mu.reshape((na, nz))

def compute_stationary_with_dist(cp, cm, a_grid, z_grid, pr_z, prob_tp, psi, mu_init=None, max_iter=200, diag_out=None, key="pre"):
    na, nz = len(a_grid), len(z_grid); n_states = na * nz * 2
    Ap, Am = bivariate_eval_matrix(a_grid, z_grid, cp, A_MIN, A_MAX, Z_MIN, Z_MAX), bivariate_eval_matrix(a_grid, z_grid, cm, A_MIN, A_MAX, Z_MIN, Z_MAX)
    rows, cols, data = [], [], []
    for ia in range(na):
        for iz in range(nz):
            for it in range(2): # 0=p, 1=m
                s = ia*nz*2 + iz*2 + it; apr = Ap[ia, iz] if it == 0 else Am[ia, iz]
                ial, w1, w2 = get_interpolation_weights(apr, a_grid)
                if w1 > 1e-9: rows.append(ial*nz*2 + iz*2 + it); cols.append(s); data.append(psi*w1)
                if w2 > 1e-9: rows.append((ial+1)*nz*2 + iz*2 + it); cols.append(s); data.append(psi*w2)
                for izp in range(nz):
                    ptp = prob_tp[izp]
                    if w1 > 1e-9:
                        rows.append(ial*nz*2+izp*2+0); cols.append(s); data.append((1-psi)*pr_z[izp]*ptp*w1)
                        rows.append(ial*nz*2+izp*2+1); cols.append(s); data.append((1-psi)*pr_z[izp]*(1-ptp)*w1)
                    if w2 > 1e-9:
                        rows.append((ial+1)*nz*2+izp*2+0); cols.append(s); data.append((1-psi)*pr_z[izp]*ptp*w2)
                        rows.append((ial+1)*nz*2+izp*2+1); cols.append(s); data.append((1-psi)*pr_z[izp]*(1-ptp)*w2)
    Q = sparse.csr_matrix((data, (rows, cols)), shape=(n_states, n_states))
    if mu_init is None: mu = np.ones(n_states) / n_states
    else: mu = mu_init.ravel()
    for i in range(max_iter):
        m_new = Q @ mu; m_new /= m_new.sum()
        diff = np.max(np.abs(m_new - mu))
        if diag_out is not None: diag_out['dist_errs'][key].append(diff)
        if i % 10 == 0: print(f"    [Dist withDist {key}] it={i}, diff={diff:.2e}")
        if diff < 1e-10: break
        mu = m_new
    mu_f = mu.reshape((na, nz, 2))
    return mu_f[:,:,0], mu_f[:,:,1]

def find_equilibrium_nested(params, distortions=False, diag_out=None, label="post"):
    delta, alpha, nu, lam, beta, sigma, psi = params
    na_h = 600; nz_h = 40
    a_h = np.exp(np.linspace(np.log(A_MIN+A_SHIFT), np.log(A_MAX+A_SHIFT), na_h)) - A_SHIFT
    M_v = np.concatenate([np.linspace(0.0, 0.998, 38), [0.999, 0.9995]])
    z_h = (1 - M_v)**(-1/ETA)
    pr_z = np.zeros(nz_h); pr_z[0] = M_v[0]; pr_z[1:] = M_v[1:] - M_v[:-1]; pr_z /= pr_z.sum()
    prob_tp = 1 - np.exp(-Q_DIST * z_h)
    nodes_a, nodes_z, T_full = generate_bivariate_nodes_matrix(A_MIN, A_MAX, Z_MIN, Z_MAX)
    T_inv = np.linalg.inv(T_full)
    z_quad = z_h.copy(); z_w = pr_z.copy()
    r_min, r_max = -0.15, 0.08
    coeffs_curr = None; dist_curr = None
    
    for out_iter in range(12):
        r = (r_min + r_max) / 2
        print(f"\n[Outer SS {label}] Iter {out_iter}: r={r:.6f}, bounds=[{r_min:.4f}, {r_max:.4f}]")
        w_low, w_high = 0.01, 3.5
        for inn_iter in range(25):
            w = (w_low + w_high) / 2
            if not distortions:
                coeffs = solve_policy_spectral_nested(params, w, r, nodes_a, nodes_z, T_inv, z_quad, z_w, coeffs_curr, diag_out, f"{label}_r{out_iter}_w{inn_iter}")
                dist = compute_stationary_analytical(coeffs, a_h, z_h, pr_z, psi, mu_init=dist_curr, max_iter=150, diag_out=diag_out, key=f"{label}_r{out_iter}_w{inn_iter}")
                ed_L = labor_excess_mu_nodist(w, r, dist, a_h, z_h, params)
                coeffs_curr, dist_curr = coeffs, dist
            else:
                cp, cm = solve_policy_spectral_with_dist_nested(params, w, r, nodes_a, nodes_z, T_inv, z_quad, z_w, prob_tp, coeffs_curr, diag_out, f"{label}_r{out_iter}_w{inn_iter}")
                mu_p, mu_m = compute_stationary_with_dist(cp, cm, a_h, z_h, pr_z, prob_tp, psi, mu_init=dist_curr, max_iter=150, diag_out=diag_out, key=f"{label}_r{out_iter}_w{inn_iter}")
                ed_L = labor_excess_with_dist(w, r, mu_p, mu_m, a_h, z_h, params, TAU_PLUS, TAU_MINUS)
                coeffs_curr, dist_curr = (cp, cm), np.concatenate([mu_p.ravel(), mu_m.ravel()])
            
            if diag_out is not None:
                diag_out['edL_history'][f"{label}_r{out_iter}"].append(ed_L)
            
            if inn_iter % 3 == 0 or abs(ed_L) < 1e-4:
                print(f"    [Wage Bisection] it={inn_iter}, w={w:.4f}, edL={ed_L:.3e}, bracket=[{w_low:.3f}, {w_high:.3f}]")

            if ed_L > 0: w_low = w
            else: w_high = w
            if abs(ed_L) < 1e-5: break
        
        if not distortions:
            dist = compute_stationary_analytical(coeffs, a_h, z_h, pr_z, psi, mu_init=dist_curr, max_iter=250)
            ed_K = capital_excess_mu_nodist(r, w, dist, a_h, z_h, params)
            ed_L = labor_excess_mu_nodist(w, r, dist, a_h, z_h, params)
        else:
            mu_p, mu_m = compute_stationary_with_dist(cp, cm, a_h, z_h, pr_z, prob_tp, psi, mu_init=dist_curr, max_iter=250)
            ed_K = capital_excess_with_dist(r, w, mu_p, mu_m, a_h, z_h, params, TAU_PLUS, TAU_MINUS)
            ed_L = labor_excess_with_dist(w, r, mu_p, mu_m, a_h, z_h, params, TAU_PLUS, TAU_MINUS)
            
        if diag_out is not None:
            diag_out['edK_history'][label].append(ed_K)
        
        # Plot final policy for this r iteration
        if not distortions: plot_current_policy_status(coeffs, f"{label} r{out_iter} FINAL", filename="policy_post_final.png")
        else: plot_current_policy_status(cp, f"{label} r{out_iter} plus FINAL", filename="policy_pre_final.png")
            
        print(f"  [RESULT r={r:.4f}] final_w={w:.4f}, edL={ed_L:.4e}, edK={ed_K:.4e}")
        
        # CRITICAL: Only allow outer convergence if inner market actually cleared
        if abs(ed_K) < 2e-3 and abs(ed_L) < 5e-4: 
            print("  [GE SUCCESS] Both markets cleared!")
            break
        
        if ed_K > 0: r_min = r
        else: r_max = r

        if out_iter == 11:
            print("  [GE WARNING] Outer loop finished without full clearing.")
    if not distortions:
        K, L, Y, En, A, Ext = get_aggregates_analytical(dist, a_h, z_h, w, r, lam, delta, alpha, nu)
    else:
        K, L, Y, En, A, Ext = get_aggregates_with_dist(mu_p, mu_m, a_h, z_h, w, r, lam, delta, alpha, nu, TAU_PLUS, TAU_MINUS)
    res = {'w': w, 'r': r, 'Y': Y, 'K': K, 'A': A, 'TFP': Y/max(((K**alpha)*((1-En)**(1-alpha)))**(1-nu), 1e-8),
           'a_grid': a_h, 'z_grid': z_h, 'prob_z': pr_z, 'share_entre': En, 'ExtFin_Y': Ext/max(Y, 1e-8)}
    if distortions:
        res['coeffs_plus'], res['coeffs_minus'], res['mu_plus'], res['mu_minus'] = cp, cm, mu_p, mu_m
    else:
        res['coeffs'] = coeffs
    return res

def solve_policy_spectral_nested(params, w, r, nodes_a, nodes_z, T_inv, quad_z, quad_w, coeffs_init, diag_out=None, key="post", show_plot=False):
    delta, alpha, nu, lam, beta, sigma, psi = params
    # Initialize with a small 'seed' of savings if no warm start to avoid zero-asset trap
    if coeffs_init is not None:
        coeffs = coeffs_init
    else:
        # A simple linear rule: save 10% of max assets as a starting guess
        seed_vals = 0.1 * nodes_a.repeat(N_CHEBY_Z) + 0.05 * A_MAX
        coeffs = T_inv @ seed_vals

    for i in range(MAX_ITER_POLICY):
        c_new = solve_policy_bivariate_update(coeffs, nodes_a, nodes_z, T_inv, beta, sigma, psi, w, r, lam, delta, alpha, nu, A_MIN, A_MAX, Z_MIN, Z_MAX, quad_z, quad_w)
        diff = np.max(np.abs(c_new - coeffs))
        if diag_out is not None: 
            diag_out['policy_errs'][key].append(diff)
        if i % 5 == 0: 
            print(f"      [Policy spectral] it={i}, diff={diff:.2e}")
        if diff < 1e-7: break
        coeffs = 0.5 * coeffs + 0.5 * c_new
    if show_plot: plot_current_policy_status(coeffs, f"{key} final")
    return coeffs

def solve_policy_spectral_with_dist_nested(params, w, r, nodes_a, nodes_z, T_inv, quad_z, quad_w, prob_tp, coeffs_in, diag_out=None, key="pre", show_plot=False):
    delta, alpha, nu, lam, beta, sigma, psi = params
    if coeffs_in is not None:
        cp, cm = coeffs_in
    else:
        cp = np.zeros(N_CHEBY_A * N_CHEBY_Z); cm = np.zeros(N_CHEBY_A * N_CHEBY_Z)
    for i in range(MAX_ITER_POLICY):
        cnp = solve_policy_bivariate_update_with_dist(cp, cm, nodes_a, nodes_z, T_inv, beta, sigma, psi, w, r, lam, delta, alpha, nu, A_MIN, A_MAX, Z_MIN, Z_MAX, quad_z, quad_w, prob_tp, TAU_PLUS, TAU_MINUS, True)
        cnm = solve_policy_bivariate_update_with_dist(cp, cm, nodes_a, nodes_z, T_inv, beta, sigma, psi, w, r, lam, delta, alpha, nu, A_MIN, A_MAX, Z_MIN, Z_MAX, quad_z, quad_w, prob_tp, TAU_PLUS, TAU_MINUS, False)
        diff = max(np.max(np.abs(cnp - cp)), np.max(np.abs(cnm - cm)))
        if diag_out is not None: 
            diag_out['policy_errs'][key].append(diff)
        if i % 5 == 0: 
            print(f"      [Policy spectral pre] it={i}, diff={diff:.2e}")
        if diff < 1e-7: break
        cp, cm = 0.5 * cp + 0.5 * cnp, 0.5 * cm + 0.5 * cnm
    if show_plot: plot_current_policy_status(cp, f"{key} plus final")
    return cp, cm

# =============================================================================
# Transition Path (Algorithm B.2 logic)
# =============================================================================

def solve_transition_v2(pre_eq, post_eq, params, T=250, diag_out=None):
    delta, alpha, nu, lam, beta, sigma, psi = params
    a_h = post_eq['a_grid']; z_h = post_eq['z_grid']; pr_z = post_eq['prob_z']
    
    # 1. Initial Guess for r and w paths
    t_idx = np.arange(T)
    r_path = np.linspace(pre_eq['r'], post_eq['r'], T)
    w_path = np.linspace(pre_eq['w'], post_eq['w'], T)
    
    mu_0 = pre_eq['mu_plus'] + pre_eq['mu_minus']
    
    for outer_tpi in range(15):
        print(f"\n[Outer TPI] iteration {outer_tpi+1}")
        
        # Inner loop: Wages w_t for fixed r_t
        for inner_tpi in range(10):
            # A. Backward Induction Policies
            pols = [None] * T; pols[T-1] = post_eq['coeffs']
            for t in range(T-2, -1, -1):
                pols[t] = solve_policy_transition_at_t_simple(pols[t+1], w_path[t], r_path[t], w_path[t+1], r_path[t+1], params)
            
            # B. Forward Distribution Update and Labor Market Clearing
            mu_t = mu_0.copy()
            w_new = np.zeros(T); max_edl = 0.0
            for t in range(T):
                edl = labor_excess_mu_nodist(w_path[t], r_path[t], mu_t, a_h, z_h, params)
                max_edl = max(max_edl, abs(edl))
                w_new[t] = w_path[t] * (1 + 0.2 * edl)
                if t < T - 1:
                    Ap = bivariate_eval_matrix(a_h, z_h, pols[t], A_MIN, A_MAX, Z_MIN, Z_MAX)
                    mu_t = update_dist_matrix_nodist(Ap, mu_t, len(a_h), len(z_h), pr_z, PSI, a_h)
            
            w_path = 0.5 * w_path + 0.5 * w_new
            if max_edl < 1e-3: break
            
        # C. Capital Market Clearing for r_t
        mu_t = mu_0.copy(); r_new = np.zeros(T); max_edk = 0.0
        for t in range(T):
            edk = capital_excess_mu_nodist(r_path[t], w_path[t], mu_t, a_h, z_h, params)
            max_edk = max(max_edk, abs(edk))
            r_new[t] = r_path[t] + 0.01 * edk
            if t < T - 1:
                Ap = bivariate_eval_matrix(a_h, z_h, pols[t], A_MIN, A_MAX, Z_MIN, Z_MAX)
                mu_t = update_dist_matrix_nodist(Ap, mu_t, len(a_h), len(z_h), pr_z, PSI, a_h)
        
        r_path = 0.5 * r_path + 0.5 * r_new
        if diag_out is not None:
            diag_out['tpi_errs_L'].append(max_edl)
            diag_out['tpi_errs_K'].append(max_edk)
        print(f"  Max ED_L={max_edl:.4f}, Max ED_K={max_edk:.4f}")
        if max_edk < 3e-3: break

    # Final Pass for storage
    res_path = {'t': t_idx, 'w': w_path, 'r': r_path, 'Y': np.zeros(T), 'K': np.zeros(T), 'L': np.zeros(T), 'A': np.zeros(T)}
    mu_t = mu_0.copy()
    for t in range(T):
        K, L, Y, En, A, Ext = get_aggregates_analytical(mu_t, a_h, z_h, w_path[t], r_path[t], lam, delta, alpha, nu)
        res_path['Y'][t], res_path['K'][t], res_path['A'][t] = Y, K, A
        if t < T - 1:
            Ap = bivariate_eval_matrix(a_h, z_h, pols[t], A_MIN, A_MAX, Z_MIN, Z_MAX)
            mu_t = update_dist_matrix_nodist(Ap, mu_t, len(a_h), len(z_h), pr_z, PSI, a_h)
    return res_path

@njit(cache=False, parallel=True)
def solve_policy_transition_update_v2(c_curr, c_next, n_a, n_z, T_inv, beta, sig, psi, wt, rt, wt1, rt1, lam, d, a, nu, amin, amax, zmin, zmax, qz, qw):
    nz_c = N_CHEBY_Z; nt = len(n_a) * nz_c; target = np.zeros(nt)
    for i in prange(nt):
        ia, iz = i // nz_c, i % nz_c; cur_a, cur_z = n_a[ia], n_z[iz]
        p, _, _, _ = solve_entrepreneur_single(cur_a, cur_z, wt, rt, lam, d, a, nu)
        inc = max(p, wt) + (1 + rt) * cur_a
        ap = bivariate_eval(cur_a, cur_z, c_curr, amin, amax, zmin, zmax)
        ap = max(amin, min(ap, inc - 1e-6))
        # Mu at t+1
        p1, _, _, _ = solve_entrepreneur_single(ap, cur_z, wt1, rt1, lam, d, a, nu)
        inc1 = max(p1, wt1) + (1 + rt1) * ap
        app = bivariate_eval(ap, cur_z, c_next, amin, amax, zmin, zmax)
        mu_p = max(inc1 - app, 1e-9)**(-sig)
        e_mu = 0.0
        for k in range(len(qz)):
            pk, _, _, _ = solve_entrepreneur_single(ap, qz[k], wt1, rt1, lam, d, a, nu)
            ick = max(pk, wt1) + (1+rt1)*ap
            akk = bivariate_eval(ap, qz[k], c_next, amin, amax, zmin, zmax)
            e_mu += qw[k] * max(ick - akk, 1e-9)**(-sig)
        exp_mu = psi * mu_p + (1-psi)*e_mu
        c_tar = (beta*(1+rt)*exp_mu)**(-1.0/sig)
        target[i] = max(amin, min(inc - c_tar, amax))
    return T_inv @ target

def solve_policy_transition_at_t_simple(c_next, wt, rt, wt1, rt1, params):
    d, a, nu, lam, beta, sig, psi = params
    n_a, n_z, T_inv = generate_bivariate_nodes_matrix(A_MIN, A_MAX, Z_MIN, Z_MAX)[0:3]; T_inv = np.linalg.inv(T_inv)
    qz = n_z; qw = np.ones(len(n_z))/len(n_z) # Simple integration for transition step
    c = c_next.copy()
    for _ in range(20):
        c_new = solve_policy_transition_update_v2(c, c_next, n_a, n_z, T_inv, beta, sig, psi, wt, rt, wt1, rt1, lam, d, a, nu, A_MIN, A_MAX, Z_MIN, Z_MAX, qz, qw)
        if np.max(np.abs(c_new - c)) < 1e-6: break
        c = 0.5 * c + 0.5 * c_new
    return c

# =============================================================================
# Main
# =============================================================================

def main():
    print("\n" + "="*50)
    print("BUERA & SHIN V2: SPECTRAL NESTED SOLVER")
    print("="*50)
    
    # Grid Plot
    plot_grid_comparison()
    print("Diagnostic: 'plots/asset_grid_comparison.png' saved.")

    # Transition diagnostics
    from collections import defaultdict
    diagnostics = {
        'policy_errs': defaultdict(list),
        'dist_errs': defaultdict(list),
        'edL_history': defaultdict(list),
        'edK_history': defaultdict(list),
        'tpi_errs_L': [], 'tpi_errs_K': []
    }

    params = (DELTA, ALPHA, NU, LAMBDA, BETA, SIGMA, PSI)
    
    # SS
    print("\n" + "="*70)
    print("[STEP 1] Post-reform SS (Nested)")
    print("="*70)
    post = find_equilibrium_nested(params, distortions=False, diag_out=diagnostics, label="post")
    
    print("\n" + "="*70)
    print("[STEP 2] Pre-reform SS (Nested)")
    print("="*70)
    pre = find_equilibrium_nested(params, distortions=True, diag_out=diagnostics, label="pre")
    
    # Transition
    print("\n" + "="*70)
    print("[STEP 3] Transition Path (Alg B.2)")
    print("="*70)
    trans = solve_transition_v2(pre, post, params, T=25, diag_out=diagnostics)
    
    # Summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"{'Variable':<20} {'Pre-Reform':>15} {'Post-Reform':>15} {'Change %':>12}")
    print("-" * 62)
    for name, key in [('Output (Y)', 'Y'), ('TFP', 'TFP'), ('Capital (K)', 'K'),
                      ('Assets (A)', 'A'), ('Wage (w)', 'w'), ('Interest Rate (r)', 'r'),
                      ('Ext.Fin/Y', 'ExtFin_Y'), ('Entre Share', 'share_entre')]:
        v0, v1 = pre[key], post[key]
        chg = (v1 - v0) / abs(v0) * 100 if v0 != 0 else 0
        print(f"{name:<20} {v0:>15.4f} {v1:>15.4f} {chg:>11.2f}%")
    print("="*70)

    # Plotting Output
    plt.figure()
    plt.plot(trans['t'], trans['Y'], label='Output Y')
    plt.title("Transition Path - Output")
    plt.savefig("v2_transition.png"); plt.close()
    
    # Final Policies and Diagnostics
    plot_policy_comparison(pre, post)
    plot_diagnostics(diagnostics)
    print("\nCompleted. v2_transition.png, plots/policy_comparison_v2.png, and plots/convergence_diagnostics.png saved.")

if __name__ == '__main__':
    main()
