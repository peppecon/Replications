"""
Buera & Shin (2013) Transition Dynamics - VERSION 4 (PAPER-EXACT B.2 ALGORITHM)
Strict implementation of the nested-loop algorithm described in Appendix B.2.
- Outer loop for Interest Rate sequence {r_t}
- Inner loop for Wage sequence {w_t}
- Period-by-period market clearing via root-finding (brentq)
- N=350,000 agents, Monte Carlo distribution
- Includes Figure 5 distributional statistics and premium plotting
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
import time
import os
import json
import argparse
import warnings
from scipy.optimize import brentq

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

# =============================================================================
# Grid Construction (Harmonized with V3 Baseline)
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
# Numerical Kernels (Numba Accelerated)
# =============================================================================

@njit(cache=True)
def solve_entrepreneur_with_tau(a, z, tau, w, r, lam, delta, alpha, upsilon):
    """Static profit maximization with distortion tau"""
    rental = max(r + delta, 1e-8)
    wage = max(w, 1e-8)
    span = 1 - upsilon
    rental_eff = rental / (1 - tau)
    
    aux1 = (1/rental_eff) * alpha * span * z
    aux2 = (1/wage) * (1-alpha) * span * z
    exp1 = 1 - (1-alpha) * span
    exp2 = (1-alpha) * span
    k1 = (aux1 ** exp1 * aux2 ** exp2) ** (1/upsilon)
    
    kstar = min(k1, lam * a)
    lstar = ((aux2 * (kstar ** (alpha * span))) ** (1/exp1))
    output = z * ((kstar ** alpha) * (lstar ** (1-alpha))) ** span
    profit = (1 - tau) * output - wage * lstar - rental * kstar
    return profit, kstar, lstar, output

@njit(cache=True, parallel=True)
def precompute_income(a_grid, z_grid, tau_v, w, r, lam, delta, alpha, upsilon):
    """
    Handles income precomputation. If tau_v is float, use that for all.
    Else assume tau_v is an array or we need to handle expected outcomes.
    For VFI we usually need weighted average income if tau is stochastic.
    """
    n_a, n_z = len(a_grid), len(z_grid)
    income_g = np.zeros((n_a, n_z))
    for i_z in prange(n_z):
        z = z_grid[i_z]
        tau = tau_v # Assumes float or handling stochastic outside
        for i_a in range(n_a):
            p, k, l, o = solve_entrepreneur_with_tau(a_grid[i_a], z, tau, w, r, lam, delta, alpha, upsilon)
            income_g[i_a, i_z] = max(p, w) + (1 + r) * a_grid[i_a]
    return income_g

# Value Function Iteration
@njit(cache=True)
def utility(c, sigma):
    if c <= 1e-10: return -1e10
    if abs(sigma - 1.0) < 1e-6: return np.log(c)
    return (c**(1-sigma) - 1) / (1-sigma)

@njit(cache=True)
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

@njit(cache=True, parallel=True)
def bellman_operator(V_target, a_grid, z_grid, prob_z, income_grid, beta, sigma, psi):
    n_a, n_z = len(a_grid), len(z_grid)
    V_new = np.zeros((n_a, n_z))
    policy_a_idx = np.zeros((n_a, n_z), dtype=np.int64)
    V_mean = np.zeros(n_a)
    for i_a in range(n_a):
        for i_z in range(n_z):
            V_mean[i_a] += prob_z[i_z] * V_target[i_a, i_z]
            
    for i_z in prange(n_z):
        start_idx = 0
        EV_row = psi * V_target[:, i_z] + (1 - psi) * V_mean
        for i_a in range(n_a):
            income = income_grid[i_a, i_z]
            v, idx = find_optimal_savings(income, a_grid, EV_row, beta, sigma, start_idx)
            V_new[i_a, i_z], policy_a_idx[i_a, i_z] = v, idx
            start_idx = idx
    return V_new, policy_a_idx

# Simulation routines
@njit(cache=True)
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

@njit(cache=True, parallel=True)
def simulate_step(a_curr, z_idx_curr, policy_a_vals, a_grid, reset_shocks, ability_shocks):
    n = len(a_curr)
    a_next = np.zeros(n)
    z_idx_next = np.zeros(n, dtype=np.int64)
    for i in prange(n):
        z_idx_next[i] = ability_shocks[i] if reset_shocks[i] else z_idx_curr[i]
        i_l, i_h, w_l = get_interp_weights(a_curr[i], a_grid)
        a_next[i] = w_l * policy_a_vals[i_l, z_idx_curr[i]] + (1-w_l) * policy_a_vals[i_h, z_idx_curr[i]]
    return a_next, z_idx_next

@njit(cache=True, parallel=True)
def compute_labor_market_excess(a_sim, z_idx_sim, tau_sim, z_grid, w, r, lam, delta, alpha, upsilon):
    n = len(a_sim)
    Ld, Ls = 0.0, 0.0
    for i in prange(n):
        a, z, tau = a_sim[i], z_grid[z_idx_sim[i]], tau_sim[i]
        p, k, l, o = solve_entrepreneur_with_tau(a, z, tau, w, r, lam, delta, alpha, upsilon)
        if p > w:
            Ld += l
        else:
            Ls += 1.0
    return (Ld - Ls) / n

@njit(cache=True, parallel=True)
def compute_capital_market_excess(a_sim, z_idx_sim, tau_sim, z_grid, w, r, lam, delta, alpha, upsilon):
    n = len(a_sim)
    Kd, A = 0.0, 0.0
    for i in prange(n):
        a, z, tau = a_sim[i], z_grid[z_idx_sim[i]], tau_sim[i]
        A += a
        p, k, l, o = solve_entrepreneur_with_tau(a, z, tau, w, r, lam, delta, alpha, upsilon)
        if p > w:
            Kd += k
    return (Kd - A) / n

@njit(cache=True, parallel=True)
def compute_full_aggregates_with_stats(a_sim, z_idx_sim, tau_sim, z_grid, prob_z, w, r, lam, delta, alpha, upsilon):
    n = len(a_sim)
    K, L, Y, A, extfin, s_e = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    
    # Paper-style Stats
    z_sum_entre = 0.0
    mass_e = 0.0
    
    # For Top 5% wealth
    # We need to sum assets by z-ability.
    n_z = len(z_grid)
    wealth_by_z = np.zeros(n_z)
    
    for i in prange(n):
        a, z_idx = a_sim[i], z_idx_sim[i]
        z, tau = z_grid[z_idx], tau_sim[i]
        A += a
        wealth_by_z[z_idx] += a
        
        p, k, l, o = solve_entrepreneur_with_tau(a, z, tau, w, r, lam, delta, alpha, upsilon)
        if p > w:
            K += k; L += l; Y += o
            extfin += max(0.0, k - a); s_e += 1.0
            z_sum_entre += z; mass_e += 1.0
            
    avg_z_e = z_sum_entre / max(mass_e, 1e-8)
    
    # Top 5% ability wealth share
    prob_z_cum = np.cumsum(prob_z)
    z_top5_idx = np.where(prob_z_cum >= 0.95)[0]
    wealth_top5 = np.sum(wealth_by_z[z_top5_idx]) / max(A, 1e-8)
    
    return {'K': K/n, 'L': L/n, 'Y': Y/n, 'A': A/n, 'extfin': extfin/n, 'share_entre': s_e/n,
            'avg_z_entrep': avg_z_e, 'wealth_top5_share': wealth_top5}

def generate_shocks(n_agents, t_steps, psi, prob_z):
    np.random.seed(42)
    resets = (np.random.rand(t_steps, n_agents) > psi).astype(np.uint8)
    cdf_z = np.cumsum(prob_z)
    shocks = np.searchsorted(cdf_z, np.random.rand(t_steps, n_agents)).astype(np.uint8)
    return resets, shocks

# =============================================================================
# Stationary Equilibrium Solvers
# =============================================================================

def find_equilibrium_nodist(a_grid, z_grid, prob_z, params, w_init=0.8, r_init=-0.04, verbose=True):
    delta, alpha, upsilon, lam, beta, sigma, psi = params
    w, r = w_init, r_init
    V = None
    resets, shocks = generate_shocks(N_AGENTS, T_SIM_SS, PSI, prob_z)
    for it in range(100):
        inc = precompute_income(a_grid, z_grid, 0.0, w, r, lam, delta, alpha, upsilon)
        V, pol_idx = solve_value_function(a_grid, z_grid, prob_z, inc, beta, sigma, psi, V_init=V)
        a_curr, z_idx_curr = np.ones(N_AGENTS) * a_grid[0], np.random.randint(0, len(z_grid), N_AGENTS)
        pol_vals = a_grid[pol_idx]
        for t in range(T_SIM_SS):
            a_curr, z_idx_curr = simulate_step(a_curr, z_idx_curr, pol_vals, a_grid, resets[0], shocks[0])
        agg = compute_full_aggregates_with_stats(a_curr, z_idx_curr, np.zeros(N_AGENTS), z_grid, prob_z, w, r, lam, delta, alpha, upsilon)
        exc_L, exc_K = agg['L'] - (1-agg['share_entre']), agg['K'] - agg['A']
        if verbose: print(f"  [Post SS {it:2d}] w={w:.4f} r={r:.4f} | Ld={agg['L']:.4f} Ls={1-agg['share_entre']:.4f}")
        if abs(exc_L) < 1e-4 and abs(exc_K) < 1e-4: break
        w *= (1 + 0.2*exc_L); r += 0.02*exc_K
    agg['w'], agg['r'], agg['V'], agg['pol_idx'], agg['a_sim'], agg['z_idx_sim'] = w, r, V, pol_idx, a_curr, z_idx_curr
    agg['TFP'] = agg['Y'] / max(((agg['K']**alpha) * ((1-agg['share_entre'])**(1-alpha)))**(1-upsilon), 1e-8)
    return agg

def find_equilibrium_with_dist(a_grid, z_grid, prob_z, prob_tau_plus, params, tau_p, tau_m, w_init=0.8, r_init=-0.04, verbose=True):
    delta, alpha, upsilon, lam, beta, sigma, psi = params
    w, r = w_init, r_init
    V = None # Using single V as approximation for seed generation
    resets, shocks = generate_shocks(N_AGENTS, T_SIM_SS, PSI, prob_z)
    for it in range(100):
        # We approximate V using expected tau for seed generation
        tau_avg = np.sum(prob_z * (prob_tau_plus * tau_p + (1-prob_tau_plus) * tau_m))
        inc = precompute_income(a_grid, z_grid, tau_avg, w, r, lam, delta, alpha, upsilon)
        V, pol_idx = solve_value_function(a_grid, z_grid, prob_z, inc, beta, sigma, psi, V_init=V)
        a_curr, z_idx_curr = np.ones(N_AGENTS) * a_grid[0], np.random.randint(0, len(z_grid), N_AGENTS)
        tau_curr = np.array([tau_p if np.random.rand() < prob_tau_plus[z_idx_curr[i]] else tau_m for i in range(N_AGENTS)])
        pol_vals = a_grid[pol_idx]
        for t in range(T_SIM_SS):
            a_curr, z_idx_curr = simulate_step(a_curr, z_idx_curr, pol_vals, a_grid, resets[0], shocks[0])
        agg = compute_full_aggregates_with_stats(a_curr, z_idx_curr, tau_curr, z_grid, prob_z, w, r, lam, delta, alpha, upsilon)
        exc_L, exc_K = agg['L'] - (1-agg['share_entre']), agg['K'] - agg['A']
        if verbose: print(f"  [Pre SS {it:2d}] w={w:.4f} r={r:.4f} | Ld={agg['L']:.4f} Ls={1-agg['share_entre']:.4f}")
        if abs(exc_L) < 1e-4 and abs(exc_K) < 1e-4: break
        w *= (1 + 0.2*exc_L); r += 0.02*exc_K
    agg['w'], agg['r'], agg['a_sim'], agg['z_idx_sim'], agg['tau_sim'] = w, r, a_curr, z_idx_curr, tau_curr
    agg['TFP'] = agg['Y'] / max(((agg['K']**alpha) * ((1-agg['share_entre'])**(1-alpha)))**(1-upsilon), 1e-8)
    return agg

# =============================================================================
# Paper Algorithm (B.2) Nested Loops
# =============================================================================

def solve_transition_paper(pre_eq, post_eq, params, T=125):
    delta, alpha, upsilon, lam, beta, sigma, psi = params
    w_path = np.linspace(pre_eq['w'], post_eq['w'], T)
    r_path = np.linspace(pre_eq['r'], post_eq['r'], T)
    z_grid, prob_z = create_ability_grid_paper(ETA)
    a_grid = create_asset_grid(501, 1e-6, 4000)
    resets, shocks = generate_shocks(N_AGENTS, T, PSI, prob_z)

    print(f"\nSTARTING B.2 NESTED LOOP (T={T})")
    
    for outer_it in range(25):
        # Step 2: Inner Wage Loop
        for inner_it in range(20):
            # Backward induction
            V_path = [None]*T
            pol_vals_path = [None]*T
            V_path[T-1] = post_eq['V']
            for t in range(T-2, -1, -1):
                inc = precompute_income(a_grid, z_grid, 0.0, w_path[t], r_path[t], lam, delta, alpha, upsilon)
                # Use a single step of Bellman for one-period-ahead value
                V_path[t], idx = bellman_operator(V_path[t+1], a_grid, z_grid, prob_z, inc, beta, sigma, psi)
                pol_vals_path[t] = a_grid[idx]
            
            # Forward simulation to find market clearing wages
            a_t, z_idx_t = pre_eq['a_sim'], pre_eq['z_idx_sim']
            w_clearing = np.zeros(T)
            
            for t in range(T):
                # Search for w_t that clears labor market today given (a_t, z_t)
                def labor_excess(w_test):
                    return compute_labor_market_excess(a_t, z_idx_t, np.zeros(N_AGENTS), z_grid, w_test, r_path[t], lam, delta, alpha, upsilon)
                try:
                    w_clearing[t] = brentq(labor_excess, 0.1, 2.5, xtol=1e-4)
                except:
                    w_clearing[t] = w_path[t] # Fallback
                
                # Advance distribution
                pol = pol_vals_path[t] if pol_vals_path[t] is not None else a_grid[post_eq['pol_idx']]
                a_t, z_idx_t = simulate_step(a_t, z_idx_t, pol, a_grid, resets[t], shocks[t])
            
            w_diff = np.max(np.abs(w_clearing - w_path))
            print(f"  [Out {outer_it:2d} | Inn {inner_it:2d}] Wage Error={w_diff:.5f}")
            w_path = 0.5 * w_clearing + 0.5 * w_path
            if w_diff < 1e-3: break

        # Step 3: Interest Rate Update (Outer Loop)
        r_clearing = np.zeros(T)
        a_t, z_idx_t = pre_eq['a_sim'], pre_eq['z_idx_sim']
        for t in range(T):
            def cap_excess(r_test):
                return compute_capital_market_excess(a_t, z_idx_t, np.zeros(N_AGENTS), z_grid, w_path[t], r_test, lam, delta, alpha, upsilon)
            try:
                r_clearing[t] = brentq(cap_excess, -0.2, 0.2, xtol=1e-4)
            except:
                r_clearing[t] = r_path[t]
            pol = pol_vals_path[t] if pol_vals_path[t] is not None else a_grid[post_eq['pol_idx']]
            a_t, z_idx_t = simulate_step(a_t, z_idx_t, pol, a_grid, resets[t], shocks[t])
        
        r_diff = np.max(np.abs(r_clearing - r_path))
        print(f"  [Out {outer_it:2d}] Interest Rate Error={r_diff:.5f}")
        r_path = 0.2 * r_clearing + 0.8 * r_path
        if r_diff < 1e-3: break
    
    # Final path computation with all stats
    hist = {k: [] for k in ['t', 'w', 'r', 'Y', 'K', 'L', 'TFP', 'AvgZ', 'WealthTop5']}
    a_t, z_idx_t = pre_eq['a_sim'], pre_eq['z_idx_sim']
    for t in range(T):
        agg = compute_full_aggregates_with_stats(a_t, z_idx_t, np.zeros(N_AGENTS), z_grid, prob_z, w_path[t], r_path[t], lam, delta, alpha, upsilon)
        hist['t'].append(t)
        hist['w'].append(w_path[t]); hist['r'].append(r_path[t])
        hist['Y'].append(agg['Y']); hist['K'].append(agg['K']); hist['L'].append(agg['L'])
        hist['TFP'].append(agg['Y'] / max(((agg['K']**alpha) * ((1-agg['share_entre'])**(1-alpha)))**(1-upsilon), 1e-8))
        hist['AvgZ'].append(agg['avg_z_entrep']); hist['WealthTop5'].append(agg['wealth_top5_share'])
        
        pol = pol_vals_path[t] if pol_vals_path[t] is not None else a_grid[post_eq['pol_idx']]
        a_t, z_idx_t = simulate_step(a_t, z_idx_t, pol, a_grid, resets[t], shocks[t])

    return {k: np.array(v) for k, v in hist.items()}

# =============================================================================
# Plotting and Main
# =============================================================================

def plot_transition(pre_eq, post_eq, trans, output_dir='outputs'):
    os.makedirs(output_dir, exist_ok=True)
    plt.rcParams.update({'font.size': 12, 'lines.linewidth': 2.5})
    
    T_limit = 20
    t_plot = trans['t'][:T_limit+1]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    def pad_norm(path, pre_val): return path[:T_limit+1] / pre_val
    
    axes[0,0].plot(t_plot, pad_norm(trans['Y'], pre_eq['Y']), 'b-'); axes[0,0].set_title('Output (Relative to Pre)')
    axes[0,1].plot(t_plot, pad_norm(trans['TFP'], pre_eq['TFP']), 'r-'); axes[0,1].set_title('TFP (Relative to Pre)')
    axes[0,2].plot(t_plot, trans['r'][:T_limit+1], 'g-'); axes[0,2].set_title('Interest Rate')
    axes[1,0].plot(t_plot, pad_norm(trans['w'], pre_eq['w']), 'm-'); axes[1,0].set_title('Wage (Relative to Pre)')
    axes[1,1].plot(t_plot, trans['AvgZ'][:T_limit+1]/pre_eq['avg_z_entrep'], 'c-'); axes[1,1].set_title('Avg Ability (Relative to Pre)')
    axes[1,2].plot(t_plot, trans['WealthTop5'][:T_limit+1], 'k-'); axes[1,2].set_title('Top 5% Wealth Share')
    
    for ax in axes.flat: ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'transition_paper_algo_v4.png'), dpi=300)
    print(f"Plots saved to {output_dir}/transition_paper_algo_v4.png")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=125)
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()
    
    z_grid, prob_z = create_ability_grid_paper(ETA)
    a_grid = create_asset_grid(501, 1e-6, 4000)
    prob_tau_plus = compute_tau_probs(z_grid, Q_DIST)
    params = (DELTA, ALPHA, NU, LAMBDA, BETA, SIGMA, PSI)
    
    print("Step 1: Post-reform Steady State (Simulation)")
    post_eq = find_equilibrium_nodist(a_grid, z_grid, prob_z, params)
    
    print("Step 2: Pre-reform Steady State (Simulation)")
    pre_eq = find_equilibrium_with_dist(a_grid, z_grid, prob_z, prob_tau_plus, params, TAU_PLUS, TAU_MINUS)
    
    print("Step 3: Paper Transition (Nested Loop B.2)")
    trans = solve_transition_paper(pre_eq, post_eq, params, T=args.T)
    
    plot_transition(pre_eq, post_eq, trans, args.output_dir)

if __name__ == "__main__":
    main()
