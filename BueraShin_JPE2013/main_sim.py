"""
Replication of Buera & Shin (2010) - Financial Frictions and the Persistence of History

VERSION 3: FURTHER OPTIMIZED
Key improvements over v2:
1. Howard's Policy Iteration Acceleration (multiple value updates per policy)
2. Exploit policy function monotonicity (savings increasing in assets)
3. Pre-compute ALL entrepreneur solutions vectorized before VFI
4. Warm-start VFI from previous lambda solution
5. Binary search for optimal savings (exploiting concavity)
6. Vectorized stationary distribution computation
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numba import njit, prange
from scipy import sparse
import time
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Model Parameters
# =============================================================================

SIGMA = 1.5      # Risk aversion (CRRA)
BETA = 0.904     # Discount factor
ALPHA = 0.33     # Capital share
NU = 0.21        # Entrepreneur's share (span-of-control = 1 - nu)
DELTA = 0.06     # Depreciation rate
ETA = 4.15       # Pareto tail parameter
PSI = 0.894      # Ability persistence

print("=" * 60)
print("Buera-Shin (2010) Replication - V4 SIMULATION")
print("Financial Frictions and the Persistence of History")
print("=" * 60)

# Simulation Parameters (Appendix B.1)
N_AGENTS = 350000
T_SIM = 500


# =============================================================================
# Grid Construction
# =============================================================================

def create_ability_grid_paper(eta):
    """
    Paper's exact 40-point ability discretization.
    Dynamically computed based on ETA to ensure consistent scale.
    """
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


def create_asset_grid(n_a, a_min, a_max, a_scale=2):
    """Asset grid with curvature scaling (Default power 2 to match V3)"""
    a_grid = a_min + (a_max - a_min) * np.linspace(0, 1, n_a) ** a_scale
    a_grid[0] = max(a_grid[0], 1e-6)
    return a_grid


# =============================================================================
# Entrepreneur Problem - Fully Vectorized
# =============================================================================

@njit(cache=True)
def solve_entrepreneur_single(a, z, w, r, lam, delta, alpha, upsilon):
    """Solve entrepreneur problem for single (a,z) state"""
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


@njit(cache=True, parallel=True)
def precompute_entrepreneur_all(a_grid, z_grid, w, r, lam, delta, alpha, upsilon):
    """
    IMPROVEMENT: Pre-compute entrepreneur solutions for ALL (a,z) pairs at once.
    This is called once per GE iteration, not inside VFI.
    """
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


# =============================================================================
# Value Function Iteration - With Howard Acceleration & Monotonicity
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
    """
    IMPROVEMENT: Binary-like search exploiting monotonicity.
    The optimal savings policy is increasing in income/assets.
    Start search from start_idx (previous optimal for lower asset).
    """
    n_a = len(a_grid)
    best_val = -1e15
    best_idx = start_idx

    # Search forward from start_idx
    for i_a_prime in range(start_idx, n_a):
        c = income - a_grid[i_a_prime]
        if c <= 1e-10:
            break
        val = utility(c, sigma) + beta * EV_row[i_a_prime]
        if val > best_val:
            best_val = val
            best_idx = i_a_prime
        # REMOVED: elif val < best_val - 1e-8: break
        # We search the full range to avoid local maxima from occupational kinks.

    return best_val, best_idx


@njit(cache=True, parallel=True)
def bellman_operator_fast(V, a_grid, z_grid, prob_z, income_grid,
                          beta, sigma, psi):
    """
    IMPROVEMENT: Faster Bellman operator
    - Uses pre-computed income (no entrepreneur solve inside)
    - Exploits policy monotonicity
    """
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

    # Main loop (parallelized over z)
    for i_z in prange(n_z):
        start_idx = 0  # Track for monotonicity

        for i_a in range(n_a):
            income = income_grid[i_a, i_z]

            # Find optimal savings with monotonicity
            best_val, best_idx = find_optimal_savings_monotone(
                income, a_grid, EV[:, i_z], beta, sigma, start_idx
            )

            V_new[i_a, i_z] = best_val
            policy_a_idx[i_a, i_z] = best_idx
            start_idx = best_idx  # Next search starts here (monotonicity)

    return V_new, policy_a_idx


@njit(cache=True, parallel=True)
def howard_acceleration(V, policy_a_idx, a_grid, z_grid, prob_z,
                        income_grid, beta, sigma, psi, n_howard=10):
    """
    IMPROVEMENT: Howard's Policy Iteration Acceleration
    Given a fixed policy, iterate the value function multiple times.
    This converges faster than updating policy every iteration.
    """
    n_a = len(a_grid)
    n_z = len(z_grid)

    for _ in range(n_howard):
        # Pre-compute expected values
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
                              tol=1e-5, max_iter=500, n_howard=10, verbose=False):
    """
    IMPROVEMENT: VFI with Howard acceleration and warm start
    """
    n_a, n_z = len(a_grid), len(z_grid)

    # Initial guess (or warm start from previous)
    if V_init is not None:
        V = V_init.copy()
    else:
        V = np.zeros((n_a, n_z))
        for i_a in range(n_a):
            c_guess = max(income_grid[i_a, 0] - a_grid[0], 0.01)
            V[i_a, :] = utility(c_guess, sigma) / (1 - beta)

    for iteration in range(max_iter):
        # Policy improvement step
        V_new, policy_a_idx = bellman_operator_fast(
            V, a_grid, z_grid, prob_z, income_grid, beta, sigma, psi
        )

        diff = np.max(np.abs(V_new - V))

        if diff < tol:
            if verbose:
                print(f"    VFI converged in {iteration+1} iter (diff={diff:.2e})")
            return V_new, policy_a_idx

        # Howard acceleration: update value multiple times with fixed policy
        if n_howard > 0 and diff > tol * 10:
            V = howard_acceleration(V_new, policy_a_idx, a_grid, z_grid, prob_z,
                                   income_grid, beta, sigma, psi, n_howard)
        else:
            V = V_new

    if verbose:
        print(f"    VFI: max iter reached (diff={diff:.2e})")

    return V, policy_a_idx


# =============================================================================
# Stationary Distribution - Simulation Method (Appendix B.1)
# =============================================================================

@njit(cache=True)
def get_interp_indices(x, x_grid):
    """Get indices and weights for linear interpolation"""
    n = len(x_grid)
    if x <= x_grid[0]:
        return 0, 0, 1.0
    if x >= x_grid[-1]:
        return n-2, n-1, 0.0
    
    # Binary search for interval
    low = 0
    high = n - 1
    while high - low > 1:
        mid = (low + high) // 2
        if x_grid[mid] > x:
            high = mid
        else:
            low = mid
    
    weight_low = (x_grid[high] - x) / (x_grid[high] - x_grid[low])
    return low, high, weight_low

@njit(cache=True, parallel=True)
def simulation_step_fixed(a_current, z_idx_current, policy_a_vals, a_grid, 
                         reset_shocks, ability_shocks):
    """One period of simulation using pre-generated shocks for determinism"""
    n_agents = len(a_current)
    a_next = np.zeros(n_agents)
    z_idx_next = np.zeros(n_agents, dtype=np.int64)
    
    for i in prange(n_agents):
        # 1. Update Ability using fixed shocks
        if reset_shocks[i]:
            z_idx_next[i] = ability_shocks[i]
        else:
            z_idx_next[i] = z_idx_current[i]
            
        # 2. Update Assets (linear interpolation of policy)
        a = a_current[i]
        i_low, i_high, w_low = get_interp_indices(a, a_grid)
        
        pol_low = policy_a_vals[i_low, z_idx_current[i]]
        pol_high = policy_a_vals[i_high, z_idx_current[i]]
        a_next[i] = w_low * pol_low + (1.0 - w_low) * pol_high
        
    return a_next, z_idx_next

@njit(cache=True, parallel=True)
def compute_sim_aggregates_parallel(a_sim, z_idx_sim, a_grid, z_grid, w, r, lam, delta, alpha, upsilon):
    """Compute aggregates from simulation cross-section using Numba reductions"""
    n_agents = len(a_sim)
    
    K_sum = 0.0
    L_sum = 0.0
    Y_sum = 0.0
    extfin_sum = 0.0
    share_entre_sum = 0.0
    A_sum = 0.0
    
    rental = max(r + delta, 1e-8)
    wage = max(w, 1e-8)
    span = 1 - upsilon
    
    for i in prange(n_agents):
        a = a_sim[i]
        z = z_grid[z_idx_sim[i]]
        A_sum += a
        
        # Solve static problem
        aux1 = (1/rental) * alpha * span * z
        aux2 = (1/wage) * (1-alpha) * span * z
        k1 = ( (aux1 ** (1-(1-alpha)*span)) * (aux2 ** ((1-alpha)*span)) ) ** (1/upsilon)
        
        kstar = min(k1, lam * a)
        inside_lab = (1/wage) * (1-alpha) * span * z * (kstar ** (alpha * span))
        lstar = inside_lab ** (1/(1-(1-alpha)*span))
        
        output = z * ((kstar ** alpha) * (lstar ** (1-alpha))) ** span
        profit = output - wage * lstar - rental * kstar
        
        if profit > wage:
            K_sum += kstar
            L_sum += lstar
            Y_sum += output
            extfin_sum += max(0.0, kstar - a)
            share_entre_sum += 1.0
            
    return (K_sum/n_agents, L_sum/n_agents, Y_sum/n_agents, 
            A_sum/n_agents, extfin_sum/n_agents, share_entre_sum/n_agents)

def generate_fixed_shocks(n_agents, t_sim, psi, n_z, prob_z):
    """Pre-generate shocks using compact memory types"""
    print(f"   Pre-generating {n_agents} x {t_sim} shocks (uint8)...")
    np.random.seed(42)
    
    # uint8 saves 8x memory vs int64/float64
    reset_shocks = (np.random.random((t_sim, n_agents)) > psi).astype(np.uint8)
    
    cdf_z = np.cumsum(prob_z)
    ability_shocks = np.zeros((t_sim, n_agents), dtype=np.uint8)
    for t in range(t_sim):
        ability_shocks[t, :] = np.searchsorted(cdf_z, np.random.random(n_agents)).astype(np.uint8)
        
    return reset_shocks, ability_shocks

def run_simulation_appendix(a_grid, z_grid, prob_z, policy_a_idx, psi, w, r, lam, delta, alpha, upsilon, 
                           fixed_shocks):
    """Run full simulation with automatic size detection"""
    reset_shocks, ability_shocks = fixed_shocks
    t_sim, n_agents = reset_shocks.shape
    
    # Fixed initial state
    np.random.seed(123)
    a_current = np.ones(n_agents) * a_grid[0]
    z_idx_current = np.random.randint(0, len(z_grid), n_agents)
        
    policy_a_vals = a_grid[policy_a_idx]
    
    for t in range(t_sim):
        a_current, z_idx_current = simulation_step_fixed(
            a_current, z_idx_current, policy_a_vals, a_grid, 
            reset_shocks[t], ability_shocks[t]
        )
        
    K, L, Y, A, extfin, share_entre = compute_sim_aggregates_parallel(
        a_current, z_idx_current, a_grid, z_grid, w, r, lam, delta, alpha, upsilon
    )
    
    return {'K': K, 'L': L, 'Y': Y, 'A': A, 'extfin': extfin, 'share_entre': share_entre}


# =============================================================================
# Equilibrium Solver
# =============================================================================

def find_equilibrium_sim(a_grid, z_grid, prob_z, params, fixed_shocks,
                          w_init=0.80, r_init=-0.04, V_init=None,
                          max_iter=100, tol=1e-3, verbose=True):
    """Find stationary equilibrium using simulation-based distribution"""
    delta, alpha, upsilon, lam, beta, sigma, psi = params

    w, r = w_init, r_init
    best_error = np.inf
    best_result = None
    V_current = V_init
    
    # Price adjustment parameters
    w_step = 0.3
    r_step = 0.05 if lam < 1.5 else 0.02
    exc_L_prev = 0.0
    exc_K_prev = 0.0

    for iteration in range(max_iter):
        # 1. Pre-compute profit grid for VFI
        (profit_grid, kstar_grid, lstar_grid, output_grid,
         is_entrep_grid, income_grid) = precompute_entrepreneur_all(
            a_grid, z_grid, w, r, lam, delta, alpha, upsilon
        )

        # 2. Solve value function (optimized grid-based VFI)
        V_current, policy_a_idx = solve_value_function_fast(
            a_grid, z_grid, prob_z, income_grid,
            beta, sigma, psi, V_init=V_current,
            tol=1e-5, max_iter=500, n_howard=15, verbose=False
        )

        # 3. Simulate to find distribution and compute aggregates
        agg = run_simulation_appendix(
            a_grid, z_grid, prob_z, policy_a_idx, psi, w, r, lam, delta, alpha, upsilon,
            fixed_shocks
        )

        exc_K = agg['K'] - agg['A']
        exc_L = agg['L'] - (1 - agg['share_entre'])

        if verbose:
            print(f"  Iter {iteration+1}: w={w:.4f}, r={r:.4f}, "
                  f"ExcL={exc_L:.4f}, ExcK={exc_K:.4f}")

        total_error = abs(exc_L) + abs(exc_K)

        if total_error < best_error:
            best_error = total_error
            best_result = {
                'w': w, 'r': r, 'agg': agg.copy(),
                'is_entrep': is_entrep_grid.copy(),
                'V': V_current.copy()
            }

        if abs(exc_L) < tol and abs(exc_K) < tol:
            if verbose:
                print(f"  Converged!")
            break

        # Adaptive damping for oscillations (if sign flips, reduce step)
        if iteration > 0:
            if exc_L * exc_L_prev < 0:
                w_step *= 0.5
            if exc_K * exc_K_prev < 0:
                r_step *= 0.5

        # Price updates (more aggressive for negative r, but guarded)
        eff_r_step = r_step * (2.0 if r < 0 else 1.0)
        
        # In autarky (λ=1), ExcK error is very flat. If small, slow down updates.
        if lam == 1.0 and abs(exc_K) < 0.01:
            eff_r_step *= 0.2
        
        w_new = w * (1 + w_step * exc_L)
        r_new = r + eff_r_step * exc_K

        w_new = max(0.01, min(2.5, w_new))
        r_new = max(-0.25, min(0.12, r_new))

        # Damping logic
        damping = 0.5
        w = damping * w + (1 - damping) * w_new
        r = damping * r + (1 - damping) * r_new

        # Store for next iteration
        exc_L_prev = exc_L
        exc_K_prev = exc_K

    # Return the final iteration's results (more consistent with logs)
    result = {
        'w': w, 'r': r, 'agg': agg.copy(),
        'is_entrep': is_entrep_grid.copy(),
        'V': V_current.copy()
    }
    agg = result['agg']

    span = 1 - upsilon
    L_s = 1 - agg['share_entre']
    TFP = agg['Y'] / max(((agg['K'] ** alpha) * (L_s ** (1-alpha))) ** span, 1e-8)

    return {
        'w': result['w'], 'r': result['r'],
        'Y': agg['Y'], 'K': agg['K'], 'L': agg['L'], 'A': agg['A'],
        'TFP': TFP,
        'extfin': agg['extfin'],
        'ext_fin_to_gdp': agg['extfin'] / agg['Y'] if agg['Y'] > 0 else 0,
        'share_entre': agg['share_entre'],
        'K_Y': agg['K'] / agg['Y'] if agg['Y'] > 0 else 0,
        'V': result['V']  # Return for warm-starting next lambda
    }


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":

    total_start = time.time()

    # Grid parameters
    n_a = 2001
    a_min = 1e-6
    a_max = 4000
    a_scale = 3

    print("\n1. Setting up model grids...")
    z_grid, prob_z = create_ability_grid_paper(ETA)
    a_grid = create_asset_grid(n_a, a_min, a_max, a_scale=2)
    n_z = len(z_grid)

    print(f"   Ability grid: {n_z} points (paper's exact discretization)")
    print(f"   Ability range: [{z_grid.min():.4f}, {z_grid.max():.4f}]")
    print(f"   Asset grid: {n_a} points, range [{a_grid.min():.6f}, {a_grid.max():.2f}]")

    # Fixed shocks for simulation (Appendix B.1)
    fixed_shocks = generate_fixed_shocks(N_AGENTS, T_SIM, PSI, n_z, prob_z)

    # JIT warmup
    print("\n   Warming up JIT compilation...")
    warmup_start = time.time()

    # Warmup entrepreneur
    _ = solve_entrepreneur_single(1.0, 1.5, 1.0, 0.04, 2.0, DELTA, ALPHA, NU)

    # Warmup precompute
    _ = precompute_entrepreneur_all(a_grid[:10], z_grid[:5], 1.0, 0.04, 2.0,
                                     DELTA, ALPHA, NU)

    # Warmup Bellman
    V_test = np.zeros((10, 5))
    income_test = np.ones((10, 5))
    _ = bellman_operator_fast(V_test, a_grid[:10], z_grid[:5], prob_z[:5],
                              income_test, BETA, SIGMA, PSI)

    # Warmup Howard
    policy_test = np.zeros((10, 5), dtype=np.int64)
    _ = howard_acceleration(V_test, policy_test, a_grid[:10], z_grid[:5], prob_z[:5],
                           income_test, BETA, SIGMA, PSI, 2)

    # Warmup simulation
    policy_test = np.zeros((10, 5), dtype=np.int64)
    dummy_fixed = (np.zeros((2, 10), dtype=np.uint8), np.zeros((2, 10), dtype=np.uint8))
    _ = run_simulation_appendix(a_grid[:10], z_grid[:5], prob_z[:5], policy_test, PSI, 
                                 1.0, 0.04, 2.0, DELTA, ALPHA, NU, dummy_fixed)

    print(f"   JIT warmup: {time.time() - warmup_start:.1f}s")

    # ==========================================================================
    # Figure 2
    # ==========================================================================

    print("\n" + "=" * 60)
    print("2. Computing Figure 2: Long-run Effect of Financial Frictions")
    print("=" * 60)

    lambda_values = [np.inf, 2.0, 1.75, 1.5, 1.25, 1.2, 1.15, 1.1, 1.0]

    initial_guesses = {
        np.inf: (0.80, -0.04),
        1.0:    (0.50, -0.05),
    }

    results_list = []
    prev_w, prev_r = None, None
    prev_V = None  # For warm-starting VFI

    for i, lam in enumerate(lambda_values):
        print(f"\n--- Lambda = {lam} ({i+1}/{len(lambda_values)}) ---")

        start_time = time.time()
        params = (DELTA, ALPHA, NU, lam, BETA, SIGMA, PSI)

        if i == 0:
            w_init, r_init = initial_guesses.get(lam, (0.17, 0.05))
            V_init = None
        else:
            w_init, r_init = prev_w, prev_r
            V_init = prev_V
        result = find_equilibrium_sim(
            a_grid, z_grid, prob_z, params, fixed_shocks,
            w_init=w_init, r_init=r_init, V_init=V_init,
            max_iter=100, tol=2e-3, verbose=True
        )
        result['lambda'] = lam
        results_list.append(result)

        prev_w, prev_r = result['w'], result['r']
        prev_V = result.get('V', None)

        elapsed = time.time() - start_time
        print(f"   Result: Y={result['Y']:.4f}, ExtFin/Y={result['ext_fin_to_gdp']:.3f}, "
              f"%Entre={result['share_entre']*100:.1f}%")
        print(f"   Time: {elapsed:.1f}s")

    # ==========================================================================
    # Plotting
    # ==========================================================================

    print("\n" + "=" * 60)
    print("3. Plotting results")
    print("=" * 60)

    lambdas = [r['lambda'] for r in results_list]
    ext_fin_ratios = [r['ext_fin_to_gdp'] for r in results_list]
    Ys = [r['Y'] for r in results_list]
    TFPs = [r['TFP'] for r in results_list]
    interest_rates = [r['r'] for r in results_list]

    Y_perfect = Ys[0]
    TFP_perfect = TFPs[0]

    Y_normalized = [y / Y_perfect if Y_perfect > 0 else 0 for y in Ys]
    TFP_normalized = [t / TFP_perfect if TFP_perfect > 0 else 0 for t in TFPs]

    # Figure 2
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    ax1.plot(ext_fin_ratios, Y_normalized, 'b-o', label='GDP', linewidth=2, markersize=10)
    ax1.plot(ext_fin_ratios, TFP_normalized, 'r--s', label='TFP', linewidth=2, markersize=10)
    ax1.set_xlabel('External Finance to GDP', fontsize=14)
    ax1.set_ylabel('Relative to Perfect Credit (λ=∞)', fontsize=14)
    ax1.set_title('GDP and TFP vs Financial Development', fontsize=16)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0.5, 1.15])

    ax2 = axes[1]
    ax2.plot(ext_fin_ratios, interest_rates, 'g-^', linewidth=2, markersize=10)
    ax2.set_xlabel('External Finance to GDP', fontsize=14)
    ax2.set_ylabel('Interest Rate', fontsize=14)
    ax2.set_title('Equilibrium Interest Rate', fontsize=16)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

    plt.tight_layout()
    plt.savefig('plots/figure2_replication_v4.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("\nFigure 2 saved to 'plots/figure2_replication_v4.png'")

    # Diagnostic plots
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))

    lam_plot = [l if l != np.inf else 100 for l in lambdas]

    axes2[0, 0].plot(lam_plot, ext_fin_ratios, 'ko-', linewidth=2, markersize=10)
    axes2[0, 0].set_xlabel('λ', fontsize=12)
    axes2[0, 0].set_ylabel('External Finance / GDP', fontsize=12)
    axes2[0, 0].set_title('Financial Development vs λ', fontsize=14)
    axes2[0, 0].set_xscale('log')
    axes2[0, 0].grid(True, alpha=0.3)

    axes2[0, 1].plot(lam_plot, Y_normalized, 'b-o', label='GDP', linewidth=2, markersize=10)
    axes2[0, 1].plot(lam_plot, TFP_normalized, 'r--s', label='TFP', linewidth=2, markersize=10)
    axes2[0, 1].set_xlabel('λ', fontsize=12)
    axes2[0, 1].set_ylabel('Relative to Perfect Credit', fontsize=12)
    axes2[0, 1].set_title('Output and Productivity vs λ', fontsize=14)
    axes2[0, 1].set_xscale('log')
    axes2[0, 1].legend()
    axes2[0, 1].grid(True, alpha=0.3)

    wages = [r['w'] for r in results_list]
    axes2[1, 0].plot(ext_fin_ratios, wages, 'm-d', linewidth=2, markersize=10)
    axes2[1, 0].set_xlabel('External Finance / GDP', fontsize=12)
    axes2[1, 0].set_ylabel('Wage', fontsize=12)
    axes2[1, 0].set_title('Equilibrium Wage', fontsize=14)
    axes2[1, 0].grid(True, alpha=0.3)

    frac_entrep = [r['share_entre'] for r in results_list]
    axes2[1, 1].plot(ext_fin_ratios, [f*100 for f in frac_entrep], 'c-p', linewidth=2, markersize=10)
    axes2[1, 1].set_xlabel('External Finance / GDP', fontsize=12)
    axes2[1, 1].set_ylabel('Entrepreneurs (%)', fontsize=12)
    axes2[1, 1].set_title('Fraction of Entrepreneurs', fontsize=14)
    axes2[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('plots/figure2_diagnostics_v4.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Diagnostic plots saved to 'plots/figure2_diagnostics_v4.png'")

    # ==========================================================================
    # Summary
    # ==========================================================================

    print("\n" + "=" * 60)
    print("4. Summary Table")
    print("=" * 60)
    print(f"\n{'Lambda':>8} {'ExtFin/GDP':>12} {'GDP':>10} {'TFP':>10} {'r':>10} "
          f"{'w':>10} {'K/Y':>10} {'%Entre':>10}")
    print("-" * 88)
    for r in results_list:
        lam_str = "inf" if r['lambda'] == np.inf else f"{r['lambda']:.2f}"
        y_norm = r['Y']/Y_perfect if Y_perfect > 0 else 0
        tfp_norm = r['TFP']/TFP_perfect if TFP_perfect > 0 else 0
        print(f"{lam_str:>8} {r['ext_fin_to_gdp']:>12.4f} {y_norm:>10.4f} "
              f"{tfp_norm:>10.4f} {r['r']:>10.4f} {r['w']:>10.4f} "
              f"{r['K_Y']:>10.3f} {r['share_entre']*100:>9.1f}%")

    total_time = time.time() - total_start

    print("\n" + "=" * 60)
    print("Key findings:")
    print("=" * 60)
    print(f"- Perfect credit (λ=∞): Y={Y_perfect:.4f}")
    print(f"- Financial autarky (λ=1): GDP={Y_normalized[-1]:.2f}, TFP={TFP_normalized[-1]:.2f}")
    print(f"- Interest rate range: [{min(interest_rates):.4f}, {max(interest_rates):.4f}]")
    print(f"- GDP loss: up to {(1-min(Y_normalized))*100:.1f}%")
    print(f"- TFP loss: up to {(1-min(TFP_normalized))*100:.1f}%")

    print("\n" + "=" * 60)
    print(f"TOTAL RUNTIME: {total_time:.1f}s ({total_time/60:.1f} min)")
    print("=" * 60)
