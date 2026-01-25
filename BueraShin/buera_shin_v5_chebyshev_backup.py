"""
Replication of Buera & Shin (2010) - Financial Frictions and the Persistence of History

VERSION 5: CHEBYSHEV PROJECTION METHOD (DEBUGGED)
Uses Euler-equation iteration with Chebyshev polynomial approximation for the asset policy.

Key features:
1. Approximate a'(a,z) using Chebyshev polynomials in a for each discrete z
2. Solve Euler equation at collocation nodes using root-finding
3. Fit Chebyshev coefficients after each iteration
4. Large-scale simulation for stationary distribution
5. NUMBA JIT compilation for critical paths
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from numba import njit, prange
import time
import warnings
import os

warnings.filterwarnings('ignore')


# =============================================================================
# Chebyshev Functions (Numba-compatible)
# =============================================================================

@njit(cache=True)
def chebyshev_nodes(n):
    """Compute Chebyshev nodes in [-1, 1]"""
    nodes = np.zeros(n)
    for k in range(n):
        nodes[k] = np.cos(((2*(k+1)-1)/(2*n))*np.pi)
    return nodes


@njit(cache=True)
def chebyshev_basis(x, p):
    """Evaluate Chebyshev polynomials T_0, ..., T_{p-1} at x"""
    T = np.zeros(p)
    T[0] = 1.0
    if p > 1:
        T[1] = x
    for j in range(1, p-1):
        T[j+1] = 2*x*T[j] - T[j-1]
    return T


@njit(cache=True)
def chebyshev_basis_matrix(x_arr, p):
    """Evaluate Chebyshev polynomials at multiple points. Returns (n, p) matrix"""
    n = len(x_arr)
    T = np.zeros((n, p))
    for i in range(n):
        T[i, :] = chebyshev_basis(x_arr[i], p)
    return T


@njit(cache=True)
def map_to_cheb(a, a_min, a_max, log_min, log_max, a_shift):
    """Map asset value to Chebyshev domain [-1, 1] via log-transform"""
    a_safe = max(a, a_min)
    log_a = np.log(a_safe + a_shift)
    x = 2.0 * (log_a - log_min) / (log_max - log_min) - 1.0
    # Clamp to [-1, 1] for numerical safety
    return max(-1.0, min(1.0, x))


@njit(cache=True)
def map_from_cheb(x, log_min, log_max, a_shift):
    """Map from Chebyshev domain to asset space"""
    log_a = (log_min + log_max) / 2.0 + ((log_max - log_min) / 2.0) * x
    return np.exp(log_a) - a_shift


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

print("=" * 70)
print("Buera-Shin (2010) Replication - V5 CHEBYSHEV PROJECTION (DEBUGGED)")
print("Financial Frictions and the Persistence of History")
print("=" * 70)


# =============================================================================
# Grid Construction
# =============================================================================

def create_ability_grid_paper(eta):
    """Paper's exact 40-point ability discretization (page 235)"""
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


# =============================================================================
# Chebyshev Approximator Class
# =============================================================================

class ChebyshevApproximator:
    """
    Chebyshev polynomial approximation for asset policy a'(a,z).
    Uses LOG-MAPPING to handle wide asset domain [a_min, a_max].
    """

    def __init__(self, a_min, a_max, n_nodes, n_poly, a_shift=1.0):
        self.a_min = a_min
        self.a_max = a_max
        self.n_nodes = n_nodes
        self.n_poly = n_poly
        self.a_shift = a_shift

        # Log-transformed bounds
        self.log_min = np.log(a_min + a_shift)
        self.log_max = np.log(a_max + a_shift)

        # Chebyshev nodes in [-1, 1]
        self.cheb_nodes = chebyshev_nodes(n_nodes)

        # Map to asset space via log-transform
        self.a_nodes = np.array([map_from_cheb(x, self.log_min, self.log_max, a_shift)
                                  for x in self.cheb_nodes])

        # Precompute Chebyshev basis at nodes: shape (n_nodes, n_poly)
        self.T_at_nodes = chebyshev_basis_matrix(self.cheb_nodes, n_poly)

    def eval_policy(self, a, gamma_z):
        """Evaluate policy a'(a, z) for a single z"""
        x = map_to_cheb(a, self.a_min, self.a_max, self.log_min, self.log_max, self.a_shift)
        T = chebyshev_basis(x, self.n_poly)
        a_prime = np.dot(gamma_z, T)
        return np.clip(a_prime, self.a_min, self.a_max)

    def fit_coefficients(self, a_values):
        """Fit Chebyshev coefficients from policy values at nodes"""
        if self.n_nodes == self.n_poly:
            gamma = np.linalg.solve(self.T_at_nodes, a_values)
        else:
            gamma = np.linalg.lstsq(self.T_at_nodes, a_values, rcond=None)[0]
        return gamma


# =============================================================================
# Numba-JIT Functions for Fast Computation
# =============================================================================

@njit(cache=True)
def eval_policy_jit(a, gamma_z, n_poly, a_min, a_max, log_min, log_max, a_shift):
    """JIT-compiled policy evaluation"""
    x = map_to_cheb(a, a_min, a_max, log_min, log_max, a_shift)
    T = chebyshev_basis(x, n_poly)
    a_prime = 0.0
    for m in range(n_poly):
        a_prime += gamma_z[m] * T[m]
    return max(a_min, min(a_max, a_prime))


@njit(cache=True)
def solve_entrepreneur_jit(a, z, w, r, lam, delta, alpha, upsilon):
    """JIT-compiled entrepreneur problem solver"""
    rental = max(r + delta, 1e-8)
    wage = max(w, 1e-8)

    span = 1 - upsilon
    aux1 = (1/rental) * alpha * span * z
    aux2 = (1/wage) * (1-alpha) * span * z
    exp1 = 1 - (1-alpha) * span
    exp2 = (1-alpha) * span

    # Unconstrained optimal capital
    k1 = (aux1 ** exp1 * aux2 ** exp2) ** (1/upsilon)

    # Apply collateral constraint
    if lam < 1e10:  # Finite lambda
        kstar = min(k1, lam * a)
    else:  # lambda = infinity
        kstar = k1

    # Optimal labor
    inside_lab = (1/wage) * (1-alpha) * span * z * (kstar ** (alpha * span))
    lstar = inside_lab ** (1/exp1)

    # Output and profit
    output = z * ((kstar ** alpha) * (lstar ** (1-alpha))) ** span
    profit = output - wage * lstar - rental * kstar

    return profit, kstar, lstar, output


@njit(cache=True)
def compute_income_jit(a, z, w, r, lam, delta, alpha, upsilon):
    """Compute total income (including asset return) after occupational choice"""
    profit, _, _, _ = solve_entrepreneur_jit(a, z, w, r, lam, delta, alpha, upsilon)

    if profit > w:
        return profit + (1 + r) * a
    else:
        return w + (1 + r) * a


@njit(cache=True)
def euler_residual_jit(a_prime, a, z_idx, z_grid, prob_z, w, r, lam,
                       gamma, n_z, n_poly, a_min, a_max, log_min, log_max, a_shift,
                       delta, alpha, upsilon, sigma, beta, psi):
    """
    JIT-compiled Euler residual.

    Euler equation: u'(c) = beta * (1+r) * E[u'(c') | z]
    Residual: F(a') = u'(c) - beta * (1+r) * E[u'(c')]

    F is INCREASING in a' (higher a' → lower c → higher u'(c))
    """
    c_min = 1e-10

    # Current income and consumption
    income = compute_income_jit(a, z_grid[z_idx], w, r, lam, delta, alpha, upsilon)
    c = income - a_prime

    if c <= c_min:
        return 1e10  # Infeasible - consumption too low

    mu_current = c ** (-sigma)

    # Expected marginal utility next period
    # E[u'(c') | z] = psi * u'(c'(a', z)) + (1-psi) * sum_k prob_z[k] * u'(c'(a', z_k))
    mu_next_sum = 0.0
    mu_next_own = 0.0

    for k in range(n_z):
        # Income at (a', z_k)
        income_next = compute_income_jit(a_prime, z_grid[k], w, r, lam, delta, alpha, upsilon)

        # Policy a''(a', z_k) from current Chebyshev approximation
        a_double_prime = eval_policy_jit(a_prime, gamma[:, k], n_poly, a_min, a_max,
                                          log_min, log_max, a_shift)

        # Consumption next period
        c_next = max(income_next - a_double_prime, c_min)
        mu_next_k = c_next ** (-sigma)

        mu_next_sum += prob_z[k] * mu_next_k
        if k == z_idx:
            mu_next_own = mu_next_k

    E_mu = psi * mu_next_own + (1 - psi) * mu_next_sum

    return mu_current - beta * (1 + r) * E_mu


# =============================================================================
# Euler Equation Solver (CORRECTED CORNER LOGIC)
# =============================================================================

def solve_euler_at_node(a, z_idx, z_grid, prob_z, w, r, lam,
                        cheb_approx, gamma, delta, alpha, upsilon, sigma, beta, psi):
    """
    Solve Euler equation for optimal a' at a single collocation node (a, z).

    IMPORTANT: The Euler residual F(a') is INCREASING in a'.
    - F > 0: u'(c) > beta*(1+r)*E[u'(c')] → want to consume more → save LESS
    - F < 0: u'(c) < beta*(1+r)*E[u'(c')] → want to consume less → save MORE
    """
    c_min = 1e-10
    a_min = cheb_approx.a_min
    a_max = cheb_approx.a_max
    n_poly = cheb_approx.n_poly
    log_min = cheb_approx.log_min
    log_max = cheb_approx.log_max
    a_shift = cheb_approx.a_shift
    n_z = len(z_grid)

    # Compute income
    income = compute_income_jit(a, z_grid[z_idx], w, r, lam, delta, alpha, upsilon)

    # Feasible bounds for a'
    lo = a_min
    hi = min(a_max, income - c_min)

    if hi <= lo + 1e-10:
        return lo  # Forced to minimum savings

    # Define residual function
    def F(a_prime):
        return euler_residual_jit(a_prime, a, z_idx, z_grid, prob_z, w, r, lam,
                                   gamma, n_z, n_poly, a_min, a_max, log_min, log_max, a_shift,
                                   delta, alpha, upsilon, sigma, beta, psi)

    # Evaluate at endpoints
    try:
        F_lo = F(lo)
        F_hi = F(hi)
    except:
        return 0.5 * (lo + hi)

    # CORRECTED CORNER SOLUTION LOGIC:
    # F is increasing in a', so:
    # - If F(lo) >= 0: even at minimum saving, LHS >= RHS → save less → a' = lo
    # - If F(hi) <= 0: even at maximum saving, LHS <= RHS → save more → a' = hi
    # - Otherwise: interior solution (F changes sign)

    if F_lo >= 0:
        # Want to save less, but already at minimum
        return lo

    if F_hi <= 0:
        # Want to save more, but already at maximum
        return hi

    # Interior solution: F(lo) < 0 and F(hi) > 0
    try:
        a_prime_opt = brentq(F, lo, hi, xtol=1e-8, maxiter=100)
    except ValueError:
        # This shouldn't happen if F is monotonic, but fallback
        a_prime_opt = 0.5 * (lo + hi)
    except:
        a_prime_opt = 0.5 * (lo + hi)

    return np.clip(a_prime_opt, a_min, a_max)


def solve_policy_chebyshev(w, r, lam, gamma_init, cheb_approx, z_grid, prob_z,
                           delta, alpha, upsilon, sigma, beta, psi,
                           max_iter=150, tol=1e-5, damping=0.7, verbose=False):
    """
    Solve for the asset policy function using Euler-equation iteration.
    """
    n_nodes = cheb_approx.n_nodes
    n_z = len(z_grid)
    n_poly = cheb_approx.n_poly
    a_nodes = cheb_approx.a_nodes

    gamma = gamma_init.copy()

    for iteration in range(max_iter):
        # Evaluate current policy at nodes
        a_old = np.zeros((n_nodes, n_z))
        for j in range(n_z):
            for i in range(n_nodes):
                a_old[i, j] = cheb_approx.eval_policy(a_nodes[i], gamma[:, j])

        # Solve Euler equation at each node
        a_new = np.zeros((n_nodes, n_z))
        for j in range(n_z):
            for i in range(n_nodes):
                a_new[i, j] = solve_euler_at_node(
                    a_nodes[i], j, z_grid, prob_z, w, r, lam,
                    cheb_approx, gamma, delta, alpha, upsilon, sigma, beta, psi
                )

        # Fit new Chebyshev coefficients for each z
        gamma_new = np.zeros((n_poly, n_z))
        for j in range(n_z):
            gamma_new[:, j] = cheb_approx.fit_coefficients(a_new[:, j])

        # Damped update
        gamma_updated = (1 - damping) * gamma + damping * gamma_new

        # Check convergence
        max_diff = np.max(np.abs(a_new - a_old))

        if verbose and iteration % 25 == 0:
            print(f"    Policy iter {iteration}: max |a'_new - a'_old| = {max_diff:.2e}")

        if max_diff < tol:
            if verbose:
                print(f"    Policy converged in {iteration+1} iterations")
            break

        gamma = gamma_updated

    return gamma


# =============================================================================
# Simulation for Stationary Distribution
# =============================================================================

@njit(cache=True)
def simulate_stationary_jit(gamma, a_nodes, z_grid, prob_z, prob_z_cumsum,
                             w, r, lam, n_poly, a_min, a_max, log_min, log_max, a_shift,
                             delta, alpha, upsilon, psi, N, T, seed):
    """JIT-compiled simulation for stationary aggregates."""
    np.random.seed(seed)
    n_z = len(z_grid)

    # Initialize population at median asset node
    a = np.ones(N) * a_nodes[len(a_nodes)//2]
    z_idx = np.zeros(N, dtype=np.int64)

    # Initial z distribution
    for i in range(N):
        u = np.random.random()
        for j in range(n_z):
            if u <= prob_z_cumsum[j]:
                z_idx[i] = j
                break

    # Burn-in simulation
    for t in range(T):
        # Update assets using policy
        for i in range(N):
            a[i] = eval_policy_jit(a[i], gamma[:, z_idx[i]], n_poly, a_min, a_max,
                                    log_min, log_max, a_shift)

        # Update ability
        for i in range(N):
            if np.random.random() > psi:
                u = np.random.random()
                for j in range(n_z):
                    if u <= prob_z_cumsum[j]:
                        z_idx[i] = j
                        break

    # Compute aggregates
    K_total = 0.0
    L_total = 0.0
    Y_total = 0.0
    A_total = 0.0
    extfin_total = 0.0
    n_entre = 0

    for i in range(N):
        ai = a[i]
        zi = z_grid[z_idx[i]]

        profit, kstar, lstar, output = solve_entrepreneur_jit(
            ai, zi, w, r, lam, delta, alpha, upsilon
        )

        A_total += ai

        if profit > w:
            K_total += kstar
            L_total += lstar
            Y_total += output
            extfin_total += max(0.0, kstar - ai)
            n_entre += 1

    return K_total/N, L_total/N, Y_total/N, A_total/N, extfin_total/N, float(n_entre)/N


def simulate_stationary(gamma, cheb_approx, z_grid, prob_z, w, r, lam,
                        delta, alpha, upsilon, psi,
                        N=150000, T=300, seed=12345):
    """Wrapper for JIT-compiled simulation"""
    prob_z_cumsum = np.cumsum(prob_z)

    K, L, Y, A, extfin, share_entre = simulate_stationary_jit(
        gamma, cheb_approx.a_nodes, z_grid, prob_z, prob_z_cumsum,
        w, r, lam, cheb_approx.n_poly, cheb_approx.a_min, cheb_approx.a_max,
        cheb_approx.log_min, cheb_approx.log_max, cheb_approx.a_shift,
        delta, alpha, upsilon, psi, N, T, seed
    )

    return {
        'K': K, 'L': L, 'Y': Y, 'A': A,
        'extfin': extfin, 'share_entre': share_entre
    }


# =============================================================================
# General Equilibrium Loop
# =============================================================================

def find_equilibrium_chebyshev(lam, w_init, r_init, gamma_init,
                                cheb_approx, z_grid, prob_z,
                                delta, alpha, upsilon, sigma, beta, psi,
                                max_ge_iter=80, tol_ge=2e-3, verbose=True):
    """Find stationary equilibrium for a given lambda."""
    w, r = w_init, r_init
    gamma = gamma_init.copy()

    best_error = np.inf
    best_result = None

    for ge_iter in range(max_ge_iter):
        # Solve policy function
        gamma = solve_policy_chebyshev(
            w, r, lam, gamma, cheb_approx, z_grid, prob_z,
            delta, alpha, upsilon, sigma, beta, psi,
            max_iter=100, tol=1e-5, damping=0.7, verbose=False
        )

        # Simulate for aggregates
        agg = simulate_stationary(
            gamma, cheb_approx, z_grid, prob_z, w, r, lam,
            delta, alpha, upsilon, psi,
            N=120000, T=250, seed=12345
        )

        # Compute excess demands
        exc_K = agg['K'] - agg['A']
        exc_L = agg['L'] - (1 - agg['share_entre'])

        if verbose:
            print(f"  GE iter {ge_iter+1}: w={w:.4f}, r={r:.4f}, "
                  f"ExcL={exc_L:.4f}, ExcK={exc_K:.4f}, Y={agg['Y']:.4f}")

        total_error = abs(exc_L) + abs(exc_K)

        if total_error < best_error:
            best_error = total_error
            best_result = {
                'w': w, 'r': r, 'agg': agg.copy(), 'gamma': gamma.copy()
            }

        if abs(exc_L) < tol_ge and abs(exc_K) < tol_ge:
            if verbose:
                print(f"  Converged!")
            break

        # Adaptive price updates
        # For labor market: excess demand → raise wage
        # For capital market: excess demand (K > A) → raise interest rate
        step_w = 0.3
        step_r = 0.5

        w_new = w * (1 + step_w * exc_L)
        r_new = r + step_r * exc_K

        # Bounds
        w_new = max(0.05, min(3.0, w_new))
        r_new = max(-0.08, min(0.15, r_new))

        # Damping
        damping_ge = 0.6
        w = damping_ge * w + (1 - damping_ge) * w_new
        r = damping_ge * r + (1 - damping_ge) * r_new

    # Extract best result
    result = best_result
    agg = result['agg']

    TFP = agg['Y'] / (agg['K'] ** ALPHA * agg['L'] ** (1 - ALPHA)) if agg['K'] > 0 and agg['L'] > 0 else 0

    return {
        'w': result['w'], 'r': result['r'],
        'Y': agg['Y'], 'K': agg['K'], 'L': agg['L'], 'A': agg['A'],
        'TFP': TFP,
        'extfin': agg['extfin'],
        'ext_fin_to_gdp': agg['extfin'] / agg['Y'] if agg['Y'] > 0 else 0,
        'share_entre': agg['share_entre'],
        'K_Y': agg['K'] / agg['Y'] if agg['Y'] > 0 else 0,
        'gamma': result['gamma']
    }


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":

    total_start = time.time()

    # Chebyshev parameters
    N_NODES = 30     # Number of Chebyshev nodes
    N_POLY = 30      # Polynomial order
    A_MIN = 1e-4
    A_MAX = 3500
    A_SHIFT = 1.0    # For log-mapping

    print("\n1. Setting up model grids and Chebyshev approximation...")
    z_grid, prob_z = create_ability_grid_paper(ETA)
    n_z = len(z_grid)

    print(f"   Ability grid: {n_z} points")
    print(f"   Ability range: [{z_grid.min():.4f}, {z_grid.max():.4f}]")

    # Create Chebyshev approximator
    cheb_approx = ChebyshevApproximator(A_MIN, A_MAX, N_NODES, N_POLY, A_SHIFT)

    print(f"   Chebyshev nodes: {N_NODES}")
    print(f"   Polynomial order: {N_POLY}")
    print(f"   Asset domain: [{A_MIN}, {A_MAX}] (log-mapped)")
    print(f"   Asset nodes range: [{cheb_approx.a_nodes.min():.4f}, {cheb_approx.a_nodes.max():.4f}]")

    # Initialize policy coefficients (guess: a' = 0.95*a, slight savings)
    gamma_init = np.zeros((N_POLY, n_z))
    for j in range(n_z):
        a_init_values = 0.95 * cheb_approx.a_nodes
        gamma_init[:, j] = cheb_approx.fit_coefficients(a_init_values)

    # JIT warmup
    print("\n   Warming up JIT compilation...")
    warmup_start = time.time()
    _ = chebyshev_basis(0.5, 5)
    _ = compute_income_jit(1.0, 1.5, 1.0, 0.04, 2.0, DELTA, ALPHA, NU)
    test_gamma = np.zeros((5, 3))
    _ = euler_residual_jit(0.5, 1.0, 0, z_grid[:3], prob_z[:3], 1.0, 0.04, 2.0,
                           test_gamma, 3, 5, A_MIN, A_MAX,
                           cheb_approx.log_min, cheb_approx.log_max, A_SHIFT,
                           DELTA, ALPHA, NU, SIGMA, BETA, PSI)
    # Warmup simulation
    prob_z_cumsum = np.cumsum(prob_z[:3])
    _ = simulate_stationary_jit(test_gamma, cheb_approx.a_nodes[:5], z_grid[:3],
                                 prob_z[:3], prob_z_cumsum, 1.0, 0.04, 2.0,
                                 5, A_MIN, A_MAX, cheb_approx.log_min, cheb_approx.log_max, A_SHIFT,
                                 DELTA, ALPHA, NU, PSI, 100, 10, 123)
    print(f"   JIT warmup: {time.time() - warmup_start:.1f}s")

    # ==========================================================================
    # Figure 2 Computation
    # ==========================================================================

    print("\n" + "=" * 70)
    print("2. Computing Figure 2: Long-run Effect of Financial Frictions")
    print("   Using Chebyshev Projection Method (DEBUGGED)")
    print("=" * 70)

    lambda_values = [np.inf, 2.0, 1.75, 1.5, 1.25, 1.0]

    # Better initial guesses based on v3 results
    initial_guesses = {
        np.inf: (1.73, 0.045),
        2.0:    (1.45, -0.02),
        1.75:   (1.40, -0.035),
        1.5:    (1.35, -0.04),
        1.25:   (1.28, -0.045),
        1.0:    (1.20, -0.055),
    }

    results_list = []
    prev_w, prev_r = None, None
    prev_gamma = gamma_init.copy()

    for i, lam in enumerate(lambda_values):
        print(f"\n--- Lambda = {lam} ({i+1}/{len(lambda_values)}) ---")

        start_time = time.time()

        if i == 0:
            w_init, r_init = initial_guesses.get(lam, (1.7, 0.045))
        else:
            w_init, r_init = prev_w, prev_r

        result = find_equilibrium_chebyshev(
            lam, w_init, r_init, prev_gamma,
            cheb_approx, z_grid, prob_z,
            DELTA, ALPHA, NU, SIGMA, BETA, PSI,
            max_ge_iter=80, tol_ge=2e-3, verbose=True
        )
        result['lambda'] = lam
        results_list.append(result)

        prev_w, prev_r = result['w'], result['r']
        prev_gamma = result.get('gamma', prev_gamma)

        elapsed = time.time() - start_time
        print(f"   Result: Y={result['Y']:.4f}, ExtFin/Y={result['ext_fin_to_gdp']:.3f}, "
              f"%Entre={result['share_entre']*100:.1f}%")
        print(f"   Time: {elapsed:.1f}s")

    # ==========================================================================
    # Plotting
    # ==========================================================================

    print("\n" + "=" * 70)
    print("3. Plotting results")
    print("=" * 70)

    # Create plots directory
    plots_dir = os.path.join(os.path.dirname(__file__), 'plots')
    os.makedirs(plots_dir, exist_ok=True)

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
    ax1.set_title('GDP and TFP vs Financial Development\n(Chebyshev Projection)', fontsize=14)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0.5, 1.15])

    ax2 = axes[1]
    ax2.plot(ext_fin_ratios, interest_rates, 'g-^', linewidth=2, markersize=10)
    ax2.set_xlabel('External Finance to GDP', fontsize=14)
    ax2.set_ylabel('Interest Rate', fontsize=14)
    ax2.set_title('Equilibrium Interest Rate', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

    plt.tight_layout()
    fig2_path = os.path.join(plots_dir, 'figure2_replication_v5_chebyshev.png')
    plt.savefig(fig2_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nFigure 2 saved to '{fig2_path}'")

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
    diag_path = os.path.join(plots_dir, 'figure2_diagnostics_v5_chebyshev.png')
    plt.savefig(diag_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Diagnostic plots saved to '{diag_path}'")

    # ==========================================================================
    # Summary
    # ==========================================================================

    print("\n" + "=" * 70)
    print("4. Summary Table")
    print("=" * 70)
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

    print("\n" + "=" * 70)
    print("Key findings:")
    print("=" * 70)
    print(f"- Perfect credit (λ=∞): Y={Y_perfect:.4f}")
    print(f"- Financial autarky (λ=1): GDP={Y_normalized[-1]:.2f}, TFP={TFP_normalized[-1]:.2f}")
    print(f"- Interest rate range: [{min(interest_rates):.4f}, {max(interest_rates):.4f}]")
    print(f"- GDP loss: up to {(1-min(Y_normalized))*100:.1f}%")
    print(f"- TFP loss: up to {(1-min(TFP_normalized))*100:.1f}%")

    print("\n" + "=" * 70)
    print(f"TOTAL RUNTIME: {total_time:.1f}s ({total_time/60:.1f} min)")
    print("=" * 70)
