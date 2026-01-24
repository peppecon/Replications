"""
Replication of Buera & Shin (2010) - Financial Frictions and the Persistence of History

This code replicates:
1. The baseline stationary equilibrium (Section 3.1)
2. Figure 2: Long-run effect of financial frictions (Section 3.2)

Following Appendix B.1 for the numerical algorithm.

IMPROVED VERSION: Based on Matlab VFI Toolkit implementation:
- Uses paper's exact ability discretization (40 points)
- Larger asset grid with curvature scaling (500 points, range [1e-6, 4000])
- Analytical stationary distribution computation (not simulation)
- Cleaner entrepreneur profit formula
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
from scipy import sparse
from scipy.sparse.linalg import eigs
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Model Parameters (Section 3.1.1)
# =============================================================================

# Preferences
SIGMA = 1.5      # Risk aversion (CRRA)
BETA = 0.904     # Discount factor (calibrated to match r = 4.5%)

# Technology
ALPHA = 0.33     # Capital share in variable factors
NU = 0.21        # Entrepreneur's share (span-of-control = 1 - nu = 0.79)
DELTA = 0.06     # Depreciation rate

# Entrepreneurial ability shocks
ETA = 4.15       # Pareto tail parameter
PSI = 0.894      # Probability of retaining ability

print("=" * 60)
print("Buera-Shin (2010) Replication - IMPROVED VERSION")
print("Financial Frictions and the Persistence of History")
print("=" * 60)


# =============================================================================
# Ability Grid - Paper's Exact Discretization (Page 235)
# =============================================================================

def create_ability_grid_paper(eta):
    """
    Create the ability grid following the paper's exact method (page 235):

    "The entrepreneurial ability e is assumed to be a truncated and discretized version
    of a Pareto distribution whose probability density is eta*e^-(eta+1) for e>=1.
    We discretize the support of the ability distribution into 40 grid points {e_1,...,e_40}.
    Denoting the cdf of the original Pareto distribution by M(e)=1-e^(-eta), we choose
    e_1 and e_38 such that M(e_1)=0.633 and M(e_38)=0.998. Indexing the grid points by j,
    we construct e_j to be equidistant from j=1,...,38. The two largest values on the
    grid are given by e_39 and e_40 which satisfy M(e_39)=0.999 and M(e_40)=0.9995.
    Finally the corresponding probability mass w(e_j) for 2<=j<=40 is given by
    [M(e_j)-M(e_{j-1})]/M(e_40) and w(e_1)=M(e_1)/M(e_40)."

    Pareto CDF: M(e) = 1 - e^(-eta) => inverse: e = (1-M)^(-1/eta)
    """
    n_z = 40

    # CDF values for the grid
    M_values = np.zeros(n_z)
    M_values[:38] = np.linspace(0.633, 0.998, 38)
    M_values[38] = 0.999
    M_values[39] = 0.9995

    # Inverse CDF: e = (1 - M)^(-1/eta)
    # But paper says CDF is M(e) = 1 - e^(-eta), so e = (1-M)^(-1/eta)
    z_grid = (1 - M_values) ** (-1/eta)

    # Probability masses
    prob_z = np.zeros(n_z)
    prob_z[0] = M_values[0] / M_values[-1]
    for j in range(1, n_z):
        prob_z[j] = (M_values[j] - M_values[j-1]) / M_values[-1]

    # Ensure sums to 1
    prob_z = prob_z / prob_z.sum()

    return z_grid, prob_z


def create_asset_grid(n_a, a_min, a_max, a_scale=2):
    """
    Create asset grid with curvature scaling (more points near zero).
    Following Matlab: a_grid = a_min + (a_max-a_min) * linspace(0,1,n_a)^a_scale
    """
    a_grid = a_min + (a_max - a_min) * np.linspace(0, 1, n_a) ** a_scale
    a_grid[0] = max(a_grid[0], 1e-6)  # Avoid zero
    return a_grid


# =============================================================================
# Core model functions (JIT compiled for speed)
# =============================================================================

@njit(cache=True)
def utility(c, sigma):
    """CRRA utility function"""
    if c <= 1e-10:
        return -1e10
    if abs(sigma - 1.0) < 1e-6:
        return np.log(c)
    else:
        return (c ** (1 - sigma) - 1) / (1 - sigma)


@njit(cache=True)
def solve_entrepreneur(a, z, w, r, lam, delta, alpha, upsilon):
    """
    Solve the entrepreneur's static maximization problem.

    Following Matlab's solve_entre.m exactly:
    - k1 = unconstrained optimal capital
    - kstar = min(k1, lambda*a) - collateral constraint
    - lstar = optimal labor given kstar
    - profit = output - wages - capital costs

    Returns: (profit, kstar, lstar)
    """
    # Ensure positive rental rate
    rental = max(r + delta, 1e-8)
    wage = max(w, 1e-8)

    # Compute unconstrained optimal capital k1
    # From the FOCs (see Matlab code):
    aux1 = (1/(rental)) * alpha * (1-upsilon) * z
    aux2 = (1/wage) * (1-alpha) * (1-upsilon) * z
    exponent = 1 - (1-alpha)*(1-upsilon)
    inside = (aux1 ** exponent) * (aux2 ** ((1-alpha)*(1-upsilon)))
    k1 = inside ** (1/upsilon)

    # Apply collateral constraint
    kstar = min(k1, lam * a)

    # Optimal labor given kstar
    inside_lab = (1/wage) * (1-alpha) * (1-upsilon) * z * (kstar ** (alpha*(1-upsilon)))
    denom = 1 - (1-alpha)*(1-upsilon)
    lstar = inside_lab ** (1/denom)

    # Compute profit
    output = z * ((kstar ** alpha) * (lstar ** (1-alpha))) ** (1-upsilon)
    profit = output - wage * lstar - rental * kstar

    return profit, kstar, lstar


@njit(cache=True)
def compute_output(kstar, lstar, z, alpha, upsilon):
    """Compute entrepreneurial output"""
    if kstar <= 0 or lstar <= 0:
        return 0.0
    return z * ((kstar ** alpha) * (lstar ** (1-alpha))) ** (1-upsilon)


# =============================================================================
# Value Function Iteration
# =============================================================================

@njit(cache=True)
def bellman_operator(V, a_grid, z_grid, prob_z, w, r,
                     delta, alpha, upsilon, lam, beta, sigma, psi):
    """
    One step of the Bellman operator.
    Uses the exact transition matrix: pi_z = psi*I + (1-psi)*ones*prob_z'
    """
    n_a = len(a_grid)
    n_z = len(z_grid)

    V_new = np.zeros((n_a, n_z))
    policy_a_idx = np.zeros((n_a, n_z), dtype=np.int64)
    policy_occ = np.zeros((n_a, n_z))  # 1 = entrepreneur, 0 = worker

    for i_z in range(n_z):
        z = z_grid[i_z]

        # Expected continuation value
        # E[V(a', z')] = psi * V(a', z) + (1-psi) * sum_z'' prob(z'') * V(a', z'')
        EV = np.zeros(n_a)
        for i_a_prime in range(n_a):
            ev_same = psi * V[i_a_prime, i_z]
            ev_new = 0.0
            for i_z_new in range(n_z):
                ev_new += prob_z[i_z_new] * V[i_a_prime, i_z_new]
            ev_new *= (1 - psi)
            EV[i_a_prime] = ev_same + ev_new

        for i_a in range(n_a):
            a = a_grid[i_a]

            # Compute profit from entrepreneurship
            profit, kstar, lstar = solve_entrepreneur(a, z, w, r, lam, delta, alpha, upsilon)

            # Occupational choice: entrepreneur if profit > wage
            if profit > w:
                income = profit + (1 + r) * a
                is_entrep = 1.0
            else:
                income = w + (1 + r) * a
                is_entrep = 0.0

            # Find optimal savings
            best_val = -1e10
            best_idx = 0

            for i_a_prime in range(n_a):
                a_prime = a_grid[i_a_prime]
                c = income - a_prime

                if c > 1e-10:
                    val = utility(c, sigma) + beta * EV[i_a_prime]
                    if val > best_val:
                        best_val = val
                        best_idx = i_a_prime
                else:
                    break  # a_grid is increasing, so no need to check further

            V_new[i_a, i_z] = best_val
            policy_a_idx[i_a, i_z] = best_idx
            policy_occ[i_a, i_z] = is_entrep

    return V_new, policy_a_idx, policy_occ


def solve_value_function(a_grid, z_grid, prob_z, w, r, params,
                        tol=1e-5, max_iter=1000, verbose=False):
    """
    Solve for the value function using value function iteration
    """
    delta, alpha, upsilon, lam, beta, sigma, psi = params

    n_a = len(a_grid)
    n_z = len(z_grid)

    # Initial guess
    V = np.zeros((n_a, n_z))
    for i_a in range(n_a):
        for i_z in range(n_z):
            c_guess = max(w + r * a_grid[i_a], 0.01)
            V[i_a, i_z] = utility(c_guess, sigma) / (1 - beta)

    for iteration in range(max_iter):
        V_new, policy_a_idx, policy_occ = bellman_operator(
            V, a_grid, z_grid, prob_z, w, r, delta, alpha, upsilon,
            lam, beta, sigma, psi
        )

        diff = np.max(np.abs(V_new - V))
        V = V_new

        if diff < tol:
            if verbose:
                print(f"    VFI converged in {iteration+1} iterations (diff={diff:.2e})")
            break

    return V, policy_a_idx, policy_occ


# =============================================================================
# Stationary Distribution - Analytical Method (not simulation)
# =============================================================================

def compute_stationary_distribution(policy_a_idx, a_grid, z_grid, prob_z, psi):
    """
    Compute stationary distribution analytically using the Eigenvector method.

    Transition matrix Q where Q[s', s] = Pr(s' | s)
    State s = (a, z) with s = i_a * n_z + i_z
    """
    n_a = len(a_grid)
    n_z = len(z_grid)
    n_states = n_a * n_z

    # Build transition matrix (sparse for efficiency)
    # Q[s', s] = Pr(transition from s to s')
    rows = []
    cols = []
    data = []

    for i_a in range(n_a):
        for i_z in range(n_z):
            s = i_a * n_z + i_z
            i_a_prime = policy_a_idx[i_a, i_z]

            for i_z_prime in range(n_z):
                # Transition probability for z
                if i_z_prime == i_z:
                    p_z = psi + (1 - psi) * prob_z[i_z_prime]
                else:
                    p_z = (1 - psi) * prob_z[i_z_prime]

                if p_z > 1e-12:
                    s_prime = i_a_prime * n_z + i_z_prime
                    rows.append(s_prime)
                    cols.append(s)
                    data.append(p_z)

    Q = sparse.csr_matrix((data, (rows, cols)), shape=(n_states, n_states))

    # Find stationary distribution: solve Q' * mu = mu
    # This is equivalent to finding the eigenvector with eigenvalue 1
    try:
        eigenvalues, eigenvectors = eigs(Q, k=1, which='LM')
        mu = np.abs(eigenvectors[:, 0].real)
        mu = mu / mu.sum()
    except:
        # Fallback: power iteration
        mu = np.ones(n_states) / n_states
        for _ in range(1000):
            mu_new = Q @ mu
            mu_new = mu_new / mu_new.sum()
            if np.max(np.abs(mu_new - mu)) < 1e-10:
                break
            mu = mu_new
        mu = mu_new

    # Reshape to (n_a, n_z)
    dist = mu.reshape((n_a, n_z))

    return dist


# =============================================================================
# Aggregate Variables
# =============================================================================

@njit(cache=True)
def compute_aggregates_on_grid(a_grid, z_grid, policy_occ, w, r, lam, delta, alpha, upsilon):
    """
    Compute K, L, Y, extfin for each grid point
    """
    n_a = len(a_grid)
    n_z = len(z_grid)

    K_grid = np.zeros((n_a, n_z))
    L_grid = np.zeros((n_a, n_z))
    Y_grid = np.zeros((n_a, n_z))
    extfin_grid = np.zeros((n_a, n_z))

    for i_a in range(n_a):
        for i_z in range(n_z):
            a = a_grid[i_a]
            z = z_grid[i_z]

            profit, kstar, lstar = solve_entrepreneur(a, z, w, r, lam, delta, alpha, upsilon)

            if policy_occ[i_a, i_z] > 0.5:  # Entrepreneur
                K_grid[i_a, i_z] = kstar
                L_grid[i_a, i_z] = lstar
                Y_grid[i_a, i_z] = compute_output(kstar, lstar, z, alpha, upsilon)
                extfin_grid[i_a, i_z] = max(0.0, kstar - a)

    return K_grid, L_grid, Y_grid, extfin_grid


def compute_aggregate_moments(dist, a_grid, z_grid, policy_occ, w, r, lam, delta, alpha, upsilon):
    """
    Compute aggregate moments given the stationary distribution
    """
    n_a = len(a_grid)
    n_z = len(z_grid)

    K_grid, L_grid, Y_grid, extfin_grid = compute_aggregates_on_grid(
        a_grid, z_grid, policy_occ, w, r, lam, delta, alpha, upsilon
    )

    # Aggregate by integrating over distribution
    K = np.sum(dist * K_grid)
    L = np.sum(dist * L_grid)
    Y = np.sum(dist * Y_grid)
    extfin = np.sum(dist * extfin_grid)

    # Total assets
    A_grid = np.outer(a_grid, np.ones(n_z))
    A = np.sum(dist * A_grid)

    # Share of entrepreneurs
    share_entre = np.sum(dist * policy_occ)

    return {
        'K': K, 'L': L, 'Y': Y, 'A': A,
        'extfin': extfin, 'share_entre': share_entre
    }


# =============================================================================
# Equilibrium solver
# =============================================================================

def find_equilibrium(a_grid, z_grid, prob_z, params,
                    w_init=0.172, r_init=0.0476,
                    max_iter=50, tol=1e-3, verbose=True):
    """
    Find stationary equilibrium using iterative price adjustment.

    GE conditions:
    1. Capital market: K = A (capital demand = asset supply)
    2. Labor market: L = 1 - share_entre (labor demand = worker supply)
    """
    delta, alpha, upsilon, lam, beta, sigma, psi = params

    w = w_init
    r = r_init

    best_error = np.inf
    best_result = None

    for iteration in range(max_iter):
        # Solve value function
        V, policy_a_idx, policy_occ = solve_value_function(
            a_grid, z_grid, prob_z, w, r, params, verbose=False
        )

        # Compute stationary distribution
        dist = compute_stationary_distribution(policy_a_idx, a_grid, z_grid, prob_z, psi)

        # Compute aggregates
        agg = compute_aggregate_moments(
            dist, a_grid, z_grid, policy_occ, w, r, lam, delta, alpha, upsilon
        )

        # GE conditions (excess demands)
        exc_K = agg['K'] - agg['A']  # Capital demand - supply
        exc_L = agg['L'] - (1 - agg['share_entre'])  # Labor demand - worker supply

        if verbose:
            print(f"  Iter {iteration+1}: w={w:.4f}, r={r:.4f}, "
                  f"ExcL={exc_L:.4f}, ExcK={exc_K:.4f}")

        total_error = abs(exc_L) + abs(exc_K)

        if total_error < best_error:
            best_error = total_error
            best_result = {
                'w': w, 'r': r,
                'agg': agg.copy(),
                'dist': dist.copy(),
                'policy_occ': policy_occ.copy()
            }

        # Check convergence
        if abs(exc_L) < tol and abs(exc_K) < tol:
            if verbose:
                print(f"  Converged!")
            break

        # Update prices using gradient-based adjustment
        # Excess labor demand -> raise wage
        # Excess capital demand -> raise interest rate
        w_new = w * (1 + 0.3 * exc_L)
        r_new = r + 0.01 * exc_K

        # Bounds
        w_new = max(0.01, min(2.0, w_new))
        r_new = max(-0.06, min(0.12, r_new))

        # Damped update
        w = 0.5 * w + 0.5 * w_new
        r = 0.5 * r + 0.5 * r_new

    # Use best result
    result = best_result
    w, r = result['w'], result['r']
    agg = result['agg']

    # Compute TFP: Y / (K^(1/3) * L^(2/3))
    if agg['K'] > 0 and agg['L'] > 0:
        TFP = agg['Y'] / (agg['K'] ** (1/3) * agg['L'] ** (2/3))
    else:
        TFP = 0

    return {
        'w': w, 'r': r,
        'Y': agg['Y'],
        'K': agg['K'],
        'L': agg['L'],
        'A': agg['A'],
        'TFP': TFP,
        'extfin': agg['extfin'],
        'ext_fin_to_gdp': agg['extfin'] / agg['Y'] if agg['Y'] > 0 else 0,
        'share_entre': agg['share_entre'],
        'K_Y': agg['K'] / agg['Y'] if agg['Y'] > 0 else 0
    }


# =============================================================================
# Main execution
# =============================================================================

if __name__ == "__main__":

    # Grid parameters (following Matlab implementation)
    n_a = 501        # Number of asset grid points (Matlab uses 1001)
    a_min = 1e-6     # Lower bound
    a_max = 4000     # Upper bound (same as Matlab)
    a_scale = 2      # Curvature (same as Matlab)

    # Create grids
    print("\n1. Setting up model grids...")
    z_grid, prob_z = create_ability_grid_paper(ETA)
    a_grid = create_asset_grid(n_a, a_min, a_max, a_scale)
    n_z = len(z_grid)

    print(f"   Ability grid: {n_z} points (paper's exact discretization)")
    print(f"   Ability range: [{z_grid.min():.4f}, {z_grid.max():.4f}]")
    print(f"   Asset grid: {n_a} points, range [{a_grid.min():.6f}, {a_grid.max():.2f}]")
    print(f"   Asset grid curvature: {a_scale}")

    # ==========================================================================
    # Figure 2: Long-run effect of financial frictions
    # ==========================================================================

    print("\n" + "=" * 60)
    print("2. Computing Figure 2: Long-run Effect of Financial Frictions")
    print("=" * 60)

    # Range of lambda values (following Matlab)
    lambda_values = [np.inf, 1000.0, 100.0, 10.0, 2.0, 1.75, 1.5, 1.25, 1.0]

    # Initial guesses for prices (from Matlab code)
    # For lambda=inf: r=0.0472, w=0.171
    initial_guesses = {
        np.inf: (0.171, 0.0472),
        2.0:    (0.15, 0.03),
        1.75:   (0.14, 0.02),
        1.5:    (0.13, 0.01),
        1.25:   (0.12, 0.00),
        1.0:    (0.10, -0.02),
    }

    results_list = []
    prev_w, prev_r = None, None

    for i, lam in enumerate(lambda_values):
        print(f"\n--- Lambda = {lam} ({i+1}/{len(lambda_values)}) ---")

        # Pack parameters
        params = (DELTA, ALPHA, NU, lam, BETA, SIGMA, PSI)

        # Get initial guess (use previous result or default)
        if i == 0:
            w_init, r_init = initial_guesses.get(lam, (0.17, 0.05))
        else:
            w_init, r_init = prev_w, prev_r

        result = find_equilibrium(
            a_grid, z_grid, prob_z, params,
            w_init=w_init, r_init=r_init,
            max_iter=50, tol=1e-3, verbose=True
        )
        result['lambda'] = lam
        results_list.append(result)

        prev_w, prev_r = result['w'], result['r']

        print(f"   Result: Y={result['Y']:.4f}, K/Y={result['K_Y']:.3f}, "
              f"ExtFin/Y={result['ext_fin_to_gdp']:.3f}, %Entre={result['share_entre']*100:.1f}%")

    # ==========================================================================
    # Plot results
    # ==========================================================================

    print("\n" + "=" * 60)
    print("3. Plotting results")
    print("=" * 60)

    # Extract data for plotting
    lambdas = [r['lambda'] for r in results_list]
    ext_fin_ratios = [r['ext_fin_to_gdp'] for r in results_list]
    Ys = [r['Y'] for r in results_list]
    TFPs = [r['TFP'] for r in results_list]
    interest_rates = [r['r'] for r in results_list]

    # Normalize by perfect credit (lambda = inf, first entry)
    Y_perfect = Ys[0]
    TFP_perfect = TFPs[0]

    Y_normalized = [y / Y_perfect if Y_perfect > 0 else 0 for y in Ys]
    TFP_normalized = [t / TFP_perfect if TFP_perfect > 0 else 0 for t in TFPs]

    # Create Figure 2
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left panel: GDP and TFP vs External Finance/GDP
    ax1 = axes[0]
    ax1.plot(ext_fin_ratios, Y_normalized, 'b-o', label='GDP', linewidth=2, markersize=10)
    ax1.plot(ext_fin_ratios, TFP_normalized, 'r--s', label='TFP', linewidth=2, markersize=10)
    ax1.set_xlabel('External Finance to GDP', fontsize=14)
    ax1.set_ylabel('Relative to Perfect Credit (λ=∞)', fontsize=14)
    ax1.set_title('GDP and TFP vs Financial Development', fontsize=16)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0.5, 1.15])

    # Right panel: Interest Rate
    ax2 = axes[1]
    ax2.plot(ext_fin_ratios, interest_rates, 'g-^', linewidth=2, markersize=10)
    ax2.set_xlabel('External Finance to GDP', fontsize=14)
    ax2.set_ylabel('Interest Rate', fontsize=14)
    ax2.set_title('Equilibrium Interest Rate', fontsize=16)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

    plt.tight_layout()
    plt.savefig('/home/nuagsire/Dropbox/PhD Bocconi/Replications/BueraShin/figure2_replication.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("\nFigure 2 saved to 'figure2_replication.png'")

    # ==========================================================================
    # Additional diagnostic plots
    # ==========================================================================

    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Lambda vs External Finance
    ax = axes2[0, 0]
    lam_plot = [l if l != np.inf else 100 for l in lambdas]
    ax.plot(lam_plot, ext_fin_ratios, 'ko-', linewidth=2, markersize=10)
    ax.set_xlabel('λ (Financial Friction Parameter)', fontsize=12)
    ax.set_ylabel('External Finance / GDP', fontsize=12)
    ax.set_title('Financial Development vs λ', fontsize=14)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    # Plot 2: GDP and TFP vs Lambda
    ax = axes2[0, 1]
    ax.plot(lam_plot, Y_normalized, 'b-o', label='GDP', linewidth=2, markersize=10)
    ax.plot(lam_plot, TFP_normalized, 'r--s', label='TFP', linewidth=2, markersize=10)
    ax.set_xlabel('λ', fontsize=12)
    ax.set_ylabel('Relative to Perfect Credit', fontsize=12)
    ax.set_title('Output and Productivity vs λ', fontsize=14)
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Wages
    wages = [r['w'] for r in results_list]
    ax = axes2[1, 0]
    ax.plot(ext_fin_ratios, wages, 'm-d', linewidth=2, markersize=10)
    ax.set_xlabel('External Finance / GDP', fontsize=12)
    ax.set_ylabel('Wage', fontsize=12)
    ax.set_title('Equilibrium Wage', fontsize=14)
    ax.grid(True, alpha=0.3)

    # Plot 4: Fraction entrepreneurs
    frac_entrep = [r['share_entre'] for r in results_list]
    ax = axes2[1, 1]
    ax.plot(ext_fin_ratios, [f*100 for f in frac_entrep], 'c-p', linewidth=2, markersize=10)
    ax.set_xlabel('External Finance / GDP', fontsize=12)
    ax.set_ylabel('Entrepreneurs (%)', fontsize=12)
    ax.set_title('Fraction of Entrepreneurs', fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/nuagsire/Dropbox/PhD Bocconi/Replications/BueraShin/figure2_diagnostics.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Diagnostic plots saved to 'figure2_diagnostics.png'")

    # ==========================================================================
    # Summary table
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

    print("\n" + "=" * 60)
    print("Key findings (compare with paper's Figure 2):")
    print("=" * 60)
    print(f"- Perfect credit (λ=∞): Y={Y_perfect:.4f}, ExtFin/GDP={ext_fin_ratios[0]:.4f}")
    print(f"- Financial autarky (λ=1): GDP={Y_normalized[-1]:.2f}, TFP={TFP_normalized[-1]:.2f} of perfect credit")
    print(f"- Interest rate range: [{min(interest_rates):.4f}, {max(interest_rates):.4f}]")
    print(f"- GDP loss from frictions: up to {(1-min(Y_normalized))*100:.1f}%")
    print(f"- TFP loss from frictions: up to {(1-min(TFP_normalized))*100:.1f}%")

    print("\n" + "=" * 60)
    print("Replication complete!")
    print("=" * 60)
