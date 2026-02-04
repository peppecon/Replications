#!/usr/bin/env python3
"""
Test script for the Newton-based policy solver in transition_cheb.py

This script:
1. Tests the Newton solver on a single (w, r) pair
2. Compares Newton vs Picard iteration convergence
3. Tests the full equilibrium solver
"""

import numpy as np
import time
import sys

# Add current directory to path
sys.path.insert(0, '.')

from transition_cheb import (
    # Parameters
    SIGMA, BETA, ALPHA, NU, DELTA, ETA, PSI, LAMBDA,
    A_MIN, A_MAX, Z_MIN, Z_MAX, N_CHEBY_A, N_CHEBY_Z,
    TAU_PLUS, TAU_MINUS, Q_DIST,
    # Functions
    generate_bivariate_nodes_matrix,
    solve_entrepreneur_single,
    solve_policy_newton,
    solve_policy_spectral_nested,
    solve_policy_bivariate_update,
    euler_residuals_nodist_numba,
    euler_residuals_scaled_numba,
    bivariate_eval,
    find_equilibrium_nested
)

def test_single_policy_solve():
    """Test different solver methods for a single (w, r) pair."""
    print("=" * 70)
    print("TEST 1: Solver Method Comparison")
    print("=" * 70)

    params = (DELTA, ALPHA, NU, LAMBDA, BETA, SIGMA, PSI)
    w, r = 0.8, 0.02  # Typical equilibrium-ish values

    # Setup
    nodes_a, nodes_z, T_full = generate_bivariate_nodes_matrix(A_MIN, A_MAX, Z_MIN, Z_MAX)
    T_inv = np.linalg.inv(T_full)

    # Simple quadrature using histogram grid
    nz_h = 40
    M_v = np.concatenate([np.linspace(0.0, 0.998, 38), [0.999, 0.9995]])
    z_h = (1 - M_v)**(-1/ETA)
    pr_z = np.zeros(nz_h)
    pr_z[0] = M_v[0]
    pr_z[1:] = M_v[1:] - M_v[:-1]
    pr_z /= pr_z.sum()
    z_quad = z_h.copy()
    z_w = pr_z.copy()

    print(f"\nTest prices: w={w}, r={r}")
    print(f"Chebyshev nodes: {N_CHEBY_A} x {N_CHEBY_Z} = {N_CHEBY_A * N_CHEBY_Z} coefficients")

    results = {}

    # Test 1: Picard with Anderson Acceleration
    print("\n--- Method 1: Picard with Anderson Acceleration (100 iters) ---")
    t0 = time.time()
    coeffs_1, success_1 = solve_policy_newton(
        params, w, r, nodes_a, nodes_z, T_full, T_inv, z_quad, z_w,
        coeffs_init=None, tol=1e-8, max_iter=100, verbose=True,
        method='picard_accelerated'
    )
    time_1 = time.time() - t0
    scaled_res_1 = np.max(np.abs(euler_residuals_scaled_numba(
        coeffs_1, nodes_a, nodes_z, BETA, SIGMA, PSI, w, r, LAMBDA, DELTA, ALPHA, NU,
        A_MIN, A_MAX, Z_MIN, Z_MAX, z_quad, z_w
    )))
    results['picard_accel'] = (coeffs_1, success_1, time_1, scaled_res_1)
    print(f"Time: {time_1:.2f}s, Success: {success_1}, Scaled residual: {scaled_res_1:.2e}")

    # Test 2: Least Squares (Levenberg-Marquardt)
    print("\n--- Method 2: Least Squares (Levenberg-Marquardt) ---")
    t0 = time.time()
    coeffs_2, success_2 = solve_policy_newton(
        params, w, r, nodes_a, nodes_z, T_full, T_inv, z_quad, z_w,
        coeffs_init=None, tol=1e-8, max_iter=100, verbose=True,
        n_picard_warmup=20, method='least_squares'
    )
    time_2 = time.time() - t0
    scaled_res_2 = np.max(np.abs(euler_residuals_scaled_numba(
        coeffs_2, nodes_a, nodes_z, BETA, SIGMA, PSI, w, r, LAMBDA, DELTA, ALPHA, NU,
        A_MIN, A_MAX, Z_MIN, Z_MAX, z_quad, z_w
    )))
    results['least_squares'] = (coeffs_2, success_2, time_2, scaled_res_2)
    print(f"Time: {time_2:.2f}s, Success: {success_2}, Scaled residual: {scaled_res_2:.2e}")

    # Test 3: Original Picard (for comparison)
    print("\n--- Method 3: Original Picard (100 iterations, fixed damping) ---")
    t0 = time.time()
    coeffs_3 = solve_policy_spectral_nested(
        params, w, r, nodes_a, nodes_z, T_inv, z_quad, z_w,
        coeffs_init=None, diag_out=None, key="test", show_plot=False
    )
    time_3 = time.time() - t0
    scaled_res_3 = np.max(np.abs(euler_residuals_scaled_numba(
        coeffs_3, nodes_a, nodes_z, BETA, SIGMA, PSI, w, r, LAMBDA, DELTA, ALPHA, NU,
        A_MIN, A_MAX, Z_MIN, Z_MAX, z_quad, z_w
    )))
    success_3 = scaled_res_3 < 0.05
    results['original_picard'] = (coeffs_3, success_3, time_3, scaled_res_3)
    print(f"Time: {time_3:.2f}s, Success: {success_3}, Scaled residual: {scaled_res_3:.2e}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Method':<30} {'Time (s)':<12} {'Success':<10} {'|R_scaled|':<12}")
    print("-" * 70)
    for name, (_, success, time_val, res) in results.items():
        print(f"{name:<30} {time_val:<12.2f} {str(success):<10} {res:<12.2e}")

    # Find best method
    best_method = min(results.keys(), key=lambda k: results[k][3])
    print(f"\nBest method: {best_method} (lowest scaled residual)")

    # Return success if any method converged well
    best_res = results[best_method][3]
    return best_res < 0.05


def test_equilibrium_solver():
    """Test full equilibrium solver with Newton."""
    print("\n" + "=" * 70)
    print("TEST 2: Full Equilibrium Solver (Post-Reform, Newton)")
    print("=" * 70)

    params = (DELTA, ALPHA, NU, LAMBDA, BETA, SIGMA, PSI)

    t0 = time.time()
    result = find_equilibrium_nested(params, distortions=False, diag_out=None,
                                     label="test_post", use_newton=True)
    time_total = time.time() - t0

    print("\n" + "-" * 50)
    print("EQUILIBRIUM RESULTS:")
    print("-" * 50)
    print(f"  Wage (w):        {result['w']:.6f}")
    print(f"  Interest rate:   {result['r']:.6f}")
    print(f"  Output (Y):      {result['Y']:.4f}")
    print(f"  Capital (K):     {result['K']:.4f}")
    print(f"  TFP:             {result['TFP']:.4f}")
    print(f"  Entre share:     {result['share_entre']:.4f}")
    print(f"  Ext.Fin/Y:       {result['ExtFin_Y']:.4f}")
    print(f"\n  Total time:      {time_total:.1f}s")

    # Sanity checks
    checks_passed = True
    if result['w'] <= 0 or result['w'] > 5:
        print("  [FAIL] Wage out of reasonable range")
        checks_passed = False
    if result['r'] < -0.2 or result['r'] > 0.2:
        print("  [FAIL] Interest rate out of reasonable range")
        checks_passed = False
    if result['Y'] <= 0:
        print("  [FAIL] Output should be positive")
        checks_passed = False
    if result['share_entre'] < 0 or result['share_entre'] > 1:
        print("  [FAIL] Entrepreneur share should be in [0,1]")
        checks_passed = False

    if checks_passed:
        print("\n  [PASS] All sanity checks passed!")

    return checks_passed


def test_policy_monotonicity():
    """Check that the policy function is monotonic in assets."""
    print("\n" + "=" * 70)
    print("TEST 3: Policy Function Monotonicity Check")
    print("=" * 70)

    params = (DELTA, ALPHA, NU, LAMBDA, BETA, SIGMA, PSI)
    w, r = 0.8, 0.02

    nodes_a, nodes_z, T_full = generate_bivariate_nodes_matrix(A_MIN, A_MAX, Z_MIN, Z_MAX)
    T_inv = np.linalg.inv(T_full)

    nz_h = 40
    M_v = np.concatenate([np.linspace(0.0, 0.998, 38), [0.999, 0.9995]])
    z_h = (1 - M_v)**(-1/ETA)
    pr_z = np.zeros(nz_h)
    pr_z[0] = M_v[0]
    pr_z[1:] = M_v[1:] - M_v[:-1]
    pr_z /= pr_z.sum()

    coeffs, _ = solve_policy_newton(
        params, w, r, nodes_a, nodes_z, T_full, T_inv, z_h, pr_z,
        coeffs_init=None, tol=1e-8, verbose=False,
        n_picard_warmup=10, use_scaling=True
    )

    # Check monotonicity along asset dimension for multiple z values
    a_test = np.linspace(A_MIN, A_MAX, 100)
    z_indices = [0, len(z_h)//4, len(z_h)//2, 3*len(z_h)//4, len(z_h)-1]

    all_monotonic = True
    for iz in z_indices:
        z_test = z_h[iz]
        policy_vals = np.array([bivariate_eval(a, z_test, coeffs, A_MIN, A_MAX, Z_MIN, Z_MAX) for a in a_test])

        # Check if non-decreasing
        diffs = np.diff(policy_vals)
        n_violations = np.sum(diffs < -1e-6)

        if n_violations == 0:
            print(f"  [PASS] Policy is monotonic in assets (z={z_test:.2f})")
        else:
            print(f"  [WARN] {n_violations} monotonicity violations at z={z_test:.2f}, min diff: {np.min(diffs):.2e}")
            all_monotonic = False

    return all_monotonic


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("NEWTON SOLVER TEST SUITE")
    print("=" * 70)

    results = []

    # Run tests
    results.append(("Single Policy Solve", test_single_policy_solve()))
    results.append(("Policy Monotonicity", test_policy_monotonicity()))

    # Only run full equilibrium test if requested (slow)
    if len(sys.argv) > 1 and sys.argv[1] == "--full":
        results.append(("Full Equilibrium", test_equilibrium_solver()))
    else:
        print("\n[INFO] Skipping full equilibrium test. Run with --full to include.")

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: [{status}]")
        all_passed = all_passed and passed

    print("=" * 70)
    if all_passed:
        print("All tests passed!")
    else:
        print("Some tests failed.")
    print("=" * 70)
