#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simplified Neoclassical Growth Model - Deterministic Version
Only capital k as state variable (no productivity shocks)
Uses Chebyshev polynomial projection method
DIRECT APPROXIMATION: Approximates c(k) directly without log-exp transformation
"""

import numpy as np
from scipy import optimize as opt
import sys
import os

# Add parent directory to path to import functions_library
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)
from functions_library import *

# Graphics imports
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['figure.figsize'] = (9, 6)
from matplotlib import cm
from mpl_toolkits.mplot3d.axes3d import Axes3D

# ============================================================================
# MODEL PARAMETERS
# ============================================================================
β = 0.99     # Discount factor
α = 0.33      # Capital share
δ = 0.025     # Depreciation rate
γ = 1         # Risk aversion (CRRA parameter)

# Damping parameter for iteration
dampening = 1  # Damping parameter for consumption updates

# ============================================================================
# CHEBYSHEV APPROXIMATION SETUP
# ============================================================================
n_k = 10      # Number of Chebyshev nodes for capital
p_k = n_k     # Polynomial order (typically equals number of nodes)

# Calculate steady state capital
k_ss = (β * α / (1 - β * (1 - δ)))**(1 / (1 - α))

# Capital domain bounds
k_low = 0.5 * k_ss
k_high = 1.5 * k_ss

# Get Chebyshev nodes in [-1, 1] domain
cheb_nodes_k = Chebyshev_Nodes(n_k).ravel()

# Map Chebyshev nodes to economic domain [k_low, k_high]
k_grid = Change_Variable_Fromcheb(k_low, k_high, cheb_nodes_k)

# Number of coefficients
n_coeffs = p_k

# ============================================================================
# CONSUMPTION POLICY FUNCTION (Chebyshev approximation - DIRECT)
# ============================================================================
def c_cheb(k, gamma, k_low, k_high, p_k):
    """
    Returns Chebyshev approximation of consumption as a function of capital k
    DIRECT APPROXIMATION: c(k) = gamma @ T_k (no log-exp transformation)
    
    Parameters:
    -----------
    k : scalar
        Capital value
    gamma : array
        Chebyshev coefficients (length p_k) - approximates c(k) directly
    k_low, k_high : scalars
        Capital domain bounds
    p_k : int
        Polynomial order
    
    Returns:
    --------
    c : scalar
        Consumption value (ensured to be positive)
    """
    # Transform to Chebyshev domain [-1, 1]
    k_cheb = Change_Variable_Tocheb(k_low, k_high, k)
    
    # Evaluate Chebyshev polynomials at k_cheb
    # T has shape (p_k, 1) - each row is a polynomial order
    T_k = Chebyshev_Polynomials_Recursion_mv(np.array([k_cheb]), p_k)
    
    # Direct approximation: c(k) = gamma @ T_k
    # gamma is (p_k,), T_k is (p_k, 1), so gamma @ T_k gives scalar
    c = float(gamma @ T_k)
    
    # Ensure consumption is positive (enforce lower bound)
    c = max(c, 1e-10)
    
    return c

# ============================================================================
# COMPUTE EULER ERRORS AND UPDATE CONSUMPTION
# ============================================================================
def compute_euler_errors_and_update(c_values, k_grid, k_low, k_high, p_k, gamma):
    """
    Computes Euler errors and returns updated consumption values
    
    Parameters:
    -----------
    c_values : array
        Current consumption values at grid points
    k_grid : array
        Capital grid points
    k_low, k_high : scalars
        Capital domain bounds
    p_k : int
        Polynomial order
    gamma : array
        Current Chebyshev coefficients (used to evaluate c at k_prime)
    
    Returns:
    --------
    c_new : array
        Updated consumption values
    euler_errors : array
        Euler errors at each grid point
    """
    n = len(k_grid)
    c_new = np.zeros(n)
    euler_errors = np.zeros(n)
    
    for i_k in range(n):
        k = k_grid[i_k]
        c = c_values[i_k]
        
        # Compute next period capital from resource constraint
        k_prime = (1 - δ) * k + k**α - c
        
        # Ensure k_prime is positive and within reasonable bounds
        if k_prime <= 0 or k_prime > 2 * k_high:
            # If invalid, keep current consumption
            c_new[i_k] = c
            euler_errors[i_k] = 1e10
            continue
        
        # Evaluate consumption at k_prime using Chebyshev approximation
        c_prime = c_cheb(k_prime, gamma, k_low, k_high, p_k)
        
        # Euler equation: 1 = β * (c'/c)^(-γ) * R'
        R_prime = α * k_prime**(α - 1) + (1 - δ)
        consumption_ratio = c_prime / c
        
        # Euler error: 1 - β * (c'/c)^(-γ) * R'
        euler_error = 1 - β * (consumption_ratio)**(-γ) * R_prime
        euler_errors[i_k] = euler_error
        
        # Update consumption using the Euler equation structure
        # From Euler: (c'/c)^(-γ) = 1/(β * R')
        # So: c'/c = (β * R')^(1/γ)
        # This gives: c_target = c' / (β * R')^(1/γ)
        # But c' depends on c through k', so we use a damped update
        if R_prime > 0 and c_prime > 1e-10:
            # Compute target consumption from Euler equation
            c_target = c_prime / (β * R_prime)**(1/γ)
            # Ensure target is reasonable
            c_target = max(c_target, 1e-10)
            c_target = min(c_target, k**α + (1-δ)*k)
            
            # Damped update: c_new = (1-λ)*c_old + λ*c_target
            c_new[i_k] = (1 - dampening) * c + dampening * c_target
        else:
            c_new[i_k] = c  # Keep current if R_prime or c_prime invalid
        
        # Ensure consumption stays positive and reasonable
        c_new[i_k] = max(c_new[i_k], 1e-10)
        c_new[i_k] = min(c_new[i_k], k**α + (1-δ)*k)  # Can't consume more than available
    
    return c_new, euler_errors

# ============================================================================
# INVERT CONSUMPTION TO GET GAMMA COEFFICIENTS
# ============================================================================
def invert_consumption_to_gamma(c_values, k_grid, k_low, k_high, p_k):
    """
    Inverts consumption values at grid points to get Chebyshev coefficients
    DIRECT APPROXIMATION: c(k_i) = gamma @ T_k_i
    
    Parameters:
    -----------
    c_values : array
        Consumption values at grid points
    k_grid : array
        Capital grid points
    k_low, k_high : scalars
        Capital domain bounds
    p_k : int
        Polynomial order
    
    Returns:
    --------
    gamma : array
        Chebyshev coefficients
    """
    n = len(k_grid)
    
    # Build matrix of Chebyshev polynomials evaluated at grid points
    T_matrix = np.zeros((n, p_k))
    for i in range(n):
        k = k_grid[i]
        k_cheb = Change_Variable_Tocheb(k_low, k_high, k)
        T_k = Chebyshev_Polynomials_Recursion_mv(np.array([k_cheb]), p_k)
        T_matrix[i, :] = T_k.ravel()
    
    # Solve for gamma: c_values = T_matrix @ gamma
    # Use least squares if n != p_k, direct solve if n == p_k
    if n == p_k:
        gamma = np.linalg.solve(T_matrix, c_values)
    else:
        gamma = np.linalg.lstsq(T_matrix, c_values, rcond=None)[0]
    
    return gamma

# ============================================================================
# SOLVE THE MODEL
# ============================================================================
print("="*80)
print("DETERMINISTIC NEOCLASSICAL GROWTH MODEL - CHEBYSHEV PROJECTION (DIRECT)")
print("="*80)
print(f"\nModel Parameters:")
print(f"  β = {β}")
print(f"  α = {α}")
print(f"  δ = {δ}")
print(f"  γ = {γ}")
print(f"\nSteady-state capital: k_ss = {k_ss:.6f}")
print(f"Capital domain: [{k_low:.6f}, {k_high:.6f}]")
print(f"Number of Chebyshev nodes: {n_k}")
print(f"Number of coefficients: {n_coeffs}")
print(f"\nNOTE: Using DIRECT approximation c(k) = gamma @ T_k (no log-exp transformation)")

# ============================================================================
# FIXED-POINT ITERATION ALGORITHM
# ============================================================================
c_ss = k_ss**α - δ * k_ss  # Steady-state consumption

# Initialize consumption at grid points (constant at steady state)
c_current = np.full(n_k, c_ss)

# Initialize gamma to approximate constant function c(k) = c_ss
# We need gamma such that gamma @ T_k = c_ss for all k
# For constant function, only T_0 (constant term) should be non-zero
gamma_current = invert_consumption_to_gamma(c_current, k_grid, k_low, k_high, p_k)

print(f"\nInitial consumption: c = c_ss = {c_ss:.6f} everywhere")
print(f"Initial gamma (first 5): {gamma_current[:5]}")
print(f"\nStarting fixed-point iteration...")
print("="*80)

# Fixed-point iteration parameters
max_iter = 2000
tol = 1e-8

for iter in range(max_iter):
    # Compute Euler errors and update consumption
    c_new, euler_errors = compute_euler_errors_and_update(c_current, k_grid, k_low, k_high, p_k, gamma_current)
    
    # Check convergence: max Euler error
    max_error = np.max(np.abs(euler_errors))
    mean_error = np.mean(np.abs(euler_errors))
    
    # Print progress every 10 iterations
    if iter % 10 == 0 or max_error < tol:
        print(f"Iteration {iter:4d}: max |Euler error| = {max_error:.6e}, mean = {mean_error:.6e}")
    
    # Check convergence
    if max_error < tol:
        print(f"\nConverged after {iter} iterations!")
        break
    
    # Update consumption with damping (only update if errors are reasonable)
    if max_error < 1.0:  # Only update if errors aren't too large
        c_current = (1 - dampening) * c_current + dampening * c_new
    else:
        # If errors are too large, use smaller step
        c_current = (1 - dampening * 0.1) * c_current + dampening * 0.1 * c_new
    
    # Invert consumption to get new gamma coefficients
    gamma_current = invert_consumption_to_gamma(c_current, k_grid, k_low, k_high, p_k)
    
    # Recompute consumption from gamma to ensure consistency
    c_current = np.array([c_cheb(k, gamma_current, k_low, k_high, p_k) for k in k_grid])

if iter == max_iter - 1:
    print(f"\nWarning: Reached maximum iterations ({max_iter})")

gamma_opt = gamma_current
print(f"\nFinal max |Euler error|: {max_error:.6e}")
print(f"Final mean |Euler error|: {mean_error:.6e}")
print(f"Optimal coefficients (first 5): {gamma_opt[:5]}")

# ============================================================================
# PLOT RESULTS
# ============================================================================
print("\nGenerating plots...")

# Fine grid for plotting
k_grid_fine = np.linspace(k_low, k_high, 200)

# Evaluate consumption policy function on fine grid
c_policy = np.zeros_like(k_grid_fine)
for i in range(len(k_grid_fine)):
    c_policy[i] = c_cheb(k_grid_fine[i], gamma_opt, k_low, k_high, p_k)

# Compute capital transition: k' = k^α - c(k) + (1-δ)*k
k_prime_policy = np.zeros_like(k_grid_fine)
for i in range(len(k_grid_fine)):
    k = k_grid_fine[i]
    c = c_policy[i]
    k_prime_policy[i] = (1 - δ) * k + k**α - c

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Consumption policy function
ax1 = axes[0, 0]
ax1.plot(k_grid_fine, c_policy, 'b-', linewidth=2, label='Chebyshev approximation (direct)')
ax1.scatter(k_grid, [c_cheb(k, gamma_opt, k_low, k_high, p_k) for k in k_grid], 
           c='red', s=100, marker='o', edgecolors='black', linewidths=1.5, 
           label=f'Chebyshev nodes ({n_k})', zorder=5)
ax1.axvline(k_ss, color='green', linestyle='--', linewidth=2, label=f'Steady-state (k={k_ss:.3f})')
ax1.set_xlabel('k (Capital)', fontsize=12, fontweight='bold')
ax1.set_ylabel('c (Consumption)', fontsize=12, fontweight='bold')
ax1.set_title('Consumption Policy Function (Direct Approximation)', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Capital transition function
ax2 = axes[0, 1]
ax2.plot(k_grid_fine, k_prime_policy, 'b-', linewidth=2, label="k' = f(k)")
ax2.plot([k_low, k_high], [k_low, k_high], 'r--', linewidth=2, label='45° line')
ax2.scatter(k_grid, [k**α - c_cheb(k, gamma_opt, k_low, k_high, p_k) + (1-δ)*k 
                    for k in k_grid], 
           c='red', s=100, marker='o', edgecolors='black', linewidths=1.5, 
           label='Chebyshev nodes', zorder=5)
ax2.axvline(k_ss, color='green', linestyle='--', linewidth=2)
ax2.axhline(k_ss, color='green', linestyle='--', linewidth=2)
ax2.set_xlabel('k (Current Capital)', fontsize=12, fontweight='bold')
ax2.set_ylabel("k' (Next Period Capital)", fontsize=12, fontweight='bold')
ax2.set_title('Capital Transition Function', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Euler errors at collocation points
ax3 = axes[1, 0]
euler_errors = []
for k in k_grid:
    c = c_cheb(k, gamma_opt, k_low, k_high, p_k)
    k_prime = (1 - δ) * k + k**α - c
    c_prime = c_cheb(k_prime, gamma_opt, k_low, k_high, p_k)
    R_prime = α * k_prime**(α - 1) + (1 - δ)
    consumption_ratio = c_prime / c
    # Euler error: 1 - β * (c'/c)^(-γ) * R'
    euler_error = 1 - β * (consumption_ratio)**(-γ) * R_prime
    euler_errors.append(euler_error)

ax3.scatter(k_grid, np.abs(euler_errors), c='blue', s=100, marker='o', 
           edgecolors='black', linewidths=1.5)
ax3.axhline(0, color='red', linestyle='--', linewidth=1)
ax3.set_xlabel('k (Capital)', fontsize=12, fontweight='bold')
ax3.set_ylabel('|Euler Error|', fontsize=12, fontweight='bold')
ax3.set_title('Euler Errors at Collocation Points', fontsize=14, fontweight='bold')
ax3.set_yscale('log')
ax3.grid(True, alpha=0.3)

# Plot 4: Consumption vs Capital (with steady-state reference)
ax4 = axes[1, 1]
# Compute steady-state consumption
c_ss = k_ss**α - δ * k_ss
ax4.plot(k_grid_fine, c_policy, 'b-', linewidth=2, label='Policy function')
ax4.axvline(k_ss, color='green', linestyle='--', linewidth=2)
ax4.axhline(c_ss, color='green', linestyle='--', linewidth=2, label=f'Steady-state (c={c_ss:.3f})')
ax4.scatter([k_ss], [c_ss], c='green', s=200, marker='*', edgecolors='black', 
           linewidths=2, zorder=10, label='Steady-state point')
ax4.set_xlabel('k (Capital)', fontsize=12, fontweight='bold')
ax4.set_ylabel('c (Consumption)', fontsize=12, fontweight='bold')
ax4.set_title('Policy Function with Steady-State', fontsize=14, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()

# Save figure
output_path = 'NGM_figures/deterministic_Chebyshev_results_direct.png'
os.makedirs('NGM_figures', exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved plot: {output_path}")
plt.close()

# Print summary statistics
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)
print(f"Max Euler error: {max(np.abs(euler_errors)):.6e}")
print(f"Mean Euler error: {np.mean(np.abs(euler_errors)):.6e}")
print(f"RMS Euler error: {np.sqrt(np.mean(np.array(euler_errors)**2)):.6e}")
print(f"\nSteady-state values:")
print(f"  k_ss = {k_ss:.6f}")
print(f"  c_ss = {c_ss:.6f}")
print(f"  Policy function at k_ss: c({k_ss:.6f}) = {c_cheb(k_ss, gamma_opt, k_low, k_high, p_k):.6f}")
print(f"  Error: {abs(c_cheb(k_ss, gamma_opt, k_low, k_high, p_k) - c_ss):.6e}")

print("\nDone!")

