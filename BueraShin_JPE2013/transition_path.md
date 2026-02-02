# Transition Path Implementations - Buera & Shin (2010)

This document summarizes the three transition path implementations for the Buera & Shin (2010) model with idiosyncratic distortions.

## Overview

All three files compute:
1. **Post-reform equilibrium**: λ=1.35 without distortions (τ=0)
2. **Pre-reform equilibrium**: λ=1.35 with idiosyncratic output wedges
3. **Transition path**: Perfect-foresight TPI when distortions are removed

## File Comparison

| File | Policy Method | Distribution | Speed | Accuracy |
|------|--------------|--------------|-------|----------|
| `transition_buera_shin.py` | Chebyshev spectral | Analytical (sparse) | Slowest | Highest |
| `transition_path_v3_howardpi.py` | Grid VFI + Howard PI | Analytical (sparse) | Medium | High |
| `transition_path_v4_simulation.py` | Grid VFI + Howard PI | Monte Carlo (350k) | Fastest | Good |

## Distortion Model

The pre-reform economy has idiosyncratic output wedges:

```
profit = (1-τ) * z * (k^α * l^(1-α))^(1-ν) - wl - (r+δ)k
```

**Parameters:**
- τ_plus = 0.57 (tax, correlated with high ability)
- τ_minus = -0.15 (subsidy, correlated with low ability)
- Pr(τ=τ_plus | z) = 1 - exp(-q*z), where q = 1.55
- Joint persistence: with prob ψ=0.894 stay at (z,τ), else redraw both

## Implementation Details

### Entrepreneur Problem
All files use effective productivity `z_eff = (1-τ)*z` to solve the static problem:
```python
z_eff = (1.0 - tau) * z
# FOCs use z_eff for both capital and labor
```

### Value Function with Distortions
The two tau states are **coupled** because the expected value depends on both:
```
E[V' | z, τ] = ψ·V_τ(a', z) + (1-ψ)·E_{z',τ'}[V(a', z', τ')]
```

### Equilibrium Solver (Simultaneous Update)
All files use a single-loop price update with adaptive damping:
```python
# Simultaneous update (not nested loops)
w_new = w * (1 + w_step * exc_L)
r_new = r + r_step * exc_K

# Adaptive step (reduce if sign flips)
if exc_L * exc_L_prev < 0: w_step *= 0.5
if exc_K * exc_K_prev < 0: r_step *= 0.5

# Damped update
w = 0.5 * w + 0.5 * w_new
r = 0.5 * r + 0.5 * r_new
```

### Transition Path Algorithm (TPI)
1. Initialize price paths (w_t, r_t) from pre to post steady state
2. **Backward pass**: Solve policies from T-1 to 0 using continuation values
3. **Forward pass**: Update distribution from 0 to T-1 using policies
4. Compute excess demands and update prices
5. Repeat until convergence

## Usage

```bash
# Chebyshev version (most accurate)
python transition_buera_shin.py --T 250

# Howard PI version (faster)
python transition_path_v3_howardpi.py --T 250

# Simulation version (fastest)
python transition_path_v4_simulation.py --T 50
```

## Output

Each script produces:
- Stationary equilibrium summaries (JSON)
- Transition path data (CSV)
- Plots of aggregate dynamics (PNG)
