# Buera & Shin (2013) - Replication

This repository contains a Python replication of the model in **"Financial Frictions and the Persistence of History"** by Francisco J. Buera and Yongseok Shin (Journal of Political Economy, 2013).

## Scripts Overview

The repository includes several scripts implementing stationary equilibrium and transition dynamics using different numerical methods:

### Stationary Equilibrium
- **[main_sim.py](main_sim.py)**: Computes the stationary equilibrium using Value Function Iteration (VFI) with Howard acceleration and a simulation-based stationary distribution.
- **[main_cheb.py](main_cheb.py)**: Computes the stationary equilibrium using Chebyshev spectral collocation and an analytical stationary distribution (Power Iteration).
- **[benchmark_vfi.py](benchmark_vfi.py)**: A performance-optimized VFI benchmark for the stationary equilibrium.

### Transition Dynamics
- **[transition_howard.py](transition_howard.py)**: Implements transition dynamics using VFI with Howard acceleration and Monte Carlo simulation for market clearing.
- **[transition_cheb.py](transition_cheb.py)**: Implements transition dynamics using Chebyshev spectral collocation and a nested solver.
- **[transition_sim.py](transition_sim.py)**: Implements the simulation-based transition dynamics following Algorithm B.2 in the paper's appendix.

### Plotting
- **[plot_paper_figures.py](plot_paper_figures.py)**: Generates the main figures of the paper, including policy functions, transition dynamics, and wealth/ability distributions.

## Methodology and Documentation

The project explores two main numerical methods for solving the model. Detailed documentation for each is available in the `docs` folder:

1. **Chebyshev Functional Approximation**: Located in [`docs/time_iteration`](docs/time_iteration). This method uses spectral collocation to approximate the value and policy functions.
2. **Howard Policy Improvement for VFI**: Located in [`docs/howard_policy_improvement`](docs/howard_policy_improvement). This method accelerates standard VFI by performing multiple value updates for a fixed policy.

> [!IMPORTANT]
> The **Howard Policy Improvement** method is simpler to implement and works significantly better in terms of stability and performance for this specific model.

## Usage

To run the main simulation for the stationary equilibrium:
```bash
python main_sim.py
```

To compute the transition path:
```bash
python transition_howard.py
```

To generate the plots after running the simulations:
```bash
python plot_paper_figures.py
```
