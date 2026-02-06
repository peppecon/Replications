# Buera & Shin (2013) - Replication

This repository contains a Python replication of **"Financial Frictions and the Persistence of History"** by Francisco J. Buera and Yongseok Shin (*Journal of Political Economy*, 2013).

## Model Overview

The paper studies how financial frictions interact with initial resource misallocation to generate slow transition dynamics following economic reforms. Key features:

- **Occupational choice**: Agents choose between being workers or entrepreneurs
- **Span-of-control technology**: $y = z(k^\alpha l^{1-\alpha})^{1-\nu}$ where $z$ is entrepreneurial ability
- **Collateral constraint**: $k \leq \lambda a$ (entrepreneurs can only use capital up to $\lambda$ times their wealth)
- **Idiosyncratic distortions**: Pre-reform taxes $\tau^+$ and subsidies $\tau^-$ that misallocate resources
- **Pareto ability distribution**: $z \sim \text{Pareto}(\eta)$ with $\eta = 4.15$

### Calibration (Table 1 from paper)

| Parameter | Value | Description |
|-----------|-------|-------------|
| $\alpha$ | 0.33 | Capital share in production |
| $\nu$ | 0.21 | Returns to scale (span of control = $1-\nu$) |
| $\delta$ | 0.06 | Depreciation rate |
| $\beta$ | 0.92 | Discount factor |
| $\sigma$ | 1.5 | Risk aversion |
| $\psi$ | 0.10 | Death/exit probability |
| $\lambda$ | 1.35 | Collateral constraint |
| $\eta$ | 4.15 | Pareto tail parameter |
| $\tau^+$ | 0.57 | Tax on high-ability entrepreneurs |
| $\tau^-$ | -0.15 | Subsidy to low-ability entrepreneurs |
| $q$ | 1.55 | Distortion assignment parameter |

## Repository Structure

```
BueraShin_JPE2013/
├── main_sim.py              # Stationary equilibrium (VFI + simulation)
├── main_cheb.py             # Stationary equilibrium (Chebyshev collocation)
├── transition_howard.py     # Transition dynamics (VFI + Howard acceleration)
├── transition_cheb.py       # Transition dynamics (Chebyshev collocation)
├── plot_paper_figures.py    # Generate all paper figures
├── outputs/                 # Generated results and figures
│   ├── steady_states.npz
│   ├── transition_path.npz
│   ├── transition_neoclassical.npz
│   └── fig_*.png
└── docs/                    # Method documentation
```

## Scripts

### Stationary Equilibrium

| Script | Method | Description |
|--------|--------|-------------|
| `main_sim.py` | VFI + Simulation | Value Function Iteration with Howard acceleration, simulation-based distribution |
| `main_cheb.py` | Chebyshev | Spectral collocation with analytical distribution (Power Iteration) |

### Transition Dynamics

| Script | Method | Description |
|--------|--------|-------------|
| `transition_howard.py` | VFI + Howard | **Recommended**. Implements Algorithm B.2 with nested price clearing |
| `transition_cheb.py` | Chebyshev | Time iteration with spectral methods |

### Plotting

| Script | Description |
|--------|-------------|
| `plot_paper_figures.py` | Generates all figures replicating the paper |

## Usage

### 1. Compute Steady States and Transition

```bash
# Using conda environment
conda activate phd_econ

# Run transition (computes both steady states and transition path)
python transition_howard.py --out outputs --T 125 --N 350000

# Options:
#   --T      Transition horizon (default: 125 years)
#   --N      Number of simulated agents (default: 350,000)
#   --na     Asset grid points (default: 601)
#   --amax   Maximum assets (default: 300)
#   --burn   Burn-in periods for steady state (default: 500)
```

### 2. Generate Figures

```bash
python plot_paper_figures.py --out outputs
```

## Output Files

### Data Files (`outputs/`)

| File | Contents |
|------|----------|
| `steady_states.npz` | Pre/post reform steady state policies, distributions, prices |
| `transition_path.npz` | Benchmark transition ($\lambda=1.35$): prices, aggregates, micro metrics |
| `transition_neoclassical.npz` | Neoclassical transition ($\lambda=\infty$) for comparison |

### Figures (`outputs/`)

| Figure | Description | Paper Reference |
|--------|-------------|-----------------|
| `fig_aggregate_dynamics_I.png` | GDP, TFP, Investment Rate | Figure 3 |
| `fig_aggregate_dynamics_II.png` | Capital Stock, Interest Rates | Figure 4 |
| `fig_micro_implications.png` | Avg. Entrepreneurial Ability, Wealth Share Top 5% | Figure 5 |
| `fig_policy_functions.png` | Asset policy $a'(a)$ for different ability percentiles | - |
| `fig_asset_policy_contour.png` | 2D contour of $a'(a,z)$ | - |
| `fig_occupational_contour.png` | Occupational choice boundaries pre/post reform | - |
| `fig_wealth_distribution.png` | Wealth distribution comparison | - |
| `fig_ability_distribution.png` | Ability grid and Pareto distribution | - |

## Key Results Replicated

1. **Slow transition**: Capital stock takes ~10.5 years to cover half the distance to new steady state (vs 5.5 years in neoclassical model)

2. **Endogenous TFP dynamics**: TFP increases ~47% over the transition as resources reallocate from subsidized to productive entrepreneurs

3. **Hump-shaped investment**: Investment rate initially drops then rises, reflecting the evolution of the wealth distribution

4. **Wealth concentration**: Top 5% ability agents increase their wealth share from ~12% to ~48% as they save to overcome collateral constraints

## Numerical Methods

### Howard Policy Improvement (Recommended)

The `transition_howard.py` script uses:
- Value Function Iteration with Howard acceleration (25 policy improvement steps)
- Monte Carlo simulation for distribution evolution
- Nested price clearing: inner loop for wages, outer loop for interest rates
- Relaxation parameters: $\eta_w = 0.35$, $\eta_r = 0.20$

### Chebyshev Collocation (Alternative)

The `transition_cheb.py` script uses:
- Spectral collocation with Chebyshev polynomials
- Time iteration via Euler equation
- Analytical distribution updates

> **Note**: The Howard method is more stable and recommended for this model.

## Dependencies

- Python 3.8+
- NumPy
- SciPy
- Numba (JIT compilation)
- Matplotlib

## References

- Buera, F. J., & Shin, Y. (2013). Financial frictions and the persistence of history: A quantitative exploration. *Journal of Political Economy*, 121(2), 221-272.

## Author

Replication by PhD student, Bocconi University.
