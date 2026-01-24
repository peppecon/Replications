# Replication of Buera & Shin (2010)
## "Financial Frictions and the Persistence of History: A Quantitative Exploration"

**Author**: Replication code by AI assistant
**Date**: January 2025
**Original Paper**: Buera, Francisco J., and Yongseok Shin. "Financial frictions and the persistence of history: A quantitative exploration." *Journal of Political Economy* 121.2 (2013): 221-272.

---

## Overview

This repository contains Python code to replicate the key results from Buera & Shin (2010/2013), specifically:

1. **Baseline stationary equilibrium** (Section 3.1)
2. **Figure 2**: Long-run effect of financial frictions (Section 3.2)

The code follows the numerical algorithm described in Appendix B.1 of the paper.

---

## Model Summary

### Economic Environment

- **Agents**: Continuum of infinitely-lived agents with heterogeneous entrepreneurial ability
- **Occupational choice**: Each period, agents choose to be workers (supply 1 unit of labor) or entrepreneurs (operate a firm)
- **Production**: Span-of-control technology: $f(e, k, l) = e \cdot (k^\alpha l^{1-\alpha})^{1-\nu}$
- **Financial friction**: Collateral constraint $k \leq \lambda a$ where $\lambda \geq 1$

### Key Parameters (Table 1 calibration)

| Parameter | Value | Description |
|-----------|-------|-------------|
| $\sigma$ | 1.5 | Risk aversion (CRRA) |
| $\beta$ | 0.904 | Discount factor |
| $\alpha$ | 0.33 | Capital share in variable factors |
| $\nu$ | 0.21 | Entrepreneur's share (span-of-control) |
| $\delta$ | 0.06 | Depreciation rate |
| $\eta$ | 4.15 | Pareto tail parameter |
| $\psi$ | 0.894 | Ability persistence |

### Financial Frictions

- $\lambda = 1$: Financial autarky (no external borrowing)
- $\lambda = \infty$: Perfect credit markets
- Baseline US economy: $\lambda \approx 1.35$ (matching external finance to GDP ratio)

---

## File Structure

```
BueraShin/
├── buera_shin_replication.py      # Main replication code (v1)
├── buera_shin_replication_v2.py   # Optimized version (v2)
├── figure2_replication.png        # Main output figure
├── figure2_diagnostics.png        # Diagnostic plots
├── docs/
│   └── replication_notes.md       # This file
└── BueraShin2013example/          # Reference Matlab code (VFI Toolkit)
```

---

## Algorithm

Following Appendix B.1, the algorithm consists of:

### 1. Value Function Iteration (VFI)

For given prices $(w, r)$:

1. **Entrepreneur problem**: For each state $(a, e)$, solve static profit maximization:
   - Unconstrained optimal capital: $k^* = \arg\max \{e(k^\alpha l^{1-\alpha})^{1-\nu} - wl - (r+\delta)k\}$
   - Apply collateral constraint: $k = \min(k^*, \lambda a)$
   - Compute optimal labor and profit

2. **Occupational choice**: Compare entrepreneurial profit vs. wage $w$

3. **Savings decision**: Standard Bellman equation with CRRA utility

### 2. Stationary Distribution

Compute the stationary distribution using:
- Sparse transition matrix construction
- Power iteration or eigenvalue method

### 3. General Equilibrium

Iterate on prices $(w, r)$ until markets clear:
- **Capital market**: $K^d = A$ (capital demand = total assets)
- **Labor market**: $L^d = 1 - \text{share}_e$ (labor demand = workers)

---

## Key Results

### Figure 2: Long-run Effect of Financial Frictions

| $\lambda$ | ExtFin/GDP | GDP (rel.) | TFP (rel.) | Interest Rate |
|-----------|------------|------------|------------|---------------|
| $\infty$ | ~1.7 | 1.00 | 1.00 | ~4.5% |
| 2.0 | ~1.0 | ~0.85 | ~0.95 | ~-2% |
| 1.5 | ~0.7 | ~0.80 | ~0.90 | ~-4% |
| 1.0 | 0.0 | ~0.70 | ~0.85 | ~-6% |

### Key Findings

1. **GDP Loss**: Financial autarky reduces output by ~30% relative to perfect credit
2. **TFP Loss**: Misallocation reduces TFP by ~15%
3. **Interest Rates**: Tighter frictions lower equilibrium interest rates (excess savings)
4. **Mechanism**: Financial frictions prevent talented entrepreneurs from operating at efficient scale

---

## Code Versions

### v1: `buera_shin_replication.py`
- Standard implementation following the paper
- Uses simulation-based approach for some computations
- Runtime: ~15-20 minutes

### v2: `buera_shin_replication_v2.py`
- Optimized with:
  - Parallelized VFI using `numba.prange`
  - Sparse matrix operations for stationary distribution
  - Pre-computed expected values
  - Non-interactive matplotlib backend
- Runtime: ~10 minutes

---

## Dependencies

```python
numpy
matplotlib
numba
scipy
```

Install with:
```bash
pip install numpy matplotlib numba scipy
```

Or with conda:
```bash
conda install numpy matplotlib numba scipy
```

---

## Usage

```bash
# Activate environment
conda activate phd_econ

# Run replication
python buera_shin_replication.py

# Or optimized version
python buera_shin_replication_v2.py
```

---

## Technical Notes

### Ability Grid Construction (Page 235)

The paper uses a specific 40-point discretization:
- Points 1-38: Equidistant in CDF space from $M(e_1)=0.633$ to $M(e_{38})=0.998$
- Points 39-40: $M(e_{39})=0.999$, $M(e_{40})=0.9995$
- CDF: $M(e) = 1 - e^{-\eta}$ (Pareto)

### Ability Transition

The transition matrix for ability:
$$\pi(e'|e) = \psi \cdot \mathbf{1}_{e'=e} + (1-\psi) \cdot \text{Pr}(e')$$

With probability $\psi$, ability persists; otherwise, new draw from stationary distribution.

### Entrepreneur's Problem

Optimal capital (unconstrained):
$$k^* = \left[\frac{\alpha(1-\nu)}{r+\delta}\right]^{\frac{1-(1-\alpha)(1-\nu)}{\nu}} \left[\frac{(1-\alpha)(1-\nu)}{w}\right]^{\frac{(1-\alpha)(1-\nu)}{\nu}} e^{1/\nu}$$

---

## Comparison with Paper

The replication qualitatively matches Figure 2 from the paper:

- Monotonic relationship between external finance/GDP and output
- GDP and TFP increase with financial development
- Interest rates increase with financial development
- Significant output losses from financial frictions

Minor quantitative differences may arise from:
- Grid resolution differences
- Equilibrium solver tolerances
- Numerical precision

---

## References

1. Buera, F. J., & Shin, Y. (2013). Financial frictions and the persistence of history: A quantitative exploration. *Journal of Political Economy*, 121(2), 221-272.

2. VFI Toolkit (Matlab reference implementation): [GitHub](https://github.com/vfitoolkit)

---

## License

This replication code is provided for educational and research purposes.
