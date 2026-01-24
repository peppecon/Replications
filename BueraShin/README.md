# Buera & Shin (2010) Replication

This folder contains multiple Python implementations of the replication for "Financial Frictions and the Persistence of History" (Buera & Shin, 2010).

## Implementation Versions

1.  **[buera_shin_v1_baseline.py](buera_shin_v1_baseline.py)**:
    - Baseline version.
    - Uses paper's exact ability discretization (40 points).
    - Analytical stationary distribution computation.
    - Slower execution compared to optimized versions.

2.  **[buera_shin_v2_vectorized.py](buera_shin_v2_vectorized.py)**:
    - **Optimized for speed**.
    - Vectorized entrepreneur problem solving.
    - Pre-computed transition matrices for the ability process.
    - Parallelized Value Function Iteration (VFI) using `numba.prange`.

3.  **[buera_shin_v3_howard_pi.py](buera_shin_v3_howard_pi.py)**:
    - **Further Optimized (Recommended)**.
    - Implements **Howard's Policy Iteration Acceleration** (multiple value updates per policy).
    - Exploits policy function monotonicity (savings increase in assets).
    - Uses binary search for optimal savings.
    - Warm-starts VFI from previous solutions across lambda values.

## Results

All generated plots and diagnostics are saved in the **[plots/](plots/)** directory to keep the workspace clean.
