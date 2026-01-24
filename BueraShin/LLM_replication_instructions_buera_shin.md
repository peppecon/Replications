# Instructions for an LLM to Replicate Buera–Shin (Financial Frictions and the Persistence of History)

## Goal
Replicate the paper’s **baseline stationary equilibrium** and the main **long-run financial-frictions exercise** (Section 3.2 / Figure 2), using the model and numerical algorithm described in the paper.

## Read-first checklist (paper guidance)
1. Read the **model section** that defines households, entrepreneurs, technology, and financial frictions.
2. Read the **calibration section** (parameter values, distributions, and normalization choices).
3. Read **Appendix B.1 (Computing the stationary equilibrium)** carefully and treat it as the authoritative algorithm description.
4. Identify exactly how the paper defines:
   - “GDP” / output per worker
   - “TFP” (aggregate productivity measure)
   - “External finance to GDP” (the x-axis of Figure 2)
   - The ability process (including the “reset” probability and the ability distribution)

> When uncertain, default to the paper’s definitions and note the page/section you used.

## Replication deliverables
Produce these outputs:
1. A working solver that finds a **stationary equilibrium** consistent with Appendix B.1.
2. A table of baseline steady-state moments that match the paper’s reporting (as closely as possible).
3. A reproduction of the **Figure 2-style plots**:
   - GDP and TFP (normalized by perfect credit) vs external finance to GDP
   - Interest rate vs external finance to GDP
4. A short replication note describing:
   - which sections/appendices you implemented
   - any deviations from the paper (if necessary)
   - how close your moments/plots are to the paper’s figures

## Implementation plan (high level, let the LLM fill in details)
### Step 1 — Reconstruct the model from the paper
- Implement the household problem exactly as described in the model section.
- Implement the entrepreneurial technology and the financial friction parameter **λ** as in the paper.
- Implement the ability process using the persistence/reset mechanism described in the paper (see where they define **ψ** and the ability distribution).

### Step 2 — Calibration and discretization
- Use the paper’s baseline calibration (Section 3.1 / calibration tables).
- Discretize any continuous state(s) as needed, following standard practice and the paper’s stated approach.
- Ensure the discretization is sufficient to replicate reported steady-state statistics.

### Step 3 — Solve the stationary equilibrium (Appendix B.1)
Follow Appendix B.1 verbatim:
1. Guess the interest rate in the invariant distribution.
2. Guess the wage in the invariant distribution.
3. Given (r,w), solve the individual problem in stationary equilibrium.
4. Simulate a large number of individuals for T periods and use the cross-section at period T as an approximation to the invariant distribution.
5. Update the wage until the labor market clears at period T.
6. Update the interest rate until the capital (asset) market clears at period T.
7. Iterate until both markets clear.

### Step 4 — Validate the baseline steady state
- Confirm that the steady-state equilibrium matches the paper’s baseline moments and qualitative patterns.
- Run basic robustness checks mentioned by the paper (e.g., larger T should not materially change the invariant distribution).

### Step 5 — Long-run financial frictions exercise (Section 3.2 / Figure 2)
- Vary **λ** over the range described in Section 3.2 (including λ=1 and λ→∞).
- For each λ, solve for the stationary equilibrium and compute:
  - steady-state GDP (normalized by perfect credit)
  - steady-state TFP (normalized by perfect credit)
  - equilibrium interest rate
  - external finance to GDP (use the paper’s definition)
- Reproduce the shape and magnitude described in the text around Figure 2.

### Step 6 — Optional sensitivity experiments (Section 3.2 discussion)
- Repeat the long-run exercise for alternative values of:
  - **ψ** (shock persistence)
  - **η** (ability dispersion)
- Use the alternative values explicitly mentioned in the paper and report how results change.

## Sanity checks (qualitative, from the paper’s narrative)
- Tightening financial frictions (lower λ) should:
  - reduce GDP and TFP (relative to perfect credit)
  - lower the equilibrium interest rate
  - reduce external finance to GDP
- The magnitude of GDP/TFP changes should be in the ballpark described in Section 3.2.

## Reporting format
- Provide:
  - a short “Replication log” listing the paper sections implemented (with page/section references)
  - final plots and summary tables
  - the list of any numerical settings (grid sizes, simulation length T, number of agents N) needed to reproduce your results

## What NOT to do
- Do not invent definitions of GDP, TFP, or external finance. Use the paper’s definitions.
- Do not skip Appendix B.1 details. Use the same equilibrium-finding structure.
- Do not add extra frictions or open-economy features unless explicitly replicating an extension (do baseline first).
