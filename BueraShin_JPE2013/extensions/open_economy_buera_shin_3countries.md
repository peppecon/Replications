# 3-Country Open-Economy Buera–Shin (JPE 2013 Style) — No Trade, With Capital/Asset Flows

This document states a **3-country** version of a Buera–Shin style economy that allows **cross-country asset holding / capital inflows** *without* modeling international trade. Countries can be interpreted as **EU2004**, **EU15**, and **ROW**.

Two variants are included:
1) **World bond (single interest rate)** — simplest, delivers capital inflows via net foreign assets but one common `r`.
2) **Country assets + portfolio/integration friction (Kleinman-style)** — delivers **country-specific** interest rates `r_i` and smooth capital reallocation.

---

## 1. Countries and agents

- Countries indexed by `i ∈ {1,2,3}` (EU2004, EU15, ROW).
- Population weights `N_i` (or masses), labor is **immobile** across countries.
- Each household in country `i` has state `(a,e)`:
  - `a ≥ 0` risk-free wealth / net worth / collateral
  - `e ∈ E` entrepreneurial ability

### Preferences
Households maximize expected discounted utility:
\[
\max \; \mathbb{E}\sum_{t\ge 0}\beta^t u(c_t), \qquad
u(c)=\frac{c^{1-\sigma}}{1-\sigma}.
\]

### Ability process (Buera–Shin reset)
\[
e_{t+1}=\begin{cases}
e_t & \text{w.p. } \psi\\
\tilde e \sim \pi_0 & \text{w.p. } 1-\psi
\end{cases}
\]
Let `Π(e'|e)` be the implied transition matrix.

---

## 2. Technology and financial frictions

### Entrepreneurial technology in country `i`
An entrepreneur with ability `e` operates:
\[
y = A_i\, e\, (k^\alpha \ell^{1-\alpha})^{1-\nu}.
\]

### Collateral/credit constraint in country `i`
\[
k \le \lambda_i\, a.
\]
`λ_i` governs the tightness of financial frictions (higher `λ_i` = easier external finance).

---

## 3. Entrepreneur’s static problem (country `i`)

Given prices `(w_i, r_i)` (or `(w_i, r)` in the world-bond case), define the user cost:
\[
R_i \equiv r_i + \delta.
\]
Entrepreneur solves, for each state `(a,e)`:
\[
\pi_i(a,e;w_i,r_i)=\max_{k,\ell\ge 0}
\left\{A_i e (k^\alpha \ell^{1-\alpha})^{1-\nu}-w_i\ell-R_i k\right\}
\quad \text{s.t.}\quad k\le \lambda_i a.
\]
This delivers policy functions:
- `k_i(a,e;w_i,r_i)` capital demand
- `ℓ_i(a,e;w_i,r_i)` labor demand
- `y_i(a,e;w_i,r_i)` output
- `π_i(a,e;w_i,r_i)` profit

### Occupational choice (Buera–Shin simplification)
Define the entrepreneur indicator:
\[
\chi_i(a,e) \equiv \mathbf{1}\{\pi_i(a,e;w_i,r_i) \ge w_i\}.
\]
Current income is `max{w_i, π_i(a,e)}`.

---

## 4. Household dynamic problem (two open-economy variants)

### 4.1 Variant A: World bond (single interest rate `r`)
Household in country `i` chooses savings `a'`:
\[
V_i(a,e)=\max_{a'\ge 0}\Big\{
u(c)+\beta\,\mathbb{E}[V_i(a',e')\mid e]
\Big\}
\]
subject to:
\[
c + a' = \max\{w_i,\pi_i(a,e;w_i,r)\} + (1+r)a.
\]

**State variables:** `(a,e)` per country (no extra portfolio state).

---

### 4.2 Variant B: Country assets + portfolio/integration friction (Kleinman-style)
Here there are **country-specific returns** `{r_i}` on capital installed in each country. Households do *not* choose a full portfolio at the micro level; instead, a **mutual-fund / reduced-form portfolio rule** allocates world savings across countries as a function of returns and an integration parameter `ν_int`.

Household DP remains 1D in wealth:
\[
V_i(a,e)=\max_{a'\ge 0}\Big\{
u(c)+\beta\,\mathbb{E}[V_i(a',e')\mid e]
\Big\}
\]
with:
\[
c + a' = \max\{w_i,\pi_i(a,e;w_i,r_i)\} + (1+r_i^H)a.
\]
- `r_i` is the return relevant for firms/capital in country `i`.
- `r_i^H` is the return paid to savers in country `i` (you can set `r_i^H=r_i`, or define a separate deposit rate/spread if desired).

#### Portfolio allocation rule (illustrative Kleinman-style mapping)
Let total world savings be:
\[
S \equiv \sum_{j=1}^3 N_j \int a'_j(a,e)\, d\mu_j.
\]
The mutual fund allocates shares:
\[
\omega_i(r_1,r_2,r_3;\nu_{int}) \in (0,1),\qquad \sum_i \omega_i=1.
\]
Example smooth/logit form:
\[
\omega_i = \frac{\exp(\nu_{int}\,r_i)}{\sum_{j=1}^3 \exp(\nu_{int}\,r_j)}.
\]
Then capital supplied to country `i` is:
\[
K_i^s = \omega_i(r_1,r_2,r_3;\nu_{int})\, S.
\]
**Interpretation:** raising `ν_int` (integration) makes the allocation more sensitive to return differentials and can generate large **capital inflows** (e.g., into EU2004 after accession).

---

## 5. Stationary competitive equilibrium (3 countries)

A stationary equilibrium consists of:
- Prices `{w_i}` and either a common `r` (Variant A) or `{r_i}` (Variant B)
- Household savings policies `a'_i(a,e)` and implied occupational rule `χ_i(a,e)`
- Static entrepreneur policies `(k_i, ℓ_i, y_i, π_i)`
- Invariant distributions `μ_i(a,e)` for each country

such that, for each country `i`:

### (i) Household optimality
Given prices, `a'_i` solves the household DP and `χ_i(a,e)=1{π_i(a,e)≥w_i}`.

### (ii) Entrepreneur optimality
Given prices and collateral constraint, `(k_i,ℓ_i)` solve the static profit maximization.

### (iii) Stationary distributions
Distributions are invariant under policies and the ability process:
\[
\mu_i = \mathcal{T}_i(\mu_i; a'_i, \Pi).
\]
(Compute via long simulation at large `T` as in Buera–Shin, or via a Markov transition operator on the grid.)

### (iv) Labor market clearing (each country)
\[
L_i^d \equiv \int \chi_i(a,e)\,\ell_i(a,e;w_i,\cdot)\, d\mu_i
\;=\;
L_i^s \equiv \int (1-\chi_i(a,e))\, d\mu_i.
\]

### (v) Capital / asset market clearing
#### Variant A (world bond, single `r`)
Country capital demand:
\[
K_i^d \equiv \int \chi_i(a,e)\,k_i(a,e;w_i,r)\, d\mu_i.
\]
Aggregate asset supply in country `i`:
\[
A_i \equiv \int a\, d\mu_i.
\]
**World clearing condition:**
\[
\sum_{i=1}^3 N_i K_i^d \;=\; \sum_{i=1}^3 N_i A_i.
\]
This pins down the common world interest rate `r`. Wages `{w_i}` are pinned down by each country’s labor clearing equation.

#### Variant B (country assets + portfolio friction, `{r_i}`)
Country capital demand:
\[
K_i^d \equiv \int \chi_i(a,e)\,k_i(a,e;w_i,r_i)\, d\mu_i.
\]
Total world savings:
\[
S \equiv \sum_{j=1}^3 N_j \int a'_j(a,e)\, d\mu_j.
\]
Capital supply to country `i` from the mutual fund:
\[
K_i^s = \omega_i(r_1,r_2,r_3;\nu_{int})\, S.
\]
**Three clearing conditions:**
\[
K_i^d(w_i,r_i) = K_i^s(r_1,r_2,r_3;\nu_{int}),\qquad i=1,2,3.
\]
These jointly pin down `{r_i}` along with labor clearing for `{w_i}`.

---

## 6. How interest rates are determined (intuition)

- **Variant A:** there is one `r`, determined by **global** capital demand equaling **global** savings. Countries can have different wages because labor markets clear locally.
- **Variant B:** each country has its own `r_i`, determined by equating that country’s capital demand to the **portfolio-allocated supply**. Lower integration frictions (higher `ν_int`) shift `K_i^s` toward higher-return countries and generate capital inflows.

---

## 7. Notes for implementation (high level)

- State space remains `(a,e)` per country; the open-economy extension changes **market clearing** and, in Variant B, adds a **portfolio allocation block**.
- Solve prices with a nested fixed point (Buera–Shin Appendix B.1 style):
  - inner loop: for each country `i`, adjust `w_i` until labor clears
  - outer loop: adjust `r` (Variant A) or `{r_i}` (Variant B) until capital clearing holds
- Compute moments of interest per country and for the world:
  - GDP, TFP, interest rates
  - external finance to GDP (use the same definition as in the paper)
  - net foreign asset positions if you want to report inflows explicitly (especially Variant A)

---

## 8. Optional “EU2004 accession” experiment

Model accession as a permanent change in integration frictions in Variant B:
- Pre-accession: low `ν_int` (high barriers)
- Post-accession: higher `ν_int` (more integration)

Then compare stationary equilibria before/after and measure:
- change in `{r_i}`, `{w_i}`
- change in `K_i^s` (capital inflows) and output/TFP in EU2004
- changes in the occupational composition and capital allocation across ability types
