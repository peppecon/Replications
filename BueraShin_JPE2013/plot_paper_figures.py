import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import argparse

# Global Style Settings for "QJE" Look
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'lines.linewidth': 2.5,
    'axes.linewidth': 1.5,
    'xtick.major.size': 6,
    'ytick.major.size': 6,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'figure.titlesize': 18,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

# Model Parameters (match Buera & Shin calibration)
ALPHA  = 0.33
NU     = 0.21         # 1-nu = span of control
DELTA  = 0.06
LAMBDA = 1.35         # Collateral constraint
TAU_PLUS = 0.57       # Tax distortion
TAU_MINUS = -0.15     # Subsidy distortion

def interpolate_policy(a_dense, a_grid, pol_idx, z_idx):
    """
    Linearly interpolates the policy function for a dense grid of asset holdings.
    """
    # Policy values at grid points (in levels)
    pol_vals_grid = a_grid[pol_idx[:, z_idx]]
    
    # Interpolate
    pol_dense = np.interp(a_dense, a_grid, pol_vals_grid)
    
    return pol_dense

def interpolate_policy_2d(a_dense, z_dense, a_grid, z_grid, pol_idx):
    """
    Performs 2D interpolation for the policy surface.
    pol_idx is (n_a, n_z) indices.
    """
    from scipy.interpolate import RegularGridInterpolator
    
    # Get values on the original grid
    vals = a_grid[pol_idx]
    
    # Setup interpolator
    interp = RegularGridInterpolator((a_grid, z_grid), vals, bounds_error=False, fill_value=None)
    
    # Create meshgrid for dense points
    AA, ZZ = np.meshgrid(a_dense, z_dense, indexing='ij')
    points = np.stack([AA.ravel(), ZZ.ravel()], axis=-1)
    
    # Interpolate
    pol_dense = interp(points).reshape(AA.shape)
    
    return AA, ZZ, pol_dense

def solve_occupational_choice(a, z, w, r, lam, delta, alpha, nu, tau=0.0):
    """
    Returns 1 if agent chooses to be an entrepreneur, 0 otherwise.
    Replicates the logic of solve_firm_no_tau from the simulation.
    """
    z_eff = (1.0 - tau) * z
    if z_eff <= 0: return 0.0
    
    rental = r + delta
    span = 1.0 - nu
    
    # Closed-form optimal unconstrained k
    aux1 = (alpha * span * z_eff) / max(rental, 1e-10)
    aux2 = ((1.0 - alpha) * span * z_eff) / max(w, 1e-10)
    exp1 = 1.0 - (1.0 - alpha) * span
    exp2 = (1.0 - alpha) * span

    k_uncon = (aux1 ** exp1 * aux2 ** exp2) ** (1.0 / nu)
    k = min(k_uncon, lam * a)
    
    # Labor from FOC given k
    l = (aux2 * (k ** (alpha * span))) ** (1.0 / exp1)
    
    # Output and Profit
    y = z_eff * ((k ** alpha) * (l ** (1.0 - alpha))) ** span
    profit = y - w * l - rental * k
    
    return 1.0 if profit > w else 0.0

def plot_policy_functions(steady_states_path, outdir):
    if not os.path.exists(steady_states_path):
        print(f"File not found: {steady_states_path}")
        return

    data = np.load(steady_states_path)
    a_grid = data['a_grid']
    z_grid = data['z_grid']
    n_z = len(z_grid)

    # Pre-Reform
    pre_polp = data['pre_polp']

    # Post-Reform
    post_pol = data['post_pol']

    # Setup dense grid for smoothing
    limit_a = 200.0
    a_dense = np.linspace(a_grid[0], limit_a, 1000)

    # Use meaningful percentiles: 50th (median), 75th, 90th, 99th (max)
    # With equal-probability bins, index i corresponds to percentile ~ i/n_z
    idx_p50 = n_z // 2                    # ~50th percentile
    idx_p75 = int(0.75 * n_z)             # ~75th percentile
    idx_p90 = int(0.90 * n_z)             # ~90th percentile
    idx_p99 = n_z - 1                     # ~99th percentile (max in grid)

    z_p50 = z_grid[idx_p50]
    z_p75 = z_grid[idx_p75]
    z_p90 = z_grid[idx_p90]
    z_p99 = z_grid[idx_p99]

    print(f"  z-grid: min={z_grid[0]:.3f}, max={z_grid[-1]:.3f}, n_z={n_z}")
    print(f"  Selected percentiles: p50={z_p50:.2f}, p75={z_p75:.2f}, p90={z_p90:.2f}, p99={z_p99:.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = ['#1F77B4', '#2CA02C', '#FF7F0E', '#D62728']
    styles = ['--', '-.', '-', '-']
    indices = [idx_p50, idx_p75, idx_p90, idx_p99]
    labels = [f'p50 (z={z_p50:.2f})', f'p75 (z={z_p75:.2f})',
              f'p90 (z={z_p90:.2f})', f'p99 (z={z_p99:.2f})']

    # --- Plot Pre-Reform ---
    ax = axes[0]
    for idx, col, sty, lab in zip(indices, colors, styles, labels):
        pol = interpolate_policy(a_dense, a_grid, pre_polp, idx)
        ax.plot(a_dense, pol, color=col, linestyle=sty, label=lab, linewidth=2.0)
    ax.plot(a_dense, a_dense, color='gray', linestyle=':', linewidth=1.5, label='45° line')
    ax.set_title("Pre-Reform Steady State")
    ax.set_xlabel(r"Current Assets $a$")
    ax.set_ylabel(r"Next Period Assets $a'$")
    ax.set_xlim(0, limit_a)
    ax.set_ylim(0, limit_a)
    ax.legend(frameon=False, loc='upper left', fontsize=11)

    # --- Plot Post-Reform ---
    ax = axes[1]
    for idx, col, sty, lab in zip(indices, colors, styles, labels):
        pol = interpolate_policy(a_dense, a_grid, post_pol, idx)
        ax.plot(a_dense, pol, color=col, linestyle=sty, label=lab, linewidth=2.0)
    ax.plot(a_dense, a_dense, color='gray', linestyle=':', linewidth=1.5, label='45° line')
    ax.set_title("Post-Reform Steady State")
    ax.set_xlabel(r"Current Assets $a$")
    ax.set_xlim(0, limit_a)
    ax.set_ylim(0, limit_a)
    ax.legend(frameon=False, loc='upper left', fontsize=11)

    for ax in axes:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    plt.tight_layout()
    save_path = os.path.join(outdir, "fig_policy_functions.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Generated: {save_path}")

def load_transition_data(filepath, tmin, tmax):
    """Helper to load and prepare transition data for plotting."""
    if not os.path.exists(filepath):
        return None

    data = np.load(filepath)
    t_win = np.arange(tmin, tmax + 1)
    n_win = len(t_win)

    pre_Y = float(data['pre_Y'])
    pre_K = float(data['pre_K'])
    pre_TFP = float(data['pre_TFP'])
    pre_r = float(data['pre_r'])

    # Populate arrays
    Y_plot = np.zeros(n_win)
    K_plot = np.zeros(n_win)
    TFP_plot = np.zeros(n_win)
    r_plot = np.zeros(n_win)

    for i, tt in enumerate(t_win):
        if tt < 0:
            Y_plot[i] = pre_Y
            K_plot[i] = pre_K
            TFP_plot[i] = pre_TFP
            r_plot[i] = pre_r
        else:
            t_idx = int(tt)
            if t_idx < len(data['Y']):
                Y_plot[i] = data['Y'][t_idx]
                K_plot[i] = data['K'][t_idx]
                TFP_plot[i] = data['TFP'][t_idx]
                r_plot[i] = data['r'][t_idx]
            else:
                Y_plot[i] = data['Y'][-1]
                K_plot[i] = data['K'][-1]
                TFP_plot[i] = data['TFP'][-1]
                r_plot[i] = data['r'][-1]

    # Compute investment rate as deviation from pre-reform
    # I_t = K_{t+1} - (1-delta)*K_t, then I/Y deviation
    delta = DELTA
    IY_pre = (delta * pre_K) / pre_Y  # Steady state I/Y = delta*K/Y
    IY_dev = np.zeros(n_win)

    # Use the full K series for proper calculation
    K_full = np.concatenate([[pre_K] * abs(tmin), data['K'][:]])

    for i, tt in enumerate(t_win):
        idx = tt - tmin  # Index into K_full
        if idx + 1 < len(K_full):
            kt = K_full[idx]
            kt_next = K_full[idx + 1]
        else:
            kt = K_full[-1]
            kt_next = K_full[-1]  # Steady state

        it = kt_next - (1 - delta) * kt
        yt = Y_plot[i]
        iy_ratio = it / yt if yt > 1e-10 else 0
        IY_dev[i] = iy_ratio - IY_pre

    # Load micro metrics if available
    avg_z = data.get('avg_z_entre', None)
    ws_top5 = data.get('wealth_share_top5', None)
    pre_avg_z = float(data.get('pre_avg_z_entre', 1.0))
    pre_ws_top5 = float(data.get('pre_wealth_share_top5', 0.25))

    avg_z_plot = np.zeros(n_win)
    ws_top5_plot = np.zeros(n_win)

    if avg_z is not None and len(avg_z) > 0:
        for i, tt in enumerate(t_win):
            if tt < 0:
                avg_z_plot[i] = pre_avg_z
                ws_top5_plot[i] = pre_ws_top5
            else:
                t_idx = int(tt)
                if t_idx < len(avg_z):
                    avg_z_plot[i] = avg_z[t_idx]
                    ws_top5_plot[i] = ws_top5[t_idx]
                else:
                    avg_z_plot[i] = avg_z[-1]
                    ws_top5_plot[i] = ws_top5[-1]

    # Post-reform steady state values (from end of transition)
    post_Y = float(data['Y'][-1])
    post_K = float(data['K'][-1])
    post_TFP = float(data['TFP'][-1])
    post_r = float(data['r'][-1])

    # Post-reform I/Y in steady state
    IY_post = (delta * post_K) / post_Y
    IY_dev_post = IY_post - IY_pre  # Deviation from pre-reform

    return {
        't_win': t_win,
        'Y_n': Y_plot / pre_Y,
        'K_n': K_plot / pre_K,
        'TFP_n': TFP_plot / pre_TFP,
        'r': r_plot,
        'IY_dev': IY_dev,
        'avg_z_n': avg_z_plot / pre_avg_z if pre_avg_z > 0 else avg_z_plot,
        'ws_top5': ws_top5_plot,
        # Pre-reform values (normalized = 1.0 for Y, K, TFP)
        'pre_Y': pre_Y, 'pre_K': pre_K, 'pre_TFP': pre_TFP, 'pre_r': pre_r,
        # Post-reform steady state values (normalized)
        'post_Y_n': post_Y / pre_Y,
        'post_K_n': post_K / pre_K,
        'post_TFP_n': post_TFP / pre_TFP,
        'post_r': post_r,
        'IY_dev_post': IY_dev_post,
        'pre_avg_z_n': 1.0,  # Normalized pre = 1.0
        'post_avg_z_n': avg_z_plot[-1] / pre_avg_z if pre_avg_z > 0 else 1.0,
        'pre_ws_top5': pre_ws_top5,
        'post_ws_top5': ws_top5_plot[-1] if len(ws_top5_plot) > 0 else 0.0,
    }


def plot_transition_dynamics(transition_path, outdir, tmin=-4, tmax=20):
    """
    Plots transition dynamics comparing benchmark (λ=1.35) and neoclassical (λ=∞).
    Replicates Figures 3, 4, and 5 from Buera & Shin (2013).
    """
    # Load benchmark transition
    bench = load_transition_data(transition_path, tmin, tmax)
    if bench is None:
        print(f"File not found: {transition_path}")
        return

    # Try to load neoclassical comparison
    nc_path = os.path.join(outdir, "transition_neoclassical.npz")
    nc = load_transition_data(nc_path, tmin, tmax)

    t_win = bench['t_win']

    print(f"Loaded benchmark transition: pre_Y={bench['pre_Y']:.4f}, pre_K={bench['pre_K']:.4f}")
    if nc:
        print(f"Loaded neoclassical transition for comparison")

    # =========================================================================
    # Figure 3 & 4: Aggregate Dynamics
    # =========================================================================
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    def plot_series(ax, x, y_bench, y_nc, title, pre_ss=None, post_ss=None, is_level=False):
        mask_pre = x < 0
        mask_post = x >= 0

        # Benchmark (λ=1.35) - solid black
        if np.any(mask_pre):
            ax.plot(x[mask_pre], y_bench[mask_pre], color='black', linewidth=2.5)
        if np.any(mask_post):
            ax.plot(x[mask_post], y_bench[mask_post], color='black', linewidth=2.5, label='λ=1.35')

        # Neoclassical (λ=∞) - dotted
        if y_nc is not None:
            if np.any(mask_pre):
                ax.plot(x[mask_pre], y_nc[mask_pre], color='black', linewidth=2.0, linestyle=':')
            if np.any(mask_post):
                ax.plot(x[mask_post], y_nc[mask_post], color='black', linewidth=2.0, linestyle=':', label='λ=∞')

        ax.axvline(0, color='gray', linestyle='-', linewidth=0.8, alpha=0.7)

        # Pre-reform steady state reference line (blue dashed)
        if pre_ss is not None:
            ax.axhline(pre_ss, color='#1f77b4', linestyle='--', linewidth=1.5, alpha=0.8, label='Pre-SS')

        # Post-reform steady state reference line (red dashed)
        if post_ss is not None:
            ax.axhline(post_ss, color='#d62728', linestyle='--', linewidth=1.5, alpha=0.8, label='Post-SS')

        ax.set_title(title, fontsize=14)
        ax.tick_params(direction='in', length=5)
        ax.spines['right'].set_visible(True)
        ax.spines['top'].set_visible(True)
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')

    # Row 1: GDP, TFP, Investment Rate
    plot_series(axes[0, 0], t_win, bench['Y_n'], nc['Y_n'] if nc else None, "GDP (normalized)",
                pre_ss=1.0, post_ss=bench['post_Y_n'])
    plot_series(axes[0, 1], t_win, bench['TFP_n'], nc['TFP_n'] if nc else None, "TFP Measure (normalized)",
                pre_ss=1.0, post_ss=bench['post_TFP_n'])
    plot_series(axes[0, 2], t_win, bench['IY_dev'], nc['IY_dev'] if nc else None,
                "Investment Rate (deviation)", pre_ss=0.0, post_ss=bench['IY_dev_post'])

    # Row 2: Capital, Interest Rates
    plot_series(axes[1, 0], t_win, bench['K_n'], nc['K_n'] if nc else None, "Capital Stock (normalized)",
                pre_ss=1.0, post_ss=bench['post_K_n'])
    plot_series(axes[1, 1], t_win, bench['r'], nc['r'] if nc else None, "Interest Rates (level)",
                pre_ss=bench['pre_r'], post_ss=bench['post_r'], is_level=True)

    # Legend in bottom right panel
    axes[1, 2].axis('off')
    # Add a combined legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        axes[1, 2].legend(handles, labels, loc='center', frameon=True, fontsize=12)

    for ax in axes.flat:
        if ax.get_visible():
            ax.set_xlabel("Years after reform")
            ax.set_xlim(tmin, tmax)

    plt.tight_layout()
    save_path = os.path.join(outdir, "fig_paper_replication.png")
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Generated: {save_path}")

    # =========================================================================
    # Figure 5: Micro Implications (if data available)
    # =========================================================================
    if np.any(bench['avg_z_n'] > 0):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Avg Entrepreneurial Ability (normalized)
        ax = axes[0]
        mask_pre = t_win < 0
        mask_post = t_win >= 0
        if np.any(mask_pre):
            ax.plot(t_win[mask_pre], bench['avg_z_n'][mask_pre], color='black', linewidth=2.5)
        if np.any(mask_post):
            ax.plot(t_win[mask_post], bench['avg_z_n'][mask_post], color='black', linewidth=2.5)
        ax.axvline(0, color='gray', linestyle='-', linewidth=0.8, alpha=0.7)
        ax.axhline(1.0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
        ax.set_title("Avg. Entrep. Ability (normalized)", fontsize=14)
        ax.set_xlabel("Years after reform")
        ax.set_xlim(tmin, tmax)

        # Wealth Share of Top 5% Ability
        ax = axes[1]
        if np.any(mask_pre):
            ax.plot(t_win[mask_pre], bench['ws_top5'][mask_pre], color='black', linewidth=2.5)
        if np.any(mask_post):
            ax.plot(t_win[mask_post], bench['ws_top5'][mask_post], color='black', linewidth=2.5)
        ax.axvline(0, color='gray', linestyle='-', linewidth=0.8, alpha=0.7)
        ax.set_title("Wealth Share of Top 5% Ability", fontsize=14)
        ax.set_xlabel("Years after reform")
        ax.set_xlim(tmin, tmax)
        ax.set_ylim(0, 0.7)

        for ax in axes:
            ax.tick_params(direction='in', length=5)
            ax.spines['right'].set_visible(True)
            ax.spines['top'].set_visible(True)

        plt.tight_layout()
        save_path = os.path.join(outdir, "fig_micro_implications.png")
        plt.savefig(save_path, dpi=200)
        plt.close()
        print(f"Generated: {save_path}")
    else:
        print("  Skipping micro implications plot (no data available)")

def plot_wealth_distribution(steady_states_path, outdir):
    if not os.path.exists(steady_states_path):
        print(f"File not found: {steady_states_path}")
        return

    data = np.load(steady_states_path)
    a_grid = data['a_grid']
    post_mu = data['post_mu']
    pre_mu_p = data['pre_mu_p']
    pre_mu_m = data['pre_mu_m']

    post_mass_a = np.sum(post_mu, axis=1)
    pre_mass_a = np.sum(pre_mu_p, axis=1) + np.sum(pre_mu_m, axis=1)
    
    post_mass_a /= np.sum(post_mass_a)
    pre_mass_a /= np.sum(pre_mass_a)
    
    post_cdf = np.cumsum(post_mass_a)
    pre_cdf = np.cumsum(pre_mass_a)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. SAMPLED HISTOGRAM
    def get_distribution_samples(mass, grid, n_samples=100000):
        # We sample from the discrete grid points proportional to their mass
        return np.random.choice(grid, size=n_samples, p=mass)

    pre_samples = get_distribution_samples(pre_mass_a, a_grid)
    post_samples = get_distribution_samples(post_mass_a, a_grid)

    ax = axes[0]
    bins = np.linspace(0, 50, 60) # focused on wealth below 50
    ax.hist(pre_samples, bins=bins, color='gray', alpha=0.4, label='Pre-Reform', density=True)
    ax.hist(post_samples, bins=bins, histtype='step', color='black', linewidth=3, label='Post-Reform', density=True)
    
    ax.set_title("Asset Distribution (Histogram)")
    ax.set_xlabel("Assets $a$")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlim(0, 50)
    
    # 2. CDF
    ax = axes[1]
    ax.plot(a_grid, pre_cdf, color='gray', linestyle='--', linewidth=2, label='Pre-Reform CDF')
    ax.plot(a_grid, post_cdf, color='black', linestyle='-', linewidth=2, label='Post-Reform CDF')
    ax.set_title("Asset CDF")
    ax.set_xlabel("Assets $a$")
    ax.set_ylabel("Cumulative Probability")
    ax.set_xlim(0, 100) 
    ax.legend(frameon=False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.grid(True, linestyle=':', alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(outdir, "fig_wealth_distribution.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Generated: {save_path}")

def plot_ability_distribution(steady_states_path, outdir):
    if not os.path.exists(steady_states_path):
        print(f"File not found: {steady_states_path}")
        return

    data = np.load(steady_states_path)
    z_grid = data['z_grid']
    n_z = len(z_grid)

    # The grid uses equal-probability bins, so each z has equal prob mass = 1/n_z
    prob_z = np.ones(n_z) / n_z

    # Also compute the theoretical Pareto PDF for comparison
    eta = 4.15
    z_dense = np.linspace(z_grid[0], z_grid[-1], 500)
    pdf_pareto = eta * (z_dense ** (-eta - 1))
    # Normalize to integrate to 1 over the shown range
    dz = z_dense[1] - z_dense[0]
    pdf_pareto = pdf_pareto / (np.sum(pdf_pareto) * dz)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Left: Grid point locations with equal probability ---
    ax = axes[0]
    ax.stem(z_grid, prob_z, linefmt='#1F77B4', markerfmt='o', basefmt=' ')
    ax.set_title(f"Ability Grid Points (n={n_z}, equal probability)")
    ax.set_xlabel("Ability $z$")
    ax.set_ylabel("Probability Mass")
    ax.set_xlim(z_grid[0] - 0.05, z_grid[-1] + 0.1)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.axhline(1/n_z, color='gray', linestyle='--', alpha=0.5, label=f'Equal mass = 1/{n_z}')
    ax.legend(frameon=False)

    # --- Right: Grid density vs Pareto PDF ---
    ax = axes[1]
    # Show how grid points are distributed: histogram of z values
    ax.hist(z_grid, bins=30, density=True, color='#1F77B4', alpha=0.6, label='Grid point density')
    ax.plot(z_dense, pdf_pareto, color='#D62728', linewidth=2, label=f'Pareto PDF ($\\eta$={eta})')
    ax.set_title("Grid Density vs Pareto Distribution")
    ax.set_xlabel("Ability $z$")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlim(z_grid[0] - 0.05, z_grid[-1] + 0.1)

    plt.tight_layout()
    save_path = os.path.join(outdir, "fig_ability_distribution.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Generated: {save_path}")

def plot_asset_policy_contour(steady_states_path, outdir):
    if not os.path.exists(steady_states_path):
        print(f"File not found: {steady_states_path}")
        return

    data = np.load(steady_states_path)
    a_grid = data['a_grid']
    z_grid = data['z_grid']
    post_pol = data['post_pol']
    pre_polp = data['pre_polp']  # Taxed type

    n_z = len(z_grid)
    print(f"  Contour plot: z_grid range [{z_grid[0]:.2f}, {z_grid[-1]:.2f}], n_z={n_z}")

    # Create dense grid for smooth contours
    limit_a = 150.0
    a_dense = np.linspace(a_grid[0], limit_a, 150)
    z_dense = np.linspace(z_grid[0], z_grid[-1], 150)

    # Interpolate both surfaces
    AA_pre, ZZ_pre, pol_pre = interpolate_policy_2d(a_dense, z_dense, a_grid, z_grid, pre_polp)
    AA_post, ZZ_post, pol_post = interpolate_policy_2d(a_dense, z_dense, a_grid, z_grid, post_pol)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    levels = np.linspace(0, limit_a, 25)

    # --- Plot 1: Pre-Reform ---
    ax = axes[0]
    cf1 = ax.contourf(AA_pre, ZZ_pre, pol_pre, levels=levels, cmap='viridis')
    ax.contour(AA_pre, ZZ_pre, pol_pre, levels=levels[::2], colors='white', linewidths=0.4, alpha=0.4)
    ax.set_title("Pre-Reform: $a'(a, z)$ (Taxed)", fontsize=14)
    cbar1 = fig.colorbar(cf1, ax=ax, shrink=0.85)
    cbar1.set_label("$a'$", fontsize=12)

    # --- Plot 2: Post-Reform ---
    ax = axes[1]
    cf2 = ax.contourf(AA_post, ZZ_post, pol_post, levels=levels, cmap='viridis')
    ax.contour(AA_post, ZZ_post, pol_post, levels=levels[::2], colors='white', linewidths=0.4, alpha=0.4)
    ax.set_title("Post-Reform: $a'(a, z)$ (Undistorted)", fontsize=14)
    cbar2 = fig.colorbar(cf2, ax=ax, shrink=0.85)
    cbar2.set_label("$a'$", fontsize=12)

    for ax in axes:
        ax.set_xlabel("Current Assets $a$", fontsize=12)
        ax.set_ylabel("Ability $z$", fontsize=12)
        # Mark key percentile lines on z-axis
        for pct, ls in [(0.5, '--'), (0.75, ':'), (0.9, '-.')]:
            z_pct = z_grid[int(pct * n_z)]
            ax.axhline(z_pct, color='white', linestyle=ls, linewidth=0.8, alpha=0.6)

    plt.suptitle(f"Asset Savings Policy $a'(a,z)$ — z-grid: [{z_grid[0]:.2f}, {z_grid[-1]:.2f}]",
                 fontsize=16, y=0.98)
    plt.tight_layout()

    save_path = os.path.join(outdir, "fig_asset_policy_contour.png")
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Generated: {save_path}")

def plot_occupational_choice_contour(steady_states_path, outdir):
    if not os.path.exists(steady_states_path):
        print(f"File not found: {steady_states_path}")
        return

    data = np.load(steady_states_path)
    a_grid = data['a_grid']
    z_grid = data['z_grid']
    
    # Prices
    post_w, post_r = data['post_w'], data['post_r']
    pre_w, pre_r = data['pre_w'], data['pre_r']
    
    # Create dense grid
    limit_a = 50.0 
    a_dense = np.linspace(a_grid[0], limit_a, 200)
    z_dense = np.linspace(z_grid[0], z_grid[-1], 200)
    AA, ZZ = np.meshgrid(a_dense, z_dense, indexing='ij')

    # Compute choices
    OCC_post = np.zeros(AA.shape)
    OCC_pre_plus = np.zeros(AA.shape)
    OCC_pre_minus = np.zeros(AA.shape)
    
    for i in range(len(a_dense)):
        for j in range(len(z_dense)):
            OCC_post[i, j] = solve_occupational_choice(a_dense[i], z_dense[j], post_w, post_r, LAMBDA, DELTA, ALPHA, NU, tau=0.0)
            OCC_pre_plus[i, j] = solve_occupational_choice(a_dense[i], z_dense[j], pre_w, pre_r, LAMBDA, DELTA, ALPHA, NU, tau=TAU_PLUS)
            OCC_pre_minus[i, j] = solve_occupational_choice(a_dense[i], z_dense[j], pre_w, pre_r, LAMBDA, DELTA, ALPHA, NU, tau=TAU_MINUS)

    # Pre-reform aggregate choice for shading: 
    # 0 = Worker for both
    # 1 = Entre for subsidized only (Sensitive zone)
    # 2 = Entre for both
    OCC_pre_agg = OCC_pre_plus + OCC_pre_minus

    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    
    # Precise Color Palette
    color_worker = '#F8F9F9'  # Very Light Gray
    color_entre  = '#FEEBEE'  # Soft Peach/Red
    color_zone   = '#E3F2FD'  # Soft Sky Blue

    # Create handles for the legend
    patch_worker = mpatches.Patch(color=color_worker, label='Always Worker')
    patch_zone   = mpatches.Patch(color=color_zone, label='Dependent on distortion draw')
    patch_entre  = mpatches.Patch(color=color_entre, label='Always Entrepreneur')

    fig, axes = plt.subplots(1, 2, figsize=(18, 10))
    
    # --- Plot 1: Pre-Reform ---
    ax = axes[0]
    ax.contourf(AA, ZZ, OCC_pre_agg, levels=[-0.5, 0.5, 1.5, 2.5], colors=[color_worker, color_zone, color_entre])
    
    l1 = ax.contour(AA, ZZ, OCC_pre_plus, levels=[0.5], colors='#D32F2F', linewidths=3.5)
    l3 = ax.contour(AA, ZZ, OCC_pre_minus, levels=[0.5], colors='#388E3C', linewidths=3.5)
    
    ax.set_title("Pre-Reform Choice Boundary", fontsize=20, fontweight='bold', pad=20)
    
    # --- Plot 2: Post-Reform ---
    ax = axes[1]
    ax.contourf(AA, ZZ, OCC_post, levels=[-0.5, 0.5, 1.5], colors=[color_worker, color_entre])
    l_post = ax.contour(AA, ZZ, OCC_post, levels=[0.5], colors='black', linewidths=3.5)
    ax.set_title("Post-Reform Choice Boundary", fontsize=20, fontweight='bold', pad=20)
    
    # Common Formatting
    for ax in axes:
        ax.set_xlabel("Assets $a$", fontsize=16)
        ax.set_ylabel("Ability $z$", fontsize=16)
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlim(0, limit_a)
        ax.set_ylim(z_grid[0], z_grid[-1])
        ax.tick_params(axis='both', which='major', labelsize=12)

    # Create a Unified Legend at the bottom
    h1, _ = l1.legend_elements(); h3, _ = l3.legend_elements(); hp, _ = l_post.legend_elements()
    
    handles = [
        h1[0], h3[0], hp[0],
        patch_worker, patch_zone, patch_entre
    ]
    labels = [
        f'Threshold (Taxed $\\tau={TAU_PLUS}$)',
        f'Threshold (Subsidized $\\tau={TAU_MINUS}$)',
        'Unified Entry Threshold',
        'Always Worker Region',
        'Dependent on distortion draw',
        'Always Entrepreneur Region'
    ]
    
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=16, 
               frameon=True, facecolor='white', framealpha=1.0, shadow=True,
               bbox_to_anchor=(0.5, -0.05))

    plt.suptitle("Impact of Distortions on Occupational Choice", fontsize=26, fontweight='bold', y=1.05)
    plt.tight_layout()
    
    save_path = os.path.join(outdir, "fig_occupational_contour.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Generated: {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="outputs", help="Directory containing .npz files")
    args = parser.parse_args()

    print(f"Generating fancy plots from: {args.out}")
    
    steady_path = os.path.join(args.out, "steady_states.npz")
    trans_path = os.path.join(args.out, "transition_path.npz")
    
    plot_policy_functions(steady_path, args.out)
    plot_wealth_distribution(steady_path, args.out)
    plot_transition_dynamics(trans_path, args.out)
    plot_ability_distribution(steady_path, args.out)
    plot_asset_policy_contour(steady_path, args.out)
    plot_occupational_choice_contour(steady_path, args.out)

if __name__ == "__main__":
    main()
