import numpy as np
import matplotlib.pyplot as plt
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

def plot_policy_functions(steady_states_path, outdir):
    if not os.path.exists(steady_states_path):
        print(f"File not found: {steady_states_path}")
        return

    data = np.load(steady_states_path)
    a_grid = data['a_grid']
    z_grid = data['z_grid']
    
    # Pre-Reform
    pre_polp = data['pre_polp']
    
    # Post-Reform
    post_pol = data['post_pol']
    
    # Setup dense grid for smoothing
    limit_a = 200.0
    a_dense = np.linspace(a_grid[0], limit_a, 1000)
    
    # Indices for High and Median ability
    idx_high = len(z_grid) - 1
    idx_med  = len(z_grid) // 2
    z_high_val = z_grid[idx_high]
    z_med_val  = z_grid[idx_med]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Plot Pre-Reform ---
    ax = axes[0]
    pol_high = interpolate_policy(a_dense, a_grid, pre_polp, idx_high)
    ax.plot(a_dense, pol_high, color='#D62728', linestyle='-', label=f'High z ({z_high_val:.2f})')
    pol_med = interpolate_policy(a_dense, a_grid, pre_polp, idx_med)
    ax.plot(a_dense, pol_med, color='#1F77B4', linestyle='--', label=f'Median z ({z_med_val:.2f})')
    ax.plot(a_dense, a_dense, color='gray', linestyle=':', linewidth=2.0)
    ax.set_title("Pre-Reform Steady State")
    ax.set_xlabel(r"Current Assets $a$")
    ax.set_ylabel(r"Next Period Assets $a'$")
    ax.set_xlim(0, limit_a)
    ax.set_ylim(0, limit_a)
    
    mask_grid = a_grid <= limit_a
    ag_sub = a_grid[mask_grid]
    pol_high_raw = a_grid[pre_polp[mask_grid, idx_high]]
    pol_med_raw = a_grid[pre_polp[mask_grid, idx_med]]
    ax.scatter(ag_sub, pol_high_raw, s=10, color='#D62728', alpha=0.3, label='Grid Points')
    ax.scatter(ag_sub, pol_med_raw, s=10, color='#1F77B4', alpha=0.3)
    ax.legend(frameon=False, loc='upper left')
    
    # --- Plot Post-Reform ---
    ax = axes[1]
    pol_high = interpolate_policy(a_dense, a_grid, post_pol, idx_high)
    ax.plot(a_dense, pol_high, color='#D62728', linestyle='-', label=f'High z ({z_high_val:.2f})')
    pol_med = interpolate_policy(a_dense, a_grid, post_pol, idx_med)
    ax.plot(a_dense, pol_med, color='#1F77B4', linestyle='--', label=f'Median z ({z_med_val:.2f})')
    pol_high_raw = a_grid[post_pol[mask_grid, idx_high]]
    pol_med_raw = a_grid[post_pol[mask_grid, idx_med]]
    ax.scatter(ag_sub, pol_high_raw, s=10, color='#D62728', alpha=0.3, label='Grid Points')
    ax.scatter(ag_sub, pol_med_raw, s=10, color='#1F77B4', alpha=0.3)
    ax.plot(a_dense, a_dense, color='gray', linestyle=':', linewidth=2.0)
    ax.set_title("Post-Reform Steady State")
    ax.set_xlabel(r"Current Assets $a$")
    ax.set_xlim(0, limit_a)
    ax.set_ylim(0, limit_a)
    ax.legend(frameon=False, loc='upper left')

    for ax in axes:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    plt.tight_layout()
    save_path = os.path.join(outdir, "fig_policy_functions.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Generated: {save_path}")

def plot_transition_dynamics(transition_path, outdir, tmin=-4, tmax=20):
    if not os.path.exists(transition_path):
        print(f"File not found: {transition_path}")
        return

    data = np.load(transition_path)
    t = data['t']
    t_win = np.arange(tmin, tmax + 1)
    n_win = len(t_win)
    
    pre_Y = data['pre_Y']
    pre_K = data['pre_K']
    pre_TFP = data['pre_TFP']
    pre_w = data['pre_w']
    pre_r = data['pre_r']
    
    print(f"DEBUG: Loaded Data from {transition_path}")
    print(f"DEBUG: pre_Y={pre_Y:.4f}, pre_K={pre_K:.4f}")

    # --- POPULATE PLOTTING DATA ---
    Y_plot = np.zeros(n_win)
    K_plot = np.zeros(n_win)
    TFP_plot = np.zeros(n_win)
    w_plot = np.zeros(n_win)
    r_plot = np.zeros(n_win)
    
    for i, tt in enumerate(t_win):
        if tt < 0:
            Y_plot[i] = pre_Y
            K_plot[i] = pre_K
            TFP_plot[i] = pre_TFP
            w_plot[i] = pre_w
            r_plot[i] = pre_r
        else:
            t_idx = int(tt)
            if t_idx < len(data['Y']):
                Y_plot[i] = data['Y'][t_idx]
                K_plot[i] = data['K'][t_idx]
                TFP_plot[i] = data['TFP'][t_idx]
                w_plot[i] = data['w'][t_idx]
                r_plot[i] = data['r'][t_idx]
            else:
                Y_plot[i] = data['Y'][-1]
                K_plot[i] = data['K'][-1]
                TFP_plot[i] = data['TFP'][-1]
                w_plot[i] = data['w'][-1]
                r_plot[i] = data['r'][-1]

    # --- CALCULATE INVESTMENT ---
    delta = 0.06 
    IY_dev = np.zeros(n_win)
    IY_pre = (delta * pre_K) / pre_Y
    
    for i, tt in enumerate(t_win):
        kt = K_plot[i]
        tt_next = tt + 1
        if tt_next < 0:
            kt_next = pre_K
        elif tt_next < len(data['K']):
            kt_next = data['K'][int(tt_next)]
        else:
            kt_next = data['K'][-1]
        
        it = kt_next - (1-delta)*kt
        yt = Y_plot[i]
        iy_ratio = it / yt if yt != 0 else 0
        IY_dev[i] = iy_ratio - IY_pre

    # --- NORMALIZE TRANSITION VARS ---
    Y_n   = Y_plot / pre_Y
    K_n   = K_plot / pre_K
    TFP_n = TFP_plot / pre_TFP
    w_n   = w_plot / pre_w
    r_val = r_plot 
    
    print(f"DEBUG: Normalized Y[0] = {Y_n[t_win==0][0]:.4f}")

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    def style_subplot(ax, x, y, title, ylabel=None, is_level=False):
        mask_pre = x < 0
        mask_post = x >= 0
        
        if np.any(mask_pre):
            ax.plot(x[mask_pre], y[mask_pre], color='black', linewidth=3.0)
        if np.any(mask_post):
            ax.plot(x[mask_post], y[mask_post], color='black', linewidth=3.0)
             
        ax.axvline(0, color='gray', linestyle='-', linewidth=0.8, alpha=1.0)
        
        if not is_level:
             # Logic to detect if normalized (centered at 1) or deviation (centered at 0)
             # Normalized values are usually > 0.5.
             ref_val = 1.0 if np.mean(np.abs(y)) > 0.5 else 0.0
             ax.axhline(ref_val, color='gray', linestyle='-', linewidth=0.5)
        
        ax.set_title(title, fontsize=14)
        if ylabel: ax.set_ylabel(ylabel, fontsize=12)
        ax.tick_params(direction='in', length=5)
        ax.spines['right'].set_visible(True)
        ax.spines['top'].set_visible(True)
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        
    style_subplot(axes[0,0], t_win, Y_n, "GDP (normalized)")
    style_subplot(axes[0,1], t_win, TFP_n, "TFP Measure (normalized)")
    style_subplot(axes[0,2], t_win, IY_dev, "Investment Rate (deviation)")
    
    style_subplot(axes[1,0], t_win, K_n, "Capital Stock (normalized)")
    style_subplot(axes[1,1], t_win, r_val, "Interest Rates (Level)", is_level=True)
    
    axes[1,2].axis('off')

    for ax in axes.flat:
        if ax.get_visible():
            ax.set_xlabel("Years after reform")
            ax.set_xlim(-4, 20)

    plt.tight_layout()
    save_path = os.path.join(outdir, "fig_paper_replication.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Generated: {save_path}")

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
    
    # Re-calculate prob_z based on the same logic as the main script
    # Pareto PDF: f(z) = eta * z^(-eta-1)
    eta = 4.15
    pdf = eta * (z_grid ** (-eta - 1))
    
    # Normalize to show relative mass
    pdf_norm = pdf / np.sum(pdf)

    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.bar(z_grid, pdf_norm, width=0.02, color='#1F77B4', alpha=0.7, label='Grid Point Probability Mass')
    ax.plot(z_grid, pdf_norm, color='#D62728', marker='o', markersize=4, linestyle='--', linewidth=1, label='PDF Trend')
    
    ax.set_title("Ability Distribution (z-grid)")
    ax.set_xlabel("Ability $z$")
    ax.set_ylabel("Probability Mass")
    ax.legend(frameon=False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.grid(True, linestyle=':', alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(outdir, "fig_ability_distribution.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Generated: {save_path}")

def plot_policy_functions_3d(steady_states_path, outdir):
    if not os.path.exists(steady_states_path):
        print(f"File not found: {steady_states_path}")
        return

    data = np.load(steady_states_path)
    a_grid = data['a_grid']
    z_grid = data['z_grid']
    post_pol = data['post_pol']  # shape (n_a, n_z)
    
    # Create dense grid specifically for 3D visualization
    limit_a = 50.0  # Focused on the most active part of the distribution
    a_dense = np.linspace(a_grid[0], limit_a, 40)
    z_dense = np.linspace(z_grid[0], z_grid[-1], 30)
    
    AA, ZZ, pol_dense = interpolate_policy_2d(a_dense, z_dense, a_grid, z_grid, post_pol)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot surface
    surf = ax.plot_surface(AA, ZZ, pol_dense, cmap='viridis', edgecolor='none', alpha=0.8)
    
    # Add wireframe for better perspective
    ax.plot_wireframe(AA, ZZ, pol_dense, color='black', linewidth=0.3, alpha=0.3)
    
    # Set labels
    ax.set_xlabel(r"Current Assets $a$")
    ax.set_ylabel(r"Ability $z$")
    ax.set_zlabel(r"Next Period Assets $a'$")
    ax.set_title("Post-Reform Asset Policy Function $a'(a, z)$")
    
    # Colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label="Next Period Assets")
    
    # Improve view angle
    ax.view_init(elev=25, azim=-125)
    
    save_path = os.path.join(outdir, "fig_policy_3d.png")
    plt.savefig(save_path, dpi=200)
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
    plot_policy_functions_3d(steady_path, args.out)

if __name__ == "__main__":
    main()
