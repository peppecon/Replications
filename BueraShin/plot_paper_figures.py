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
    This reconstructs the 'effective' policy used in the simulation, eliminating
    the visual 'staircase' effect of the discrete grid.
    """
    # Policy values at grid points (in levels)
    pol_vals_grid = a_grid[pol_idx[:, z_idx]]
    
    # Interpolate
    # np.interp expects x to be increasing, which a_grid is.
    pol_dense = np.interp(a_dense, a_grid, pol_vals_grid)
    
    return pol_dense

def plot_policy_functions(steady_states_path, outdir):
    if not os.path.exists(steady_states_path):
        print(f"File not found: {steady_states_path}")
        return

    data = np.load(steady_states_path)
    a_grid = data['a_grid']
    z_grid = data['z_grid']
    
    # Pre-Reform
    pre_polp = data['pre_polp'] # Matrix of indices (na, nz)
    
    # Post-Reform
    post_pol = data['post_pol'] # Matrix of indices
    
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
    # High Z
    pol_high = interpolate_policy(a_dense, a_grid, pre_polp, idx_high)
    ax.plot(a_dense, pol_high, color='#D62728', linestyle='-', label=f'High z ({z_high_val:.1f})')
    
    # Median Z
    pol_med = interpolate_policy(a_dense, a_grid, pre_polp, idx_med)
    ax.plot(a_dense, pol_med, color='#1F77B4', linestyle='--', label=f'Median z ({z_med_val:.1f})')
    
    # 45-degree line
    ax.plot(a_dense, a_dense, color='gray', linestyle=':', linewidth=2.0)
    
    ax.set_title("Pre-Reform Steady State")
    ax.set_xlabel(r"Current Assets $a$")
    ax.set_ylabel(r"Next Period Assets $a'$")
    ax.set_xlim(0, limit_a)
    ax.set_ylim(0, limit_a)
    
    # Overlay raw grid choices (to show "points chosen on the grid")
    # Subsample for visibility if scale is large, but for 200 it's fine.
    # We plot the actual grid points (a_grid[i], a_grid[pre_polp[i, z]])
    mask_grid = a_grid <= limit_a
    ag_sub = a_grid[mask_grid]
    pol_high_raw = a_grid[pre_polp[mask_grid, idx_high]]
    pol_med_raw = a_grid[pre_polp[mask_grid, idx_med]]
    
    ax.scatter(ag_sub, pol_high_raw, s=10, color='#D62728', alpha=0.3, label='Grid Points')
    ax.scatter(ag_sub, pol_med_raw, s=10, color='#1F77B4', alpha=0.3)

    ax.legend(frameon=False, loc='upper left')
    
    # --- Plot Post-Reform ---
    ax = axes[1]
    # High Z
    pol_high = interpolate_policy(a_dense, a_grid, post_pol, idx_high)
    ax.plot(a_dense, pol_high, color='#D62728', linestyle='-', label=f'High z ({z_high_val:.1f})')
    
    # Median Z
    pol_med = interpolate_policy(a_dense, a_grid, post_pol, idx_med)
    ax.plot(a_dense, pol_med, color='#1F77B4', linestyle='--', label=f'Median z ({z_med_val:.1f})')
    
    # Overlay raw grid choices
    pol_high_raw = a_grid[post_pol[mask_grid, idx_high]]
    pol_med_raw = a_grid[post_pol[mask_grid, idx_med]]
    
    ax.scatter(ag_sub, pol_high_raw, s=10, color='#D62728', alpha=0.3, label='Grid Points')
    ax.scatter(ag_sub, pol_med_raw, s=10, color='#1F77B4', alpha=0.3)
    
    # 45-degree line
    ax.plot(a_dense, a_dense, color='gray', linestyle=':', linewidth=2.0)

    ax.set_title("Post-Reform Steady State")
    ax.set_xlabel(r"Current Assets $a$")
    ax.set_xlim(0, limit_a)
    ax.set_ylim(0, limit_a)
    ax.legend(frameon=False, loc='upper left')

    # Despine top and right for cleaner look
    for ax in axes:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    plt.tight_layout()
    save_path = os.path.join(outdir, "fig_policy_functions.pdf")
    plt.savefig(save_path)
    # Also save png for quick preview
    plt.savefig(save_path.replace('.pdf', '.png'))
    plt.close()
    print(f"Generated: {save_path}")

def plot_transition_dynamics(transition_path, outdir, tmin=-4, tmax=20):
    if not os.path.exists(transition_path):
        print(f"File not found: {transition_path}")
        return

    data = np.load(transition_path)
    
    # Load data
    t = data['t'] # 0 to T
    # Prepare full window vectors
    t_win = np.arange(tmin, tmax + 1)
    n_win = len(t_win)
    
    # Extract pre-values
    pre_Y = data['pre_Y']
    pre_K = data['pre_K']
    pre_TFP = data['pre_TFP']
    pre_w = data['pre_w']
    pre_r = data['pre_r']
    
    # Arrays to plot
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
            # tt is index if t starts at 0
            if tt < len(t):
                Y_plot[i] = data['Y'][tt]
                K_plot[i] = data['K'][tt]
                TFP_plot[i] = data['TFP'][tt]
                w_plot[i] = data['w'][tt]
                r_plot[i] = data['r'][tt]
            else:
                # Out of bounds (shouldn't happen with default settings)
                Y_plot[i] = data['Y'][-1]
                K_plot[i] = data['K'][-1]
                TFP_plot[i] = data['TFP'][-1]
                w_plot[i] = data['w'][-1]
                r_plot[i] = data['r'][-1]

    # Normalize
    Y_n = Y_plot / pre_Y
    K_n = K_plot / pre_K
    TFP_n = TFP_plot / pre_TFP
    w_n = w_plot / pre_w
    # Interest rate: gross rate ratio
    r_n = (1.0 + r_plot) / (1.0 + pre_r)

    # Plotting
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # Helper to standardize subplots
    def style_subplot(ax, x, y, title, ylabel=None):
        ax.plot(x, y, color='#2C3E50', linewidth=3.0)
        ax.axvline(0, color='gray', linestyle='--', linewidth=1.5, alpha=0.8)
        ax.set_title(title)
        if ylabel: ax.set_ylabel(ylabel)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.grid(True, linestyle=':', alpha=0.4)

    style_subplot(axes[0,0], t_win, Y_n, "Output (Y)", "Relative to Pre-Reform")
    style_subplot(axes[0,1], t_win, TFP_n, "TFP", "Relative to Pre-Reform")
    style_subplot(axes[0,2], t_win, K_n, "Capital (K)", "Relative to Pre-Reform")
    
    style_subplot(axes[1,0], t_win, w_n, "Wage (w)", "Relative to Pre-Reform")
    style_subplot(axes[1,1], t_win, r_n, "Interest Rate (1+r)", "Ratio (1+r)/(1+r_pre)")
    
    # Excess Demand (Absolute)
    # Reconstruct full ED vectors
    ED_L_full = np.zeros(n_win)
    ED_K_full = np.zeros(n_win)
    for i, tt in enumerate(t_win):
        if tt >= 0 and tt < len(t):
            ED_L_full[i] = data['ED_L'][tt]
            ED_K_full[i] = data['ED_K'][tt]
            
    axes[1,2].plot(t_win, np.abs(ED_L_full), label='Labor Market', color='#E67E22', linewidth=2.5)
    axes[1,2].plot(t_win, np.abs(ED_K_full), label='Capital Market', color='#8E44AD', linewidth=2.5)
    axes[1,2].axvline(0, color='gray', linestyle='--', linewidth=1.5, alpha=0.8)
    axes[1,2].set_title("Market Clearing Errors (Abs)")
    axes[1,2].legend(frameon=False)
    axes[1,2].spines['right'].set_visible(False)
    axes[1,2].spines['top'].set_visible(False)
    axes[1,2].grid(True, linestyle=':', alpha=0.4)

    # Common X label
    for ax in axes.flat:
        ax.set_xlabel("Time Periods")

    plt.tight_layout()
    save_path = os.path.join(outdir, "fig_transition_dynamics.pdf")
    plt.savefig(save_path)
    plt.savefig(save_path.replace('.pdf', '.png'))
    plt.close()
    print(f"Generated: {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="outputs_finall", help="Directory containing .npz files")
    args = parser.parse_args()

    print(f"Generating fancy plots from: {args.out}")
    
    steady_path = os.path.join(args.out, "steady_states.npz")
    trans_path = os.path.join(args.out, "transition_path.npz")
    
    plot_policy_functions(steady_path, args.out)
    plot_transition_dynamics(trans_path, args.out)

if __name__ == "__main__":
    main()
