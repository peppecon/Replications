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
    
    print(f"DEBUG: Loaded Data from {transition_path}")
    print(f"DEBUG: pre_Y={pre_Y:.4f}, pre_K={pre_K:.4f}")
    if len(data['Y']) > 0:
        Y0 = data['Y'][0]
        K0 = data['K'][0]
        Y_n0 = Y0 / pre_Y if pre_Y != 0 else 0
        print(f"DEBUG: Y[0]={Y0:.4f}, K[0]={K0:.4f}")
        print(f"DEBUG: Y_norm[0]={Y_n0:.4f}")
        
        if Y_n0 < 0.1:
            print("\n" + "!"*60)
            print("WARNING: Normalized GDP is near ZERO.")
            print(f"This implies Y[0] ({Y0:.4f}) is much smaller than pre_Y ({pre_Y:.4f}).")
            print("Check if pre_Y is artificially large (e.g. infinite lambda artifact?)")
            print("!"*60 + "\n")
    
    # Arrays to plot
    Y_plot = np.zeros(n_win)
    K_plot = np.zeros(n_win)
    TFP_plot = np.zeros(n_win)
    w_plot = np.zeros(n_win)
    r_plot = np.zeros(n_win)
    
    # Calculate Investment (Derived)
    # I_t = K_{t+1} - (1-delta)K_t
    # In the data, K[t] is capital stock at time t.
    # So I[t] implies change from K[t] to K[t+1].
    # We need K_{t+1}. In our saved path, K is likely length T.
    # We can compute I up to T-1.
    
    # Construct full time series for K including pre-ss history
    # t_win goes from -4 to 20.
    K_full = np.zeros(n_win + 1) # need one extra for I calculation of last point
    
    # Fill K_full
    for i, tt in enumerate(t_win):
        if tt < 0:
            K_full[i] = pre_K
        else:
            if tt < len(t):
                K_full[i] = data['K'][tt]
            else:
                K_full[i] = data['K'][-1]
    # Extra point
    K_full[-1] = K_full[-2] # assume steady state at end
    
    delta = 0.06 # Fixed parameter from paper/code
    I_full = K_full[1:] - (1 - delta) * K_full[:-1]
    
    # We also need Y for I/Y
    Y_full = np.zeros(n_win)
    for i, tt in enumerate(t_win):
        if tt < 0:
            Y_full[i] = pre_Y
        else:
            if tt < len(t):
                Y_full[i] = data['Y'][tt]
            else:
                Y_full[i] = data['Y'][-1]
                
    IY_ratio = I_full / Y_full
    
    # Pre-reform I/Y
    IY_pre = (pre_K - (1-delta)*pre_K) / pre_Y # = delta * K / Y
    
    # Deviation from pre-reform level
    IY_dev = IY_ratio - IY_pre

    # Normalize others
    Y_n = Y_plot / pre_Y
    K_n = K_plot / pre_K
    TFP_n = TFP_plot / pre_TFP
    w_n = w_plot / pre_w
    # Interest rate: Levels (not ratio)
    r_level = r_plot

    # Layout: 5 panels like paper (Fig 3 top row, Fig 4 bottom row)
    # Row 1: GDP, TFP, Investment Rate
    # Row 2: Capital, Interest Rate, (Blank or Wage)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # Helper to standardize subplots
    def style_subplot(ax, x, y, title, ylabel=None, is_level=False):
        # Split data at t=0
        mask_pre = x < 0
        mask_post = x >= 0
        
        # Style: Black solid line only
        if np.any(mask_pre):
            ax.plot(x[mask_pre], y[mask_pre], color='black', linewidth=3.0)
            
        if np.any(mask_post):
            ax.plot(x[mask_post], y[mask_post], color='black', linewidth=3.0)
             
        ax.axvline(0, color='gray', linestyle='-', linewidth=0.8, alpha=1.0) # Paper has solid vertical line
        
        # Reference line (y=1 for normalized, y=0 for devs, or pre-level)
        if not is_level:
             # Check if deviations (mean near 0) or normalized (mean near 1)
             if np.abs(np.mean(y[:3])) < 0.1: # Likely deviation
                 ax.axhline(0.0, color='gray', linestyle='-', linewidth=0.5)
             else:
                 ax.axhline(1.0, color='gray', linestyle='-', linewidth=0.5)
        
        ax.set_title(title, fontsize=14)
        if ylabel: ax.set_ylabel(ylabel, fontsize=12)
        
        # Ticks inward
        ax.tick_params(direction='in', length=5)
        
        # Box frame like the paper (ticks on all sides? Paper has box)
        # But QJE style usually despine top/right.
        # The user image shows box plots (ticks on top/right but no labels).
        # Let's keep typical clean style but maybe enabling top/right spines if desired.
        # Sticking to current clean looks.
        ax.spines['right'].set_visible(True)
        ax.spines['top'].set_visible(True)
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        
    style_subplot(axes[0,0], t_win, Y_n, "GDP (normalized)")
    style_subplot(axes[0,1], t_win, TFP_n, "TFP Measure (normalized)")
    style_subplot(axes[0,2], t_win, IY_dev, "Investment Rate (deviation from Pre-SS)")
    
    style_subplot(axes[1,0], t_win, K_n, "Capital Stock (normalized)")
    style_subplot(axes[1,1], t_win, r_level, "Interest Rates (Level)", is_level=True)
    
    # Wage or unused
    # style_subplot(axes[1,2], t_win, w_n, "Wage (normalized)")
    # Keep it blank or put Wage
    axes[1,2].axis('off') # Remove 6th plot if not needed, paper implies 5 panels in description

    # Common X label
    for ax in axes.flat:
        if ax.get_visible():
            ax.set_xlabel("Years after reform")
            ax.set_xlim(-4, 20)

    plt.tight_layout()
    save_path = os.path.join(outdir, "fig_paper_replication.pdf")
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
