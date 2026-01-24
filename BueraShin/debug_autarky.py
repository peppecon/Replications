import numpy as np
import buera_shin_v3_howard_pi as v3
import buera_shin_v4_simulation as v4
import time

# Params for autarky
lam = 1.0
params = (v3.DELTA, v3.ALPHA, v3.NU, lam, v3.BETA, v3.SIGMA, v3.PSI)
z_grid, prob_z = v3.create_ability_grid_paper(v3.ETA)
a_grid = v3.create_asset_grid(501, 1e-6, 4000, 2)

# Fixed shocks for v4
fixed_shocks = v4.generate_fixed_shocks(v4.N_AGENTS, 500, v3.PSI, len(z_grid), prob_z)

print("Testing v2 (Vectorized) at lambda=1...")
res_v2 = v2.find_equilibrium(a_grid, z_grid, prob_z, params, 
                             w_init=1.1, r_init=-0.05, max_iter=200)

print("\nTesting v3 (Analytical) at lambda=1...")
# Use the stable GE solver from v3 (which I fixed earlier)
res_v3 = v3.find_equilibrium_fast(a_grid, z_grid, prob_z, params, 
                                  w_init=1.1, r_init=-0.05, max_iter=50)

print("\nTesting v4 (Simulation) at lambda=1 with T=500...")
# Override T_SIM globally for this test
v4.T_SIM = 500
res_v4 = v4.find_equilibrium_sim(a_grid, z_grid, prob_z, params, fixed_shocks,
                                 w_init=1.1, r_init=-0.05, max_iter=50)

print("\nComparison:")
print(f"v3: w={res_v3['w']:.4f}, r={res_v3['r']:.4f}, Y={res_v3['Y']:.4f}")
print(f"v4: w={res_v4['w']:.4f}, r={res_v4['r']:.4f}, Y={res_v4['Y']:.4f}")
