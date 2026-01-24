import time
import numpy as np
import buera_shin_v2_vectorized as v2
import buera_shin_v3_howard_pi as v3

# Parameters
params_osc = (v2.DELTA, v2.ALPHA, v2.NU, 1.25, v2.BETA, v2.SIGMA, v2.PSI)

# Grids
z_grid, prob_z = v2.create_ability_grid_paper(v2.ETA)
a_grid = v2.create_asset_grid(501, 1e-6, 4000)

print("Starting v2 benchmark...")
start = time.time()
res_v2 = v2.find_equilibrium(a_grid, z_grid, prob_z, params_osc, verbose=True, max_iter=40)
end = time.time()
print(f"v2 completed in {end-start:.2f}s")

print("\nStarting v3 benchmark...")
start = time.time()
res_v3 = v3.find_equilibrium_fast(a_grid, z_grid, prob_z, params_osc, verbose=True, max_iter=40, w_init=0.12, r_init=0.00)
end = time.time()
print(f"v3 completed in {end-start:.2f}s")
