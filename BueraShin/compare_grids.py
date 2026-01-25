import numpy as np

def create_ability_grid_paper(eta):
    n_z = 40
    M_values = np.zeros(n_z)
    M_values[:38] = np.linspace(0.633, 0.998, 38)
    M_values[38] = 0.999
    M_values[39] = 0.9995
    z_grid = (1 - M_values) ** (-1/eta)
    
    prob_z = np.zeros(n_z)
    prob_z[0] = M_values[0] / M_values[-1]
    for j in range(1, n_z):
        prob_z[j] = (M_values[j] - M_values[j-1]) / M_values[-1]
    prob_z = prob_z / prob_z.sum()
    return z_grid, prob_z

ETA = 4.15
z_py, p_py = create_ability_grid_paper(ETA)

import os
mat_dir = "/home/nuagsire/Dropbox/PhD Bocconi/Replications/BueraShin_JPE20213/BueraShin2013_v6/inputs"
z_mat = np.loadtxt(os.path.join(mat_dir, "support.dat"))
p_mat = np.loadtxt(os.path.join(mat_dir, "dist.dat"))
p_mat = p_mat / p_mat.sum()

print("Ability Grid Comparison:")
print(f"PY range:  [{z_py.min():.4f}, {z_py.max():.4f}]")
print(f"MAT range: [{z_mat.min():.4f}, {z_mat.max():.4f}]")
print(f"PY mean:   {np.sum(z_py*p_py):.4f}")
print(f"MAT mean:  {np.sum(z_mat*p_mat):.4f}")

print("\nWeights (first 5):")
print(f"PY:  {p_py[:5]}")
print(f"MAT: {p_mat[:5]}")

print("\nLog-scaling MAT support?")
z_mat_exp = np.exp(z_mat)
print(f"MAT exp range: [{z_mat_exp.min():.4f}, {z_mat_exp.max():.4f}]")
print(f"MAT exp mean:  {np.sum(z_mat_exp*p_mat):.4f}")
