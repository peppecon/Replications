import numpy as np

# Paper parameters
ALPHA = 0.33
NU = 1 - 0.79 # 0.21
DELTA = 0.06
LAMBDA = np.inf

# MATLAB reported values for Lambda = inf
w_mat = 0.172
r_mat = 0.0476
# K_mat = 0.745, L_mat = 0.946, Y_mat = 0.307

def solve_py(a, z, w, r, lam):
    rental = r + 0.06
    wage = w
    span = 0.79 # 1 - NU
    
    # aux1 = (1/rental) * alpha * span * z
    aux1 = (1/rental) * 0.33 * 0.79 * z
    aux2 = (1/wage) * (1-0.33) * 0.79 * z
    
    exp1 = 1 - (1-0.33)*0.79
    exp2 = (1-0.33)*0.79
    k1 = (aux1**exp1 * aux2**exp2)**(1/0.21)
    
    kstar = min(k1, lam*a)
    lstar = ((1/wage)*(1-0.33)*0.79*z*kstar**(0.33*0.79))**(1/exp1)
    output = z * (kstar**0.33 * lstar**(1-0.33))**0.79
    profit = output - wage*lstar - rental*kstar
    return profit, kstar, lstar, output

# Test with mean z from support.dat (interpreted as level)
z_mean = 0.2737
p, k, l, y = solve_py(1.0, z_mean, w_mat, r_mat, LAMBDA)
print(f"Mean Entrepreneur (z={z_mean:.4f}, a=1.0):")
print(f"PY: Y={y:.4f}, K={k:.4f}, L={l:.4f}, Profit={p:.4f}")

# Compare to worker
print(f"Worker wage: {w_mat:.4f}")
if p > w_mat:
    print("Agent is Entrepreneur")
else:
    print("Agent is Worker")
