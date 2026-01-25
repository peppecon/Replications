#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 18:41:10 2023

@author: peppecon
"""

import sys
import os

# Get the directory of this script
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.append(script_dir)

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from functions_library import *
from scipy.interpolate import Rbf,CubicSpline



# %%
""" Warm up: Part 1, approximate a function of two variables """

value = 20
n_x = value
n_y = value
p_x = value
p_y = value
kron_xy,cheb_nodes_x,cheb_nodes_y = Tenser_Product_bv(n_x,n_y,p_x,p_y)

# Define a function to find the optimal gamma_0
x_min = 0
x_max = 1
y_min = -1
y_max = 0
x_grid = Change_Variable_Fromcheb(x_min,x_max,cheb_nodes_x)
y_grid = Change_Variable_Fromcheb(y_min,y_max,cheb_nodes_y)


xy, yx = np.meshgrid(x_grid,y_grid)
grid_xy = np.array((xy.ravel(), yx.ravel())).T


""" I define the Approximating and the Residual Function in the main script as they are
    application specific """
    
def Approximating_Function(gamma,kron_xy):    
    approx_func = gamma @ kron_xy    
    return approx_func

#function to minimize
def Residual_Function(gamma,kron_xy,func,grid_xy):
    x = grid_xy[:,0]
    y = grid_xy[:,1]
    target_fun = func(x,y)
    residuals = target_fun - Approximating_Function(gamma,kron_xy)
    SSR = np.sum(residuals**2)        
    return SSR

''' Pick initial values for the optimization '''
gamma_0 = np.ones(n_x*n_y)

''' Define the function you want to approximate '''
func = lambda x,y: np.exp(x)*np.exp(y)


''' Find optimal gamma by minimizing the residual function '''
q = lambda x: Residual_Function(x,kron_xy,func,grid_xy) 
res = minimize(q, gamma_0, options={'disp': True})
gamma_star = res.x



""" Plot the Approximation Results """

plt.close('all')
try:
    plt.style.use('seaborn-v0_8')
except OSError:
    try:
        plt.style.use('seaborn')
    except OSError:
        plt.style.use('default') 



''' Simulate data to plot the real function '''
sim_points = 100
x_sim = np.linspace(x_min, x_max,sim_points)
y_sim = np.linspace(y_min, y_max,sim_points)
X, Y = np.meshgrid(x_sim, y_sim)
Z = func(X,Y)

from matplotlib import cm
my_col = cm.jet(Z/np.amax(Z))


''' Prepare the data from the approximated function '''
#adj_nodes_X = Change_Variable_Fromcheb(x_min,x_max,cheb_nodes_x)
#adj_nodes_Y = Change_Variable_Fromcheb(y_min,y_max,cheb_nodes_y)
X_approx_grid, Y_approx_grid = np.meshgrid(x_grid,y_grid)
Z_approx_vec = gamma_star @ kron_xy
Z_approx_grid = Z_approx_vec.reshape((n_x,n_y))

ax = plt.figure().add_subplot(projection='3d')

ax.plot_surface(X_approx_grid, Y_approx_grid, Z_approx_grid, edgecolor='blue', lw=0.8, alpha=0)
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors = my_col,linewidth=0, alpha = 0.2,antialiased=False)
ax.scatter(X_approx_grid, Y_approx_grid, Z_approx_grid, marker='o', c = 'red', s = 25, alpha=1)

ax.set(xlim=(x_min, x_max), ylim=(y_min, y_max), zlim=(np.min(Z), np.max(Z)),
       xlabel='X', ylabel='Y', zlabel='Z')

plt.savefig(f'latex/func11_{value}.png')
plt.show()


# %% Plot approximation results along approximated functions with new points
""" Plot the Approximation Results """

plt.close('all')
try:
    plt.style.use('seaborn-v0_8')
except OSError:
    try:
        plt.style.use('seaborn')
    except OSError:
        plt.style.use('default') 


''' Prepare the data from the approximated function '''
#adj_nodes_X = Change_Variable_Fromcheb(x_min,x_max,cheb_nodes_x)
#adj_nodes_Y = Change_Variable_Fromcheb(y_min,y_max,cheb_nodes_y)
X_approx_new, Y_approx_new = np.meshgrid(x_sim,y_sim)
kron_xy_new = Tenser_Product_new_points(x_sim,y_sim,p_x,p_y)
Z_approx_vec_new = gamma_star @ kron_xy_new
Z_approx_new = Z_approx_vec_new.reshape((sim_points,sim_points))


ax = plt.figure().add_subplot(projection='3d')

ax.plot_surface(X_approx_new, Y_approx_new, Z_approx_new, edgecolor='blue', lw=0.4, rstride=4, cstride=4,
                alpha=0.4)
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors = my_col,linewidth=0, alpha = 0.2,antialiased=False)
ax.scatter(X_approx_grid, Y_approx_grid, Z_approx_grid, marker='o', c = 'red', s = 25, alpha=1)

ax.set(xlim=(x_min, x_max), ylim=(y_min, y_max), zlim=(np.min(Z), np.max(Z)),
       xlabel='X', ylabel='Y', zlabel='Z')
plt.savefig(f'latex/func12_{value}.png')
plt.show()

# %%
""" Warm up: Part 2, approximate a function of one variable, with a max operator """

value = 20
n_x = value
p_x = value

T_x,cb_nodes_x2 = Tx(n_x,p_x)

''' Define the domanin of the function you want to approximate '''
x_min = 0
x_max = 2

''' Convert the Chebyshev grid to the original Domain '''
grid_x = Change_Variable_Fromcheb(x_min,x_max,cb_nodes_x2)


''' I define the Approximating and the Residual Function in the main script as they are
    application specific, but they could easily be generalized with a unique function '''
    
def Approximating_Function(gamma,T_x):    
    approx_func = gamma @ T_x    
    return approx_func

def Residual_Function(gamma,T_x,func,grid_x):
    x = grid_x
    target_fun = func(x)
    residuals = target_fun - Approximating_Function(gamma,T_x)
    SSR = np.sum(residuals**2)        
    return SSR

''' Pick initial values for the optimization '''
gamma_0 = np.ones(n_x)

''' Define the function you want to approximate '''
func = lambda x: np.maximum(0,x-1)


''' Find optimal gamma by minimizing the residual function '''
q = lambda x: Residual_Function(x,T_x,func,grid_x) 
res = minimize(q, gamma_0, options={'disp': True})
gamma_star = res.x

""" Obtain approximated function at collocation points """
y_points = gamma_star @ T_x


""" Plot the Approximation Results """

''' Simulate data to plot the real function '''
x_sim = np.linspace(x_min, x_max,100)
y_sim = func(x_sim)

T_x_new,cb_nodes_x2_new = Tx_new_points(x_sim,value)

''' Prepare the data from the approximated function '''
# y_approx = gamma_star @ T_x
y_approx = gamma_star @ T_x_new


""" Plot the Approximation Results with Interpolation"""

# spl = CubicSpline(grid_x[::-1], y_approx[::-1]) #need strictly increasing seq.
# rbf = Rbf(grid_x[::-1], y_approx[::-1])
# lint = lambda x: np.interp(x,grid_x[::-1],y_approx[::-1])

plt.close('all')
plt.figure(figsize=(18, 9))
#plt.style.use('fivethirtyeight') 
#plt.style.use('seaborn-v0_8-darkgrid') 
try:
    plt.style.use('seaborn-v0_8')
except OSError:
    try:
        plt.style.use('seaborn')
    except OSError:
        plt.style.use('default') 
#plt.title(f"N=P={value}",fontsize=20)
plt.plot(x_sim, y_sim, linewidth='3.5', linestyle='solid', c='blue', alpha=0.8, label="True Function")
plt.plot(x_sim, y_approx, linewidth='3.5', linestyle='solid', c='orange', alpha=0.8, label="Approximated Function")
# plt.plot(x_sim, spl(x_sim), linewidth='2.5', linestyle='dashed', c='green', alpha=0.7, label="Approx. Func - Spline")
# plt.plot(x_sim, rbf(x_sim), linewidth='2.5', linestyle='dashdot', c='magenta', alpha=0.6, label="Approx. Func - RBF")
# plt.plot(x_sim, lint(x_sim), linewidth='2.5', linestyle= (0, (3, 1, 1, 1)), c='cyan', alpha=0.8, label="Approx. Func - Linear")
plt.scatter(grid_x, y_points, c='red', marker='o', s=100, alpha=1, label="Collocation Points")
plt.legend(fontsize=20)
plt.savefig(f'latex/func2_{value}.png')
plt.show()

