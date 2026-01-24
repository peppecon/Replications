#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 10:37:26 2023

@author: peppecon
"""


import numpy as np
import multiprocessing as mp
import time


from simulate_Rust_dataset import opt_policy_path,simulate_data,get_df
from first_stage_estimators import first_stage,optimal_cutoffs
from simulate_alternative_policies import simulate_vf_nb,alternative_policies
from scipy.optimize import minimize




''' 
Define the parameters for the dynamic optimization problem:
    - mu_nu : mean of the private shocks;
    - sigma_nu: variance of the private shocks;
    - R: replacement cost;
    - mu: maintenance cost;
    - state_space : state space for mileage evolution;
    - beta: discount factor;
    - T: time periods;
    - N: number of observations;
    - nS: number of (forward) simulation, needed to compute the value function;
    - nI: number of alternative policies simulated;
'''
    
mu_nu = 0
sigma_nu = 1
R = 4
mu = 1
S = np.array([1,2,3,4,5])
beta = 0.9
T = 170
N = 50
nS = 250
nI = 200
theta_true = [R,mu]

''' Simulate the data '''
sigma,_,_ = opt_policy_path(T,R,mu,S,beta)
T_prime = T-50
sigma_prime = sigma[:T_prime,:]
sim_series = simulate_data(sigma_prime,N,S)
df = get_df(sim_series,T_prime,N)


''' First Stage: recover optimal policy and transition probabilities '''
p,F = first_stage(df,S,T_prime)
cvf_diff = optimal_cutoffs(p)


''' Second Stage: get true value functions and simulate alternative policies '''
W_true = simulate_vf_nb(cvf_diff,mu_nu,sigma_nu,F,beta,S,T_prime,nS)
W_prime = alternative_policies(cvf_diff,mu_nu,sigma_nu,F,beta,S,T_prime,nS,nI,0.5)


''' Minimum Distance Estimator '''

def Q_func(theta,W_prime,W_true):
    # print(f"R = {theta[0]}, mu = {theta[1]}")
    R=theta[0]
    mu=theta[1]
    vf_true = W_true @ np.array([R,mu,1])
    vf_alt = W_prime.T @ np.array([R,mu,1])
    g = vf_true - vf_alt
    ''' Below we use the minimum because the error is given by the alternatives 
        which uses a shocked policy and grant a value higher than what estimated
        true the data. If what is estimated is optimal, then NO VALUES SHOULD BE
        HIGHER!!! (Here there is the MPE assumption) '''
    dev = np.sum(np.minimum(g,0)**2)
    Q = dev/len(vf_alt)
    return Q
    
''' Pick initial values for the optimization '''
R0 = 3
mu0 = 1.5


''' Minimize the deviation from Markov policy '''
q = lambda x: Q_func([x[0],x[1]],W_prime,W_true)  
res = minimize(q, (R0, mu0), options={'disp': True})
print(res.x)


# %% Monte Carlo Experiment
''' Run the Monte Carlo experiment using parallelization '''

N_pools= mp.cpu_count()
npool=8
pool = mp.Pool(npool)

print(f"Using {npool} CPUs out of {N_pools} total CPUs") 

init_values = np.array([R0,mu0])


def func_par(runs):
    
    cols = len(init_values)
    params = np.zeros((runs,cols))
    
    for i in range(runs):
        print(f"Monte Carlo run {i}")
        sigma,_,_ = opt_policy_path(T,R,mu,S,beta)
        sim_series = simulate_data(sigma,N,S)
        df = get_df(sim_series,T,N)
        p,F = first_stage(df,S,T)
        cvf_diff = optimal_cutoffs(p)
        W_true = simulate_vf_nb(cvf_diff,mu_nu,sigma_nu,F,beta,S,T,nS)
        W_prime = alternative_policies(cvf_diff,mu_nu,sigma_nu,F,beta,S,T,nS,nI,0.5)
        q = lambda x: Q_func([x[0],x[1]],W_prime,W_true)  
        params[i,:] = minimize(q, (init_values[0], init_values[1]), method='SLSQP', options={'disp': False}).x
    
    R0 = np.sum(params[:,0])/runs
    mu0 = np.sum(params[:,1])/runs
    
    return R0,mu0,params


t = time.time()
runs = 50
params = pool.map(func_par,[runs]*npool)
elapsed = time.time() - t
print(f"It take {elapsed} to run {runs*npool} Monte Carlo simulations splitted on {npool} cores")
