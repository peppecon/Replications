#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 11:28:32 2023

@author: peppecon
"""

import numpy as np
import numba as nb

@nb.njit
def vf_simulation_cvf_nb(cvf_diff,mu_nu,sigma_nu,F,beta,S,T):
    nu_0 = np.random.normal(mu_nu, sigma_nu, T)
    nu_1 = np.random.normal(mu_nu, sigma_nu, T)
    
    disc = np.zeros(T)
    for i in range(0,T):
        disc[i] = beta**i
        
    # disc = nb.typed.List([beta**i for i in range(0,T)])
    W1,W2,W3 = [0,0,0]
    s0 = np.random.randint(min(S),max(S))
    # a = np.zeros(T,dtype=int)
    # s = np.zeros(T,dtype=int)
    a = np.zeros(T)
    s = np.zeros(T)
    s[0] = s0  
    for i in range(0,T-1):
        if nu_0[i] - nu_1[i] <= cvf_diff[int(s[i]-1)]:
            a[i] = 1                        
        else:
            a[i] = 0
        #Transition of the state
        S_float = S*1.0
        s[i+1] = S_float @ F[int(a[i]),int(s[i]-1),:]
        s[i+1] = int(s[i+1])
    
    # Getting advantage of parameters' linearity    
    W1 = disc @ a
    W2 = disc*s @ (1-a)
    W3 = disc*(nu_1) @ a + disc*(nu_0) @ (1-a)
        
    return W1,W2,W3,a

@nb.njit
def simulate_vf_nb(cvf_diff,mu_nu,sigma_nu,F,beta,S,T,nS):
    W1_sims = np.zeros(nS)
    W2_sims = np.zeros(nS)
    W3_sims = np.zeros(nS)    
    for s in range(0,nS):
        W1_sims[s],W2_sims[s],W3_sims[s],_ = vf_simulation_cvf_nb(cvf_diff,mu_nu,sigma_nu,F,beta,S,T)
        
    W1_sims_final = np.sum(W1_sims)/nS
    W2_sims_final = np.sum(W2_sims)/nS
    W3_sims_final = np.sum(W3_sims)/nS
    
    return -W1_sims_final,-W2_sims_final,W3_sims_final

def alternative_policies(cvf_diff,mu_nu,sigma_nu,F,beta,S,T,nS,nI,sigma):
    W1_alt = np.zeros(nI)
    W2_alt = np.zeros(nI)
    W3_alt = np.zeros(nI)   
    for policy in range(0,nI):
        # print(f"Computing alternative policy simulations # {policy+1}!")
        cvf_diff_new = np.random.normal(cvf_diff,sigma,len(S))
        W1_alt[policy],W2_alt[policy],W3_alt[policy] = simulate_vf_nb(cvf_diff_new,mu_nu,sigma_nu,F,beta,S,T,nS)
    return np.array([[W1_alt],[W2_alt],[W3_alt]])
