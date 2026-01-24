#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 10:36:24 2023

@author: peppecon
"""


import pandas as pd
import numpy as np
from scipy.stats import norm



''' 
First of all we are going to generate the data by backward solving the problem.
For illustration purposes, I do the first two backward iterations manually, and
then I create a function to recursively calculate the optimal policies, starting
from period T-2. 
'''


def s_tilde(s,state_space):
    ''' 
    Function for the transition of the states when no replacement occurs.
        
    Arguments:
    ----------
    s = current state --> int
    state_space = state space --> np.array, size Sx1
    
    Output:
    ----------
    s_tilde = next period state --> int
    
    '''
    s_tilde = min(s+1,max(state_space))
    return s_tilde



def optimal_policy_T(R,mu,S):
    ''' 
    Get the optimal policy for period T.
    
    Arguments:
    ----------
    R = replacement cost --> int
    mu = maintenance cost --> int
    S = state space --> np.array, size Sx1
    
    Output:
    ----------
    sigma_T = policy functions at period T --> np.array, size Sx1
    
    '''
    sigma_T = norm.cdf((mu*S - R)/np.sqrt(2))
    return sigma_T




def optimal_policy_Tm1(R,mu,S,beta):
    ''' 
    Optimal policy for period T-1, returns optimal policy functions and
    conditional value functions
    
    Arguments:
    ----------
    R = replacement cost --> int
    mu = maintenance cost --> int
    S = state space --> np.array, size Sx1
    beta = discount factor --> int
    
    Output:
    ----------
    sigma_T-1 = policy functions at period T-1 --> np.array, size Sx1
    v_1 = conditional value function if replace --> np.array, size Sx1
    v_0 = conditional value function if do not replace --> np.array, size Sx1
    
    '''
    sigma_T = optimal_policy_T(R,mu,S)
    v_0 = np.zeros(len(S))
    v_1 = np.zeros(len(S))
    
    for s in S:
        s_prime = s_tilde(s,S)
        ''' When a = 1, next period state will be s=1. Hence next period policy
            is automatically sigma(1) '''
        pr_a1 = sigma_T[min(S)]
        exp_vf1 = pr_a1*(-R) + (1 - pr_a1)*(-mu)
        v_1[s-1] = -R + beta*exp_vf1
        ''' When a = 0, instead, next period state evolves according to the function
            stilde, therefore next period policy depends on today s as well '''
        pr_a0 = sigma_T[s_prime-1]
        exp_vf0 = pr_a0*(-R) + (1 - pr_a0)*(-mu*s_prime)
        v_0[s-1] = -mu*s + beta*exp_vf0
    
    sigma_Tm1 = norm.cdf((v_1 - v_0)/np.sqrt(2))
    return sigma_Tm1,v_1,v_0




def optimal_policy_Tmx(sigma_Tp1,v_0Tp1,v_1Tp1,R,mu,S,beta):
    ''' 
    Optimal policy for period T-X, returns optimal policy functions and
    conditional value functions
    
    Arguments:
    ----------
    sigma_Tp1 = policy functions at period T+1 --> np.array, size Sx1
    v_0Tp1 = cond vf at t+1 if replace --> np.array, size Sx1
    v_1Tp1 = cond vf at t+1 if do not replace --> np.array, size Sx1
    R = replacement cost --> int
    mu = maintenance cost --> int
    S = state space --> np.array, size Sx1
    beta = discount factor --> int
    
    Output:
    ----------
    sigma_T-X = policy functions at period T-X --> np.array, size Sx1
    v_1 = conditional value function if replace --> np.array, size Sx1
    v_0 = conditional value function if do not replace --> np.array, size Sx1
    
    '''    
    v_0 = np.zeros(len(S))
    v_1 = np.zeros(len(S))
    v_0Tp1 = np.array(v_0Tp1)
    v_1Tp1 = np.array(v_1Tp1)
    
    #This part could be vectorized    
    for s in S:
        s_prime = s_tilde(s,S)
        pr_a1 = sigma_Tp1[min(S)]
        exp_vf1 = pr_a1*(-R + beta*v_1Tp1[s-1]) + (1 - pr_a1)*(-mu + beta*v_0Tp1[s-1])
        v_1[s-1] = -R + beta*exp_vf1
        pr_a0 = sigma_Tp1[s_prime-1]
        exp_vf0 = pr_a0*(-R + beta*v_1Tp1[s-1]) + (1 - pr_a0)*(-mu*s_prime + beta*v_0Tp1[s-1])
        v_0[s-1] = -mu*s + beta*exp_vf0
    
    sigma_Tmx = norm.cdf((v_1 - v_0)/np.sqrt(2))
    return sigma_Tmx,v_1,v_0



def opt_policy_path(T,R,mu,S,beta):
    ''' 
    Obtain policy functions and conditional value functions by backward induction
    
    Arguments:
    ----------
    T = number of time periods --> int
    R = replacement cost --> int
    mu = maintenance cost --> int
    S = state space --> np.array, size Sx1
    beta = discount factor --> int
    
    Output:
    ----------
    sigma = full matrix of policy functions --> np.array, size TxS
    vc_1Tp1 = full matrix of conditional vf when a == replace --> np.array, size TxS
    vc_0Tp1 = full matrix of conditional vf when a == do not replace --> np.array, size TxS
    '''
    
    sigma = np.zeros((T,len(S)))
    vc_0Tp1 = np.zeros((T,len(S)))
    vc_1Tp1 = np.zeros((T,len(S)))
    sigma[T-1,:] = optimal_policy_T(R,mu,S)
    sigma[T-2,:] = optimal_policy_Tm1(R,mu,S,beta)[0]
    vc_1Tp1[T-2,:] = optimal_policy_Tm1(R,mu,S,beta)[1]
    vc_0Tp1[T-2,:] = optimal_policy_Tm1(R,mu,S,beta)[2]
    
    for t in range(T-3,-1,-1):
        # print(t)
        # print(f"{t+1} = {sigma[t+1,:]}")
        # print(sigma[t,:])
        sigma[t,:] = optimal_policy_Tmx(sigma[t+1,:],vc_0Tp1[t+1,:],vc_1Tp1[t+1,:],R,mu,S,beta)[0]
        vc_1Tp1[t,:] = optimal_policy_Tmx(sigma[t+1,:],vc_0Tp1[t+1,:],vc_1Tp1[t+1,:],R,mu,S,beta)[1]
        vc_0Tp1[t,:] = optimal_policy_Tmx(sigma[t+1,:],vc_0Tp1[t+1,:],vc_1Tp1[t+1,:],R,mu,S,beta)[2]
        
    return sigma,vc_1Tp1,vc_0Tp1
        

    
def simulate_data(sigma,N,S):
    ''' 
    Generate a dataset by forward simulating the data NxT drawing 
    random number for the initial mileage s0.
    
    Arguments:
    ----------
    sigma = full matrix of policy functions --> np.array, size TxS
    N = # of observations --> int
    S = state space --> np.array, size Sx1
    
    Output:
    ----------
    actions = matrix containing simulated optimal actions --> np.array, size NxT
    states = matrix containing simulated state space --> np.array, size NxT
    '''
    T = int(np.size(sigma)/len(S))
    actions = np.zeros([N,T])
    states = np.zeros([N,T])
    for i in range(0,N):
        # print(f"Simulation n-{i}")
        s0 = np.random.randint(min(S),max(S))
        states[i,0] = s0
        p0 = np.random.random()
        if p0 < sigma[0,s0-1]:
            actions[i,0] = 1
            states[i,1] = min(S) 
        else:
            actions[i,0] = 0
            states[i,1] = int(s_tilde(s0,S))
        for t in range(1,T):
            p = np.random.random()
            if p < sigma[t,int(states[i,t])-1]:
                actions[i,t] = 1
                if t < T-1:
                    states[i,t+1] = min(S) 
            else:
                actions[i,t] = 0
                if t < T-1:
                    states[i,t+1] = int(s_tilde(states[i,t],S))
    
    return actions,states



def get_df(df_init,T,N):
    ''' 
    Return the dataset in Pandas Dataframe format, this dataset format is 
    similar to the original one in Rust 1987.
    
    Arguments:
    ----------
    df_init = outcome of the function simulate_data --> tuple size Ax1: (np.array(NxT),...,np.array(NxT))
    N = # of observations --> int
    S = state space --> np.array, size Sx1
    
    
    Output:
    ----------
    df = dataframe containing buses information per multiple-periods --> pd.Dataframe 
    
    '''
    actions = df_init[0]
    states = df_init[1]
    index_T = [[i for i in range(0,T)]*N]
    index_N0 = [[i]*T for i in range(1,N+1)]
    new_indexN = []
    [new_indexN.extend(index_N0[i]) for i in range(0,len(index_N0))]
    arrays = [new_indexN,index_T[0]]
    multi_period_buses = pd.MultiIndex.from_arrays(arrays, names=('N', 'T'))
    states_df = states.flatten()
    actions_df = actions.flatten()
    df = pd.DataFrame(data={'states': states_df, 'actions': actions_df},
                      index=multi_period_buses)        
    return df
        
