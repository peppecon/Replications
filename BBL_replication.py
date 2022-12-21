# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 18:39:26 2022

@author: Piero De Dominicis
"""



import pandas as pd
import numpy as np
import random 
from scipy.stats import norm
from scipy.optimize import minimize



''' 
Define the parameters for the problem:
    - mu_nu : mean of the private shocks;
    - sigma_nu: variance of the private shocks;
    - R: replacement cost;
    - mu: maintenance cost;
    - state_space : state space for mileage evolution;
    - beta: discount factor;
    - T: time periods;
    - N: number of observations;
    - nS: number of (forward) simulation, needed to compute the value func;
    - nI: number of alternative policies simulated;
'''
    
mu_nu = 0
sigma_nu = 1
R = 4
mu = 1
state_space = np.array([1,2,3,4,5])
beta = 0.9
T = 1000
N = 400
nS = 1000
nI = 200
theta_true = [R,mu]



''' 
First of all we are going to generate the data by backward solving the problem.
For didactical purposes, I do the first two backward iterations manually, and
then I create a function to recursively calculate the optimal policies, starting
from period T-2. 
'''


def s_tilde(s,state_space):
    ''' 
    Function for the transition of the states when no replacement occurs.
        
    Arguments:
    ----------
    s = int
    state_space = np.array, size Sx1
    
    Output:
    ----------
    s_tilde = int
    
    '''
    s_tilde = min(s+1,max(state_space))
    return s_tilde


def optimal_policy_T(R=R,mu=mu,S=state_space):
    ''' 
    Get the optimal policy for period T.
    
    Arguments:
    ----------
    R = int
    mu = int
    state_space = np.array, size Sx1
    
    Output:
    ----------
    sigma_T = np.array, size Sx1
    
    '''
    sigma_T = norm.cdf((mu*S - R)/np.sqrt(2))
    return sigma_T

def optimal_policy_Tm1(R=R,mu=mu,S=state_space,beta=beta):
    ''' Optimal policy for period T-1'''
    sigma_T = optimal_policy_T()
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

def optimal_policy_Tmx(sigma_Tp1,v_0Tp1,v_1Tp1,R=R,mu=mu,S=state_space,beta=beta):
    ''' Optimal policy for period T - x '''
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

def opt_policy_path(T=T,R=R,mu=mu,S=state_space,beta=beta):
    sigma = np.zeros((T,len(S)))
    vc_0Tp1 = np.zeros((T,len(S)))
    vc_1Tp1 = np.zeros((T,len(S)))
    sigma[T-1,:] = optimal_policy_T()
    sigma[T-2,:] = optimal_policy_Tm1()[0]
    vc_1Tp1[T-2,:] = optimal_policy_Tm1()[1]
    vc_0Tp1[T-2,:] = optimal_policy_Tm1()[2]
    
    for t in range(T-3,-1,-1):
        # print(t)
        # print(f"{t+1} = {sigma[t+1,:]}")
        # print(sigma[t,:])
        sigma[t,:] = optimal_policy_Tmx(sigma_Tp1=sigma[t+1,:],v_0Tp1=vc_0Tp1[t+1,:],
                                          v_1Tp1=vc_1Tp1[t+1,:])[0]
        vc_1Tp1[t,:] = optimal_policy_Tmx(sigma_Tp1=sigma[t+1,:],v_0Tp1=vc_0Tp1[t+1,:],
                                          v_1Tp1=vc_1Tp1[t+1,:])[1]
        vc_0Tp1[t,:] = optimal_policy_Tmx(sigma_Tp1=sigma[t+1,:],v_0Tp1=vc_0Tp1[t+1,:],
                                          v_1Tp1=vc_1Tp1[t+1,:])[2]
        
    return sigma,vc_1Tp1,vc_0Tp1
        
        
''' Generate a dataset by forward simulating the data NxT drawing random number
    for s0 '''
    
def simulate_data(sigma,N=N,S=state_space):
    T = int(np.size(sigma)/len(S))
    actions = np.zeros([N,T])
    states = np.zeros([N,T])
    for i in range(0,N):
        print(f"Simulation n-{i}")
        s0 = random.randint(min(S),max(S))
        states[i,0] = s0
        p0 = random.random()
        if p0 < sigma[0,s0-1]:
            actions[i,0] = 1
            states[i,1] = min(S) 
        else:
            actions[i,0] = 0
            states[i,1] = int(s_tilde(s0,S))
        for t in range(1,T):
            p = random.random()
            if p < sigma[t,int(states[i,t])-1]:
                actions[i,t] = 1
                if t < T-1:
                    states[i,t+1] = min(S) 
            else:
                actions[i,t] = 0
                if t < T-1:
                    states[i,t+1] = int(s_tilde(states[i,t],S))
    
    return actions,states

def get_df(df_init,T=T,N=N):
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
        

            

def first_stage(df,S=state_space):
    #Given that actions and states are discrete we can simply count "bins"
    #Non-parametric way to estimate p and F from the data
    num_actions = len(df.actions.unique())
    p = np.zeros([len(S),num_actions])
    F = np.zeros([num_actions,len(S),len(S)])
    df['states_tp1'] = df.states.shift(-1)
    df['states_tp1'].loc[:,T-1] = None
    for j in df.actions.unique():
        total = df.states.value_counts().sort_index()
        p[:,int(j)] = df.states.loc[df.actions == j].value_counts().sort_index()/total    
        for k in [x-1 for x in S]:
            values = df['states_tp1'].loc[(df.actions == j) & (df.states == k+1)].value_counts()
            total = np.sum(values)
            list_keys = [x for x in values.keys()]
            for index in list_keys:
                F[int(j),k,int(index)-1] = values[values.index == index].values[0]/total
    return p,F



# This is for Logit errors
# def optimal_policy(p=p):
#     choice_vf_diff = np.log(p[:,1]/p[:,0])
#     opt_pol = norm.cdf(choice_vf_diff/np.sqrt(2))
#     return opt_pol

def optimal_cutoffs(p):
    choice_vf_diff = norm.ppf(p[:,1])*np.sqrt(2)
    return choice_vf_diff


# Simulations with Logit errors
# def vf_simulation(p_hat=p_hat,F=F,beta=beta,
#                   S=state_space,T=1000):
#     nu_0 = np.random.normal(mu_nu, sigma_nu, T)
#     nu_1 = np.random.normal(mu_nu, sigma_nu, T)
#     disc = [beta**i for i in range(0,T)]
#     W1,W2,W3 = [0,0,0]
#     s0 = random.randint(min(S),max(S))
#     a = np.zeros(T,dtype=int)
#     s = np.zeros(T,dtype=int)
#     s[0] = s0
#     p = np.random.uniform(0,1,T)
#     for i in range(0,T-1):
#         if p[i] <= p_hat[s[i]-1]:
#             a[i] = 1                        
#         else:
#             a[i] = 0
#         #Transition of the state
#         s[i+1] = S @ F[a[i],s[i]-1,:]
#         s[i+1] = int(s[i+1])
        
#     # Getting advantage of parameters' linearity    
#     W1 = disc @ a
#     W2 = disc*s @ (1-a)
#     W3 = disc*(nu_1) @ a + disc*(nu_0) @ (1-a)
        
#     return W1,W2,W3,a

def vf_simulation_cvf(cvf_diff,F,beta=beta,
                  S=state_space,T=T):
    nu_0 = np.random.normal(mu_nu, sigma_nu, T)
    nu_1 = np.random.normal(mu_nu, sigma_nu, T)
    disc = [beta**i for i in range(0,T)]
    W1,W2,W3 = [0,0,0]
    s0 = random.randint(min(S),max(S))
    a = np.zeros(T,dtype=int)
    s = np.zeros(T,dtype=int)
    s[0] = s0
    for i in range(0,T-1):
        if nu_0[i] - nu_1[i] <= cvf_diff[s[i]-1]:
            a[i] = 1                        
        else:
            a[i] = 0
        #Transition of the state
        s[i+1] = S @ F[a[i],s[i]-1,:]
        s[i+1] = int(s[i+1])
        
    # Getting advantage of parameters' linearity    
    W1 = disc @ a
    W2 = disc*s @ (1-a)
    W3 = disc*(nu_1) @ a + disc*(nu_0) @ (1-a)
        
    return W1,W2,W3,a

def simulate_vf(cvf_diff,F,nS=nS):
    W1_sims = np.zeros(nS)
    W2_sims = np.zeros(nS)
    W3_sims = np.zeros(nS)    
    for s in range(0,nS):
        W1_sims[s],W2_sims[s],W3_sims[s],_ = vf_simulation_cvf(cvf_diff=cvf_diff,F=F)
        
    W1_sims_final = np.sum(W1_sims)/nS
    W2_sims_final = np.sum(W2_sims)/nS
    W3_sims_final = np.sum(W3_sims)/nS
    
    return -W1_sims_final,-W2_sims_final,W3_sims_final




def alternative_policies(cvf_diff,F,nI=nI,sigma=0.5,s=state_space):
    W1_alt = np.zeros(nI)
    W2_alt = np.zeros(nI)
    W3_alt = np.zeros(nI)   
    for policy in range(0,nI):
        print(f"Computing alternative policy simulations # {policy+1}!")
        cvf_diff_new = np.random.normal(cvf_diff,sigma,len(s))
        W1_alt[policy],W2_alt[policy],W3_alt[policy] = simulate_vf(cvf_diff=cvf_diff_new,F=F)
    return np.array([[W1_alt],[W2_alt],[W3_alt]])

''' We exploit the linearity of the profit function, therefore we first compute
    W1,W2,W3 and then we minimize on linear objectives. This reduces the computational
    burden by quite a lot. '''
    


def Q_func(theta,W_prime,W_true):
    print(f"R = {theta[0]}, mu = {theta[1]}")
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

''' Run the code '''
sigma,_,_ = opt_policy_path()
df_init = simulate_data(sigma)
p,F = first_stage(get_df(df_init))
cvf_diff = optimal_cutoffs(p)
W_true = simulate_vf(cvf_diff,F)
W_prime = alternative_policies(cvf_diff,F)



''' Pick initial values for the optimization '''
R0 = 1
mu0 = 1


''' Minimize the deviation from Markov policy '''
q = lambda x: Q_func([x[0],x[1]],W_prime,W_true)  
res = minimize(q, (R0, mu0), options={'disp': True})
print(res.x)
