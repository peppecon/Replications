# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 18:37:35 2022

@author: Piero De Dominicis
"""

import numpy as np
import matplotlib.pyplot as plt
# from scipy.optimize import fsolve


'''
Parameters
'''

rho = 0.891
sig2_u = .67**2
sig2_nu = .89**2
sig2_ep = (1-rho)**2*sig2_u
sig2_et = rho*sig2_u

max_iters = 10000
eps = 1e-12
periods = 20

A = np.array([[1+rho,-rho,0],[1,0,0],[0,0,rho]])
B = np.array([[1,0,0],[0,0,0],[0,1,0]])
C = np.array([[1,0,1],[1,0,0]])
D = np.array([[0,0,0],[0,0,1]])
V = np.array([sig2_ep,sig2_et,sig2_nu])
sigma_V = np.diag(V)
P_tm1 = np.eye(3)
#dims = (P_tm1@C.T).shape
#starting_values = np.ones(dims).ravel()


# def get_K(ini_vec):
#     ini_vec_mat = ini_vec.reshape(dims)
#     res = (ini_vec_mat @ Q - P_tm1@C.T).ravel()
#     return res

for i in range(0,max_iters):
    Q = C@P_tm1@C.T + D@sigma_V@D.T
    K = P_tm1@C.T@np.linalg.inv(Q)
    # K = fsolve(get_K,starting_values)
    # K = K.reshape(dims)
    P_hat = P_tm1 - K@C@P_tm1
    P_tm1_new = A@P_hat@A.T + B@sigma_V@B.T
    sq_dev = np.sum(P_tm1_new-P_tm1,1)**2
    dev = np.sum(sq_dev)/9
    P_tm1 = P_tm1_new
    if dev < eps:
        break


IKCA  = (np.eye(3)-K@C)@A
zero3 = np.zeros([3,3])
H1 = np.hstack([A,zero3,zero3])
H2 = np.hstack([K@C@A,IKCA,zero3])
H3 = np.hstack([zero3,zero3,zero3])  
H = np.vstack([H1,H2,H3])
W = np.vstack([B,K@C@B+K@D,zero3])

T = np.array([[1,0,1,0,0,0,0,0,0],
                  [0,0,0,1/(1-rho),-rho/(1-rho),0,0,0,0]])


c_eps = np.zeros([periods])
a_eps = np.zeros([periods])
c_eta = np.zeros([periods])
a_eta = np.zeros([periods])
c_nu = np.zeros([periods])
a_nu = np.zeros([periods])

''' Permanent Shock '''
perm_shock = np.array([sig2_ep**0.5,0,0])
X = W @ perm_shock;

for t in range(0,periods):
    c_eps[t] = T[1].T@X;
    a_eps[t] = T[0].T@X;
    X = H@X;
    
''' Transitory Shock '''
trans_shock = np.array([0,sig2_et**0.5,0])
X = W @ trans_shock;

for t in range(0,periods):
    c_eta[t] = T[1].T@X;
    a_eta[t] = T[0].T@X;
    X = H@X;
    
''' Noise Shock '''
noise_shock = np.array([0,0,sig2_nu**0.5])
X = W @ noise_shock;

for t in range(0,periods):
    c_nu[t] = T[1].T@X;
    a_nu[t] = T[0].T@X;
    X = H@X;    
    


plt.style.use('fivethirtyeight') 

subtitle_font=10
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 6))
axes[0].plot(c_eps, '-', c='blue', label='Consumption')
axes[0].plot(a_eps, '--', c='green', label='Productivity')
axes[0].set_title('Panel A. Permanent shock',fontsize=subtitle_font)
axes[0].legend(fontsize=8, frameon=False)
axes[1].plot(c_eta, '-', c='blue', label='Consumption')
axes[1].plot(a_eta, '--', c='green', label='Productivity')
axes[1].set_title('Panel B. Transitory shock',fontsize=subtitle_font)
axes[2].plot(c_nu, '-', c='blue', label='Consumption')
axes[2].plot(a_nu, '--', c='green', label='Productivity')
axes[2].set_title('Panel C. Noise shock',fontsize=subtitle_font)
fig.suptitle('Figure 1: Impulse Responses')
for i in axes:
    i.set_ylim([-0.2,0.8])
plt.savefig('Figure1.png')


    