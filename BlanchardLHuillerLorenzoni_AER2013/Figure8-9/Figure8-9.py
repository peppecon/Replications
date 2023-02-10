# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 12:09:24 2023

@author: Piero De Dominicis
"""

'''
Parameters
'''

import numpy as np
import matplotlib.pyplot as plt


rho = 0.891
sig2_u = .0067**2
sig2_nu = .0089**2
sig2_ep = (1-rho)**2*sig2_u
sig2_et = rho*sig2_u

max_iters = 1000
eps = 1e-20
N=20

# %% 
''' Consumer's Expectations '''

A = np.array([[1+rho,-rho,0],[1,0,0],[0,0,rho]])
B = np.array([[1,0,0],[0,0,0],[0,1,0]])
C = np.array([[1,0,1],[1,0,0]])
D = np.array([[0,0,0],[0,0,1]])
V = np.array([sig2_ep,sig2_et,sig2_nu])
sigma_V = np.diag(V)
P_tm1 = np.eye(3)


for i in range(0,max_iters):
    Q = C@P_tm1@C.T + D@sigma_V@D.T
    K = P_tm1@C.T@np.linalg.inv(Q)
    P_hat = P_tm1 - K@C@P_tm1
    P_tm1_new = A@P_hat@A.T + B@sigma_V@B.T
    sq_dev = np.sum(P_tm1_new-P_tm1,1)**2
    dev = np.sum(sq_dev)/9
    P_tm1 = P_tm1_new
    if dev < eps:
        break


IKCA  = (np.eye(3)-K@C)@A


# %% 
''' Econometrician's Inference '''


zero3 = np.zeros([3,3])
H1 = np.hstack([A,zero3,zero3])
H2 = np.hstack([K@C@A,IKCA,zero3])
H3 = np.hstack([zero3,zero3,zero3])  
H = np.vstack([H1,H2,H3])
W = np.vstack([B,K@C@B+K@D,np.eye(3)])


T = np.array([[1,0,1,0,0,0,0,0,0],
                  [0,0,0,1/(1-rho),-rho/(1-rho),0,0,0,0]])


P_tm1 = np.eye(9)


for i in range(0,max_iters):
    Q = T@P_tm1@T.T 
    K = P_tm1@T.T@np.linalg.inv(Q)
    P_hat = P_tm1 - K@T@P_tm1
    P_tm1_new = H@P_hat@H.T + W@sigma_V@W.T
    sq_dev = np.sum(P_tm1_new-P_tm1,1)**2
    dev = np.sum(sq_dev)/81
    P_tm1 = P_tm1_new
    if dev < eps:
        break
    
# %%
''' Kalman smoother, econometrician '''
P12 = P_tm1
P22 = P_tm1
G = T.T@np.linalg.inv(T@P_tm1@T.T)@T


Psmooth = np.zeros([N,P22.shape[0],P22.shape[1]])

for j in range(0,N):
    P22 = P22 - P12@G@P12.T
    P12 = P12@(np.eye(9)-T.T@K.T)@H.T
    Psmooth[j,:,:] = P22

plt.style.use('fivethirtyeight') 

subtitle_font=20
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 8))
axes[0].plot(np.sqrt(Psmooth[:,0,0]), '-', c='blue', label='Consumption')
axes[0].set_title('State X',fontsize=subtitle_font)
axes[0].legend(fontsize=8, frameon=False)
axes[1].plot(np.sqrt(Psmooth[:,2,2]), '-', c='blue', label='Consumption')
axes[1].set_title('State Z',fontsize=subtitle_font)
axes[1].set_xlabel("Number of leads")
fig.tight_layout()
fig.savefig('Figure8.png')


subtitle_font=20
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 6))
axes[0].plot(np.sqrt(Psmooth[:,6,6]/sig2_ep), '-', c='blue')
axes[0].set_title('Permanent Shock',fontsize=subtitle_font)
axes[0].legend(fontsize=8, frameon=False)
axes[0].set_xlabel("Number of leads")
axes[1].plot(np.sqrt(Psmooth[:,7,7]/sig2_et), '-', c='blue')
axes[1].set_title('Transitory Shock',fontsize=subtitle_font)
axes[1].set_xlabel("Number of leads")
axes[2].plot(np.sqrt(Psmooth[:,8,8]/sig2_nu), '-', c='blue')
axes[2].set_title('Noise Shock',fontsize=subtitle_font)
axes[2].set_xlabel("Number of leads")

fig.tight_layout()
fig.savefig('Figure9.png')