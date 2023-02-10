# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 15:21:26 2023

@author: Piero De Dominicis
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io

''' Load Data '''
mat = scipy.io.loadmat('original_data.mat')
dpty = mat['dpty']
dcons = mat['dcons']


dm_dpty = dpty - np.mean(dpty)
dm_dcons = dcons - np.mean(dcons)


dpty_cm = np.cumsum(dm_dpty)
dcons_cm = np.cumsum(dm_dcons)


plt.figure(figsize=(12, 6))
plt.style.use('fivethirtyeight') 
plt.title("Original Data",fontsize=20)
plt.plot(dpty_cm, linewidth='2.5', linestyle='--', c='orange', label="Productivity")
plt.plot(dcons_cm, linewidth='2.5', linestyle='-', c='blue',label="Consumption")
plt.legend()
plt.savefig('Data.png')
plt.show()




# %%
''' Define the function that computes consumer's expectations '''

''' The econometrician impose the structure of the model, so these values are given '''
B = np.array([[1,0,0],[0,0,0],[0,1,0]])
C = np.array([[1,0,1],[1,0,0]])
D = np.array([[0,0,0],[0,0,1]])

def consumers_expectations(rho,sig2_u,sig2_nu,B=B,C=C,D=D):
    
    max_iters = 1000
    eps = 1e-12
    sig2_ep = (1-rho)**2*sig2_u
    sig2_et = rho*sig2_u

    A = np.array([[1+rho,-rho,0],[1,0,0],[0,0,rho]])
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
    zero3 = np.zeros([3,3])
    H1 = np.hstack([A,zero3,zero3])
    H2 = np.hstack([K@C@A,IKCA,zero3])
    H3 = np.hstack([zero3,zero3,zero3])  
    H = np.vstack([H1,H2,H3])
    W = np.vstack([B,K@C@B+K@D,zero3])
    
    T = np.array([[1,0,1,0,0,0,0,0,0],
                      [0,0,0,1/(1-rho),-rho/(1-rho),0,0,0,0]])
    
    return IKCA, H, W, T, sigma_V

# %%
''' Define the Likelihood function '''

def likelihood(theta,S):
       
    rho = 0.891
    sig2_u = theta[0]
    sig2_nu = theta[1]
    
    
    IKCA, H, W, T, sigma_V = consumers_expectations(rho,sig2_u,sig2_nu)
    
    # change this
    var,row = T.shape
    periods = S.shape[1]
    S_tm1 = np.zeros([periods,var,1])
    P_tm1 = np.zeros([periods+1,row,row])
    P_tm1[0,:,:] = np.eye(row)
    P = np.zeros([periods,row,row])
    X_tm1 = np.zeros([periods+1,row,1])
    X = np.zeros([periods,row,1])
    Q = np.zeros([periods,var,var])
    ll = np.zeros([periods,1])
    K = np.zeros([periods,row,var])
        
    for t in range(0,periods):
               
        S_tm1[t,:,:] = T@X_tm1[t,:,:]
        # Q[t,:,:] = T@P_tm1[t,:,:]@T.T + D@sigma_V@D.T
        Q[t,:,:] = T@P_tm1[t,:,:]@T.T
        fc_error = S[:,t,:] - S_tm1[t,:,:]
        last_ll_term = fc_error.T @ np.linalg.inv(Q[t,:,:]) @ fc_error
        
        ll[t] = -np.log(2*np.pi) -0.5*np.log(np.linalg.det(Q[t,:,:])) - 0.5*last_ll_term
        
        K[t,:,:] = P_tm1[t,:,:]@T.T@np.linalg.inv(Q[t,:,:])
        ''' Updating Equations '''
        X[t,:,:] = X_tm1[t,:,:] + K[t,:,:] @ (S[:,t,:] - S_tm1[t,:,:])
        P[t,:,:] = P_tm1[t,:,:] - K[t,:,:] @ T @ P_tm1[t,:,:]
        X_tm1[t+1,:,:] = H @ X[t,:,:] 
        P_tm1[t+1,:,:] = H @ P[t,:,:] @ H.T +  W@sigma_V@W.T
    
    likelihood = np.sum(ll)
    print(-likelihood)
    return -likelihood


# %%

from scipy.optimize import minimize
from datetime import datetime
start_time = datetime.now() 

# S = np.expand_dims(np.array([dcons_cm[1:],dpty_cm[1:]]),axis=2)
S = np.expand_dims(np.array([dpty_cm[1:],dcons_cm[1:]]),axis=2)


# Minimization of -LogLikelihood, constrained
# initial guesses (we estimate the variances)
# theta_0 = np.array([0.5,0.58**2,0.81**2])
theta_0 = np.array([0.005,0.005])

# Nelder-Mead, BFGS, 
res = minimize(likelihood, theta_0,args=(S),method="Nelder-Mead",options={'disp': True,'maxiter':10000})
end_time = datetime.now()
print('Total Time: {}'.format(end_time - start_time))


# %%

''' Compute filtered expectations from the estimated paramters '''

rho = 0.891
sig2_u = res.x[0]
sig2_nu = res.x[1]
IKCA, H, W, T, sigma_V = consumers_expectations(rho,sig2_u,sig2_nu) 

var,row = T.shape
periods = S.shape[1]
S_tm1 = np.zeros([periods,var,1])
P_tm1 = np.zeros([periods+1,row,row])
P = np.zeros([periods,row,row])
X_tm1 = np.zeros([periods+1,row,1])
X = np.zeros([periods,row,1])
Q = np.zeros([periods,var,var])
ll = np.zeros([periods,1])
K = np.zeros([periods,row,var])
L = np.zeros([periods,row,row])
fc_error = np.zeros([periods,var,1])



for t in range(0,periods):
    
    S_tm1[t,:,:] = T@X_tm1[t,:,:]
    # Q[t,:,:] = T@P_tm1[t,:,:]@T.T + D@sigma_V@D.T
    Q[t,:,:] = T@P_tm1[t,:,:]@T.T 
    fc_error[t,:,:] = S[:,t,:] - S_tm1[t,:,:]
    K[t,:,:] = P_tm1[t,:,:]@T.T@np.linalg.pinv(Q[t,:,:])
    L[t,:,:] = H - H @ K[t,:,:] @ T
    X[t,:,:] = X_tm1[t,:,:] + K[t,:,:] @ (S[:,t,:] - S_tm1[t,:,:])
    P[t,:,:] = P_tm1[t,:,:] - K[t,:,:] @ T @ P_tm1[t,:,:]
    X_tm1[t+1,:,:] = H @ X[t,:,:]
    P_tm1[t+1,:,:] = H @ P[t,:,:] @ H.T +  W@sigma_V@W.T


# %% 
''' Compute the Kalman Smoothing '''

r = np.zeros([periods,row,1])
a_sm = np.zeros([periods,row,1])
eta_sm = np.zeros([periods,3,1])

L[periods-1] = np.eye(row)

for t in range(1,periods):
   
    r[periods-t-1,:,:] = T.T@np.linalg.inv(Q[periods-t,:,:])@fc_error[periods-t,:,:] + L[periods-t,:,:].T@r[periods-t,:,:]
    a_sm[periods-t-1,:,:] = X_tm1[periods-t,:,:] + P_tm1[periods-t,:,:]@r[periods-t-1,:,:]
    eta_sm[periods-t,:,:] = sigma_V@W.T@r[periods-t,:,:]
    
a_t_t_inf = (a_sm[:,3,:] - rho*a_sm[:,4,:])/(1-rho) 
a_t_inf = (a_sm[:,0,:] - rho*a_sm[:,1,:])/(1-rho) 

# %%
''' Figure 3 '''

date = np.arange(1970.25,2008,0.25)

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 6),
                         sharey='row', sharex='row')
subtitle_font=16
circle_size = 12
plt.style.use('fivethirtyeight') 
axes[0].plot(date[0:-1],a_sm[0:-1,0,:],linestyle='-', c='red', label=r"Smoothed $x_t$")
axes[0].plot(date[0:-1],a_sm[0:-1,3,:], linestyle='--', c='blue', label=r"Smoothed $x_{t|t}$")
# axes[0].set_title('Smoothed Estimates of the Permanent Component of Productivity',fontsize=subtitle_font)
axes[0].legend(loc="lower right")
axes[1].plot(date[0:-1],a_t_inf[0:-1], linestyle='-', c='red', label=r"Smoothed $x_{t+\infty}$")
axes[1].plot(date[0:-1],a_t_t_inf[0:-1], linestyle='--', c='blue', label=r"Smoothed $x_{t+\infty|t}$")
axes[1].legend(loc="lower right")

# axes[1].set_title('Title2',fontsize=subtitle_font)
fig.tight_layout()
fig.savefig('Figure3.png')


# %%
''' Figure 4 '''

fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 6),
                         sharey='row', sharex='row')
subtitle_font=16
circle_size = 12
plt.style.use('fivethirtyeight') 
axes[0].plot(date,eta_sm[:,0,:],linestyle='-', c='black',linewidth=2)
axes[0].set_title(r'Permanent shock $\epsilon$',fontsize=subtitle_font)
axes[1].plot(date,eta_sm[:,1,:],linestyle='-', c='black',linewidth=2)
axes[1].set_title(r'Transitory shock $\eta$',fontsize=subtitle_font)
axes[2].plot(date,eta_sm[:,2,:],linestyle='-', c='black',linewidth=2)
axes[2].set_title(r'Noise shock $\nu$',fontsize=subtitle_font)

# axes[1].set_title('Title2',fontsize=subtitle_font)
fig.tight_layout()
fig.savefig('Figure4.png')
