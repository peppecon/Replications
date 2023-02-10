# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 17:09:19 2023

@author: Piero De Dominicis
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# from scipy.optimize import minimize


'''
Parameters
'''

rho = 0.891
sig2_u = .0067**2
sig2_nu = .0089**2
sig2_ep = (1-rho)**2*sig2_u
sig2_et = rho*sig2_u

max_iters = 100000
eps = 1e-12
periods = 20
N_obs = 1000

A = np.array([[1+rho,-rho,0],[1,0,0],[0,0,rho]])
B = np.array([[1,0,0],[0,0,0],[1,1,0]])
C = np.array([[1,0,1],[1,0,0]])
D = np.array([[0,0,0],[0,0,1]])
V = np.array([sig2_ep,sig2_et,sig2_nu])
sigma_V = np.diag(V)
P_tm1 = np.eye(3)


''' 
Parameters for VAR:
    - num_lags: number of lags for the VAR
    - iterations: iterations to find the steady state values 
    - T_graph: number of periods for the IRFs plot
    - IRF_periods: number of periods used to calculate the IRFs
'''
num_lags = 4
iterations = 500
T_graph = 25
IRF_periods = 200



# %%
''' Derive the True IRFs '''


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
    
# %%
''' Simulate series for consumption and productivity using the true model '''
X_sim = np.zeros([9,1])
#np.random.seed(14)
sd = np.array([sig2_ep**0.5,sig2_et**0.5, sig2_nu**0.5])
c_sim  = np.zeros([N_obs+1,1])
a_sim  = np.zeros([N_obs+1,1])
dc_sim = np.zeros([N_obs+1,1])
da_sim = np.zeros([N_obs+1,1])


a_sim[0] = T[0]@X
c_sim[0] = T[1]@X
da_sim[0] = 0
dc_sim[0] = 0

for t in range(1,N_obs+1):
    sim_shocks = sd*np.random.randn(3,1).T
    X_sim = H@X_sim + W@sim_shocks.T
    a_sim[t] = T[0]@X_sim
    c_sim[t] = T[1]@X_sim
    da_sim[t] = a_sim[t] - a_sim[t-1]
    dc_sim[t] = c_sim[t] - c_sim[t-1]

''' Let us plot the artificial data'''
plt.figure(figsize=(12, 6))
plt.style.use('fivethirtyeight') 
plt.title("Simulated Data",fontsize=20)
plt.plot(a_sim, linewidth='2', linestyle='--', c='orange', label="Productivity")
plt.plot(c_sim, linewidth='2', linestyle='-', c='blue',label="Consumption")
plt.legend()
# plt.xlabel('t')
# plt.ylabel('Series')
plt.savefig('simulated_series.png')
plt.show()


''' Johansen Test '''
from statsmodels.tsa.vector_ar.vecm import coint_johansen
df = pd.DataFrame({'a':a_sim[:,0],'c':c_sim[:,0]})
joh_model3 = coint_johansen(df,0,4) # k_ar_diff +1 = K

def joh_output(res):
    output = pd.DataFrame([res.lr2,res.lr1],
                          index=['max_eig_stat',"trace_stat"])
    print(output.T,'\n')
    print("Critical values(90%, 95%, 99%) of max_eig_stat\n",res.cvm,'\n')
    print("Critical values(90%, 95%, 99%) of trace_stat\n",res.cvt,'\n')

joh_output(joh_model3)



y = a_sim;
intercept = np.ones([c_sim.shape[0],c_sim.shape[1]])
X_c = np.hstack([intercept,c_sim])
beta_coint = np.linalg.inv(X_c.T @ X_c) @ (X_c.T @ y)
ECT = y - X_c @ beta_coint;
diff = a_sim - c_sim
plt.plot(diff)
plt.plot(ECT)
plt.show()

y_1 = da_sim[1+num_lags:N_obs+1]
y_2 = dc_sim[1+num_lags:N_obs+1]
da_lag = np.zeros([N_obs - num_lags,num_lags])
dc_lag = np.zeros([N_obs - num_lags,num_lags])
for p in range(0,num_lags):
    lag = p+1 #adjust due to Pythonic indexing
    da_lag[:,p] = np.array(da_sim[1+num_lags-lag:N_obs+1-lag]).T
    dc_lag[:,p] = np.array(dc_sim[1+num_lags-lag:N_obs+1-lag]).T
    
vec_ones = np.ones([N_obs-num_lags,1])

X_coint = np.hstack([vec_ones,
                     da_lag,
                     dc_lag,
                     ECT[num_lags:N_obs]])
    

# ''' For VEC with 4 lags '''   
# X_coint = np.hstack([vec_ones,
#                      da_sim[1+num_lags-1:N_obs+1-1],
#                      da_sim[1+num_lags-2:N_obs+1-2],
#                      da_sim[1+num_lags-3:N_obs+1-3],
#                      da_sim[1+num_lags-4:N_obs+1-4],
#                      dc_sim[1+num_lags-1:N_obs+1-1],
#                      dc_sim[1+num_lags-2:N_obs+1-2],
#                      dc_sim[1+num_lags-3:N_obs+1-3],
#                      dc_sim[1+num_lags-4:N_obs+1-4],
#                      ECT[num_lags:N_obs]])



''' Compute the VAR coefficients through standard OLS '''
# func_a = lambda x: np.sum(((y_1 - X_coint @ x).ravel())**2)
# func_c = lambda x: np.sum(((y_1 - X_coint @ x).ravel())**2)
# betaA = minimize(func_a,np.ones([X_coint.shape[1],1]))
# betaC = minimize(func_c,np.ones([X_coint.shape[1],1]))
# beta_a = np.array([betaA.x]).T
# beta_c = np.array([betaC.x]).T

beta_a = np.linalg.inv(X_coint.T @ X_coint) @ X_coint.T @ y_1
beta_c = np.linalg.inv(X_coint.T @ X_coint) @ X_coint.T @ y_2


# %%

''' Estimate the Variance Covariance matrix '''
eps_a = y_1 - X_coint @ beta_a
eps_c = y_2 - X_coint @ beta_c
sigmat = np.zeros([2,2])
sigmat[0,0] = (eps_a.T @ eps_a)/(N_obs - num_lags)
sigmat[1,0] = (eps_a.T @ eps_c)/(N_obs - num_lags)
sigmat[0,1] = (eps_c.T @ eps_a)/(N_obs - num_lags)
sigmat[1,1] = (eps_c.T @ eps_c)/(N_obs - num_lags)


''' Compute the steady state '''
xit = np.ones([iterations,1])*0.5
cit = np.ones([iterations,1])*0.5
y_1it = np.ones([iterations,1])*0.5
y_2it = np.ones([iterations,1])*0.5

XitC = np.zeros([iterations,num_lags*2 + 2]) # +2 because of int + EC term

for t in range(num_lags,iterations):
    ECTit = xit[t-1] - beta_coint[0] - beta_coint[1]*cit[t-1]
    ECTit = ECTit[:,None]
    da_reg = y_1it[t-num_lags:t][::-1].T
    dc_reg = y_1it[t-num_lags:t][::-1].T
    XitC[t,:] = np.hstack([np.ones([1,1]),da_reg,dc_reg,ECTit])
    y_1it[t] = XitC[t,:]@beta_a
    y_2it[t] = XitC[t,:]@beta_c
    xit[t] = y_1it[t] + xit[t-1]
    cit[t] = y_2it[t] + cit[t-1]
    
y_1ss = y_1it[iterations-1]
y_2ss = y_2it[iterations-1]

# %% 
''' Compute Impulse Response Functions '''

IMPC = np.zeros([iterations,4])


y_1it = np.zeros([iterations,1])
y_1it[:num_lags-1] = y_1ss
y_1it[num_lags-1] = 1+y_1ss
y_2it = np.zeros([iterations,1])
y_2it[:num_lags] = y_2ss
xit[num_lags-1] = xit[iterations-1] + 1
cit[num_lags-1] = cit[iterations-1]

for t in range(num_lags,iterations):
    ECTit = xit[t-1] - beta_coint[0] - beta_coint[1]*cit[t-1]
    ECTit = ECTit[:,None]
    da_reg = y_1it[t-num_lags:t][::-1].T
    dc_reg = y_1it[t-num_lags:t][::-1].T
    XitC[t,:] = np.hstack([np.ones([1,1]),da_reg,dc_reg,ECTit])
    y_1it[t] = XitC[t,:]@beta_a
    y_2it[t] = XitC[t,:]@beta_c
    xit[t] = y_1it[t] + xit[t-1]
    cit[t] = y_2it[t] + cit[t-1]
    
IMPC[:,0] = (y_1it - y_1ss).ravel()
IMPC[:,2] = (y_2it - y_2ss).ravel()

y_1it = np.zeros([iterations,1])
y_1it[:num_lags] = y_1ss
y_2it = np.zeros([iterations,1])
y_2it[num_lags-1] = y_2ss + 1
y_2it[:num_lags-1] = y_2ss
xit[num_lags-1] = xit[iterations-1] 
cit[num_lags-1] = cit[iterations-1] + 1

for t in range(num_lags,iterations):
    ECTit = xit[t-1] - beta_coint[0] - beta_coint[1]*cit[t-1]
    ECTit = ECTit[:,None]
    da_reg = y_1it[t-num_lags:t][::-1].T
    dc_reg = y_1it[t-num_lags:t][::-1].T
    XitC[t,:] = np.hstack([np.ones([1,1]),da_reg,dc_reg,ECTit])
    y_1it[t] = XitC[t,:]@beta_a
    y_2it[t] = XitC[t,:]@beta_c
    xit[t] = y_1it[t] + xit[t-1]
    cit[t] = y_2it[t] + cit[t-1]
    
IMPC[:,1] = (y_1it - y_1ss).ravel()
IMPC[:,3] = (y_2it - y_2ss).ravel()

emats = np.zeros([2,2])
emats[0,0] = np.sum(IMPC[num_lags-1:iterations,0])
emats[0,1] = np.sum(IMPC[num_lags-1:iterations,1])
emats[1,0] = np.sum(IMPC[num_lags-1:iterations,2])
emats[1,1] = np.sum(IMPC[num_lags-1:iterations,3]) - 5

mat = np.linalg.cholesky(emats @ sigmat @ emats.T)
mat = np.linalg.inv(mat)  
BQ = np.linalg.inv(mat @ emats)


BQIMPC = np.zeros([IRF_periods,4])
for t in range(num_lags-1,200):
    BQIMPC[t,0] = IMPC[t,0]*BQ[0,0] + IMPC[t,1]*BQ[1,0]
    BQIMPC[t,1] = IMPC[t,0]*BQ[0,1] + IMPC[t,1]*BQ[1,1]
    BQIMPC[t,2] = IMPC[t,2]*BQ[0,0] + IMPC[t,3]*BQ[1,0]
    BQIMPC[t,3] = IMPC[t,2]*BQ[0,1] + IMPC[t,3]*BQ[1,1]



CUMBQIMPC = np.zeros([IRF_periods,4])

for i in range(0,4):
    for t in range(num_lags-1,IRF_periods):
        CUMBQIMPC[t,i] = BQIMPC[t,i]+CUMBQIMPC[t-1,i]



fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 6),
                         sharey='row', sharex='row')
subtitle_font=16
axes[0, 0].plot(CUMBQIMPC[num_lags:T_graph-1,0], '--', c='black')
axes[0, 0].plot(a_eps, '-', c='red')
axes[0, 0].set_title('Panel A. Productivity: permanent shock',fontsize=subtitle_font)
axes[0, 1].plot(CUMBQIMPC[num_lags:T_graph-1,1], '--', c='black')
axes[0, 1].plot(a_eta, '-', c='green')
axes[0, 1].plot(a_nu, '-', c='blue')
axes[0, 1].set_title('Panel B. Productivity: temporary and noise shock',fontsize=subtitle_font)
axes[1, 0].plot(CUMBQIMPC[num_lags:T_graph-1,2], '--', c='black')
axes[1, 0].plot(c_eps, '-', c='red')
axes[1, 0].set_title('Panel C. Consumption: permanent shock',fontsize=subtitle_font)
axes[1, 1].plot(CUMBQIMPC[num_lags:T_graph-1,3], '--', c='black')
axes[1, 1].plot(c_eta, '-', c='green')
axes[1, 1].plot(c_nu, '-', c='blue')
axes[1, 1].set_title('Panel D. Consumption: temporary and noise shock',fontsize=subtitle_font)
# axes[0, 0].set_ylabel(r'')
# axes[0, 1].set_ylabel(r'')
# axes[1, 0].set_ylabel(r'')
# axes[1, 1].set_ylabel(r'')
fig.tight_layout()
plt.savefig('Figure2.png')
