# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 15:06:01 2023

@author: Piero De Dominicis
"""

'''
Parameters
'''

import numpy as np
import matplotlib.pyplot as plt


rho = 0.891
sig2_u = .67**2
sig2_nu = .89**2
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
    Q = T@P_tm1@T.T + D@sigma_V@D.T
    K = P_tm1@T.T@np.linalg.inv(Q)
    P_hat = P_tm1 - K@T@P_tm1
    P_tm1_new = H@P_hat@H.T + W@sigma_V@W.T
    sq_dev = np.sum(P_tm1_new-P_tm1,1)**2
    dev = np.sum(sq_dev)/81
    P_tm1 = P_tm1_new
    if dev < eps:
        break
    
# %%
''' Variance Decomposition '''

sd = np.array([sig2_ep**0.5,sig2_et**0.5, sig2_nu**0.5])

# figure(1)
# subplot(2,1,1)
# plot(1:N,vd_a(1,:),'ko',1:N,vd_a(2,:),'k.',1:N,vd_a(3,:),'k*','LineWidth',1.5)
# title('productivity')
# subplot(2,1,2)
# plot(1:N,vd_c(1,:),'ko',1:N,vd_c(2,:),'k.',1:N,vd_c(3,:),'k*','LineWidth',1.5)
# title('consumption')
# legend('permanent technology shock','temporary technology shock','noise shock','Location','SouthOutside','Orientation','vertical')


cum_var = (T@W*sd.T)**2
vd_a = np.zeros([3,N])
vd_c = np.zeros([3,N])
vd_a[:,0] = cum_var[0,:]/np.sum(cum_var[0,:])
vd_c[:,0] = cum_var[1,:]/np.sum(cum_var[1,:])

    
for i in range(0,N-1):
    if i == 0:
        Q = H
    else:
        Q = Q@H
    cum_var += (T@Q@W*sd.T)**2
    vd_a[:,i+1] = cum_var[0,:]/np.sum(cum_var[0,:])
    vd_c[:,i+1] = cum_var[1,:]/np.sum(cum_var[1,:])
    
    
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 6),
                         sharey='row', sharex='row')
subtitle_font=16
circle_size = 12
plt.style.use('fivethirtyeight') 
axes[0].plot(vd_a[0,:],linestyle='', c='red', marker='o', markersize = circle_size,
             mfc='none',label="Permanent Technology Shock")
axes[0].plot(vd_a[1,:], linestyle='', c='blue', marker='o', markersize = circle_size,
             label="Temporary Technology Shock")
axes[0].plot(vd_a[2,:], linestyle='', c='green', marker='*', markersize = circle_size,
             label="Noise Shock")
axes[0].set_title('Productivity',fontsize=subtitle_font)
axes[1].plot(vd_c[0,:], linestyle='', c='red', marker='o', markersize = circle_size,
             mfc='none',label="Permanent Technology Shock")
axes[1].plot(vd_c[1,:], linestyle='', c='blue', marker='o', markersize = circle_size,
             label="Temporary Technology Shock")
axes[1].plot(vd_c[2,:], linestyle='', c='green', marker='*', markersize = circle_size,
             label="Noise Shock")
axes[1].set_title('Consumption',fontsize=subtitle_font)
# axes[0, 0].set_ylabel(r'')
# axes[0, 1].set_ylabel(r'')
# axes[1, 0].set_ylabel(r'')
# axes[1, 1].set_ylabel(r'')
for ax in axes:
    ax.set_xticks(range(1,N+1,2)) 
    ax.set_xticklabels(range(1,N+1,2), fontsize=12)
fig.tight_layout()
# handles, labels = [(a + b) for a, b in zip(axes[0].get_legend_handles_labels(), axes[1].get_legend_handles_labels())]
handles, labels = axes[0].get_legend_handles_labels()
lgd = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=1)
fig.savefig('Figure_Table4.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
# fig.savefig('Figure_Table4.png')
np.savetxt("var_dec.txt", vd_c[:,[0,3,7,11]].T, delimiter=' & ', fmt='%2.3f', newline=' \\\\\n')


