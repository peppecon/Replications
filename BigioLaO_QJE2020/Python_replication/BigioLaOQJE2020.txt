#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 16:08:14 2023

@author: peppecon
"""

import os
import numpy as np
import matplotlib.pyplot as plt
#import scipy.io

from mat4py import loadmat


''' Change directory '''
pathfile_linux="/home/nuagsire/Dropbox/PhD Bocconi/2nd year courses/Advanced Macroeconomics III/Replication/BigioLaO/Python_replication"
os.chdir(pathfile_linux)
# pathfile_windows="C:\\Users\\Eurhope\\Dropbox\\PhD Bocconi\\2nd year courses\\Advanced Macroeconomics III\\Replication\\BigioLaO\\Python_replication"
# os.chdir(pathfile_windows)

from func_library import network_hm



''' Load Data '''
IOvars = loadmat('IOvars.mat')
IOparams = loadmat('IOparams.mat')
#IOseries_original = scipy.io.loadmat('IOseries.mat')
IOseries = loadmat('IOseries.mat')
IOshock = loadmat('IOshock.mat')


def value(key):
    try:
        value = IOvars[f'{key}']
    except KeyError:
        try:
            value = IOparams[f'{key}']
        except KeyError:
            try:
                value = IOseries[f'{key}']
            except KeyError:
                value = IOshock[f'{key}']
                
    return np.array(value)

            
    


beg_year = 1997         # first year
end_year = 2014         # final year
N_years = 18            # Number of Years
NsecGZ = value('phi_GZ_mat').shape[0]       # Number of Industries
dYearsvec = range(beg_year+1, end_year, 1)
Yearsvec = range(beg_year, end_year, 1)
onevec = np.ones([NsecGZ,1])

''' Set Parameters '''
v_vec = value('v_i_vec')
alpha_vec = value('alpha_i_vec')
eta_vec = value('eta_i_vec')

''' Define Horizontal Economy '''
alphahor = 0.9999*onevec
Whor = (1/NsecGZ)*np.ones([NsecGZ,NsecGZ])

''' Symmetric Shocks Time Series: uses average phi_GZ '''
xsec_mlogphiGZ = np.ones([N_years,1])
for jj in range(0,N_years):
    xsec_mlogphiGZ[jj] = np.mean(value('logphi_GZ_mat')[:,jj])

logphiGZ_sym_mat = np.ones([NsecGZ,1])*xsec_mlogphiGZ.T
phiGZ_sym_mat = np.exp(logphiGZ_sym_mat)
Dlogphi_GZsym = np.diff(xsec_mlogphiGZ)

# %% Run Simulations
""" Run Simulations """
''' Declare Variables '''

''' Baseline with GZ shocks '''
log_A_GZ_DRS                = np.ones([1,N_years])
neglogLambda_GZ_DRS         = np.ones([1,N_years])
etabarGZ_DRS                = np.ones([1,N_years])


''' Baseline with Symmetric GZ Shocks '''
log_A_GZsym_DRS             = np.ones([1,N_years])
neglogLambda_GZsym_DRS      = np.ones([1,N_years])
etabar_GZsym_DRS            = np.ones([1,N_years])

''' Horizontal with symmetric GZ shocks '''
log_A_hor_GZsym_DRS         = np.ones([1,N_years])
neglogLambda_hor_GZsym_DRS  = np.ones([1,N_years])
etabar_hor_GZsym_DRS        = np.ones([1,N_years])

''' Horizontal with asymmetric GZ shocks '''
log_A_hor_GZasym_DRS        = np.ones([1,N_years])
neglogLambda_hor_GZasym_DRS = np.ones([1,N_years])
etabar_hor_GZasym_DRS       = np.ones([1,N_years])

''' CRS Baseline with GZ shocks '''
log_A_GZ_CRS                = np.ones([1,N_years])
neglogLambda_GZ_CRS         = np.ones([1,N_years])
etabarGZ_CRS                = np.ones([1,N_years])

''' CRS Baseline with Symmetric GZ Shocks '''
log_A_GZsym_CRS             = np.ones([1,N_years])
neglogLambda_GZsym_CRS      = np.ones([1,N_years])
etabar_GZsym_CRS            = np.ones([1,N_years])

''' CRS Horizontal with symmetric GZ shocks '''
log_A_hor_GZsym_CRS         = np.ones([1,N_years])
neglogLambda_hor_GZsym_CRS  = np.ones([1,N_years])
etabar_hor_GZsym_CRS        = np.ones([1,N_years])

''' CRS Horizontal with asymmetric GZ shocks '''
log_A_hor_GZasym_CRS        = np.ones([1,N_years])
neglogLambda_hor_GZasym_CRS = np.ones([1,N_years])
etabar_hor_GZasym_CRS       = np.ones([1,N_years])


Wii = value('W_2007');        #Set input-output matrix to 2007 tables

''' Extract Value from Dictionaries to simplify notations in the loop '''
phi_GZ_mat = value('phi_GZ_mat')
logphi_GZ_mat = value('logphi_GZ_mat')
eta_CRS_vec = value('eta_CRS_vec')

for ii in range(0,N_years):
    
    ''' DRS with GZ Shocks '''
    log_A_GZ_DRS[0,ii],neglogLambda_GZ_DRS[0,ii],etabarGZ_DRS[0,ii] = network_hm(phi_GZ_mat[:,ii],logphi_GZ_mat[:,ii],Wii,v_vec,alpha_vec,eta_vec,NsecGZ)
    ''' DRS with Symmetric GZ Shocks '''
    log_A_GZsym_DRS[0,ii],neglogLambda_GZsym_DRS[0,ii],etabar_GZsym_DRS[0,ii] = network_hm(phiGZ_sym_mat[:,ii],logphiGZ_sym_mat[:,ii],Wii,v_vec,alpha_vec,eta_vec,NsecGZ)
    ''' DRS Horizontal with Symmetric GZ Shocks '''
    log_A_hor_GZsym_DRS[0,ii],neglogLambda_hor_GZsym_DRS[0,ii],etabar_hor_GZsym_DRS[0,ii] = network_hm(phiGZ_sym_mat[:,ii],logphiGZ_sym_mat[:,ii],Whor,v_vec,alphahor,eta_vec,NsecGZ)
    ''' DRS Horizontal with GZ shocks '''
    log_A_hor_GZasym_DRS[0,ii],neglogLambda_hor_GZasym_DRS[0,ii],etabar_hor_GZasym_DRS[0,ii] = network_hm(phi_GZ_mat[:,ii],logphi_GZ_mat[:,ii],Whor,v_vec,alphahor,eta_vec,NsecGZ)
    
    
    ''' CRS with GZ Shocks '''
    log_A_GZ_CRS[0,ii],neglogLambda_GZ_CRS[0,ii],etabarGZ_CRS[0,ii] = network_hm(phi_GZ_mat[:,ii],logphi_GZ_mat[:,ii],Wii,v_vec,alpha_vec,eta_CRS_vec,NsecGZ)
    ''' CRS with Symmetric GZ Shocks '''
    log_A_GZsym_CRS[0,ii],neglogLambda_GZsym_CRS[0,ii],etabar_GZsym_CRS[0,ii] = network_hm(phiGZ_sym_mat[:,ii],logphiGZ_sym_mat[:,ii],Wii,v_vec,alpha_vec,eta_CRS_vec,NsecGZ)
    ''' CRS Horizontal with Symmetric GZ Shocks '''
    log_A_hor_GZsym_CRS[0,ii],neglogLambda_hor_GZsym_CRS[0,ii],etabar_hor_GZsym_CRS[0,ii] = network_hm(phiGZ_sym_mat[:,ii],logphiGZ_sym_mat[:,ii],Whor,v_vec,alphahor,eta_CRS_vec,NsecGZ)
    ''' CRS Horizontal with GZ shocks '''
    log_A_hor_GZasym_CRS[0,ii],neglogLambda_hor_GZasym_CRS[0,ii],etabar_hor_GZasym_CRS[0,ii] = network_hm(phi_GZ_mat[:,ii],logphi_GZ_mat[:,ii],Whor,v_vec,alphahor,eta_CRS_vec,NsecGZ)
    


etabarDRS=np.mean(etabarGZ_DRS);
etabarCRS=np.mean(etabarGZ_CRS);

# %% Labor Wedge Calculations 
""" Labor Wedge Calculations """

''' DRS '''
logLWedgeGZ           = -neglogLambda_GZ_DRS
logLWedge_GZsym       = -neglogLambda_GZsym_DRS
logLWedge_hor_GZsym   = -neglogLambda_hor_GZsym_DRS
logLWedge_hor_GZasym  = -neglogLambda_hor_GZasym_DRS

''' CRS '''
logLWedgeGZ_CRS           = -neglogLambda_GZ_CRS
logLWedge_GZsym_CRS       = -neglogLambda_GZsym_CRS
logLWedge_hor_GZsym_CRS   = -neglogLambda_hor_GZsym_CRS
logLWedge_hor_GZasym_CRS  = -neglogLambda_hor_GZasym_CRS



""" Take differences """

''' TFP: DRS Baseline with GZ '''
DlogTFP_GZ_DRS              =np.diff(log_A_GZ_DRS)
DlogTFP_GZsym_DRS           =np.diff(log_A_GZsym_DRS)
DlogTFP_hor_GZsym_DRS       =np.diff(log_A_hor_GZsym_DRS)
DlogTFP_hor_GZasym_DRS      =np.diff(log_A_hor_GZasym_DRS)

''' TFP: CRS Baseline with GZ '''
DlogTFP_GZ_CRS              =np.diff(log_A_GZ_CRS)
DlogTFP_GZsym_CRS           =np.diff(log_A_GZsym_CRS)
DlogTFP_hor_GZsym_CRS       =np.diff(log_A_hor_GZsym_CRS)
DlogTFP_hor_GZasym_CRS      =np.diff(log_A_hor_GZasym_CRS)

''' Labor Wedge: DRS Baseline with GZ Shocks '''
DlogLWedge_GZ_DRS           = np.diff(logLWedgeGZ)
DlogLWedge_GZsym_DRS        = np.diff(logLWedge_GZsym)
DlogLWedge_hor_GZsym_DRS    = np.diff(logLWedge_hor_GZsym)
DlogLWedge_hor_GZasym_DRS   = np.diff(logLWedge_hor_GZasym)

''' 'Labor Wedge: CRS Baseline with GZ Shocks '''
DlogLWedge_GZ_CRS           = np.diff(logLWedgeGZ_CRS)
DlogLWedge_GZsym_CRS        = np.diff(logLWedge_GZsym_CRS)
DlogLWedge_hor_GZsym_CRS    = np.diff(logLWedge_hor_GZsym_CRS)
DlogLWedge_hor_GZasym_CRS   = np.diff(logLWedge_hor_GZasym_CRS)


# %% Figures for Decomposition of network_hm Effect 
""" Figures for Decomposition of network_hm Effect """
''' Figures in main text for network_hm multiplier '''

plt.rcParams['text.usetex'] = True
#plt.style.use('seaborn') 
plt.style.use('seaborn-v0_8-whitegrid') # fivethirtyeight is name of style


subtitle_font=18
legend_font = 12
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
axes[0].plot(dYearsvec[8:13],DlogTFP_hor_GZsym_DRS[0,8:13], linewidth='3.5', linestyle='solid', c='blue', label=r'Horizontal, Symmetric $\phi$')
axes[0].plot(dYearsvec[8:13],DlogTFP_GZsym_DRS[0,8:13], linewidth='3.5', linestyle='dashed', c='red', label=r'IO Network, Symmetric $\phi$')
axes[0].plot(dYearsvec[8:13],DlogTFP_GZ_DRS[0,8:13], linewidth='3.5', linestyle=(0, (3, 1, 1, 1)), c='green', label=r'IO Network, Asymmetric $\phi$')
axes[0].set_ylabel(r'$\Delta \log A$',fontsize=subtitle_font)
axes[0].set_title(r'$\Delta \log A$ with DRS',fontsize=subtitle_font)
axes[0].legend(fontsize=legend_font, frameon=False)
axes[1].plot(dYearsvec[8:13],DlogTFP_hor_GZsym_CRS[0,8:13], linewidth='3.5',linestyle='solid', c='blue', label=r'Horizontal, Symmetric $\phi$')
axes[1].plot(dYearsvec[8:13],DlogTFP_GZsym_CRS[0,8:13], linewidth='3.5',linestyle='dashed', c='red', label=r'IO Network, Symmetric $\phi$')
axes[1].plot(dYearsvec[8:13],DlogTFP_GZ_CRS[0,8:13], linewidth='3.5',linestyle=(0, (3, 1, 1, 1)), c='green', label=r'IO Network, Asymmetric $\phi$')
axes[1].set_title(r'$\Delta \log A$ with CRS',fontsize=subtitle_font)
axes[1].set_ylabel(r'$\Delta \log A$',fontsize=subtitle_font)
axes[1].legend(fontsize=legend_font, frameon=False)
fig.suptitle('Figure 1')
plt.savefig('latex/figure1.png')
plt.show()

subtitle_font=18
legend_font = 13
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
axes[0].plot(dYearsvec[8:13],DlogLWedge_hor_GZsym_DRS[0,8:13], linewidth='3.5', linestyle='solid', c='blue', label=r'Horizontal, Symmetric $\phi$')
axes[0].plot(dYearsvec[8:13],DlogLWedge_GZsym_DRS[0,8:13], linewidth='3.5', linestyle='dashed', c='red', label=r'IO Network, Symmetric $\phi$')
axes[0].plot(dYearsvec[8:13],DlogLWedge_GZ_DRS[0,8:13], linewidth='3.5', linestyle=(0, (3, 1, 1, 1)), c='green', label=r'IO Network, Asymmetric $\phi$')
axes[0].set_title(r'$\Delta \log \Lambda$ with DRS',fontsize=subtitle_font)
axes[0].set_ylabel(r'$\Delta \log \Lambda$',fontsize=subtitle_font)
axes[0].legend(fontsize=legend_font, frameon=False)
axes[1].plot(dYearsvec[8:13],DlogLWedge_hor_GZsym_CRS[0,8:13], linewidth='3.5',linestyle='solid', c='blue', label=r'Horizontal, Symmetric $\phi$')
axes[1].plot(dYearsvec[8:13],DlogLWedge_GZsym_CRS[0,8:13], linewidth='3.5',linestyle='dashed', c='red', label=r'IO Network, Symmetric $\phi$')
axes[1].plot(dYearsvec[8:13],DlogLWedge_GZ_CRS[0,8:13], linewidth='3.5',linestyle=(0, (3, 1, 1, 1)), c='green', label=r'IO Network, Asymmetric $\phi$')
axes[1].set_title(r'$\Delta \log \Lambda$ with CRS',fontsize=subtitle_font)
axes[1].set_ylabel(r'$\Delta \log \Lambda$',fontsize=subtitle_font)
axes[1].legend(fontsize=legend_font, frameon=False)
fig.suptitle('Figure 2')
plt.savefig('latex/figure2.png')
plt.show()



# %% Model-implied STATISTICS
""" Model-implied STATISTICS """
''' Table in main text for network_hm multiplier '''


GZ_decomp2009=np.array([DlogTFP_hor_GZsym_DRS[0,13], DlogTFP_GZsym_DRS[0,13], DlogTFP_GZ_DRS[0,13],np.nan,
                        DlogTFP_hor_GZsym_CRS[0,13], DlogTFP_GZsym_CRS[0,13], DlogTFP_GZ_CRS[0,13], np.nan,
                        DlogLWedge_hor_GZsym_DRS[0,13], DlogLWedge_GZsym_DRS[0,13], DlogLWedge_GZ_DRS[0,13], DlogLWedge_GZsym_DRS[0,13]/DlogLWedge_hor_GZsym_DRS[0,13],
                        DlogLWedge_hor_GZsym_CRS[0,13], DlogLWedge_GZsym_CRS[0,13], DlogLWedge_GZ_CRS[0,13], DlogLWedge_GZsym_CRS[0,13]/DlogLWedge_hor_GZsym_CRS[0,13]])

print('Table in the main text');
print('The Labor Wedge network_hm Multiplier during the 2008-2009 Financial Crisis');
GZ_decomp2009 = GZ_decomp2009.T




# %% Aggregate Labor and Consumption in the data
""" Aggregate Labor and Consumption in the data """
''' Aggregate labor in the data '''
aggLabor    =np.sum(value('l_it_mat'),0)
log_aggL    =np.log(aggLabor)

''' Aggregate consumption in the data: total consumption divided by aggregate price level '''
aggPC          =np.sum(value('pc_it_mat'))
log_idealP     =np.diag(value('v_it_mat').T @ value('logp_mat')).T
log_aggC       =np.log(aggPC)-log_idealP

Dlog_aggC     =np.diff(log_aggC)
Dlog_aggL     =np.diff(log_aggL)


# %% Calculate EQUILIBRIUM OUTPUT AND LABOR
""" Calculate EQUILIBRIUM OUTPUT AND LABOR """
''' Parameter Values '''
epsilon=0.5
gamma=0.1

''' Aggregate Labor Wedge in the Data '''
Dlog_dataLW  =(1+epsilon)*Dlog_aggL-(1-gamma)*Dlog_aggC

denom_DRS       =(1+epsilon)-etabarDRS*(1-gamma)
denom_CRS       =(1+epsilon)-etabarCRS*(1-gamma)

Gamma_l_DRS     =(1-gamma)/denom_DRS
Gamma_c_DRS     =(1+epsilon)/denom_DRS
Lambda_l_DRS      =1/denom_DRS
Lambda_c_DRS      =etabarDRS/denom_DRS

Gamma_l_CRS     =(1-gamma)/denom_CRS
Gamma_c_CRS     =(1+epsilon)/denom_CRS
Lambda_l_CRS      =1/denom_CRS
Lambda_c_CRS      =etabarCRS/denom_CRS


''' Equil Labor ''' 
''' Baseline with GZ '''
DlogEQLGZ               =Gamma_l_DRS*DlogTFP_GZ_DRS +Lambda_l_DRS*DlogLWedge_GZ_DRS
DlogEQL_GZsym           =Gamma_l_DRS*DlogTFP_GZsym_DRS +Lambda_l_DRS*DlogLWedge_GZsym_DRS
DlogEQL_hor_GZsym       =Gamma_l_DRS*DlogTFP_hor_GZsym_DRS +Lambda_l_DRS*DlogLWedge_hor_GZsym_DRS
DlogEQL_hor_GZasym      =Gamma_l_DRS*DlogTFP_hor_GZasym_DRS +Lambda_l_DRS*DlogLWedge_hor_GZasym_DRS

''' CRS Baseline with GZ '''
DlogEQLGZ_CRS           =Gamma_l_CRS*DlogTFP_GZ_CRS +Lambda_l_CRS*DlogLWedge_GZ_CRS
DlogEQL_GZsym_CRS       =Gamma_l_CRS*DlogTFP_GZsym_CRS +Lambda_l_CRS*DlogLWedge_GZsym_CRS
DlogEQL_hor_GZsym_CRS   =Gamma_l_CRS*DlogTFP_hor_GZsym_CRS +Lambda_l_CRS*DlogLWedge_hor_GZsym_CRS
DlogEQL_hor_GZasym_CRS  =Gamma_l_CRS*DlogTFP_hor_GZasym_CRS +Lambda_l_CRS*DlogLWedge_hor_GZasym_CRS


''' Equil Consumption '''
''' Baseline with GZ '''
DlogEQCGZ               =Gamma_c_DRS*DlogTFP_GZ_DRS +Lambda_c_DRS*DlogLWedge_GZ_DRS
DlogEQC_GZsym           =Gamma_c_DRS*DlogTFP_GZsym_DRS +Lambda_c_DRS*DlogLWedge_GZsym_DRS
DlogEQC_hor_GZsym       =Gamma_c_DRS*DlogTFP_hor_GZsym_DRS +Lambda_c_DRS*DlogLWedge_hor_GZsym_DRS
DlogEQC_hor_GZasym      =Gamma_c_DRS*DlogTFP_hor_GZasym_DRS +Lambda_c_DRS*DlogLWedge_hor_GZasym_DRS

''' CRS Baseline with GZ '''
DlogEQCGZ_CRS           =Gamma_c_CRS*DlogTFP_GZ_CRS +Lambda_l_CRS*DlogLWedge_GZ_CRS
DlogEQC_GZsym_CRS       =Gamma_c_CRS*DlogTFP_GZsym_CRS +Lambda_l_CRS*DlogLWedge_GZsym_CRS
DlogEQC_hor_GZsym_CRS   =Gamma_c_CRS*DlogTFP_hor_GZsym_CRS +Lambda_l_CRS*DlogLWedge_hor_GZsym_CRS
DlogEQC_hor_GZasym_CRS  =Gamma_c_CRS*DlogTFP_hor_GZasym_CRS +Lambda_l_CRS*DlogLWedge_hor_GZasym_CRS


# %% FIGURES for Model-Implied TFP and Labor Wedge
""" FIGURES for Model-Implied TFP and Labor Wedge """
''' Figures in online appendix '''

subtitle_font=15
legend_font = 12
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
axes[0].plot(Yearsvec,DlogTFP_GZ_DRS.T, linewidth='3.5', linestyle='dashed', c='midnightblue', label=r'$\Delta \log $TFP, DRS')
axes[0].plot(Yearsvec,DlogTFP_GZ_CRS.T, linewidth='3.5', linestyle=(0, (3, 1, 1, 1)), c='mediumvioletred', label=r'$\Delta \log $TFP, CRS')
#axes[0].plot(dYearsvec,Dlog_dataTFP, linewidth='3.5', linestyle=(0, (3, 1, 1, 1)), c='cyan', label=r'$\Delta \log $TFP, CRS')
axes[0].set_ylabel(r'$\Delta \log TFP$',fontsize=subtitle_font)
#axes[0].set_title(r'$\Delta \log A$ with DRS',fontsize=subtitle_font)
axes[0].legend(fontsize=legend_font, frameon=False)
axes[1].plot(Yearsvec,DlogLWedge_GZ_DRS.T, linewidth='3.5', linestyle='dashed', c='midnightblue', label=r'$\Delta \log $TFP, DRS')
axes[1].plot(Yearsvec,DlogLWedge_GZ_CRS.T, linewidth='3.5', linestyle=(0, (3, 1, 1, 1)), c='mediumvioletred', label=r'$\Delta \log $TFP, CRS')
axes[1].plot(Yearsvec,Dlog_dataLW.T, linewidth='3.5', linestyle='dotted', c='seagreen', label=r'$\Delta \log \Lambda$, data')
#axes[1].set_title(r'$\Delta \log A$ with CRS',fontsize=subtitle_font)
axes[1].set_ylabel(r'$\Delta \log \Lambda$',fontsize=subtitle_font)
axes[1].legend(fontsize=legend_font, frameon=False)
fig.suptitle('Figure 3')
plt.savefig('latex/figure3.png')
plt.show()


# %% Decomposition of Output and Labor by TFP and Labor Wedge


subtitle_font=15
legend_font = 12
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 6))
Y = np.hstack((Gamma_c_DRS*DlogTFP_GZ_DRS.T,Lambda_c_DRS*DlogLWedge_GZ_DRS.T))
axes[0,0].fill_between(Yearsvec, Y[:,0],color='midnightblue',edgecolor='black', alpha=1, label='TFP')
axes[0,0].fill_between(Yearsvec, Y[:,1],color='seagreen', edgecolor='black', alpha=0.5, label='Labor Wedge')
# axes[0,0].fill_between(Yearsvec, Y[:,0],color='midnightblue',edgecolor='black', alpha=1)
# axes[0,0].fill_between(Yearsvec, Y[:,1],color='seagreen', edgecolor='black', alpha=0.5)
axes[0,0].set_ylabel(r'$\Delta \log C$',fontsize=subtitle_font)
axes[0,0].legend(fontsize=legend_font, frameon=False,loc='lower left')
axes[0,0].set_title(r'$\Delta \log C$, DRS',fontsize=subtitle_font)


Y = np.hstack((Gamma_c_CRS*DlogTFP_GZ_DRS.T,Lambda_c_CRS*DlogLWedge_GZ_DRS.T))
axes[0,1].fill_between(Yearsvec, Y[:,0],color='midnightblue',edgecolor='black', alpha=1)
axes[0,1].fill_between(Yearsvec, Y[:,1],color='seagreen', edgecolor='black', alpha=0.5)
axes[0,1].set_ylabel(r'$\Delta \log C$',fontsize=subtitle_font)
axes[0,1].set_title(r'$\Delta \log C$, CRS',fontsize=subtitle_font)


Y = np.hstack((Gamma_l_DRS*DlogTFP_GZ_DRS.T,Lambda_l_DRS*DlogLWedge_GZ_DRS.T))
axes[1,0].fill_between(Yearsvec, Y[:,0],color='midnightblue',edgecolor='black', alpha=1)
axes[1,0].fill_between(Yearsvec, Y[:,1],color='seagreen', edgecolor='black', alpha=0.5)
axes[1,0].set_ylabel(r'$\Delta \log L$',fontsize=subtitle_font)
axes[1,0].set_title(r'$\Delta \log L$, DRS',fontsize=subtitle_font)

Y = np.hstack((Gamma_l_CRS*DlogTFP_GZ_DRS.T,Lambda_l_CRS*DlogLWedge_GZ_DRS.T))
axes[1,1].fill_between(Yearsvec, Y[:,0],color='midnightblue',edgecolor='black', alpha=1)
axes[1,1].fill_between(Yearsvec, Y[:,1],color='seagreen', edgecolor='black', alpha=0.5)
axes[1,1].set_ylabel(r'$\Delta \log L$',fontsize=subtitle_font)
axes[1,1].set_title(r'$\Delta \log L$, CRS',fontsize=subtitle_font)
fig.suptitle('Figure 4')
plt.savefig('latex/figure4.png')
plt.show()



