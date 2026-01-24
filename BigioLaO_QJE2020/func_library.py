#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 18:12:06 2023

@author: peppecon
"""

import numpy as np

def network_hm(phi_vec,logphi_vec,W_mat,v_vec,alpha_vec,eta_vec,NsecGZ):
    
    onevec = np.ones([NsecGZ,1])
   
    I = np.eye(NsecGZ)
    #G = onevec.T @ value('logphi_GZ_mat')[:,1]
    #W_mat = Wii
    #G = onevec.T @ W_mat
    
    ''' Test variables 
    ii = 0
    phi_vec = phi_GZ_mat[:,ii]
    logphi_vec = logphi_GZ_mat[:,ii]
    W_mat = Wii '''
    
    
    phi_vec = phi_vec.reshape(-1,1) 

    #one = (((onevec - alpha_vec) @ onevec.T).T * W_mat)
    #two = onevec@(phi_vec*eta_vec).T
    #three = v_vec @ (onevec - (phi_vec*eta_vec)).T
    #four = I - (((onevec - alpha_vec) @ onevec.T).T * W_mat)*(onevec@(phi_vec*eta_vec).T) (error here)
    a_vec = np.linalg.inv(I - (((onevec - alpha_vec) @ onevec.T) * W_mat).T*(onevec@(phi_vec*eta_vec).T) - v_vec @ (onevec - (phi_vec*eta_vec)).T) @ v_vec
    log_a = np.log(a_vec)
    
    B = np.linalg.inv(I - ((eta_vec*(onevec - alpha_vec)) @ onevec.T) * W_mat)
    q = ((v_vec.T @ B)*(eta_vec.T)).T
    d = (v_vec.T @ ((onevec @ (onevec-eta_vec).T)*B)).T

    psi = onevec.T @ ((onevec - (phi_vec*eta_vec))*a_vec)
    neglogLambda = np.log(1+psi)
    
    etabar =1-(d.T @ onevec)
    
    ''' Compute all the things '''
    qlogphi = q.T @ logphi_vec;
    log_A = qlogphi + neglogLambda - (d.T @ log_a)
    
    return log_A,neglogLambda,etabar
    
