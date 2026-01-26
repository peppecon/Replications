#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 11:44:07 2023

@author: peppecon
"""

import numpy as np

def Chebyshev_Polynomials_Recursion_mv(x,p):
    ''' Multivariate case '''    
    T = np.zeros([p,len(x)])
    T[0,:] = 1
    T[1,:] = x    
    for j in range(1,p-1):
        T[j+1,:] = 2*x*T[j,:] - T[j-1,:]    
    return T


def Change_Variable_Tocheb(bmin, bmax, b):    
    x = 2*b/(bmax - bmin) - (bmin+bmax)/(bmax-bmin)
    return x



def Change_Variable_Fromcheb(bmin, bmax, x):             
    # b_rev = (x + 1)*(bmax - bmin)/2 + bmin    
    b_rev = (bmin + bmax)/2 + ((bmax - bmin)/2)*x
    return b_rev
    


def Chebyshev_Nodes(n):    
    cheb_nodes = np.zeros([n,1])    
    for k in range(0,n):
        ''' We have to adjust the Chebyshev nodes formula because Python starts
            counting from 0, hence we need k+1, but the indexing is still k'''
        cheb_nodes[k] = np.cos(((2*(k+1)-1)/(2*n))*np.pi)    
    return cheb_nodes       


def Tx(n_x,p_x):
    ''' Get the Tx operator - univariate '''
    cheb_nodes_x = Chebyshev_Nodes(n_x).ravel()
    T_x = Chebyshev_Polynomials_Recursion_mv(cheb_nodes_x,p_x)
    return T_x,cheb_nodes_x


def Tx_new_points(X,p_x):
    ''' Get the Tx operator - univariate '''
    cheb_nodes_x = Change_Variable_Tocheb(np.min(X), np.max(X), X)
    T_x = Chebyshev_Polynomials_Recursion_mv(cheb_nodes_x,p_x)
    return T_x,cheb_nodes_x



def Tenser_Product_bv(n_x,n_y,p_x,p_y):
    """ Bivariate Tenser Product:
        obtain the tenser product between Tx and Ty """
    cheb_nodes_x = Chebyshev_Nodes(n_x).ravel()
    cheb_nodes_y = Chebyshev_Nodes(n_y).ravel()
    ''' Visualize the grid '''
    #cheb_nodes_xy, cheb_nodes_yx = np.meshgrid(cheb_nodes_x,cheb_nodes_y)
    #grid_cheb_xy = np.array((cheb_nodes_xy.ravel(), cheb_nodes_yx.ravel())).T
    #plt.plot(cheb_nodes_xy, cheb_nodes_yx, marker='o', color='k', linestyle='none')
    #plt.show()

    T_x = Chebyshev_Polynomials_Recursion_mv(cheb_nodes_x,p_x)
    T_y = Chebyshev_Polynomials_Recursion_mv(cheb_nodes_y,p_y)

    # I have to transpose T_x cause the first column as the first node, the second
    # column the second node and so on and so forth.
    # These below should be equivalent (just give a different format in Python)
    #tens_xy = np.tensordot(T_x.T,T_y.T, axes = 0)
    kron_xy = np.kron(T_x,T_y)
    return kron_xy,cheb_nodes_x,cheb_nodes_y

def Tenser_Product_new_points(X,Y,p_x,p_y):
    """ Bivariate Tenser Product:
        obtain the tenser product between Tx and Ty """
    cheb_nodes_x = Change_Variable_Tocheb(np.min(X), np.max(X), X)
    cheb_nodes_y = Change_Variable_Tocheb(np.min(Y), np.max(Y), Y)
    
   
    ''' Visualize the grid '''
    #cheb_nodes_xy, cheb_nodes_yx = np.meshgrid(cheb_nodes_x,cheb_nodes_y)
    #grid_cheb_xy = np.array((cheb_nodes_xy.ravel(), cheb_nodes_yx.ravel())).T
    #plt.plot(cheb_nodes_xy, cheb_nodes_yx, marker='o', color='k', linestyle='none')
    #plt.show()

    T_x = Chebyshev_Polynomials_Recursion_mv(cheb_nodes_x,p_x)
    T_y = Chebyshev_Polynomials_Recursion_mv(cheb_nodes_y,p_y)

    # I have to transpose T_x cause the first column as the first node, the second
    # column the second node and so on and so forth.
    # These below should be equivalent (just give a different format in Python)
    #tens_xy = np.tensordot(T_x.T,T_y.T, axes = 0)
    kron_xy = np.kron(T_x,T_y)
    return kron_xy

