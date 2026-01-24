#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 11:24:35 2023

@author: peppecon
"""

import numpy as np
from scipy.stats import norm

def first_stage(df,S,T):
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



def optimal_cutoffs(p):
    choice_vf_diff = norm.ppf(p[:,1])*np.sqrt(2)
    return choice_vf_diff
