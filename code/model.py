#!/usr/bin/env python3

__appname__ = '[App_name_here]'
__author__ = 'Pablo Lechon (plechon@ucm.es)'
__version__ = '0.0.1'

## IMPORTS ##

import sys
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import itertools
import matplotlib.pylab as plt

## CONSTANTS ##

global seed; seed = 69

## FUNCTIONS ##
def maintenance(c, cost = 0.1):
    '''
    Calculates maintenance cost based on number of metabolic pathways
    '''

    #Get number of species
    s = np.shape(c)[0]
    #Get number of pathways of each species
    n_path = c.sum(axis = 1)
    fluc = np.random.normal(0, 0.1, size = s)
    for i in range(len(fluc)):
        while abs(fluc[i]) > 0.1:
            fluc[i] = np.random.normal(0, 0.1)
    m = cost*n_path + fluc
    return(m)

def rescaling(v):
    '''
    Rescaling vector v to be between 0 and 1
    '''

    return((v - min(v) * np.heaviside(-min(v), 0)) / \
           (sum(v - min(v) * np.heaviside(-min(v), 0) )))

def crossfeeding_matrix(demands, m, kf):
    '''
    Create metabolic matrix
    '''

    #Base level cross-feeding is given by a Dirichlet distribution with low
    #variance
    D_base = np.random.dirichlet(100*np.ones(m), m)
    #Base level cross-feeding is given by a uniform distribution. 
    #D_base = 1/m*np.ones(shape = (m, m))
    #Initialite matrix of metabolic cross-feeding based on demands
    D_dem = np.ones(shape = (m, m), dtype = int)
    #Create iterator with non-diagonal indices of matrix
    ind = itertools.permutations(np.arange(m), 2)
    for (i, j) in ind:
        D_dem[i, j] = (demands[j]-demands[i] ) \
        * np.heaviside((demands[j] - demands[i]), 0.5)
    
    #Get normalization constants for each column
    D_resc = (rescaling(D_dem.reshape(m*m, 1))).reshape(m,m)
    norm = np.sum(D_dem, axis = 1)
    D_norm = D_dem / norm[:, None]
    D_norm[np.isnan(D_norm)] = 0
    #Add base metabolic matrix and demand-based metabolic matrix weighted by k
    D = (1 - kf) * D_base + kf * D_norm
    #Rescale  rows of D so that they are all positive and add up to 1
    for i in range(m):
        D[i,:] = rescaling(D[i,:])
    return(D)

def sampling_probability(kc, cumulative_demand, m):
    '''
    Compute the sampling probability vector
    '''

    if not any(cumulative_demand):
        return(1/m*np.ones(m))
    else:
        base = (1 - kc) * 1 / m
        regime = kc * cumulative_demand/sum(cumulative_demand)
        #Transform nans to 0
        regime[np.isnan(regime)] = 0
        #Rescale to get probabilities
        score = base + regime
        #Rescale to get probabilities
        #probability =  (score - min(score) * np.heaviside(-min(score), 0)) / \
        #               (1 - m * min(score) * np.heaviside(-min(score), 0))
        return(score)

def preference_matrix(m, s, kc, beta, n_r):
    '''
    Create metabolic preferences matrix
    '''

    #Prealocate matrix
    c = np.zeros(shape = (s, m), dtype = int)
    for i in range(s):
        #Sample number of imported metabolites
        n = 0
        #All species having the same number of preferences
        if n_r:
            n = round(n_r)
        #Number of preferences follows an exponential distribution with rate
        #beta
        elif beta:
            #Make sure that it is greater than 1, but smaller than the total number
            #of resources
            while n < 1 or n > m/3:
                n = round(np.random.exponential(beta))
        #Compute the cumulative demand
        cum = np.sum(c[0:i,:], axis = 0)
        #Get sampling probability
        prob = sampling_probability(kc, cum, m)
        #Draw indices with probability p
        ind = np.random.choice(range(m), n, replace = False, p = prob)
        #Flip the ind positions in the vector of strategies for species sp
        c[i,ind] = 1
    return c

def equations(t, z, params):

    '''
    Diferential equations of marsland model in matrix form

    Parameters: 
                s (int): number of species
                m (int): number of resources
                g (sx1): proportionality constant harvested energy --> abundance
                N (sx1): abundance of each strain
                c (sxm): preference matrix of the community
                l (mx1): leakage factor of each resource
                x (sx1): maintenance cost of each species
                D (mxm): metabolic matrix of community
                coal (bool): is the assembly a coalescence event, or not?

    Output:
                list (1x(s+m)): abundance of each species and resource after
                                one time iteration
    '''

    #Unpack parameter values
    s, m,  g, c, l, x, D, K, t, coal = map(params.get, 
                                     ('s','m','g','c','l','x','D', 'K', 't',
                                      'coal'))
    #Separate species and resource vector and reshape them to columns vectors
    N = np.array(z[0:s]).reshape(s,1)
    R = np.array(z[s:m+s]).reshape(m,1)
    if not coal :
        #Compute one iteration step
        dNdt = g * N * (c @ ((1 - l) * R) - x)
        #Normal equation for resources
        dRdt = K - 1 / t * R - (c.transpose() @ N) * R + \
               D.transpose() @ ((l * R) * c.transpose()) @ N
    if coal:
        #Extend the vector of resources
        Re = np.vstack( [R]*(int(np.shape(D)[0]/m)) )
        #Compute one iteration step
        dNdt = g * N * (c @ ((1 - l) * Re) - x)
        #Construct matrix to sum the effects on resource due to each community
        sum_mat = np.concatenate([np.identity(m)]*(int(np.shape(D)[0]/m)), 
                                 axis = 1)
        #Sum the effect of the two communities in the last term
        #1 / t * R
        dRdt = K - sum_mat @ ((c.transpose() @ N) * Re + \
               D.transpose() @ ((l * Re) * c.transpose()) @ N)
    return(list(dNdt.reshape(s))+list(dRdt.reshape(m)))
    
