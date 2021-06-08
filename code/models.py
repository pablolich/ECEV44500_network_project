#!/usr/bin/env python3

__appname__ = '[models.py]'
__author__ = 'Pablo Lechon (plechon@uchicago.edu)'
__version__ = '0.0.1'

## IMPORTS ##

import numpy as np

## FUNCTIONS ##

def lotka_volterra(t,  N, params):


    '''
    Differential equations of a GLV 

    Parameters: 
                s (int): number of species
                r (sx1): growth rate of each species
                A (sxs): Matrix of interactions

     Output:
                 list (1xs): abundance of each species after one time 
                             iteration
    ''' 
     
    #Unpack parameter values
    s, r, A, = map(params.get, ('s', 'r', 'A'))
    D_N = np.diag(N)
    #Reshape N to column vector
    N = np.array(N).reshape(s,1)
    #Perform one iteration for variation of species abundances
    dNdt = D_N @ (r + A @ N)
    return(list(dNdt.reshape(s)))

def consumer_resouce_crossfeeding(t, z, params):

    '''
    Diferential equations of marsland model in matrix form

    Parameters: 
                s (int): number f species
                m (int): number of resources
                g (sx1): proportionality constant harvested energy --> abundance
                N (sx1): abundance of each strain
                c (sxm): preference matrix of the community
                l (mx1): leakage factor of each resource
                x (sx1): maintenance cost of each species
                D (mxm): metabolic matrix of community

    Output:
                list (1x(s+m)): abundance of each species and resource after
                                one time iteration
    '''

    #Unpack parameter values
    s, m,  g, c, l, x, D, K, t = map(params.get,('s','m','g','c','l','x','D', 
                                     'K', 't'))
    #Separate species and resource vector and reshape them to columns vectors
    N = np.array(z[0:s]).reshape(s,1)
    R = np.array(z[s:m+s]).reshape(m,1)
    #Compute one iteration step for species and resouce abundandces
    dNdt = g * N * (c @ ((1 - l) * R) - x)
    dRdt = K - 1 / t * R  - (c.transpose() @ N) * R + \
           D.transpose() @ ((l * R) * c.transpose()) @ N
    return(list(dNdt.reshape(s))+list(dRdt.reshape(m)))
