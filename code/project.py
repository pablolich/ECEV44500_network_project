#!/usr/bin/env python3

__appname__ = '[project.py]'
__author__ = 'Pablo Lechon (plechon@ucm.es)'
__version__ = '0.0.1'

## IMPORTS ##

import sys
from models import *
import numpy as np
import pandas as pd
import itertools
from scipy.integrate import solve_ivp
import progressbar
import matplotlib.pylab as plt

## FUNCTIONS ##

def metabolic_matrix(m):
    '''
    Generage vector of concentration parameters of a Dirichlet distribution

    Parameters: 
        m (int): Number of metabolites
    '''
    #Get concentration parameters for each row (all the same)
    c_vec = np.ones(m) 
    #Sample matrix 
    D = np.random.dirichlet(c_vec, m)
    return D

def preferences_number(beta, m):
    '''
    Sample the number of preferences from an exponential distribution
    '''
    #Make sure that it is greater than 1, but smaller than the total number
    #of resources
    n = 0
    while n < 1 or n > m/3:
        n = round(np.random.exponential(beta))
    return n

def community_facilitation(c1, c2,  D, l1, l2):
    '''
    Compute the community-level facilitation, i.e., the total amount of energy
    that is leaked to the environment and captured by other species (or the 
    same one)
    '''
    return (c1 @ D @ np.transpose(c2))

def community_competition(c1, c2, D, l1, l2):
    '''
    Compute the community/communities-level competition, i.e., the total amount
    of energy that is overlappingly required by all species pairs in the 
    community/mix
    '''
    shap1 = np.shape(c1)
    shap2 = np.shape(c2)
    #Calculate amount of competition in the community
    C = c1@ np.transpose(c2)
    #Get upper diagonal (diag included) indices of matrix of competition.
    #Note that these are the only indices that we want because Cb is symmetric
    ind = np.triu_indices(np.shape(C)[0])
    n_sp = len(ind[0])
    #Preallocate matrix of biotic competition
    Cb = np.zeros(shape = np.shape(C))
    Cb1 = np.zeros(shape = np.shape(C))
    for i in range(n_sp):
        a = ind[0][i]
        b = ind[1][i]
        c_a = c1[a,:]
        c_b = c2[b,:]
        Sab = c_a + c_b
        Pab  = c_a * c_b
        Cb[a, b] = Sab @ D @ np.transpose(Pab)
    #Make this matrix symmetric (copy the upper triangular part to the lower
    #one)
    i_lower = np.tril_indices(shap1[0], -1)
    Cb[i_lower] = Cb.T[i_lower]
    return((1-l1)*C + l1*Cb)

def preferences_matrix(s, m, nr, beta = 5):
    '''
    Construct preference matrix
    Parameters:
        Kc (float): Taxonomic heterogeneity constant
        m (int): Number of resources
        s (int): Number of species
        c (int): Number of classes 
    '''
    #Preallocate matrix of preferences
    c_mat = np.zeros(shape = (s, m))
    for i in range(s):
        #Preallocate preferences vector
        c_vec = np.zeros(m)
        #Sample number of preferences
        #n = preferences_number(beta, m = m)
        #Draw indices with probability p
        ind = np.random.choice(range(m), nr[i], replace = False,
                               p = np.repeat(1/m, m))
        #Switch up those indices
        c_vec[ind] = 1
        c_mat[i,:] = c_vec
    return c_mat

def growth_rates(l, c, R0, z):
    '''
    Calculate growth rates of GLV according to the consumer resouce parameters
    Parameters:
        l (float): Leakage value
        c (sxm): Preferences matrix
        R0 (mx1): Initial suppply of resources
        z (sx1): Cost of each species
    '''
    return((1-l) * (c @ R0 - z))


def main(argv):
    '''Main function'''

    #Exponential rate to draw preferences
    beta = 5
    #Cost per pathway 
    cost_path = 0.1
    #Number of simulations
    n_sim = 250
    #Leakage values
    l = np.arange(0.01, 0.95, 0.04)
    #l = np.array([0.01, 0.1, 0.2, 0.5, 0.7, 0.9])
    #Number of species
    s = np.arange(10, 60, 5, dtype = int) 
    #s = np.array([10, 20, 30, 40, 50, 60])
    #Create N-D parameter grid 
    product = itertools.product(range(n_sim), l, s)
    #Create column names of data frame
    col_names = ['n_sim', 'l', 's']
    #Create dataframe for storing parameter values and simulation results
    df = pd.DataFrame(data = np.array(list(product)), columns = col_names)
    #Number of iterations
    n_it = len(df)
    #Add endpoint columns for CR and GLV
    df['cr_mean'] = np.zeros(n_it)
    df['glv_mean'] = np.zeros(n_it)
    #Flag for divergent dynamics
    df['div'] = np.zeros(n_it)
    ##Preallocate matrices of endpoints for each model
    #endpoints_mat_CR = np.zeros(shape = (n_it, s))
    #endpoints_mat_GLV = np.zeros(shape = (n_it, s))
    for i in progressbar.progressbar(range(n_it)):
        #Extract number of species to variable s
        s = int(df['s'][i])
        m = s
        #Number of reactions
        nr = np.random.randint(1, m, s)
        #Sample preference matrix
        c = preferences_matrix(s, m, nr)
        #Sample metabolic matrix
        D = np.random.dirichlet(np.ones(m), m)
        #Create time vector
        tspan = tuple([1, 1e4])
        #Set initial conditions
        z0 = list(np.ones(s))+list(2*np.ones(m))
        #Calculate vector of costs
        x = cost_path*nr.reshape(s, 1)
        #Create a dictionary of parameters for CR
        params_CR = {'s':s,
                     'm':m,
                     'g':np.ones(s).reshape(s,1),
                     'c':c,
                     'l':np.repeat(df['l'][i], m).reshape(m, 1),
                     'x':x,
                     'D':D,
                     'K':20*np.ones(m).reshape(m,1),
                     't':0.5*np.ones(m).reshape(m,1)
                    }
        #Integrate CR
        sol_CR = solve_ivp(lambda t,z: consumer_resouce_crossfeeding(t,z, params_CR),
                           tspan, z0,
                           method = 'BDF', atol = 0.01 )
        #Calculate competititon and facilitation matrices
        F = community_facilitation(c, c, D, df['l'][i], df['l'][i])
        C = community_competition(c, c, D, df['l'][i], df['l'][i])
        #Get matrix A of interactions as F - C
        A = -F - C
        #Get growth rates
        r = growth_rates(df['l'][i], c, 2*np.ones(m).reshape(m, 1), x) 
        #r = np.repeat(1, s).reshape(s, 1)
        #Set initial conditions
        N0 = list(np.ones(s))
        #Create a dictionary of parameters for GLV
        params_GLV = {'s':s,
                      'r':r,
                      'A':A
                     }
        #Integrate GLV
        sol_GLV = solve_ivp(lambda t, N: lotka_volterra(t, N, params_GLV),
                            tspan, N0,
                            method = 'BDF', atol = 0.01 )
        #Stable abundances
        x_glv = sol_GLV.y[:, -1]
        x_cr = sol_CR.y[0:s, -1]
        #Set divergent indices
        div_glv = np.where(abs(x_glv) > 1e3)[0]
        div_cr = np.where(abs(x_cr) > 1e3)[0]
        #If integration diverges, flag it
        if any(div_glv) or any(div_cr):
            df.loc[i, 'div'] = 1
            continue
        #Store mean abundance
        df.loc[i, 'glv_mean'] = np.mean(x_glv)
        df.loc[i, 'cr_mean'] = np.mean(x_cr)
    #Save output to data
    df.to_csv("../data/results_fc.txt", index = False)

    return 0

## CODE ##

if (__name__ == '__main__'):
    status = main(sys.argv)
    sys.exit(status)
     
     
