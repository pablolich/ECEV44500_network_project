#!/usr/bin/env python3

__appname__ = '[assembly_clean.py]'
__author__ = 'Pablo Lechon (plechon@ucm.es)'
__version__ = '0.0.1'

## IMPORTS ##

import sys
from model import *
import itertools
from functions_clean import *
import pandas as pd
import progressbar
#Erase
import matplotlib.pylab as plt

## CONSTANTS ##

global seed; seed = 69

## FUNCTIONS ##

def main(argv):
    '''Main function'''

    #Environment with s strains and m metabolites
    m = 60
    s = 60
    #Partition families
    n_fam = 4
    n_memb = 15
    #Exponential rate to draw preferences
    beta = 5
    #Create a dictionary of parameters
    params = {'g':np.ones(s).reshape(s,1),
              's':s,
              'm':m,
              'K':20*np.ones(m).reshape(m,1),
              't':0.5*np.ones(m).reshape(m,1),
              'coal':0 #This is not a coalescence event
             }
    #Create time vector
    tspan = tuple([1, 1e4])
    #Set initial conditions
    z0 = list(np.ones(s))+list(2*np.ones(m))
    #Create parameter vectors
    kc = np.linspace(0.01, 0.9, num = 3)
    kf = np.linspace(0, 0.99, num = 3)
    K = np.linspace(0, 0.9, num = 2)
    l = np.array([0.1, 0.2, 0.3, 0.4, 0.50,
                  0.50, 0.6, 0.7, 0.8, 0.9])
    nsim = range(100)
    #Create N-D parameter grid 
    product = itertools.product(kc, kf, K, l, nsim)
    #Create column names of data frame
    col_names = ['kc', 'kf', 'K', 'l', 'n_sim']
    #Create dataframe for storing parameter values and simulation results
    df = pd.DataFrame(data = np.array(list(product)), columns = col_names)
    #Preallocate columns for C0 and F0 calculations
    ncol = len(df)
    df['C0av'] = np.zeros(ncol)
    df['C0bav'] = np.zeros(ncol)
    df['F0av'] = np.zeros(ncol)
    #Preallocate columns for C and F calculations (after community assembly)
    df['Cav'] = np.zeros(ncol)
    df['Cbav'] = np.zeros(ncol)
    df['Fav'] = np.zeros(ncol)
    #Preallocate column for richness
    df['r'] = np.zeros(ncol, dtype = int)
    #Preallocte column for total resouce abundance at equilibrium
    df['ER'] = np.zeros(ncol)
    #Preallocte column for resouce abundance standard deviation
    df['std'] = np.zeros(ncol)
    #Preallocte column for resouce abundance mean
    df['mean'] = np.zeros(ncol)
    #Preallocate columns for preference and metabolic matrices
    all_c = pd.DataFrame(data = np.zeros(shape = (s*ncol, m + 1), dtype = int))
    #Note that I add one column at the end to specify wether that species is 
    #extinct or not after assembly.
    all_D = pd.DataFrame(data = np.zeros(shape = (m*ncol, m)))
    all_abundances = pd.DataFrame(data = np.zeros(shape = (ncol, s)))
    #Set indices to number each community
    all_c = all_c.set_index(np.repeat(np.arange(ncol), s))
    all_D = all_D.set_index(np.repeat(np.arange(ncol), m))
    #Run simulations across the parameter grid
    for i in progressbar.progressbar(range(ncol)):
       #Create vector of species classes
        sc_vec = np.repeat(np.arange(n_fam), n_memb)
        #Draw preference matrix
        if df['K'][i] == 0:
            c = preference_matrix(m, s, df['kc'][i],
                                  beta = beta, n_r = None)
        else:
            c = gen_pref_matrix(df['K'][i], df['kc'][i], m, n_memb, sc_vec, 
                                sc_vec)
        #Compute demands as the sum of the rows of c
        demands = np.sum(c, axis = 0)
        #Sample metabolic matrix
        #Get matrix of classes
        Mc = class_matrix(m, n_fam)
        D = general_metabolic_matrix(0.05, df['kf'][i], demands, 
                                     df['K'][i], Mc, m, n_memb)
        #Store in dataframe
        all_D[m*i:m*(i+1)]= D
        #Compute costs
        maint_vec = maintenance(c) 
        #Calculate competititon and facilitation  
        F_cy = community_facilitation(c, c, D, df['l'][i], df['l'][i])
        C_cy = community_competition(c, c, D, df['l'][i], df['l'][i])
        #Average non-zero elements to get community level facilitation and 
        #competition leaving out the 0 of the diagonal
        df.loc[i, 'F0av'] = F_cy[1]
        df.loc[i, 'C0av'] = C_cy[1]
        df.loc[i, 'C0bav'] = C_cy[2]
        #Calculate cost of each species
        maint_vec = maintenance(c)
        #Add sampled strategies, metabolic, costs, and leakage
        #to the parameter dictionary
        params['D'] = D
        params['c'] = c
        params['x'] = maint_vec.reshape(s,1)
        params['l'] = df['l'][i]*np.ones(m).reshape(m,1)
        #Solve diferential equations
        sol = solve_ivp(lambda t,z: equations(t,z, params),
                        tspan, z0,
                        method = 'BDF', atol = 0.0001 )
        #Record resource abundance at equilibrium
        stable_concentration = sol.y[s:s+m, -1]
        df.loc[i, 'ER'] = sum(stable_concentration)
        df.loc[i, 'std'] = np.std(stable_concentration)
        df.loc[i, 'mean'] = np.mean(stable_concentration)
        #Get abundance of species at stable state
        stable_abundance = sol.y[0:s,-1]
        #Store in dataframe
        all_abundances.iloc[i] = stable_abundance
        #Get indices of extant species
        ind_extant = np.where(stable_abundance > 1)[0]
        #Record species richness
        df.loc[i, 'r'] = len(ind_extant)
        #Create a vector of extant species
        extant = -1*np.ones(shape = (s, 1))
        #Flip to 1 the extant ones
        extant[ind_extant] = 1
        c_tot = np.hstack([c, extant])
        #Store in data frame
        all_c[s*i:s*(i+1)] = c_tot
        #Get rid of these rows  in the matrix of preferences
        c_assembly = c[ind_extant,:]
        #Recalculate competition and facilitation 
        F_cy = community_facilitation(c_assembly, c_assembly, D, 
                                      df['l'][i], df['l'][i])
        C_cy = community_competition(c_assembly, c_assembly, D,
                                     df['l'][i], df['l'][i])
        #Store
        df.loc[i, 'Fav'] = F_cy[1]
        df.loc[i, 'Cav'] = C_cy[1]
        df.loc[i, 'Cbav'] = C_cy[2]

    #Save results
    if len(sys.argv) > 1:
        df.to_csv('../data/simulation_results_'+sys.argv[1]+'.csv',
                  index = False)
        all_c.to_csv('../data/c_matrices_'+sys.argv[1]+'.csv')
        all_D.to_csv('../data/D_matrices_'+sys.argv[1]+'.csv')
        all_abundances.to_csv('../data/abundances_'+sys.argv[1]+'.csv')
    else:
        df.to_csv('../data/simulation_results.csv', index = False)
        all_c.to_csv('../data/c_matrices.csv')
        all_D.to_csv('../data/D_matrices.csv')
        all_abundances.to_csv('../data/abundances.csv')
    return 0

## CODE ##

if (__name__ == '__main__'):
    status = main(sys.argv)
    sys.exit(status)
     

