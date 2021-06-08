#!/usr/bin/env python3

__appname__ = '[functions_cleaned.py]'
__author__ = 'Pablo Lechon (plechon@ucm.es)'
__version__ = '0.0.1'

## IMPORTS ##

import sys
import numpy as np
from model import equations
from scipy.integrate import solve_ivp

## CONSTANTS ##


## FUNCTIONS ##

def joint_system(c_1, D_1, N_1, l1, x1, c_2, D_2, N_2, l2, x2):
    '''Create vectors and matrix of the joint system'''
    #Get number of metabolites and species in each community
    m = np.shape(c_2)[1]
    s1 = len(N_1)
    s2 = len(N_2)
    #Create off-diagonal zeros of joint system
    dO1 = np.zeros( shape = (np.shape(D_1)[0], np.shape(D_2)[1]), dtype = int )
    dO2 = np.zeros( shape = (np.shape(D_2)[0], np.shape(D_1)[1]), dtype = int )
    cO1 = np.zeros( shape = (np.shape(c_1)[0], np.shape(c_2)[1]), dtype = int )
    cO2 = np.zeros( shape = (np.shape(c_2)[0], np.shape(c_1)[1]), dtype = int )
    #Create matrices and vectors of the joint system
    D = np.asarray( np.bmat([[D_1, dO1], [dO2, D_2]]) )
    C = np.asarray( np.bmat([[c_1, cO1], [cO2, c_2]]) )
    N = np.hstack( [N_1, N_2] ) #These will be reshaped inside model equations
    l = np.vstack( [(np.ones(np.shape(c_1)[1])*l1).reshape(np.shape(c_1)[1], 1),
                    (np.ones(np.shape(c_2)[1])*l2).reshape(np.shape(c_2)[1], 1)] ) 
    x = np.vstack( [x1.reshape(s1, 1), x2.reshape(s2, 1)] )
    #Create dictionary to return system
    system = {'D':D, 'C':C, 'N':N, 'l':l, 'x':x}
    return(system)

def boolean_similarity(v, v1, v2):
    '''
    Calculate how more (less) similar is vector v to v1 than to v2

    Parameters: 
        v (1:s1+s2): Vector of absence-precence (1-0) values for species in the
                     resulting community
        v1 (1:s1+s2): Vector of absence-precnce values for species in 
                      community 1 with s2 zeros at the end
        v2 (1:s2+s1): Vector of absence-precnce values for species in 
                      community 2 with s1 preceeding zeros
    '''

    #Calculate normalized similarities of v with original communities
    s1 = np.dot(v, v1)/sum(v1)
    s2 = np.dot(v, v2)/sum(v2)
    #Calculate the difference between these similarities
    return(s1-s2)

def community_facilitation(c1, c2,  D, l1, l2):
    '''
    Compute the community-level facilitation, i.e., the total amount of energy
    that is leaked to the environment and captured by other species (or the 
    same one)
    '''
    
    #Calculate the amount of facilitation in the community
    F = c1 @ D @ np.transpose(c2)
    #Merge
    both = np.array([l1*(np.sum(F)-np.trace(F))/(F.size - len(F)), 
                     l1*np.mean(F)])
    #Factor with sumations
    #(l1 * (1 - l2))/(1 - l1 * (1 - l2))
    #(l1-1)*l1/(np.log(l1*(1-l1))) * 
    return (both)

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
    ind = np.triu_indices(np.shape(C)[0], k = 1)
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
    #Merge
    both = np.array([np.sum(C[ind]), 
                     np.mean((1-l1)*C[ind]),
                     np.mean(l1*Cb[ind])])
    return(both)

def class_matrix(m, c):
    '''
    Create a square matrix with 1 for elements of the c-sized box diagonal and 
    0 for elements in the off box diagonals.

    Parameters: 
        m (int): size of the matrix
        c (int): size of the boxes
    '''
    #Initialize matrix
    M = np.zeros(shape = (m, m))
    #Create vector of indices for upper diagonal
    inds = np.triu_indices(m)
    n_it = len(inds[0])
    #Loop over all index pairs of the upper diagonal
    for ind in range(n_it):
        i = inds[0][ind]
        j = inds[1][ind]
        #Number of resources per class 
        mc = m/c
        if np.floor(i/mc) == np.floor(j/mc):
            #The element is in a class
            M[i, j] = 1
            M[j, i] = 1
    return(M)

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

def preferences_vector(Kc, m, nc, sc):
    '''
    Generate vector of metabolic preferences for species s 
    Parameters:
        Kc (float): Taxonomic heterogeneity constant
        m (int): Number of resources
        nc (int): Number of species per resource class
        sc (int): Class to which the species belongs
    '''
    #Preallocate preferences vector
    c_vec = np.zeros(m)
    #Get high and low probabilities
    high = 1/m*(1 + Kc*(m/nc - 1))
    low = 1/m*(1-Kc)
    #Sample number of preferences
    n = preferences_number(beta = 5, m = m)
    #What is the class of species s?
    prob = np.zeros(m)
    prob[nc*sc:nc*(sc+1)] = 1
    ind_0 = np.where(prob == 0)[0]
    prob[ind_0] = low
    ind_1 = np.where(prob == 1)[0]
    prob[ind_1] = high
    #Draw indices with probability p
    ind = np.random.choice(range(m), n, replace = False, p = prob)
    c_vec[ind] = 1
    return(c_vec)

def preferences_matrix(Kc, m, nc, c, s):
    '''
    Construct preference matrix
    Parameters:
        Kc (float): Taxonomic heterogeneity constant
        m (int): Number of resources
        s (int): Number of species
        nc (int): Number of species per resource class
        c (int): Number of classes 
    '''
    #Preallocate matrix of preferences
    c_mat = np.zeros(shape = (s, m))
    #Equal number of species in each class
    Mc = int(np.floor(s/c))
    #Create vector of classes
    class_vec = np.repeat(np.arange(c), Mc)
    for i in range(s):
        c_mat[i,:] = preferences_vector(Kc, m, Mc, 
                                        class_vec[i])
    return c_mat

def concentration_vector(kf, spar, M, alpha, m, Mc):
    '''
    Create concentration vector for each row (substrate)
    '''
    #Preallocate vector
    d = np.zeros(m)
    for prod in range(m):
        #Assign value to each element of the vector
        if M[alpha, prod]: 
            #When product and substrate belong to the same class assign a 
            #smaller value
            d[prod] = (1 - kf)/(spar*Mc)
        else:
            #When product and substrate belong to a different class assign a
            #bigger value
            d[prod] = 1/(spar*Mc)
        #Note that the values of d match independently of  the class of 
        #product and substrate when kf = 0.
    return d

#Functions for general metabolic sampling


def constant_a(T, m, nc, kc, Tc, Kc):
    '''Return constant a'''
    return (Kc + nc/m*(1 - Kc))/((Kc + 1)*(nc/m*(1 - kc) + Tc/T*kc)) 

def prob_piecewise(Kc, kc, m, nc, sc, mc, dnorm, a):
    '''
    Generate probability vector of preference j for species s
    Parameters:
        Kc (float): Taxonomic heterogeneity constant
        kc (float): Competitive factor
        m (int): Total number of resources
        nc (int): Number of resources per class
        sc (int): Class to which species s belongs
        mc (int): Class to which resource j belongs
        dnorm (1xm): Normalized demand of resource j 
    '''
    if sc == mc:
        #resource is in the class to which the species belongs
        return ((1-kc)*1/m + dnorm*kc)*a*(1 + Kc)
    else:
        #resource is not in the class to which the species belongs
        return 1/m*(1-Kc)

def gen_prob_vec(Kc, kc, d, m, nc, sc, mc_vec):
    '''
    Compute sampling probability for the m preferences and species s
    Parameters:
        Kc (float): Taxonomic heterogeneity constant
        kc (float): Competitive factor
        d (1xm): Number of consumers consumer each resource
        m (int): Number of resources
        nc (int): Number of resources per class
        sc (int): Class to which species s belongs
        mc_vec (1xm): Vector of classes to which each resource belong
    '''
    #Calculate total number of demands
    T = np.sum(d)
    #Calculate demands within current class
    Tc = np.array([d[nc*i:nc*(i+1)].sum() for i in range(m//nc)])
    #Calculate normalization constants a and b
    a = constant_a(T, m, nc, kc, Tc, Kc)
    a = np.repeat(a, nc)
    #Preallocate vector of probabilities of each metabolite
    p_j = np.zeros(m)
    #Assign probability to each metabolite
    for j in range(m):
        p_j[j] = prob_piecewise(Kc, kc, m, nc, sc, mc_vec[j], d[j]/T, a[j])
    return p_j

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

def gen_pref_matrix(Kc, kc, m, nsp, sc_vec, mc_vec, beta = 5):
    '''
    Sample preference matrix
    Parameters:
        Kc (float): Taxonomic heterogeneity constant
        kc (float): Competitive factor
        m (int): Number of resources
        nsp (int): Number of resources per class
        sc_vec (1xm): Vector of classes to which each species belong
        mc_vec (1xm): Vector of classes to which each resource belong
    '''
    #Number of species
    s = len(sc_vec)
    #Preallocate matrix
    C = np.zeros(shape = (s, m))
    #Initialize vector of demands 
    d = np.ones(m)
    for alpha in range(s):
        #Sampling probability of each metabolite
        prob = gen_prob_vec(Kc, kc, d, m, nsp, sc_vec[alpha], mc_vec)
        #Sample number of consumed metabolites
        n = 0
        while n < 1 or n > m/3:
            n = round(np.random.exponential(beta))
        #Sample indices of the matrix
        ind = np.random.choice(range(m), n, replace = False, p = prob)
        C[alpha, ind] = 1
        #Calculate demands at iteration alpha
        d = np.sum(C, axis = 0)
    return C

def general_metabolic_matrix(s, kf, d, Kf, M, m, nsp):
    '''
    Generage vector of concentration parameters of a Dirichlet distribution

    Parameters: 
        kf (float): Facilitation constant
        s  (float): Sparsity parameter
        nsp (int): Number of individuals in each class
        M (mxm array): Matrix of classes
    '''
    n_it = len(M)
    #Preallocate metabolic matrix
    D = np.zeros(shape = (n_it, n_it))
    #Sample rows of D according to the concentration vector
    for i in range(n_it):
        #Sample concentration vector 
        c_vec = general_concentration_vector(kf, s, M, d, Kf, m, i, nsp)
        D[i,:] = np.random.dirichlet(c_vec)
    return D

def general_concentration_vector(kf, spar, M, d, Kf, m, alpha, nsp):
    '''
    Create concentration vector for each row (substrate)
    '''
    #Preallocate vector
    q = np.zeros(m) 
    for prod in range(m):
        #Assign value to each element of the vector
        if M[alpha, prod]: 
            #When product and substrate belong to the same class assign a 
            #smaller value
            q[prod] = (1 + kf*d[prod])*(1 - Kf)/(spar*nsp)
        else:
            #When product and substrate belong to a different class assign a
            #smaller value
            q[prod] = (1 + kf*d[prod])*(1 + Kf)/(spar*nsp)
        #Note that the values of d match independently of  the class of 
        #product and substrate when kf = 0.
    return q
