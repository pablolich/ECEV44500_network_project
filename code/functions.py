#!/usr/bin/env python3

__appname__ = '[functions.py]'
__author__ = 'Pablo Lechon (plechon@ucm.es)'
__version__ = '0.0.1'

## IMPORTS ##

import sys
import numpy as np
from model import equations
from scipy.integrate import solve_ivp

## CONSTANTS ##


## FUNCTIONS ##
def pairwise_competition(l, c_a, c_b):
    return((1 - l) * c_a @ c_b.T)

def pairwise_facilitation(D, l, c_a, c_b):
    return(l * c_a @ D @ c_b.T)

def interaction_matrix(l, c, D, interaction = 'both'):
    '''
    Calculate level of competition/facilitation/cohesion between all species 
    pairs
    '''
    #Get number of species and metabolites
    s = np.shape(c)[0]
    m = np.shape(c)[1]
    #Since the interaction matrix is symmetric, get the indices of the non-zero
    #elements (those on the upper triangle with offset of 1 over the main 
    #diagonal)
    ind_u = np.triu_indices(s, k = 1)
    ind_l = np.tril_indices(s, k = -1)
    ind = np.hstack([ind_u, ind_l])
    #Prealocate vector of interaction
    interaction_vec = np.zeros(len(ind[0]))
    #Calculate interaction indices
    for i in range(len(ind[0])):
        #Get species indices
        ith = ind[0][i]
        jth = ind[1][i]
        #Find preferences of each species
        c_ith = c[ith,:]
        c_jth = c[jth,:]
        #Find shared preferences and normalize
        common = len(np.where(c_ith == c_jth)[0])/len(c_ith)
        if interaction == 'both':
            #Compute and store facilitation level
            interaction_vec[i] = pairwise_competition(l, c_ith, c_jth) - \
                                 pairwise_facilitation(D, l, c_ith, c_jth)
        elif interaction == 'competition':
            interaction_vec[i] = pairwise_competition(l, c_ith, c_jth) 
        elif interaction == 'facilitation':
            interaction_vec[i] = pairwise_facilitation(D, l, c_ith, c_jth)
        else:
            print('Unknown type of interaction')
            raise
    #Create cohesion matrix
    interaction_matrix = np.zeros(shape = (s, s))
    #Reshape ind to correctly index a matrix
    ind = tuple([ind[0], ind[1]])
    #Populate cohesion matrix
    interaction_matrix[ind] = interaction_vec
    return interaction_matrix

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

def competition_test(l, k1, k2):
    '''
    Test wether communities with higher kc have higher levels of total
    competition
    '''

    #Set parameters
    m = s = 50
    n_com = 100 
    n_k1 = len(k1)
    n_k2 = len(k2)
    #Prealocate vector of community-level competition
    C_ten = np.zeros(shape = (n_k1, n_com, n_k2))
    F_ten = np.zeros(shape = (n_k1, n_com, n_k2))
    #Generate preferences matrix with varying kc
    for k_1 in range(n_k1):
        for k_2 in range(n_k2):
            for sim in  range(n_com):
                #Sample a metabolic matrix
                c = preference_matrix(m, s, kc = k1[k_1],  beta = 5, n_r = None)
                C = interaction_matrix(l, c, D = 2, interaction = 'competition') 
                C_ten[k_1, sim, k_2] = np.mean(C[np.triu_indices(len(C))])
                #Calculate facilitation
                cum_demand = np.sum(c, 0)
                D = crossfeeding_matrix(cum_demand, m, k2[k_2])
                F = interaction_matrix(l, c, D = D, interaction = 'facilitation') 
                F_ten[k_1, sim,  k_2] = np.mean(F[np.triu_indices(len(F))])
    return(C_ten, F_ten)

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

def extinctions(v, v1, v2):
    '''
    Calculate extinctions in each communty based on the presence absence 
    vectors before and after community coalescence

    Parameters:
        v (1:s1+s2): Vector of absence-precence (1-0) values for species in the
                     resulting community
        v1 (1:s1+s2): Vector of absence-precnce values for species in 
                      community 1 with s2 zeros at the end
        v2 (1:s2+s1): Vector of absence-precnce values for species in 
                      community 2 with s1 preceeding zeros
    '''
    #Number of surviving species of each community in the final mix
    surv1 = np.dot(v1, v)
    surv2 = np.dot(v2, v)
    #Number of extinct species of each community in the final mix normalized
    #by the total number of species in  the original community
    ext1 = 1 - surv1/np.sum(v1)
    ext2 = 1 - surv2/np.sum(v2) 
    return np.array([ext1, ext2])

def cohesion_looser(bool_12, r1, D1, c1, l1, r2, D2, c2, l2, looser):
    if looser == 1:
        #Select presence/absence vector form loosing community (1)
        bool_after = bool_12[0:r1]
        #Get indices of extinct species
        ext_ind = np.where(bool_after == 0)[0]
        #Find the preference matrix of extinct species
        c_ext = c1[ext_ind, :]
        #Calculate facilitation of this group of species
        F_ext = community_facilitation(c_ext, c_ext,  D1, l1, l1)[1]
        #Calculate competition of this group of species
        C_ext = community_competition(c_ext, c_ext, l1, l1)[1]
    elif looser == -1:
        #Select presence/absence vector form loosing community (1)
        bool_after = bool_12[r1:]
        #Get indices of extinct species
        ext_ind = np.where(bool_after == 0)[0]
        #Find the preference matrix of extinct species
        c_ext = c2[ext_ind, :]
        #Calculate facilitation of this group of species
        F_ext = community_facilitation(c_ext, c_ext,  D2, l2, l2)[1]
        #Calculate competition of this group of species
        C_ext = community_competition(c_ext, c_ext, l2, l1)[1]
    else:
        F_ext = 0
        C_ext = 0
    return np.array([F_ext, C_ext])

    

def potential_harvest(l, c, demands):
    '''
    Calculate the average potential harvest of community members
    '''
    ##Get indices where there is a 0 in the demand
    #ind_0 = np.where(demands == 0)[0]
    ##Set to 1 to avoid nans
    #demands[ind_0] = 1
    return(np.mean((1-l)*np.sum(c, axis = 1)))

def facilitation_in(l, c1, c2, D, demands, inter = False):
    '''
    Calculate facilitation to species i from all the rest
    '''
    #Get number of metabolites
    m = D.shape[0]
    #Get indices where there is a 0 in the demand
    ind_0 = np.where(demands == 0)[0]
    #Set to 1 to avoid nans
    demands[ind_0] = 1
    if inter:
        #Get number of species in community 1
        s1 = c1.shape[0]
        s2 = c2.shape[0]
        #Initialize vector of facilitation
        facilitation = np.zeros(s1) 
        for sp in range(s1):
            #Calculate facilitation from all species in community 2 to each of
            #the species in community 1
            facilitation[sp] = l*np.ones(s2) @ c2 @ \
                               (D @ (c1[sp,:]/demands).reshape(m, 1))
    else: 
        s = c1.shape[0]
        #If only one species survives, the facilitation is 0
        if s == 1:
            return(0)
        #Initialize vector of facilitation
        facilitation = np.zeros(s) 
        for sp in range(s):
            #Get rid of focal strain 
            c_other = np.delete(c1, sp, axis = 0)
            facilitation[sp] = l*np.ones(s-1) @ c_other @ \
                               (D @ (c1[sp,:]/demands).reshape(m, 1))
    return(np.mean(facilitation))

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

#def inter_comm_abiotic_competition(c_mats, D_mats, l_vecs, ,


#def inter_comm_facilitation(comms):
#    '''
#    Compute the inter-community facilitation matrix
#    '''
#    n_comm = len(comms)
#    F = np.zeros( shape = (n_comm, n_comm) )
#    for i in range(n_comm):
#        for j in range(n_comm):
#            F[i,j] = community_facilitation(comms[i].c, comms[j].c, 
#                                            comms[i].D, comms[i].l)
#    return np.mean(F)

class Community:
    def __init__(self, kf, kc, r, c, n, x, l, D, n_coal):
        self.kf = kf
        self.kc = kc
        self.r = r 
        self.c = c
        self.n = n
        self.x = x
        self.l = l
        self.D = D
        self.s = np.shape(c)[0]
        self.m = np.shape(c)[1]
        self.coal = n_coal
        self.source = np.repeat(n_coal, r)
        self.path = np.mean(np.sum(self.c, axis = 1))
    
    def coalescence(self, comm, resources):
        '''Perform coalescence between 2 communities'''
     
        #Create joint system
        #Remember to take away the extinct species
        ext_system = joint_system(self.c, self.D, self.n, self.l, self.x, 
                                  comm.c, comm.D, comm.n, comm.l, comm.x)
        
        #Create the vector of supplied resources
        kappa = np.zeros(self.m)
        kappa[resources] = 1
        #Create dictionary of parameters
        params = {'g':np.ones(self.r + comm.r).reshape(self.r + comm.r, 1),
                  's':self.r + comm.r,
                  'm':self.m,
                  'K':20*kappa.reshape(self.m, 1),
                  't':0.5*np.ones(self.m).reshape(self.m, 1),
                  'coal':self.coal,#This is a coalescence event
                  'D':ext_system['D'],
                  'c':ext_system['C'],
                  'x':ext_system['x'],
                  'l':ext_system['l']}
        #Initial conditions
        z0 = list(ext_system['N']) + list(2*np.ones(self.m)) 
        #Create time vector
        tspan = tuple([1, 1e4])
        #Integrate diferential equations
        try:
            sol = solve_ivp(lambda t,z: equations(t,z, params),
                            tspan, z0,
                            method = 'BDF', atol = 0.0001 )
        except:
            #The solution diverges
            sol = None  
        return(params, sol)

def extinctions_one_two(abundances, s1):
    '''
    Calculates the remaining species in each community after coalescence
    '''
    abundances1 = abundances[0:s1]
    present1 = len(np.where(abundances1 > 1)[0])
    abundances2 = abundances[s1:len(abundances)]
    present2 = len(np.where(abundances2 > 1)[0])
    return np.array([present1, present2])

def remove_chunk(extint_ind, D, m):
    '''
    Remove a matrix inside a bigger matrix
    '''
    #Create matrix of bools of the size of Dtot
    bool_mat = np.ones(shape = np.shape(D), dtype = bool)
    shape = np.array(np.shape(D))
    #Create crosses of False to be removed from D
    for i in extint_ind:
        #Create chunks of False on the rows
        bool_mat[i*m:(i+1)*m, :] = False
        #Create chunks of False on the columns
        bool_mat[:, i*m:(i+1)*m] = False
    #Subtract the chunks from D
    D = D[bool_mat]
    #Reshape D to be square again
    D = D.reshape(tuple(shape - len(extint_ind)*m))
    return D

def remove_square(k, s, A):
    '''
    Remove the kth square matrix of size s inside matrix A
    '''
    return(A[s*k:s*(k+1), s*k:s*(k+1)])

def remove_all(A, s):
    '''
    Collect all square matrices a of size m from matrices  separatelly from 
    the bigger matrix A
    '''
    #Get number of matrices to collect
    n_mat = np.shape(A)[0]//s
    #Preallocate tensor to store all  D matrices
    all_a = np.zeros(shape = (n_mat, s, s))
    for i in range(n_mat):
        all_a[i,:,:] = remove_square(i, s, A)
    return all_a


def remove_stripe(extinct, c, m):
    '''
    Remove a vertical stripe along a matrix
    '''
    #Create a boolean matrix to mask c
    shape = np.shape(c)
    bool_mat = np.ones(shape, dtype = bool)
    #Make a vertical stripe of Falses in the place of extinct
    #communities
    for i in extinct:
        bool_mat[:,(i)*m:(i+1)*m] = False
    cm = c[bool_mat]
    cm = cm.reshape(shape[0], shape[1] - m*len(extinct))
    return cm

def full_c(mix_comm):
    '''
    Create the mixed  matrix with elements in the off diagonals. Specifically,
    each row is a copy of the matrix in the box-diagonal
    '''
    #Get vector of distinct communities and how many species each community has 
    values, counts = np.unique(mix_comm.source, return_counts = True)
    #Get cummulative sum of cunts, add 0 at the begginning
    cum_counts = np.hstack([0, np.cumsum(counts)])
    #Initialize matrix
    c_mix = np.zeros(np.shape(mix_comm.c))
    m = mix_comm.m
    #Loop over each communtiy
    for i in range(len(values)):
        #Repeat c_mat of the ith community, as many times as different
        #communities are
        c_chunk = mix_comm.c[cum_counts[i]:cum_counts[i+1], i*m:(i+1)*m]
        c_mix[cum_counts[i]:cum_counts[i+1],:] = np.tile(c_chunk, len(values))
    return c_mix

def generalized_odometer(n_vec, v_vec):
    '''
    Returns the index of the generalized odometer corresponding to the values
    of each of its digits
    '''
    prod_vec = [np.prod(n_vec[0:i-1]) for i in range(len(n_vec))]
    return prod_vec 

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

def metabolic_matrix(kf, s, Mc, M, m):
    '''
    Generage vector of concentration parameters of a Dirichlet distribution

    Parameters: 
        kf (float): Facilitation constant
        s  (float): Sparsity parameter
        Mc (int): Number of individuals in each class
        M (mxm array): Matrix of classes
        m (int): Number of metabolites
    '''
    n_it = len(M)
    #Preallocate metabolic matrix
    D = np.zeros(shape = (n_it, n_it))
    #Sample rows of D according to the concentration vector
    for i in range(n_it):
        #Sample concentration vector 
        c_vec = concentration_vector(kf, s, M, i, m, Mc)
        D[i,:] = np.random.dirichlet(c_vec)
    return D

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

def gen_pref_matrix(Kc, kc, m, nsp, sc_vec, mc_vec, beta = 5):
    '''
    Samplekpreference matrix
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
