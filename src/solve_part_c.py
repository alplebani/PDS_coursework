#!/usr/bin/env python
import scipy
from math import log
from scipy.integrate import simpson
from scipy.optimize import minimize
from scipy.stats import norm, expon
from functools import partial
from scipy.integrate import quad
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import cumulative_trapezoid
from scipy.stats import truncnorm, truncexpon
import argparse

plt.style.use('../mphil.mplstyle')


# ===============================================
# Helper functions for the model distributions
# ===============================================

class Model ():
    '''
    Statistical model. Contains all functions needed to run the code
    '''
    def __init__(self, M, f, lamda, sigma, mu, alpha, beta, is_normalised):
        self.M = M
        self.f = f
        self.lamda = lamda
        self.sigma = sigma
        self.mu = mu
        if alpha >= beta:
            print('Error! alpha must be greater than beta')
            print('Exiting code')
            exit(1)
        else:
            self.alpha = alpha
            self.beta = beta
        self.is_normalised = is_normalised
    
    def background(self):
        '''
        Distribution of the background model
        '''
        
        return expon.pdf(self.M, scale=1./self.lamda)


    def signal(self):
        '''
        Distribution of the signal model
        '''
        
        return norm.pdf(self.M, loc=self.mu, scale=self.mu)
    
    
    def norm_factors(self):
        '''
        Function that returns the normalisation factors for signal and background pdfs
        '''

        Norm_bkg = np.trapz(self.background(), self.M) # normalisation factor for the signal
        Norm_sig = np.trapz(self.signal(), self.M) # normalisation factor for the background
        
        return Norm_sig, Norm_bkg


    def pdf(self):
        '''
        Function that returns the probability distribution function, with flag to have it normalised or not
        '''
        
        N_sig, N_bkg = self.norm_factors()

        if self.is_normalised:
            return (self.f * self.signal() / N_sig) + ((1 - self.f) * self.background() / N_bkg)
        else:
            return self.f * self.signal() + (1 - self.f) * self.background()
        
    def cdf(self):
        '''
        Function that returns the cumulative distribution function for the pdf
        '''
        
        return cumulative_trapezoid(self.pdf(), self.M, initial=self.alpha) # integral of the cdf from alpha 
        
    
        





    


def main():
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-a', '--alpha', help='Value of lower limit of distribution', type=float, default=5., required=False)
    parser.add_argument('-b', '--beta', help='Value of the upper limit of the distribution', type=float, default=5.6, required=False)
    parser.add_argument('-n', '--nentries', help='Number of models to be tested', type=int, required=False, default=1000)
    args = parser.parse_args()
    
    if args.alpha >= args.beta:
        print('Error! alpha must be smaller than beta. Exiting the code!')
        exit(4)
    
    print("Executing exercise c)")
    print("=======================================")
    
    print("Testing to see if pdf is normalised")
    print("=======================================")
    
    n_entries = args.nentries
    
    print('Now testing {} models'.format(n_entries))
    print("=======================================")
    
    my_alpha = args.alpha
    my_beta = args.beta
        
    x = np.linspace(my_alpha, my_beta, n_entries) # select n_entries points in the selected range  
    
    np.random.seed(4999) # random seed to have always same results, chosen as my birthday 
    
    f_values = np.random.random(n_entries)
    lamda_values = np.random.uniform(0, 1, n_entries)
    mu_values = np.random.uniform(1, 10, n_entries)
    sigma_values = np.random.uniform(0.1, 1.1, n_entries)
    
    my_model = []
    
    for i in range(n_entries):
        my_model.append(Model(M=x, f=f_values[i], lamda=lamda_values[i], mu=mu_values[i], sigma=sigma_values[i], alpha=my_alpha, beta=my_beta, is_normalised=True))
    
    
    
    integral = []
    
    if n_entries < 50: # For more than 50 entries the plot is too messy and it takes a long time to open
        plt.figure(figsize=(15,10))
        print('Showing plot with all the distributions')
        print("=======================================")
    
    for i in range(n_entries):
        integral.append(np.trapz(my_model[i].pdf(), x)) 
        
        if n_entries < 50: 
            plt.plot(x, my_model[i].pdf())            
            plt.xlabel('M')
            plt.ylabel('pdf(M)')
        
    print('Mean of {0} different models: {1} +- {2}'.format(n_entries, np.mean(integral), np.var(integral)))
    
    if n_entries < 50:
        plt.title('{0} different models for the pdf. Mean = {1}'.format(n_entries, np.mean(integral)))
        plt.savefig('../plots/Part_c_{0}_{1}_{2}_entries.pdf'.format(my_alpha, my_beta, n_entries))
        print("=======================================")
        plt.show()
        print('Saving pdf file at ../plots/Part_c_{0}_{1}_{2}_entries.pdf'.format(my_alpha, my_beta, n_entries))


if __name__ == "__main__":
    print("=======================================")
    print("Initialising coursework")
    print("=======================================")
    main()
    print("=======================================")
    print("Code finished. Exiting!")
    print("=======================================")
