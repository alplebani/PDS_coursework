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
    
    # def norm_sig(self):
    #     '''
    #     Normalisation factor for the signal
    #     '''
        
    #     return 1./(norm.cdf(self.beta, loc=self.mu, scale=self.sigma) - norm.cdf(self.alpha, loc=self.mu, scale=self.sigma))
    
    # def norm_bkg(self):
    #     '''
    #     Normalisation factor for the backgrouund
    #     '''
        
    #     return 1./(expon.cdf(self.beta, scale=1./self.lamda) - expon.cdf(self.alpha, scale=1./self.lamda))
    
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
    
    print("Executing exercise c)")
    print("=======================================")
    
    print("Testing to see if pdf is normalised")
    print("=======================================")
    
    n_entries = int(input("Select number of points to generate: "))
    
    my_alpha = 0
    my_beta = 10
        
    x = np.linspace(my_alpha, my_beta, n_entries) # select n_entries points in the selected range   
    
    f_values = np.random.random(n_entries)
    lamda_values = np.random.uniform(0, 1, n_entries)
    mu_values = np.random.uniform(5, 5.1, n_entries)
    sigma_values = np.random.uniform(0.1, 1.1, n_entries)
    
    my_model = []
    
    for i in range(n_entries):
        my_model.append(Model(M=x, f=f_values[i], lamda=lamda_values[i], mu=mu_values[i], sigma=sigma_values[i], alpha=my_alpha, beta=my_beta, is_normalised=True))
    
    # plt.figure()
    # for i in range(n_entries):
    #     plt.plot(x, my_model[i].pdf())
    
    # plt.show()
    
    integral = []
    
    
    
    for i in range(n_entries):
        integral.append(np.trapz(my_model[i].pdf(), x)) 
        
    print(np.mean(integral))
    print(np.var(integral))


if __name__ == "__main__":
    print("=======================================")
    print("Initialising coursework")
    print("=======================================")
    main()
