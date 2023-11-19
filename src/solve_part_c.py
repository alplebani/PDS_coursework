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


# ===============================================
# Helper functions for the model distributions
# ===============================================

class model ():
    '''
    Statistical model. Contains all functions needed to run the code
    '''
    def __init__(self, M, f, lamda, sigma, mu, alpha, beta):
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

    def norm_factor(self):
        '''
        Function that returns the normalisation factor to be applied to the PDF in order to have unitary probability
        '''
        
        N_bkg = (1 - self.f) * (expon.cdf(self.beta, scale=1./self.lamda) - expon.cdf(self.alpha, scale=1./self.lamda)) # Background norm factor
        N_sig = self.f * (norm.cdf(self.beta, loc=self.mu, scale=self.sigma) - norm.cdf(self.alpha, loc=self.mu, scale=self.sigma)) # Signal norm factor
        
        return 1./(N_bkg + N_sig)

    def pdf(self, is_normalised):
        '''
        Function that returns the probability distribution function 
        '''

        if is_normalised:
            return self.norm_factor() * (self.f * self.signal() + (1 - self.f) * self.background())
        else: 
            return self.f * self.signal() + (1 - self.f) * self.background()



    


def main():
    print("Hello World!")

if __name__ == "__main__":
    main()
