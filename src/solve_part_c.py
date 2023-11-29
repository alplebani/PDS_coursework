#!/usr/bin/env python
import scipy
from math import log
from scipy.integrate import simpson
from scipy.special import erf
from scipy.optimize import minimize
from scipy.stats import norm, expon
from functools import partial
from scipy.integrate import quad
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import cumulative_trapezoid
from scipy.stats import truncnorm, truncexpon
import argparse
from scipy.stats.sampling import NumericalInversePolynomial as NIP
import itertools
from iminuit.cost import BinnedNLL, UnbinnedNLL
from iminuit import Minuit
import time

plt.style.use('mphil.mplstyle')


# ===============================================
# Helper functions for the model distributions
# ===============================================

class Model ():
    '''
    Statistical model. Contains all functions needed to run the code
    '''
    def __init__(self, f, lamda, sigma, mu, alpha, beta, is_normalised):
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
    
    def background(self, x):
        '''
        Distribution of the background model
        '''
        
        return expon.pdf(x, scale=1./self.lamda)


    def signal(self, x):
        '''
        Distribution of the signal model
        '''
        
        return norm.pdf(x, loc=self.mu, scale=self.sigma)
    
    def N_S(self, x):
        '''
        Normalisation factor of the signal
        '''
        
        if self.f != 0:
            return np.trapz(self.signal(x), x) # to avoid scenarios where we don't have a signal distribution
        else:
            return 1
    
    def N_B(self, x):
        '''
        Normalisation factor of the background
        '''
        
        return np.trapz(self.background(x), x)
    
    def norm_factors(self, x):
        '''
        Function that returns the normalisation factors for signal and background pdfs
        '''
        
        return self.N_S(x), self.N_B(x)
    
    def pdf_signal(self, x):
        '''
        Normalised distribution of the signal
        '''
        
        return self.signal(x)/self.N_S(x)
    
    def pdf_background(self, x):
        '''
        Normalised distribution of the background
        '''
        
        return (1 - self.f) * self.background(x)/self.N_B(x)

    def pdf(self, x):
        '''
        Function that returns the probability distribution function
        '''

        if self.is_normalised:
            return (self.f * self.signal(x) / self.N_S(x)) + ((1 - self.f) * self.background(x) / self.N_B(x))
        else:
            return self.f * self.signal(x) + (1 - self.f) * self.background(x)
        
    def cdf(self, x):
        '''
        Function that returns the cumulative distribution function
        '''
        
        bkg = - np.exp(- self.lamda * self.alpha) + np.exp(- self.lamda * x)
        sig = 0.5 * (erf((x - self.mu)/(np.sqrt(2) * self.sigma)) - erf((self.alpha - self.mu)/(np.sqrt(2) * self.sigma)))
        
        return self.f * sig / self.N_S(x) + (1 - self.f) * bkg / self.N_B(x)
    
    def accept_reject(self, size):
        '''
        Function used to generate data according to the pdf using the accept/reject method
        '''
        
        a = np.random.uniform(0, 1, size)
        x = np.linspace(self.alpha, self.beta, size)
        pdf = self.pdf(x)
        return x[np.where(a < pdf / np.max(pdf))[0][:size]]


def main():
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-a', '--alpha', help='Value of lower limit of distribution', type=float, default=5., required=False)
    parser.add_argument('-b', '--beta', help='Value of the upper limit of the distribution', type=float, default=5.6, required=False)
    parser.add_argument('-n', '--nentries', help='Number of models to be tested', type=int, required=False, default=1000)
    parser.add_argument('-p', '--points', help="Number of points you  want to generate", type=int, required=False, default=100000)
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
    lamda_values = np.random.uniform(0.1, 1, n_entries)
    mu_values = np.random.uniform(my_alpha, my_beta, n_entries)
    sigma_values = np.random.uniform(0.01, 1, n_entries)
    
    my_model = []
    
    for i in range(n_entries):
        my_model.append(Model(f=f_values[i], lamda=lamda_values[i], mu=mu_values[i], sigma=sigma_values[i], alpha=my_alpha, beta=my_beta, is_normalised=True))
    
    integral = []
    
    if n_entries < 50: # For more than 50 entries the plot is too messy and it takes a long time to open
        plt.figure(figsize=(15,10))
        print('Showing plot with all the distributions')
        print("=======================================")
    
    for i in range(n_entries):
        integral.append(np.trapz(my_model[i].pdf(x), x)) 
        
        if n_entries < 50: 
            plt.plot(x, my_model[i].pdf(x))            
            plt.xlabel('M')
            plt.ylabel('pdf(M)')
        
    print('Mean of {0} different models: {1} +- {2}'.format(n_entries, np.mean(integral), np.var(integral)))
    
    if n_entries < 50:
        plt.title('{0} different models for the pdf. Mean = {1}'.format(n_entries, np.mean(integral)))
        plt.savefig('plots/Part_c_{0}_{1}_{2}_entries.pdf'.format(my_alpha, my_beta, n_entries))
        print("=======================================")
        plt.show()
        print('Saving pdf file at plots/Part_c_{0}_{1}_{2}_entries.pdf'.format(my_alpha, my_beta, n_entries))

    print("=======================================")
    print('Part c finished, moving on to part d now...')
    print("Executing exercise d)")
    print("=======================================")
    
    print("Plotting true distributions")
    print("=======================================")
    
    x = np.linspace(5., 5.6, 1000)
    true_model = Model(f=0, lamda=0.5, mu=5.28, sigma=0.018, alpha=5, beta=5.6, is_normalised=True)
    
    plt.figure(figsize=(15,10))
    plt.plot(x, true_model.signal(x), label='Signal', c='r', ls='--')
    plt.plot(x, true_model.background(x), label='Background', c='b', ls='-.')
    plt.plot(x, true_model.pdf(x), label='Signal+background', color='green')
    plt.xlabel('M')
    plt.ylabel('PDF(M)')
    plt.title('True PDF')
    plt.legend()
    plt.savefig('plots/true_pdf.pdf')
    print("=======================================")
    print('Saving pdf file at plots/true_pdf.pdf')
    plt.show()
    
    exit(0)
    
    print('Part d finished, moving on to part e now...')
    print("Executing exercise e)")
    print("=======================================")
    print('Generating sample')
    
    entries = args.points

    x = np.linspace(my_alpha, my_beta, entries)
    
    true_model = Model(f=0.1, lamda=0.5, mu=5.28, sigma=0.018, alpha=5, beta=5.6, is_normalised=True)
    
    data = true_model.accept_reject(size=entries)
    
    plt.figure(figsize=(15,10))
    plt.hist(data, bins=100,range=(args.alpha, args.beta), density=True, label='Data')
    plt.plot(x, true_model.pdf(x), label='True pdf')
    plt.legend()
    plt.xlabel('M')
    plt.ylabel('PDF(M)')
    plt.title('High-statistics sample with {} events'.format(entries))
    plt.savefig('plots/part_e.pdf')
    print("=======================================")
    print('Saving pdf file at plots/part_e.pdf')
    # plt.show()    
    
    def pdf(x, f, mu, lamda, sigma):
        '''
        Function that returns the pdf of the model
        '''
        
        model = Model(f=f, mu=mu, lamda=lamda, sigma=sigma, alpha=5., beta=5.6, is_normalised=True)
        
        return model.pdf(x)
    
    nLL = UnbinnedNLL(data, pdf)
    mi = Minuit(nLL, mu=5.28, f=0.1, lamda=0.5, sigma=0.018)
    mi.migrad()
    print(mi)
    
    hat_f, hat_mu, hat_lamda, hat_sigma = mi.values
    
    plt.figure(figsize=(15,10))
    plt.hist(data, bins=100, density=True)
    plt.plot(x, true_model.pdf(x), label='True model')
    fit_model = Model(f=hat_f, mu=hat_mu, sigma=hat_sigma, lamda=hat_lamda, alpha=my_alpha, beta=my_beta, is_normalised=True)
    plt.plot(x, fit_model.pdf(x), label='Fit model', color='green')
    plt.plot(x, fit_model.signal(x)/5., label='Fit signal / 5.', c='r', ls='--')
    plt.plot(x, fit_model.background(x), label='Fit background', c='b', ls='-.')
    plt.title('Post-fit distribution')
    plt.xlabel('M')
    plt.ylabel('PDF(M)')
    plt.legend()
    plt.savefig("plots/fit_e.pdf")
    print("=======================================")
    print('Saving pdf file at plots/fit_e.pdf')
    # plt.show()
    
    print("=======================================")
    print('Part e finished, moving on to part f now...')
    print("Executing exercise f)")
    print("=======================================")
    
    
    nLL = UnbinnedNLL(data, pdf)
    mi = Minuit(nLL, mu=5.28, f=0, lamda=0.5, sigma=0.018)
    mi.values['f'] = 0
    mi.fixed['f'] = True
    
    mi.migrad()
    mi.hesse()
    print(*mi.values)
    hat_f, hat_mu, hat_lamda, hat_sigma = mi.values
    plt.figure()
    plt.hist(data, bins=100, density=True)
    plt.plot(x, true_model.pdf(x), label='True model')
    fit_model = Model(f=hat_f, mu=hat_mu, sigma=hat_sigma, lamda=hat_lamda, alpha=my_alpha, beta=my_beta, is_normalised=True)
    plt.plot(x, fit_model.pdf(x), label='Fit model', color='green')
    plt.plot(x, fit_model.background(x), label='Fit background', c='b', ls='-.')
    plt.legend()
    plt.show()
    
    

if __name__ == "__main__":
    print("=======================================")
    print("Initialising coursework")
    print("=======================================")
    main()
    print("=======================================")
    print("Code finished. Exiting!")
    print("=======================================")
