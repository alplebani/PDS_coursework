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
from scipy.stats import truncnorm, truncexpon, chi2
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

class New_Model ():
    '''
    Statistical model. Contains all functions needed to run the code
    '''
    def __init__(self, f1, f2, lamda, sigma, mu_1, mu_2, alpha=5., beta=5.6, is_normalised=True):
        self.f1 = f1
        self.f2 = f2
        self.lamda = lamda
        self.sigma = sigma
        self.mu_1 = mu_1
        self.mu_2 = mu_2
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


    def signal_1(self, x):
        '''
        Distribution of the signal model s1
        '''
        
        return norm.pdf(x, loc=self.mu_1, scale=self.sigma)
    
    def signal_2(self, x):
        '''
        Distribution of the signal model s2
        '''
        
        return norm.pdf(x, loc=self.mu_2, scale=self.sigma)
    
    def N_S_1(self):
        '''
        Normalisation factor of the signal s1
        '''
        
        return 0.5 * (erf((self.beta - self.mu_1)/(np.sqrt(2) * self.sigma)) - erf((self.alpha - self.mu_1)/(np.sqrt(2) * self.sigma)))
    
    def N_S_2(self):
        '''
        Normalisation factor of the signal s1
        '''
        
        return 0.5 * (erf((self.beta - self.mu_2)/(np.sqrt(2) * self.sigma)) - erf((self.alpha - self.mu_2)/(np.sqrt(2) * self.sigma)))


    
    def N_B(self):
        '''
        Normalisation factor of the background
        '''
        
        return np.exp(- self.lamda * self.alpha) - np.exp(-self.lamda * self.beta)
    
    def pdf_signal_1(self, x):
        '''
        Normalised distribution of the signal
        '''
        
        return self.signal_1(x)/self.N_S_1()
 
    def pdf_signal_2(self, x):
        '''
        Normalised distribution of the signal
        '''
        
        return self.signal_2(x)/self.N_S_2() 
    
    def pdf_background(self, x):
        '''
        Normalised distribution of the background
        '''
        
        return self.background(x)/self.N_B()

    def pdf(self, x):
        '''
        Function that returns the probability distribution function
        '''

        if self.is_normalised:
            return (self.f1 * self.pdf_signal_1(x) + self.f2 * self.pdf_signal_2(x) +  (1 - self.f1 - self.f2) * self.pdf_background(x))
        else:
            return self.f1 * self.signal_1(x) + self.f2 * self.signal_2(x) + (1 - self.f1 - self.f2) * self.background(x)
        
    def cdf(self, x):
        '''
        Function that returns the cumulative distribution function
        '''
        
        bkg = - np.exp(- self.lamda * self.alpha) + np.exp(- self.lamda * x)
        sig_1 = 0.5 * (erf((x - self.mu_1)/(np.sqrt(2) * self.sigma)) - erf((self.alpha - self.mu_1)/(np.sqrt(2) * self.sigma)))
        sig_2 = 0.5 * (erf((x - self.mu_2)/(np.sqrt(2) * self.sigma)) - erf((self.alpha - self.mu_2)/(np.sqrt(2) * self.sigma)))
        
        return self.f1 * sig_1 / self.N_S_1(x) + self.f2 * sig_2 / self.N_S_2(x) +  (1 - self.f1 - self.f2) * bkg / self.N_B(x)
    
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
    parser.add_argument('-f', '--fit', help='Flag whether you want to re-do the fits in part f) or if you just want to load the data', type=bool, required=False, default=False)
    args = parser.parse_args()
    
    if args.alpha >= args.beta:
        print('Error! alpha must be smaller than beta. Exiting the code!')
        exit(4)
    
    print("Executing exercise g)")
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
    
    f1_values = np.random.random(n_entries)
    f2_values = np.random.random(n_entries)
    lamda_values = np.random.uniform(0.1, 1, n_entries)
    mu1_values = np.random.uniform(my_alpha, my_beta, n_entries)
    mu2_values = np.random.uniform(my_alpha, my_beta, n_entries)
    sigma_values = np.random.uniform(0.01, 1, n_entries)
    
    my_model = []
    
    for i in range(n_entries):
        my_model.append(New_Model(f1=f1_values[i], f2=f2_values[i], lamda=lamda_values[i], mu_1=mu1_values[i], mu_2=mu2_values[i], sigma=sigma_values[i], alpha=my_alpha, beta=my_beta, is_normalised=True))
    
    integral = []
    
    if n_entries < 50: # For more than 50 entries the plot is too messy and it takes a long time to open
        plt.figure(figsize=(15,10))
        print('Showing plot with all the distributions')
        print("=======================================")
    
    for i in range(n_entries):
        integral.append(np.trapz(my_model[i].pdf_background(x), x)) 
        
        if n_entries < 50: 
            plt.plot(x, my_model[i].pdf(x))            
            plt.xlabel('M')
            plt.ylabel('pdf(M)')
        
    print('Mean of {0} different models: {1} +- {2}'.format(n_entries, np.mean(integral), np.var(integral)))
    
    if n_entries < 50:
        plt.title('{0} different models for the pdf. Mean = {1}'.format(n_entries, np.mean(integral)))
        plt.savefig('plots/Part_g_{0}_{1}_{2}_entries.pdf'.format(my_alpha, my_beta, n_entries))
        print("=======================================")
        plt.show()
        print('Saving pdf file at plots/Part_g_{0}_{1}_{2}_entries.pdf'.format(my_alpha, my_beta, n_entries))

    print("=======================================")
    print('Part c finished, moving on to part d now...')
    print("Executing exercise d)")
    print("=======================================")
    
    print("Plotting true distributions")
    print("=======================================")
    
    x = np.linspace(5., 5.6, 1000)
    true_model = New_Model(f1=0.1, f2=0.05, lamda=0.5, mu_1=5.28, mu_2=5.35, sigma=0.018, alpha=5, beta=5.6, is_normalised=True)
    s1_model = New_Model(f1=1, f2=0, lamda=0.5, mu_1=5.28, mu_2=5.35, sigma=0.018, alpha=5, beta=5.6, is_normalised=True)
    s2_model = New_Model(f1=0, f2=1, lamda=0.5, mu_1=5.28, mu_2=5.35, sigma=0.018, alpha=5, beta=5.6, is_normalised=True)
    bkg_model = New_Model(f1=0, f2=0, lamda=0.5, mu_1=5.28, mu_2=5.35, sigma=0.018, alpha=5, beta=5.6, is_normalised=True)
    
    

    
    plt.figure(figsize=(15,10))
    plt.plot(x, s1_model.pdf(x)/10., label='S1', c='r', ls='--')
    plt.plot(x, s2_model.pdf(x)/20., label='S2', c='orange', ls='--')
    plt.plot(x, bkg_model.pdf(x)*0.85, label='Background', c='b', ls='-.')
    plt.plot(x, true_model.pdf(x), label='Signal+background', color='green')
    plt.xlabel('M')
    plt.ylabel('PDF(M)')
    plt.title('True PDF')
    plt.legend()
    plt.savefig('plots/part_g_true_pdf.pdf')
    print("=======================================")
    print('Saving pdf file at plots/part_g_true_pdf.pdf')
    plt.show()
    
    print('Part d finished, moving on to part e now...')
    print("Executing exercise e)")
    print("=======================================")
    print('Generating sample')
    
    entries = args.points

    x = np.linspace(my_alpha, my_beta, entries)
    
    true_model = New_Model(f1=0.1, f2=0.05, lamda=0.5, mu_1=5.28, mu_2=5.35, sigma=0.018, alpha=5, beta=5.6, is_normalised=True)
    
    data = true_model.accept_reject(size=entries)
    
    plt.figure(figsize=(15,10))
    plt.hist(data, bins=100,range=(args.alpha, args.beta), density=True, label='Data')
    plt.plot(x, true_model.pdf(x), label='True pdf')
    plt.legend()
    plt.xlabel('M')
    plt.ylabel('PDF(M)')
    plt.title('High-statistics sample with {} events'.format(entries))
    plt.savefig('plots/part_g.pdf')
    print("=======================================")
    print('Saving pdf file at plots/part_g.pdf')
    plt.show()    
    
    def pdf(x, f1, f2, mu_1, mu_2, lamda, sigma):
        '''
        Function that returns the pdf of the model, easier to handle for the fit
        '''
        
        model = New_Model(f1=f1, f2=f2, mu_1=mu_1, mu_2=mu_2, lamda=lamda, sigma=sigma)
        
        return model.pdf(x)
    
    def s1_pdf(x, f1, mu_1, sigma, lamda):
        '''
        Function that returns the pdf of the background-only model, easier to handle for the fit
        '''
        
        model = New_Model(f1=f1, f2=0, mu_1=mu_1, mu_2=5.35, lamda=lamda, sigma=sigma)
        
        return model.pdf(x)
    
    
    nLL = UnbinnedNLL(data, pdf)
    mi = Minuit(nLL, mu_1=5.28, mu_2=5.35, f1=0.1, f2=0.05, lamda=0.5, sigma=0.018)
    mi.migrad()
    print(mi)
    
    hat_f1, hat_f2, hat_mu1, hat_mu2, hat_lamda, hat_sigma = mi.values
    
    plt.figure(figsize=(15,10))
    plt.hist(data, bins=100, density=True)
    plt.plot(x, true_model.pdf(x), label='True model')
    fit_model = New_Model(f1=hat_f1, f2=hat_f2, mu_1=hat_mu1, mu_2=hat_mu2, sigma=hat_sigma, lamda=hat_lamda, alpha=my_alpha, beta=my_beta, is_normalised=True)
    bkg_model = New_Model(f1=hat_f1, f2=0, mu_1=hat_mu1, mu_2=hat_mu2, sigma=hat_sigma, lamda=hat_lamda, alpha=my_alpha, beta=my_beta, is_normalised=True)
    plt.plot(x, fit_model.pdf(x), label='Fit model', color='green')
    plt.plot(x, fit_model.signal_1(x)/10., label='Fit signal / 10.', c='r', ls='--')
    plt.plot(x, fit_model.signal_2(x)/20., label='Fit signal / 10.', c='orange', ls='--')
    plt.plot(x, bkg_model.pdf(x)*0.85, label='Fit background', c='b', ls='-.')
    plt.title('Post-fit distribution')
    plt.xlabel('M')
    plt.ylabel('PDF(M)')
    plt.legend()
    plt.savefig("plots/fit_g.pdf")
    print("=======================================")
    print('Saving pdf file at plots/fit_g.pdf')
    plt.show()
    
    print("=======================================")
    print('Part e finished, moving on to part f now...')
    print("Executing exercise f)")
    print("=======================================")
    print('Test: fit background only')
    
    nLL = UnbinnedNLL(data, s1_pdf)
    mi = Minuit(nLL, f1=0.15, mu_1=5.315, sigma=0.018, lamda=0.4)
    
    mi.migrad()
    mi.hesse()
    print(mi)
    hat_f1, hat_mu_1, hat_sigma, hat_lamda = mi.values
    plt.figure(figsize=(15,10))
    plt.hist(data, bins=100, density=True)
    plt.plot(x, true_model.pdf(x), label='True model')
    bkg_model = New_Model(f1=hat_f1, f2=0, mu_1=hat_mu_1, mu_2=5.35, sigma=hat_sigma, lamda=hat_lamda)
    plt.plot(x, bkg_model.pdf(x), label='Fit background', c='b', ls='-.')
    plt.legend()
    plt.show()
    
    # print("=======================================")
    # print('Now moving on to part f)')
    # print("=======================================")
    
    # number_of_models = [50, 100, 200, 500, 750, 1000, 1100, 1250, 1500, 1750, 2000]
    # N_datasets = 1000
    
    # discovery_rates = []

    # if args.fit:
    #     for mod in number_of_models:
    #         print("=======================================")
    #         print("Evaluating now {} data points".format(mod))
            
    #         my_model = Model(f=0.1, lamda=0.5, sigma=0.018, mu=5.28)
    #         significances = []
    #         fails_H0 = 0
    #         fails_H1 = 0
    #         for i in range(N_datasets):
    #             data = my_model.accept_reject(size=mod)
    #             nLL_H0 = UnbinnedNLL(data, bkg_pdf)
    #             mi_H0 = Minuit(nLL_H0, lamda=0.5)
    #             mi_H0.migrad(iterate=10)
    #             mi_H0.hesse()
                
    #             mi_H0_min = mi_H0.fval
    #             nLL_H1 = UnbinnedNLL(data, pdf)
    #             mi_H1 = Minuit(nLL_H1, mu=5.28, f=0.1, lamda=0.5, sigma=0.018)
    #             mi_H1.migrad(iterate=10)
    #             mi_H1.hesse()
    #             if mi_H0.valid == False and mi_H1.valid == True:
    #                 fails_H0 += 1
    #                 continue
    #             elif mi_H0.valid == True and mi_H1.valid == False:
    #                 fails_H1 += 1
    #                 continue
    #             mi_H1_min = mi_H1.fval
                
    #             T = mi_H0_min - mi_H1_min
    #             sb_chisq = T
    #             sb_ndof = 1
    #             sb_pval = 1 - chi2.cdf(sb_chisq, sb_ndof)
    #             sb_sig = chi2.ppf(1 - sb_pval, 1)**0.5
    #             significances.append(sb_sig)
            
    #         print("Failed fits for H0 : {}".format(fails_H0))
    #         print("Failed fits for H1 : {}".format(fails_H1))
    #         val_sig = [value for value in significances if value >= 5.0]
    #         if len(significances) == 0:
    #             discovery_rates.append(0)
    #             continue
    #         disc_rate = float(len(val_sig))/(float(len(significances)))
    #         discovery_rates.append(disc_rate)
        
    #     else:
    #         discovery_rates = np.load("data/discovery_rates.npy")
    # plt.figure(figsize=(15,10))   
    # plt.scatter(number_of_models, discovery_rates)
    # plt.axhline(y=0.9, c='r', ls='--')
    # plt.title('Discovery rate vs number of points simulated')
    # plt.xlabel('Number of points simulated')
    # plt.ylabel('Discovery rate')
    # plt.xscale('log')
    # plt.savefig('plots/part_f.pdf')
    # print("=======================================")
    # print('Saving pdf file at plots/part_f.pdf')
    # print('Saving np array for future uses')
    # np.save('data/discovery_rates.npy', discovery_rates)
    # for i in range(len(discovery_rates)):
    #     print("=======================================")    
    #     print('Number of points = {0}, discovery rate = {1}'.format(number_of_models[i], discovery_rates[i]))    
    # plt.show()

        
            
            
    
    

if __name__ == "__main__":
    print("=======================================")
    print("Initialising coursework")
    print("=======================================")
    main()
    print("=======================================")
    print("Code finished. Exiting!")
    print("=======================================")
