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
from Helpers.HelperFunctions import Model, pdf, bkg_pdf

plt.style.use('mphil.mplstyle')

def main():
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-a', '--alpha', help='Value of lower limit of distribution', type=float, default=5., required=False)
    parser.add_argument('-b', '--beta', help='Value of the upper limit of the distribution', type=float, default=5.6, required=False)
    parser.add_argument('-n', '--nentries', help='Number of models to be tested', type=int, required=False, default=1000)
    parser.add_argument('-p', '--points', help="Number of points you  want to generate", type=int, required=False, default=100000)
    parser.add_argument('-f', '--fit', help='Flag whether you want to re-do the fits in part f) or if you just want to load the data', action='store_true', required=False, default=False)
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
    true_model = Model(f=0.1, lamda=0.5, mu=5.28, sigma=0.018, alpha=5, beta=5.6, is_normalised=True)
    bkg_model = Model(f=0, lamda=0.5, mu=5.28, sigma=0.018, alpha=5, beta=5.6, is_normalised=True)
    sig_model = Model(f=1, lamda=0.5, mu=5.28, sigma=0.018, alpha=5, beta=5.6, is_normalised=True)
    
    plt.figure(figsize=(15,10))
    plt.plot(x, sig_model.pdf(x)/10., label='Signal', c='r', ls='--')
    plt.plot(x, bkg_model.pdf(x)*0.9, label='Background', c='b', ls='-.')
    plt.plot(x, true_model.pdf(x), label='Signal+background', color='green')
    plt.xlabel('M')
    plt.ylabel('PDF(M)')
    plt.title('True PDF')
    plt.legend()
    plt.savefig('plots/true_pdf.pdf')
    print("=======================================")
    print('Saving pdf file at plots/true_pdf.pdf')
    plt.show()
    
    print('Part d finished, moving on to part e now...')
    print("Executing exercise e)")
    print("=======================================")
    print('Generating sample')
    
    entries = args.points

    x = np.linspace(my_alpha, my_beta, entries)
    
    true_model = Model(f=0.1, lamda=0.5, mu=5.28, sigma=0.018)
    
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
    plt.show()    

    nLL = UnbinnedNLL(data, pdf)
    mi = Minuit(nLL, mu=5.28, f=0.1, lamda=0.5, sigma=0.018)
    mi.migrad()
    
    hat_f, hat_mu, hat_lamda, hat_sigma = mi.values
    
    plt.figure(figsize=(15,10))
    plt.hist(data, bins=100, density=True)
    plt.plot(x, true_model.pdf(x), label='True model')
    fit_model = Model(f=hat_f, mu=hat_mu, sigma=hat_sigma, lamda=hat_lamda, alpha=my_alpha, beta=my_beta, is_normalised=True)
    bkg_model = Model(f=0, mu=hat_mu, sigma=hat_sigma, lamda=hat_lamda, alpha=my_alpha, beta=my_beta, is_normalised=True)
    plt.plot(x, fit_model.pdf(x), label='Fit model', color='green')
    plt.plot(x, fit_model.signal(x)/10., label='Fit signal', c='r', ls='--')
    plt.plot(x, bkg_model.pdf(x)*0.9, label='Fit background', c='b', ls='-.')
    plt.title('Post-fit distribution')
    plt.xlabel('M')
    plt.ylabel('PDF(M)')
    plt.legend()
    plt.savefig("plots/fit_e.pdf")
    print("=======================================")
    print('Saving pdf file at plots/fit_e.pdf')
    plt.show()
    
    print("=======================================")
    print('Part e finished, moving on to part f now...')
    print("Executing exercise f)")
    print("=======================================")
    print('Test: fit background only')
    
    nLL = UnbinnedNLL(data, bkg_pdf)
    mi = Minuit(nLL, lamda=0.4)
    
    mi.migrad()
    mi.hesse()
    hat_lamda = float(mi.values['lamda'])
    plt.figure(figsize=(15,10))
    plt.hist(data, bins=100, density=True)
    plt.plot(x, true_model.pdf(x), label='True model')
    bkg_model = Model(f=0, mu=5.28, sigma=0.018, lamda=hat_lamda)
    plt.plot(x, bkg_model.pdf(x), label='Fit background', c='b', ls='-.')
    plt.legend()
    plt.show()
    
    print("=======================================")
    print('Now moving on to part f)')
    print("=======================================")
    
    number_of_models = [50, 100, 200, 500, 750, 1000, 1025, 1050, 1075, 1100, 1250, 1500, 1750, 2000]
    N_datasets = 1000
    
    discovery_rates = []
    
    print(args.fit)

    if args.fit:
        for mod in number_of_models:
            print("=======================================")
            print("Evaluating now {} data points".format(mod))
            
            my_model = Model(f=0.1, lamda=0.5, sigma=0.018, mu=5.28)
            significances = []
            fails_H0 = 0
            fails_H1 = 0
            for i in range(N_datasets):
                data = my_model.accept_reject(size=mod)
                nLL_H0 = UnbinnedNLL(data, bkg_pdf)
                mi_H0 = Minuit(nLL_H0, lamda=0.5)
                mi_H0.migrad(iterate=10)
                mi_H0.hesse()
                
                mi_H0_min = mi_H0.fval
                nLL_H1 = UnbinnedNLL(data, pdf)
                mi_H1 = Minuit(nLL_H1, mu=5.28, f=0.1, lamda=0.5, sigma=0.018)
                mi_H1.migrad(iterate=10)
                mi_H1.hesse()
                if mi_H0.valid == False and mi_H1.valid == True:
                    fails_H0 += 1
                    continue
                elif mi_H0.valid == True and mi_H1.valid == False:
                    fails_H1 += 1
                    continue
                mi_H1_min = mi_H1.fval
                
                T = mi_H0_min - mi_H1_min
                sb_chisq = T
                sb_ndof = 1
                sb_pval = 1 - chi2.cdf(sb_chisq, sb_ndof)
                sb_sig = chi2.ppf(1 - sb_pval, 1)**0.5
                significances.append(sb_sig)
            
            print("Failed fits for H0 : {}".format(fails_H0))
            print("Failed fits for H1 : {}".format(fails_H1))
            val_sig = [value for value in significances if value >= 5.0]
            if len(significances) == 0:
                discovery_rates.append(0)
                continue
            disc_rate = float(len(val_sig))/(float(len(significances)))
            discovery_rates.append(disc_rate)
        
    else:
        discovery_rates = np.load("data/discovery_rates.npy")
            
    plt.figure(figsize=(15,10))   
    plt.scatter(number_of_models, discovery_rates)
    plt.axhline(y=0.9, c='r', ls='--')
    plt.title('Discovery rate vs number of points simulated')
    plt.xlabel('Number of points simulated')
    plt.ylabel('Discovery rate')
    plt.xscale('log')
    plt.savefig('plots/part_f.pdf')
    print("=======================================")
    print('Saving pdf file at plots/part_f.pdf')
    print('Saving np array for future uses')
    np.save('data/discovery_rates.npy', discovery_rates)
    for i in range(len(discovery_rates)):
        print("=======================================")    
        print('Number of points = {0}, discovery rate = {1}'.format(number_of_models[i], discovery_rates[i]))    
    plt.show()

        
            
            
    
    

if __name__ == "__main__":
    print("=======================================")
    print("Initialising coursework")
    print("=======================================")
    main()
    print("=======================================")
    print("Code finished. Exiting!")
    print("=======================================")
