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
from Helpers.HelperFunctions import Model, New_Model, pdf, pdf_new_model

plt.style.use('mphil.mplstyle')
  

def main():
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-p', '--points', help="Number of points you  want to generate", type=int, required=False, default=100000)
    parser.add_argument('-f', '--fit', help='Flag whether you want to re-do the fits in part f) or if you just want to load the data', required=False, default=False, action='store_true')
    parser.add_argument('--plots', help='Flag: if selected, will show all plots, otherwise it will only save them', required=False, action='store_true')
    args = parser.parse_args()
    
    my_alpha = 5.0
    my_beta = 5.6
    
    x = np.linspace(my_alpha, my_beta, 1000)
    true_model = New_Model(f1=0.1, f2=0.05, lamda=0.5, mu_1=5.28, mu_2=5.35, sigma=0.018, alpha=5, beta=5.6, is_normalised=True)
        
    plt.figure(figsize=(15,10))
    plt.plot(x, true_model.pdf_signal_1(x)/10., label='S1', c='r', ls='--')
    plt.plot(x, true_model.pdf_signal_2(x)/20., label='S2', c='orange', ls='--')
    plt.plot(x, true_model.pdf_background(x)*0.85, label='Background', c='b', ls='-.')
    plt.plot(x, true_model.pdf(x), label='Total PDF', color='green')
    plt.xlabel('M')
    plt.ylabel('PDF(M)')
    plt.title('True PDF')
    plt.legend()
    plt.savefig('plots/Part_g/true_pdf.pdf')
    print('Saving pdf file at plots/Part_g/true_pdf.pdf')    
    
    entries = args.points

    x = np.linspace(my_alpha, my_beta, entries)
    
    true_model = New_Model(f1=0.1, f2=0.05, lamda=0.5, mu_1=5.28, mu_2=5.35, sigma=0.018, alpha=5, beta=5.6, is_normalised=True)
    
    data = true_model.accept_reject(size=entries)

    nLL = UnbinnedNLL(data, pdf_new_model)
    mi = Minuit(nLL, mu_1=5.28, mu_2=5.35, f1=0.1, f2=0.05, lamda=0.5, sigma=0.018)
    mi.migrad()
    
    hat_f1, hat_f2, hat_mu1, hat_mu2, hat_lamda, hat_sigma = mi.values
    fit_model = New_Model(f1=hat_f1, f2=hat_f2, mu_1=hat_mu1, mu_2=hat_mu2, sigma=hat_sigma, lamda=hat_lamda, alpha=my_alpha, beta=my_beta, is_normalised=True)
    
    plt.figure(figsize=(15,10))
    plt.hist(data, bins=100, density=True)
    plt.plot(x, true_model.pdf(x), label='True model')
    plt.plot(x, fit_model.pdf(x), label='Fit total PDF', color='green')
    plt.plot(x, fit_model.pdf_signal_1(x)/10., label='Fit signal S1', c='r', ls='--')
    plt.plot(x, fit_model.pdf_signal_2(x)/20., label='Fit signal S2', c='orange', ls='--')
    plt.plot(x, fit_model.pdf_background(x)*0.85, label='Fit background', c='b', ls='-.')
    plt.title('Post-fit distribution')
    plt.xlabel('M')
    plt.ylabel('PDF(M)')
    plt.legend()
    plt.savefig("plots/Part_g/fit.pdf")
    print("=======================================")
    print('Saving pdf file at plots/Part_g/fit.pdf')
    
    number_of_models = [2000, 3000, 3250, 3500, 3750, 4000, 4250, 4500, 4750, 5000, 10000]
    N_datasets = 1000
    
    discovery_rates = []

    if args.fit:
        for mod in number_of_models:
            print("=======================================")
            print("Evaluating now {} data points".format(mod))
            
            my_model = New_Model(f1=0.1, f2=0.05, mu_1=5.28, mu_2=5.35, lamda=0.5, sigma=0.018)
            significances = []
            fails_H0 = 0
            fails_H1 = 0
            for i in range(N_datasets):
                data = my_model.accept_reject(size=mod)
                nLL_H0 = UnbinnedNLL(data, pdf)
                mi_H0 = Minuit(nLL_H0, lamda=0.5, f=0.15, mu=5.3, sigma=0.018)
                mi_H0.migrad(iterate=10)
                mi_H0.hesse()
                
                mi_H0_min = mi_H0.fval
                nLL_H1 = UnbinnedNLL(data, pdf_new_model)
                mi_H1 = Minuit(nLL_H1, mu_1=5.28, f1=0.1, f2=0.05, mu_2=5.35, lamda=0.5, sigma=0.018)
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
        discovery_rates = np.load("data/discovery_rates_g.npy")
        
    plt.figure(figsize=(15,10))   
    plt.scatter(number_of_models, discovery_rates)
    plt.axhline(y=0.9, c='r', ls='--')
    plt.title('Discovery rate vs number of points simulated')
    plt.xlabel('Number of points simulated')
    plt.ylabel('Discovery rate')
    plt.xscale('log')
    plt.savefig('plots/Part_g/discovery_rates.pdf')
    print("=======================================")
    print('Saving pdf file at plots/Part_g/discovery_rates.pdf')
    print('Saving np array for future uses')
    np.save('data/discovery_rates_g.npy', discovery_rates)
    for i in range(len(discovery_rates)):
        print("=======================================")    
        print('Number of points = {0}, discovery rate = {1}'.format(number_of_models[i], discovery_rates[i]))    
    if args.plots:
        plt.show()

   
if __name__ == "__main__":
    print("=======================================")
    print("Initialising exercise g")
    print("=======================================")
    start_time = time.time()
    main()
    end_time = time.time()
    print("=======================================")
    print("Exercise g finished. Exiting!")
    print("Time it took to run the code : {} seconds". format(end_time - start_time))
    print("=======================================")

