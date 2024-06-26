#!/usr/bin/env python
import numpy as np
from matplotlib import pyplot as plt
import argparse
from iminuit.cost import UnbinnedNLL
from iminuit import Minuit
import time
from scipy.stats import chi2
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Helpers.HelperFunctions import Model, pdf, bkg_pdf

plt.style.use('mphil.mplstyle')

def main():
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-f', '--fit', help='Flag whether you want to re-do the fits in part f) or if you just want to load the data', action='store_true', required=False, default=False)
    parser.add_argument('--plots', help='Flag: if selected, will show all plots, otherwise it will only save them', required=False, action='store_true')
    args = parser.parse_args()
    
    np.random.seed(4999)
        
    sizes = [400, 500, 600, 650, 700, 750, 800, 850, 900, 1000]

    N_datasets = 1000
    
    discovery_rates = []

    
    my_model = Model(f=0.1, lamda=0.5, sigma=0.018, mu=5.28)

    if args.fit:
        uncertainties = []
        for size in sizes:
            print("=======================================")
            print("Evaluating now {} data points".format(size))
            
            significances = []
            fails_H0 = 0
            fails_H1 = 0
            for i in range(N_datasets):
                data = my_model.accept_reject(size=size)
                nLL_H0 = UnbinnedNLL(data, bkg_pdf)
                mi_H0 = Minuit(nLL_H0, lamda=0.4)
                mi_H0.migrad(iterate=5)
                mi_H0.hesse()
                
                mi_H0_min = mi_H0.fval
                nLL_H1 = UnbinnedNLL(data, pdf)
                mi_H1 = Minuit(nLL_H1, mu=5.28, f=0.1, lamda=0.5, sigma=0.018)
                mi_H1.migrad(iterate=5)
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
                sb_ndof = 3 # difference between free parameters of numerator (4) and denominator (1)
                sb_pval = 1 - chi2.cdf(sb_chisq, sb_ndof)
                significances.append(sb_pval)
            
            print("Failed fits for H0 : {}".format(fails_H0))
            print("Failed fits for H1 : {}".format(fails_H1))
            val_sig = [value for value in significances if value <= 2.9e-7]
            if len(significances) == 0:
                discovery_rates.append(0)
                continue
            disc_rate = float(len(val_sig))/(float(len(significances)))
            discovery_rates.append(disc_rate)
            uncertainties.append(np.sqrt((disc_rate * (1 - disc_rate)) / N_datasets))
            print("=======================================")
            print('Number of points = {0}, discovery rate = {1}'.format(size, disc_rate))
        
    else:
        discovery_rates = np.load("data/discovery_rates.npy")
        uncertainties = np.sqrt(discovery_rates * (1 - discovery_rates) / float(N_datasets)) # binomial uncertainty
        for i in range(len(discovery_rates)):
            print("=======================================")
            print('Number of points = {0}, discovery rate = {1}'.format(sizes[i], discovery_rates[i]))
    
    
    plt.figure(figsize=(15,10))   
    plt.errorbar(sizes, discovery_rates, yerr=uncertainties, fmt='ko')
    plt.axhline(y=0.9, c='r', ls='--')
    plt.title('Discovery rate vs number of points simulated')
    plt.xlabel('Number of points simulated')
    plt.ylabel('Discovery rate')
    plt.savefig('plots/Part_f/discovery_rates.pdf')
    print("=======================================")
    print('Saving pdf file at plots/Part_f/discovery_rates.pdf')
    print('Saving np array for future uses')
    np.save('data/discovery_rates.npy', discovery_rates)   
    if args.plots:
        plt.show()
        
        
if __name__ == "__main__":
    print("=======================================")
    print("Initialising exercise f")
    print("=======================================")
    start_time = time.time()
    main()
    end_time = time.time()
    print("=======================================")
    print("Exercise f finished. Exiting!")
    print("Time it took to run the code : {} seconds". format(end_time - start_time))
    print("=======================================")
