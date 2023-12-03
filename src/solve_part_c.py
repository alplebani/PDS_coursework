#!/usr/bin/env python
import numpy as np
from matplotlib import pyplot as plt
import argparse
import time
from Helpers.HelperFunctions import Model

plt.style.use('mphil.mplstyle')

def main():
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-a', '--alpha', help='Value of lower limit of distribution', type=float, default=5., required=False)
    parser.add_argument('-b', '--beta', help='Value of the upper limit of the distribution', type=float, default=5.6, required=False)
    parser.add_argument('-n', '--nentries', help='Number of models to be tested', type=int, required=False, default=1000)
    parser.add_argument('--plots', help='Flag: if selected, will show all plots, otherwise it will only save them', required=False, action='store_true')
    args = parser.parse_args()
    
    if args.alpha >= args.beta:
        print('Error! alpha must be smaller than beta. Exiting the code!')
        exit(4)
    
    my_alpha = args.alpha
    my_beta = args.beta
    
    print("Executing exercise c)")
    print("=======================================")
    
    print("Testing to see if pdf is normalised")
    print("=======================================")
    print('Chosen values : alpha = {0}, beta = {1}'.format(my_alpha, my_beta))
    print("=======================================")
    
    n_entries = args.nentries
    
    print('Now testing {} models, and then evaluating average of integrals'.format(n_entries))
    print("=======================================")
        
    x = np.linspace(my_alpha, my_beta, n_entries) # select n_entries points in the selected range  
    
    np.random.seed(4999) # random seed to have always same results, chosen as my birthday 
    
    f_values = np.random.uniform(0, 1, n_entries) # random uniformly distributed values between 0 and 1 for fraction of signal
    lamda_values = np.random.uniform(0.1, 1, n_entries) # random uniformly distributed values between 0.1 and 1 for lamda
    mu_values = np.random.uniform(my_alpha, my_beta, n_entries) # random uniformly distributed values between alpha and beta for mu
    sigma_values = np.random.uniform(0.01, 1, n_entries) # random uniformly distributed values between 0.01 and 1 for sigma
    
    my_model = []
    integral = []
    my_models = []
    
    # Generating n_entries different models with the random parameters above and evaluating integral
    
    for i in range(n_entries):
        my_model = Model(f=f_values[i], lamda=lamda_values[i], mu=mu_values[i], sigma=sigma_values[i], alpha=my_alpha, beta=my_beta, is_normalised=True)
        if args.plots:
            my_models.append(my_model)
        integral.append(np.trapz(my_model.pdf(x), x))             
        
    print('Mean of {0} different models: {1} +- {2}'.format(n_entries, np.mean(integral), np.var(integral)))
    
    if args.plots:
        plt.figure(figsize=(15,10))
        for i in range(n_entries):
            plt.plot(x, my_models[i].pdf(x))            
        plt.xlabel('M')
        plt.ylabel('pdf(M)')
        plt.title('{0} different models for the pdf. Mean = {1}'.format(n_entries, np.mean(integral)))
        plt.savefig('plots/Part_c/a_{0}_{1}_{2}_entries.pdf'.format(my_alpha, my_beta, n_entries))
        print("=======================================")
        print('Saving pdf file at plots/Part_c/a_{0}_{1}_{2}_entries.pdf'.format(my_alpha, my_beta, n_entries)) 
        print('Showing plot with all the distributions')
        print("=======================================")
        plt.show()  
            

if __name__ == "__main__":
    print("=======================================")
    print("Initialising exercise c")
    print("=======================================")
    start_time = time.time()
    main()
    end_time = time.time()
    print("=======================================")
    print("Exercise c finished. Exiting!")
    print("Time it took to run the code : {} seconds". format(end_time - start_time))
    print("=======================================")
    
