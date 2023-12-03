#!/usr/bin/env python
import numpy as np
from matplotlib import pyplot as plt
import argparse
from iminuit.cost import UnbinnedNLL
from iminuit import Minuit
import time
from Helpers.HelperFunctions import Model, pdf

plt.style.use('mphil.mplstyle')

def main():
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-p', '--points', help="Number of points you  want to generate", type=int, required=False, default=100000)
    parser.add_argument('--plots', help='Flag: if selected, will show all plots, otherwise it will only save them', required=False, action='store_true')
    args = parser.parse_args()
    
    my_alpha = 5.0
    my_beta = 5.6
    
    print('Generating sample')
    
    entries = args.points

    x = np.linspace(my_alpha, my_beta, entries)
    
    true_model = Model(f=0.1, lamda=0.5, mu=5.28, sigma=0.018)
    
    data = true_model.accept_reject(size=entries)

    nLL = UnbinnedNLL(data, pdf)
    mi = Minuit(nLL, mu=5.28, f=0.1, lamda=0.5, sigma=0.018)
    mi.migrad()
    
    hat_f, hat_mu, hat_lamda, hat_sigma = mi.values
    fit_model = Model(f=hat_f, mu=hat_mu, sigma=hat_sigma, lamda=hat_lamda, alpha=my_alpha, beta=my_beta, is_normalised=True)
    
    
    plt.figure(figsize=(15,10))
    plt.hist(data, bins=100, density=True)
    plt.plot(x, true_model.pdf(x), label='True model')
    plt.plot(x, fit_model.pdf(x), label='Fit model', color='green')
    plt.plot(x, fit_model.pdf_signal(x)/10., label='Fit signal', c='r', ls='--')
    plt.plot(x, fit_model.pdf_background(x)*0.9, label='Fit background', c='b', ls='-.')
    plt.title('Post-fit distribution')
    plt.xlabel('M')
    plt.ylabel('PDF(M)')
    plt.legend()
    plt.savefig("plots/Part_e/fit.pdf")
    print("=======================================")
    print('Saving pdf file at plots/Part_e/fit.pdf')
    if args.plots:
        plt.show()
    

if __name__ == "__main__":
    print("=======================================")
    print("Initialising exercise e")
    print("=======================================")
    start_time = time.time()
    main()
    end_time = time.time()
    print("=======================================")
    print("Exercise e finished. Exiting!")
    print("Time it took to run the code : {} seconds". format(end_time - start_time))
    print("=======================================")
