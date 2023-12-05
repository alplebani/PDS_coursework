#!/usr/bin/env python
import numpy as np
from matplotlib import pyplot as plt
import argparse
import time
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Helpers.HelperFunctions import Model

plt.style.use('mphil.mplstyle')

def main():
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--plots', help='Flag: if selected, will show all plots, otherwise it will only save them', required=False, action='store_true')
    args = parser.parse_args()
    
    my_alpha = 5.0
    my_beta = 5.6
    
    print("Plotting true distributions")
    
    x = np.linspace(my_alpha, my_beta, 1000)
    true_model = Model(f=0.1, lamda=0.5, mu=5.28, sigma=0.018, alpha=5, beta=5.6, is_normalised=True)
    
    plt.figure(figsize=(15,10))
    plt.plot(x, true_model.pdf_signal(x)/10., label='Signal', c='r', ls='--')
    plt.plot(x, true_model.pdf_background(x)*0.9, label='Background', c='b', ls='-.')
    plt.plot(x, true_model.pdf(x), label='Signal+background', color='green')
    plt.xlabel('M')
    plt.ylabel('PDF(M)')
    plt.title('True PDF')
    plt.legend()
    plt.savefig('plots/Part_d/true_pdf.pdf')
    print("=======================================")
    print('Saving pdf file at plots/Part_d/true_pdf.pdf')
    if args.plots:
        plt.show()
    

if __name__ == "__main__":
    print("=======================================")
    print("Initialising exercise d")
    print("=======================================")
    start_time = time.time()
    main()
    end_time = time.time()
    print("=======================================")
    print("Exercise d finished. Exiting!")
    print("Time it took to run the code : {} seconds". format(end_time - start_time))
    print("=======================================")
