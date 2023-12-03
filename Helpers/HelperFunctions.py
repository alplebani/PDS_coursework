# Helpers/HelperFunctions.py
import scipy
from math import log
from scipy.integrate import simpson
from scipy.special import erf
from scipy.optimize import minimize
from scipy.stats import norm, expon
from functools import partial
from scipy.integrate import quad
import numpy as np

# ===============================================
# Helper functions for the model distributions
# ===============================================

class Model():
    '''
    Statistical model. Contains all functions needed to run the code.
    '''
    def __init__(self, f, lamda, sigma, mu, alpha=5., beta=5.6, is_normalised=True):
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
    
    def N_S(self):
        '''
        Normalisation factor of the signal
        '''
        
        return 0.5 * (erf((self.beta - self.mu)/(np.sqrt(2) * self.sigma)) - erf((self.alpha - self.mu)/(np.sqrt(2) * self.sigma)))

    
    def N_B(self):
        '''
        Normalisation factor of the background
        '''
        if self.alpha >= 0:
            return np.exp(- self.lamda * self.alpha) - np.exp(-self.lamda * self.beta)
        else: 
            return 1 - np.exp(-self.lamda * self.beta)
    
    def norm_factors(self):
        '''
        Function that returns the normalisation factors for signal and background pdfs
        '''
        
        return self.N_S(), self.N_B()
    
    def pdf_signal(self, x):
        '''
        Normalised distribution of the signal
        '''
        
        return self.signal(x)/self.N_S()
    
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
            return (self.f * self.pdf_signal(x) + (1 - self.f) * self.pdf_background(x))
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
        if self.alpha >= 0:
            return np.exp(- self.lamda * self.alpha) - np.exp(-self.lamda * self.beta)
        else: 
            return 1 - np.exp(-self.lamda * self.beta)
        
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


def pdf(x, f, mu, lamda, sigma):
    '''
    Function that returns the pdf of the model, easier to handle for the fit
    '''
    
    model = Model(f=f, mu=mu, lamda=lamda, sigma=sigma)
    
    return model.pdf(x)
    
def bkg_pdf(x, lamda):
    '''
    Function that returns the pdf of the background-only model, easier to handle for the fit
    '''
    
    model = Model(f=0, mu=5.28, lamda=lamda, sigma=0.018)
    
    return model.pdf(x)

def pdf_new_model(x, f1, f2, mu_1, mu_2, lamda, sigma):
    '''
    Function that returns the pdf of the model, easier to handle for the fit
    '''
    
    model = New_Model(f1=f1, f2=f2, mu_1=mu_1, mu_2=mu_2, lamda=lamda, sigma=sigma)
    
    return model.pdf(x)