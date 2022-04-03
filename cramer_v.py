# From https://www.statology.org/cramers-v-in-python/ with small modifications

#load necessary packages and functions
import scipy.stats as stats
import numpy as np

def cramer(data):
    #Chi-squared test statistic, sample size, and minimum of rows and columns
    X2 = stats.chi2_contingency(data, correction=False)[0]
    n = np.sum(data)
    minDim = min(data.shape)-1

    #calculate Cramer's V 
    V = np.sqrt((X2/n) / minDim)
    return V