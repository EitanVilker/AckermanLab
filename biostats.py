import parse_csv
import pandas as pd
import numpy as np
from statsmodels.stats.multitest import fdrcorrection
from scipy.stats import pearsonr, spearmanr, kendalltau

def ANOVA(group_list):
    pass

def multipel_comparisons():
    pass

def two_sample_t_test(group_0, group_1):
    pass

def correlation(group_0, group_1, type="pearson"):
    pass

def fdr_correction(p_value_list):
    rejected, corrected_p_values = fdrcorrection(p_value_list)
    return rejected, corrected_p_values

# Run 190 ANOVA tests
for feature in features:
    ANOVA()

# Run FDR correction on p-values
