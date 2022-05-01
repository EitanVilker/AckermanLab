# Author @ Eitan Vilker
# Credit for multiple comparisons guide to https://s-nako.work/2020/01/one-way-anovaanalysis-of-variance-and-multiple-comparisons-in-python/

import parse_csv

import pandas as pd
import numpy as np
import bisect

# Stats modules
from statsmodels.stats.multitest import fdrcorrection
import statsmodels.formula.api as smf
import statsmodels.api as sm
import statsmodels.stats.anova as anova
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import pearsonr, spearmanr, kendalltau, f_oneway, contingency
from cramer_v import cramer

def three_way_ANOVA(a, b, c):
    f, p = f_oneway(a, b, c)
    return f, p

def tukey(group_names, *args):
    groups_list = []
    for i in range(len(args)):
        for j in range(len(args[i])):
            groups_list.append(group_names[i])
    groups = np.hstack(args)
    return pairwise_tukeyhsd(groups, groups_list)

def correlation(group_0, group_1, correlation="pearson"):
    if correlation == "pearson":
        correlation_coefficient, p_value = pearsonr(group_0, group_1)
    elif correlation == "spearman":
        correlation_coefficient, p_value = spearmanr(group_0, group_1)
    elif correlation == "kendalltau":
        correlation_coefficient, p_value = kendalltau(group_0, group_1)
    elif correlation == "cramer":
        contingency_table = pd.crosstab(index=group_0, columns=group_1, margins=True)
        # print(contingency_table)
        r = cramer(contingency_table)
        return r, None
    else:
        print("\n\nINVALID CORRELATION USED")
    
    return correlation_coefficient, p_value

def find_best_features(file='41591_2018_161_MOESM3_ESM.csv', feature_count=190):

    # Separate treatments
    group_a, group_b, group_c = parse_csv.separate_into_groups(file)
    feature_value_tuples = []
    keys = [] # Holds the p-values used for sorting
    all_p_values = []
    feature_indices = []
    feature_names = []
    
    # Run ANOVA on all features to collect all the features that initially pass the significance test
    for i in range(1, feature_count + 1):
        feature_name = group_a.columns[i]
        a = group_a.iloc[:, i]
        b = group_b.iloc[:, i]
        c = group_c.iloc[:, i]
        f, p = three_way_ANOVA(a, b, c)
        # if feature_name[len(feature_name) - 3: len(feature_name)] == "IgA":
        #     print(feature_name + "," + str(f) + "," + str(p))
        # print("Feature: " + feature_name + ", p: " + str(p))
        # if p < 0.05:
        if True:
            feature_indices.append(i)
            feature_names.append(group_a.columns[i])

    # print("Features after ANOVA: " + str(feature_names))
    # print("Count: " + str(len(feature_names)))

    # Run multiple comparisons (Tukey) on the ANOVA-selected features and output the features whose pairs of 
    # treatments had statistically significant difference in means
    for i in feature_indices:
        feature_name = group_a.columns[i]
        a = group_a.iloc[:, i]
        b = group_b.iloc[:, i]
        c = group_c.iloc[:, i]

        group_names =  ["IM239", "IM mosaic", "AE239"]

        # Run multiple comparison tests between each treatment
        result = tukey(group_names, a, b, c)
        p_values = result.pvalues

        # if feature_name[len(feature_name) - 3: len(feature_name)] == "IgA":
        #     print(feature_name)
        #     print(result)

        # Insert information into sorted list
        index = bisect.bisect(keys, p_values[0]); feature_value_tuples.insert(index, (p_values[0], result, feature_name, "AE239 vs IM mosaic")); keys.insert(index, p_values[0])
        index = bisect.bisect(keys, p_values[1]); feature_value_tuples.insert(index, (p_values[1], result, feature_name, "AE239 vs IM239"));keys.insert(index, p_values[1])
        index = bisect.bisect(keys, p_values[2]); feature_value_tuples.insert(index, (p_values[2], result, feature_name, "IM mosaic vs IM239")); keys.insert(index, p_values[2])

    # Output best p-values/features
    # for i in range(100):
    #     fvt = feature_value_tuples[i]
        # print("Feature:  " + str(fvt[2]) + ", Treatments: " + str(fvt[3]) + ", P-value: " + str("{:.2f}".format(fvt[0])) + ", Meandiffs: " + str(fvt[1].meandiffs))

    # count = 0
    # for i in keys:
    #     if i < 0.05:
    #         count += 1
    # print("Count before FDR: " + str(count))

    # Run FDR correction on p-values
    print("\n\nFDR")
    rejected, corrected = fdrcorrection(keys)
    for i in range(len(corrected)):
        feature_name = feature_value_tuples[i][2]
        # Negative meandiff means group 1 is larger than group 2
        # if feature_name[len(feature_name) - 3: len(feature_name)] == "IgA":
        # if feature_name[3:6] == "IgG" or feature_name[1:4] == "IgG":
        # if feature_name[1:5] == "FcgR" or feature_name[:4] == "rR2A" or feature_name[:4] == "rR3A" or feature_name[:3] == "R3A" or feature_name[:3] == "R2A":
        # if feature_name[:3] == "MIP" or feature_name[:3] == "IFN":
        if feature_name[:9] == "R2A.4.low":
            if True: #corrected[i] < 0.05:
                result = feature_value_tuples[i][1]
                print("\n" + feature_name)
                print(result.pvalues)
                print(result)
                print(corrected[i])

        # if i % 10 == 0:
            # print(corrected[i])
            # print(str(feature_value_tuples[i][2]) + "," + str(feature_value_tuples[i][3]))
    count = 0
    for i in corrected:
        if i < 0.05:
            count += 1
    # print("Count after FDR: " + str(count))

    correlations_keys = []
    correlations = []
    attributes, classifier1, classifier2, classifier3, classifier4, classifier5, classifier6, averages, standard_deviations, mean_animals = parse_csv.parse_csv(190, 60, filename=filename)
    # Look at all pairs of features
    feature_count = len(feature_names)
    for i in range(feature_count - 1):
        feature_i = feature_names[i]
        for j in range(i + 1, feature_count):
            feature_j = feature_names[j]
            r, p = correlation(attributes[feature_i], attributes[feature_j])
            if r > 0.8:
                index = bisect.bisect(correlations_keys, r); correlations.insert(index, (r, feature_i, feature_j)); keys.insert(index, r)
    
    # print(correlations)
    # print("Len correlations: " + str(len(correlations)))
    


filename = "41591_2018_161_MOESM3_ESM.csv"
find_best_features(file=filename)
attributes, classifier1, classifier2, classifier3, classifier4, classifier5, classifier6, averages, standard_deviations, mean_animals = parse_csv.parse_csv(190, 60, filename=filename)

# folder = "Correlations/"
# file_path = folder + "c4.csv"
# data = pd.read_csv(file_path)
# predictions = data["Targets"]
# targets = data["Predictions"]

# r, p = correlation(targets, predictions, correlation="cramer")
# print(r)

# targets = attributes["ADCP gp140"]
# predictions = attributes["ADNP gp140"]
# r, p = correlation(targets, predictions, correlation="pearson")
# # print("Pearson r: " + str(r))
# print(r)
# print(p)
# r, p = correlation(targets, predictions, correlation="spearman")
# # print("Spearman r: " + str(r))
# print(r)
# print(p)
# r, p = correlation(targets, predictions, correlation="kendalltau")
# # print("Kendalltau r: " + str(r))
# print(r)
# print(p)

# group_a, group_b, group_c = parse_csv.separate_into_groups(filename)
# a = group_a.iloc[:, 3]
# b = group_b.iloc[:, 3]
# c = group_c.iloc[:, 3]

# group_names =  ["IM239", "IM mosaic", "AE239"]

# # Run multiple comparison tests between each treatment
# result = tukey(group_names, a, b, c)
# print(result)