# Basic libraries
import numpy as np
import sys
import random
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import tensorflow.keras as keras

# Local Python scripts
import parse_csv as p
import nn_functions as nn
import RNN
import evaluate as e
import graphing as g
import cox_based_model as cox

# ML functions
from sklearn import linear_model, random_projection
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.nonparametric import kaplan_meier_estimator
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans


random.seed(None)

''' Function to shuffle the labels of the subjects randomly for permutation testing '''
def randomize_labels(classifier, classifier_type=3):
    print("\n\nClassifier type: " + str(classifier_type))
    for i in range(len(classifier)):
        if classifier_type == 1:
            classifier[i] = random.randint(0, 1)
        elif classifier_type == 2:
            classifier[i] = random.uniform(0.0, 7.5)
        elif classifier_type == 3:
            classifier[i] = random.randint(0, 2)
        elif classifier_type == 4:
            classifier[i] = random.randint(1, 13)
        elif classifier_type == 5:
            # classifier[i] = random.randint(0, 2)
            rand = random.randint(0, 2)
            print(rand)
            if rand == 0:
                classifier[i] = "IM239"
            elif rand == 1:
                classifier[i] = "AE239"
            elif rand == 2:
                classifier[i] = "IM mosaic"

testing_binary_classifier = False
testing_large_data_set = True
feature_count = 190
subject_count = 60

if testing_large_data_set:
    filename = '41591_2018_161_MOESM3_ESM.csv'
else:
    filename = '41591_2018_161_MOESM4_ESM.csv'

''' split into input and output variables '''
if testing_large_data_set:
    attributes, classifier1, classifier2, classifier3, classifier4, classifier5, classifier6, averages, standard_deviations, mean_animals = p.parse_csv(feature_count, subject_count, filename=filename)
    classifier2 = classifier2.tolist()
else:
    attributes, classifier1 = p.parse_csv_small(filename, 4, 18)
    classifier2 = None
    classifier3 = None
    classifier4 = None

permuting = False
removing_outliers = True

if len(sys.argv) == 1:
    classifier = classifier3
if len(sys.argv) > 1:
    classifier = str(sys.argv[1])
    if classifier == "1":
        classifier = classifier1
    elif classifier == "2":
        classifier = classifier2
    elif classifier == "3":
        classifier = classifier3
    elif classifier == "4":
        classifier = classifier4
    elif classifier == "5":
        classifier = classifier5
    elif classifier == "6":
        classifier = classifier6
if len(sys.argv) > 2:
    if sys.argv[2] == "1":
        print("setting permuting to True")
        permuting = True
if len(sys.argv) > 3:
    adding_artificial_subjects = int(sys.argv[3])
else:
    adding_artificial_subjects = 0
if permuting:
    randomize_labels(classifier, classifier_type=int(sys.argv[1]))

if removing_outliers:
    CI_val = 1.645
    for i in range(feature_count):
        for j in range(subject_count):
            val = attributes.iat[j, i]
            if val > CI_val:
                attributes.iat[j, i] = CI_val
            elif val < -CI_val:
                attributes.iat[j, i] = -CI_val

holdout_attributes = np.array([])
holdout_classifiers = np.array([])
if adding_artificial_subjects > 0:

    attributes, classifier, holdout_attributes, holdout_classifiers, subject_count = cox.get_holdouts(attributes, classifier)
    attributes = cox.add_artificial_subjects(attributes, classifier, original_subject_count=subject_count, additional_subject_count=adding_artificial_subjects)

''' Regression by group. For some reason works better/only if the groups are run individually '''
# nn.ridge_regression(attributes, classifier3, test_count=1)
# group_a, group_b, group_c = p.separate_into_groups(filename)
# attributes, classifier1, classifier2, classifier3, classifier4, averages, standard_deviations = p.parse_csv(185, 20, data=group_a, group_number=0)
# nn.ridge_regression(attributes, classifier2, test_count=1000, classifier_type=3)
# attributes, classifier1, classifier2, classifier3, classifier4, averages, standard_deviations = p.parse_csv(185, 20, data=group_b, group_number=1)
# nn.ridge_regression(attributes, classifier2, test_count=1000, classifier_type=3)
# attributes, classifier1, classifier2, classifier3, classifier4, averages, standard_deviations = p.parse_csv(185, 20, data=group_c, group_number=2)
# nn.ridge_regression(attributes, classifier2, test_count=1000, classifier_type=3)

''' Perform feature selection '''
# current_best_feature_count = 0
# current_best_loss = 999999999
# current_best_features = None

# for i in range(100):

#     loss, features = nn.lin_reg_feature_selection(attributes, classifier2, i + 1, test_count=100, classifier_type=2)
#     if loss < current_best_loss:
#         current_best_loss = loss
#         current_best_feature_count = i + 1
#         current_best_features = features
# print("\nCurrent best feature count: " + str(current_best_feature_count))
# print("\nCurrent best loss: " + str(current_best_loss))
# print("\nCurrent best features: ")
# print(current_best_features)


''' LASSO regression '''
# l2_penalty_mse, best_l2, best_model, importance = nn.lasso_regression(attributes, classifier3)
# print("\nANALYSIS \n\nl2 penalty: ")
# print(l2_penalty_mse)
# print("Best 12: ")
# print(best_l2)
# print("Best model: ")
# print(best_model)
# print("Importance: ")
# print(importance)

''' LASSO regression that also gets weights '''
# best_alpha, coefficients, importance = nn.grid_search_lasso(attributes, classifier2, test_count=1)
# print("\nAlpha: ")
# print(best_alpha)
# print("Coefficients: ")
# print(coefficients)
# print("Importance: ")
# print(importance)
# print("Surviving features: ")
# print(surviving_features)

''' LARS '''
# nn.lasso_lars(attributes, classifier2, test_count=100, classifier_type=2)

''' Multi-Layer Perceptron '''
# nn.mlp(attributes, classifier3, test_count=100, classifier_type=3)

''' Basic Linear Regression '''
# nn.linear_regression(attributes, classifier, test_count=1, classifier_type=4)


''' Random Projections '''

# Gaussian
# nn.gaussian_random_projection(attributes, classifier2, classifier_type=2, attributes_wanted=4, prediction_method=nn.ridge_regression)

# Sparse
# nn.sparse_random_projection(attributes, classifier2, classifier_type=2, prediction_method=nn.linear_regression)


''''' Cox based 3-step feature selection '''''

''' Switch features and subjects for the purpose of clustering '''
# features = attributes.transpose()
# cluster_count=9
# clustering, labels = cox.identify_clusters(features, clusters_wanted=cluster_count)

# print(clustering)

# print("params: ")
# print(clustering.get_params())
# print("\nClusters: ")
# print(labels)

# ''' Graph clusters '''
# # plt.scatter(features.iloc[:, 0], features.iloc[:, 1], c=labels, s=50, cmap='viridis')
# # plt.title("Features separated into 9 clusters")
# # plt.show()

# ''' Remove bad features '''
# best_features = cox.get_best_features_by_pearson(labels, features, classifier, cluster_count=cluster_count)
# print("\nBest features: ")
# print(best_features)

# features_to_remove = []
# for i in range(185):
#     if attributes.columns.values[i] not in best_features:
#         features_to_remove.append(attributes.columns.values[i])

# for feature in features_to_remove:
#     attributes = attributes.drop(feature, axis=1)

# ''' Run model and permute '''
# # estimator = nn.lasso_lars(attributes, classifier, test_count=100)
# estimator = linear_model.Ridge(alpha=0.1, solver='cholesky')

# score, permutation_scores, pvalue = e.permutation_test_score(estimator, attributes, classifier)

# print("Score: " + str(score))
# print("Permutation scores: ")
# print(permutation_scores)
# print("pvalue: " + str(pvalue))

# estimator.fit(attributes, classifier)

# print(estimator.predict(attributes))
# print(estimator.score(attributes, classifier))

''' Create scatter plots of feature comparisons '''
# g.scatter(attributes['rIgG_high.SIVsmE543.gp140'], attributes['G146.151.SIV.gp120.biotinylated.peptide.IgA'], classifier, "rIgG_high.SIVsmE543.gp140", "G146.151.SIV.gp120.biotinylated.peptide.IgA")
# g.scatter(attributes['MIP1β'], attributes['Bisecting'], classifier, "MIP1β", 'Bisecting')
# g.scatter(attributes['R2A.4.high.J08.V1V2.mac239.AVI.His'], attributes['R2A.4.high.C1.TR'], classifier, "R2A.4.high.J08.V1V2.mac239.AVI.His", 'R2A.4.high.C1.TR')
# g.scatter(attributes['rIgG_high.SIVsmE543.gp140'], attributes['R2A.4.high.C1.TR'], classifier, "rIgG_high.SIVsmE543.gp140", 'R2A.4.high.C1.TR')
# g.scatter(attributes['ADCP gp140'], attributes['ADNP gp140'], classifier, "ADCP gp140", "ADNP gp140")
# g.scatter(attributes['rIgG_high.SIVsmE543.gp140'], attributes['Bisecting'], classifier, "rIgG_high.SIVsmE543.gp140", "Bisecting")
# g.scatter(attributes['MIP1β'], attributes['G146.151.SIV.gp120.biotinylated.peptide.IgA'], classifier, "MIP1β", "G146.151.SIV.gp120.biotinylated.peptide.IgA")

# nn.pca_transform_attributes(attributes, classifier)

# print(attributes)
# nn.lasso_lars(attributes, classifier, classifier_type=5, test_count=1000)

# cox.survival_analysis(attributes, classifier)

# print("Before permuting: ")
# print(classifier)

# print("After permuting")
# print(classifier)

# reduced_features = cox.sequential_feature_selection(attributes, classifier)
# # nn.ridge_regression(reduced_features, classifier, classifier_type=3)
# knn = KNeighborsClassifier(n_neighbors=5)
# knn.fit(reduced_features, classifier)
# for i in range(60 + adding_artificial_subjects):
#     row = [reduced_features[i, :]]
#     print(classifier[i])
#     print(knn.predict_proba(row))

# print(classifier)
# estimator = CoxPHSurvivalAnalysis(alpha=1.0).fit(attributes, classifier)
# surv_funcs = estimator.predict_survival_function(mean_animals.transpose())
# # tup_list1 = []
# # tup_list2 = []
# # for i in range(60):
# #     print(classifier6[i])
# #     tup_list1.append(classifier6[i][0])
# #     tup_list2.append(classifier6[i][1])
# # time, survival_prob = kaplan_meier_estimator(tup_list1, tup_list2)
# # plt.step(time, survival_prob, where="post")
# plt.ylabel("est. probability of survival $\hat{S}(t)$")
# plt.xlabel("challenges $t$")
# for fn in surv_funcs:
#     plt.step(fn.x, fn(fn.x), where="post")

# plt.ylim(0, 1)
# plt.show()

''' RNN '''
RNN.RNN_model(attributes, classifier, holdout_attributes=holdout_attributes, holdout_classifiers=holdout_classifiers, input_dim=190, subject_count=60+adding_artificial_subjects, artificial_count=adding_artificial_subjects)