# Basic libraries
import numpy as np
import sys
import random
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import tensorflow.keras as keras

# Local Python scripts
import parse_csv as p
import ml_functions as ml
import NN
import evaluate as e

# ML functions
from sklearn import linear_model, random_projection
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sksurv.nonparametric import kaplan_meier_estimator
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline, make_pipeline

import warnings
warnings.filterwarnings("ignore")
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

''' Adjusts feature values for the purpose of assessing directionality '''
def change_feature_values(attributes, subject_count=60, feature_index=None, change_amount=-2):
    for i in range(subject_count):
        if feature_index is not None:
            attributes[i, feature_index] += change_amount
    return attributes         


testing_binary_classifier = False
testing_large_data_set = True
permuting = False
holdout_proportion = 0.2
separating_holdouts = False
removing_outliers = True
adding_arm_feature = False
removing_select_features = False
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

if removing_select_features:
    feature_count -= 4

if removing_outliers:
    CI_val = 1.645
    for i in range(feature_count):
        for j in range(subject_count):
            val = attributes.iat[j, i]
            if val > CI_val:
                attributes.iat[j, i] = CI_val
            elif val < -CI_val:
                attributes.iat[j, i] = -CI_val

# Add feature for arm labels because it is highly informative using one-hot encoding
if adding_arm_feature:

    arm = classifier5.copy()
    for i in range(len(arm)):
        if arm[i] == "IM239":
            arm[i] = 1
        else:
            arm[i] = 0
    attributes["IM239"] = arm
    arm = classifier5.copy()
    for i in range(len(arm)):
        if arm[i] == "IM mosaic":
            arm[i] = 1
        else:
            arm[i] = 0
    attributes["IM mosaic"] = arm
    arm = classifier5.copy()
    for i in range(len(arm)):
        if arm[i] == "AE239":
            arm[i] = 1
        else:
            arm[i] = 0
    attributes["AE239"] = arm

    feature_count += 3

if separating_holdouts:
    holdout_attributes = np.array([])
    holdout_classifiers = np.array([])
    attributes, classifier, holdout_attributes, holdout_classifiers, subject_count = ml.get_holdouts(attributes, classifier, holdout_proportion=holdout_proportion)

print(adding_artificial_subjects)
if adding_artificial_subjects > 0:
    attributes, classifier = ml.add_artificial_subjects(attributes, classifier, original_subject_count=subject_count, additional_subject_count=adding_artificial_subjects, feature_count=feature_count)


''' Perform feature selection '''
# current_best_feature_count = 0
# current_best_loss = 999999999
# current_best_features = None

# for i in range(100):

#     loss, features = ml.lin_reg_feature_selection(attributes, classifier2, i + 1, test_count=100, classifier_type=2)
#     if loss < current_best_loss:
#         current_best_loss = loss
#         current_best_feature_count = i + 1
#         current_best_features = features
# print("\nCurrent best feature count: " + str(current_best_feature_count))
# print("\nCurrent best loss: " + str(current_best_loss))
# print("\nCurrent best features: ")
# print(current_best_features)


''' LASSO regression '''
# l2_penalty_mse, best_l2, best_model, importance = ml.lasso_regression(attributes, classifier3)
# print("\nANALYSIS \n\nl2 penalty: ")
# print(l2_penalty_mse)
# print("Best 12: ")
# print(best_l2)
# print("Best model: ")
# print(best_model)
# print("Importance: ")
# print(importance)

''' LASSO regression that also gets weights '''
# best_alpha, coefficients, importance = ml.grid_search_lasso(attributes, classifier2, test_count=1)
# print("\nAlpha: ")
# print(best_alpha)
# print("Coefficients: ")
# print(coefficients)
# print("Importance: ")
# print(importance)
# print("Surviving features: ")
# print(surviving_features)

''' LARS '''
# ml.lasso_lars(attributes, classifier2, test_count=100, classifier_type=2)

''' Multi-Layer Perceptron '''
# ml.mlp(attributes, classifier3, test_count=100, classifier_type=3)

''' Linear Regression '''
# ml.linear_regression(attributes, classifier, test_count=100, classifier_type=int(sys.argv[1]))

''' Logistic Regression '''
# ml.logistic_regression(attributes, classifier, test_count=100, classifier_type=int(sys.argv[1]))

''' SVM '''
# ml.SVM(attributes, classifier, test_count=100)

''' KNN '''
# ml.KNN(attributes, classifier, test_count=100, classifier_type=int(sys.argv[1]))

''' SGD '''
# ml.SGD(attributes, classifier, test_count=100, classifier_type=int(sys.argv[1]))

''' Ridge Regression '''
# ml.ridge_regression(attributes, classifier, test_count=100, classifier_type=int(sys.argv[1]))

''' Random Projections '''

# Gaussian
# ml.gaussian_random_projection(attributes, classifier2, classifier_type=2, attributes_wanted=4, prediction_method=ml.ridge_regression)

# Sparse
# ml.sparse_random_projection(attributes, classifier2, classifier_type=2, prediction_method=ml.linear_regression)


''''' 3-step feature selection '''''

''' Switch features and subjects for the purpose of clustering '''
# features = attributes.transpose()
# cluster_count=9
# clustering, labels = ml.identify_clusters(features, clusters_wanted=cluster_count)

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
# best_features = ml.get_best_features_by_pearson(labels, features, classifier, cluster_count=cluster_count)
# print("\nBest features: ")
# print(best_features)

# features_to_remove = []
# for i in range(185):
#     if attributes.columns.values[i] not in best_features:
#         features_to_remove.append(attributes.columns.values[i])

# for feature in features_to_remove:
#     attributes = attributes.drop(feature, axis=1)

# ''' Run model and permute '''
# # estimator = ml.lasso_lars(attributes, classifier, test_count=100)
# # estimator = linear_model.Ridge(alpha=0.1, solver='cholesky')
# ml.ridge_regression(attributes, classifier, test_count=100, classifier_type=int(sys.argv[1]))

# score, permutation_scores, pvalue = e.permutation_test_score(estimator, attributes, classifier)

# print("Score: " + str(score))
# print("Permutation scores: ")
# print(permutation_scores)
# print("pvalue: " + str(pvalue))

# estimator.fit(attributes, classifier)

# print(estimator.predict(attributes))
# print(estimator.score(attributes, classifier))

# ml.pca_transform_attributes(attributes, classifier)

# print(attributes)
# ml.lasso_lars(attributes, classifier, classifier_type=5, test_count=1000)

# ml.survival_analysis(attributes, classifier)

''' Sequential Feature Selection '''
sfs = ml.sequential_feature_selection(attributes, classifier)
reduced_features = sfs.transform(attributes)
# estimator = linear_model.Ridge(alpha=0.1, solver='saga')
# estimator = KNeighborsClassifier(n_neighbors=5)
estimator = linear_model.LinearRegression()
# estimator = make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3, penalty='elasticnet'))
estimator.fit(reduced_features, classifier)

''' Identifying directionality and weights of features '''
feature_weight_dict = {}
feature_weight_dict["Base Model"] = 0
for i in range(60 + adding_artificial_subjects):
    row = [reduced_features[i, :]]
    prediction = estimator.predict(row).item()
    feature_weight_dict["Base Model"] += prediction / 60

for i in range(len(sfs.k_feature_names_)):
    current_feature = sfs.k_feature_names_[i]
    feature_weight_dict[current_feature] = 0
    reduced_features_copy = reduced_features.copy()
    reduced_features_copy = change_feature_values(reduced_features_copy, feature_index=i, change_amount=-1)
    # reduced_features_copy = np.delete(reduced_features_copy, i, 1)
    # estimator.fit(reduced_features, classifier)
    for j in range(60 + adding_artificial_subjects):
        row = [reduced_features_copy[j, :]]
        prediction = estimator.predict(row).item()
        feature_weight_dict[current_feature] += prediction / 60


print("\n\nGetting weights, so to speak")
print(feature_weight_dict)

base_val = feature_weight_dict["Base Model"]
feature_weight_dict = {}
feature_weight_dict["Base Model"] = base_val

for i in range(len(sfs.k_feature_names_)):
    current_feature = sfs.k_feature_names_[i]
    feature_weight_dict[current_feature] = 0
    reduced_features_copy = reduced_features.copy()
    reduced_features_copy = change_feature_values(reduced_features_copy, feature_index=i, change_amount=1)
    # reduced_features_copy = np.delete(reduced_features_copy, i, 1)
    # estimator.fit(reduced_features, classifier)
    for j in range(60 + adding_artificial_subjects):
        row = [reduced_features_copy[j, :]]
        prediction = estimator.predict(row).item()
        feature_weight_dict[current_feature] += prediction / 60

print("\n\nGetting weights, so to speak")
print(feature_weight_dict)

''' Neural Network '''
# holdout_attributes, holdout_classifiers = ml.add_artificial_subjects(holdout_attributes, holdout_classifiers, doing_holdouts=True, feature_count=feature_count, additional_subject_count=20, original_subject_count=subject_count)
# NN.NN_model(attributes, classifier, holdout_attributes=holdout_attributes, holdout_classifiers=holdout_classifiers, input_dim=feature_count, subject_count=60+adding_artificial_subjects, artificial_count=adding_artificial_subjects)