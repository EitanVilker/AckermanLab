import parse_csv as p
import nn_functions as nn
import cox_based_model as cox

testing_binary_classifier = False
testing_large_data_set = True

if testing_large_data_set:
    filename = '41591_2018_161_MOESM3_ESM.csv'
else:
    filename = '41591_2018_161_MOESM4_ESM.csv'

# split into input and output variables
if testing_large_data_set:
    attributes, classifier1, classifier2, classifier3, classifier4, averages, standard_deviations = p.parse_csv(185, 60, filename=filename)
    classifier2 = classifier2.tolist()
else:
    attributes, classifier1 = p.parse_csv_small(filename, 4, 18)
    classifier2 = None
    classifier3 = None
    classifier4 = None


# Ignore the wall of commented out blocks unless you wish to test, I have coded up many different types of models
# and want to be able to execute them quickly. Simpler models can be run just using the functions in nn_functions.py

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
# nn.linear_regression(attributes, classifier2, test_count=1, classifier_type=2)


''' Random Projections '''

# Gaussian
# nn.gaussian_random_projection(attributes, classifier2, classifier_type=2, attributes_wanted=4, prediction_method=nn.ridge_regression)

# Sparse
# nn.sparse_random_projection(attributes, classifier2, classifier_type=2, prediction_method=nn.linear_regression)


''' Cox based 3-step feature selection '''

# Switch features and subjects for the purpose of clustering
features = attributes.transpose()

clustering, labels = cox.identify_clusters(features)

print(clustering)

print("params: ")
print(clustering.get_params())
print("\nClusters: ")
print(labels)

best_features = cox.get_best_features_by_pearson(labels, features, classifier3)
print("\nBest features: ")
print(best_features)

for i in range(185):
    if attributes.columns.values[i] not in best_features:
        attributes.drop(attributes.columns.values[i], axis=1)

nn.lasso_lars(attributes, classifier3)