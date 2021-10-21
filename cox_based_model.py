import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import evaluate as e
import parse_csv as p

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn import linear_model, random_projection
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, train_test_split, GridSearchCV
from sklearn.feature_selection import RFECV
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from scipy.stats import pearsonr

''' Function that takes a separated pandas dataframe of attributes and classifiers and 
runs a linear regression model '''
def linear_regression(attributes, classifier, test_size=0.2, test_count=100, classifier_type=3):

    accuracy = 0
    total_predictions = 0
    loss_over_samples = 0

    for state in range(test_count):

        x_train, x_test, y_train, y_test = train_test_split(attributes, classifier, test_size=test_size, random_state=state)
        clf = linear_model.LinearRegression().fit(x_train, y_train)
        predictions = clf.predict(x_test)
        answers = y_test
        # answers = y_test.tolist()

        loss, successes, prediction_count = e.evaluate(predictions, answers)
        accuracy += successes
        loss_over_samples += loss
        total_predictions += prediction_count

        # print("ANSWERS: \n")
        # print(answers)
        # print("PREDICTIONS: \n")
        # print(predictions)

    accuracy /= total_predictions
    loss_over_samples /= total_predictions

    print("\nACCURACY: ")
    print(accuracy)
    print("\nLOSS OVER SAMPLES:")
    print(loss_over_samples)


''' Function that takes a separated pandas dataframe of attributes and classifiers and 
runs a tenfold LASSO regression model '''
def lasso_regression(attributes, classifier, test_size=0.2, test_count=100, classifier_type=3):
    
    accuracy = 0
    total_predictions = 0
    loss_over_samples = 0

    l2_val=np.logspace(-4,15,num=10)
    folds = KFold(10)
    l2_penalty_mse=[]
    best_mse=None
    best_l2=None
    best_model=None
    
    for l2 in l2_val:
        mse=0
        for train_index, test_index in folds.split(attributes):
            lasso = linear_model.Lasso(l2, tol=0.001)
            model = lasso.fit(np.array(attributes)[train_index], np.array(classifier)[train_index])
            y_pred = model.predict(np.array(attributes)[test_index])
            mse += np.sum((np.array(classifier)[test_index] - y_pred)**2)
            
        mse/=10
        l2_penalty_mse.append(mse)
        if (best_mse == None or mse < best_mse):
            best_mse = mse
            best_l2 = l2
            best_model = model

    predictions = best_model.predict(attributes)
    answers = classifier
    loss, successes, prediction_count = e.evaluate(predictions, answers)
    accuracy += successes
    loss_over_samples += loss
    total_predictions += prediction_count
    accuracy /= total_predictions
    loss_over_samples /= total_predictions

    print(answers)
    print(predictions)

    print("\nACCURACY: ")
    print(accuracy)
    print("\nLOSS OVER SAMPLES:")
    print(loss_over_samples)

    return l2_penalty_mse, best_l2, best_model, importance

''' Function that takes a separated pandas dataframe of attributes and classifiers and 
runs a LASSO regression model to generate weights for the utility of each variable '''
def grid_search_lasso(attributes, classifier, test_size=0.2, test_count=100):

    for state in range(test_count):
        pipeline = Pipeline([('scaler', StandardScaler()), ('model',linear_model.Lasso())])
        search = GridSearchCV(pipeline, {'model__alpha':np.arange(0.1,10,0.1)}, cv=10, scoring="neg_mean_squared_error",verbose=3)
        x_train, x_test, y_train, y_test = train_test_split(attributes, classifier, test_size=test_size, random_state=state)
        search.fit(x_train, y_train)

    best_alpha = search.best_params_
    coefficients = search.best_estimator_.named_steps['model'].coef_
    importance = np.abs(coefficients)
    # surviving_features = np.array(attributes)[importance > 0]

    return best_alpha, coefficients, importance

''' Function that breaks attributes into clusters based on a certain number of clusters and an affinity, 
which can be manhattan, l1, l2, etc. '''
def identify_clusters(attributes, clusters_wanted=9, affinity="euclidean"):
    clustering = AgglomerativeClustering(n_clusters=clusters_wanted, affinity=affinity, linkage='ward')
    a = clustering.fit(attributes)
    b = clustering.fit_predict(attributes)
    return a, b

''' Function to get the Pearson correlation coefficients for each feature to pick the best ones '''
def get_best_features_by_pearson(labels, features, classifier, correlation_threshold=0.7, cluster_count=9):

    best_features = []

    # Initialize 2D matrix of clusters
    cluster_matrix = []
    for i in range(cluster_count):
        cluster_matrix.append([])
    
    # Add features to corresponding clusters within the matrix
    for i in range(len(labels)):
        # cluster_matrix[labels[i]].append(features[i])
        cluster_matrix[labels[i]].append(features.index.values[i])

    # Examine each cluster and get best feature
    for i in range(cluster_count):
        current_best_feature = None
        current_best_correlation = 0
        for j in range(len(cluster_matrix[i])):
            # Get pearson coefficient for this feature
            feature_values = features.loc[cluster_matrix[i][j]]
            correlation_coefficient, p_value = pearsonr(feature_values, classifier)
            if abs(correlation_coefficient) > current_best_correlation:
                current_best_feature = cluster_matrix[i][j]
                current_best_correlation = correlation_coefficient
        if current_best_feature is not None:
            best_features.append(current_best_feature)

    # TODO Figure out pairwise correlations
            
    return best_features


def recursive_feature_selection(attributes, classifier, estimator=None):

    if estimater is None:
        estimator = SVR(kernel="linear")
    selector = RFECV(estimator, step=1, cv=5)
    selector = selector.fit(attributes, classifier)