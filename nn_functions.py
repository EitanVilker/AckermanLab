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
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from mlxtend.feature_selection import SequentialFeatureSelector as SFS


''' Function that takes a separated pandas dataframe of attributes and classifiers and 
runs a logistic regression model. Will not work for continuous classifiers such as 
Log Peak Viral Load (classifier2) '''
def logistic_regression(attributes, classifier, classifier_type=3, test_size=0.2, test_count=100):

    accuracy = 0
    total_predictions = 0
    loss_over_samples = 0

    for state in range(test_count):

        x_train, x_test, y_train, y_test = train_test_split(attributes, classifier, test_size=test_size, random_state=state)
        clf = linear_model.LogisticRegression(random_state=0).fit(x_train, y_train)
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
runs a multi-layer perceptron model '''
def mlp(attributes, classifier, test_size=0.2, test_count=100, classifier_type=3):

    accuracy = 0
    total_predictions = 0
    loss_over_samples = 0

    for state in range(test_count):

        x_train, x_test, y_train, y_test = train_test_split(attributes, classifier, test_size=test_size, random_state=state)
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
        clf = clf.fit(x_train, y_train)
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
runs a ridge linear regression model '''
def ridge_regression(attributes, classifier, test_size=0.2, test_count=100, classifier_type=3):

    accuracy = 0
    total_predictions = 0
    loss_over_samples = 0

    for state in range(test_count):

        x_train, x_test, y_train, y_test = train_test_split(attributes, classifier, test_size=test_size, random_state=state)
        clf = linear_model.Ridge(alpha=.1).fit(x_train, y_train)
        predictions = clf.predict(x_test)
        if classifier_type == 2:
            answers = y_test
        else:
            answers = y_test.tolist()

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

    print(answers)
    print(predictions)

    print("\nACCURACY: ")
    print(accuracy)
    print("\nLOSS OVER SAMPLES:")
    print(loss_over_samples)

''' Function that takes a separated pandas dataframe of attributes and classifiers 
and runs a linear regression model using the feature_select function in parse_csv.py '''
def lin_reg_feature_selection(attributes, classifier, feature_count, test_size=0.2, test_count=100, classifier_type=3):
    
    accuracy = 0
    total_predictions = 0
    loss_over_samples = 0

    attributes = p.feature_select(attributes, classifier, feature_count)

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

    return loss_over_samples, attributes

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
        pipeline = Pipeline([('scaler',StandardScaler()), ('model',linear_model.Lasso())])
        search = GridSearchCV(pipeline, {'model__alpha':np.arange(0.1,10,0.1)}, cv = 5, scoring="neg_mean_squared_error",verbose=3)
        x_train, x_test, y_train, y_test = train_test_split(attributes, classifier, test_size=test_size, random_state=state)
        search.fit(x_train, y_train)

    best_alpha = search.best_params_
    coefficients = search.best_estimator_.named_steps['model'].coef_
    importance = np.abs(coefficients)
    # surviving_features = np.array(attributes)[importance > 0]

    return best_alpha, coefficients, importance

''' Function to transform attributes using PCA. 
Works but oddly makes results slightly worse when plugged into another model '''
def pca_transform_attributes(attributes, classifier):
    pca = PCA(0.95)
    pca.fit(attributes)
    attributes = pca.transform(attributes)
    principalComponents = pca.fit_transform(classifier)
    return attributes

''' Function that takes a separated pandas dataframe of attributes and classifiers and 
runs a LARS regression model to generate weights for the utility of each variable '''
def lasso_lars(attributes, classifier, test_size=0.2, test_count=100, classifier_type=3):

    accuracy = 0
    total_predictions = 0
    loss_over_samples = 0

    for state in range(test_count):

        x_train, x_test, y_train, y_test = train_test_split(attributes, classifier, test_size=test_size, random_state=state)
        clf = linear_model.LassoLars(alpha=.1, normalize=False).fit(x_train, y_train)
        predictions = clf.predict(x_test)
        if classifier_type == 2:
            answers = y_test
        else:
            answers = y_test.tolist()

        loss, successes, prediction_count = e.evaluate(predictions, answers)
        accuracy += successes
        loss_over_samples += loss
        total_predictions += prediction_count

        print("\n\nCoefficients: ")
        print(clf.coef_path_)
        # print("feature names: ")
        # print(clf.feature_names_in_)
        path = clf.coef_.tolist()
        coefficient_nums = []
        for i in range(len(path)):
            if abs(path[i]) > 0.001:
                coefficient_nums.append(i)
        print("coefficient numbers: ")
        print(coefficient_nums)
        # print("ANSWERS: \n")
        # print(answers)
        # print("PREDICTIONS: \n")
        # print(predictions)

    accuracy /= total_predictions
    loss_over_samples /= total_predictions

    print(answers)
    print(predictions)

    print("\nACCURACY: ")
    print(accuracy)
    print("\nLOSS OVER SAMPLES:")
    print(loss_over_samples)


def gaussian_random_projection(attributes, classifier, attributes_wanted=10, test_size=0.2, test_count=100, classifier_type=3, prediction_method=linear_regression):

    transformer = random_projection.GaussianRandomProjection(n_components=attributes_wanted)
    adjusted_attributes = transformer.fit_transform(attributes)
    prediction_method(adjusted_attributes, classifier, classifier_type=classifier_type, test_count=test_count, test_size=test_size)

def sparse_random_projection(attributes, classifier, attributes_wanted=10, test_size=0.2, test_count=100, classifier_type=3, prediction_method=linear_regression):

    transformer = random_projection.SparseRandomProjection(n_components=attributes_wanted)
    adjusted_attributes = transformer.fit_transform(attributes)
    prediction_method(adjusted_attributes, classifier, classifier_type=classifier_type, test_count=test_count, test_size=test_size)