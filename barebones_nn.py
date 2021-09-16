# first neural network with keras tutorial
import numpy
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
import parse_csv as p
import parse_csv_small as ps
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn import linear_model
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import matplotlib.pyplot as plt

testing_binary_classifier = False
testing_large_data_set = True

if testing_large_data_set:
    filename = '41591_2018_161_MOESM3_ESM.csv'
else:
    filename = '41591_2018_161_MOESM4_ESM.csv'

# split into input (X) and output (y) variables
if testing_large_data_set:
    attributes, classifier1, classifier2, classifier3, classifier4, blank1, blank2 = p.parse_csv(filename, 185, 60)
    classifier2 = classifier2.tolist()
else:
    attributes, classifier1 = ps.parse_csv_small(filename, 4, 18)
    classifier2 = None
    classifier3 = None
    classifier4 = None

accuracy = 0
total_predictions = 0
loss_over_samples = 0

pca = PCA(0.95)
pca.fit(attributes)
attributes = pca.transform(attributes)

# principalComponents = pca.fit_transform(classifier3)


for state in range(1):
    x_train, x_test, y_train, y_test = train_test_split(attributes, classifier2, test_size=0.2, random_state=state)
    # clf = linear_model.LogisticRegression(random_state=0).fit(x_train, y_train)
    # clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    # clf = clf.fit(x_train, y_train)
    # clf = linear_model.LinearRegression().fit(x_train, y_train)
    clf = linear_model.Ridge(alpha=.1).fit(x_train, y_train)
    predictions = clf.predict(x_test)
    answers = y_test
    # answers = y_test.tolist()

    prediction_length = len(predictions)
    for i in range(prediction_length):
        total_predictions += 1
        if predictions[i] == answers[i]:
            accuracy += 1
            loss_over_samples += 0
        else:
            if testing_binary_classifier:
                loss_over_samples += round(abs(predictions[i] - answers[i]))
            else:
                print(predictions[i] - answers[i])
                print(round(predictions[i] - round(answers[i])))
                loss_over_samples += round(abs(predictions[i] - answers[i]))
    
    print("ANSWERS: \n")
    print(answers)
    print("PREDICTIONS: \n")
    print(predictions)

accuracy /= total_predictions
loss_over_samples /= total_predictions

print("\nACCURACY: ")
print(accuracy)
print("\nLOSS OVER SAMPLES:")
print(loss_over_samples)