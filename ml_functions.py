import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

from evaluate import evaluate
import parse_csv as p

from sklearn.model_selection import cross_val_score, KFold
from sklearn import linear_model, random_projection
from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.feature_selection import RFECV
from sklearn.neighbors import KNeighborsClassifier, kneighbors_graph
from sklearn.svm import SVR, SVC
from sklearn.preprocessing import StandardScaler
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from scipy.stats import pearsonr, spearmanr, kendalltau


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
        # answers = y_test
        answers = y_test.tolist()

        loss, successes, prediction_count = evaluate(predictions, answers)
        accuracy += successes
        loss_over_samples += loss
        total_predictions += prediction_count

        # print(successes/prediction_count)
        print(-1 * mean_squared_error(answers, predictions))

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
        # answers = y_test
        answers = y_test.tolist()

        loss, successes, prediction_count = evaluate(predictions, answers)
        accuracy += successes
        loss_over_samples += loss
        total_predictions += prediction_count

        print(successes/prediction_count)

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

''' Function to run a basic SVM model '''
def SVM(attributes, classifier, test_size=0.2, test_count=100):

    accuracy = 0
    total_predictions = 0
    loss_over_samples = 0

    for state in range(test_count):

        x_train, x_test, y_train, y_test = train_test_split(attributes, classifier, test_size=test_size, random_state=state)
        clf = SVC(kernel='linear').fit(x_train, y_train)
        predictions = clf.predict(x_test)
        # answers = y_test
        answers = y_test.tolist()

        loss, successes, prediction_count = evaluate(predictions, answers, nominal=False)
        accuracy += successes
        loss_over_samples += loss
        total_predictions += prediction_count

        # print(successes/prediction_count)
        print(-1 * mean_squared_error(answers, predictions))

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

''' Function to run a simple KNN model '''
def KNN(attributes, classifier, test_size=0.2, test_count=1, classifier_type=3):
    accuracy = 0
    total_predictions = 0
    loss_over_samples = 0

    for state in range(test_count):

        x_train, x_test, y_train, y_test = train_test_split(attributes, classifier, test_size=test_size, random_state=state)
        clf = KNeighborsClassifier(n_neighbors=5).fit(x_train, y_train)
        predictions = clf.predict(x_test)
        # answers = y_test
        answers = y_test.tolist()

        loss, successes, prediction_count = evaluate(predictions, answers, nominal=True)
        accuracy += successes
        loss_over_samples += loss
        total_predictions += prediction_count

        print(successes/prediction_count)

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

''' Function to run SGD model '''
def SGD(attributes, classifier, test_size=0.2, test_count=1, classifier_type=3):

    accuracy = 0
    total_predictions = 0
    loss_over_samples = 0

    for state in range(test_count):

        x_train, x_test, y_train, y_test = train_test_split(attributes, classifier, test_size=test_size, random_state=state)
        clf = make_pipeline(StandardScaler(), linear_model.SGDClassifier(max_iter=1000, tol=1e-3)).fit(x_train, y_train)
        predictions = clf.predict(x_test)
        # answers = y_test
        answers = y_test.tolist()

        loss, successes, prediction_count = evaluate(predictions, answers, nominal=True)
        accuracy += successes
        loss_over_samples += loss
        total_predictions += prediction_count

        print(successes/prediction_count)

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
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(50, 50), random_state=state)
        clf = clf.fit(x_train, y_train)
        predictions = clf.predict(x_test)
        if classifier_type == 1:
            answers = y_test
        else:
            answers = y_test.tolist()

        loss, successes, prediction_count = evaluate.evaluate(predictions, answers)
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

        loss, successes, prediction_count = evaluate(predictions, answers, nominal=False)
        accuracy += successes
        loss_over_samples += loss
        total_predictions += prediction_count

        # print(loss/prediction_count)
        print(mean_squared_error(answers, predictions))

        # print("ANSWERS: \n")
        # print(answers)
        # print("PREDICTIONS: \n")
        # print(predictions)

    accuracy /= total_predictions
    loss_over_samples /= total_predictions

    print(answers)
    print(predictions)

    # print("\nACCURACY: ")
    # print(accuracy)
    # print("\nLOSS OVER SAMPLES:")
    # print(loss_over_samples)
    print("\nMSE:")
    print(mean_squared_error(answers, predictions))

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

        loss, successes, prediction_count = evaluate(predictions, answers)
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

''' Function to transform attributes using PCA. 
Works but oddly makes results slightly worse when plugged into another model '''
def pca_transform_attributes(attributes, classifier):
    pca = PCA(2)
    # pca.fit(attributes)
    # attributes = pca.transform(attributes)
    projected = pca.fit_transform(attributes)
    # principalComponents = pca.fit_transform(classifier)
    plt.scatter(projected[:, 0], projected[:, 1], edgecolor='none', alpha=0.5, cmap=plt.cm.get_cmap('Spectral', 10))
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.colorbar()
    plt.show()
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

        loss, successes, prediction_count = evaluate(predictions, answers)
        accuracy += successes
        loss_over_samples += loss
        total_predictions += prediction_count

        # print("\n\nCoefficients: ")
        # print(clf.coef_path_)
        # print("feature names: ")
        # print(clf.feature_names_in_)
        path = clf.coef_.tolist()
        coefficient_nums = []
        for i in range(len(path)):
            if abs(path[i]) > 0.001:
                coefficient_nums.append(i)
        # print("coefficient numbers: ")
        # print(coefficient_nums)
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

    return clf

''' Function that runs a Gaussian random projection on the features before passing them into an ML function '''
def gaussian_random_projection(attributes, classifier, attributes_wanted=10, test_size=0.2, test_count=100, classifier_type=3, prediction_method=linear_regression):

    transformer = random_projection.GaussianRandomProjection(n_components=attributes_wanted)
    adjusted_attributes = transformer.fit_transform(attributes)
    prediction_method(adjusted_attributes, classifier, classifier_type=classifier_type, test_count=test_count, test_size=test_size)

''' Function that runs a sparse random projection on the features before passing them into an ML function '''
def sparse_random_projection(attributes, classifier, attributes_wanted=10, test_size=0.2, test_count=100, classifier_type=3, prediction_method=linear_regression):

    transformer = random_projection.SparseRandomProjection(n_components=attributes_wanted)
    adjusted_attributes = transformer.fit_transform(attributes)
    prediction_method(adjusted_attributes, classifier, classifier_type=classifier_type, test_count=test_count, test_size=test_size)

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
    loss, successes, prediction_count = evaluate(predictions, answers)
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

    return l2_penalty_mse, best_l2, best_model

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
    connectivity = kneighbors_graph(attributes, n_neighbors=10, include_self=False)
    clustering = AgglomerativeClustering(n_clusters=clusters_wanted, affinity=affinity, linkage='ward', connectivity=connectivity)
    # clustering = SpectralClustering(n_clusters=clusters_wanted, affinity="nearest_neighbors", assign_labels="kmeans")
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
            # correlation_coefficient, p_value = spearmanr(feature_values, classifier)
            if abs(correlation_coefficient) > current_best_correlation:
                current_best_feature = cluster_matrix[i][j]
                current_best_correlation = correlation_coefficient
        if current_best_feature is not None:
            best_features.append(current_best_feature)

    print("Best features before pairwise check")
    print(best_features)

    # Remove pairwise correlations
    print("\nRemoving pairwise correlations")
    while True:
        go_again = False
        for feature in best_features:
            if not go_again:
                for other_feature in best_features:
                    if feature != other_feature and not go_again:
                        correlation_coefficient, p_value = pearsonr(features.loc[feature], features.loc[other_feature])
                        # correlation_coefficient, p_value = spearmanr(features.loc[feature], features.loc[other_feature])
                        if correlation_coefficient > correlation_threshold:
                            best_features.remove(feature)
                            go_again = True
        if not go_again:
            break

    return best_features

''' Function to split attributes into ten folds, with each fold containing the indices of two subjects for
each arm in one part and the remaining ones in the second part '''
def split_into_folds(group_count=3, subject_count=60, fold_count=10):

    random.seed(None)
    folds = []

    group_a, group_b, group_c = [], [], []
    for i in range(subject_count // group_count):
        group_a.append(i)
        group_b.append(i + subject_count // group_count)
        group_c.append(i + 2 * subject_count // group_count)

    # For each of the folds
    for i in range(fold_count):

        # Initialize empty two-part fold
        fold = []
        fold.append([])
        fold.append([])

        # Add test animals to second part of each fold
        group_a_copy = group_a.copy()
        for j in range(2):
            rand = random.randint(0, len(group_a_copy) - 1)
            fold[1].append(group_a[rand])
            group_a_copy.pop(rand)

        group_b_copy = group_b.copy()
        for j in range(2):
            rand = random.randint(0, len(group_b_copy) - 1)
            fold[1].append(group_b[rand])
            group_b_copy.pop(rand)

        group_c_copy = group_c.copy()
        for j in range(2):
            rand = random.randint(0, len(group_c_copy) - 1)
            fold[1].append(group_c[rand])
            group_c_copy.pop(rand)

        # Add remaining animals to training part of the fold
        for j in group_a_copy:
            fold[0].append(j)
        for j in group_b_copy:
            fold[0].append(j)
        for j in group_c_copy:
            fold[0].append(j)

        fold[1].sort()
        folds.append(fold)

    return folds

''' Function to run a cox survival analysis. Doesn't work yet, will likely remove '''
def survival_analysis(attributes, classifier, fold_count=10, cluster_count=9):

    # Get clusters
    features = attributes.transpose()
    clustering, labels = identify_clusters(features)

    # Get folds
    folds = split_into_folds()
    for fold in folds:

        features_copy = features.copy()
        classifier_copy = classifier.copy()
        labels_copy = labels.copy()

        # Remove test set subjects from features and classifiers
        for index in reversed(range(len(fold[1]))):
            print("index: " + str(fold[1][index]))
            del classifier_copy[fold[1][index]]
            # labels_copy = np.delete(labels_copy, fold[1][index])
            features_copy.drop(fold[1][index], 1)
        
        print(features_copy)
        
        # Rank features in clusters by correlations to time-to-infection for train set
        print("labels: " + str(len(labels_copy)) + ", features: " + str(len(features_copy.columns)) + ", classifier: " + str(len(classifier_copy)))
        best_features = get_best_features_by_pearson(labels_copy, features_copy, classifier_copy)

''' Function to run recursive backward selection, running an ML model with increasing numbers of features '''
def recursive_feature_selection(attributes, classifier, estimator=None):

    if estimator is None:
        estimator = SVR(kernel="linear")
    selector = RFECV(estimator, step=1, cv=5)
    selector = selector.fit(attributes, classifier)

    return selector

''' Function to run sequential forward selection, running an ML model with increasing numbers of features '''
def sequential_feature_selection(attributes, classifier, estimator=None, test_count=100, test_size=0.2):

    if estimator is None:
        # estimator = KNeighborsClassifier(n_neighbors=5)
        # estimator = make_pipeline(StandardScaler(), linear_model.SGDClassifier(max_iter=1000, tol=1e-3, penalty='elasticnet'))
        # estimator = linear_model.Ridge(alpha=0.1, solver="saga")
        estimator = linear_model.LinearRegression()
        # estimator = SVC(kernel='linear')

    if estimator is not None:
        sfs = SFS(estimator, k_features=22, forward=True, floating=False, verbose=2,scoring='neg_mean_squared_error', cv=10)
        # sfs = SFS(estimator, k_features=22, forward=True, floating=False, verbose=2,scoring='balanced_accuracy', cv=10)
        sfs.fit(attributes, classifier)
        print(sfs.k_feature_names_)
        print(sfs.k_score_)
        print(sfs.subsets_)
        return sfs

''' Function to separate holdouts for validation testing '''
def get_holdouts(attributes, classifier, holdout_proportion=0.2, subject_count=60):
    holdout_attributes = []
    holdout_classifiers = []
    to_remove = []
    for i in range(subject_count // int(1 / holdout_proportion)):
        # Choose a random real subject index that hasn't been chosen yet
        rand = random.randint(0, subject_count - i - 1)
        while rand in to_remove:
            rand = random.randint(0, subject_count - i - 1)
        to_remove.append(rand)
        
        # Move the subject to holdouts
        holdout_attributes.append(attributes.loc[rand])
        holdout_classifiers.append(classifier.loc[rand])
        classifier.pop(rand)

    attributes = attributes.drop(to_remove)
    holdout_attributes = np.asarray(holdout_attributes).astype('float32')
    # holdout_classifiers = np.asarray(holdout_classifiers).astype('float32')
    holdout_classifiers = np.asarray(holdout_classifiers)
    subject_count -= subject_count // int(1 / holdout_proportion)

    return attributes, classifier, holdout_attributes, holdout_classifiers, subject_count

''' Function to add randomized subjects within the bounds of the min and max feature values to compensate for insufficient data '''
def add_artificial_subjects(attributes, classifier, doing_holdouts=False, holdout_proportion=0.2, additional_subject_count=60, feature_count=190, original_subject_count=60):

    for i in range(additional_subject_count):
        new_subject = []
        if not doing_holdouts:
            rand = random.randint(0, original_subject_count - 1)
        else:
            rand = random.randint(0, original_subject_count // int(1 / holdout_proportion) - 1)
            print("rand: " + str(rand))
        for j in range(feature_count):
            if doing_holdouts:
                feature_val = attributes[rand, j]
            else:
                feature_val = attributes.iloc[rand, j]
            # Assumption here is that attributes have been standardized so new features should be within 3/10 of a standard deviation
            new_subject.append(random.uniform(feature_val - 0.3, feature_val + 0.3))
        if doing_holdouts:
            print(classifier[rand])
            attributes = np.vstack((attributes, new_subject))
            classifier = np.hstack((classifier, classifier[rand]))
        else:
            attributes = attributes.append(pd.Series(new_subject, index=attributes.columns[:len(new_subject)]), ignore_index=True)
            classifier = classifier.append(pd.Series(classifier.iloc[rand]))            

    return attributes, classifier