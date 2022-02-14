# first neural network with keras tutorial
import numpy
from keras.models import Sequential
from keras.layers import Dense
import parse_csv as p
import parse_csv_small as ps
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import matplotlib.pyplot as plt

def plot_tool_test(sfs_obj, clf, x_train, y_train, x_test, y_test):
    number_features = []
    feature_list = []
    for key in sfs_obj.get_metric_dict().keys():
        number_features.append(key)
        feature_list.append(list(sfs_obj.get_metric_dict()[key]['feature_names']))

    accuracies = []

    print("\nFeature List: ")
    print(feature_list)

    for l in feature_list:

        # print(x_train.iloc[l])
        print(x_train[l])

        # x_train = x_train[l].copy()
        # x_test = x_test[l].copy()
        # clf_temp = clf
        # clf_temp.fit(x_train, y_train)
        # preds = clf_temp.predict(x_test)
        # accuracies.append(accuracy_score(y_test, preds))
    return number_features, accuracies

def plot_subsets(search_obj, x_train, y_train, x_test, y_test, clf=None, ylim=[0,1]):
    if clf == None:
        clf = search_obj.estimator
    num, acc = plot_tool_test(search_obj, clf, x_train, y_train, x_test, y_test)
    fig1 = plot_sfs(search_obj.get_metric_dict(), kind='std_dev')
    plt.ylim(ylim)
    plt.title('Sequential Forward Selection (w. StdDev)')
    plt.grid()
    plt.xticks(rotation=90)
    plt.tight_layout()
    fig1.set_figwidth(15)
    plt.plot(num, acc)
    plt.show()
    pass

testing_binary_classifier = True
testing_large_data_set = True

if testing_large_data_set:
    filename = '41591_2018_161_MOESM3_ESM.csv'
else:
    filename = '41591_2018_161_MOESM4_ESM.csv'

# split into input (X) and output (y) variables
if testing_large_data_set:
    attributes, classifier1, classifier2 = p.parse_csv(filename, 185, 60)
else:
    attributes, classifier1 = ps.parse_csv_small(filename, 4, 18)
    classifier2 = None

# define base model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(4, input_dim=4, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model

# clf = LogisticRegression(solver='lbfgs')
# x_train = attributes
# y_train = classifier1
# x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=88)

# stratkfold = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=88).split(x_train, y_train)
# cv=list(stratkfold)

# sfs = SFS(clf, 
#         k_features='best', 
#         forward=True, 
#         floating=True, 
#         verbose=0,
#         scoring='accuracy',
#         cv=cv,
#         n_jobs=-1)
# sfs.fit(x_train, y_train)

# plot_subsets(sfs, x_train, y_train, x_test, y_test, clf=clf)


# # evaluate model
# estimator = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=5, verbose=0)
# kfold = KFold(n_splits=10)
# results = cross_val_score(estimator, attributes, classifier1, cv=kfold)
# print("Baseline: %.2f (%.2f) MSE" % (results.mean(), results.std()))

# # define the keras model
# model = Sequential()
# model.add(Dense(12, input_dim=4, activation='relu'))
# model.add(Dense(8, activation='relu'))
# if testing_binary_classifier:
#     model.add(Dense(1, activation='sigmoid'))

# # compile the keras model
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the keras model on the dataset
# if testing_binary_classifier:
#     model.fit(attributes, classifier1, validation_split=0.2, epochs=150, batch_size=10)
# else:
#     model.fit(attributes, classifier2, validation_split=0.2, epochs=150, batch_size=10)

# # evaluate the keras model
# _, accuracy = model.evaluate(attributes, classifier1)
# print('Accuracy: %.2f' % (accuracy*100))

# # print('Weights: ')
# # print(model.get_weights())
# weights = model.get_weights()
# counter = 0
# for array in weights:
#     outfile = "output" + str(counter) + ".csv"
#     numpy.savetxt(outfile, array, delimiter=",")
#     counter += 1

# predictions = [[8.4,17.52,141045.5,973.5], [11.8,28.46,163886.5,1995.5], [8.3,12.81,116096.75,589.25], [10,19.33,132087.5,929.25]]

# prediction_results = model.predict_classes(predictions)
# print(prediction_results)