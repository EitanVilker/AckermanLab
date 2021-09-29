import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import feature_selection


''' If filename given, returns attributes and classifiers based on the csv.
If not, uses the data object being passed in. 
This is to make it easier if only parts of the dataset are being tested, 
like when working with individual arms '''
def parse_csv(dim_x, dim_y, filename=False, data=False, group_number=0):

    if filename != False:
        if data == False:
            data = pd.read_csv(filename)
    elif data == False:
        return None
    attributes = data.drop('Group', axis=1)
    attributes = attributes.drop('Time to Infection', axis=1) # Classifier 4
    attributes = attributes.drop('Resisted Infection?', axis=1) # Classifier 1
    attributes = attributes.drop('Log Peak VL', axis=1) # Classifier 2
    attributes = attributes.drop('Log Set Pt VL', axis=1)

    classifier1 = data['Resisted Infection?']
    classifier2 = data['Log Peak VL']
    classifier3 = data['Time to Infection']
    classifier4 = classifier3.copy() # Classifier 3

    # Normalize attributes
    averages = attributes.mean(axis=0)
    standard_deviations = attributes.std(axis=0)
    # for i in range(dim_x):
    #     for j in range(dim_y):
    #         attributes.iat[j, i] = (attributes.iat[j, i] - averages[i]) / standard_deviations[i]

    scaler = preprocessing.StandardScaler().fit(attributes)
    # print(scaler.mean_)
    scaled_attributes = scaler.transform(attributes)
    
    separate_into_buckets(classifier3, group_number)
            
    return scaled_attributes, classifier1, classifier2, classifier3, classifier4, averages, standard_deviations

''' Function to separate the attributes into the three individual arms '''
def separate_into_groups(filename):

    data = pd.read_csv(filename)
    group_a = data.copy()
    group_b = data.copy()
    group_c = data.copy()

    group_a = group_a.loc[(group_a["Group"] == "IM239")]
    group_b = group_b.loc[(group_b["Group"] == "IM mosaic")]
    group_c = group_c.loc[(group_c["Group"] == "AE239")]

    return group_a, group_b, group_c

''' Function to create a new classifier that lumps the Time to Infection classifier into 4 "buckets."
The quartiles are not identified automatically, 
as there was no point in making that functional when it is highly unlikely new data are collected '''
def separate_into_buckets(classifier3, group_number=0):

    # blank, bins = pd.qcut(classifier3, 4, retbins=True) # Identify the quartile ranges
    # print(bins)
    for i in range(group_number * 20, group_number * 20 + len(classifier3)):
        if classifier3[i] <= 2:
            classifier3[i] = 0

        elif classifier3[i] <= 4.5:
            classifier3[i] = 1

        elif classifier3[i] <= 12:
            classifier3[i] = 2

        else:
            classifier3[i] = 3

''' Function that allows a model to make predictions using raw data,
transforming it using the average and standard deviation in lieu of the StandardScaler '''
def predict_raw_data(model, using_nn, averages, standard_deviations, new_data):

    for i in range(len(new_data)):
        new_data[i] = (new_data[i] - averages[i]) / standard_deviations[i]

    if using_nn:
        print(model.predict_classes([new_data]))
    else:
        print(model.predict([new_data]))

''' Function that uses SelectKBest to pick the most useful features.
Probably strictly inferior to more complex approaches like LASSO '''
def feature_select(features, classifiers, feature_count):

    modified_features = feature_selection.SelectKBest(feature_selection.f_classif, k=feature_count).fit_transform(features, classifiers)
    return modified_features