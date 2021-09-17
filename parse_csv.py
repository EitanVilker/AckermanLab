import pandas as pd
import numpy as np
from sklearn import preprocessing

def parse_csv(dim_x, dim_y, filename=False, data=False):

    if filename != False:
        if data == False:
            data = pd.read_csv(filename)

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
    
    separate_into_buckets(classifier3)
            
    return scaled_attributes, classifier1, classifier2, classifier3, classifier4, averages, standard_deviations

def separate_into_groups(filename):

    data = pd.read_csv(filename)
    group_a = data.copy()
    group_b = data.copy()
    group_c = data.copy()

    length = len(data)
    for i in range(length):

        group_value = data['Group'][i]

        if group_value == "IM239":
            group_b.drop(i)
            group_c.drop(i)

        elif group_value == "IM mosaic":
            group_a.drop(i)
            group_c.drop(i)

        elif group_value == "AE239":
            group_a.drop(i)
            group_b.drop(i)

    return group_a, group_b, group_c


def separate_into_buckets(classifier3):

    # blank, bins = pd.qcut(classifier3, 4, retbins=True) # Identify the quartile ranges
    # print(bins)
    for i in range(len(classifier3)):
        if classifier3[i] <= 2:
            classifier3[i] = 0

        elif classifier3[i] <= 4.5:
            classifier3[i] = 1

        elif classifier3[i] <= 12:
            classifier3[i] = 2

        else:
            classifier3[i] = 3

def predict_raw_data(model, using_nn, averages, standard_deviations, new_data):

    for i in range(len(new_data)):
        new_data[i] = (new_data[i] - averages[i]) / standard_deviations[i]

    if using_nn:
        print(model.predict_classes([new_data]))
    else:
        print(model.predict([new_data]))