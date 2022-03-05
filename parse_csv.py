import pandas as pd
import numpy as np
from array import array
from sklearn import preprocessing, feature_selection

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
    attributes = attributes.drop('Time to Infection', axis=1) # Classifier 4 - Challenges
    attributes = attributes.drop('Resisted Infection?', axis=1) # Classifier 1 - Resisted Infection y/n
    attributes = attributes.drop('Log Peak VL', axis=1) # Classifier 2 - Log Peak Viral Load
    attributes = attributes.drop('Log Set Pt VL', axis=1)

    classifier1 = data['Resisted Infection?']
    classifier2 = data['Log Peak VL']
    classifier3 = data['Time to Infection'] # Classifier 3 - Buckets
    classifier4 = classifier3.copy() # Copy classifier 3 before it is sorted into buckets
    classifier5 = data['Group'] # Classifier 5 - Arm Labels

    group_labels = {}
    i = 0
    for classifier in classifier5:
        if classifier not in group_labels:
            group_labels[classifier] = i
            i += 1
        classifier = group_labels[classifier]

    # Normalize attributes
    averages = attributes.mean(axis=0)
    standard_deviations = attributes.std(axis=0)
    for i in range(dim_x):
        for j in range(dim_y):
            attributes.iat[j, i] = (attributes.iat[j, i] - averages[i]) / standard_deviations[i]

    # scaler = preprocessing.StandardScaler().fit(attributes)
    # print(scaler.mean_)
    # scaled_attributes = scaler.transform(attributes)

    separate_into_buckets(classifier3, group_number)
    # convert_groups_to_ints(classifier5)

    # Assemble classifier 6 - tuples for survival analysis
    classifier6 = np.zeros(60, dtype=('bool, int32'))
    for i in range(dim_y):
        # np.append(classifier6, [classifier1[i], classifier4[i]])
        bool_convert = classifier1[i]
        if bool_convert.item(0) == 0:
            bool_convert = False
        elif bool_convert.item(0) == 1:
            bool_convert = True
        classifier6[i] = (bool_convert, classifier4[i])
        # np.append(classifier6, (bool_convert, classifier4[i]))
        # classifier6.append(np.array([classifier1[i], classifier4[i]], dtype=object))

    # print(classifier6)

    mean_animals = pd.DataFrame()
    # For each challenge count
    for i in range(1, 14):
        subject_count = 0

        mean_attributes = []
        # Create list of zeroed features
        for j in range(dim_x):
            mean_attributes.append(0)

        # For each subject
        for j in range(dim_y):
            subject_count += 1
            # If the subject has the right challenge count
            if int(classifier4[j]) == i:
                # For each feature
                for k in range(dim_x):
                    mean_attributes[k] += attributes.iat[j, k] # Double check order j, k
        
        # Divide every feature value by subject count to get average
        if subject_count != 0:
            for j in range(dim_x):
                mean_attributes[j] /= subject_count
        mean_animals[str(i)] = mean_attributes
    
    return attributes, classifier1, classifier2, classifier3, classifier4, classifier5, classifier6, averages, standard_deviations, mean_animals

''' Parses the smaller csv, MOESM4. Rarely used so may be out of date '''
def parse_csv_small(filename, dim_x, dim_y):

    data = pd.read_csv(filename)
    attributes = data.drop('Primate ID', axis=1)
    attributes = attributes.drop('Challenges', axis=1) # Classifier 1

    classifier1 = data['Challenges']

    # Normalize attributes
    averages = attributes.mean(axis=0)
    standard_deviations = attributes.std(axis=0)
    for i in range(dim_x):
        for j in range(dim_y):
            attributes.iat[i, j] = (attributes.iat[i, j] - averages[i]) / standard_deviations[i]

    return attributes, classifier1

''' Check if string can be converted to float '''
def is_float(element):
    try:
        float(element)
        return True
    except ValueError:
        return False

''' Function to help with formatting nn output so that they can easily be collected as data '''
def get_usable_data_nn(folder, file1, file2, file3, file4, file5):
    data = open(folder + "/" + file1)
    out_file1 = open(folder + "/" + file2, "w")
    out_file2 = open(folder + "/" + file3, "w")
    out_file3 = open(folder + "/" + file4, "w")
    out_file4 = open(folder + "/" + file5, "w")
    lines = data.readlines()
    for line in lines:
        count = 0
        line = line.split()

        for item in line:
            if is_float(item):
                if count == 0:
                    out_file1.write(item + "\n")
                elif count == 1:
                    out_file2.write(item + "\n")
                elif count == 2:
                    out_file3.write(item + "\n")
                elif count == 3:
                    out_file4.write(item + "\n")
                count += 1
    data.close()
    out_file1.close()
    out_file2.close()
    out_file3.close()
    out_file4.close()

''' Function to help with formatting nn output so that they can easily be collected as data '''
def get_usable_data_sfs(folder, file1, file2, file3, file4, file5):
    data = open(folder + "/" + file1)
    out_file1 = open(folder + "/" + file2, "w")
    out_file2 = open(folder + "/" + file3, "w")
    out_file3 = open(folder + "/" + file4, "w")
    out_file4 = open(folder + "/" + file5, "w")
    lines = data.readlines()
    for line in lines:
        count = 0
        line = line.split()
        if len(line) == 1:
            out_file1.write(line[0] + "\n")
        else:
            word = line[0]
            word = word[2:]
            out_file2.write(word)
            for i in range(1, len(line) - 1):
                out_file2.write("," + line[i])
            word = line[len(line) - 1]
            out_file2.write("," + word[:len(word) - 2] + "\n")

    data.close()
    out_file1.close()
    out_file2.close()
    out_file3.close()
    out_file4.close()

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

# def parse_csv_into_features(filename, dim_x, dim_y):
#     data = pd.read_csv(filename)
#     attributes = data.drop('Group', axis=1)
#     attributes = attributes.drop('Time to Infection', axis=1) # Classifier 4
#     attributes = attributes.drop('Resisted Infection?', axis=1) # Classifier 1
#     attributes = attributes.drop('Log Peak VL', axis=1) # Classifier 2
#     attributes = attributes.drop('Log Set Pt VL', axis=1)
#     attributes = attributes.copy().transpose()
#     return attributes

''' Function to create a new classifier that lumps the Time to Infection classifier into 4 "buckets."
The quartiles are not identified automatically, 
as there was no point in making that functional when it is highly unlikely new data are collected '''
def separate_into_buckets(classifier3, group_number=0):

    # blank, bins = pd.qcut(classifier3, 4, retbins=True) # Identify the quartile ranges
    # print(bins)
    # Paper uses 1-4, 5-9, 10+
    # for i in range(group_number * 20, group_number * 20 + len(classifier3)):
    #     if classifier3[i] <= 2:
    #         classifier3[i] = 0

    #     elif classifier3[i] <= 4.5:
    #         classifier3[i] = 1

    #     elif classifier3[i] <= 12:
    #         classifier3[i] = 2

    #     else:
    #         classifier3[i] = 3

    for i in range(group_number * 20, group_number * 20 + len(classifier3)):
        if classifier3[i] <= 4:
            classifier3[i] = 0
        elif classifier3[i] <= 9:
            classifier3[i] = 1
        else:
            classifier3[i] = 2

''' Function to set up classifier5, which allows for predictions of groups '''
def convert_groups_to_ints(classifier5):
    for i in range(len(classifier5)):
        if classifier5[i] == "IM239":
            classifier5[i] = 0
        elif classifier5[i] == "IM mosaic":
            classifier5[i] = 1
        elif classifier5[i] == "AE239":
            classifier5[i] = 2

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


if __name__ == "__main__":
    # get_usable_data_nn("output", "output0.csv", "output1.csv", "output2.csv", "output3.csv", "output4.csv")
    get_usable_data_sfs("output", "output0.csv", "output1.csv", "output2.csv", "output3.csv", "output4.csv")