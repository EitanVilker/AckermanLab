import pandas as pd
import numpy as np

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
            attributes.iat[j, i] = (attributes.iat[j, i] - averages[i]) / standard_deviations[i]
            
    return attributes, classifier1
