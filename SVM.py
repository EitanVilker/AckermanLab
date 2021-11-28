from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

small_set = True
testing_binary_classifier = False

if small_set:
    filename = "41591_2018_161_MOESM4_ESM.csv"
else:
    filename = "41591_2018_161_MOESM3_ESM.csv"

data = pd.read_csv(filename)

if small_set:
    attributes = data.drop('Primate ID', axis=1)
    attributes = attributes.drop('Challenges', axis=1)
    classifier2 = data['Challenges']

else:
    attributes = data.drop('Group', axis=1)
    attributes = attributes.drop('Time to Infection', axis=1)
    attributes = attributes.drop('Resisted Infection?', axis=1) # Classifier 1
    attributes = attributes.drop('Log Peak VL', axis=1) # Classifier 2
    attributes = attributes.drop('Log Set Pt VL', axis=1)
    classifier1 = data['Resisted Infection?']
    classifier2 = data['Log Peak VL']

if testing_binary_classifier:
    X_train, X_test, y_train, y_test = train_test_split(attributes, classifier1, test_size = 0.20)
else:
    X_train, X_test, y_train, y_test = train_test_split(attributes, classifier2, test_size = 0.20)

svclassifier = SVC(kernel='sigmoid')
svclassifier.fit(X_train, y_train)

y_pred = svclassifier.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

# sheet_ranges = workbook['range names']

# X = [[0, 0], [1, 1]]
# y = [0, 1]
# clf = svm.SVC()
# clf.fit(X, y)

print(svclassifier.predict([[8.4,17.52,141045.5,973.5], [11.8,28.46,163886.5,1995.5], [8, 7, 0.4, 2]]))