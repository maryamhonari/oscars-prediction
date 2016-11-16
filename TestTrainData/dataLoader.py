import csv
import numpy as np
from sklearn import preprocessing

#
# converts a csv file to 2D array
def csvToArray(filename):
    ret = []
    with open(filename) as x:
        entryreader = csv.reader(x, delimiter=',')
        for row in entryreader:
            ret.append(row)
    return ret

feat_train = csvToArray('feat_train.csv')
feat_test = csvToArray('feat_test.csv')
label_train = csvToArray('label_train.csv')
label_test = csvToArray('label_test.csv')

#
# The first rows are names of columns
feature_names = feat_train[0]
label_names = label_train[0]

#
# These lines remove the column names and turn lists to numpy arrays
# If you need lists, use for example data_train.tolist() to convert them back to lists
feat_train = np.array(feat_train)[1:, :].astype(float)
feat_test = np.array(feat_test)[1:, :].astype(float)
label_train = np.array(label_train)[1:, :].astype(int)
label_test = np.array(label_test)[1:, :].astype(int)


####################### Save important features before scaling ###########
train_years = np.copy(feat_train[:, feature_names.index('title_year')])
test_years = np.copy(feat_test[:, feature_names.index('title_year')])
print train_years, test_years

################# Do scaling if you need it ###################
# scaling the feature columns
# Don't scale 'original row' feature
for i in range(1, len(feat_train[0])):
    feat_train[:, i] = preprocessing.scale(feat_train[:, i])

for i in range(1, len(feat_test[0])):
    feat_test[:, i] = preprocessing.scale(feat_test[:, i])


# Just printing
print '-> ', feature_names
print '-> ', label_names

################## Your code goes here #########################