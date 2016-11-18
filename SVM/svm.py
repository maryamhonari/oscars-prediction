from __future__ import division
from sklearn import svm
import csv
import numpy as np
import math
from sklearn import preprocessing
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from preprocessor import DataPreprocessor
from enum import Enum

class Label(Enum):
    Nominee = 0
    Winner = 1
    NumOfAwards = 2

#converts a csv file to 2D array
def csvToArray(filename):
    ret = []
    with open(filename) as x:
        entryreader = csv.reader(x, delimiter=',')
        for row in entryreader:
            ret.append(row)
    return ret

def writeCSV(data, header, fileName):
    all = []
    header = np.array(header).tolist()
    data = np.array(data).tolist()

    all.append(header)
    for i in range(len(data)):
        all.append(data[i])

    with open(fileName + '.csv', 'wb') as f:
        writer = csv.writer(f)
        writer.writerows(all)

def getCols(labelOfInterest=0):

    """

    :param labelOfInterest: #0 means nominee , 1 means winner (for best picture), 2 means number of awards
    :return:
    """

    # feature selection based on correlation values
    correlation = csvToArray("feature_correlation_results.csv")

    cols = []
    colsNames = []

    threshold = 0.1
    for i in range(1, len(correlation)):
        curName = correlation[i][0]
        if labelOfInterest == Label.NumOfAwards and ('Won' in curName):  # we should ignore this when predicting num of awards
            threshold = 0.1
            continue

        if labelOfInterest == Label.Winner and ('Won' in curName):
            threshold = 0.1
            continue

        if labelOfInterest == Label.Nominee and (('Won' in curName)):
            threshold = 0.1
            continue

        if curName in feature_names:
            if math.fabs(float(correlation[i][1 + labelOfInterest.value])) > threshold:
                cols.append(feature_names.index(curName))
                colsNames.append(curName)

    print 'favoritCols = ', len(cols)

    return cols

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

train_years = np.copy(feat_train[:, feature_names.index('title_year')])
test_years = np.copy(feat_test[:, feature_names.index('title_year')])

print test_years

#
# scaling the feature columns
# don't scale 'original row' feature
for i in range(1, len(feat_train[0])):
    feat_train[:, i] = preprocessing.scale(feat_train[:, i])

for i in range(1, len(feat_test[0])):
    feat_test[:, i] = preprocessing.scale(feat_test[:, i])

labelOfInterest = Label.Winner

###=================== SVR =======================
cols = getCols(Label.NumOfAwards)

reg = svm.SVR(C=0.005, kernel='poly', degree=3, max_iter=1000000)

predicted = cross_validation.cross_val_predict(reg, feat_train[:, cols], label_train[:, Label.NumOfAwards.value], cv=10)

reg.fit(feat_train[:, cols], label_train[:, Label.NumOfAwards.value])

predicted = reg.predict(feat_test[:, cols])

print 'SVR cross_val_predict = ', r2_score(label_test[:, Label.NumOfAwards.value], predicted)

###=================== SVC =======================
cols = getCols(labelOfInterest)

clf = svm.SVC(C=1.5, kernel='poly', degree=3, max_iter=1000000)

predicted = cross_validation.cross_val_predict(clf, feat_train[:, cols], label_train[:, labelOfInterest.value], cv=10)

clf.fit(feat_train[:, cols], label_train[:, labelOfInterest.value])

predicted = clf.predict(feat_test[:, cols])
svm_distances = clf.decision_function(feat_test[:, cols])


#select one from a bunch
winners = [-1] * 2020
distances = [-10000] * 2020
reality = [-1] * 2020

for i in range(len(predicted)):
    year = int(test_years[i])
    val = svm_distances[i]
    if label_test[i][labelOfInterest.value] == 1: # this is the id of real winner
        reality[year] = int(feat_test[i][0])
    if val > distances[year]:
        distances[year] = val
        winners[year] = int(feat_test[i][0])


total = 0
correct = 0

for i in range(1928, 2016):
    if winners[i] != -1:
        total += 1
        if winners[i] == reality[i]:
            correct += 1
        print 'year %d winner guess is %d and reality is %d' % (i, winners[i], reality[i])


print 'correct percent = ', correct/total

has_year = [0] * 2020
winners = [-1] * 2020
nominees_count = [0] * 2020


for i in range(len(feat_train)):
    year = int(train_years[i])
    has_year[year] = 1
    if label_train[i][0] == 1:
        nominees_count[year] += 1
    if label_train[i][labelOfInterest.value] == 1:
        winners[year] = int(feat_train[i][0])

for i in range(2020):
    if has_year[i] == 1:
        print 'winner guess on year %d is %d and #nominees = %d' % (i, winners[i], nominees_count[i])


print 'SVC cross_val_predict = ', accuracy_score(label_test[:, labelOfInterest.value], predicted)

# for i in range(len(feat_test)):
#     print feat_test[i][0]

# sum = 0
# sum_true = 0
# for i in range(len(label_test)):
#     if label_test[i][labelOfInterest.value] == 1:
#         sum += 1
#         if predicted[i] == 1:
#             sum_true += 1
#         print '%d: \t t=%d p=%d' % (feat_test[i][0], label_test[i][labelOfInterest.value], predicted[i])
#
# print 'total = %d, percent = %f' % (sum, sum_true/sum)
