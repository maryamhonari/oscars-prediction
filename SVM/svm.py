from __future__ import division
from sklearn import svm
import csv
import numpy as np
import time
import math
from sklearn.metrics import explained_variance_score
from sklearn import preprocessing
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score


#converts a csv file to 2D array
def csvToArray(filename):
    ret = []
    with open(filename) as x:
        entryreader = csv.reader(x, delimiter=',')
        for row in entryreader:
            ret.append(row)
    return ret

def getTrainTestRowsAndCols(labelOfInterest, titleYearIdx):
    """

    :param labelOfInterest: #0 means nominee , 1 means winner (for best picture), 2 means number of awards
    :return:
    """
    trainRows = []
    testRows = []

    for i in range(len(features)):
        if labelOfInterest != 1:
            if int(math.floor(features[i][titleYearIdx])) % 4 == 0:
                testRows.append(i)
            else:
                trainRows.append(i)
        else:
            if labels[i][0] == 1:
                if int(math.floor(features[i][titleYearIdx])) % 4 == 0:
                    testRows.append(i)
                else:
                    trainRows.append(i)

    # prints percentage of train and test
    print len(trainRows) / len(features), len(testRows) / len(features)

    # feature selection based on correlation values
    correlation = csvToArray("feature_correlation_results.csv")

    favoriteCols = []
    favoriteColsNames = []

    threshold = 0.1
    for i in range(1, len(correlation)):
        if labelOfInterest == 2 and ('Won' in correlation[i][0]):  # we should ignore this when predicting num of awards
            threshold = 0.1
            continue

        if labelOfInterest == 1 and ('Won' in correlation[i][0]):
            threshold = 0.1
            continue

        if labelOfInterest == 0 and (('Won' in correlation[i][0])):
            threshold = 0.1
            continue

        if correlation[i][0] in featIdxMap:
            if math.fabs(float(correlation[i][1 + labelOfInterest])) > threshold:
                # print correlation[i][0]
                favoriteCols.append(featIdxMap[correlation[i][0]])
                favoriteColsNames.append(correlation[i][0])

    print 'favoritCols = ', len(favoriteCols)

    print favoriteColsNames

    # making test set half positive and half negative
    # removes some of negative instances
    # positive could mean that the instance has been nominated for OR has won best picture!

    return trainRows, testRows, favoriteCols


features = csvToArray("features.csv")
featNames = features[0]
features = features[1:]
print len(features), len(features[0])

labels = csvToArray("labels.csv")
labelNames = labels[0]
labels = labels[1:]
print len(labels), len(labels[0])

labels = np.array(labels).astype(int)
features = np.array(features).astype(float)

featIdxMap = dict()
for i in range(len(featNames)):
    featIdxMap[featNames[i]] = i

#getting title_year column number
titleYearIdx = -1
for i in range(len(featNames)):
    if featNames[i] == 'title_year':
        titleYearIdx = i
print 'year index = ', titleYearIdx

labelOfInterest = 1

trainRowsReg, testRowsReg, favColsReg = getTrainTestRowsAndCols(2, titleYearIdx)
trainRows, testRows, favCols = getTrainTestRowsAndCols(labelOfInterest, titleYearIdx)

for i in range(len(features[0])):
    if (i in favCols) or (i in favColsReg):
        features[:, i] = preprocessing.scale(features[:, i])

reg = svm.SVR(C=0.005, kernel='poly', degree=3, max_iter=1000000)

predicted = cross_validation.cross_val_predict(reg, (features[trainRowsReg, :])[:, favColsReg], labels[trainRowsReg, 2], cv=10)
print 'SVR cross_val_predict = ', r2_score(labels[trainRowsReg, 2], predicted)

# scores = cross_validation.cross_val_score(reg, (features[trainRowsReg, :])[:, favColsReg], labels[trainRowsReg, 2], cv=10)
# print scores, 'avg = ', sum(scores)/10

###=================== SVC =======================

clf = svm.SVC(C=1.5, kernel='poly', degree=3, max_iter=1000000)

predicted = cross_validation.cross_val_predict(clf, (features[trainRows, :])[:, favCols],
                                               labels[trainRows, labelOfInterest], cv=10)

print 'SVC cross_val_predict = ', accuracy_score(labels[trainRows, labelOfInterest], predicted)

sum = 0
sum_true = 0
for h in range(len(trainRows)):
    i = trainRows[h]
    if labels[i][labelOfInterest] == 0:
        sum += 1
        if predicted[h] == 0:
            sum_true += 1
        print '%d: \t t=%d p=%d' % (i, labels[i][labelOfInterest], predicted[h])

print 'total = %d, percent = %f' % (sum, sum_true/sum)

# scores = cross_validation.cross_val_score(clf, (features[trainRows, :])[:, favCols], labels[trainRows, labelOfInterest], cv=10)
# print scores, 'avg = ', sum(scores)/10