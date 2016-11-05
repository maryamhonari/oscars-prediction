from __future__ import division
from sklearn import svm
import csv
import numpy as np
import time
import math

#converts a csv file to 2D array
def csvToArray(filename):
    ret = []
    with open(filename) as x:
        entryreader = csv.reader(x, delimiter=',')
        for row in entryreader:
            ret.append(row)
    return ret

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

labelOfInterest = 1 #0 means nominee , 1 means winner (for best picture)

#choosing train/test instances based on title_year
trainRows = []
testRows = []

nomineeSum = 0

for i in range(len(features)):
    if labelOfInterest == 0:
        if float(features[i][titleYearIdx]) > 2010:
            testRows.append(i)
        else:
            trainRows.append(i)
    elif labelOfInterest == 1:
        if labels[i][0] == 1:
            nomineeSum += 1
            #if float(features[i][titleYearIdx]) > 2010:
            if int(math.floor(features[i][titleYearIdx])) % 4 == 0:
                testRows.append(i)
            else:
                trainRows.append(i)


#prints percentage of train and test
if labelOfInterest == 0:
    print len(trainRows) / len(features) , len(testRows) / len(features)
elif labelOfInterest == 1:
    print len(trainRows) / nomineeSum, len(testRows) / nomineeSum


#feature selection based on correlation values
correlation = csvToArray("feature_correlation_results.csv")

favoriteCols = []
favoriteColsNames = []

for i in range(1, len(correlation)):
    if correlation[i][0] in featIdxMap:
        if math.fabs(float(correlation[i][1 + labelOfInterest])) > 0.1:
            # print correlation[i][0]
            favoriteCols.append(featIdxMap[correlation[i][0]])
            favoriteColsNames.append(correlation[i][0])

print 'favoritCols = ', len(favoriteCols)

print favoriteColsNames

start_time = time.time()


# making test set half positive and half negative
# removes some of negative instances
# positive could mean that the instance has been nominated for OR has won best picture!

tmp = []
for i in range(len(testRows)):
    if labels[testRows[i]][labelOfInterest] == 1:
        tmp.append(testRows[i])

positiveNum = len(tmp)
for i in range(len(testRows)):
    if labels[testRows[i]][labelOfInterest] == 0 and positiveNum > 0:
        tmp.append(testRows[i])
        positiveNum -= 1

testRows = tmp

print 'test length = ', len(testRows)

#fitting SVM Classifier to data
clf = svm.SVC(kernel='poly', degree=3, max_iter=1000000)
clf.fit((features[trainRows, :])[:, favoriteCols], labels[trainRows, labelOfInterest])
y_pred = clf.predict((features[testRows, :])[:, favoriteCols])
y_test = labels[testRows, labelOfInterest]
acc = np.mean((y_test-y_pred)==0)

#prints actual label and predicted label to compare
print '======================'
for i in range(len(y_pred)):
    print (testRows[i] + 2), ' = ', y_pred[i], ' ', y_test[i]

print 'accuracy = %f' %(np.mean((y_test-y_pred)==0))