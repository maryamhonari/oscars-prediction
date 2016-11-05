from __future__ import division
from sklearn import svm
import csv
import numpy as np
import time
import math
from sklearn.metrics import f1_score



features = []

with open("features.csv") as feat:
    entryreader = csv.reader(feat, delimiter=',')
    for row in entryreader:
        features.append(row)

featNames = features[0]
features = features[1:]

print len(features), len(features[0])

labels = []

with open("labels.csv") as lbl:
    entryreader = csv.reader(lbl, delimiter=',')
    for row in entryreader:
        labels.append(row)

labelNames = labels[0]
labels = labels[1:]

print len(labels), len(labels[0])

labels = np.array(labels).astype(int)
features = np.array(features).astype(float)

labels = np.array(labels)
features = np.array(features)

featIdxMap = dict()
for i in range(len(featNames)):
    featIdxMap[featNames[i]] = i

correlation = []

with open("feature_correlation_results.csv") as corr:
    entryreader = csv.reader(corr, delimiter=',')
    for row in entryreader:
        correlation.append(row)





titleYearIdx = -1

for i in range(len(featNames)):
    if featNames[i] == 'title_year':
        titleYearIdx = i

print 'year index = ', titleYearIdx

trainRows = []
testRows = []

for i in range(len(features)):
    if float(features[i][titleYearIdx]) > 2010:
        testRows.append(i)
    else:
        trainRows.append(i)

print len(trainRows) / len(features) , len(testRows) / len(features)

favoriteCols = []

for i in range(1, len(correlation)):
    if correlation[i][0] in featIdxMap:
        if math.fabs(float(correlation[i][1])) > 0.1:
            print correlation[i][0]
            favoriteCols.append(featIdxMap[correlation[i][0]])

print 'favoritCols = ', len(favoriteCols)

start_time = time.time()

#clf = svm.SVC(kernel='linear')

#### temporary customization of test and train
tmp = []
for i in range(len(testRows)):
    if labels[testRows[i]][0] == 1:
        tmp.append(testRows[i])

posNo = len(tmp)

for i in range(len(testRows)):
    if labels[testRows[i]][0] == 0 and posNo > 0:
        tmp.append(testRows[i])
        posNo -= 1

testRows = tmp

print 'test length = ', len(testRows)

#clf = svm.SVC(kernel='linear', cache_size=1000)
#clf = svm.SVC(C=10)
clf=svm.SVC(C=1.0, kernel='poly', degree=3, gamma='auto', coef0=0.0, shrinking=True,
          probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=1000,
          decision_function_shape=None, random_state=None)

print 'here'

#(tf[:,[91,1063]])[[0,3,4],:]
clf.fit((features[trainRows, :])[:, favoriteCols], labels[trainRows, 0])

print 'there'

y_pred = clf.predict((features[testRows, :])[:, favoriteCols])
y_test = labels[testRows, 0]

print 'accuracy = %f' %(np.mean((y_test-y_pred)==0))


#score = clf.score((features[testRows, :])[:, favoriteCols], labels[testRows, 0])
#print 'fscore = %s' % (f1_score(y_true, y_pred, average='macro') )



print("--- %s seconds ---" % (time.time() - start_time))


y = labels[testRows, 0]

# for i in range(len(y)):
#     if y[i] == '1':
#         print testRows[i]
#
print '======================'

for i in range(len(y_pred)):
    print (testRows[i] + 2), ' = ', y_pred[i], ' ', y_test[i]


#Taken from Calob's lab
# for i in range(1, 101):
# 	C = i/10.0
# 	clf = svm.SVC(C=C)
# 	clf = clf.fit(data, categories)
#
# 	score = clf.score(test, ans)
# 	print 'C = %.2f: %s' % (C, score)
