from __future__ import division
import csv
import numpy as np
from sklearn import preprocessing
from sklearn import cross_validation
from enum import Enum
from sklearn import metrics
import math
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor

class Label(Enum):
    Nominee = 0
    Winner = 1
    NumOfAwards = 2

#
# converts a csv file to 2D array
def csvToArray(filename):
    ret = []
    with open(filename) as x:
        entryreader = csv.reader(x, delimiter=',')
        for row in entryreader:
            ret.append(row)
    return ret

def getCols(label_of_interest=0):
    """
    :param label_of_interest: #0 means nominee , 1 means winner (for best picture), 2 means number of awards
    :return:
    """
    # feature selection based on correlation values
    correlation = csvToArray("../OscarDataset/feature_correlation.csv")

    cols = []
    colsNames = []

    threshold = 0.1
    for i in range(1, len(correlation)):
        curName = correlation[i][0]
        if label_of_interest == Label.NumOfAwards and ('Won' in curName):  # we should ignore this when predicting num of awards
            continue

        if label_of_interest == Label.Winner and ('Won' in curName):
            continue

        if label_of_interest == Label.Nominee:
            if 'Won' in curName or 'Nominated' in curName:
                continue

        if curName in feature_names:
            if math.fabs(float(correlation[i][1 + label_of_interest.value])) > threshold:
                cols.append(feature_names.index(curName))
                colsNames.append(curName)

    # print 'favoritCols = ', len(cols)

    return cols

feat_train = csvToArray('../OscarDataset/feat_train.csv')
feat_test = csvToArray('../OscarDataset/feat_test.csv')
label_train = csvToArray('../OscarDataset/label_train.csv')
label_test = csvToArray('../OscarDataset/label_test.csv')
feat_test_2016 = csvToArray('../OscarDataset/feat_test_2016.csv')

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
feat_test_2016 = np.array(feat_test_2016)[1:, :].astype(float)

# print feature_names

####################### Save important features before scaling ###########
train_years = np.copy(feat_train[:, feature_names.index('title_year')])
test_years = np.copy(feat_test[:, feature_names.index('title_year')])
# print train_years, test_years

################# Do scaling if you need it ###################
# scaling the feature columns
# Don't scale 'original row' feature
for i in range(1, len(feat_train[0])):
    feat_train[:, i] = preprocessing.scale(feat_train[:, i])

for i in range(1, len(feat_test[0])):
    feat_test[:, i] = preprocessing.scale(feat_test[:, i])

for i in range(1, len(feat_test_2016[0])):
    feat_test_2016[:, i] = preprocessing.scale(feat_test_2016[:, i])


# Just printing
# print '-> ', feature_names
# print '-> ', label_names

################## Your code goes here #########################
###=================== AdaBoost =======================
cols = getCols(Label.NumOfAwards)

reg = AdaBoostRegressor(DecisionTreeRegressor(max_depth=20), n_estimators=300, random_state=None)

predicted = cross_validation.cross_val_predict(reg, feat_train[:, cols], label_train[:, Label.NumOfAwards.value], cv=10)

reg.fit(feat_train[:, cols], label_train[:, Label.NumOfAwards.value])

predicted = reg.predict(feat_test[:, cols])

print 'Awards - AdaBoost regressor Score: %0.2f' % (metrics.r2_score(label_test[:, Label.NumOfAwards.value], predicted))

##=================== AdaBoost Nomine ================
label_of_interest = Label.Nominee

cols = getCols(label_of_interest)

clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                         algorithm="SAMME",
                         n_estimators=200)

predicted = cross_validation.cross_val_predict(clf, feat_train[:, cols], label_train[:, label_of_interest.value], cv=10)

clf.fit(feat_train[:, cols], label_train[:, label_of_interest.value])

predicted = clf.predict(feat_test[:, cols])

f1=open('./adaboost_predicted_for_nominee.txt', 'w')
for i in range(len(predicted)):
    f1.write(str(predicted[i]) + '\n')

score = metrics.f1_score(label_test[:, label_of_interest.value], predicted)
prec = metrics.precision_score(label_test[:, label_of_interest.value], predicted)
recall = metrics.recall_score(label_test[:, label_of_interest.value], predicted)

print("Nomination - AdaBoost Precision: %0.2f" % (prec))
print("Nomination - AdaBoost Recall: %0.2f" % (recall))
print("Nomination - AdaBoost F-Score: %0.2f" % (score))

###=================== Nominee 2016 =====================
distances = clf.decision_function(feat_test_2016[:, cols])

print 'Possible nominees:'

raw_2016 = csvToArray('../OscarDataset/movies_from_2016.csv')
title_idx = raw_2016[0].index('movie_title')
raw_2016 = raw_2016[1:]

most_likely = []
for h in range(20):
    max = -10000
    max_idx = -1
    for i in range(len(distances)):
        if distances[i] > max:
            max = distances[i]
            max_idx = int(feat_test_2016[i][0]) - 2
    most_likely.append(max_idx)
    distances[max_idx] = -10000

for i in range(len(most_likely)):
    print raw_2016[most_likely[i]][title_idx]

###=================== SVC Winner =======================
label_of_interest = Label.Winner

cols = getCols(label_of_interest)

clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                         algorithm="SAMME",
                         n_estimators=200)

predicted = cross_validation.cross_val_predict(clf, feat_train[:, cols], label_train[:, label_of_interest.value], cv=10)

clf.fit(feat_train[:, cols], label_train[:, label_of_interest.value])

predicted = clf.predict(feat_test[:, cols])

print '#### Predicted values ####'

f1=open('./adaboost_predicted_for_win.txt', 'w')
for i in range(len(predicted)):
    f1.write(str(predicted[i]) + '\n')

score = metrics.f1_score(label_test[:, label_of_interest.value], predicted)
prec = metrics.precision_score(label_test[:, label_of_interest.value], predicted)
recall = metrics.recall_score(label_test[:, label_of_interest.value], predicted)

print("Win - AdaBoost Precision: %0.2f" % (prec))
print("Win - AdaBoost Recall: %0.2f" % (recall))
print("Win - AdaBoost F-Score: %0.2f" % (score))

ada_distances = clf.decision_function(feat_test[:, cols])

real_winner_id = [-1] * 2020
nominees_count = [0] * 2020

year_movie_dist = []
for i in range(2020):
    year_movie_dist.append([])

for i in range(len(predicted)):
    year = int(test_years[i])
    dist = round(ada_distances[i], 3)
    id = int(feat_test[i][0])

    if label_test[i][label_of_interest.value] == 1:
        real_winner_id[year] = id

    if label_test[i][0] == 1:
        nominees_count[year] += 1

    year_movie_dist[year].append((id, dist))

K_top = 5
all_top_k = []

total = 0
correct = 0

for i in range(1928, 2016):
    if len(year_movie_dist[i]) == 0:
        continue

    top = []
    for h in range(K_top):
        max_idx = -1
        max_dist = -1000
        for j in range(len(year_movie_dist[i])):
            if (year_movie_dist[i][j][0] not in top) and (year_movie_dist[i][j][1] > max_dist):
                max_dist = year_movie_dist[i][j][1]
                max_idx = year_movie_dist[i][j][0]

        top.append(max_idx)

    all_top_k.append(top)
    print 'Year %d: total=%d #nominees=%d real_winner=%d, top-k=%s' % (i, len(year_movie_dist[i]), nominees_count[i], real_winner_id[i], (''.join([str(x) + ' ' for x in top])))

    total += 1
    if real_winner_id[i] in top:
        correct += 1

print 'correct percent = ', correct/total

