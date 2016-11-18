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

def getTrainTestRowsAndCols(features, labels, labelOfInterest, years):

    """

    :param labelOfInterest: #0 means nominee , 1 means winner (for best picture), 2 means number of awards
    :return:
    """

    train = []
    test = []

    for i in range(len(features)):
        if labelOfInterest != Label.Winner:
            if int(math.floor(years[i])) % 4 == 0:
                test.append(i)
            else:
                train.append(i)
        else:
            if labels[i][0] == 1:
                if int(math.floor(years[i])) % 4 == 0:
                    test.append(i)
                else:
                    train.append(i)

    # prints percentage of train and test
    print len(train) / len(features), len(test) / len(features)

    # feature selection based on correlation values
    correlation = csvToArray("feature_correlation_results.csv")

    cols = []
    colsNames = []

    threshold = 0.1
    for i in range(1, len(correlation)):
        if labelOfInterest == Label.NumOfAwards and ('Won' in correlation[i][0]):  # we should ignore this when predicting num of awards
            threshold = 0.1
            continue

        if labelOfInterest == Label.Winner and ('Won' in correlation[i][0]):
            threshold = 0.1
            continue

        if labelOfInterest == Label.Nominee and (('Won' in correlation[i][0])):
            threshold = 0.1
            continue

        if correlation[i][0] in featIdxMap:
            if math.fabs(float(correlation[i][1 + labelOfInterest.value])) > threshold:
                # print correlation[i][0]
                cols.append(featIdxMap[correlation[i][0]])
                colsNames.append(correlation[i][0])

    print 'favoritCols = ', len(cols)

    # print colsNames

    # making test set half positive and half negative
    # removes some of negative instances
    # positive could mean that the instance has been nominated for OR has won best picture!

    return train, test, cols



ignore_list = ['genres', 'plot_keywords', 'movie_title', 'movie_imdb_link']
correlation = csvToArray("feature_correlation_ultimate.csv")

for i in range(1, len(correlation)):
    if math.fabs(float(correlation[i][1])) < 0.08 \
            and math.fabs(float(correlation[i][2])) < 0.08 \
            and math.fabs(float(correlation[i][3])) < 0.08:
        ignore_list.append(correlation[i][0])

print len(ignore_list)

preprocessor  = DataPreprocessor(['Nominated Best Picture', 'Won Best Picture', 'Num of Awards']
                                 , ignore_list
                                 ,'movie_metadata_ultimate.csv')

preprocessor.preprocess()
preprocessor.numerify()

features = preprocessor.features_numerical

labels = preprocessor.labels_numerical
feat_names = preprocessor.column_headers

labels = np.array(labels).astype(int)
features = np.array(features).astype(float)

labels = np.transpose(labels)

print len(labels), len(labels[0])
print len(features), len(features[0])

featIdxMap = dict()

#getting title_year column number
for i in range(len(feat_names)):
    featIdxMap[feat_names[i]] = i

# print featIdxMap

#We need this for splitting data
yearIdx = featIdxMap['title_year']
years = np.copy(features[:, yearIdx])


########## Creating New DataSet ################
train, test, cols = getTrainTestRowsAndCols(features, labels, Label.Nominee, years)
tmp = []
tmp.append('Original Row')
tmp.extend(feat_names)
feat_names = tmp

features = np.insert(features, 0, [x+2 for x in range(len(features))], axis=1)

writeCSV(features[train, :], feat_names, "feat_train")
writeCSV(features[test, :], feat_names, "feat_test")
writeCSV(labels[train, :], preprocessor.class_labels, "label_train")
writeCSV(labels[test, :], preprocessor.class_labels, "label_test")
