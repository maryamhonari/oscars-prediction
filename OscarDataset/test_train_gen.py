from __future__ import division
import csv
import numpy as np
import math
from preprocessor import DataPreprocessor
from enum import Enum
import scipy.stats
from sklearn import preprocessing


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

    return train, test

preprocessor  = DataPreprocessor(['Nominated Best Picture', 'Won Best Picture', 'Num of Awards']
                                 , ['genres', 'plot_keywords', 'movie_title', 'movie_imdb_link']
                                 ,'movies_all_features.csv')

preprocessor.preprocess()
preprocessor.numerify(scale=False)

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
train, test = getTrainTestRowsAndCols(features, labels, Label.Nominee, years)
correlation_results = np.zeros((len(feat_names), 3))

print('\n\n\nThis is correlation results:')
print("feature,nominated_best_picture,won_best_picture,num_of_awards")

scaler = preprocessing.StandardScaler()
scaled_features = np.copy(features)
scaled_features = scaler.fit_transform(scaled_features).tolist()
scaled_features = np.array(scaled_features)

for i in range(len(feat_names)):
    vec1 = np.array(scaled_features[train, i], dtype=float)

    result = str(feat_names[i]) + ','
    for j in range(3):
        vec2 = np.array(labels[train, j], dtype=float)
        vec2 = np.copy(vec2)

        val = 0
        if np.std(vec1)!=0 and np.std(vec2)!=0:
            val = round(scipy.stats.pearsonr(vec1, vec2)[0], 3)
        result += str(val)
        if j < 2:
            result += ','
        correlation_results[i, j] = val
    print result

############ Add original_rows ##################
tmp = []
tmp.append('original_row')
tmp.extend(feat_names)
feat_names = tmp
features = np.insert(features, 0, [x+2 for x in range(len(features))], axis=1)

cols = []
cols.append(0)

for i in range(0, len(feat_names) - 1):
    if correlation_results[i, 0] > 0.1 \
            or correlation_results[i, 1] > 0.1 \
            or correlation_results[i, 2] > 0.1:
        cols.append(i + 1)

print 'Feature names'
tmp = []
for i in cols:
    tmp.append(feat_names[i])
feat_names = tmp

writeCSV((features[train, :])[:, cols], feat_names, "feat_train")
writeCSV((features[test, :])[:, cols], feat_names, "feat_test")
writeCSV(labels[train, :], preprocessor.class_labels, "label_train")
writeCSV(labels[test, :], preprocessor.class_labels, "label_test")