# -*- coding: utf-8 -*-

import numpy
import csv
from prep import DataPreprocessor
from sklearn import preprocessing
from sklearn.preprocessing import Imputer

# prep = DataPreprocessor(['Nominated Best Picture',
#                         'Won Best Picture',
#                          'Num of Awards'],

prep = DataPreprocessor(['Nominated Best Picture',
                        'Won Best Picture',
                         'Num of Awards'],

                        ['genres',
                         'plot_keywords',
                         'movie_title',
                         'movie_imdb_link'],

                                'Data.csv')
prep.preprocess()


feat_num_map = prep.feat_num_map #this has problems
features = prep.features
labels = prep.labels


## writing labels to csv
labels = numpy.array(labels)
labels = labels.transpose()
all = []
all.append(['Nominated Best Picture',
            'Won Best Picture',
            'Num of Awards'])
all.extend(labels.tolist())

with open('labels.csv', 'w') as fp:
    a = csv.writer(fp, delimiter=',', quotechar='\"')
    a.writerows(all)


string_valued_columns = [
    'director_name',
    'actor_2_name',
    'actor_3_name',
    'actor_1_name',
    'language',
    'country',
    'content_rating',
    'color'
]

features = numpy.array(features)

for x in feat_num_map:
    for y in string_valued_columns:
        if x == y:
            col = int(feat_num_map[x])
            le = preprocessing.LabelEncoder()
            le.fit(features[:, col])
            features[: , col] = le.transform(features[:, col])

            #0's resulted from LabelEncoder are actually missing values
            for n, i in enumerate(features[: , col]):
                if int(i) == 0:
                    features[n, col] = ''

for i in range(len(features)):
    for j in range(len(features[0])):
        if len(features[i][j]) == 0:
            features[i][j] = -1
        else:
            features[i][j] = float(features[i][j])

#taking care of missing values by replacing each of them with mean of the column
imp = Imputer(missing_values=-1, strategy='mean', axis=0)
features = imp.fit_transform(features)

#extracting column names in the right order from the map
columnNames = []
for i in range(len(feat_num_map)):
    for x in feat_num_map:
        if feat_num_map[x] == i:
            columnNames.append(x)
            break

columnNames = numpy.array(columnNames)

#adding column names to features array
#features = numpy.insert(features, 0, columnNames, axis=0)
all = []
all.append(columnNames.tolist())
all.extend(features.tolist())

with open('features.csv', 'w') as fp:
    a = csv.writer(fp, delimiter=',', quotechar='\"')
    a.writerows(all)