#!/usr/bin/python

import csv
from preprocessor import DataPreprocessor
from sklearn.ensemble import ExtraTreesClassifier

"""
Applies sklearn's library to find out which features are the most important.
"""

# Author: Omar Elazhary <omazhary@gmail.com>
# License: MIT

preprocessor = DataPreprocessor(['Nominated Best Picture',
                                'Won Best Picture', 'Num of Awards'],
                                ['genres', 'plot_keywords'],
                                'movies_original.csv')
preprocessor.preprocess()

preprocessor.add_feature(preprocessor.labels[0])

preprocessor.numerify()

model = ExtraTreesClassifier()

result = [
        [''],
        ['Nominated Best Picture'],
        ['Won Best Picture'],
        ['Num of Awards'],
        ]
result[0].extend(preprocessor.column_headers)
for index, label_vector in enumerate(preprocessor.labels_numerical):
    model.fit(preprocessor.features_numerical, label_vector)
    result[index + 1].extend(model.feature_importances_)
with open('feature_importance_original.csv', 'wb') as outcsvfile:
    outwriter = csv.writer(outcsvfile, delimiter=',')
    for index, vector in enumerate(result[0]):
        new_row = []
        new_row.append(result[0][index])
        new_row.append(result[1][index])
        new_row.append(result[2][index])
        new_row.append(result[3][index])
        outwriter.writerow(new_row)
