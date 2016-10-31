#!/usr/bin/python

import numbers
import decimal
import random
import scipy.stats
import numpy as np
from preprocessor import DataPreprocessor
from sklearn import preprocessing

"""
Calculates the correlation between each feature and the class labels, in order
to identify the most influential features.
"""

# Author: Omar Elazhary <omazhary@gmail.com>
# License: None

# Load and preprocess the data

preprocessor = DataPreprocessor(['Nominated Best Picture',
                                'Won Best Picture', 'Num of Awards'],
                                ['genres', 'plot_keywords'],
                                'final_dataset_no_duplicates.csv')
preprocessor.preprocess()

feature_vectors = preprocessor.split_features()

# Need to convert textual attributes to a numeric format.

for index, features in enumerate(feature_vectors):
    test_index = 0
    if (features[test_index] == ''):
        test_index = random.randrange(1, len(features))
    if (not isinstance(features[test_index], numbers.Number) and
            not isinstance(features[test_index], decimal.Decimal)):
        encoder = preprocessing.LabelEncoder()
        encoder.fit(features)
        feature_vectors[index] = encoder.transform(features)

# Assuming that the features are normally distributed, we can check the
# feature-label correlation using the pearson correlation coefficient.

print "feature,nominated_best_picture,won_best_picture,num_of_awards"
for index, feature_vector in enumerate(feature_vectors):
    result = preprocessor.column_headers[index] + ","
    for label_vector in preprocessor.labels:
        feature_array = np.array(feature_vectors[index]).astype(np.float)
        label_array = np.array(label_vector).astype(np.float)
        result += str(scipy.stats.pearsonr(feature_array, label_array)[0])
        result += ","
    print result[:-1]
