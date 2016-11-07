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
                                ['genres', 'plot_keywords', 'movie_imdb_link'],
                                'movies_original.csv')
preprocessor.preprocess()

preprocessor.numerify()

# Assuming that the features are normally distributed, we can check the
# feature-label correlation using the pearson correlation coefficient.

features = map(list, zip(*preprocessor.features_numerical))

print "feature,nominated_best_picture,won_best_picture,num_of_awards"
for index, feature_vector in enumerate(features):
    result = preprocessor.column_headers[index] + ","
    for label_vector in preprocessor.labels:
        feature_array = np.array(feature_vector).astype(np.float)
        label_array = np.array(label_vector).astype(np.float)
        result += str(scipy.stats.pearsonr(feature_array, label_array)[0])
        result += ","
    print result[:-1]
