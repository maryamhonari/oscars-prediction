#!/usr/bin/python

import csv
from preprocessor import DataPreprocessor
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier

"""
Applies clustering algorithms to the movie dataset in order to discover
meaningful relationships between data tuples.
"""

# Author: Omar Elazhary <omazhary@gmail.com>
# License: MIT

preprocessor = DataPreprocessor(['Nominated Best Picture',
                                'Won Best Picture', 'Num of Awards'],
                                ['genres', 'plot_keywords'],
                                'final_dataset_no_duplicates.csv')
preprocessor.preprocess()

preprocessor.add_feature(preprocessor.labels[0])

preprocessor.numerify()

preprocessor.create_test_set(0.3, 0, True)

km = KMeans(n_clusters=2, init='k-means++', max_iter=5000, n_init=1,
        verbose=False)

km.fit_predict(preprocessor.features_numerical)

print km.labels_
print preprocessor.labels_numerical[0]

print("Homogeneity: %0.3f" % metrics.homogeneity_score(
    preprocessor.labels_numerical[0], km.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(
    preprocessor.labels_numerical[0] , km.labels_))
print("V-Measure: %0.3f" % metrics.v_measure_score(
    preprocessor.labels_numerical[0], km.labels_))
print("Adjusted Rand-Index: %0.3f"
        % metrics.adjusted_rand_score(preprocessor.labels_numerical[0],
            km.labels_))
print("Silhouette Coefficient: %0.3f"
        % metrics.silhouette_score(preprocessor.labels_numerical[0],
            km.labels_))
