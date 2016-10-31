#!/usr/bin/python

from preprocessor import DataPreprocessor
from sklearn.cluster import KMeans
from sklearn import metrics

"""
Applies clustering algorithms to the movie dataset in order to discover
meaningful relationships between data tuples.
"""

# Author: Omar Elazhary <omazhary@gmail.com>
# License: None

preprocessor = DataPreprocessor(['Nominated Best Picture',
                                'Won Best Picture', 'Num of Awards'],
                                ['genres', 'plot_keywords'],
                                'final_dataset_no_duplicates.csv')
preprocessor.preprocess()
preprocessor.numerify()

km = KMeans(n_clusters=2, init='k-means++', max_iter=5000, n_init=1,
        verbose=False)

km.fit(preprocessor.features_numerical)

print("Homogeneity: %0.3f" % metrics.homogeneity_score(preprocessor.labels[0],
    km.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(preprocessor.labels[0]
    , km.labels_))
print("V-Measure: %0.3f" % metrics.v_measure_score(preprocessor.labels[0],
    km.labels_))
print("Adjusted Rand-Index: %0.3f"
        % metrics.adjusted_rand_score(preprocessor.labels[0], km.labels_))
print("Silhouette Coefficient: %0.3f"
        % metrics.silhouette_score(preprocessor.labels[0], km.labels_))
