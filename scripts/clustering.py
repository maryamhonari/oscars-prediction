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

# Preprocess data:
preprocessor = DataPreprocessor(['Nominated Best Picture',
                                'Won Best Picture', 'Num of Awards'],
                                ['genres', 'plot_keywords', 'movie_imdb_link',
                                    'director_name', 'actor_3_facebook_likes',
                                    'actor_2_name', 'actor_1_facebook_likes',
                                    'actor_1_name', 'movie_title',
                                    'cast_total_facebook_likes',
                                    'actor_3_name', 'facenumber_in_poster',
                                    'language', 'country', 'content_rating',
                                    'budget', 'actor_2_facebook_likes',
                                    'aspect_ratio'],
                                'movies_original.csv')
preprocessor.preprocess()

preprocessor.add_feature(preprocessor.labels[0])

preprocessor.numerify()

# Create test set:
preprocessor.create_test_set(0.3, 0, True)

# Start clustering and CV:
km = KMeans(n_clusters=2, init='k-means++', max_iter=5000, n_init=2,
        verbose=False)
km.fit_predict(preprocessor.features_numerical)

# Statisitics about my clusters:
print("Homogeneity: %0.3f" % metrics.homogeneity_score(
    preprocessor.labels_numerical[0], km.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(
    preprocessor.labels_numerical[0] , km.labels_))
print("V-Measure: %0.3f" % metrics.v_measure_score(
    preprocessor.labels_numerical[0], km.labels_))

# Testing:
print km.score(preprocessor.test_features)
