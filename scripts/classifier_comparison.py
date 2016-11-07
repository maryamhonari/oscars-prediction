#!/usr/bin/python

from preprocessor import DataPreprocessor
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LinearRegression
from sklearn import cross_validation

"""
Applies various algorithms on the same data with cross-validation (10-fold),
and reports on the output.
"""

# Author: Omar Elazhary <omazhary@gmail.com>
# License: MIT

# Preprocess data:
preprocessor_nomination = DataPreprocessor(['Nominated Best Picture',
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
preprocessor_nomination.preprocess()
preprocessor_nomination.numerify()
preprocessor_win = DataPreprocessor(['Nominated Best Picture',
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
preprocessor_win.preprocess()
preprocessor_win.numerify()
preprocessor_awards = DataPreprocessor(['Nominated Best Picture',
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
preprocessor_awards.preprocess()
preprocessor_awards.numerify()

# Create test set:
preprocessor_nomination.create_test_set(0.3, 0, True)
preprocessor_win.create_test_set(0.3, 1, True)
preprocessor_awards.create_test_set(0.3, 2, True)

# Prepare Classifiers:
classifiers_nomination = [
                Perceptron(),
        ]
classifiers_win = [
                Perceptron(),
        ]
regressors = [
                LinearRegression(),
        ]

# Run training and cross-validation:
for clf in classifiers_nomination:
    scores = cross_validation.cross_val_score(clf,
            preprocessor_nomination.features_numerical,
            preprocessor_nomination.labels_numerical[0], cv=10)
    print("Nomination - %s Accuracy: %0.2f (+/- %0.2f)"
            % (type(clf).__name__, scores.mean(), scores.std() * 2))
for clf in classifiers_win:
    scores = cross_validation.cross_val_score(clf,
            preprocessor_win.features_numerical,
            preprocessor_win.labels_numerical[1], cv=10)
    print("Win - %s Accuracy: %0.2f (+/- %0.2f)"
            % (type(clf).__name__, scores.mean(), scores.std() * 2))
for reg in regressors:
    scores = cross_validation.cross_val_score(reg,
            preprocessor_awards.features_numerical,
            preprocessor_awards.labels_numerical[2], cv=10)
    print("Awards - %s Accuracy: %0.2f (+/- %0.2f)"
            % (type(reg).__name__, scores.mean(), scores.std() * 2))
