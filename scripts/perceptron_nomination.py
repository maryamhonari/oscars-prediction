#!/usr/bin/python

from preprocessor import DataPreprocessor
from sklearn.linear_model import Perceptron

"""
Uses a perceptron in order to determine if a movie gets nominated.
"""

# Author: Omar Elazhary <omazhary@gmail.com>
# License: MIT

# Preprocess the data:

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

preprocessor.numerify()

# Create the test set:

preprocessor.create_test_set(0.3, 0, True)

# Perform cross-validation:

clf = Perceptron()
clf = clf.fit(preprocessor.features_numerical,
              preprocessor.labels_numerical[0])

"""
scores = cross_validation.cross_val_score(clf, preprocessor.features_numerical,
        preprocessor.labels_numerical[2], cv=10)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
"""

score = clf.score(preprocessor.test_features, preprocessor.test_labels)

print("Accuracy after testing (no CV): %3.2f%%") % (score * 100)
