#!/usr/bin/python

from preprocessor import DataPreprocessor
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LinearRegression
from sklearn import cross_validation
from sklearn import metrics
import argparse

"""
Applies various algorithms on the same data with cross-validation (10-fold),
and reports on the output.
"""

# Author: Omar Elazhary <omazhary@gmail.com>
# License: MIT

parser = argparse.ArgumentParser(
        description='Run CV and/or Testing on implemented algorithms.')
parser.add_argument(
        '--no-cv', default=False, action='store_true',
        help='Whether or not to output cv results')
parser.add_argument(
        '--no-test', default=False, action='store_true',
        help='Whether or not to output test results')

args = vars(parser.parse_args())

# Preprocess data:
lbls = ['Nominated Best Picture', 'Won Best Picture', 'Num of Awards']
feat_ignore = ['genres', 'plot_keywords',
               'movie_imdb_link',
               'director_name',
               'actor_3_facebook_likes',
               'actor_2_name',
               'actor_1_facebook_likes',
               'actor_1_name',
               'movie_title',
               'cast_total_facebook_likes',
               'actor_3_name',
               'facenumber_in_poster',
               'language',
               'country',
               'content_rating',
               'budget',
               'actor_2_facebook_likes',
               'aspect_ratio']
data_file = 'training_data.csv'
test_file = 'testing_data.csv'

prep_nom = DataPreprocessor(lbls, feat_ignore, data_file)
prep_nom.preprocess()
prep_nom.numerify()
prep_win = DataPreprocessor(lbls, feat_ignore, data_file)
prep_win.preprocess()
prep_win.numerify()
prep_awd = DataPreprocessor(lbls, feat_ignore, data_file)
prep_awd.preprocess()
prep_awd.numerify()

# Create test set:
test_nom = DataPreprocessor(lbls, feat_ignore, test_file)
test_nom.preprocess()
test_nom.numerify()
test_win = DataPreprocessor(lbls, feat_ignore, test_file)
test_win.preprocess()
test_win.numerify()
test_awards = DataPreprocessor(lbls, feat_ignore, test_file)
test_awards.preprocess()
test_awards.numerify()

# Prepare Classifiers:
classifiers_nomination = [
                Perceptron(penalty='l2'),
        ]
classifiers_win = [
                Perceptron(penalty='l1'),
        ]
regressors = [
                LinearRegression(normalize=True),
        ]

# Run training and cross-validation:
print("Training...")
if not args['no_cv']:
    print("### Cross validation enabled.")
for clf in classifiers_nomination:
    clf = clf.fit(prep_nom.features_numerical, prep_nom.labels_numerical[0])
    if not args['no_cv']:
        scores = cross_validation.cross_val_score(clf,
                                                  prep_nom.features_numerical,
                                                  prep_nom.labels_numerical[0],
                                                  cv=10, scoring="f1_macro")
        print("Nomination - %s Accuracy: %0.2f (+/- %0.2f)"
              % (type(clf).__name__, scores.mean(), scores.std() * 2))
for clf in classifiers_win:
    clf = clf.fit(prep_win.features_numerical, prep_win.labels_numerical[1])
    if not args['no_cv']:
        scores = cross_validation.cross_val_score(clf,
                                                  prep_win.features_numerical,
                                                  prep_win.labels_numerical[1],
                                                  cv=10, scoring="f1_macro")
        print("Win - %s Accuracy: %0.2f (+/- %0.2f)"
              % (type(clf).__name__, scores.mean(), scores.std() * 2))
for reg in regressors:
    reg = reg.fit(prep_awd.features_numerical,
                  prep_awd.labels_numerical[2])
    if not args['no_cv']:
        scores = cross_validation.cross_val_score(reg,
                                                  prep_awd.features_numerical,
                                                  prep_awd.labels_numerical[2],
                                                  cv=10)
        print("Awards - %s Accuracy: %0.2f (+/- %0.2f)"
              % (type(reg).__name__, scores.mean(), scores.std() * 2))

# Run testing:
if not args['no_test']:
    print("### Testing against test set...")
    for clf in classifiers_nomination:
        score = metrics.f1_score(test_nom.labels_numerical[0],
                                 clf.predict(test_nom.features_numerical))
        print("Nomination - %s F-Score: %0.2f" % (type(clf).__name__, score))
    for clf in classifiers_win:
        score = metrics.f1_score(test_win.labels_numerical[1],
                                 clf.predict(test_win.features_numerical))
        print("Win - %s F-Score: %0.2f" % (type(clf).__name__, score))
    for reg in regressors:
        score = reg.score(test_awards.features_numerical,
                          test_awards.labels_numerical[2])
        print("Awards - %s Accuracy: %0.2f" % (type(reg).__name__, score))
