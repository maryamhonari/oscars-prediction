#!/usr/bin/python

from preprocessor import DataPreprocessor
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LinearRegression
from sklearn import cross_validation
from sklearn import metrics
import argparse
import csv

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

# Filter out features based on correlation:
print("Filtering out weakly correlated features...")
nom_ignore = []
win_ignore = []
awd_ignore = []
rows = 0
with open('feature_correlation_scaled.csv', 'rb') as corr_file:
    entryreader = csv.reader(corr_file, delimiter=',')
    for row in entryreader:
        # Ignore first row:
        if not rows == 0:
            nom_corr = float(row[1])
            win_corr = float(row[2])
            awd_corr = float(row[3])
            if nom_corr > -0.1 and nom_corr < 0.1:
                nom_ignore.append(row[0])
            if win_corr > -0.1 and win_corr < 0.1:
                win_ignore.append(row[0])
            if awd_corr > -0.1 and awd_corr < 0.1:
                awd_ignore.append(row[0])
        rows += 1

# Preprocess data:
print("Preprocessing the data...")
lbls = ['Nominated Best Picture', 'Won Best Picture', 'Num of Awards']
data_file = 'training_data.csv'
test_file = 'testing_data.csv'

prep_nom = DataPreprocessor(lbls, nom_ignore, data_file)
prep_nom.preprocess()
prep_nom.numerify([8])
prep_win = DataPreprocessor(lbls, win_ignore, data_file)
prep_win.preprocess()
prep_win.numerify([8])
prep_awd = DataPreprocessor(lbls, awd_ignore, data_file)
prep_awd.preprocess()
prep_awd.numerify([8])

# Create test set:
test_nom = DataPreprocessor(lbls, nom_ignore, test_file)
test_nom.preprocess()
test_nom.numerify([8])
test_win = DataPreprocessor(lbls, win_ignore, test_file)
test_win.preprocess()
test_win.numerify([8])
test_awards = DataPreprocessor(lbls, awd_ignore, test_file)
test_awards.preprocess()
test_awards.numerify([8])

# Prepare Classifiers:
classifiers_nomination = [
                Perceptron(penalty='l2'),
        ]
classifiers_win = [
                Perceptron(penalty='l1'),
        ]
regressors = [
                LinearRegression(),
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
        print("Nomination - %s Score: %0.2f (+/- %0.2f)"
              % (type(clf).__name__, scores.mean(), scores.std() * 2))
for clf in classifiers_win:
    clf = clf.fit(prep_win.features_numerical, prep_win.labels_numerical[1])
    if not args['no_cv']:
        scores = cross_validation.cross_val_score(clf,
                                                  prep_win.features_numerical,
                                                  prep_win.labels_numerical[1],
                                                  cv=10, scoring="f1_macro")
        print("Win - %s Score: %0.2f (+/- %0.2f)"
              % (type(clf).__name__, scores.mean(), scores.std() * 2))
for reg in regressors:
    reg = reg.fit(prep_awd.features_numerical,
                  prep_awd.labels_numerical[2])
    if not args['no_cv']:
        scores = cross_validation.cross_val_score(reg,
                                                  prep_awd.features_numerical,
                                                  prep_awd.labels_numerical[2],
                                                  cv=10)
        print("Awards - %s Score: %0.2f (+/- %0.2f)"
              % (type(reg).__name__, scores.mean(), scores.std() * 2))

# Run testing:
if not args['no_test']:
    print("### Testing against test set...")
    for clf in classifiers_nomination:
        predictions = clf.predict(test_nom.features_numerical)
        score = metrics.f1_score(test_nom.labels_numerical[0], predictions)
        prec = metrics.precision_score(test_nom.labels_numerical[0],
                                            predictions)
        recall = metrics.recall_score(test_nom.labels_numerical[0],
                                      predictions)
        print("Nomination - %s Precision: %0.2f" % (type(clf).__name__, prec))
        print("Nomination - %s Recall: %0.2f" % (type(clf).__name__, recall))
        print("Nomination - %s F-Score: %0.2f" % (type(clf).__name__, score))
    for clf in classifiers_win:
        predictions = clf.predict(test_win.features_numerical)
        score = metrics.f1_score(test_win.labels_numerical[1],
                                 clf.predict(test_win.features_numerical))
        prec = metrics.precision_score(test_win.labels_numerical[1],
                                            predictions)
        recall = metrics.recall_score(test_win.labels_numerical[1],
                                      predictions)
        print("Win - %s Precision: %0.2f" % (type(clf).__name__, prec))
        print("Win - %s Recall: %0.2f" % (type(clf).__name__, recall))
        print("Win - %s F-Score: %0.2f" % (type(clf).__name__, score))
    for reg in regressors:
        score = reg.score(test_awards.features_numerical,
                          test_awards.labels_numerical[2])
        print("Awards - %s Score: %0.2f" % (type(reg).__name__, score))
