#!/usr/bin/python

from preprocessor import DataPreprocessor
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LinearRegression
from sklearn import cross_validation
from sklearn import metrics
import numpy as np
import argparse
import csv

"""
Applies various algorithms on the same data with cross-validation (10-fold),
and reports on the output.
"""

# Author: Omar Elazhary <omazhary@gmail.com>
# License: MIT

def precision_at_k(y_true, y_predicted, confidence, k):
    """
    Calculates the precision at k results as a scoring metric.
    Parameters:
        y_true - A list of true/correct labels to be used as a reference.
        y_predicted - A list of predicted labels aligned with y_true.
        confidence - A list of confidence scores aligned with y_true.
    Returns:
        The precision at k score.
    """
    # Reorder based on confidence (max to min). The further the point is from
    # the hyperplane, the more certain we are it's classified correctly.
    confidence = np.asarray(confidence)
    np.absolute(confidence)
    indices = confidence.argsort()[::-1]
    k_max_true = []
    k_max_predicted = []
    for index in indices:
        if y_true[index] == 1:
            k_max_true.append(y_true[index])
            k_max_predicted.append(y_predicted[index])
        if len(k_max_true) == k:
            break
    return metrics.recall_score(k_max_true, k_max_predicted)

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

prep_nom = DataPreprocessor(lbls, nom_ignore, 'movies_all_features_nom.csv')
prep_nom.preprocess()
prep_nom.numerify()
prep_win = DataPreprocessor(lbls, win_ignore, 'movies_all_features_won.csv')
prep_win.preprocess()
prep_win.numerify()
prep_awd = DataPreprocessor(lbls, awd_ignore, 'movies_all_features_nom.csv')
prep_awd.preprocess()
prep_awd.numerify()

# Create test set:
print("Extracting test set...")
test_instances = []
with open('testing_indices.csv', 'rb') as test_inst_file:
    entryreader = csv.reader(test_inst_file, delimiter=',')
    for row in entryreader:
        test_instances.append(int(row[0]))
prep_nom.create_test_set(test_instances)
prep_win.create_test_set(test_instances)
prep_awd.create_test_set(test_instances)

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
    k = 10
    for clf in classifiers_nomination:
        predictions = clf.predict(prep_nom.test_features)
        confidence = clf.decision_function(prep_nom.test_features)
        score = metrics.f1_score(prep_nom.test_labels[0], predictions)
        prec = metrics.precision_score(prep_nom.test_labels[0],
                                       predictions)
        recall = metrics.recall_score(prep_nom.test_labels[0],
                                      predictions)
        patk = precision_at_k(prep_nom.test_labels[0], predictions,
                              confidence, k)
        print("Nomination - %s Precision: %0.2f" % (type(clf).__name__, prec))
        print("Nomination - %s Recall: %0.2f" % (type(clf).__name__, recall))
        print("Nomination - %s F-Score: %0.2f" % (type(clf).__name__, score))
        print("Nomination - %s Precision at %d: %0.2f" % (type(clf).__name__,
              k, patk))
    for clf in classifiers_win:
        predictions = clf.predict(prep_win.test_features)
        confidence = clf.decision_function(prep_win.test_features)
        score = metrics.f1_score(prep_win.test_labels[1],
                                 clf.predict(prep_win.test_features))
        prec = metrics.precision_score(prep_win.test_labels[1],
                                       predictions)
        recall = metrics.recall_score(prep_win.test_labels[1],
                                      predictions)
        patk = precision_at_k(prep_win.test_labels[1], predictions,
                              confidence, k)
        print("Win - %s Precision: %0.2f" % (type(clf).__name__, prec))
        print("Win - %s Recall: %0.2f" % (type(clf).__name__, recall))
        print("Win - %s F-Score: %0.2f" % (type(clf).__name__, score))
        print("Win - %s Precision at %d: %0.2f" % (type(clf).__name__,
              k, patk))
    for reg in regressors:
        score = reg.score(prep_awd.test_features,
                          prep_awd.test_labels[2])
        print("Awards - %s Score: %0.2f" % (type(reg).__name__, score))
