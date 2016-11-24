#!/usr/bin/python

from preprocessor import DataPreprocessor
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LinearRegression
from sklearn import cross_validation
from sklearn import metrics

"""
Applies various algorithms on the same data with cross-validation (10-fold),
and reports on the output.
"""

# Author: Omar Elazhary <omazhary@gmail.com>
# License: MIT

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

prep_nom = DataPreprocessor(lbls, feat_ignore, data_file)
prep_nom.preprocess()
prep_nom.numerify()
prep_win = DataPreprocessor(lbls, feat_ignore, data_file)
prep_win.preprocess()
prep_win.numerify()
prep_awards = DataPreprocessor(lbls, feat_ignore, data_file)
prep_awards.preprocess()
prep_awards.numerify()

# Create test set:
prep_nom.create_test_set(0.3, 0, True)
prep_win.create_test_set(0.3, 1, True)
prep_awards.create_test_set(0.3, 2, True)

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
print("Training and cross validating...")
for clf in classifiers_nomination:
    scores = cross_validation.cross_val_score(clf, prep_nom.features_numerical,
                                              prep_nom.labels_numerical[0],
                                              cv=10, scoring="f1_macro")
    print("Nomination - %s Accuracy: %0.2f (+/- %0.2f)"
          % (type(clf).__name__, scores.mean(), scores.std() * 2))
for clf in classifiers_win:
    scores = cross_validation.cross_val_score(clf,
                                              prep_win.features_numerical,
                                              prep_win.labels_numerical[1],
                                              cv=10, scoring="f1_macro")
    print("Win - %s Accuracy: %0.2f (+/- %0.2f)"
          % (type(clf).__name__, scores.mean(), scores.std() * 2))
for reg in regressors:
    scores = cross_validation.cross_val_score(reg,
                                              prep_awards.features_numerical,
                                              prep_awards.labels_numerical[2],
                                              cv=10)
    print("Awards - %s Accuracy: %0.2f (+/- %0.2f)"
          % (type(reg).__name__, scores.mean(), scores.std() * 2))

# Run testing:
print("### Testing against random test set...")
for clf in classifiers_nomination:
    clf = clf.fit(prep_nom.features_numerical, prep_nom.labels_numerical[0])
    score = metrics.f1_score(prep_nom.test_labels,
                             clf.predict(prep_nom.test_features))
    print("Nomination - %s F-Score: %0.2f" % (type(clf).__name__, score))
for clf in classifiers_win:
    clf = clf.fit(prep_win.features_numerical, prep_win.labels_numerical[1])
    score = metrics.f1_score(prep_win.test_labels,
                             clf.predict(prep_win.test_features))
    print("Win - %s F-Score: %0.2f" % (type(clf).__name__, score))
for reg in regressors:
    reg = reg.fit(prep_awards.features_numerical,
                  prep_awards.labels_numerical[2])
    score = reg.score(prep_awards.test_features, prep_awards.test_labels)
    print("Awards - %s Accuracy: %0.2f" % (type(reg).__name__, score))
