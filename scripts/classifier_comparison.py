#!/usr/bin/python

from preprocessor import DataPreprocessor
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LinearRegression
from sklearn import cross_validation, svm
from sklearn.metrics import roc_curve, auc
from scipy import interp
from itertools import cycle
import matplotlib.pyplot as plt
import numpy as np

"""
Applies various algorithms on the same data with cross-validation (10-fold),
and reports on the output.
"""

# Author: Omar Elazhary <omazhary@gmail.com>
# License: MIT

def plot_roc(feat_data, lbl_data, title, n_folds=10):
    """
    Plots an ROC/AUC graph based on cross-validation of the data given.
    Parameters:
        feat_data - A 2D structure for feature data [n_samples, n_features].
        lbl_data - A list of labels aligned with the previous feat_data.
        n_folds - The number of cross-validation folds to perform.
    """
    # Prepare CV parameters
    X = np.asarray(feat_data)
    y = np.asarray(lbl_data)
    cv = cross_validation.StratifiedKFold(y, n_folds=10)
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    for i, (train, test) in enumerate(cv):
        trials = clf.fit(X[train], y[train]).predict(X[test])
        # Compute ROC curve and AUC
        fpr, tpr, thresholds = roc_curve(y[test], trials)
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2,
                 label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k', label='Luck')
    mean_tpr /= len(cv)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color='g', linestyle='--',
             label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC for %s - %s' % (title, type(clf).__name__))
    plt.legend(loc="lower right")
    plt.show()

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
nom_file = 'movies_for_nominations.csv'
win_file = 'movies_original.csv'

prep_nom = DataPreprocessor(lbls, feat_ignore, nom_file)
prep_nom.preprocess()
prep_nom.numerify()
prep_win = DataPreprocessor(lbls, feat_ignore, win_file)
prep_win.preprocess()
prep_win.numerify()
prep_awards = DataPreprocessor(lbls, feat_ignore, win_file)
prep_awards.preprocess()
prep_awards.numerify()



# Prepare Classifiers:
classifiers_nomination = [
                Perceptron(),
        ]
classifiers_win = [
                Perceptron(),
        ]
regressors = [
                LinearRegression(normalize=True),
        ]

# Create test set:
prep_nom.create_test_set(0.3, 0, True)
prep_win.create_test_set(0.3, 1, True)
prep_awards.create_test_set(0.3, 2, True)

# Run training and cross-validation:
print("### Training and cross validating...")
for clf in classifiers_nomination:
    scores = cross_validation.cross_val_score(clf, prep_nom.features_numerical,
                                              prep_nom.labels_numerical[0],
                                              cv=10)
    print("Nomination - %s Accuracy: %0.2f (+/- %0.2f)"
          % (type(clf).__name__, scores.mean(), scores.std() * 2))
    plot_roc(prep_nom.features_numerical, prep_nom.labels_numerical[0],
             'Nomination', 10)
for clf in classifiers_win:
    scores = cross_validation.cross_val_score(clf,
                                              prep_win.features_numerical,
                                              prep_win.labels_numerical[1],
                                              cv=10)
    print("Win - %s Accuracy: %0.2f (+/- %0.2f)"
          % (type(clf).__name__, scores.mean(), scores.std() * 2))
    plot_roc(prep_win.features_numerical, prep_win.labels_numerical[1],
             'Win', 10)
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
    score = clf.score(prep_nom.test_features, prep_nom.test_labels)
    print("Nomination - %s Accuracy: %0.2f" % (type(clf).__name__, score))
for clf in classifiers_win:
    clf = clf.fit(prep_nom.features_numerical, prep_nom.labels_numerical[0])
    score = clf.score(prep_nom.test_features, prep_nom.test_labels)
    print("Win - %s Accuracy: %0.2f" % (type(clf).__name__, score))
for reg in regressors:
    reg = reg.fit(prep_nom.features_numerical, prep_nom.labels_numerical[0])
    score = reg.score(prep_nom.test_features, prep_nom.test_labels)
    print("Nomination - %s Accuracy: %0.2f" % (type(reg).__name__, score))
