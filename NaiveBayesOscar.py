from __future__ import division
from sklearn.naive_bayes import GaussianNB 
import csv
import numpy as np
import time
import math
from sklearn.metrics import f1_score
from .utils import check_X_y, check_array




class GaussianNB():

	def __init__(self, priors=0.1):
        self._priors = priors

	def fit(self, X, y, sample_weight=None): #Fit Gaussian Naive Bayes according to X, y
	
		X, y = check_X_y(X, y)
		return self._partial_fit(X, y, np.unique(y), _refit=True,
		sample_weight=sample_weight)

	def _update_mean_variance(n_past, mu, var, X, sample_weight=None):	#Compute online update of Gaussian mean and variance

		if X.shape[0] == 0:
		return mu, var
		# Compute mean and variance of new datapoints:
		if sample_weight is not None:
            n_new = float(sample_weight.sum())
            new_mu = np.average(X, axis=0, weights=sample_weight / n_new)
            new_var = np.average((X - new_mu) ** 2, axis=0,
                                 weights=sample_weight / n_new)
        else:
            n_new = X.shape[0]
            new_var = np.var(X, axis=0)
            new_mu = np.mean(X, axis=0)

        if n_past == 0:
            return new_mu, new_var

		n_total = float(n_past + n_new)
		
		total_mu = (n_new * new_mu + n_past * mu) / n_total
		old_ssd = n_past * var
        new_ssd = n_new * new_var
        total_ssd = (old_ssd + new_ssd +
                     (n_past / float(n_new * n_total)) *
                     (n_new * mu - n_new * new_mu) ** 2)
        total_var = total_ssd / n_total

		return total_mu, total_var
		
	def partial_fit(self, X, y, classes=None, sample_weight=None):
	 
		return self._partial_fit(X, y, classes, _refit=False,
                                 sample_weight=sample_weight)
								 
	def _partial_fit(self, X, y, classes=None, _refit=False,sample_weight=None):	#Actual implementation of Gaussian NB fitting.
	 
		X, y = check_X_y(X, y)
		epsilon = 1e-9 * np.var(X, axis=0).max()

        if _refit:
            self.classes_ = None

		if _check_partial_fit_first_call(self, classes):
			n_features = X.shape[1]
			n_classes = len(self.classes_)
			self.theta_ = np.zeros((n_classes, n_features))
			self.sigma_ = np.zeros((n_classes, n_features))
			self.class_count_ = np.zeros(n_classes, dtype=np.float64)
			n_classes = len(self.classes_)
		
			if self.priors is not None:
				priors = np.asarray(self.priors)
				if len(priors) != n_classes:
					raise ValueError('Number of priors must match number of'' classes.')
				if priors.sum() != 1.0:
					raise ValueError('The sum of the priors should be 1.')
		
				if (priors < 0).any():
					raise ValueError('Priors must be non-negative.')
				self.class_prior_ = priors
			
			else:
                self.class_prior_ = np.zeros(len(self.classes_),
				dtype=np.float64)
		else:
            if X.shape[1] != self.theta_.shape[1]:
                msg = "Number of features %d does not match previous data %d."
                raise ValueError(msg % (X.shape[1], self.theta_.shape[1]))
			self.sigma_[:, :] -= epsilon
		
		
		classes = self.classes_
		
		unique_y = np.unique(y)
		unique_y_in_classes = in1d(unique_y, classes)
		
		if not np.all(unique_y_in_classes):
            raise ValueError("The target label(s) %s in y do not exist in the "
                             "initial classes %s" %
                             (unique_y[~unique_y_in_classes], classes))
		
		for y_i in unique_y:
            i = classes.searchsorted(y_i)
			X_i = X[y == y_i, :]

		if sample_weight is not None:
                sw_i = sample_weight[y == y_i]
				N_i = sw_i.sum()
				
		else:
                sw_i = None
				N_i = X_i.shape[0]




			new_theta, new_sigma = self._update_mean_variance(
							self.class_count_[i], self.theta_[i, :], self.sigma_[i, :],
							X_i, sw_i)	

			self.theta_[i, :] = new_theta
            self.sigma_[i, :] = new_sigma
			self.class_count_[i] += N_i	


		self.sigma_[:, :] += epsilon

		if self.priors is None:
			 self.class_prior_ = self.class_count_ / self.class_count_.sum()
		
		return self
		
		
	def _joint_log_likelihood(self, X):
        check_is_fitted(self, "classes_")

        X = check_array(X)
        joint_log_likelihood = []
        for i in range(np.size(self.classes_)):
            jointi = np.log(self.class_prior_[i])
            n_ij = - 0.5 * np.sum(np.log(2. * np.pi * self.sigma_[i, :]))
            n_ij -= 0.5 * np.sum(((X - self.theta_[i, :]) ** 2) /
                                 (self.sigma_[i, :]), 1)
            joint_log_likelihood.append(jointi + n_ij)

        joint_log_likelihood = np.array(joint_log_likelihood).T
        return joint_log_likelihood
	
		
		
# set features :
features = []

with open("features.csv") as feat:
    entryreader = csv.reader(feat, delimiter=',')
    for row in entryreader:
        features.append(row)

featNames = features[0]
features = features[1:]

print len(features), len(features[0])

# set labels :
labels = []

with open("labels.csv") as lbl:
    entryreader = csv.reader(lbl, delimiter=',')
    for row in entryreader:
        labels.append(row)

labelNames = labels[0]
labels = labels[1:]

print len(labels), len(labels[0])

labels = np.array(labels).astype(int)
features = np.array(features).astype(float)


labels = np.array(labels)
features = np.array(features)


featIdxMap = dict()
for i in range(len(featNames)):
    featIdxMap[featNames[i]] = i

correlation = []

with open("feature_correlation_results.csv") as corr:
    entryreader = csv.reader(corr, delimiter=',')
    for row in entryreader:
        correlation.append(row)
		
# Test and train :


titleYearIdx = -1

for i in range(len(featNames)):
    if featNames[i] == 'title_year':
        titleYearIdx = i

print 'year index = ', titleYearIdx

trainRows = []
testRows = []

for i in range(len(features)):
    if float(features[i][titleYearIdx]) > 2010:
        testRows.append(i)
    else:
        trainRows.append(i)

print len(trainRows) / len(features) , len(testRows) / len(features)

favoriteCols = []

for i in range(1, len(correlation)):
    if correlation[i][0] in featIdxMap:
        if math.fabs(float(correlation[i][1])) > 0.1:
            print correlation[i][0]
            favoriteCols.append(featIdxMap[correlation[i][0]])

print 'favoritCols = ', len(favoriteCols)

start_time = time.time()

# ??
tmp = []
for i in range(len(testRows)):
    if labels[testRows[i]][0] == 1:
        tmp.append(testRows[i])

posNo = len(tmp)

for i in range(len(testRows)):
    if labels[testRows[i]][0] == 0 and posNo > 0:
        tmp.append(testRows[i])
        posNo -= 1

testRows = tmp

print 'test length = ', len(testRows)

classes = [
        'Nominated Best Picture',
		'Won Best Picture',
    ]


clf=GaussianNB()
clf.fit(features[trainRows, :])[:, favoriteCols], labels[trainRows, 0], sample_weight=None)
clf._update_mean_variance(n_past, mu, var, X, sample_weight=None)
clf.partial_fit(features[trainRows, :])[:, favoriteCols], labels[trainRows, 0], classes=classes, sample_weight=None)
clf._partial_fit(features[trainRows, :])[:, favoriteCols], labels[trainRows, 0], classes=classes, _refit=False,sample_weight=None)
clf._joint_log_likelihood(features[trainRows, :])[:, favoriteCols])

print 'accuracy = %f' %(np.mean((y_test-y_pred)==0))

		
		
		
		
		
		
		
		
		
		
		
		
		
		 
			
			
			
		 


	 











