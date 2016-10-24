#!/usr/bin/python

from sklearn.naive_bayes import GaussianNB

# Data preprocessing:
features = []
labels = []

# Load data:
with open("../Dataset/Final.csv", "rb") as dataset:
    entryreader = csv.reader(dataset, delimiter=',')
