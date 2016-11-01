#!/usr/bin/python

import csv
import random
import numbers
import decimal
from sklearn import preprocessing


class DataPreprocessor:
    """This class transforms your csv files into inputs suitable for use with
    scikit-learn."""

    # Class variables
    features = []
    colmap = dict()
    class_labels = [
        'Nominated Best Picture',
        'Won Best Picture',
        'Num of Awards'
    ]
    labels = []
    class_label_index = []
    ignore = ['genres', 'plot_keywords']
    csv_filename = ""
    column_headers = []
    features_numerical = []
    labels_numerical = []

    def __init__(self, label_column_names_list, column_names_to_ignore_list,
                 csv_filename):
        """
        Instantiates a new DataPreprocessor object.
        Parameters:
            label_column_names_list - A list of column names that indicate
            class labels.
            column_names_to_ignore_list - A list of column names to ignore.
            csv_filename - The name of the source csv file.
        """
        self.csv_filename = csv_filename
        self.class_labels = label_column_names_list
        self.ignore = column_names_to_ignore_list
        for heading in self.class_labels:
            self.labels.append([])

    def preprocess(self):
        """
        Reads and preprocesses the data from a csv file to produce a large
        2d-arrary containing features, and a smaller 2d-array containing
        multiple labels where each is aligned with the features.
        """
        # Load data:
        counter = 0
        headers_processed = False
        temp_headers = []
        with open(self.csv_filename, "rb") as dataset:
            entryreader = csv.reader(dataset, delimiter=',')
            for row in entryreader:
                if counter == 0:
                    # The first row is column headings (special treatment).
                    temp_headers = list(row)
                    for column in row:
                        self.colmap[column] = row.index(column)
                else:
                    # Need to separate features from ignored and class label
                    # fields.
                    data_row = []
                    skippable_attributes = list(self.ignore)
                    skippable_attributes.extend(self.class_labels)
                    skippable_attributes = [self.colmap[x] for x in
                                            skippable_attributes]
                    for class_label in self.class_labels:
                        label_index = self.class_labels.index(class_label)
                        column_index = row[self.colmap[class_label]]
                        self.labels[label_index].append(column_index)
                    for index, value in enumerate(row):
                        if index not in skippable_attributes:
                            data_row.append(value)
                            if not headers_processed:
                                self.column_headers.append(temp_headers[index])
                    headers_processed = True
                    self.features.append(data_row)
                counter += 1

    def split_features(self):
        """
        Splits the already preprocessed feature matrix vertically instead of
        horizontally, so that you get a single list per feature, perfectly
        aligned with the class labels. Requires that the features already be
        preprocessed.
        """
        # Initialize the result arrays:
        result = []
        for feature in self.features[0]:
            result.append([])

        for index1, value1 in enumerate(self.features):
            for index2, value2 in enumerate(self.features[index1]):
                result[index2].append(value2)

        return result

    def numerify(self):
        """
        Converts all non-numerical attributes (features and labels) into
        numerical attributes.
        """
        # For features:
        temp = self.split_features()
        for i in range(len(temp)):
            self.features_numerical.append([])
        for index, feature_vector in enumerate(temp):
            test_index = 0
            if (feature_vector[test_index] == ''):
                test_index = random.randrange(1, len(feature_vector))
            if (not isinstance(feature_vector[test_index], numbers.Number) and
                    not isinstance(feature_vector[test_index],
                                   decimal.Decimal)):
                encoder = preprocessing.LabelEncoder()
                encoder.fit(feature_vector)
                self.features_numerical[index] = encoder.transform(
                        feature_vector)
        # Rearrange features into rows:
        self.features_numerical = map(list, zip(*self.features_numerical))
        # For labels:
        for label_vector in self.labels:
            test_index = 0
            if (label_vector[test_index] == ''):
                test_index = random.randrange(1, len(label_vector))
            if (not isinstance(label_vector[test_index], numbers.Number) and
                    not isinstance(label_vector[test_index],
                                   decimal.Decimal)):
                encoder = preprocessing.LabelEncoder()
                encoder.fit(label_vector)
                self.labels_numerical.append(encoder.transform(label_vector))

    def add_feature(self, new_feat):
        """
        Adds a new list representing a feature and its values to the matrix of
        existing features. The list must be of a length equal to that of the
        total tuples in the dataset and aligned with it.
        """
        if not isinstance(new_feat, list):
            raise TypeError('Invalid parameter type. Expecting type: list')
        if len(new_feat) != len(self.features):
            raise IndexError('Invalid list length')
        # Add the feature values to the tuples (assuming alignment)
        for index1, vector1 in enumerate(self.features):
            self.features[index1].append(new_feat[index1])
