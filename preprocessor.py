#!/usr/bin/python

import csv


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
        with open(self.csv_filename, "rb") as dataset:
            entryreader = csv.reader(dataset, delimiter=',')
            for row in entryreader:
                if counter == 0:
                    # The first row is column headings (special treatment).
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
                    self.features.append(data_row)
                counter += 1
