#!/usr/bin/python

import csv

# Data preprocessing:
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

for heading in class_labels:
    labels.append([])


# Load data:
counter = 0
with open("Final.csv", "rb") as dataset:
    entryreader = csv.reader(dataset, delimiter=',')
    for row in entryreader:
        # Everything after column 8181 is a duplicate. File still needs cleaning up.
        row = row[:8182]
        if counter == 0:
            # The first row is just column headings. It gets special treatment.
            for column in row:
                colmap[column] = row.index(column)
        else:
            # Need to separate features from processed and class label fields.
            data_row = []
            skippable_attributes = list(ignore)
            skippable_attributes.extend(class_labels)
            skippable_attributes = [colmap[x] for x in skippable_attributes]
            for class_label in class_labels:
                labels[class_labels.index(class_label)].append(row[colmap[class_label]])
            for index, value in enumerate(row):
                if index >= len(row):
                    break
                if index not in skippable_attributes:
                    data_row.append(value)
            features.append(data_row)
        counter += 1
