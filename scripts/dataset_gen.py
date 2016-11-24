#!/usr/bin/python

import csv

"""
Separates the training from test data.
"""

trinds = []
tsinds = []

training_data = []
testing_data = []

training_index = 1
testing_index = 1

with open('training_indices.csv', 'rb') as traincsvfile:
    entryreader = csv.reader(traincsvfile, delimiter=',')
    for row in entryreader:
        trinds.append(int(row[0]))

with open('testing_indices.csv', 'rb') as testcsvfile:
    entryreader = csv.reader(testcsvfile, delimiter=',')
    for row in entryreader:
        tsinds.append(int(row[0]))

with open('movies_all_features.csv', 'rb') as datacsvfile:
    entryreader = csv.reader(datacsvfile, delimiter=',')
    for row in entryreader:
        if training_index in trinds or training_index == 1:
            training_data.append(row)
        training_index += 1

with open('movies_all_features.csv', 'rb') as datacsvfile:
    entryreader = csv.reader(datacsvfile, delimiter=',')
    for row in entryreader:
        if testing_index in tsinds or testing_index == 1:
            testing_data.append(row)
        testing_index += 1

with open('training_data.csv', 'wb') as outcsvfile:
    outwriter = csv.writer(outcsvfile, delimiter=',')
    for new_row in training_data:
        outwriter.writerow(new_row)

with open('testing_data.csv', 'wb') as outcsvfile:
    outwriter = csv.writer(outcsvfile, delimiter=',')
    for new_row in testing_data:
        outwriter.writerow(new_row)
