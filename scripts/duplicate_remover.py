#!/usr/bin/python

import csv

new_attributes = []

with open("Final.csv", "rb") as csvfile:
    csvfile.seek(0)
    entryreader = csv.reader(csvfile, delimiter=',')
    for row in entryreader:
        new_attributes.append(row[:8182])
with open('final_dataset_no_duplicates.csv', 'wb') as outcsvfile:
    outwriter = csv.writer(outcsvfile, delimiter=',')
    for new_row in new_attributes:
        outwriter.writerow(new_row)
