#!/usr/bin/python

import csv

rows = 0
old_attribute_index_map = dict()
new_attribute_index_map = dict()
genre_map = dict()
keyword_map = dict()
new_attributes = []

with open("movie_metadata.csv", "rb") as csvfile:
    csvfile.seek(0)
    entryreader = csv.reader(csvfile, delimiter=',')
    # Create the first row (column names):
    cols = []
    for row in entryreader:
        # The first row is the column names, we only want two:
        if rows == 0:
            for attribute in row:
                if attribute in ("genres", "plot_keywords"):
                    old_attribute_index_map[attribute] = row.index(attribute)
                cols.append(attribute)
        rows += 1
        for attribute in row:
            prefix = ""
            process = False
            is_genre = False
            if row.index(attribute) == old_attribute_index_map["genres"]:
                prefix = "genre="
                process = True
                is_genre = True
            elif row.index(attribute) == old_attribute_index_map["plot_keywords"]:
                prefix = "keyword="
                process = True
            if process:
                process = False
                splits = attribute.split('|')
                for split in splits:
                    if is_genre:
                        genre_map[prefix+split] = len(new_attribute_index_map) - 1
                    else:
                        keyword_map[prefix+split] = len(new_attribute_index_map) - 1
    # Update the first row with the new features
    for key, value in genre_map.items():
        cols.append(key)
        new_attribute_index_map[key] = len(new_attribute_index_map) - 1
    for key, value in keyword_map.items():
        cols.append(key)
        new_attribute_index_map[key] = len(new_attribute_index_map) - 1
    new_attributes.append(cols)
    # Add entries as appropriate:
    csvfile.seek(0)
    rows = 0
    for row in entryreader:
        # We need to skip the first row.
        if rows != 0:
            temp = row
            temp.extend([0] * (len(cols) - len(row)))
            for attribute in row:
                prefix = ""
                process = False
                if row.index(attribute) == old_attribute_index_map["genres"]:
                    prefix = "genre="
                    process = True
                elif row.index(attribute) == old_attribute_index_map["plot_keywords"]:
                    prefix = "keyword="
                    process = True
                if process:
                    process = False
                    splits = attribute.split('|')
                    for split in splits:
                        temp[cols.index(prefix+split)] = 1
            row.extend(temp)
            new_attributes.append(row)
        rows += 1

with open('split_features.csv', 'wb') as outcsvfile:
    outwriter = csv.writer(outcsvfile, delimiter=',')
    for new_row in new_attributes:
        outwriter.writerow(new_row)
