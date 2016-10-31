#!/usr/bin/python

from preprocessor import DataPreprocessor

"""
Applies clustering algorithms to the movie dataset in order to discover
meaningful relationships between data tuples.
"""

# Author: Omar Elazhary <omazhary@gmail.com>
# License: None

preprocessor = DataPreprocessor(['Nominated Best Picture',
                                'Won Best Picture', 'Num of Awards'],
                                ['genres', 'plot_keywords'],
                                'final_dataset_no_duplicates.csv')
preprocessor.preprocess()
print len(preprocessor.features)
print len(preprocessor.features[0])
