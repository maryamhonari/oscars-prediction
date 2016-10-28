#!/usr/bin/python

from preprocessor import DataPreprocessor


preprocessor = DataPreprocessor(['Nominated Best Picture',
                                'Won Best Picture', 'Num of Awards'],
                                ['genres', 'plot_keywords'],
                                'final_dataset_no_duplicates.csv')
preprocessor.preprocess()
print len(preprocessor.features)
print len(preprocessor.features[0])
