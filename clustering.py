from preprocessor import DataPreprocessor


preprocessor = DataPreprocessor(['Nominated Best Picture',
                                'Won Best Picture', 'Num of Awards'],
                                ['genres', 'plot_keywords'], 'Final.csv')
preprocessor.preprocess()
print len(preprocessor.features)
print len(preprocessor.features[0])
