Some important points about the new datasets:


1) Use dataLoader.py to load the feat_train, feat_test, label_train, label_test

2) In some rows the new feature_correlation.csv has 0, this is because we only consider training rows; therefore, some columns will be all 0 and have standard-deviation of 0, so I put pearson_corr value 0 for them so that they won't be selected

3) feat_train, feat_test are not scaled if you want to scale them use a peice of code provided in dataLoader.py

4) movie_metadata.csv is the normal Kaggel dataset which also has information new added movies by Ali (~250 best picture nominees that were initially missing)

5) movies_all_features.csv is obtained from movie_metadata.csv by adding all nominee, winner, keyword, gener features. (Approximately 8000 features)

6) feat_train, feat_test contain only features that have correlation above 0.1 for at least one of the class labels

7) feat_train, feat_test, label_train, label_test were obtained by running test_train_gen.py (in OscarDataset on github) on movies_all_features.csv

8) feat_train, feat_test have a new feature/column called 'original_row' that you should ignore; it contains the original row number of that instance in movie_metadata.csv or similarly movies_all_features.csv
