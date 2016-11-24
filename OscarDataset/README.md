Some important points about the new datasets:

1) Use dataLoader.py to load the feat_train, feat_test, label_train, label_test

2) In some rows the new feature_correlation.csv has 0, this is because we only consider training rows; therefore, some columns will be all 0 and have standard-deviation of 0, so I put pearson_corr value 0 for them so that they won't be selected

3) feat_train, feat_test are not scaled if you want to scale them use a peice of code provided in dataLoader.py
