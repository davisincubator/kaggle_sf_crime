__author__ = 'kfranko'

import pandas as pd
import numpy as np


import os
os.getcwd()


data_train = pd.read_csv('train.csv')
data_test = pd.read_csv('test.csv')

print 'number of rows in training set:', len(data_train) # 878049 rows
print 'number of rows in test set:', len(data_test) # 884262 rows

training_data_columns = list(data_train.columns.values) # 9 columns
# (includes 'Category' (this is the target variable we are trying to predict)
# and 'Resolution' (how the crime incident was resolved (only in train.csv)
# and 'Descript' - detailed description of the crime incident (only in train.csv))

test_data_columns = list(data_test.columns.values) # 7 columns
training_data_columns = list(data_train.columns.values) # 7 columns

''' let's make another train/test split using the training data for validation purposes;
an alternative would be just having members make submissions through kaggle
to get our score (creating our own for internal might be easier, also gets around submission limit)
'''

from sklearn import cross_validation

features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(
...     features, labels, test_size=0.3, random_state=42)




