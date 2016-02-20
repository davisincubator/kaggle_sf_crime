__author__ = 'kfranko'

import pandas as pd
import numpy as np


import os
os.getcwd()


data_train = pd.read_csv('train.csv')
data_test = pd.read_csv('test.csv')
data_sample_submission = pd.read_csv('sampleSubmission.csv')

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


# select which features of training set will go into model

features = ['Dates', 'Descript', 'DayOfWeek', 'PdDistrict', 'Resolution', 'Address', 'X', 'Y']


test_features = ['Dates', 'DayOfWeek', 'PdDistrict', 'Address', 'X', 'Y']

data_train['PdDistrict'][1000]

data_train['Descript'][1000]

unique_descript = data_train['Descript'].unique()

unique_address = data_train['Address'].unique()

data_train['Resolution'][10600]

data_train['DayOfWeek'][0]

data_train[features].head()

# emily example of setting categories: df["B"] = df["A"].astype('category')

data_train['cat_PdDistrict'] = data_train['PdDistrict'].astype('category')
data_train['cat_DayOfWeek'] = data_train['DayOfWeek'].astype('category')

cat_features = ['cat_PdDistrict', 'cat_DayOfWeek']



# running into errors due to strings; need to preprocess:

from sklearn import preprocessing


le = preprocessing.LabelEncoder()

le.fit(data_train['Category'])
data_train['cat_trans'] = le.transform(data_train.Category)

le.fit(data_train['PdDistrict'])
data_train['district_trans'] = le.transform(data_train.PdDistrict)

le.fit(data_train['DayOfWeek'])
data_train['dayofweek_trans'] = le.transform(data_train.DayOfWeek)

le.fit(data_train['Address'])
data_train['address_trans'] = le.transform(data_train.Address)

data_train['Dates'] = pd.to_datetime(data_train['Dates'])

data_train['Month'] = data_train.Dates.map(lambda x: x.month)

data_train['Hour'] = data_train.Dates.map(lambda x: x.hour)

features = ['district_trans', 'dayofweek_trans', 'address_trans']

test_features = ['district_trans', 'dayofweek_trans', 'address_trans']

le.fit(data_train['Address'])
data_train['address_trans'] = le.transform(data_train.Address)


le = preprocessing.LabelEncoder()

le.fit(data_test['DayOfWeek'])
data_test['dayofweek_trans'] = le.transform(data_test.DayOfWeek)

le.fit(data_test['Address'])
data_test['address_trans'] = le.transform(data_test.Address)

le.fit(data_test['PdDistrict'])
data_test['district_trans'] = le.transform(data_test.PdDistrict)


features_train = data_train[features]
category_train = data_train.cat_trans
category_predict  = []
features_test  = data_test[test_features]
sales_test     = np.NaN


from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import SGDClassifier


# Create/Instantiate Random Forest Classifier
sgd = SGDClassifier()

# fit model
sgd.fit(features_train, category_train)

# Generate predictions
category_predict = sgd.predict(features_test)

unique_predictions = np.bincount(category_predict)


# try random forest:

rf = RandomForestClassifier()

# Train model
rf.fit(features_train, category_train)

# Generate predictions
random_forest_predict = rf.predict(features_test)


rf_unique_predictions = np.bincount(random_forest_predict)

# look at feature importances:

feature_importance = rf.feature_importances_

# make importances relative to max importance
feature_importance = 100.0 * (feature_importance /
                              feature_importance.max())

import matplotlib.pyplot as plt

sorted_idx         = np.argsort(feature_importance)
pos                = np.arange(sorted_idx.shape[0]) + .5

plt.subplot(1, 2, 2)
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, features)
plt.xlabel('Relative Importance')
plt.title( 'Variable Importance')
plt.show()



# predict:

pred = clf.predict(features_test)


from sklearn.metrics import accuracy_score

accuracy = accuracy_score(labels_test, pred)

print 'accuracy:', accuracy