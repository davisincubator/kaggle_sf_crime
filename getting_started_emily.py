'''
Playing with SF Crime dataset from Kaggle
Feb 19 2016
Group Work Session with DIG peeps
'''

import pandas as pd
import numpy as np


train_df = pd.read_csv('./data/train.csv')
train_df['Dates'] = pd.to_datetime(train_df['Dates'])

# how many arrest categories?
# 39

len(train_df.Category.unique())

# how many types of resolutions
len(train_df.Resolution.unique())
# 17

train_df.Category.value_counts()

'''
The variable that we want to predict is category
'''

# 10 different Police Districts
len(train_df.PdDistrict.unique())

# which districts have the crime per type?

train_df['PdDistrict'] = train_df['PdDistrict'].astype('category')
train_df['Category'] = train_df['Category'].astype('category')

crime_by_district = train_df.groupby(['PdDistrict'])['Category'].value_counts()

crime_by_day_of_week = train_df.groupby(['DayOfWeek'])['Category'].value_counts()


# in the training set we have 878K observations of 9 variables
# maybe try  making classifier based on district and day of week just to see how it does


'''
do some feature engineering and get text labeled data into integers

ex - need category to correspond to number

'''

from sklearn import preprocessing
# use label encoder to transform non numerical labels into numerical labels
le = preprocessing.LabelEncoder()

le.fit(train_df['Category'])
train_df['cat_trans'] = le.transform(train_df.Category)

le.fit(train_df['PdDistrict'])
train_df['district_trans'] = le.transform(train_df.PdDistrict)

le.fit(train_df['Address'])
train_df['address_trans'] = le.transform(train_df.Address)

le.fit(train_df['DayOfWeek'])
train_df['day_week_trans'] = le.transform(train_df.DayOfWeek)

train_df['Month'] = train_df.Dates.map(lambda x: x.month)

tr_df1 = train_df[::2]
tr_df2 = train_df[1::2]


from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
features_train = tr_df1[['district_trans', 'address_trans', 'day_week_trans']]
labels_train = tr_df1['cat_trans']
clf = SGDClassifier()
clf.fit(features_train, labels_train)

features_test = tr_df2[['district_trans', 'address_trans', 'day_week_trans']]
labels_test = tr_df2['cat_trans']

pred = clf.predict(features_test)

accuracy_score(pred, labels_test)
# 0.03 accuracy! TERRIBLE!

from sklearn.ensemble import RandomForestClassifier

clf2 = RandomForestClassifier(n_estimators= 100, max_depth= 30, min_samples_split=30)
clf2.fit(features_train, labels_train)
pred2 = clf2.predict(features_test)
accuracy_score(pred2, labels_test)
# 0.24 so already better - with no parameters

# clf2 = RandomForestClassifier(n_estimators= 100, max_depth= 30, min_samples_split=30)
# gave accuracy of 0.26


'''
let's try getting better with features
'''


# make time of day feature

train_df['hour'] = train_df.Dates.map(lambda x: x.hour)
crime_by_hour = train_df.groupby(['hour'])['Category'].value_counts()


# 8AM-5PM = daytime
# 5PM - 10PM = evening
# 10PM - 8AM = late night

pd.cut(ages, bins=[0, 18, 35, 70])

train_df['time_day'] = pd.cut(train_df.hour, bins=[0, 8, 17, 22])

pd.cut(train_df.hour, bins=[0, 8, 17, 22, 24])

# this isn't working - sadface