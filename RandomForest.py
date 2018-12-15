import pandas as pd
import numpy as np
import csv
from sklearn import cross_validation, ensemble, preprocessing, metrics
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import f1_score
from sklearn import model_selection

# Load data
a = pd.read_csv('TraData.csv',sep = ',', encoding = 'utf-8')
a = pd.DataFrame(a)

## Exam for unknown data. ##
testdata = pd.read_csv('input.csv', sep = ',', encoding = 'utf-8')
testdata = pd.DataFrame(testdata)

frames = [a, testdata]
train = pd.concat(frames)

del train['adx']
# del train['dclkVerticals']
del train['ip']
del train['spaceType']

label_encoder = preprocessing.LabelEncoder()
#adx = label_encoder.fit_transform(train["adx"])
#spaceType = label_encoder.fit_transform(train["spaceType"])
spaceId = label_encoder.fit_transform(train["spaceId"])
spaceCat = label_encoder.fit_transform(train["spaceCat"].astype(str))
adType = label_encoder.fit_transform(train["adType"])
#ip = label_encoder.fit_transform(train["ip"])
os = label_encoder.fit_transform(train["os"])
deviceType = label_encoder.fit_transform(train["deviceType"])
dclkVerticals = label_encoder.fit_transform(train["dclkVerticals"].astype(str))
publisherId = label_encoder.fit_transform(train["publisherId"])
campaignId = label_encoder.fit_transform(train["campaignId"])
advertiserId = label_encoder.fit_transform(train["advertiserId"])


# Build train and test set
X = pd.DataFrame([
#    adx,
#    spaceType,
    spaceId,
    spaceCat,
    adType,
    os,
#    ip,
    deviceType,
    dclkVerticals,
    publisherId,
    campaignId,
    advertiserId
]).T

C = a["click"]

# build rf model
# kfold = model_selection.KFold(n_splits = 4)
# model_selection.cross_val_score(rf, X, y, cv = kfold, scoring = 'f1')

D = X.loc[1:961457]
T = X.loc[961457:len(train)]

# train_X, test_X, train_y, test_y = cross_validation.train_test_split(D, C, test_size = 0.35)
rf = RandomForestClassifier(class_weight = {0:1, 1:10}, n_estimators = 40)
rf_fit = rf.fit(D, C)

testdata_y_predicted = rf.predict(T)

testdata_y_predicted[testdata_y_predicted < 0.5] = 0
testdata_y_predicted[testdata_y_predicted >= 0.5] = 1

header_test = ["click"]
output = pd.DataFrame(testdata_y_predicted)
output.to_csv('Predict.csv', index = True, header = header_test)

'''
prediction
test_y_predicted = rf.predict(test_X)

accuracy
accuracy = metrics.accuracy_score(test_y, test_y_predicted)


print(metrics.classification_report(test_y, test_y_predicted#))

test_y_predicted[test_y_predicted < 0.5] = 0
test_y_predicted[test_y_predicted >= 0.5] = 1

print(confusion_matrix(test_y, test_y_predicted))
print('acc:', (accuracy))
print('f1_score:', (f1_score(test_y, test_y_predicted, average = 'macro')))
'''
