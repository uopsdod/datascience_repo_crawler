# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import math
import quandl
import pandas as pd
import numpy as np
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression

# def print_hi(name):
#     # Use a breakpoint in the code line below to debug your script.
#     print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.

quandl.ApiConfig.api_key = "QWenB2Q_aAZ9ScPPApUb"
df = quandl.get('WIKI/GOOGL')
print(df.head())

df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]
print(df.head())

forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True) # use -99999 to represent missing data

forecast_out = int(math.ceil(0.01 * len(df))) # to predict 10% future from the size of our data set
print('forecast_out(days): ', forecast_out)

df['label'] = df[forecast_col].shift(-forecast_out) # label: what we want to predict
df.dropna(inplace=True)

X = np.array(df.drop(['label'], 1)) # X: features
X = preprocessing.scale(X) # in machine learning, we want to make our features between -1 ~ 1 for performanccce purpose
y = np.array(df['label']) # Y: label

# shuffl test data and divide into two sets
# one for training; the other for verifying
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)


## SVR
# pick classifier/algorithm
# clf = svm.SVR() # one of the classifier - Support Vector Regression (SVR)
for k in ['linear','poly','rbf','sigmoid']:
    clf = svm.SVR(kernel=k)
    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)
    print(k,confidence)

## LinearRegression
# clf = LinearRegression() # one of the classifier
# clf = LinearRegression(n_jobs=-1) # one of the classifier - use all threads available
# clf.fit(X_train, y_train)
# confidence = clf.score(X_test, y_test) # compare with expected results
# print(confidence)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("\nend");
    # print_hi('PyCharm')
    # print(df.head())


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
