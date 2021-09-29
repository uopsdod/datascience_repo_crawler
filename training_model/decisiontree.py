from scipy.io import arff
import pandas as pd
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

data = arff.loadarff('../training_dataset/feature-envy.arff')
df = pd.DataFrame(data[0]) # ? not yet clear

number_of_rows_to_show = 8
df.head(number_of_rows_to_show)

print(df)

x_row_start = 0;
x_row_end = len(df) # 不包含
x_col_start = 0;
x_col_end = -1;
X = df.iloc[x_row_start:x_row_end, x_col_start:x_col_end].values

print(X)

y_row_start = 0;
y_row_end = len(df) # 不包含
y_col_start = -1;
y_col_end = -1;
Y_data = df.iloc[y_row_start:y_row_end, -1].values # ? not clear how -1 works

print(Y_data)

# convert boolean to 1 and 0
encoder = preprocessing.LabelEncoder()
y = encoder.fit_transform(Y_data)

print(y)

# give missing data a fake value
X_copy = df.iloc[:, :-1].copy()

imputer = SimpleImputer(strategy="median")

imputer.fit(X_copy)

new_X = imputer.transform(X_copy)

new_X_df = pd.DataFrame(new_X, columns=X_copy.columns, index=X_copy.index)

new_X_df.info()

# split into training set & test/verify set
# new_X: tabble with makup missing data
# y: table with results
X_train, X_test, y_train, y_test = train_test_split(new_X, y, test_size=0.15, random_state=42)

print(X_train)

# train it

tree_clf = DecisionTreeClassifier()

scores = cross_val_score(tree_clf, X_train, y_train, cv=10, scoring="accuracy")

print("scores: " , scores)

print("scores mean (accuracy): " , str(np.mean(scores)))

scores = cross_val_score(tree_clf, X_train, y_train, cv=10, scoring="f1")

print("scores average (f1): " , str(np.average(scores)))

# more complex combination training and tests
# if your model is overfitting, reducing the number for max_depth is one way to combat overfitting – there are other ways.

depths = np.arange(1, 11)
num_leafs = [1, 5, 10, 20, 50, 100]
param_grid = { 'criterion':['gini','entropy'],'max_depth': depths, 'min_samples_leaf': num_leafs} # ? what is depth

new_tree_clf = DecisionTreeClassifier()
grid_search = GridSearchCV(new_tree_clf, param_grid, cv=10, scoring="accuracy", return_train_score=True)

grid_search_config_info = grid_search.fit(X_train, y_train)
print(grid_search_config_info)
print(grid_search.best_estimator_)

# compare the typical decisiontree training vs. optimized-with-searchgrid one

best_model = grid_search.best_estimator_

tree_clf.fit(X_train, y_train)

# Training set accuracy

print("The training accuracy is: ")

print("DT default: " + str(accuracy_score(y_train, tree_clf.predict(X_train))))

print("DT optimized: " + str(accuracy_score(y_train, best_model.predict(X_train))))



# Test set accuracy

print("\nThe test accuracy is: ")

print("DT default: " + str(accuracy_score(y_test, tree_clf.predict(X_test))))

print("DT optimized: " + str(accuracy_score(y_test, best_model.predict(X_test))))