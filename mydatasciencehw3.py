# -*- coding: utf-8 -*-
"""MyDataScienceHW3

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1dScQLGrYbdKvytIrQwxzAnO3X4m1Qzfq
"""

import json

 

def load_file(file_path):

    """

    :param file_path: path to the json file

    :return: an array in which each entry is tuple [text, classification label]

    """

    with open(file_path) as json_file:

        raw_data = json.load(json_file)

        return convert_data(raw_data)

def convert_data(raw_data):

    data = []

    for elem in raw_data:

        data.append([elem["comment"], elem["label"]])

    return data

import pandas as pd

data = load_file("Bug_tt.json")
df = pd.DataFrame(data, columns = ['Text', 'Label'])
# df.head()
# df.groupby('Label').nunique().plot(kind='bar')
# df.loc[3]['Text']

import re

import nltk

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split

from sklearn.feature_selection import chi2

import numpy as np

df['text_parsed_1'] = df['Text'].str.replace("\r", " ")

df['text_parsed_1'] = df['text_parsed_1'].str.replace("\n", " ")

df['text_parsed_1'] = df['text_parsed_1'].str.replace("  ", " ")

df['text_parsed_1'] = df['text_parsed_1'].str.replace('"', '')

df['text_parsed_1'] = df['text_parsed_1'].str.lower()

df.loc[3]['text_parsed_1']

punctuation_signs = list("?:!.,;")

df['text_parsed_2'] = df['text_parsed_1']

for punct_sign in punctuation_signs:

    df['text_parsed_2'] = df['text_parsed_2'].str.replace(punct_sign, '')

df['text_parsed_2'] = df['text_parsed_2'].str.replace("'s", "")

df.loc[3]['text_parsed_2']

# Downloading punkt and wordnet from NLTK

nltk.download('punkt')

print("------------------------------------------------------------")

nltk.download('wordnet')

# Saving the lemmatizer into an object

wordnet_lemmatizer = WordNetLemmatizer()

df.head()

nrows = len(df)

lemmatized_text_list = []

for row in range(0, nrows):

    # Create an empty list containing lemmatized words

    lemmatized_list = []

   

    # Save the text and its words into an object

    text = df.loc[row]['text_parsed_2']

    text_words = text.split(" ")

 

    # Iterate through every word to lemmatize

    for word in text_words:

        lemmatized_list.append(wordnet_lemmatizer.lemmatize(word, pos="v"))

       

    # Join the list

    lemmatized_text = " ".join(lemmatized_list)

   

    # Append to the list containing the texts

    lemmatized_text_list.append(lemmatized_text)

df['text_parsed_3'] = lemmatized_text_list

 

print(df.loc[3]['Text'])

print(df.loc[3]['text_parsed_3'])

# Downloading the stop words list

nltk.download('stopwords')

# Loading the stop words in english

stop_words = list(stopwords.words('english'))

stop_words[0:10]

df['text_parsed_4'] = df['text_parsed_3']

for stop_word in stop_words:

    regex_stopword = r"\b" + stop_word + r"\b"

    df['text_parsed_4'] = df['text_parsed_4'].str.replace(regex_stopword, '')

 

print(df.loc[3]['Text'])

print(df.loc[3]['text_parsed_4'])

df.head(1)

# remove the intermediate columns

list_columns = ["Text", "Label", "text_parsed_4"]

df = df[list_columns]

df = df.rename(columns={'text_parsed_4': 'text_parsed'})

 

df.head(1)

label_codes = {

    'Bug': 1,

    'Not_Bug': 0

}

# label mapping

df['label_code'] = df['Label']

df = df.replace({'label_code':label_codes})

df.head()

# train - test split

X_train, X_test, y_train, y_test = train_test_split(df['text_parsed'],

                                                    df['label_code'],

                                                    test_size=0.15,

                                                    random_state=42)

# Text representation

 

# Parameter election

ngram_range = (1,2)

min_df = 1

max_df = 1.

max_features = 300

 

tfidf = TfidfVectorizer(encoding='utf-8',

                        ngram_range=ngram_range,

                        stop_words=None,

                        lowercase=False,

                        max_df=max_df,

                        min_df=min_df,

                        max_features=max_features,

                        norm='l2',

                        sublinear_tf=True)

                       

features_train = tfidf.fit_transform(X_train).toarray()

labels_train = y_train

print(features_train.shape)

 

features_test = tfidf.transform(X_test).toarray()

labels_test = y_test

print(features_test.shape)

# let's train a SVM

 

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn import svm

from pprint import pprint

from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

from sklearn.model_selection import ShuffleSplit

import matplotlib.pyplot as plt

 

svc_0 =svm.SVC(random_state=42)

 

print('Parameters currently in use:\n')

pprint(svc_0.get_params())

# let's train a SVM

 

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn import svm

from pprint import pprint

from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

from sklearn.model_selection import ShuffleSplit

import matplotlib.pyplot as plt

 

svc_0 =svm.SVC(random_state=42)

 

print('Parameters currently in use:\n')

pprint(svc_0.get_params())

# C

C = [.0001, .001, .01, -1, 1]

 

# gamma

gamma = [.0001, .001, .01, .1, 1, 10, 100]

 

# degree

degree = [1, 2, 3, 4, 5]

 

# kernel

kernel = ['linear', 'rbf', 'poly']

 

# probability

probability = [True]

 

# Create the random grid

random_grid = {'C': C,

              'kernel': kernel,

              'gamma': gamma,

              'degree': degree,

              'probability': probability

             }

 

pprint(random_grid)

# First create the base model to tune

svc = svm.SVC(random_state=42)

 

# Definition of the random search

random_search = RandomizedSearchCV(estimator=svc,

                                   param_distributions=random_grid,

                                   n_iter=50,

                                   scoring='accuracy',

                                   cv=3,

                                   verbose=1,

                                   random_state=42)

 

# Fit the random search model

random_search.fit(features_train, labels_train)

print("The best hyperparameters from Random Search are:")

print(random_search.best_params_)

print("")

print("The mean accuracy of a model with these hyperparameters is:")

print(random_search.best_score_)

# random search was better

best_svc = random_search.best_estimator_

# fit the model

best_svc.fit(features_train, labels_train)

svc_pred = best_svc.predict(features_test)

print(svc_pred)

print(best_svc.predict_proba(features_test))

# Training accuracy

print("The training accuracy is: ")

print(accuracy_score(labels_train, best_svc.predict(features_train)))

 

# Test accuracy

print("The test accuracy is: ")

print(accuracy_score(labels_test, svc_pred))

 

# Classification report

print("Classification report")

print(classification_report(labels_test,svc_pred))

import seaborn as sns

 

# Confusion matrix

aux_df = df[['Label', 'label_code']].drop_duplicates().sort_values('label_code')

conf_matrix = confusion_matrix(labels_test, svc_pred)

 

plt.figure(figsize=(12.8,6))

 

sns.heatmap(conf_matrix,

            annot=True,

            xticklabels=aux_df['Label'].values,

            yticklabels=aux_df['Label'].values,

            cmap="Blues")

 

plt.ylabel('Predicted')

plt.xlabel('Actual')

plt.title('Confusion matrix')

plt.show()

# Let's see if the hyperparameter tuning process has returned a better model:

 

base_model = svm.SVC(random_state = 8)
base_model.fit(features_train, labels_train)

print("Base Model")
print("Accuracy: " + str(accuracy_score(labels_test, base_model.predict(features_test))))
print("F1 score: "+ str(f1_score(labels_test, base_model.predict(features_test))))

print("\nRandom Model")
print("Accuracy: " + str(accuracy_score(labels_test, best_svc.predict(features_test))))
print("F1 score: "+ str(f1_score(labels_test, best_svc.predict(features_test))))