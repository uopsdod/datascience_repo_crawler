import json

import nltk
import pandas as pd

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

import nltk
import ssl

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer


class NLPService:
    def __init__(self):
        # Downloading punkt and wordnet from NLTK
        self._download_nltk()

        pass

    def _download_nltk(self):
        # ref: https://stackoverflow.com/questions/38916452/nltk-download-ssl-certificate-verify-failed
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context
        # nltk.download()
        nltk.download('punkt', halt_on_error=False)
        nltk.download('wordnet', halt_on_error=False)
        nltk.download('stopwords')

    def lemmatize(self, df, feautre_name):
        wordnet_lemmatizer = WordNetLemmatizer()

        # Saving the lemmatizer into an object
        nrows = len(df)
        lemmatized_text_list = []
        for row in range(0, nrows):
            lemmatized_text = self.get_lemmatized_list(df, feautre_name, row, wordnet_lemmatizer)
            lemmatized_text_list.append(lemmatized_text)

        df[feautre_name] = lemmatized_text_list

    def get_lemmatized_list(self, df, feautre_name, row, wordnet_lemmatizer):
        # Create an empty list containing lemmatized words
        lemmatized_list = []
        # Save the text and its words into an object
        text = df.loc[row][feautre_name]
        text_words = text.split(" ")
        # Iterate through every word to lemmatize
        for word in text_words:
            lemmatized_list.append(wordnet_lemmatizer.lemmatize(word, pos="v"))
        # Join the list
        lemmatized_text = " ".join(lemmatized_list)
        return lemmatized_text

    def convert_to_tfidf_text_representation(self, X_train, feature_text):

        X_train_feature = X_train[feature_text]

        ngram_range = (1,2) # unigrams and bigrams
        min_df = 1
        max_df = 1
        max_features = 20

        tfidf_vectorizer = TfidfVectorizer(encoding='utf-8',
                                ngram_range=ngram_range,
                                stop_words=None,
                                lowercase=False,
                                max_df=max_df,
                                min_df=min_df,
                                max_features=max_features,
                                norm='l2',
                                sublinear_tf=True)

        features_tfidf = tfidf_vectorizer.fit_transform(X_train_feature).toarray()
        feature_names_out = tfidf_vectorizer.get_feature_names_out()
        feature_names_out = [feature_text + "_" + elem for elem in feature_names_out]
        features_tfidf_df = pd.DataFrame(data=features_tfidf, columns=feature_names_out)
        return features_tfidf_df

        # features_tfidf = tfidf_vectorizer.fit_transform(df[feature_text]).toarray()
        # feature_names_out = tfidf_vectorizer.get_feature_names_out()
        # feature_names_out = [feature_text + "_" + elem for elem in feature_names_out]

        # df_tfidf = pd.DataFrame(data=features_tfidf, columns=feature_names_out)
        # return df_tfidf

        # df[feature_text+'_tfidf'] = features_tfidf
        # print(features_tfidf.shape)
        # print('')
        # return features_tfidf

    def remove_punctuation_signs(self, df, feature_name):
        punctuation_signs = list("!?:.,;")
        for punct_sign in punctuation_signs:
            df.loc[:, feature_name] = df.loc[:, feature_name].str.replace(punct_sign, '', regex=True)
            # df[feature_name] = feature_replaced

    def remove_stopwords(self, df, feature_name):
        stop_words = list(stopwords.words('english'))
        for stop_word in stop_words:
            regex_stopword = r"\b" + stop_word + r"\b"
            df.loc[:, feature_name] = df.loc[:, feature_name].str.replace(regex_stopword, '', regex=True)
