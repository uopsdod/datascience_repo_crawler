import nltk
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

from myutils.datasetservice import DatasetService
from myutils.nlpservice import NLPService
from myutils.printservice import PrintService
from training_model.decisiontree_v2 import ModelDecisionTree
from training_model.logistic_regression import ModelLogisticRegression
from training_model.naivebayes_v2 import ModelNaiveBayes
from training_model.randomforest_v2 import RandomForest
from training_model.svc_v2 import ModelSVC




class Main:
    def __init__(self):
        self.modelLogisticRegression = ModelLogisticRegression()
        self.printService = PrintService()
        self.datasetService = DatasetService()
        self.nlpservice = NLPService()
        self.modelSVC = ModelSVC()
        self.modelNaiveBayes = ModelNaiveBayes()
        self.modelDecisionTree = ModelDecisionTree()
        self.modelRandomForest = RandomForest()

    def start(self):

        dataset_types = ["feature", "bug", "rating", "userexperience"]
        # dataset_types = ["rating"] # debug
        # dataset_types = ["feature"] # debug
        # dataset_types = ["bug"] # debug
        # dataset_types = ["rating", "feature"] # debug

        # model_types = ["logisticregression", "svc", "naivebayes", "decisiontree", "randomforest"]
        # model_types = ["naivebayes", "decisiontree", "randomforest"]
        # model_types = ["svc", "naivebayes", "decisiontree"]
        # model_types = ["naivebayes"]
        model_types = ["svc"]
        # model_types = ["decisiontree"]
        # model_types = ["randomforest"]
        # model_types = ["logisticregression"]

        df_all = None
        df_datasets = {}
        example_count = 0
        for dataset_type in dataset_types:
            df = self.datasetService.load_file(dataset_type)
            df_datasets[dataset_type] = df
            example_count = example_count + len(df)

            if df_all is None:
                df_all = df
            else:
                df_all = df_all.append(df)


        models_svc = {}

        for model_type in model_types:
            accuracy_sum = 0

            self.printService.print_result(f'[{model_type}]', "Accuracy(%)", "F1-score(%)")
            for dataset_type in dataset_types:
                # step: get the features you need from raw dataset
                # features_gleaned = ["title", "comment", "rating", "label"]
                # df = self.datasetService.load_file(dataset_type, features_gleaned)

                label_codes = self.datasetService.get_label_codes(dataset_type)
                df = self.datasetService.convert_to_df(df_all, df_datasets[dataset_type], label_codes)

                # data clean - convert to numeric value - fee type
                self.datasetService.convert_fee_type_to_numeric(df)


                df["length_word_cleaned"] = self.datasetService.get_word_count_of_cleaned_comment(df)

                # step: fill in null/nan fields
                self.datasetService.fill_null_val(df, "title", "")
                self.datasetService.fill_null_val(df, "comment", "")
                self.datasetService.fill_nan_mean(df, "rating")
                self.datasetService.fill_nan_mean(df, "length_word_cleaned")
                self.datasetService.fill_nan_mean(df, "sentiScore")

                # feature scaling
                self.datasetService.standardize_feature(df, "rating")
                self.datasetService.standardize_feature(df, "sentiScore")
                self.datasetService.standardize_feature(df, "fee")
                # self.datasetService.standardize_feature(df, "length_word_cleaned") # not good at all - don't use it

                # step: balance datasets
                df = self.datasetService.balance_dataset(dataset_type, df, "label")

                # step: NLP
                self.nlpservice.lemmatize(df, "title")
                self.nlpservice.lemmatize(df, "comment")
                self.nlpservice.remove_punctuation_signs(df, "comment")
                self.nlpservice.remove_stopwords(df, "comment")

                # step: NLP - convert a sentence to BOW (TF-IDF)
                ngram_range_bigram = (1, 2)
                ngram_range = (1)
                df_comment_train = self.nlpservice.convert_to_tfidf_text_representation(df, "comment")
                df_title_train = self.nlpservice.convert_to_tfidf_text_representation(df, "title")
                df_final = df_comment_train.merge(df_title_train, left_index=True, right_index=True) # join by index

                # step: add metadata features
                # if (dataset_type is "rating"):
                #     df_final["rating"] = df["rating"]
                #     pass
                # elif (dataset_type is "bug"):
                #     df_final["rating"] = df["rating"]
                # elif (dataset_type is "feature"):
                #     df_final["rating"] = df["rating"]
                # elif (dataset_type is "userexperience"):
                #     df_final["rating"] = df["rating"]

                df_final["rating"] = df["rating"]
                df_final["length_word_cleaned"] = df["length_word_cleaned"]
                df_final["sentiScore"] = df["sentiScore"]
                df_final["fee"] = df["fee"]

                # step: add result class
                df_final["label"] = df["label"]

                # step: split data for training and testing
                (X_train, X_test, y_train, y_test) = self.datasetService.split_dataset(df_final, dataset_type)

                # X_train_minmax = min_max_scaler.transform(X_train)
                # X_train["length_word_cleaned"] = X_train_minmax["length_word_cleaned"]
                # X_test_minmax = min_max_scaler.transform(X_test)

                if (model_type == "svc"):
                    accuracy_now = self.modelSVC.train_model(models_svc, dataset_type, X_test, X_train, y_test, y_train)
                elif (model_type == "naivebayes"):
                    accuracy_now = self.modelNaiveBayes.train_model(models_svc, dataset_type, X_test, X_train, y_test, y_train)
                elif (model_type == "decisiontree"):
                    accuracy_now = self.modelDecisionTree.train_model(models_svc, dataset_type, X_test, X_train, y_test, y_train)
                elif (model_type == "randomforest"):
                    accuracy_now = self.modelRandomForest.train_model(models_svc, dataset_type, X_test, X_train, y_test, y_train)
                elif (model_type == "logisticregression"):
                    accuracy_now = self.modelLogisticRegression.train_model(models_svc, dataset_type, X_test, X_train, y_test, y_train)

                accuracy_sum = accuracy_sum + accuracy_now

            mean_accuracy_overall = accuracy_sum / len(models_svc)
            print(f'mean_accuracy_overall: {mean_accuracy_overall} \n')








# entry point
main = Main();
main.start();
# ModelNaiveBayes_bk().train(["feature-envy"], "generate", "retrain")

## test
# ModelSVC().train(["feature-envy"], "generate")
# ModelSVC().train(["feature-envy"], "compare")


