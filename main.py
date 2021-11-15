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
from training_model.svc_wapper import ModelSVCWrapper


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
        self.modelSVCWrapper = ModelSVCWrapper()

    def start(self):

        # dataset_types = ["feature", "bug", "rating", "userexperience"]
        # dataset_types = ["rating"] # debug
        # dataset_types = ["feature"] # debug
        # dataset_types = ["bug"] # debug
        dataset_types = ["rating", "feature"] # debug

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

            # step: fill in null/nan fields
            self.datasetService.fill_null_val(df, "title", "")
            self.datasetService.fill_null_val(df, "comment", "")
            self.datasetService.fill_nan_mean(df, "rating")
            self.datasetService.fill_nan_mean(df, "sentiScore")

            # data clean - convert to numeric value - fee type
            self.datasetService.convert_fee_type_to_numeric(df)

            # feature scaling
            self.datasetService.standardize_feature(df, "rating")
            self.datasetService.standardize_feature(df, "sentiScore")
            self.datasetService.standardize_feature(df, "fee")
            # self.datasetService.standardize_feature(df, "length_word_cleaned") # not good at all - don't use it

            # step: NLP
            self.nlpservice.lemmatize(df, "title")
            self.nlpservice.lemmatize(df, "comment")
            self.nlpservice.remove_punctuation_signs(df, "comment")
            self.nlpservice.remove_stopwords(df, "comment")

            df_datasets[dataset_type] = df
            example_count = example_count + len(df)

            if df_all is None:
                df_all = df
            else:
                df_all = df_all.append(df)


        # reorganize index here
        df_all = self.datasetService.reorganize_index(df_all)

        # let's divide data_train data_test right here

        # step: NLP - convert a sentence to BOW (TF-IDF)
        df_comment_train = self.nlpservice.convert_to_tfidf_text_representation(df_all, "comment")
        df_title_train = self.nlpservice.convert_to_tfidf_text_representation(df_all, "title")
        df_final = df_comment_train.merge(df_title_train, left_index=True, right_index=True) # join by index

        df_final["comment"] = df_all["comment"]
        df_final["rating"] = df_all["rating"]
        df_final["sentiScore"] = df_all["sentiScore"]
        df_final["fee"] = df_all["fee"]

        df_final["length_word_cleaned"] = self.datasetService.get_word_count_of_cleaned_comment(df_final)
        self.datasetService.fill_nan_mean(df_final, "length_word_cleaned")

        # step: add result class
        df_final["label"] = df_all["label"]

        # keep it
        # df_final_dataset[dataset_type] = df_final.copy()

        for model_type in model_types:
            accuracy_sum = 0
            df_final_dataset = {}
            models_svc = {}

            self.printService.print_result(f'[{model_type}]', "Accuracy(%)", "F1-score(%)")
            for dataset_type in dataset_types:
                # step: get the features you need from raw dataset
                # features_gleaned = ["title", "comment", "rating", "label"]
                # df = self.datasetService.load_file(dataset_type, features_gleaned)

                # convert label name to numeric
                label_codes = self.datasetService.get_label_codes(dataset_type)
                df_here = self.datasetService.convert_to_df(df_final, df_datasets[dataset_type], label_codes)

                # remove comment (which is string format and for temporary use)
                df_here.drop("comment", axis=1, inplace=True)

                # step: balance datasets
                df_here = self.datasetService.balance_dataset(dataset_type, df_here, "label")

                # step: split data for training and testing
                df_here = self.datasetService.replace_label_with_zerorone(df_here, dataset_type)
                (X_train, X_test, y_train, y_test) = self.datasetService.split_dataset(df_here)

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

            ########
            # df_all_final = None
            # for dataset_type in dataset_types:
            #     df_now = df_final_dataset[dataset_type]
            #     labels_now = df_now.pop('label') # remove column b and store it in df1
            #     df_now['label'] = labels_now
            #
            #     if df_all_final is None:
            #         df_all_final = df_now
            #     else:
            #         df_all_final = df_all_final.append(df_now)
            #     pass
            ######

            # self.modelSVCWrapper.predict()
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


