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

        dataset_types = ["bug", "feature", "rating", "userexperience"]
        # dataset_types = ["rating"] # debug
        # dataset_types = ["feature"] # debug
        # dataset_types = ["bug"] # debug
        # dataset_types = ["rating", "feature"] # debug
        # dataset_types = ["bug", "rating"]

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
            # self.nlpservice.lemmatize(df, "title")
            # self.nlpservice.lemmatize(df, "comment")
            # self.nlpservice.remove_punctuation_signs(df, "comment")
            # self.nlpservice.remove_stopwords(df, "comment")

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
                df_here = self.datasetService.replace_label_with_zerorone(df_here, label_codes)
                (X_train, X_test, y_train, y_test) = self.datasetService.split_dataset(df_here)

                # X_train_minmax = min_max_scaler.transform(X_train)
                # X_train["length_word_cleaned"] = X_train_minmax["length_word_cleaned"]
                # X_test_minmax = min_max_scaler.transform(X_test)

                if (model_type == "svc"):
                    self.modelSVC.train_model(models_svc, dataset_type, X_test, X_train, y_test, y_train)
                elif (model_type == "naivebayes"):
                    self.modelNaiveBayes.train_model(models_svc, dataset_type, X_test, X_train, y_test, y_train)
                elif (model_type == "decisiontree"):
                    self.modelDecisionTree.train_model(models_svc, dataset_type, X_test, X_train, y_test, y_train)
                elif (model_type == "randomforest"):
                    self.modelRandomForest.train_model(models_svc, dataset_type, X_test, X_train, y_test, y_train)
                elif (model_type == "logisticregression"):
                    self.modelLogisticRegression.train_model(models_svc, dataset_type, X_test, X_train, y_test, y_train)

            ########

            # label df_final into 0(bug), 1(feature), 2(rating), 3(userexperience), 4(others)
            df_final.drop("comment", axis=1, inplace=True)
            label_all_codes = self.datasetService.get_label_all_codes()
            df_final_multi = self.datasetService.replace_label_with_zerorone(df_final, label_all_codes)
            df_final_multi = df_final_multi[(df_final_multi.label != 4)]

            X_final = df_final_multi.iloc[:, :-1].copy()
            y_final = df_final_multi.iloc[: , -1].copy()

            # (X_train, X_test, y_train, y_test) = self.datasetService.split_dataset(df_final_multi)

            Y_train_predict = self.modelSVCWrapper.predict(models_svc, X_final)
            Y_train_predict = np.array(Y_train_predict)

            # Accuracy
            accuracy_here = accuracy_score(y_final, Y_train_predict)
            accuracy_score_result = str(int(accuracy_here * 100))

            # F1-score
            f1_score_results = f1_score(y_final, Y_train_predict, average=None)
            for index, val in enumerate(f1_score_results):
                f1_score_results[index] = int(f1_score_results[index] * 100)

            self.printService.print_result_here(accuracy_score_result, accuracy_score_result,
                                                str(f1_score_results[0]), str(f1_score_results[0]), "generate", "bug")
            self.printService.print_result_here(accuracy_score_result, accuracy_score_result,
                                                str(f1_score_results[1]), str(f1_score_results[1]), "generate", "feature")
            self.printService.print_result_here(accuracy_score_result, accuracy_score_result,
                                                str(f1_score_results[2]), str(f1_score_results[2]), "generate", "rating")
            self.printService.print_result_here(accuracy_score_result, accuracy_score_result,
                                                str(f1_score_results[3]), str(f1_score_results[3]), "generate", "userexperience")



# entry point
main = Main();
main.start();
# ModelNaiveBayes_bk().train(["feature-envy"], "generate", "retrain")

## test
# ModelSVC().train(["feature-envy"], "generate")
# ModelSVC().train(["feature-envy"], "compare")


