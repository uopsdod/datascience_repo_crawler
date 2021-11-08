import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from myutils.datasetservice import DatasetService
from myutils.nlpservice import NLPService
from myutils.printservice import PrintService
from training_model.naivebayes import ModelSVC




class Main:
    def __init__(self):
        self.printService = PrintService()
        self.datasetService = DatasetService()
        self.nlpservice = NLPService()
        self.modelSVC = ModelSVC()
        pass

    def start(self):

        dataset_types = ["bug", "feature", "rating", "userexperience"]
        # dataset_types = ["bug", "feature"]
        # dataset_types = ["bug"]

        models_svc = {}
        accuracy_sum = 0

        for dataset_type in dataset_types:
            # step: get the features you need from raw dataset
            features_gleaned = ["title", "comment", "rating", "label"]
            df = self.datasetService.load_file(dataset_type, features_gleaned)

            # step: fill in null/nan fields
            self.datasetService.fill_null_val(df, "title", "")
            self.datasetService.fill_null_val(df, "comment", "")
            self.datasetService.fill_nan_mean(df, "rating")

            # step: balance datasets
            self.datasetService.balance_dataset(dataset_type, df, "label")

            # step: NLP - lemmatization
            self.nlpservice.lemmatize(df, "title")
            self.nlpservice.lemmatize(df, "comment")

            # step: NLP - convert a sentence to BOW (TF-IDF)
            df_comment_train = self.nlpservice.convert_to_tfidf_text_representation(df, "comment")
            df_title_train = self.nlpservice.convert_to_tfidf_text_representation(df, "title")
            df_final = df_comment_train.merge(df_title_train, left_index=True, right_index=True) # join by index

            # step: add metadata features
            df_final["rating"] = df["rating"]

            # step: add result class
            df_final["label"] = df["label"]

            # step: split data for training and testing
            (X_train, X_test, y_train, y_test) = self.modelSVC.split_dataset(df_final, dataset_type)

            # step: train it
            best_svc = self.modelSVC.train(X_train, X_test, y_train, y_test)
            models_svc[dataset_type] = best_svc

            # fit the model
            best_svc.fit(X_train, y_train)
            # print(f' predictions: \n {best_svc.predict(X_test)}')
            # print(f' predictions_proba: \n {best_svc.predict_proba(X_test)}')

            # Accuracy
            accuracy_score_result = int(accuracy_score(y_train, best_svc.predict(X_train)) * 100)
            accuracy_score_result_test = int(accuracy_score(y_test, best_svc.predict(X_test)) * 100)
            accuracy_sum = accuracy_sum + accuracy_score_result_test

            # F1-score
            f1_score_result = int(f1_score(y_train, best_svc.predict(X_train)) * 100)
            f1_score_result_test = int(f1_score(y_test, best_svc.predict(X_test)) * 100)

            self.printService.print_result(dataset_type, "Accuracy(%)", "F1-score(%)")
            self.printService.print_result_here(accuracy_score_result, accuracy_score_result_test, f1_score_result,
                                                f1_score_result_test, "generate", dataset_type)

            # print()

        mean_accuracy_overall = accuracy_sum / len(models_svc)
        print(f'mean_accuracy_overall: {mean_accuracy_overall} \n')

# entry point
main = Main();
main.start();
# ModelNaiveBayes_bk().train(["feature-envy"], "generate", "retrain")

## test
# ModelSVC().train(["feature-envy"], "generate")
# ModelSVC().train(["feature-envy"], "compare")


