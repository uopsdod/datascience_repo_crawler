import nltk
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
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
        df_all = None
        df_datasets = {}
        example_count = 0
        for dataset_type in dataset_types:
            features_gleaned = ["title", "comment", "rating", "label", "length_words"]
            df = self.datasetService.load_file(dataset_type, features_gleaned)
            df_datasets[dataset_type] = df
            example_count = example_count + len(df)

            if df_all is None:
                df_all = df
            else:
                df_all = df_all.append(df)

        # dataset_types = ["rating"] # debug
        # dataset_types = ["feature"] # debug
        # dataset_types = ["bug"] # debug
        dataset_types = ["rating", "feature"] # debug

        models_svc = {}
        accuracy_sum = 0

        for dataset_type in dataset_types:
            # step: get the features you need from raw dataset
            # features_gleaned = ["title", "comment", "rating", "label"]
            # df = self.datasetService.load_file(dataset_type, features_gleaned)

            label_codes = self.modelSVC.get_label_codes(dataset_type)
            df = self.convert_to_df(df_all, df_datasets[dataset_type], label_codes)

            self.nlpservice.remove_punctuation_signs(df, "comment")
            self.nlpservice.remove_stopwords(df, "comment")

            df["length_word_cleaned"] = self.datasetService.get_word_count_of_cleaned_comment(df)

            # step: fill in null/nan fields
            self.datasetService.fill_null_val(df, "title", "")
            self.datasetService.fill_null_val(df, "comment", "")
            self.datasetService.fill_nan_mean(df, "rating")
            self.datasetService.fill_nan_mean(df, "length_word_cleaned")
            self.datasetService.fill_nan_mean(df, "sentiScore")

            # step: balance datasets
            df = self.datasetService.balance_dataset(dataset_type, df, "label")

            # step: NLP - lemmatization
            self.nlpservice.lemmatize(df, "title")
            self.nlpservice.lemmatize(df, "comment")

            # step: NLP - convert a sentence to BOW (TF-IDF)
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

    def convert_to_df(self, df_all, df_dataset, label_codes):

        label_positive = list(label_codes)[0]
        label_negative = list(label_codes)[1]

        comments_series = df_dataset['label']
        comments_series.index = df_dataset['comment']
        comments_dict = comments_series.to_dict()

        new_rows_array = []
        for index, row in df_all.iterrows():
            comment_now = row['comment']
            if ( comment_now in comments_dict and comments_dict[comment_now] == label_positive):
                row['label'] = label_positive
                new_rows_array.append(row)
            else:
                row['label'] = label_negative
                new_rows_array.append(row)

        new_rows_df = pd.DataFrame(new_rows_array)
        return new_rows_df

# entry point
main = Main();
main.start();
# ModelNaiveBayes_bk().train(["feature-envy"], "generate", "retrain")

## test
# ModelSVC().train(["feature-envy"], "generate")
# ModelSVC().train(["feature-envy"], "compare")


