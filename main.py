import pandas as pd
import numpy as np

from myutils.datasetservice import DatasetService
from myutils.nlpservice import NLPService
from myutils.printservice import PrintService
from training_model import ModelDecisionTree
from training_model.naivebayes import ModelNaiveBayes
from training_model.naivebayes_bk import ModelNaiveBayes_bk
from training_model.randomforest import ModelRandomForest
from training_model.svc import ModelSVC



class Main:
    def __init__(self):
        self.printService = PrintService()
        self.datasetService = DatasetService()
        self.nlpservice = NLPService()
        self.modelNaiveBayes = ModelNaiveBayes()
        pass

    def start(self):

        # dataset_types = ["bug", "feature", "rating", "userexperience"]
        dataset_types = ["bug"]
        for dataset_type in dataset_types:
            # step01: get the dataset you want from raw dataset
            features_gleaned = ["title", "comment", "rating", "label"]
            df = self.datasetService.load_file(dataset_type, features_gleaned)
            self.datasetService.fill_null_val(df, "title", "")
            self.datasetService.fill_null_val(df, "comment", "")
            self.datasetService.balance_dataset(dataset_type, df, "label")

            # step02: use NLP to parse the comments
            self.nlpservice.lemmatize(df, "title")
            self.nlpservice.lemmatize(df, "comment")

            df_comment_train = self.nlpservice.convert_to_tfidf_text_representation(df, "comment")
            df_title_train = self.nlpservice.convert_to_tfidf_text_representation(df, "title")
            df_final = df_comment_train.merge(df_title_train, left_index=True, right_index=True) # join by index
            df_final["rating"] = df["rating"]
            df_final["label"] = df["label"]

            # step03: split data for training and testing
            (X_train, X_test, y_train, y_test) = self.modelNaiveBayes.split_dataset(df_final)

            self.modelNaiveBayes.train(X_train, X_test, y_train, y_test)
            print()

# entry point
main = Main();
main.start();
# ModelNaiveBayes_bk().train(["feature-envy"], "generate", "retrain")

## test
# ModelSVC().train(["feature-envy"], "generate")
# ModelSVC().train(["feature-envy"], "compare")


