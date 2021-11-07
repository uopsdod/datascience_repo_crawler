import pandas as pd

from myutils.datasetservice import DatasetService
from myutils.printservice import PrintService
from training_model import ModelDecisionTree
from training_model.naivebayes import ModelNaiveBayes
from training_model.randomforest import ModelRandomForest
from training_model.svc import ModelSVC



class Main:
    def __init__(self):
        self.printService = PrintService()
        self.datasetService = DatasetService()
        pass

    def start(self):

        dataset_types = ["bug", "feature", "rating", "userexperience"] # add two more
        for dataset_type in dataset_types:
            # dataset_type = "bug"
            features_gleaned = ["title", "comment", "label"]
            df = self.datasetService.load_file(dataset_type, features_gleaned)
            is_dataset_balance = self.datasetService.is_dataset_balance(dataset_type, df, "label")
            print("hey001")

        # while(True):
        #     print("choose intention: ")
        #     print("1. retrain models ")
        #     print("2. see specific results")
        #     print("3. see all results")
        #     print("4. exit")
        #     operation_id = input("please type # of intention: ").strip()
        #     if operation_id != "1" and operation_id != "2" and operation_id != "3" and operation_id != "4" and operation_id != "gap":
        #         print("intention_id is unknown. abort operation")
        #         continue
        #     elif operation_id == "3":
        #         self.printService.print_all_test_result()
        #         continue
        #     elif operation_id == "4":
        #         print("program ended")
        #         break
        #     elif operation_id == "gap":
        #         self.printService.print_all_test_gap_result()
        #         continue
        #     operation_picked = self.get_intention_picked(operation_id)
        #
        #     print("choose operation: ")
        #     print("1. generate training results ")
        #     print("2. compare training results")
        #     operation_id = input("please type # of operation: ").strip()
        #     if operation_id != "1" and operation_id != "2":
        #         print("operation_id is unknown. abort operation")
        #         continue
        #
        #     print("choose model: ")
        #     print("1. decision tree ")
        #     print("2. random forest ")
        #     print("3. naïve bayes ")
        #     print("4. LinearSVC ")
        #     model_id = input("please type # of model: ").strip()
        #     # print(f'You entered {model_id}')
        #     # print("please type # of model: ")
        #     model_picked = self.get_model_picked(model_id);
        #     if model_picked == "none":
        #         print("model id is unknown. abort operation")
        #         continue
        #
        #     if operation_id == "1":
        #         self.generate_training(model_picked, operation_picked)
        #     elif operation_id == "2":
        #         self.compare_training(model_picked, operation_picked)

    # def compare_training(self, model_picked, operation_picked):
    #     print("choose smell: ")
    #     print("1. Feature Envy ")
    #     print("2. Data Class ")
    #     print("3. God Class ")
    #     print("4. Long Method ")
    #     print("(ex. 1) - for feature envy")
    #     smell_ids = input("please type # of smell: ").strip()
    #     smell_picked = self.get_smell_picked(smell_ids)
    #     if smell_picked == "none":
    #         print("smell id is unknown. abort operation")
    #         return
    #     elif len(smell_picked) > 1:
    #         print("only one smell id is allowed. abort operation")
    #         return
    #
    #     # God Class, Data Class, Feature Envy, and Long Method
    #     model_picked.train(smell_picked, "compare", operation_picked)
    #
    # def generate_training(self, model_picked, operation_picked):
    #
    #     print("choose smells: ")
    #     print("1. Feature Envy ")
    #     print("2. Data Class ")
    #     print("3. God Class ")
    #     print("4. Long Method ")
    #     print("(ex. 1) - for feature envy")
    #     print("(ex. 1 3) - for feature envy, god class")
    #     print("(ex. 1 3 4) - for feature envy, god class, long method")
    #     smell_ids = input("please type # of one or more smells: ").strip()
    #     smell_picked = self.get_smell_picked(smell_ids)
    #     if smell_picked == "none":
    #         print("smell id is unknown. abort operation")
    #         return
    #
    #     # God Class, Data Class, Feature Envy, and Long Method
    #     model_picked.train(smell_picked, "generate", operation_picked)
    #
    # def get_intention_picked(self, intention_id):
    #     if intention_id == "1":
    #         return "retrain"
    #     if intention_id == "2":
    #         return "lookup"
    #     return "none"
    #
    # def get_model_picked(self, model_id):
    #     if model_id == "1":
    #         return ModelDecisionTree()
    #     if model_id == "2":
    #         return ModelRandomForest()
    #     if model_id == "3":
    #         return ModelNaiveBayes()
    #     if model_id == "4":
    #         return ModelSVC()
    #     return "none"
    #
    # def get_smell_picked(self, smell_ids):
    #     smell_str_array = []
    #     smell_ids_array = smell_ids.split(" ")
    #
    #     for smell_id in smell_ids_array:
    #         if smell_id == "1":
    #             smell_str_array.append("feature-envy")
    #         elif smell_id == "2":
    #             smell_str_array.append("data-class")
    #         elif smell_id == "3":
    #             smell_str_array.append("god-class")
    #         elif smell_id == "4":
    #             smell_str_array.append("long-method")
    #         else:
    #             return "none"
    #
    #     if len(smell_str_array) == 0:
    #         return "none"
    #
    #     return smell_str_array

# entry point
main = Main();
main.start();

## test
# ModelSVC().train(["feature-envy"], "generate")
# ModelSVC().train(["feature-envy"], "compare")


