from training_model import ModelDecisionTree
from training_model.naivebayes import ModelNaiveBayes
from training_model.randomforest import ModelRandomForest


class Main:
    def __init__(self):
        pass

    def start(self):
        print("choose operation: ")
        print("1. generate training results ")
        print("2. compare training results")
        operation_id = input("please type # of operation: ")
        if operation_id != "1" and operation_id != "2":
            print("operation_id is unknown. abort operation")
            return

        print("choose model: ")
        print("1. decision tree ")
        print("2. random forest ")
        print("3. naïve bayes ")
        print("4. SVC (XXXXXX kernel) ")
        model_id = input("please type # of model: ")
        # print(f'You entered {model_id}')
        # print("please type # of model: ")
        model_picked = self.get_model_picked(model_id);
        if model_picked == "none":
            print("model id is unknown. abort operation")
            return

        if operation_id == "1":
            self.generate_training(model_picked)
        elif operation_id == "2":
            self.compare_training(model_picked)

    def compare_training(self, model_picked):
        print("choose smell: ")
        print("1. Feature Envy ")
        print("2. Data Class ")
        print("3. God Class ")
        print("4. Long Method ")
        print("(ex. 1) - for feature envy")
        smell_ids = input("please type # of smell: ")
        smell_picked = self.get_smell_picked(smell_ids)
        if smell_picked == "none":
            print("smell id is unknown. abort operation")
            return
        elif len(smell_picked) > 1:
            print("only one smell id is allowed. abort operation")
            return

        # God Class, Data Class, Feature Envy, and Long Method
        model_picked.train(smell_picked, "compare")

    def generate_training(self, model_picked):

        print("choose smells: ")
        print("1. Feature Envy ")
        print("2. Data Class ")
        print("3. God Class ")
        print("4. Long Method ")
        print("(ex. 1) - for feature envy")
        print("(ex. 1 3) - for feature envy, god class")
        print("(ex. 1 3 4) - for feature envy, god class, long method")
        smell_ids = input("please type # of one or more smells: ")
        smell_picked = self.get_smell_picked(smell_ids)
        if smell_picked == "none":
            print("smell id is unknown. abort operation")
            return

        # God Class, Data Class, Feature Envy, and Long Method
        model_picked.train(smell_picked, "generate")

    def get_model_picked(self, model_id):
        if model_id == "1":
            return ModelDecisionTree()
        if model_id == "2":
            return ModelRandomForest()
        if model_id == "3":
            return ModelNaiveBayes()
        if model_id == "4":
            return
        return "none"

    def get_smell_picked(self, smell_ids):
        smell_str_array = []
        smell_ids_array = smell_ids.strip().split(" ")

        for smell_id in smell_ids_array:
            if smell_id == "1":
                smell_str_array.append("feature-envy")
            elif smell_id == "2":
                smell_str_array.append("data-class")
            elif smell_id == "3":
                smell_str_array.append("god-class")
            elif smell_id == "4":
                smell_str_array.append("long-method")
            else:
                return "none"

        if len(smell_str_array) == 0:
            return "none"

        return smell_str_array


# entry point
main = Main();
main.start();

## test
# ModelNaiveBayes().train(["feature-envy"], "generate")
# ModelNaiveBayes().train(["feature-envy"], "compare")


