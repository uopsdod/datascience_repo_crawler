from scipy.io import arff
import pandas as pd
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

from myutils.cacheservice import CacheService
from myutils.printservice import PrintService

is_develop_mode = False


class ModelDecisionTree:
    def __init__(self):
        self.modelname = 'decisiontree'
        self.cacheService = CacheService()
        self.printService = PrintService()
        pass

    def train(self, smell_types, operation, operation_picked):
        # print(f"train it - {smell_types}")

        # step01: load dataset from file

        if operation == "generate":
            self.printService.print_result("Smell", "Accuracy(%)", "F1-score(%)")
        elif operation == "compare":
            self.printService.print_result(smell_types[0], "Accuracy(%)", "F1-score(%)")


        for smell_type in smell_types:

            if operation_picked == 'retrain':
                self.cacheService.clear_cache(self.modelname, smell_type)

            (accuracy_score_result_cached,
             accuracy_score_result_test_cached,
             f1_score_result_cached,
             f1_score_result_test_cached) = self.cacheService.use_cache(self.modelname, smell_type)

            if accuracy_score_result_cached is not None:
                self.printService.print_result_here(accuracy_score_result_cached, accuracy_score_result_test_cached, f1_score_result_cached,
                                       f1_score_result_test_cached, operation, smell_type)
                continue

            dataset_filename = self._get_dataset_filename(smell_type)
            if dataset_filename == "none":
                return
            data = arff.loadarff(dataset_filename)

            # step02: get a dataframe from dataset
            df = pd.DataFrame(data[0])

            # step03: get independent variables(features)(x) & dependent variables(y)
            Y_data = df.iloc[:, -1].values
            encoder = preprocessing.LabelEncoder()
            y = encoder.fit_transform(Y_data)

            # step04: give missing data a mockup value
            X_copy = df.iloc[:, :-1].copy()
            imputer = SimpleImputer(strategy="median")
            imputer.fit(X_copy)
            new_X = imputer.transform(X_copy) # ? what is the difference btwn fit vs. transform
            # new_X_df = pd.DataFrame(new_X, columns=X_copy.columns, index=X_copy.index)

            # step03-2: feature selection (remove low variance) - to combat skewed data
            # p = .7
            sel = VarianceThreshold(threshold=0.8)
            new_X = sel.fit_transform(new_X)
            # step03-2: feature selection (pick K best)
            # new_X = SelectKBest(chi2, k=60).fit_transform(new_X, y)

            # step05: split into training dataframe & testing dataframe
            test_size_now = 0.15
            if smell_type == "long-method":
                test_size_now = .35
            X_train, X_test, y_train, y_test = train_test_split(new_X, y, test_size=test_size_now, random_state=42) # report: 0.35 to avoid test data too skewed?

            # step06-2: train it with multiple combinations
            param_grid = self.get_param_grid()
            best_model, accuracy_score_result, f1_score_result = self.train_helper(X_train, param_grid, y_train)

            accuracy_score_result_test = str(int(accuracy_score(y_test, best_model.predict(X_test)) * 100) )
            f1_score_result_test = str( int(f1_score(y_test, best_model.predict(X_test)) * 100) )

            # cache
            self.cacheService.update_cache(accuracy_score_result,
                                           accuracy_score_result_test,
                                           f1_score_result,
                                           f1_score_result_test,
                                           self.modelname,
                                           smell_type)

            # what to print
            self.printService.print_result_here(accuracy_score_result, accuracy_score_result_test, f1_score_result,
                                   f1_score_result_test, operation, smell_type)


    def get_param_grid(self):
        depths = np.arange(1, 6)
        min_samples = np.arange(2, 6)  # must be > than 1
        min_samples_leaf = np.arange(1, 5)
        # num_leafs = [1, 5, 10, 20, 50, 100]
        param_grid = {'criterion': ['gini', 'entropy'],
                      'splitter': ["best", "random"],
                      'max_depth': depths,
                      'min_samples_split': min_samples,
                      'min_samples_leaf': min_samples_leaf
                      }
        return param_grid

    def train_helper(self, X_train, param_grid, y_train):
        scoring_strategy_list = ["accuracy", "f1"]
        for scoring_strategy in scoring_strategy_list:
            new_tree_clf = DecisionTreeClassifier()
            # RandomSearchVC (lookup)
            grid_search = GridSearchCV(new_tree_clf, param_grid, cv=10, scoring=scoring_strategy,
                                       return_train_score=True, n_jobs=-1)

            grid_search_config_info = grid_search.fit(X_train, y_train)
            if (is_develop_mode):
                print(grid_search_config_info)
            if (is_develop_mode):
                print(grid_search.best_estimator_)

            best_model = grid_search.best_estimator_

            if scoring_strategy == "accuracy":
                accuracy_score_result = str(int(accuracy_score(y_train, best_model.predict(X_train)) * 100))
                # print(f"decision tree - {smell_type} - {scoring_strategy} - score: " + accuracy_score_result)
            elif scoring_strategy == "f1":
                f1_score_result = str(int(f1_score(y_train, best_model.predict(X_train)) * 100))
                # print(f"decision tree - {smell_type} - {scoring_strategy} - score: " + f1_score_result)
        return best_model, accuracy_score_result, f1_score_result


    def _get_dataset_filename(self, smell_type):
        dateset_folder = "training_dataset"
        if smell_type == "feature-envy":
            dateset_file = "feature-envy.arff"
        elif smell_type == "data-class":
            dateset_file = "data-class.arff"
        elif smell_type == "god-class":
            dateset_file = "god-class.arff"
        elif smell_type == "long-method":
            dateset_file = "long-method.arff"
        else:
            return "none"

        return dateset_folder + "/" + dateset_file

