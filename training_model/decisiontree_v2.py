import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier

from myutils.printservice import PrintService

is_develop_mode = False
padding_count = 20

class ModelDecisionTree:
    def __init__(self):
        # self.cacheService = CacheService()
        self.printService = PrintService()
        self.modelname = 'decisiontree'
        pass

    def train_model(self, models_svc, dataset_type, X_test, X_train, y_test, y_train):
        # self.printService.print_result(dataset_type, "Accuracy(%)", "F1-score(%)")

        # step06-2: train it with multiple combinations
        pipeline = Pipeline([
            ('decisiontree', DecisionTreeClassifier()),
        ])
        param_grid = self.get_param_grid()
        best_model = self.train_helper(X_train, y_train, param_grid, pipeline)
        models_svc[dataset_type] = best_model

        # fit the model
        best_model.fit(X_train, y_train)

        # Accuracy
        y_predict_here = best_model.predict(X_train)
        accuracy_score_result = str(int(accuracy_score(y_train, best_model.predict(X_train)) * 100))
        accuracy_score_result_test = str(int(accuracy_score(y_test, best_model.predict(X_test)) * 100) )

        # F1-score
        f1_score_result = str(int(f1_score(y_train, best_model.predict(X_train)) * 100))
        f1_score_result_test = str( int(f1_score(y_test, best_model.predict(X_test)) * 100) )

        self.printService.print_result_here(accuracy_score_result, accuracy_score_result_test, f1_score_result,
                                            f1_score_result_test, "generate", dataset_type)
        return int(accuracy_score_result_test)

    def train_helper(self, X_train, y_train, param_grid, estimators):
        scoring_strategy_list = ["accuracy", "f1"]
        for scoring_strategy in scoring_strategy_list:
            # DecisionTreeClassifier()
            # RandomSearchVC (lookup)
            grid_search = GridSearchCV(estimators, param_grid, cv=10, scoring=scoring_strategy,
                                       return_train_score=True, n_jobs=-1)

            grid_search_config_info = grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_

        return best_model

    def get_param_grid(self):
        depths = np.arange(1, 6)
        min_samples = np.arange(2, 6)  # must be > than 1
        min_samples_leaf = np.arange(1, 5)
        # num_leafs = [1, 5, 10, 20, 50, 100]
        param_grid = {'decisiontree__criterion': ['gini', 'entropy'],
                      'decisiontree__splitter': ["best", "random"],
                      'decisiontree__max_depth': depths,
                      'decisiontree__min_samples_split': min_samples,
                      'decisiontree__min_samples_leaf': min_samples_leaf
                      }
        return param_grid