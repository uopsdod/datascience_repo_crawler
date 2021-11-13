from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

from myutils.printservice import PrintService

is_develop_mode = False
padding_count = 20

class ModelSVC:
    def __init__(self):
        self.printService = PrintService()
        self.modelname = 'svc'
        pass

    # def train(self, X_train, X_test, y_train, y_test):
    #     from sklearn import svm
    #     from pprint import pprint
    #     from sklearn import svm
    #     from pprint import pprint
    #     from sklearn.model_selection import RandomizedSearchCV
    #
    #     # let's train a SVM
    #     svc_0 =svm.SVC(random_state=42)
    #     # print('Parameters currently in use:\n')
    #     # pprint(svc_0.get_params())
    #
    #     # hyperparameters
    #     random_grid = self.get_random_grid()
    #     # pprint(random_grid)
    #     # First create the base model to tune
    #     svc = svm.SVC(random_state=42)
    #
    #     # Definition of the random search
    #     random_search = RandomizedSearchCV(estimator=svc,
    #                                        param_distributions=random_grid,
    #                                        n_iter=3,
    #                                        scoring='accuracy',
    #                                        cv=3,
    #                                        verbose=1,
    #                                        random_state=42)
    #
    #     # Fit the random search model
    #     random_search.fit(X_train, y_train)
    #     if (is_develop_mode):
    #         print("The best hyperparameters from Random Search are:")
    #         print(random_search.best_params_)
    #     # print("")
    #     # print("The mean accuracy of a model with these hyperparameters is:")
    #     # print(random_search.best_score_)
    #
    #     # random search was better
    #     best_svc = random_search.best_estimator_
    #     return best_svc


    # def train_model_svc(self, models_svc, dataset_type, X_test, X_train, y_test, y_train):
    #
    #     accuracy_sum = 0
    #
    #     # 1: train SVC
    #     best_svc = self.train(X_train, X_test, y_train, y_test)
    #     models_svc[dataset_type] = best_svc
    #     # fit the model
    #     best_svc.fit(X_train, y_train)
    #     # print(f' predictions: \n {best_svc.predict(X_test)}')
    #     # print(f' predictions_proba: \n {best_svc.predict_proba(X_test)}')
    #     # Accuracy
    #     accuracy_score_result = int(accuracy_score(y_train, best_svc.predict(X_train)) * 100)
    #     accuracy_score_result_test = int(accuracy_score(y_test, best_svc.predict(X_test)) * 100)
    #     accuracy_sum = accuracy_sum + accuracy_score_result_test
    #     # F1-score
    #     f1_score_result = int(f1_score(y_train, best_svc.predict(X_train)) * 100)
    #     f1_score_result_test = int(f1_score(y_test, best_svc.predict(X_test)) * 100)
    #     self.printService.print_result(dataset_type, "Accuracy(%)", "F1-score(%)")
    #     self.printService.print_result_here(accuracy_score_result, accuracy_score_result_test, f1_score_result,
    #                                         f1_score_result_test, "generate", dataset_type)
    #     return accuracy_sum

    def train_model(self, models_svc, dataset_type, X_test, X_train, y_test, y_train):
        # self.printService.print_result(dataset_type, "Accuracy(%)", "F1-score(%)")

        # step06-2: train it with multiple combinations
        pipeline = Pipeline([
            ('svc', svm.SVC(random_state=42)),
        ])
        param_grid = self.get_param_grid()
        best_model = self.train_helper(X_train, y_train, param_grid, pipeline)
        models_svc[dataset_type] = best_model

        mysvc = pipeline.named_steps["svc"]

        # fit the model
        best_model.fit(X_train, y_train)

        # Accuracy
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
        # C = [.0001, .001, .01, 1]
        # gamma = [.0001, .001, .01, .1, 1, 10, 100]
        # degree = [1, 2, 3, 4, 5]
        # kernel = ['linear', 'rbf', 'poly']
        # probability = [True]

        # the best hyperparameter (to speed up training tryouts afterwards)
        C = [1]
        gamma = [100]
        degree = [1]
        kernel = ['rbf']
        probability = [True]
        random_grid = {'svc__C': C,
                       'svc__kernel': kernel,
                       'svc__gamma': gamma,
                       'svc__degree': degree,
                       'svc__probability': probability
                       }
        return random_grid
