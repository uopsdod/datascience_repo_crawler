from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

from myutils.printservice import PrintService

is_develop_mode = False
padding_count = 20

class ModelNaiveBayes:
    def __init__(self):
        # self.cacheService = CacheService()
        self.printService = PrintService()
        self.modelname = 'naivebayes'
        pass

    def train_model(self, models_svc, dataset_type, X_test, X_train, y_test, y_train):
        # self.printService.print_result(dataset_type, "Accuracy(%)", "F1-score(%)")

        # step06-2: train it with multiple combinations
        pipeline = Pipeline([
            ('naivebayes', GaussianNB()),
        ])
        param_grid = self.get_param_grid()
        best_model = self.train_helper(X_train, y_train, param_grid, pipeline)
        models_svc[dataset_type] = best_model

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

    def get_param_grid(self):

        param_grid = {
            'naivebayes__priors': [[0.1, 0.9], [0.3, 0.7], [0.5, 0.5], [0.8, 0.2], [0.9, 0.1]], # report: to cope with skewed data
            'naivebayes__var_smoothing': [1e-10, 1e-9, 1e-8, 1e-2] # report: to smooth the curve by adding the largest variance among features to all variance
        }
        return param_grid

    def train_helper(self, X_train, y_train, param_grid, estimators):
        scoring_strategy_list = ["accuracy", "f1"]
        for scoring_strategy in scoring_strategy_list:
            grid_search = GridSearchCV(estimators, param_grid, cv=3, scoring=scoring_strategy,
                                       return_train_score=True, n_jobs=-1)

            grid_search_config_info = grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_

        return best_model
