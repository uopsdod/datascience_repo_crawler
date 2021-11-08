from scipy.io import arff
import pandas as pd
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

from myutils.cacheservice import CacheService
from myutils.printservice import PrintService

is_develop_mode = False
padding_count = 20

class ModelNaiveBayes:
    def __init__(self):
        self.printService = PrintService()
        self.modelname = 'naivebayes'
        pass

    def split_dataset(self, df):
        # step03: get independent variables(features)(x) & dependent variables(y)
        # y = df.iloc[:, -1].values
        # encoder = preprocessing.LabelEncoder()
        # y = encoder.fit_transform(y)

        # step04: give missing data a mockup value
        # imputer = SimpleImputer(strategy="median")
        # imputer.fit(X)
        # X = imputer.transform(X)

        # step03-2: feature selection (remove low variance)
        # p = .8
        # sel = VarianceThreshold(threshold=(p * (1 - p)))
        # new_X = sel.fit_transform(X)
        # step03-2: feature selection (pick K best)
        # new_X = SelectKBest(chi2, k=60).fit_transform(new_X, y)

        label_codes = {
            'Bug': 1,
            'Not_Bug': 0
        }

        # label mapping
        df = df.replace({'label':label_codes})

        X = df.iloc[:, :-1].copy()
        y = df['label']

        # X = X.to_numpy()
        # y = y.to_numpy()

    # step05: split into training dataframe & testing dataframe
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
        return (X_train, X_test, y_train, y_test)

    def train(self, X_train, X_test, y_train, y_test):
        from sklearn import svm
        from pprint import pprint
        from sklearn import svm
        from pprint import pprint
        from sklearn.model_selection import RandomizedSearchCV

        svc_0 =svm.SVC(random_state=42)
        print('Parameters currently in use:\n')
        pprint(svc_0.get_params())

        # let's train a SVM
        svc_0 =svm.SVC(random_state=42)
        print('Parameters currently in use:\n')
        pprint(svc_0.get_params())

        # hyperparameters
        random_grid = self.get_random_grid()
        pprint(random_grid)
        # First create the base model to tune
        svc = svm.SVC(random_state=42)

        # Definition of the random search
        random_search = RandomizedSearchCV(estimator=svc,
                                           param_distributions=random_grid,
                                           n_iter=30,
                                           scoring='accuracy',
                                           cv=3,
                                           verbose=1,
                                           random_state=42)

        # Fit the random search model
        random_search.fit(X_train, y_train)
        print("The best hyperparameters from Random Search are:")
        print(random_search.best_params_)
        # print("")
        # print("The mean accuracy of a model with these hyperparameters is:")
        # print(random_search.best_score_)

        # random search was better
        best_svc = random_search.best_estimator_

        # fit the model
        best_svc.fit(X_train, y_train)
        # print(f' predictions: \n {best_svc.predict(X_test)}')
        # print(f' predictions_proba: \n {best_svc.predict_proba(X_test)}')

        # Accuracy
        print("Accuracy: ")
        print(f'training: {accuracy_score(y_train, best_svc.predict(X_train))}')
        print(f'test    : {accuracy_score(y_test, best_svc.predict(X_test))}')

        # F1-score
        print("F1-score: ")
        print(f'training: {f1_score(y_train, best_svc.predict(X_train))}')
        print(f'test    : {f1_score(y_test, best_svc.predict(X_test))}')

            # pipeline = Pipeline(['naivebayes'])
            # param_grid = self.get_param_grid()
            # best_model, accuracy_score_result, f1_score_result = self.train_helper(X_train, y_train, param_grid, pipeline)
            #
            # accuracy_score_result_test = str(int(accuracy_score(y_test, best_model.predict(X_test)) * 100) )
            # f1_score_result_test = str( int(f1_score(y_test, best_model.predict(X_test)) * 100) )
            #
            # self.printService.print_result_here(accuracy_score_result, accuracy_score_result_test, f1_score_result,
            #                                     f1_score_result_test, "", "")

    def get_random_grid(self):
        # C = [.0001, .001, .01, 1]
        # gamma = [.0001, .001, .01, .1, 1, 10, 100]
        # degree = [1, 2, 3, 4, 5]
        # kernel = ['linear', 'rbf', 'poly']
        # probability = [True]

        C = [1]
        gamma = [100]
        degree = [1]
        kernel = ['rbf']
        probability = [True]
        random_grid = {'C': C,
                       'kernel': kernel,
                       'gamma': gamma,
                       'degree': degree,
                       'probability': probability
                       }
        return random_grid

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

            if is_develop_mode:
                self.get_available_param_for_estimators(grid_search)

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

    def get_available_param_for_estimators(self, grid_search):
        for param in grid_search.get_params().keys():
            print(param)

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

