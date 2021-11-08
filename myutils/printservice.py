import json

from myutils.cacheservice import CacheService


class PrintService:
    def __init__(self):
        self.padding_count = 20
        self.cacheService = CacheService()
        pass

    def print_result(self, col1, col2, col3):
        smell_padding = col1.rjust(self.padding_count)
        accuracy_padding = col2.rjust(self.padding_count)
        f1score_padding = col3.rjust(self.padding_count)
        print(smell_padding, accuracy_padding, f1score_padding)

    def print_result_here(self, accuracy_score_result, accuracy_score_result_test, f1_score_result,
                          f1_score_result_test, operation, type):
        accuracy_score_result = str(accuracy_score_result)
        accuracy_score_result_test = str(accuracy_score_result_test)
        f1_score_result = str(f1_score_result)
        f1_score_result_test = str(f1_score_result_test)

        if operation == "generate":
            self.print_result(type, accuracy_score_result, f1_score_result)
        elif operation == "compare":
            self.print_result("training set", accuracy_score_result, f1_score_result)
            self.print_result("test set", accuracy_score_result_test, f1_score_result_test)

    def print_all_test_result(self):

        smelltypes_to_print = ['feature-envy', 'data-class', 'god-class', 'long-method']
        print(''.rjust(self.padding_count), end="")
        for word in smelltypes_to_print:
            print(word.rjust(self.padding_count), end="")
            print(word.rjust(self.padding_count), end="")
        print()
        resulttypes_to_print = ['', 'Accuracy', 'F1-Score', 'Accuracy', 'F1-Score', 'Accuracy', 'F1-Score', 'Accuracy', 'F1-Score']
        for word in resulttypes_to_print:
            print(word.rjust(self.padding_count), end="")
        print()

        for modelname in ['decisiontree','randomforest','naivebayes','svc']:
            words_to_print = []
            words_to_print.append(modelname)
            for smell_type in smelltypes_to_print:
                (accuracy_score_result_cached,
                 accuracy_score_result_test_cached,
                 f1_score_result_cached,
                 f1_score_result_test_cached) = self.cacheService.use_cache(modelname, smell_type)

                words_to_print.append(accuracy_score_result_test_cached)
                words_to_print.append(f1_score_result_test_cached)

            for word in words_to_print:
                print(word.rjust(self.padding_count), end="")
            print()

    def print_all_test_gap_result(self):

        smelltypes_to_print = ['feature-envy', 'data-class', 'god-class', 'long-method']
        print(''.rjust(self.padding_count), end="")
        for word in smelltypes_to_print:
            print(word.rjust(self.padding_count), end="")
            print(word.rjust(self.padding_count), end="")
        print()
        resulttypes_to_print = ['', 'Accuracy', 'F1-Score', 'Accuracy', 'F1-Score', 'Accuracy', 'F1-Score', 'Accuracy', 'F1-Score']
        for word in resulttypes_to_print:
            print(word.rjust(self.padding_count), end="")
        print()

        for modelname in ['decisiontree','randomforest','naivebayes','svc']:
            words_to_print = []
            words_to_print.append(modelname)
            for smell_type in smelltypes_to_print:
                (accuracy_score_result_cached,
                 accuracy_score_result_test_cached,
                 f1_score_result_cached,
                 f1_score_result_test_cached) = self.cacheService.use_cache(modelname, smell_type)

                words_to_print.append(str(int(accuracy_score_result_test_cached) - int(accuracy_score_result_cached)))
                words_to_print.append(str(int(f1_score_result_test_cached) - int(f1_score_result_cached)))

            for word in words_to_print:
                print(word.rjust(self.padding_count), end="")
            print()