import json


class PrintService:
    def __init__(self):
        self.padding_count = 20
        pass

    def print_result(self, col1, col2, col3):
        smell_padding = col1.rjust(self.padding_count)
        accuracy_padding = col2.rjust(self.padding_count)
        f1score_padding = col3.rjust(self.padding_count)
        print(smell_padding, accuracy_padding, f1score_padding)

    def print_result_here(self, accuracy_score_result, accuracy_score_result_test, f1_score_result,
                          f1_score_result_test, operation, smell_type):
        if operation == "generate":
            self.print_result(smell_type, accuracy_score_result, f1_score_result)
        elif operation == "compare":
            self.print_result("training set", accuracy_score_result, f1_score_result)
            self.print_result("test set", accuracy_score_result_test, f1_score_result_test)