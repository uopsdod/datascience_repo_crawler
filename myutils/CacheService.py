import json


class CacheService:
    def __init__(self):
        self.cache_filename = 'cache.json'
        pass

    def clear_cache(self, modelname, smell_type):
        with open(self.cache_filename) as json_file:
            data = json.load(json_file)
            if f'{modelname}_{smell_type}_accuracy_trained' in data:
                data.pop(f'{modelname}_{smell_type}_accuracy_trained', None)
                data.pop(f'{modelname}_{smell_type}_accuracy_test', None)
                data.pop(f'{modelname}_{smell_type}_f1score_trained', None)
                data.pop(f'{modelname}_{smell_type}_f1score_test', None)

        # rewrite
        self._write_to_cache(data)

    def use_cache(self, modelname, smell_type):
        with open(self.cache_filename) as json_file:
            data = json.load(json_file)
            if f'{modelname}_{smell_type}_accuracy_trained' in data:
                accuracy_score_result_cached = data[f'{modelname}_{smell_type}_accuracy_trained']
                accuracy_score_result_test_cached = data[f'{modelname}_{smell_type}_accuracy_test']
                f1_score_result_cached = data[f'{modelname}_{smell_type}_f1score_trained']
                f1_score_result_test_cached = data[f'{modelname}_{smell_type}_f1score_test']

                return accuracy_score_result_cached, accuracy_score_result_test_cached, f1_score_result_cached, f1_score_result_test_cached
                # if accuracy_score_result_cached and f1_score_result_cached:
                # self.print_result_here(accuracy_score_result_cached, accuracy_score_result_test_cached, f1_score_result_cached,
                #                        f1_score_result_test_cached, operation, smell_type)
                # return True
        return None, None, None, None

    def update_cache(self, accuracy_score_result, accuracy_score_result_test, f1_score_result, f1_score_result_test,
                     modelname, smell_type):
        with open(self.cache_filename) as json_file:
            data = json.load(json_file)

        data[f'{modelname}_{smell_type}_accuracy_trained'] = accuracy_score_result
        data[f'{modelname}_{smell_type}_f1score_trained'] = f1_score_result
        data[f'{modelname}_{smell_type}_accuracy_test'] = accuracy_score_result_test
        data[f'{modelname}_{smell_type}_f1score_test'] = f1_score_result_test

        self._write_to_cache(data)

    def _write_to_cache(self, data):
        with open(self.cache_filename, 'w') as outfile:
            json.dump(data, outfile, sort_keys=True, indent=4)
