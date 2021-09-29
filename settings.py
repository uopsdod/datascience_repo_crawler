import json
import os

# _ underscore gives hint that this function is for internal use
def _get_absolute_path(relative_path):
    script_path = os.path.dirname(__file__)
    abs_path = os.path.join(script_path, relative_path)

    return abs_path

class Settings:
    _all_configs = None
    _file = "./config.json"
    def __init__(self):
        self.config = Settings._all_configs
        self.load()

    def load(self):
        if Settings._all_configs is not None:
            return

        with open(self._file) as config_file:
            self.config = json.loads(config_file.read())
            Settings._all_configs = self.config; # if you have multiple configs, Settings._all_configs will be an arrayy

    def get_repository_file_path(self):
        return _get_absolute_path(self.config['repository_file'])

    def get_output_folder_path(self):
        return _get_absolute_path((self.config['output_folder']))

    def get_smell_detector_path(self):
        return _get_absolute_path((self.config['smell_detector']))


