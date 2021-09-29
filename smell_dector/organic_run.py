import os
import shutil
from util_module import bash

class Organic:

    _FOLDER_BASE_NAME = "smells"
    _FILE_BASE_NAME = "smells.json"

    """
    Organic run:

    Attributes
        repo_name -- name of the repository to clone 
        
        smell_detector -- path to the Organic tool
        output_folder -- path to the results file
        source_path -- path to the project'ss source code
    """
    def __init__(self, repo_name, smell_detector, output_folder, source_path):
        self.repo_name = repo_name; # instance variables
        self.smell_detector = smell_detector;
        self.source_path = source_path;
        self.output_folder = os.path.join(output_folder, repo_name, Organic._FOLDER_BASE_NAME); # output_folder/{repo_name}/smells
        self.output_file = os.path.join(self.output_folder, Organic._FILE_BASE_NAME) # output_folder/{repo_name}/smells/smells.json

        # f-Strings
        self.command = f"java -jar {self.smell_detector} -sf {self.output_file} -src {self.source_path}"

    def detect_smells(self):
        # check if folder exist, we will delete it
        if os.path.exists(self.output_folder):
            shutil.rmtree(self.output_folder) # TODO: in case of concurrency, let's remove only the directory associate to the repository

        os.makedirs(self.output_folder) # Create all the intermediate folders

        # TODO: implement decorator design pattern (in our next lab
        # run organice
        print(":: Detecting code smells ::")
        bash.run_command(self.command, False)
        print(":: Done ::")

