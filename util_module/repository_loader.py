import csv
# pep eight guidelines
class RepositoryLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load(self):
        # inner function -> robustness -> check for empty value
        def get_value(string):
            string = string.strip() # remove whitespaces

            if string: # is if not None
                return string
            else:
                return None

        with open(self.file_path) as repo_file:
            reader = csv.DictReader(repo_file, delimiter=';', quotechar='"')
            projects = []

            for row in reader:
                project = {
                    "repo_name": get_value(row["repo_name"]),
                    "git_url": get_value(row["git_url"]),
                    "branch": get_value(row["branch"]),
                    "starting_commit": get_value(row["starting_commit"]),
                    "ending_commit": get_value(row["ending_commit"])
                }
                projects.append(project)

        return projects