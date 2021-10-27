
from settings import Settings
from util_module import RepositoryLoader
from git_minor import Project


class Main:
    def __init__(self):
        self.settings = Settings()
        self.repositories = []

    def start(self):
        projects = self._load_projects()
        for project in projects:
            project.collect_statistics()
        print(project.get_statistics())
        # print(self.settings.get_repository_file_path())

    def _load_projects(self):
        projects = []
        repository_loader = RepositoryLoader(self.settings.get_repository_file_path())
        loaded_projects = repository_loader.load();

        for p in loaded_projects:
            project = Project(
                p['repo_name'],
                p['git_url'],
                p['branch'],
                p['starting_commit'],
                p['ending_commit']
            )
            projects.append(project)

        return projects

    def collect_smells_initial_commit(self):
        smell_detector = self.settings.get_smell_detector_path()
        projects = self._load_projects()
        for p in projects:
            p.detect_smells_initial_commit(smell_detector, self.settings.get_output_folder_path())

main = Main();
# main.start();
main.collect_smells_initial_commit()

# This is a sample Python script.
# Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     print("\nend");
#     main = Main();
#     main.start();

    # print_hi('PyCharm')
    # print(df.head())

