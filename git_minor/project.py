from pydriller import Repository
from util_module import RepositoryClone
from pydriller import Git
from smell_dector import Organic
from smell_dector import OrganicParser

class Project:
    def __init__(self, name, url, branch, starting_commit, ending_commit):
        self.name = name
        self.url = url
        self.branch = branch
        self.starting_commit = starting_commit
        self.ending_commit = ending_commit

        self.git = None

        self.author_modified_files = {} # {author_nwsx
        # ame: number_of_modified_files} - Map style
        self.commit_most_modified_lines = {'hash': None, 'lines': 0} # just one json format

    def collect_statistics(self):
        repo = Repository(self.url, from_commit=self.starting_commit)
        generator = repo.traverse_commits() # a generator is a thing in Python ; what is yield vs return ?

        for commit in generator:
            self.process_commit(commit)

    def process_commit(self, commit):
        author_name = commit.author.name
        modified_files = commit.files
        if author_name in self.author_modified_files:
            self.author_modified_files[author_name] += modified_files
        else:
            self.author_modified_files[author_name] = modified_files

        modifined_lines = commit.lines
        if self.commit_most_modified_lines['lines'] < modifined_lines:
            self.commit_most_modified_lines = {
                'hash': commit.hash,
                'lines': modifined_lines
            }

    def get_statistics(self):
        author_name = max(self.author_modified_files, key=self.author_modified_files.get) # ??? .get

        _hash, _lines = self.commit_most_modified_lines.values();

        return f'Project name: {self.name}\n' \
                f'\tThe author with most modifications is {author_name} with ' \
                f'{self.author_modified_files[author_name]} modified_files.\n' \
                f'\tThe commit with most modified lines is {_hash} with {_lines} lines.'

    def detect_smells_initial_commit(self, smell_detector, output_folder):
        repo_local_path = RepositoryClone.clone(self.name, self.url, self.branch, output_folder)
        print("local folder: ", repo_local_path)

        self.git = Git(repo_local_path) # reference
        self.git.checkout(self.starting_commit)

        # WHAT CONNECNTS HERE: self.name
        organic = Organic(self.name, smell_detector, output_folder, repo_local_path)
        smell_file = organic.detect_smells()

        # parse the json file
        parser = OrganicParser(smell_file)
        project_smells = parser.parse()
        print(f'Commit {self.starting_commit} contains {project_smells.get_total_smells()}')
