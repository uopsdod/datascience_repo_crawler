import os
import git
import shutil

class RepositoryClone:
    _FOLDER_BASE_NAME = "source"

    @staticmethod
    def clone(repo_name, repo_url, repo_branch, output_folder) -> str:
        # ./miner-outputs/junit/source
        repo_local_path = os.path.join(output_folder, repo_name, RepositoryClone._FOLDER_BASE_NAME)

        # check if folder exists, deleting it if so
        if os.path.exists(repo_local_path):
            shutil.rmtree(repo_local_path) # TODO: in case of concurrency, let's remove only the directory associate to the repository

        os.makedirs(repo_local_path)

        # reason for using gitpython repo : we need "clone"
        #   - pyDriller does not support clone function
        # TODO: checking if repo class was able to clone the repository successfully
        git.Repo.clone_from(repo_url, repo_local_path, branch=repo_branch)

        return repo_local_path