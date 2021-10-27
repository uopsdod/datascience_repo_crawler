import json

from smell_dector import ProjectSmells

# this will be used by both method-level and class-level
def _get_smells(data):
    smells = []
    if not data:
        return None

    for d in data:
        smells.append(d['name'])

    return smells

class OrganicParser:

    def __init__(self, smell_file_path):
        self.smell_file = smell_file_path

    def parse(self) -> ProjectSmells:
        with open(self.smell_file) as smell_file:
            data = json.load(smell_file)

        project_smells = ProjectSmells()

        # parse all the smells from the file, starting with the class-level smells
        for d in data: # data is a json array [{...},{...},[...}]
            smelly_element = ProjectSmells.SmellyElement(d['fullyQualifiedName'])
            smelly_element.add_class_level_smell(_get_smells(d['smells'])) # for class-level smells
            for d_method in d['methods']:
                smelly_element.add_method_leve_smell(_get_smells(d_method['smells']))

            project_smells.add_smelly_element(smelly_element)

        return project_smells