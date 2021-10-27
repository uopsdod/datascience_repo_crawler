


class ProjectSmells:
    def __init__(self):
        self.smelly_elements = []

    def add_smelly_element(self, smelly_element):
        self.smelly_elements.append(smelly_element)

    def get_total_smells(self) -> int:
        total = 0
        for e in self.smelly_elements:
            total += e.get_element_smells() # return int here
        return total

    class SmellyElement:
        def __init__(self, fully_qualified_name):
            self.fully_qualified_name = fully_qualified_name

            self.class_smells = []
            self.method_smells = []

        def add_class_level_smell(self, smell_list):
            if smell_list is not None:
                self.class_smells += smell_list  # add smelllist one by one


        def add_method_leve_smell(self, smell_list):
            if smell_list is not None:
                self.method_smells += smell_list  # add smelllist one by one

        def get_element_smells(self):
            return len(self.class_smells) + len(self.method_smells)