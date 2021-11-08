import json

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.utils import resample


class DatasetService:
    def __init__(self):
        pass

    def load_file(self, dataset_type, elem_keys):
        file_path = self._get_dataset_filename(dataset_type)
        with open(file_path) as json_file:
            data_raw = json.load(json_file)
            data_gleaned = self._glean_data(data_raw, elem_keys)
            df = pd.DataFrame(data_gleaned, columns = elem_keys)
            return df

    def balance_dataset(self, dataset_type, df, class_y):
        class1_count, class2_count = self._get_class_count(df, class_y)

        if (class1_count != class2_count):
            print(f'unbalanced {dataset_type}: {class1_count} vs {class2_count} ')

            df = self.upsample(df, class_y)
            class1_count, class2_count = self._get_class_count(df, class_y)

            print(f'upsampled {dataset_type}: {class1_count} vs {class2_count} ')
            return False
        else:
            print(f'balanced {dataset_type}: {class1_count} vs {class2_count} ')
        return True

    def _get_class_count(self, df, class_y):
        value_counts = df[class_y].value_counts()
        class1_count = value_counts[0]
        class2_count = value_counts[1]
        return class1_count, class2_count

    def fill_null_val(self, df, feature, default_val):
        df[feature].fillna(default_val, inplace = True)

    def fill_nan_mean(self, df, feature):
        df[feature].fillna((df[feature].mean()), inplace=True)

    def upsample(self, df, class_y):
        value_counts = df[class_y].value_counts()
        value_counts_key = value_counts.keys()
        key1 = value_counts_key[0];
        key2 = value_counts_key[1];

        class1_count = value_counts[0]
        class2_count = value_counts[1]

        class_count_max = max(class1_count, class2_count)

        # Separate majority and minority classes
        if (class1_count < class2_count):
            df_majority = df[df[class_y]==key2]
            df_minority = df[df[class_y]==key1]
        else:
            df_majority = df[df[class_y]==key1]
            df_minority = df[df[class_y]==key2]

        # Upsample minority class
        df_minority_upsampled = resample(df_minority,
                                         replace=True,                 # sample with replacement
                                         n_samples=class_count_max,    # to match majority class
                                         random_state=123)             # reproducible results

        # Combine majority class with upsampled minority class
        df_upsampled = pd.concat([df_majority, df_minority_upsampled])

        # Display new class counts
        # print(df_upsampled[class_y].value_counts())

        return df_upsampled

    def _get_dataset_filename(self, dataset_type):
        dateset_folder = "training_dataset"
        if dataset_type == "bug":
            dateset_file = "Bug_tt.json"
        elif dataset_type == "feature":
            dateset_file = "Feature_tt.json"
        elif dataset_type == "rating":
            dateset_file = "Rating_tt.json"
        elif dataset_type == "userexperience":
            dateset_file = "UserExperience_tt.json"
        else:
            return "none"

        return dateset_folder + "/" + dateset_file

    def _glean_data(self, json_file, elem_keys):
        data = []
        for elem in json_file:
            example = []
            for elem_key in elem_keys:
                example.append(elem[elem_key])
            data.append(example)
        return data


