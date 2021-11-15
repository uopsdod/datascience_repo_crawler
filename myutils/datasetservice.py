import json

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

is_develop_mode = False
class DatasetService:
    def __init__(self):
        pass

    def load_file(self, dataset_type):
        file_path = self._get_dataset_filename(dataset_type)
        with open(file_path) as json_file:
            data_raw = json.load(json_file)
            # data_gleaned = self._glean_data(data_raw, elem_keys)
            df = pd.DataFrame(data_raw)
            return df

    def balance_dataset(self, dataset_type, df, class_y):
        class1_count, class2_count = self._get_class_count(df, class_y)

        if (class1_count != class2_count):
            if (is_develop_mode):
                print(f'unbalanced {dataset_type}: {class1_count} vs {class2_count} ')

            df = self.upsample(df, class_y)
            class1_count, class2_count = self._get_class_count(df, class_y)

            if (is_develop_mode):
                print(f'upsampled {dataset_type}: {class1_count} vs {class2_count} ')
            return df
        else:
            if (is_develop_mode):
                print(f'balanced {dataset_type}: {class1_count} vs {class2_count} ')
        return df

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
        new_rows_array = []
        for index, row in df_minority_upsampled.iterrows():
            new_rows_array.append(row)

        for index, row in df_majority.iterrows():
            new_rows_array.append(row)

        index_list = list(range(len(new_rows_array)))
        new_rows_df = pd.DataFrame(new_rows_array, index=index_list)
        return new_rows_df

        # df_upsampled = df_majority.append(df_minority_upsampled)
        # df_upsampled.index = pd.RangeIndex(len(df_upsampled.index))
        # df_upsampled.reset_index()
        # df_upsampled = pd.concat([df_majority, df_minority_upsampled])

        # Display new class counts
        # print(df_upsampled[class_y].value_counts())

        # return df_upsampled

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

    def get_word_count_of_cleaned_comment(self, df):
        return df['comment'].str.split().str.len()

    def normalize_feature(self, df, feature_name):
        mean = df[feature_name].mean()
        max = df[feature_name].max()
        min = df[feature_name].min()

        df[feature_name] = df[feature_name].apply(lambda x: (x - mean) / (max - min))

    def standardize_feature(self, df, feature_name):
        mean = df[feature_name].mean()
        std = df[feature_name].std()

        df[feature_name] = df[feature_name].apply(lambda x: (x - mean) / (std))

    def convert_to_df(self, df_all_donotchange, df_dataset, label_codes):

        label_positive = list(label_codes)[0]
        label_negative = list(label_codes)[1]

        comments_series = df_dataset['label']
        comments_series.index = df_dataset['comment']
        comments_dict = comments_series.to_dict()

        new_rows_array = []
        for index, row in df_all_donotchange.iterrows():
            # convert comment as one-versus-all approach
            comment_now = row['comment']
            if ( comment_now in comments_dict and comments_dict[comment_now] == label_positive):
                row['label'] = label_positive
            else:
                row['label'] = label_negative

            new_rows_array.append(row)

        new_rows_df = pd.DataFrame(new_rows_array)
        return new_rows_df

    def reorganize_index(self, df):
        # new_index = [range(len(df.index))]
        # df.reindex(new_index)

        row_arrays = df.to_numpy()
        new_rows_df = pd.DataFrame(row_arrays, columns=df.columns)

        # count = 0
        # new_rows_array = []
        # for index, series_row in df.iterrows():
        #     series_row['myindex'] = count
        #     series_row.id = count
        #     count = count + 1
        #     new_rows_array.append(series_row)
        #
        # new_index = [range(len(df.index) - 1)]
        # new_rows_df = pd.DataFrame(new_rows_array, index=new_index)
        # new_rows_df.set_index('myindex')

        # count = 0
        # for index, row in df.iterrows():
        #
        #     row['myindex'] = count
        #     # df.loc[index, 'myindex'] = count
        #     count = count + 1

        return new_rows_df



    def convert_fee_type_to_numeric(self, df_all):

        for index, row in df_all.iterrows():
            # convert fee to number representation for training
            fee_type = row['fee']
            fee_val = 0
            if (fee_type == 'paid'):
                fee_val = -1
            elif (fee_type == 'free'):
                fee_val = 1
            else:
                fee_val = 0
            df_all.loc[index, 'fee'] = fee_val

    def split_dataset(self, df):

        X = df.iloc[:, :-1].copy()
        y = df.iloc[: , -1].copy()

        # X = X.to_numpy()
        # y = y.to_numpy()

        # step05: split into training dataframe & testing dataframe
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
        return (X_train, X_test, y_train, y_test)

    def replace_label_with_zerorone(self, df, dataset_type):
        label_codes = self.get_label_codes(dataset_type)
        # label mapping
        df = df.replace({'label': label_codes})
        return df

    def get_label_codes(self, dataset_type):
        label_codes = {}
        if (dataset_type == "bug"):
            label_codes = {
                'Bug': 1,
                'Not_Bug': 0
            }
        elif (dataset_type == "feature"):
            label_codes = {
                'Feature': 1,
                'Not_Feature': 0
            }
        elif (dataset_type == "rating"):
            label_codes = {
                'Rating': 1,
                'Not_Rating': 0
            }
        elif (dataset_type == "userexperience"):
            label_codes = {
                'UserExperience': 1,
                'Not_UserExperience': 0
            }
        return label_codes

