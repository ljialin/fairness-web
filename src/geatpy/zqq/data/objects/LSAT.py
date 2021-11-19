import geatpy as ea
from geatpy.zqq.data.objects.Data import Data
import numpy as np
import pandas as pd


class LSAT(Data):
    def __init__(self):
        Data.__init__(self)
        self.dataset_name = 'LSAT'
        self.class_attr = 'first_pf'
        self.positive_class_val = '1'
        self.negative_class_val = '0'
        self.sensitive_attrs = ['race', 'sex']
        self.privileged_class_names = ['White', 'male']
        self.categorical_features = ['region_first']
        self.features_to_keep = ['race', 'sex', 'LSAT', 'UGPA', 'region_first', 'ZFYA', 'sander_index', 'first_pf']
        self.missing_val_indicators = ['?']
        # self.preserve_data = {'race': {'White', 'Black', 'Asian-Pac-Islander'}}

    # def handle_missing_data(self, dataframe):
    #     index_missing = []
    #     for i, j in zip(range(dataframe.shape[0]), (dataframe.values.astype(str) == 'nan').sum(axis=1)):
    #         if j > 0:
    #             index_missing.append(i)
    #     data = dataframe.drop(index=index_missing)
    #     data.reset_index(inplace=True, drop=True)
    #     # for col in set(data.columns) - set(data.describe().columns):
    #     #     data[col] = data[col].astype('category')
    #     return data

    def data_specific_processing(self, dataframe):
        # adding a derived sex attribute based on personal_status

        old = dataframe['sex'] == 1
        dataframe.loc[old, 'sex'] = 'famale'
        young = dataframe['sex'] != 'famale'
        dataframe.loc[young, 'sex'] = 'male'
        return dataframe
