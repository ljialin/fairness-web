import geatpy as ea
from geatpy.zqq.data.objects.Data import Data
import numpy as np
import pandas as pd

# https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients
class Default(Data):
    def __init__(self):
        Data.__init__(self)
        self.dataset_name = 'default'
        self.class_attr = 'default_payment_next_month'
        self.positive_class_val = '1'
        self.negative_class_val = '0'
        self.sensitive_attrs = ['SEX']
        self.privileged_class_names = ['1']
        self.categorical_features = ['EDUCATION', 'MARRIAGE']
        self.features_to_keep = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
                                 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
                                 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
                                 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6',
                                 'default_payment_next_month']
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

        # dataframe.reset_index(inplace=True, drop=True)
        # for attribution in self.preserve_data:
        #     delete_classes = self.preserve_data[attribution]
        #     index_deleting = np.where(~dataframe[attribution].isin(delete_classes))
        #     dataframe = dataframe.drop(index=np.array(index_deleting[0]))
        return dataframe
