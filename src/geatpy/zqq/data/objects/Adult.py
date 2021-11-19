import geatpy as ea
from geatpy.zqq.data.objects.Data import Data
import numpy as np
import pandas as pd


class Adult(Data):
    def __init__(self):
        Data.__init__(self)
        self.dataset_name = 'adult'
        self.class_attr = 'income-per-year'
        self.positive_class_val = '>50K'
        self.negative_class_val = '<=50K'
        self.sensitive_attrs = ['race', 'sex']
        self.privileged_class_names = ['White', 'Male']
        self.categorical_features = [ 'workclass', 'education', 'marital-status', 'occupation', 
                                      'relationship', 'native-country' ]
        self.features_to_keep = [ 'age', 'workclass', 'education', 'education-num', 'marital-status',
                                  'occupation', 'relationship', 'race', 'sex', 'capital-gain',
                                  'capital-loss', 'hours-per-week', 'native-country',
                                  'income-per-year' ]
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
        """
         删除race中的data，只剩下 'White', 'Black', 'Asian-Pac-Islander'
        """
        # dataframe.reset_index(inplace=True, drop=True)
        # for attribution in self.preserve_data:
        #     delete_classes = self.preserve_data[attribution]
        #     index_deleting = np.where(~dataframe[attribution].isin(delete_classes))
        #     dataframe = dataframe.drop(index=np.array(index_deleting[0]))
        return dataframe



