import geatpy as ea
from geatpy.zqq.data.objects.Data import Data
import numpy as np
import pandas as pd


class Dutch(Data):
    def __init__(self):
        Data.__init__(self)
        self.dataset_name = 'dutch'
        self.class_attr = 'occupation'
        self.positive_class_val = '2_1'  # not sure
        self.negative_class_val = '5_4_9'  # not sure
        self.sensitive_attrs = ['sex']
        self.privileged_class_names = ['1']
        self.categorical_features = ['household_position', 'prev_residence_place',
                                 'citizenship', 'country_birth', 'edu_level', 'economic_status', 'cur_eco_activity',
                                 'Marital_status']
        self.features_to_keep = ['sex', 'age', 'household_position', 'household_size', 'prev_residence_place',
                                 'citizenship', 'country_birth', 'edu_level', 'economic_status', 'cur_eco_activity',
                                 'Marital_status', 'occupation']
        self.missing_val_indicators = ['?']
        self.preserve_data = {'sex': {'1', '2'}}

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



'''
country_birth
1    56058
3     2821
2     1541
Name: country_birth, dtype: int64

prev_residence_place
1    58943
2     1477
Name: prev_residence_place, dtype: int64

economic_status
111    51340
120     4771
112     4309
Name: economic_status, dtype: int64

sex
2    30273
1    30147
Name: sex, dtype: int64

household_size
112    17237
114    16370
113    12238
111     6529
125     6179
126     1867
Name: household_size, dtype: int64

citizenship
1    59225
2      843
3      352
Name: citizenship, dtype: int64

household_position
1122    26225
1121     9975
1110     7229
1210     6529
1131     5818
1132     2291
1140     1824
1220      529
Name: household_position, dtype: int64

Marital_status
2    36655
1    19656
4     3566
3      543
Name: Marital_status, dtype: int64

cur_eco_activity
131    11621
135    10239
138     8168
122     6505
137     5862
136     4294
133     3062
139     2661
132     2616
134     1940
111     1738
124     1714
Name: cur_eco_activity, dtype: int64

edu_level
3    22672
5    18109
2    12326
1     4513
4     2580
0      220
Name: edu_level, dtype: int64

occupation
5_4_9    31657
2_1      28763
Name: occupation, dtype: int64

age
8     8748
9     8478
7     8289
10    7880
11    7021
6     6292
5     4770
12    3881
4     3801
13    1022
14     176
15      62
Name: age, dtype: int64


'''