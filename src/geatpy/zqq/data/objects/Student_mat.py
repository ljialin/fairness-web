import geatpy as ea
from geatpy.zqq.data.objects.Data import Data
import numpy as np
import pandas as pd


class Student_mat(Data):
    def __init__(self):
        Data.__init__(self)
        self.dataset_name = 'student_mat'
        self.class_attr = 'G3'
        self.positive_class_val = 'pass'
        self.negative_class_val = 'fail'
        self.sensitive_attrs = ['age', 'sex']
        self.privileged_class_names = ['F', 'M']
        self.categorical_features = ['school', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob',
                                 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime', 'failures', 'schoolsup',
                                 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic', 'famrel',
                                 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences']
        self.features_to_keep = ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob',
                                 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime', 'failures', 'schoolsup',
                                 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic', 'famrel',
                                 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2', 'G3']
        self.missing_val_indicators = ['?']
        self.preserve_data = {'age': {15, 16, 17, 18}}

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
        # Filter as done here:
        # https://github.com/propublica/compas-analysis/blob/master/Compas%20Analysis.ipynb
        core_pass = dataframe['G3'] >= 10
        dataframe.loc[core_pass, 'G3'] = 'pass'
        score_faile = dataframe['G3'] != 'pass'
        dataframe.loc[score_faile, 'G3'] = 'fail'

        # 删除race中的data，只剩下 'White', 'Black', 'Asian-Pac-Islander'
        # dataframe.reset_index(inplace=True, drop=True)
        for attribution in self.preserve_data:
            delete_classes = self.preserve_data[attribution]
            index_deleting = np.where(~dataframe[attribution].isin(delete_classes))
            dataframe = dataframe.drop(index=np.array(index_deleting[0]))
        dataframe.reset_index(inplace=True, drop=True)
        return dataframe



'''
    school
    GP    349
    MS     46
    Name: school, dtype: int64
    
    sex
    F    208
    M    187
    Name: sex, dtype: int64
    
    age
    16    104
    17     98
    15     82
    18     82
    19     24
    20      3
    21      1
    22      1
    Name: age, dtype: int64
    
    address
    U    307
    R     88
    Name: address, dtype: int64
    
    famsize
    GT3    281
    LE3    114
    Name: famsize, dtype: int64
    
    Pstatus
    T    354
    A     41
    Name: Pstatus, dtype: int64
    
    Medu
    4    131
    2    103
    3     99
    1     59
    0      3
    Name: Medu, dtype: int64
    
    Fedu
    2    115
    3    100
    4     96
    1     82
    0      2
    Name: Fedu, dtype: int64
    
    Mjob
    other       141
    services    103
    at_home      59
    teacher      58
    health       34
    Name: Mjob, dtype: int64
    
    Fjob
    other       217
    services    111
    teacher      29
    at_home      20
    health       18
    Name: Fjob, dtype: int64
    
    reason
    course        145
    home          109
    reputation    105
    other          36
    Name: reason, dtype: int64
    
    guardian
    mother    273
    father     90
    other      32
    Name: guardian, dtype: int64
    
    traveltime
    1    257
    2    107
    3     23
    4      8
    Name: traveltime, dtype: int64
    
    studytime
    2    198
    1    105
    3     65
    4     27
    Name: studytime, dtype: int64
    
    failures
    0    312
    1     50
    2     17
    3     16
    Name: failures, dtype: int64
    
    schoolsup
    no     344
    yes     51
    Name: schoolsup, dtype: int64
    
    famsup
    yes    242
    no     153
    Name: famsup, dtype: int64
    
    paid
    no     214
    yes    181
    Name: paid, dtype: int64
    
    activities
    yes    201
    no     194
    Name: activities, dtype: int64
    
    nursery
    yes    314
    no      81
    Name: nursery, dtype: int64
    
    higher
    yes    375
    no      20
    Name: higher, dtype: int64
    
    internet
    yes    329
    no      66
    Name: internet, dtype: int64
    
    romantic
    no     263
    yes    132
    Name: romantic, dtype: int64
    
    famrel
    4    195
    5    106
    3     68
    2     18
    1      8
    Name: famrel, dtype: int64
    
    freetime
    3    157
    4    115
    2     64
    5     40
    1     19
    Name: freetime, dtype: int64
    
    goout
    3    130
    2    103
    4     86
    5     53
    1     23
    Name: goout, dtype: int64
    
    Dalc
    1    276
    2     75
    3     26
    4      9
    5      9
    Name: Dalc, dtype: int64
    
    Walc
    1    151
    2     85
    3     80
    4     51
    5     28
    Name: Walc, dtype: int64
    
    health
    5    146
    3     91
    4     66
    1     47
    2     45
    Name: health, dtype: int64
    
    absences
    0     115
    2      65
    4      53
    6      31
    8      22
    10     17
    12     12
    14     12
    3       8
    16      7
    7       7
    5       5
    18      5
    20      4
    22      3
    1       3
    13      3
    15      3
    11      3
    9       3
    28      1
    56      1
    54      1
    40      1
    38      1
    30      1
    17      1
    26      1
    25      1
    24      1
    23      1
    21      1
    19      1
    75      1
    Name: absences, dtype: int64
    
    G1
    10    51
    8     41
    11    39
    7     37
    12    35
    13    33
    9     31
    14    30
    6     24
    15    24
    16    22
    17     8
    18     8
    5      7
    19     3
    4      1
    3      1
    Name: G1, dtype: int64
    
    G2
    9     50
    10    46
    12    41
    13    37
    11    35
    15    34
    8     32
    14    23
    7     21
    5     15
    6     14
    16    13
    0     13
    18    12
    17     5
    19     3
    4      1
    Name: G2, dtype: int64
    
    G3
    pass    265
    fail    130
    Name: G3, dtype: int64




'''