"""
测试数据导入
https://github.com/JSGoedhart/fairness-comparison/tree/df503855bfc2eeb9c89c1075b661bf8de5e6d18c
"""
import os

# from geatpy.zqq.data.objects.list import DATASETS, get_dataset_names
# from geatpy.zqq.data.objects.ProcessedData import ProcessedData
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from geatpy.zqq.detect_groups import detect_allgroups, detect_groups
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
import geatpy as ea
import numpy as np
from itertools import product
from mvc.data import DataModel
import random


def make_class_attr_num(dataframe, positive_val):
    dataframe = dataframe.replace({positive_val: 1})
    # dataframe = dataframe.replace("[^1]", 0, regex=True)
    dataframe[dataframe != 1] = 0
    return dataframe


supported_tags = ["original", "numerical", "numerical-binsensitive",
                  "categorical-binsensitive", "numerical-for-NN", "original_info"]


def get_smaller(data_org, data, label, rate):
    Sss = StratifiedShuffleSplit(n_splits=2, test_size=rate, random_state=0)
    Plan1, Plan2 = Sss.split(data_org, label)

    org_data = data_org.iloc[Plan1[1]]
    data = data.loc[Plan1[1]]
    data_label = label.loc[Plan1[1]]

    org_data.reset_index(inplace=True, drop=True)
    data.reset_index(inplace=True, drop=True)
    data_label.reset_index(inplace=True, drop=True)

    return org_data, data, data_label


def analyze_data(ana_data, alldata, sensitive_attributions):
    attribution = alldata.columns
    # sensitive_attributions = dataset.get_sensitive_attributes()
    group_dict = {}
    sens_idxs_name = []
    test_num = ana_data.shape[0]
    sens_idxs = {}
    if len(sensitive_attributions) == 0:
        return sens_idxs_name, sens_idxs, group_dict

    for sens in sensitive_attributions:
        temp = []
        for attr in attribution:
            temp1 = sens + '_'
            if temp1 in attr:
                temp.append(attr)
                sens_idxs.update({attr: np.array(np.where(ana_data[attr])).reshape(1, -1)})
                sens_idxs_name.append(attr)

        group_dict.update({sens: temp})

    group_attr = []
    sens_idxs_name_final = []
    for sens in sensitive_attributions:
        group_attr.append(group_dict[sens])

    for item in product(*eval(str(group_attr))): # 对2个以上的铭感属性集合做笛卡尔积
        group = item
        flag = np.ones([1, test_num]) == np.ones([1, test_num])
        for g in group:
            flag = flag & ana_data[g]
        g_num = np.sum(flag)
        name = "+".join(str(x) for x in group)
        if g_num != 0:
            # sens_idxs_name.append(name)
            print(g_num)
            g_idx = np.array(np.where(flag)).reshape([1, g_num])
            sens_idxs.update({name: g_idx})
            if name not in sens_idxs_name:
                sens_idxs_name.append(name)
        else:
            sens_idxs.update({name: None})
            if name not in sens_idxs_name:
                sens_idxs_name.append(name)

    return sens_idxs_name, sens_idxs, group_dict


def get_header(dataname, dir):
    f = open(dir + os.sep + "{}.txt".format(dataname), 'r', encoding='utf8')
    count = 0
    attr_num = 0
    attrs = {}
    label = {}
    while 1:
        count += 1
        line = f.readline().strip()
        if count < 3: continue
        if count == 4: attr_num = int(line)
        if count >= 5:
            for i in range(attr_num):
                line = f.readline().strip()
                term = line.split(', ')
                attrs[term[0]] = []
                if term[1] == "categorical":
                    attrs[term[0]] = term[2:]
            f.readline()  # LABEL
            term = f.readline().strip().split(', ')
            label['name'] = term[0]
            label['label'] = term[1:]
            break
    f.close()

    return attrs, label


# 这里可能要加上对数值型敏感属性的处理，将他转化为离散型，不然不会进行onehot转化，
# 从而导致在for sens in sensitive_attributions这一部分没有敏感属性（识别的时候是age_ in 标签）
def df2onehot(df, attrs):
    for attr_name in attrs.keys():
        attr_items = attrs[attr_name]
        for item in attr_items:
            vals = [0 for i in range(df.shape[0])]
            for i in range(len(df[attr_name].values)):
                val = str(df[attr_name].values[i])  # 有些分类的属性可能也会用数字进行表示
                if val == item:
                    vals[i] = 1
            df.insert(df.shape[1], "{}_{}".format(attr_name, item), vals)
        if len(attr_items) > 0:
            df = df.drop(columns=attr_name)
    return df


# dir = "D:\\机器学习公平性平台\\data"
# dataname = 'default'
# df = pd.read_csv(dir + os.sep + "{}.csv".format(dataname))
# attrs, label = get_header(dataname, dir)
# df = df2onehot(df, attrs)
# print()


def load_data(dataModel, preserve_sens_in_net=1, sensitive_attributions=None):
    #datatype="numerical-for-NN"
    #is_ensemble 是读取张清泉预处理数据集用到的参数

    org_data, attrs, label = dataModel.load_data()
    data = df2onehot(org_data.copy(), attrs)

    label_name = label['name']
    data.reset_index(inplace=True, drop=True)
    data_x = data.drop(columns=label_name)
    data_label = data[label_name]

    # train+validation = 80%      test = 20%
    sss1 = StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=20210811)
    plan1, plan2 = sss1.split(data_x, data_label)

    test_org = org_data.iloc[plan1[1]]
    test_data = data_x.loc[plan1[1]]
    test_label = data_label.loc[plan1[1]]

    trainvaliddata_org = org_data.iloc[plan1[0]]
    trainvaliddata = data_x.loc[plan1[0]]
    trainvalidlabel = data_label.loc[plan1[0]]
    trainvaliddata.reset_index(inplace=True, drop=True)
    trainvalidlabel.reset_index(inplace=True, drop=True)
    trainvaliddata_org.reset_index(inplace=True, drop=True)

    # train = 60%      validation = 20%
    sss2 = StratifiedShuffleSplit(n_splits=2, test_size=0.25, random_state=20210811)
    plan3, plan4 = sss2.split(trainvaliddata, trainvalidlabel)

    train_org = trainvaliddata_org.loc[plan3[0]]
    train_data = trainvaliddata.loc[plan3[0]]
    train_label = trainvalidlabel.loc[plan3[0]]

    valid_org = trainvaliddata_org.loc[plan3[1]]
    valid_data = trainvaliddata.loc[plan3[1]]
    valid_label = trainvalidlabel.loc[plan3[1]]

    train_org.reset_index(inplace=True, drop=True)
    train_data.reset_index(inplace=True, drop=True)
    train_label.reset_index(inplace=True, drop=True)
    valid_data.reset_index(inplace=True, drop=True)
    valid_label.reset_index(inplace=True, drop=True)
    valid_org.reset_index(inplace=True, drop=True)
    test_data.reset_index(inplace=True, drop=True)
    test_label.reset_index(inplace=True, drop=True)
    test_org.reset_index(inplace=True, drop=True)

    if preserve_sens_in_net == 1:
        newtrain_data = train_data.copy()
        newtest_data = test_data.copy()
        newvalid_data = valid_data.copy()
        newdata_x = data_x.copy()
    else:
        sens_dict = []
        attribution = train_data.columns
        # sensitive_attributions = dataset.get_sensitive_attributes()
        for sens in sensitive_attributions:
            for attr in attribution:
                temp = sens + '_'
                if temp in attr:
                    sens_dict.append(attr)

        newtrain_data = train_data.copy()
        newtest_data = test_data.copy()
        newvalid_data = valid_data.copy()
        newdata_x = data_x.copy()
        newtrain_data.drop(columns=sens_dict, inplace=True)
        newtest_data.drop(columns=sens_dict, inplace=True)
        newvalid_data.drop(columns=sens_dict, inplace=True)
        newdata_x.drop(columns=sens_dict, inplace=True)

    normalize = StandardScaler()

    # Fitting only on training data
    normalize.fit(newdata_x)
    train_data_norm = normalize.transform(newtrain_data)

    # Applying same transformation to test data
    test_data_norm = normalize.transform(newtest_data)

    # Applying same transformation to validation data
    valid_data_norm = normalize.transform(newvalid_data)

    # Change labels into integer
    train_y = make_class_attr_num(train_label.copy(), dataModel.pos_label_val)
    test_y = make_class_attr_num(test_label.copy(), dataModel.pos_label_val)
    valid_y = make_class_attr_num(valid_label.copy(), dataModel.pos_label_val)

    DATA_names = ['train_data', 'train_data_norm', 'train_label', 'train_y', 'train_org',
                  'valid_data', 'valid_data_norm', 'valid_label', 'valid_y', 'valid_org',
                  'test_data', 'test_data_norm', 'test_label', 'test_y', 'test_org'
                                                                         'org_data', 'positive_class',
                  'positive_class_name',
                  'Groups_info', 'privileged_class_names', 'sens_attrs']

    DATA = dict((k, []) for k in DATA_names)

    DATA['train_data'] = train_data
    DATA['train_data_norm'] = train_data_norm
    DATA['train_label'] = train_label
    DATA['train_y'] = train_y.astype('int')

    DATA['valid_data'] = valid_data
    DATA['valid_data_norm'] = valid_data_norm
    DATA['valid_label'] = valid_label
    DATA['valid_y'] = valid_y.astype('int')

    DATA['test_data'] = test_data
    DATA['test_data_norm'] = test_data_norm
    DATA['test_label'] = test_label
    DATA['test_y'] = test_y.astype('int')

    DATA['positive_class'] = dataModel.pos_label_val

    DATA['org_data'] = org_data
    DATA['train_org'] = train_org
    DATA['test_org'] = test_org
    DATA['valid_org'] = valid_org

    DATA['positive_class_name'] = dataModel.label

    sens_idxs_name_train, sens_idxs_train, group_dict_train = analyze_data(train_data, data, sensitive_attributions)
    sens_idxs_name_valid, sens_idxs_valid, group_dict_valid = analyze_data(valid_data, data, sensitive_attributions)
    sens_idxs_name_test, sens_idxs_test, group_dict_test = analyze_data(test_data, data, sensitive_attributions)

    Groups_name = ['sens_idxs_train', 'sens_idxs_name_train', 'group_dict_train',
                   'sens_idxs_valid', 'sens_idxs_name_valid', 'group_dict_valid',
                   'sens_idxs_test', 'sens_idxs_name_test', 'group_dict_test']
    Groups_info = dict((k, []) for k in Groups_name)
    Groups_info['sens_idxs_train'] = sens_idxs_train
    Groups_info['sens_idxs_name_train'] = sens_idxs_name_train
    Groups_info['group_dict_train'] = group_dict_train
    Groups_info['sens_idxs_valid'] = sens_idxs_valid
    Groups_info['sens_idxs_name_valid'] = sens_idxs_name_valid
    Groups_info['group_dict_valid'] = group_dict_valid
    Groups_info['sens_idxs_test'] = sens_idxs_test
    Groups_info['sens_idxs_name_test'] = sens_idxs_name_test
    Groups_info['group_dict_test'] = group_dict_test

    DATA['Groups_info'] = Groups_info

    return DATA
# return DATA, data_obj


"""
ricci
adult
german
"""
