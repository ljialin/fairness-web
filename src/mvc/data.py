"""
@DATE: 2021/8/20
@Author: Ziqi Wang
@File: data.py
"""

import os
import glob

import pandas as pd
import torch
import pandas
import numpy as np
from root import PRJROOT
# from utils import FileUtils
from werkzeug.utils import secure_filename
from src.common_fair_analyze import N_SPLIT
from sklearn.preprocessing import StandardScaler
from flask_babel import gettext as _


class DataModel:
    def __init__(self, name, file_path):
        # file_path 为不带拓展名的绝对路径
        self.name = name
        self.file_path = file_path
        self.featrs = []
        self.n_featrs = set()
        self.c_featrs = set()
        self.numberical_bounds = {} # 数值型数据上下界
        self.categorical_map = {}
        self.temporal = False
        self.label = ''
        self.pos_label_val = ''
        self.neg_label_val = ''
        self.label_map = {}

        self.data = None  # 执行优化算法训练NN的读取格式
        self.data4eval = None  # 执行数据集公平性判断的读取格式（主要区别在于加了ID和数值属性分5份）

        self.errinfo = self.__load_data(file_path)
        self.__train_data = None
        self.get_processed_data()
        print()

    def __load_data(self, file_path):
        # 绝对路径
        # self.raw_data = pandas.read_csv(file_path + '.csv').applymap(str)
        rela_path = file_path[len(PRJROOT):]
        if len(rela_path) >= 9 and file_path[len(PRJROOT):].find('data/temp', 0):
            # 若为temp文件夹里的文件，则对象释放同时删除对应文件
            self.temporal = True

        #gsh add
        try:
            self.data = pandas.read_csv(self.file_path + '.csv').applymap(str)
            clazz = self.data.loc[0]
            self.data = self.data.drop(0)
            featrs = list(self.data)
            for featr in featrs:
                if clazz[featr] == "categorical":
                    values = self.data[featr]
                    values = list(set(values))
                    values = sorted(values)
                    tmp = {}
                    for i, value in enumerate(values):
                        tmp[value] = i
                    self.categorical_map[featr] = tmp
                    self.c_featrs.add(featr)
                elif clazz[featr] == "numberical":
                    self.numberical_bounds[featr] = []
                    self.n_featrs.add(featr)
                elif clazz[featr] == "label":
                    values = self.data[featr]
                    values = list(set(values))
                    values = sorted(values)
                    self.label = featr
                    if len(values) != 2:
                        return _("label_error")
                    self.pos_label_val = values[0]
                    self.neg_label_val = values[1]
                    self.label_map[values[0]] = 1
                    self.label_map[values[1]] = 0
                else:
                    return f'{_("attribute")}{featr}{_("should_be_defined_as")}numberical、categorical or label'
            featrs.remove(self.label)
            self.featrs = featrs

            # 对data进行处理
            self.data[list(self.n_featrs)] = self.data[list(self.n_featrs)].applymap(float)  # 数值型的数据转浮点

            self.data4eval = self.data.copy()
            self.data4eval.insert(0, 'ID', [*range(len(self.data4eval))])
            for n_featr in self.n_featrs:
                self.__group_n_featr(n_featr)  # 分五份
            return ''
        except:
            return _("update_error")

        #gsh add finish


    def load_data(self):
        attrs = {}
        for featr in self.featrs:
            attrs[featr] = []
            if featr in self.categorical_map.keys(): #这里可以优化时间复杂度
                for key in self.categorical_map[featr].keys():
                    attrs[featr].append(key)
        return self.data, attrs, {'name':self.label, 'label':[self.pos_label_val, self.neg_label_val]}


    def __group_n_featr(self, featr):
        # 数值型属性划分五分
        vmax = self.data4eval[featr].max()
        vmin = self.data4eval[featr].min()
        d = (vmax - vmin + 1e-5) / N_SPLIT

        legi_groups = [
            f'{vmin + i * d:.3g}-{vmin + (i + 1) * d:.3g}'
            for i in range(N_SPLIT)
        ]
        original_vals = self.data4eval[featr].values
        new_col = [
            legi_groups[int((val - vmin) / d)]
            for val in original_vals
        ]
        self.data4eval.insert(0, f'{featr} groups', new_col)

    def update_prediction(self, predictions):
        # 这个方法服务于现有模型的预测，进一步用于公平性分析，所有data用data4eval
        self.data4eval.insert(len(self.data4eval.columns), 'prediction', predictions)
        binary_predictions = [
            1 if prediction > 0.5 else 0
            for prediction in self.data4eval['prediction'].values
        ]
        print(np.sum(binary_predictions), len(binary_predictions))
        self.data4eval.insert(len(self.data4eval.columns), 'binary prediction', binary_predictions)

    def get_ctgrs(self, featr):
        # print(ctgr)
        return [key for key in self.categorical_map[featr].keys()]

    def get_raw_data(self): #旧接口，只在这里用了
        if self.data is None:
            self.data = pandas.read_csv(self.file_path + '.csv').applymap(str)
            self.data[list(self.n_featrs)] = self.data[list(self.n_featrs)].applymap(float) # 数值型的数据转浮点
            self.data.insert(0, 'ID', [*range(len(self.data))])
            for n_featr in self.n_featrs:
                self.__group_n_featr(n_featr) #除了这里还有一个地方用了
        return self.data

    def get_groups(self, featr):
        if featr in self.n_featrs:
            raise RuntimeError('数值类特征没有群体之分')
        else:
            return list(self.categorical_map[featr].keys())

    def get_processed_data(self, norm=1):
        if self.__train_data is None:
            self.__process_train_data(norm)

            # normalize = StandardScaler()
            # normalize.fit(self.__train_data[0])
            # np_data = torch.FloatTensor(normalize.transform(self.__train_data[0]))
            # self.__train_data = (np_data, self.__train_data[1])

        return self.__train_data


    def __process_train_data(self, norm):
        raw_data, attrs, label = self.load_data()
        num_input = len(self.n_featrs) + sum(
            len(self.categorical_map[c_featr])
            for c_featr in self.c_featrs
        ) #离散型变成onehot之后数据的维度数量
        np_data = np.zeros((len(raw_data), num_input), np.float32)

        start = 0
        for featr in attrs.keys():
            featr_class = attrs[featr]
            frame = raw_data[featr]
            if len(featr_class) == 0:
                end = start + 1
                np_data[:, start] = frame.to_numpy()
                start = end

        for featr in attrs.keys():
            featr_class = attrs[featr]
            frame = raw_data[featr]
            if len(featr_class) != 0: #离散型
                end = start + len(featr_class)
                row_indexs = list(range(len(raw_data)))
                col_indexs = list(featr_class.index(val) + start for val in frame)
                np_data[row_indexs, col_indexs] = 1
                start = end


        labels = list(map(int, (self.label_map[val] for val in raw_data[self.label].values)))
        # labels = list(set(raw_data[self.label].values))


        if norm:
            normalize = StandardScaler()
            normalize.fit(np_data)
            np_data = normalize.transform(np_data)

        self.__train_data = (torch.tensor(np_data), torch.tensor(labels))


    # wzq ver.
    # def __process_train_data(self):
    #     num_input = len(self.n_featrs) + sum(
    #         len(self.categorical_map[c_featr])
    #         for c_featr in self.c_featrs
    #     )
    #     # print(num_input)
    #     # raw_data = self.get_raw_data() #只用了一次 #gsh comment
    #     raw_data = self.data
    #     np_data = np.zeros((len(raw_data), num_input), np.float32)
    #
    #     start = 0
    #     for featr in self.featrs:
    #         frame = raw_data[featr]
    #         if featr in self.n_featrs:
    #             # min-max normalization
    #             end = start + 1
    #             # print(frame.to_numpy().shape)
    #             np_data[:, start] = frame.to_numpy()
    #             min_val = np_data[:, start].min()
    #             max_val = np_data[:, start].max()
    #             np_data[:, start] -= min_val
    #             np_data[:, start] /= (max_val - min_val)
    #         else:
    #             # convert onehot
    #             featr_map = self.categorical_map[featr]
    #             end = start + len(featr_map)
    #             row_indexs = list(range(len(raw_data)))
    #             col_indexs = list(featr_map[val] + start for val in frame)
    #             np_data[row_indexs, col_indexs] = 1
    #         start = end
    #
    #     labels = list(map(int, (self.label_map[val] for val in raw_data[self.label].values)))
    #     self.__train_data = (torch.tensor(np_data), torch.tensor(labels))



    def free(self):
        if self.temporal:
            os.remove(self.file_path + 'np_data = {ndarray: (32561, 108)} [[0.30136988 0.         0.         ... 0.         0.         0.        ], [0.4520548  0.         0.         ... 0.         0.         0.        ], [0.28767124 1.         0.         ... 0.         0.         0.        ], ..., [0.2739726  1.         0.      ...View as Array.txt')
            os.remove(self.file_path + '.csv')


# german_newformet
# data_model = DataModel('german', PRJROOT + 'data/german')


class DataService:
    __instance = None

    def __init__(self):
        self.datasets = {}
        self.__scan_datasets()
        pass

    def __scan_datasets(self):
        csv_set = set()
        # txt_set = set()
        valid_list = []
        for file_abspath in glob.glob(PRJROOT + 'data\\*.*'):
            file_name = file_abspath[:-4]
            if file_abspath[-4:] == '.csv':
                valid_list.append(file_name)
                csv_set.add(file_name)
        valid_list.sort()
        for file_name in valid_list:
            with open(file_name + '.csv', 'r') as f:
                next(f)
                self.datasets[file_name.split('\\')[-1]] = file_name

    @staticmethod
    def upload_dataset(data, keepfile):
        self = DataService.inst()
        if data.filename[-4:] != '.csv':
            return f'{_("dataset_file_type_error")}.csv'

        tar_path = PRJROOT + 'data'
        if not keepfile:
            tar_path += '/temp'
        data_path = os.path.join(tar_path, secure_filename(data.filename))
        if keepfile and (os.path.isfile(data_path)):
            return _("filename_conflict")
        data.save(data_path)

        name = data.filename[:-4]
        if name in self.datasets.keys():
            os.remove(data_path)
            return _("dataset_duplicate_name")
        self.datasets[name] = data_path[:-4]

        data = pd.read_csv(data_path)
        # 检查数据集是否为空
        if len(data.columns) == 0:
            return _("dataset_format_error")
        size = len(data[data.columns[0]])
        # 检查行数相同
        for each in data.columns:
            if len(data[each]) != size:
                return _("dataset_format_error")
        # 检查是否有两个以上label
        count = 0
        label_name = 0
        for i,each in enumerate(data.loc[0]):
            if each == "label":
                count += 1
                label_name = data.columns[i]
        if count != 1:
            return _("dataset_format_error")
        # 检查label是否有多个
        labels = data[label_name].to_list()
        labels.pop(0)
        labels = set(labels)
        if len(labels) != 2:
            return _("dataset_format_error")

        return f'OK:{name}'

    @staticmethod
    def inst():
        if DataService.__instance is None:
            DataService.__instance = DataService()
        return DataService.__instance


class DataView:
    def __init__(self, datasets):
        self.datasets = [item for item in datasets.keys()]
        self.selected_dataset = None
        self.c_featr_texts = []
        self.n_featr_texts = []
        self.label_text = ''

    def add_dataset(self, dataset):
        print(dataset)
        self.datasets.append(dataset)

    def update_model(self, data_model):
        self.__update_c_featrs(data_model.categorical_map)
        self.__update_n_featrs(data_model.numberical_bounds)
        self.__update_label(data_model.label, data_model.label_map)

    def __update_c_featrs(self, categorical_map):
        self.c_featr_texts.clear()
        for featr in categorical_map.keys():
            items = [featr + ': ']
            for value in categorical_map[featr].keys():
                items.append(value)
                items.append(', ')
            self.c_featr_texts.append(''.join(items[:-1]))

    def __update_n_featrs(self, numberical_bounds):
        # 获取html中需要的每个numberical features对应的文本
        self.n_featr_texts.clear()
        for featr in numberical_bounds.keys():
            items = [featr]
            end = 1
            if numberical_bounds[featr]:
                end = -1
                items.append(': ')
            for bound in numberical_bounds[featr]:
                items.append('%.3f ~ %.3f' % bound)
                items.append(', ')
            self.n_featr_texts.append(''.join(items[:end]))

    def __update_label(self, label, label_map):
        items = [label, *label_map.keys()]
        self.label_text = '%s: %s, %s' % tuple(items)


class DataController:
    tar_urls = {'/data-eval', '/model-upload', '/algo-cfg'}
    insts = {}

    def __init__(self, ip, target):
        if target not in DataController.tar_urls:
            raise RuntimeError(f'Unknown target page: {target}')
        # self.service = DataService() self.datasets的获取通过重写__getattr__实现
        self.model = None
        self.view = DataView(self.datasets)
        DataController.insts[ip] = self
        self.target = target
        pass

    def upload_dataset(self, data_fname, keepfile=False):
        res = DataService.upload_dataset(data_fname, keepfile) #合法性检查
        if res[:3] == 'OK:':
            self.view.add_dataset(res[3:]) #调用上传
            self.select_dataset(res[3:])
            return ''
        else:
            return res

    def select_dataset(self, dataset):
        file_path = DataService.inst().datasets[dataset]
        if self.model:
            del self.model
        self.model = DataModel(dataset, file_path)
        # print(self.models.numberical_bounds)
        # print(self.models.categorical_map)
        self.view.update_model(self.model)
        self.view.selected_dataset = dataset
        pass

    def __getattr__(self, item):
        if item == 'datasets':
            return DataService.inst().datasets

    def refresh(self):
        print('deleted')
        if self.model:
            self.model.free()

#
# if __name__ == '__main__':
#     data_model = DataModel('German', PRJROOT + 'data/german1')
#     train_data = data_model.get_train_data()
#     print()
#     # print(data_model.get_train_data())
#
