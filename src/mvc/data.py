"""
@DATE: 2021/8/20
@Author: Ziqi Wang
@File: data.py
"""

import os
import glob
import torch
import pandas
import numpy as np
from root import PRJROOT
# from utils import FileUtils
from werkzeug.utils import secure_filename
from src.common_fair_analyze import N_SPLIT


class DataModel:
    def __init__(self, name, file_path):
        # file_path 为不带拓展名的绝对路径
        self.name = name
        self.file_path = file_path
        self.featrs = []
        self.n_featrs = set()
        self.c_featrs = set()
        self.numberical_bounds = {}
        self.categorical_map = {}
        self.temporal = False
        self.label = ''
        self.pos_label_val = ''
        self.neg_label_val = ''
        self.label_map = {}
        self.data = None

        self.errinfo = self.__load_desc(file_path)
        self.__train_data = None

    def __load_desc(self, file_path):
        # 绝对路径
        # self.raw_data = pandas.read_csv(file_path + '.csv').applymap(str)
        rela_path = file_path[len(PRJROOT):]
        if len(rela_path) >= 9 and file_path[len(PRJROOT):].find('data/temp', 0):
            # 若为temp文件夹里的文件，则对象释放同时删除对应文件
            self.temporal = True

        #gsh add
        self.data = pandas.read_csv(self.file_path + '.csv').applymap(str)
        clazz = self.data.loc[0]
        self.data = self.data.drop(0)
        featrs = list(self.data)
        for featr in featrs:
            if clazz[featr] == "categorical":
                values = self.data[featr]
                values = list(set(values))
                tmp = {}
                for i,value in enumerate(values):
                    tmp[value] = i
                self.categorical_map[featr] = tmp
                self.c_featrs.add(featr)
            elif clazz[featr] == "numberical":
                self.numberical_bounds[featr] = []
                self.n_featrs.add(featr)
            elif clazz[featr] == "label":
                values = self.data[featr]
                values = list(set(values))
                self.label = featr
                if len(values) != 2:
                    return f'描述文件中分类属性{featr}的取值类型不足两个'
                self.pos_label_val = values[0]
                self.neg_label_val = values[1]
                self.label_map[values[0]] = 1
                self.label_map[values[1]] = 0
        featrs.remove(self.label)
        self.featrs = featrs

        # 对data进行处理
        self.data[list(self.n_featrs)] = self.data[list(self.n_featrs)].applymap(float)  # 数值型的数据转浮点
        self.data.insert(0, 'ID', [*range(len(self.data))])
        for n_featr in self.n_featrs:
            self.__group_n_featr(n_featr) #分五份
        # exit(-1)
        return ''
        #gsh add finish

        # f = open(file_path + '.txt', 'r')       # 可认为不会出现文件不存在
        # try:
        #     line = next(f).strip()
        #     while line != 'NUMBER OF ATTRIBUTES:':
        #         line = next(f).strip()
        #     num_featrs = int(next(f).strip())
        #
        #     line = next(f).strip()
        #     while line != 'ATTRIBUTES:':
        #         line = next(f).strip()
        #
        #     featr_id = 0
        #     line = next(f).strip()
        #     while line != 'LABEL:':
        #         if line == '':
        #             continue
        #         fields = [*map(lambda x: x.strip(), line.split(','))]
        #         featr_name = fields[0]
        #         if len(fields) < 2:
        #             return f'描述文件中属性{featr_name}没有指定类型'
        #         featr_type = fields[1]
        #         self.featrs.append(featr_name)
        #
        #         if featr_type == 'numberical':
        #             self.numberical_bounds[featr_name] = []
        #             self.n_featrs.add(featr_name)
        #             for bound in fields[2:]:
        #                 try:
        #                     lb, ub = map(lambda x: float(x.strip()), bound.split('~'))
        #                 except ValueError:
        #                     return f'属性{featr_name}的取值范围没有满足 %f ~ %f格式'
        #                 self.numberical_bounds[featr_name].append((lb, ub))
        #         elif featr_type == 'categorical':
        #             if len(fields) < 4:
        #                 return f'描述文件中分类属性{featr_name}的取值类型不足两个'
        #             self.c_featrs.add(featr_name)
        #             self.categorical_map[featr_name] = {}
        #             n = 0
        #             for category in fields[2:]:
        #                 self.categorical_map[featr_name][category] = n
        #                 n += 1
        #         else:
        #             return f'描述文件中属性{featr_name}的类型非法: {fields[1]}'
        #         featr_id += 1
        #         line = f.readline().strip()
        #     if num_featrs != featr_id:
        #         return '描述文件指定的属性数量与实际数量不符'
        #
        #     fields = [*map(lambda x: x.strip(), f.readline().split(','))]
        #     if len(fields) != 3:
        #         return '描述文件中LABEL:之后一行必须为逗号分隔的三个字段'
        #     self.label = fields[0]
        #     self.pos_label_val = fields[1]
        #     self.neg_label_val = fields[2]
        #     self.label_map[fields[1]] = 1
        #     self.label_map[fields[2]] = 0
        # except EOFError:
        #     return '描述文件信息缺失'
        # f.close()
        # return ''

    def __group_n_featr(self, featr):
        vmax = self.data[featr].max()
        vmin = self.data[featr].min()
        d = (vmax - vmin + 1e-5) / N_SPLIT

        legi_groups = [
            f'{vmin + i * d:.3g}-{vmin + (i + 1) * d:.3g}'
            for i in range(N_SPLIT)
        ]
        original_vals = self.data[featr].values
        new_col = [
            legi_groups[int((val - vmin) / d)]
            for val in original_vals
        ]
        self.data.insert(0, f'{featr} groups', new_col)

    def update_prediction(self, predictions):
        self.data.insert(len(self.data.columns), 'prediction', predictions)
        binary_predictions = [
            1 if prediction > 0.5 else 0
            for prediction in self.data['prediction'].values
        ]
        self.data.insert(len(self.data.columns), 'binary prediction', binary_predictions)

    def get_ctgrs(self, featr):
        # print(ctgr)
        return [key for key in self.categorical_map[featr].keys()]

    def get_raw_data(self):
        if self.data is None:
            self.data = pandas.read_csv(self.file_path + '.csv').applymap(str)
            self.data[list(self.n_featrs)] = self.data[list(self.n_featrs)].applymap(float) # 数值型的数据转浮点
            self.data.insert(0, 'ID', [*range(len(self.data))])
            for n_featr in self.n_featrs:
                self.__group_n_featr(n_featr)
        return self.data

    def get_groups(self, featr):
        if featr in self.n_featrs:
            raise RuntimeError('数值类特征没有群体之分')
        else:
            return list(self.categorical_map[featr].keys())

    def get_processed_data(self):
        if self.__train_data is None:
            self.__process_train_data()
        return self.__train_data

    def __process_train_data(self):
        num_input = len(self.n_featrs) + sum(
            len(self.categorical_map[c_featr])
            for c_featr in self.c_featrs
        )
        # print(num_input)
        raw_data = self.get_raw_data()
        np_data = np.zeros((len(raw_data), num_input), np.float32)

        start = 0
        for featr in self.featrs:
            frame = raw_data[featr]
            if featr in self.n_featrs:
                # min-max normalization
                end = start + 1
                # print(frame.to_numpy().shape)
                np_data[:, start] = frame.to_numpy()
                min_val = np_data[:, start].min()
                max_val = np_data[:, start].max()
                np_data[:, start] -= min_val
                np_data[:, start] /= (max_val - min_val)
            else:
                # convert onehot
                featr_map = self.categorical_map[featr]
                end = start + len(featr_map)
                row_indexs = list(range(len(raw_data)))
                col_indexs = list(featr_map[val] + start for val in frame)
                np_data[row_indexs, col_indexs] = 1
            start = end

        labels = list(map(int, (self.label_map[val] for val in raw_data[self.label].values)))
        self.__train_data = (torch.tensor(np_data), torch.tensor(labels))

    def free(self):
        if self.temporal:
            os.remove(self.file_path + '.txt')
            os.remove(self.file_path + '.csv')


# german_newformet
data_model = DataModel('german_newformet', PRJROOT + 'data/german_newformet')


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
            return '数据文件的文件类型必须为.csv'

        tar_path = PRJROOT + 'data'
        if not keepfile:
            tar_path += '/temp'
        data_path = os.path.join(tar_path, secure_filename(data.filename))
        if keepfile and (os.path.isfile(data_path)):
            return '文件名冲突'
        data.save(data_path)

        name = data.filename[:-4]
        if name in self.datasets.keys():
            os.remove(data_path)
            return '上传失败：上传数据集与已有数据集重名'
        self.datasets[name] = data_path[:-4]
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
