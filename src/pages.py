"""
@DATE: 2021/6/23
@Author: Ziqi Wang
@File: app.py
"""
import os
from root import PRJROOT
from utils import FileUtils
from entities import DataViewer
from process import ProcessBuilder
from werkzeug.utils import secure_filename

class DataPage:
    def __init__(self):
        self.datasets = {}          # 数据集名称，对应的文件绝对路径（不带拓展名）
        # self.numberical_attrs = None
        # self.categorical_attrs = None
        # self.label_attr = None
        self.data_viewer = None
        self.status = 'new'

        self.__scan_datasets()
        ProcessBuilder.new()

    def upload_dataset(self, desc, data, keepfile=False):
        if desc.filename[-4:] != '.txt':
            return '描述文件的文件类型必须为.txt'
        if data.filename[-4:] != '.csv':
            return '数据文件的文件类型必须为.csv'
        if desc.filename[:-4] != data.filename[:-4]:
            return '描述文件与数据文件的文件名必须一致'

        tar_path = PRJROOT + 'data'
        if not keepfile:
            tar_path += '/temp'
        desc_path = os.path.join(tar_path, secure_filename(desc.filename))
        data_path = os.path.join(tar_path, secure_filename(data.filename))
        if os.path.isfile(desc_path) or os.path.isfile(data_path):
            return '文件名冲突'
        desc.save(desc_path)
        data.save(data_path)

        with open(desc_path, 'r') as f:
            next(f)
            name = f.readline().strip()
            if name in self.datasets.keys():
                os.remove(desc_path)
                os.remove(data_path)
                return '上传数据集与已有数据集重名'
            self.datasets[name] = desc_path[:-4]
        return ''

    def select_dataset(self, name):
        file_path = self.datasets[name]
        self.data_viewer = DataViewer(file_path)
        ProcessBuilder.assign_data(self.data_viewer)
        return self.data_viewer.errinfo

    def c_attr_texts(self):
        # 获取html中需要的每个categorical attributes对应的文本
        texts = []
        for attr in self.data_viewer.categorical_map.keys():
            items = [attr + ': ']
            for value in self.data_viewer.categorical_map[attr].keys():
                items.append(value)
                items.append(', ')
            texts.append(''.join(items[:-1]))
        return texts

    def n_attr_texts(self):
        # 获取html中需要的每个numberical attributes对应的文本
        texts = []
        for attr in self.data_viewer.numberical_bounds.keys():
            items = [attr]
            end = 1
            if self.data_viewer.numberical_bounds[attr]:
                end = -1
                items.append(': ')
            for bound in self.data_viewer.numberical_bounds[attr]:
                items.append('%f ~ %f' % bound)
                items.append(', ')
            texts.append(''.join(items[:end]))
        return texts

    def label_text(self):
        # 获取html中需要的每个label对应的文本
        items = [self.data_viewer.label, *self.data_viewer.label_map.keys()]
        return '%s: %s, %s' % tuple(items)

    def __scan_datasets(self):
        csv_set = set()
        txt_set = set()
        valid_list = []
        for file_abspath in FileUtils.list_files(PRJROOT + 'data/'):
            file_name = file_abspath[:-4]
            if file_abspath[-4:] == '.txt':
                if file_name in csv_set:
                    valid_list.append(file_name)
                else:
                    txt_set.add(file_name)
            elif file_abspath[-4:] == '.csv':
                if file_name in txt_set:
                    valid_list.append(file_name)
                else:
                    csv_set.add(file_name)
        valid_list.sort()
        for file_name in valid_list:
            with open(file_name + '.txt', 'r') as f:
                next(f)
                self.datasets[f.readline().strip()] = file_name


class UserPageMgr:
    __data_page = None

    @staticmethod
    def get():
        return UserPageMgr.__data_page

    @staticmethod
    def new():
        UserPageMgr.__data_page = DataPage()
        return UserPageMgr.__data_page

    @staticmethod
    def free():
        UserPageMgr.__data_page = None
