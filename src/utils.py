"""
@DATE: 2021/6/24
@Author: Ziqi Wang
@File: utils.py
"""

import os
# import math
import pickle
from root import PRJROOT


def get_rgb_hex(*rgb):
    assert len(rgb) == 3 and max(rgb) <= 255 and min(rgb) >= 0
    res = ''
    for item in rgb:
        temp = hex(item)
        if len(temp) == 4:
            res += temp[2:]
        else:
            res += f'0{temp[2]}'
    return res

def get_count_from_series(data, key):
    try:
        return data[key]
    except KeyError:
        return 0

def auto_dire(name, path=None, fmtr=''):
    dire_id = 0
    prefix = PRJROOT if path is None else f'{PRJROOT}{path}/'
    tar = f'{prefix}'
    while os.path.exists(tar):
        suffix = fmtr % dire_id if fmtr else dire_id
        tar = f'{prefix}{name}{suffix}'
        dire_id += 1
    os.makedirs(tar)
    print(tar)
    return tar


class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(__name__, name)
        except AttributeError:
            return super().find_class(module, name)


class FileUtils:
    @staticmethod
    def list_files(path):
        files = []
        for item in os.listdir(path):
            filepath = os.path.join(path, item)
            if os.path.isfile(filepath):
                files.append(os.path.abspath(filepath))
        return files

if __name__ == '__main__':
    pass
    # print(str_upright('abc'))
    # print(FileUtils.list_files('F:\\research group\Huawei Project\\fariness-web\data'))
    # print(FileUtils.list_files('../data'))
