"""
@DATE: 2021/6/24
@Author: Ziqi Wang
@File: utils.py
"""
import pickle
import os


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
    print(FileUtils.list_files('F:\\research group\Huawei Project\\fariness-web\data'))
    print(FileUtils.list_files('../data'))
