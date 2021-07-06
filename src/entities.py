"""
@DATE: 2021/6/03
@Author: Ziqi Wang
@File: entities.py
"""

import os
from root import PRJROOT


class DataViewer:
    def __init__(self, file_path):
        # file_path 为不带拓展名的绝对路径
        self.file_path = file_path
        self.attrs = []
        self.numberical_bounds = {}
        self.categorical_map = {}
        self.del_once_free = False
        self.label = ''
        self.label_map = {}

        self.errinfo = self.__load_descfile(file_path)
        print(self.errinfo)

    def __load_descfile(self, file_path):
        if file_path[len(PRJROOT)].find('data/temp', 0):
            # 若为temp文件夹里的文件，则对象释放同时删除对应文件
            self.del_once_free = True

        f = open(file_path + '.txt', 'r')       # 可认为不会出现文件不存在
        try:
            line = next(f).strip()
            while line != 'NUMBER OF ATTRIBUTES:':
                line = next(f).strip()
            n_attrs = int(next(f).strip())

            line = next(f).strip()
            while line != 'ATTRIBUTES:':
                line = next(f).strip()

            attr_id = 0
            line = next(f).strip()
            while line != 'LABEL:':
                if line == '':
                    continue
                fields = [*map(lambda x: x.strip(), line.split(','))]
                attr_name = fields[0]
                if len(fields) < 2:
                    return f'描述文件中属性{attr_name}没有指定类型'
                attr_type = fields[1]
                self.attrs.append(attr_name)

                if attr_type == 'numberical':
                    self.numberical_bounds[attr_name] = []
                    for bound in fields[2:]:
                        try:
                            lb, ub = map(lambda x: float(x.strip()), bound.split('~'))
                        except ValueError:
                            return f'属性{attr_name}的取值范围没有满足 %f ~ %f格式'
                        self.numberical_bounds[attr_name].append((lb, ub))
                elif attr_type == 'categorical':
                    if len(fields) < 4:
                        return f'描述文件中分类属性{attr_name}的取值类型不足两个'
                    self.categorical_map[attr_name] = {}
                    n = 0
                    for category in fields[2:]:
                        self.categorical_map[attr_name][category] = n
                        n += 1
                else:
                    return f'描述文件中属性{attr_name}的类型非法: {fields[1]}'
                attr_id += 1
                line = f.readline().strip()
            if n_attrs != attr_id:
                return '描述文件指定的属性数量与实际数量不符'

            fields = [*map(lambda x: x.strip(), f.readline().split(','))]
            if len(fields) != 3:
                return '描述文件中LABEL:之后一行必须为逗号分隔的三个字段'
            self.label = fields[0]
            self.label_map[fields[1]] = 0
            self.label_map[fields[2]] = 1
        except EOFError:
            return '描述文件信息缺失'
        f.close()
        return ''

    def generate_raw_data(self):
        """
            通过self.filepath + '.csv'获取数据文件的绝对路径。
            self.attrs (list<str>) 是该数据集所有属性名的列表（不含label）
            self.numberical_bounds (dict<str, list<tuple<float>>>) 所有numberical属性的取值范围，可以通过
                self.numberical_bounds.keys()获取所有numberical属性的属性名
            self.categorical_map (dict<str, dict<str, int>>)每个categorical属性的 取值类型与对应整数标签的字典，例如
                {‘status’: {'A11': 0, 'A12': 1, 'A13': 2}}
            self.label (str) 标签属性的名称
            self.lable_map (dict<str, int>) 类别到整数标签的字典，如{'True': 0, 'False': 1}
        :return: DataFrame？
        """
        pass

    def generate_train_data(self):
        pass

    def __del__(self):
        # 析构函数，当此前DataViewer的目标是一个临时文件，则对象释放的同时删除临时文件
        if self.del_once_free:
            os.remove(self.file_path + '.txt')
            os.remove(self.file_path + '.csv')

if __name__ == '__main__':
    dataviewer = DataViewer(PRJROOT + 'data/german0')
    print(dataviewer.attrs)
    print(dataviewer.numberical_bounds)
    print(dataviewer.categorical_map)
    print(dataviewer.label)
    print(dataviewer.label_map)
