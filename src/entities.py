# """
# @DATE: 2021/6/03
# @Author: Ziqi Wang
# @File: entities.py
# """
#
# import os
#
# import pandas
#
# from root import PRJROOT
#
#
# class DataViewer:
#     def __init__(self, file_path):
#         # file_path 为不带拓展名的绝对路径
#         self.file_path = file_path
#         self.attrs = []
#         self.n_attrs = set()
#         self.c_attrs = set()
#         self.numberical_bounds = {}
#         self.categorical_map = {}
#         self.del_once_free = False
#         self.label = ''
#         self.positive_label = ''
#         self.label_map = {}
#
#         self.errinfo = self.__load_file(file_path)
#         # print(self.errinfo)
#
#     def __load_file(self, file_path):
#         self.raw_data = pandas.read_csv(file_path + '.csv').applymap(str)
#         if file_path[len(PRJROOT)].find('data/temp', 0):
#             # 若为temp文件夹里的文件，则对象释放同时删除对应文件
#             self.del_once_free = True
#
#         f = open(file_path + '.txt', 'r')       # 可认为不会出现文件不存在
#         try:
#             line = next(f).strip()
#             while line != 'NUMBER OF ATTRIBUTES:':
#                 line = next(f).strip()
#             n_attrs = int(next(f).strip())
#
#             line = next(f).strip()
#             while line != 'ATTRIBUTES:':
#                 line = next(f).strip()
#
#             attr_id = 0
#             line = next(f).strip()
#             while line != 'LABEL:':
#                 if line == '':
#                     continue
#                 fields = [*map(lambda x: x.strip(), line.split(','))]
#                 attr_name = fields[0]
#                 if len(fields) < 2:
#                     return f'描述文件中属性{attr_name}没有指定类型'
#                 attr_type = fields[1]
#                 self.attrs.append(attr_name)
#
#                 if attr_type == 'numberical':
#                     self.numberical_bounds[attr_name] = []
#                     self.n_attrs.add(attr_name)
#                     for bound in fields[2:]:
#                         try:
#                             lb, ub = map(lambda x: float(x.strip()), bound.split('~'))
#                         except ValueError:
#                             return f'属性{attr_name}的取值范围没有满足 %f ~ %f格式'
#                         self.numberical_bounds[attr_name].append((lb, ub))
#                 elif attr_type == 'categorical':
#                     if len(fields) < 4:
#                         return f'描述文件中分类属性{attr_name}的取值类型不足两个'
#                     self.c_attrs.add(attr_name)
#                     self.categorical_map[attr_name] = {}
#                     n = 0
#                     for category in fields[2:]:
#                         self.categorical_map[attr_name][category] = n
#                         n += 1
#                 else:
#                     return f'描述文件中属性{attr_name}的类型非法: {fields[1]}'
#                 attr_id += 1
#                 line = f.readline().strip()
#             if n_attrs != attr_id:
#                 return '描述文件指定的属性数量与实际数量不符'
#
#             fields = [*map(lambda x: x.strip(), f.readline().split(','))]
#             if len(fields) != 3:
#                 return '描述文件中LABEL:之后一行必须为逗号分隔的三个字段'
#             self.label = fields[0]
#             self.positive_label = fields[1]
#             self.label_map[fields[1]] = 1
#             self.label_map[fields[2]] = 0
#         except EOFError:
#             return '描述文件信息缺失'
#         f.close()
#         return ''
#
#     def get_ctgrs(self, ctgr):
#         # print(ctgr)
#         return [key for key in self.categorical_map[ctgr].keys()]
#
#     def ger_train_data(self):
#         pass
#
#     # def __del__(self):
#     #     # 析构函数，当此前DataViewer的目标是一个临时文件，则对象释放的同时删除临时文件
#     #     if self.del_once_free:
#     #         os.remove(self.file_path + '.txt')
#     #         os.remove(self.file_path + '.csv')
#
#
# if __name__ == '__main__':
#     data_viewer = DataViewer(PRJROOT + 'data/german1')
#     print(data_viewer.attrs)
#     print(data_viewer.numberical_bounds)
#     print(data_viewer.categorical_map)
#     print(data_viewer.label)
#     print(data_viewer.label_map)
