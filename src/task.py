# """
# @DATE: 2021/6/24
# @Author: Ziqi Wang
# @File: process.py
# """
# # from entities import DataViewer
# from src.problem import Problem
# from src.algorithm import Algorithm
#
#
# class Task:
#     # 不要重写 __getattr__ 函数!
#     count = 0
#
#     def __init__(self):
#         self.id = Task.count
#         Task.count += 1
#         self.name = 'Task-%04d' % self.id
#         self.ready = False
#         self.finished = False
#         self.problem = None
#         self.algorithm = Algorithm()
#
#     def get_name(self):
#         return self.name
#
#     def get_strid(self):
#         return '%04d' % self.id
#
#     def __str__(self):
#         return 'Task-%04d' % self.id
#
#
# class TaskBuilder:
#     __building_process = None
#
#     @staticmethod
#     def new():
#         if TaskBuilder.__building_process is None:
#             TaskBuilder.__building_process = Task()
#         else:
#             raise RuntimeError('Concurrency unsupport yet')
#
#     @staticmethod
#     def assign_data(data_viewer):
#         TaskBuilder.__building_process.problem = Problem(data_viewer)
#
#     @staticmethod
#     def take_out():
#         process = TaskBuilder.__building_process
#         del TaskBuilder.__building_process       # 删除引用后重新赋值，否则process也会变成None
#         TaskBuilder.__building_process = None
#         return process
