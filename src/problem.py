"""
@DATE: 2021/6/25
@Author: Ziqi Wang
@File: problem.py
"""

class Problem:
    def __init__(self, data_viewer):
        self.data_viewer = data_viewer
        self.raw_data = data_viewer.__raw_data
        self.train_data = data_viewer.get_train_data()

