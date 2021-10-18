"""
@DATE: 2021/6/24
@Author: Ziqi Wang
@File: test.py
"""

import glob

import numpy as np
from pyecharts.charts import Bar
from pandas import DataFrame
from PIL import Image
import pandas
from werkzeug.datastructures import ImmutableMultiDict

from root import PRJROOT
import socket
import time

class MyObj:
    insts = {}

    def __init__(self, ip):
        self.ip = ip
        MyObj.insts[ip] = self


def prt(*args):
    for item in args:
        print(item)

if __name__ == '__main__':
    # arr = [1, -2]
    # print(', '.join(map(str, arr)))
    data = pandas.read_csv('data/N-IF-100000.csv').applymap(str)
    data = data.sort_values('grade')[['grade', 'acceptance']]

    start = time.perf_counter_ns()
    for i in range(100000):
        tmp = data.iloc[i]['grade']
        # print(data.iloc[i]['age'])
    print((time.perf_counter_ns() - start) // 1000000)

    start = time.perf_counter_ns()
    for i in range(100000):
        tmp = data.iat[i, 0]
    print((time.perf_counter_ns() - start) // 1000000)
    # print(data)
    # print(data[['age', 'credit']])
    # for i, item in data.iterrows():
    #     print(i, item['age'])
    # for i in range(1000):
    #     print(data[['age', 'credit']].loc[i])
    # all_counts = data['status'].value_counts()
    # pos_counts = data[data['credit'] == '1']['status'].value_counts()
    # print(pos_counts)
    # print(all_counts)
    # pos_ratios = pos_counts / all_counts
    # print(pos_ratios.index)
    # for item in pos_ratios:
    #     print(item)
    # print()
    # data
    # print(data['credit'].value_counts())
    # data['age'] = data[['age']].applymap(float)
    # print(data['age'])
    # tmp = data['age'].apply(float)
    # tmp = data['age']
    # tmp.apply(float)
    # print(tmp)
    # print(data['status'])
    # frame = data[['status', 'age']]

    # counts = frame.value_counts()
    # print(counts[('A11', '45')])
    # print(type(counts))
    # print(counts.index)
    # for status, age in counts.index:
    #     print(status, age)
    # print(frame)
    # print(frame.value_counts())
    # print(socket.gethostbyname(socket.gethostname()))
    pass
