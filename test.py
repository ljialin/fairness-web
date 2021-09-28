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


class MyObj:
    insts = {}

    def __init__(self, ip):
        self.ip = ip
        MyObj.insts[ip] = self


def prt(*args):
    for item in args:
        print(item)

if __name__ == '__main__':
    v = 0.00018432
    print(f'{v:.2}')
    # data = pandas.read_csv('data/german1.csv').applymap(str)
    # data
    # print(data)
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
