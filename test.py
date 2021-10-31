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
    a = 13
    s = '0023'
    print(int(s) + a)
    # arr = [1, -2]
    # print(', '.join(map(str, arr)))
    # data = pandas.read_csv('data/N-IF-100000.csv').applymap(str)
    # data[['grade']] = data[['grade']].applymap(float)
    # data = data.sort_values('grade')[['grade', 'acceptance']]
    #
    # start = time.time()
    # for i in range(10):
    #     # tmp = data.iloc[i]['grade']
    #     print(data.iloc[i]['grade'], end=' ')
    # print((time.time() - start))
    #
    # start = time.time()
    # for i in range(10):
    #     # tmp = data.iat[i, 0]
    #     print(data.iat[i, 0], end=' ')
    # print((time.time() - start))
    #
    # start = time.time()
    # grades = data['grade'].to_numpy()
    # for i in range(10):
    #     print(grades[i], end=' ')
    # print((time.time() - start))
    pass
