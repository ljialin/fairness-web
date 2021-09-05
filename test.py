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


class MyObj:
    insts = {}

    def __init__(self, ip):
        self.ip = ip
        MyObj.insts[ip] = self


def prt(*args):
    for item in args:
        print(item)

if __name__ == '__main__':
    frame = pandas.read_csv('./data/german1.csv').applymap(str)
    subframe = frame[frame['status'] == 'A12']
    print(subframe.size)
    print(subframe.shape[1])
    print(subframe.columns)
    # a = (1, 2, 3)
    # print(*a)
    # mydict = {'a': 1, 'b': 2}
    # print(list(mydict.keys()))
    pass
