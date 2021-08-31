"""
@DATE: 2021/6/24
@Author: Ziqi Wang
@File: test.py
"""

import glob
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


if __name__ == '__main__':
    img = Image.open('./static/assets/model_py_example.png')
    img.resize((649, 393)).save('./static/assets/model_py_example1.png')
    # # 群体录取率
    # featr = 'status'
    # label = 'credit'
    # data = pandas.read_csv('./data/german1.csv').applymap(str)
    # frame = data[[featr, label]]
    # total_cnts = frame[featr].value_counts()
    # positive_cnts = (
    #     frame
    #     [frame[label] == '1']
    #     [featr].value_counts()
    # )
    # rates = positive_cnts / total_cnts
    # print(type(rates), rates)
    # 总体录取率
    gplr = len(data[frame[label] == '1']) / len(data)
    # print(gplr)

    # for group in rates.index:
    #     print(group, rates[group])
    sp_rate = rates / gplr
    print(list(sp_rate.index), sp_rate.values)
    # print(rates.index)
    pass
