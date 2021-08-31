"""
  @DATE: 2021/8/28
  @Author: Ziqi Wang
  @File: data_eval.py
"""
import random
from typing import List

import pandas
from pyecharts.charts import Bar
import pyecharts.options as chart_opts


class DataEvalModel:
    # plr: positive label rate
    theta_gf = 0.8

    def __init__(self, data_model):
        self.data = data_model.get_raw_data()
        self.label = data_model.label
        self.pos_label_val = data_model.pos_label_val

    def get_global_plr(self):
        print(self.pos_label_val)
        pl_cnt = len(self.data[self.data[self.label] == self.pos_label_val])
        return pl_cnt / len(self.data)

    def get_group_plr(self, featr) -> pandas.Series:
        frame = self.data[[featr, self.label]]
        total_cnts = frame[featr].value_counts()
        positive_cnts = (frame
            [frame[self.label] == self.pos_label_val]
            [featr].value_counts()
        )
        print(total_cnts, positive_cnts)
        return positive_cnts / total_cnts

    def analyze_gf(self, featr) -> (pandas.Series, List[str]):
        cmmts = []
        group_sp_rates = self.get_group_plr(featr) / self.get_global_plr()
        for group in group_sp_rates.index:
            if group_sp_rates[group] < self.theta_gf:
                cmmts.append(
                    f'{group}群体的正面标签率（标签值为{self.pos_label_val}的频率）'
                    f'与总体正面标签率的比值过低。建议检查该群体是否收到歧视，'
                    f'以及该特征是否为应当影响标签值的非敏感特征。'
                )
            elif group_sp_rates[group] > 1 / self.theta_gf:
                cmmts.append(
                    f'{group}群体的正面标签率（标签值为{self.pos_label_val}的频率）'
                    f'与总体正面标签率的比值过高，建议检查该群体是否受到优待，'
                    f'以及该特征是否为应当影响标签值的非敏感特征。'
                )
        if not cmmts:
            cmmts.append(f'在{featr}特征上未发现公平性问题')
        return group_sp_rates, cmmts


class DataEvalView:
    def __init__(self, data_model):
        self.name = data_model.name
        # self.data = data_model.raw_data
        self.featrs = data_model.featrs
        self.n_featrs = data_model.n_featrs
        self.c_featrs = data_model.c_featrs

        self.sens_featrs = []
        self.legi_featr = None

        self.gf_cmmts = []
        self.if_blocks = []
        self.cgf_blocks = []

    def update_gf_res(self, model, sens_featrs):
        self.sens_featrs = sens_featrs
        self.gf_cmmts.clear()

        charts = []
        for i, featr in enumerate(sens_featrs):
            group_sp_rates, cmmts = model.analyze_gf(featr)
            print(group_sp_rates.values)
            chart = (
                Bar()
                .add_xaxis(list(group_sp_rates.index))
                .add_yaxis(
                    '', list(map(float, group_sp_rates.values)),
                    label_opts=chart_opts.LabelOpts(is_show=False)
                )
                .set_global_opts(title_opts=chart_opts.TitleOpts(title=featr))
            )
            charts.append(chart)
            self.gf_cmmts.append((i, cmmts))
        return charts

    def update_cgf_res(self, model, sens_featrs, legi_featr):
        self.sens_featrs = sens_featrs
        self.legi_featr = legi_featr
        data = model.data
        pass

    def update_if_res(self, model, sens_featrs):
        self.sens_featrs = sens_featrs
        data = model.data
        pass

    # def get_gp_bar(self, data, featr):
    #     pass
    #
    # def get_cgp_bar(self, data, featr):
    #     pass


class DataEvalController:
    insts = {}

    def __init__(self, ip, data_model):
        self.model = DataEvalModel(data_model)
        self.view = DataEvalView(data_model)
        self.charts = {}
        DataEvalController.insts[ip] = self
        pass

    def gf_eval(self, sens_featrs):
        if not sens_featrs:
            return '必须选择至少一个敏感属性才能进行分析'
        charts = self.view.update_gf_res(self.model, sens_featrs)
        for i, chart in enumerate(charts):
            self.charts[f'0{i}'] = chart
        # self.charts['gf'] = [item[1] for item in charts_with_id]
        pass

    def cgf_eval(self, sens_featrs, legi_featr):
        if not sens_featrs or not legi_featr:
            return '必须选择至少一个敏感属性和一个正当属性才能进行分析'
        if set(sens_featrs) & set(legi_featr):
            return '正当特征必须是非敏感特征'
        self.view.update_cgf_res(self.model, sens_featrs, legi_featr)
        pass

    def if_eval(self, sens_featrs):
        if not sens_featrs:
            return '必须选择至少一个敏感属性才能进行分析'
        self.view.update_if_res(self.model, sens_featrs)
        pass