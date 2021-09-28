"""
  @DATE: 2021/8/28
  @Author: Ziqi Wang
  @File: data_eval.py
"""
import random
from typing import List
from fair_analyze import THRESHOLDS, N_SPLIT, CgfAnalyzeRes
import pandas as pds
from pyecharts.charts import Bar
import pyecharts.options as chart_opts

from utils import get_count_from_series


class DataEvaluator:
    # plr: positive label rate
    theta_gf = 0.8

    def __init__(self, data_model):
        self.data = data_model.get_raw_data()
        self.label = data_model.label
        self.pos_label_val = data_model.pos_label_val
        self.neg_label_val = data_model.neg_label_val
        self.n_featrs = data_model.n_featrs

    def get_global_plr(self):
        data = self.data
        label, pos_val = self.label, self.pos_label_val
        pl_cnt = len(data[data[label] == pos_val])
        return pl_cnt / len(data)

    def get_group_plr(self, featr) -> pds.Series:
        data = self.data
        label, pos_val = self.label, self.pos_label_val
        frame = data[[featr, label]]
        total_cnts = frame[featr].value_counts()
        positive_cnts = (frame
            [frame[label] == pos_val]
            [featr].value_counts()
        )
        return positive_cnts / total_cnts

    def analyze_gf(self, featr) -> (pds.Series, List[str]):
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

    def analyze_cgf(self, sens_featr, legi_featr):
        res = CgfAnalyzeRes(sens_featr)
        data = self.data
        res.sens_groups = list(data[sens_featr].unique())
        if legi_featr in self.n_featrs:
            col_name = f'{legi_featr} groups'
            if col_name not in data.columns:
                vmax = data[legi_featr].max()
                vmin = data[legi_featr].min()
                print(type(vmin), vmin, type(vmax), vmax)
                d = (vmax - vmin + 1e-5) / N_SPLIT
                legi_groups = [
                    f'{vmin + i * d:.2}-{vmin + (i + 1) * d:.2}'
                    for i in range(N_SPLIT)
                ]
                res.legi_groups = legi_groups
                original_vals = data[legi_featr].values
                new_col = [
                    legi_groups[int((val - vmin) / d)]
                    for val in original_vals
                ]
                data.insert(0, col_name, new_col)
            counts = data[[col_name, sens_featr, self.label]].value_counts()
        else:
            res.legi_groups = list(data[legi_featr].unique())
            counts = data[[legi_featr, sens_featr, self.label]].value_counts()
        pval = self.pos_label_val
        nval = self.neg_label_val
        for legi_grp in res.legi_groups:
            p_tcnt = sum(
                get_count_from_series(counts, (legi_grp, sens_grp, pval))
                for sens_grp in res.sens_groups
            )
            n_tcnt = sum(
                get_count_from_series(counts, (legi_grp, sens_grp, nval))
                for sens_grp in res.sens_groups
            )
            total_plr = p_tcnt / (p_tcnt + n_tcnt + 1e-5)
            res.data[legi_grp] = {}
            for sens_grp in res.sens_groups:
                p_cnt = get_count_from_series(counts, (legi_grp, sens_grp, pval))
                n_cnt = get_count_from_series(counts, (legi_grp, sens_grp, nval))
                plr = p_cnt / (p_cnt + n_cnt + 1e-5)
                ratio = plr / total_plr
                res.data[legi_grp][sens_grp] = ratio
                if ratio < THRESHOLDS['PLR']:
                    res.cmmts.append(
                        f'{sens_grp}群体在{legi_featr}为{legi_grp}的部分可能受到了歧视'
                    )
                elif ratio > 1 / THRESHOLDS['PLR']:
                    res.cmmts.append(
                        f'{sens_grp}群体在{legi_featr}为{legi_grp}的部分可能受到了偏爱'
                    )
        return res


class DataEvalView:
    def __init__(self, data_model):
        self.name = data_model.name
        self.featrs = data_model.featrs
        self.n_featrs = data_model.n_featrs
        self.c_featrs = data_model.c_featrs

        self.sens_featrs = []
        self.legi_featr = None

        self.gf_cmmts = []
        self.cgf_cmmts = []
        self.if_cmmts = []

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
        self.cgf_cmmts.clear()

        charts = []
        for i, sens_featr in enumerate(sens_featrs):
            res = model.analyze_cgf(sens_featr, legi_featr)
            self.cgf_cmmts.append((i, res.cmmts))
            charts.append(res.get_chart())
        return charts

    def update_if_res(self, model, sens_featrs):
        self.sens_featrs = sens_featrs
        pass


class DataEvalController:
    insts = {}

    def __init__(self, ip, data_model):
        # self.raw_data = data_model.get_raw_data()
        self.model = DataEvaluator(data_model)
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

    def cgf_eval(self, sens_featrs, legi_featr):
        if not sens_featrs or not legi_featr:
            return '必须选择至少一个敏感属性和一个正当属性才能进行分析'
        if legi_featr in sens_featrs:
            return '正当特征必须是非敏感特征'
        charts = self.view.update_cgf_res(self.model, sens_featrs, legi_featr)
        for i, chart in enumerate(charts):
            self.charts[f'1{i}'] = chart

    def if_eval(self, sens_featrs):
        if not sens_featrs:
            return '必须选择至少一个敏感属性才能进行分析'
        self.view.update_if_res(self.model, sens_featrs)
