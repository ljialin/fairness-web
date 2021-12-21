"""
  @DATE: 2021/8/28
  @Author: Ziqi Wang
  @File: data_eval.py
"""

import random
import time

import pandas as pds
import pyecharts.options as chart_opts
from typing import List
from src.common_fair_analyze import THRESHOLDS, N_SPLIT, CgfAnalyzeRes
from pyecharts.charts import Bar
from src.utils import get_count_from_series


class DataEvaluator:
    # plr: positive label rate
    theta_gf = 0.8

    def __init__(self, data_model):
        # self.data = data_model.get_raw_data() #gsh commit
        # self.processed_data = data_model.get_processed_data()
        self.data = data_model.data4eval
        self.label = data_model.label
        self.pos_label_val = data_model.pos_label_val
        self.neg_label_val = data_model.neg_label_val
        self.featrs = data_model.featrs
        self.c_featrs = data_model.c_featrs
        self.n_featrs = data_model.n_featrs
        self.categorical_map = data_model.categorical_map
        self.size = len(self.data)

        self.n_featr_span = {}
        for featr in self.n_featrs:
            self.n_featr_span[featr] = self.data[featr].max() - self.data[featr].min()

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
        key = f'{featr} groups' if featr in self.n_featrs else featr
        group_sp_rates = self.get_group_plr(key) / self.get_global_plr()
        for group in group_sp_rates.index:
            if group_sp_rates[group] < self.theta_gf:
                cmmts.append(
                    f'{group}群体的正面标签率（标签值为{self.pos_label_val}的频率）'
                    f'与总体正面标签率的比值过低。建议检查该群体是否收到歧视\\偏爱，'
                    f'以及该特征是否为应当影响标签值的非敏感特征。'
                )
            elif group_sp_rates[group] > 1 / self.theta_gf:
                cmmts.append(
                    f'{group}群体的正面标签率（标签值为{self.pos_label_val}的频率）'
                    f'与总体正面标签率的比值过高，建议检查该群体是否受到歧视\\偏爱，'
                    f'以及该特征是否为应当影响标签值的非敏感特征。'
                )
        if not cmmts:
            cmmts.append(f'对{featr}特征进行群体公平分析，未发现公平性问题')
        return group_sp_rates, cmmts

    def analyze_cgf(self, sens_featr, legi_featr):
        res = CgfAnalyzeRes(sens_featr, legi_featr)
        data = self.data
        legi_key = f'{legi_featr} groups' if legi_featr in self.n_featrs else legi_featr
        sens_key = f'{sens_featr} groups' if sens_featr in self.n_featrs else sens_featr

        counts = data[[legi_key, sens_key, self.label]].value_counts()
        res.legi_groups = list(data[legi_key].unique())
        res.sens_groups = list(data[sens_key].unique())
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
                        f'{sens_featr}为{sens_grp}的群体在{legi_featr}为{legi_grp}的部分{self.label}'
                        f'为{self.pos_label_val}的比例过低，可能受到了歧视\\偏爱'
                    )
                elif ratio > 1 / THRESHOLDS['PLR']:
                    res.cmmts.append(
                        f'{sens_featr}为{sens_grp}的群体在{legi_featr}为{legi_grp}的部分{self.label}为{self.pos_label_val}'
                        f'的比例过高，可能受到了歧视\\偏爱'
                    )
            if not res.cmmts:
                res.cmmts.append(f'以{legi_featr}为正当特征对{sens_featr}进行分析，未发现公平性问题')
        return res

    def analyze_if(self, legi_featr):
        if legi_featr in self.n_featrs:
            neg_discriminated, pos_discriminated = self.__analyze_if_numberical(legi_featr)
        else:
            neg_discriminated, pos_discriminated = self.__analyze_if_categorical(legi_featr)
        cmmts = []
        if neg_discriminated:
            cmmts.append(f'以{legi_featr}作为正当特征分析，数据集中以下{len(neg_discriminated)}个个体可能受到了歧视\\偏爱：')
            cmmts.append(', '.join(map(str, neg_discriminated)))
        if pos_discriminated:
            cmmts.append(f'以{legi_featr}作为正当特征分析，数据集中以下{len(pos_discriminated)}个个体可能受到了歧视\\偏爱：')
            cmmts.append(', '.join(map(str, pos_discriminated)))
            # cmmts.append(
            #     f'以{legi_featr}作为正当特征分析，数据集中以下{len(pos_discriminated)}个个体可能'
            #     f'受到了偏爱：\n' + ', '.join(map(str, pos_discriminated))
            # )
        if not cmmts:
            cmmts.append('当前数据集中未发现个体公平问题。')
        return cmmts

    def __analyze_if_categorical(self, legi_featr):
        neg_discrminated = []
        pos_discrminated = []
        all_counts = self.data[legi_featr].value_counts()
        pos_counts = self.data[self.data[self.label] == self.pos_label_val][legi_featr].value_counts()
        pos_ratios = pos_counts / all_counts
        for grp in pos_ratios.index:
            group_frame = self.data[self.data[legi_featr] == grp]
            if pos_ratios[grp] > 0.9:
                neg_discrminated += [
                    i for i in group_frame
                    [self.data[self.label] == self.neg_label_val]
                    ['ID']
                ]
            elif pos_ratios[grp] < 0.1:
                pos_discrminated += [
                    i for i in group_frame
                    [self.data[self.label] == self.pos_label_val]
                    ['ID']
                ]
        return neg_discrminated, pos_discrminated

    def __analyze_if_numberical(self, legi_featr):
        start_time = time.time()
        neg_discriminated, pos_discriminated = [], []
        frame = self.data[[legi_featr, self.label, 'ID']].sort_values(legi_featr)
        i = 0
        p, q = 0, 0
        p_cnt, n_cnt = 0, 0
        span = 0.05 * (frame[legi_featr].max() - frame[legi_featr].min())

        val_arr = frame[legi_featr].to_numpy()
        label_arr = frame[self.label].to_numpy()
        id_arr = frame['ID'].to_numpy()
        while i < len(self.data):
            cur_val = val_arr[i]
            cur_label = label_arr[i]
            while val_arr[p] < cur_val - span:
                if label_arr[p] == self.pos_label_val:
                    p_cnt -= 1
                else:
                    n_cnt -= 1
                p += 1
            while q < len(frame) and val_arr[q] <= cur_val + span:
                if label_arr[q] == self.pos_label_val:
                    p_cnt += 1
                else:
                    n_cnt += 1
                q += 1
            n = q - p - 1
            if n > 10:
                p_cnt_expect_self = p_cnt if cur_label == self.neg_label_val else p_cnt - 1
                p_rate = p_cnt_expect_self / n
                if p_rate > 0.9 and cur_label == self.neg_label_val:
                    neg_discriminated.append(id_arr[i])
                elif p_rate < 0.1 and cur_label == self.pos_label_val:
                    pos_discriminated.append(id_arr[i])
            i += 1
        print(f'{time.time() - start_time:.3f}s')
        return neg_discriminated, pos_discriminated


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
            # print(group_sp_rates.values)
            chart = (
                Bar()
                .set_global_opts(
                    title_opts=chart_opts.TitleOpts(title=f'对{featr}属性的分析结果'),
                    xaxis_opts=chart_opts.AxisOpts(
                        name=f'基于{featr}划分的群组', name_location='middle',
                        name_gap=25
                    ),
                    yaxis_opts=chart_opts.AxisOpts(
                        name='\n'.join('群组正标签率与总体正标签率之比'), name_location='center',
                        name_gap=25, name_rotate=0
                    ),
                )
                .add_xaxis(list(group_sp_rates.index))
                .add_yaxis(
                    '', list(map(float, group_sp_rates.values)),
                    label_opts=chart_opts.LabelOpts(is_show=False)
                )
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

    def update_if_res(self, model, legi_featr):
        self.legi_featr = legi_featr
        self.if_cmmts = model.analyze_if(legi_featr)


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

    def if_eval(self, legi_featr):
        if not legi_featr:
            return '必须选择至少一个正当属性才能进行分析'
        self.view.update_if_res(self.model, legi_featr)
