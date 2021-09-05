"""
  @Time : 2021/8/30 15:53 
  @Author : Ziqi Wang
  @File : model_eval.py 
"""
import importlib
import inspect
import os
import numpy as np
import pandas as pds
import torch
# from typing import List
import pyecharts.options as chopts
from pyecharts.charts import Radar
from werkzeug.utils import secure_filename
from src.mvc.data_eval import DataEvaluator
from src.mvc.data import DataController
from src.utils import get_rgb_hex
from root import PRJROOT


METRICS = ('Acc', 'PLR', 'PPV', 'FPR', 'FNR', 'NPV')
METRIC_UBS = {'Acc': 1, 'PLR': 1, 'PPV': 1, 'FPR': 1, 'FNR': 1, 'NPV': 1}
THRESHOLDS = {'Acc': 0.8, 'PLR': 0.8, 'PPV': 0.8, 'FPR': 0.8, 'FNR': 0.8, 'NPV': 0.8}
PREFER_HIGH = {'Acc': 1, 'PLR': 1, 'PPV': 1, 'FPR': 1, 'FNR': 0, 'NPV': 0}
NUM_METRICS = len(METRICS)
assert set(METRICS) == set(METRIC_UBS.keys())
# PLR: Positive Label Rate. Statistical Parity requeirs PLR of each group close to global PLR.


def upload_model(ip, struct_file, var_file):
    if struct_file.filename[-3:] != '.py':
        raise RuntimeError('模型结构定义文件必须是.py后缀')
    if var_file.filename[-4:] not in {'.pth', '.pkl'}:
        raise RuntimeError('模型参数文件必须是.pth或.pkl后缀')

    prefix = ip.replace('.', '_') + '__'
    struct_path = os.path.join(
        PRJROOT + 'models/temp',
        prefix + secure_filename(struct_file.filename)
    )
    struct_file.save(struct_path)
    var_path = os.path.join(
        PRJROOT + 'models/temp',
        prefix + secure_filename(var_file.filename)
    )
    var_file.save(var_path)

    module = importlib.import_module(
        'models.temp.' + secure_filename(prefix + struct_file.filename)[:-3]
    )
    funcs = inspect.getmembers(module, inspect.isfunction)

    if len(funcs) != 1:
        raise RuntimeError('模型的类定义文件必须只含一个函数')

    func = funcs[0][1]
    model = func()
    model.load_state_dict(torch.load(var_path))

    # model = torch.load(path)
    # os.remove(struct_path)
    # os.remove(var_path)
    return var_file.filename[:-4], model


class Predictor:
    def __init__(self, model):
        self.model = model

    def predict(self, data: torch.Tensor, batch: int=256, to_binary=False) -> np.ndarray:
        res = []
        start = 0
        while start < len(data):
            x, _ = data[start: min(len(data), start+batch)]
            predictions = self.model(x)
            res += predictions.tolist()
            start += batch
        res = np.array(res)
        if to_binary:
            res = np.where(res > 0.5, 1, 0)
        return res


class ModelEvaluator:
    fairness_cmmt_fmt = (
        '根据%s指标，%s群体可能受到了%s，建议考虑将%s指标作为算法优化的目标之一对该模型进行优化，'
        '或考察%s特征是否对%s有正当影响。若有，建议将该特征从敏感特征列表中去除。'
    )

    def __init__(self, predictor, data_model, data_evaltr):
        self.data_evaltr = data_evaltr
        self.predictor = predictor

        self.label = data_model.label
        self.label_map = data_model.label_map
        self.label_pval = data_model.pos_label_val
        self.label_nval = data_model.neg_label_val
        self.processed_data = data_model.get_processed_data()
        # TODO: Predict only 1 times can reduce time cost
        raw_data = data_model.get_raw_data()
        self.data_with_prediction = raw_data
        self.data_with_prediction.insert(
            len(raw_data.columns), 'prediction',
            self.predictor.predict(self.processed_data)
        )
        self.data_with_prediction.insert(
            len(raw_data.columns), 'binary prediction',
            self.predictor.predict(self.processed_data, to_binary=True)
        )

        self.__glb_metrivals = None
        self.__fair_range = None

    def get_glb_metrivals(self):
        if self.__glb_metrivals is None:
            self.__glb_metrivals = self.compute_metrics(**self.__get_confus_vals())
        return self.__glb_metrivals

    def get_fair_range(self):
        if self.__fair_range is None:
            glb_metrivals = self.get_glb_metrivals()
            self.__fair_range = [
                [glb_metrivals[mtrc] * THRESHOLDS[mtrc]
                 for mtrc in METRICS],  # Lower Bound
                [min(glb_metrivals[mtrc] / THRESHOLDS[mtrc], METRIC_UBS[mtrc])
                 for mtrc in METRICS],  # Upper Bound
            ]
        return self.__fair_range

    def analyze_gf(self, featr):
        metrivals = {}
        confus_vals = self.__get_grp_confus_vals(featr)
        cmmts = []
        for grp in confus_vals.keys():
            metrivals[grp] = self.compute_metrics(**confus_vals[grp])
            cmmts += self.make_fairness_cmmts(featr, grp, metrivals[grp])
        if not cmmts:
            cmmts.append(f'对{featr}特征进行分析，未发现公平性问题')
        return metrivals, cmmts

    def make_fairness_cmmts(self, featr, grp, metrivals):
        # make for a single group
        cmmts = []
        fair_range = self.get_fair_range()
        for i, mtrc in enumerate(METRICS):
            metrival = metrivals[mtrc]
            discrimination = ''
            if metrival < fair_range[0][i]:
                discrimination = '歧视' if PREFER_HIGH[mtrc] else '偏爱'
            elif metrival > fair_range[1][i]:
                discrimination = '偏爱' if PREFER_HIGH[mtrc] else '歧视'
            metric = mtrc + '和Equalized Odds' if mtrc in {'FPR', 'FNR'} else mtrc
            if discrimination != '':
                cmmts.append(
                    ModelEvaluator.fairness_cmmt_fmt %
                    (metric, grp, discrimination, metric, featr, featr)
                )
        return cmmts

    @staticmethod
    def compute_metrics(TP, FP, FN, TN):
        total = TP + FP + FN + TN + 1e-8
        res = {
            'Acc': (TP + TN) / total,
            'PLR': (TP + FP) / total,
            'PPV': TP / (TP + FP),
            'FPR': FP / (FP + TN),
            'FNR': FN / (TP + FN),
            'NPV': TN / (TN + FN)
        }
        return res

    def __get_confus_vals(self, frame=None):
        if frame is None:
            frame = self.data_with_prediction

        T_subframe = frame[frame[self.label] == self.label_pval]
        F_subframe = frame[frame[self.label] == self.label_nval]
        res = {
            'TP': T_subframe[T_subframe['binary prediction'] == 1].shape[0],
            'FP': T_subframe[T_subframe['binary prediction'] == 0].shape[0],
            'FN': F_subframe[F_subframe['binary prediction'] == 1].shape[0],
            'TN': F_subframe[F_subframe['binary prediction'] == 0].shape[0]
        }
        return res

    def __get_grp_confus_vals(self, featr):
        res = {}
        groups = self.data_with_prediction[featr].unique()
        for grp in groups:
            frame = self.data_with_prediction
            subframe = frame[frame[featr] == grp]
            res[grp] = self.__get_confus_vals(subframe)
        return res


class ModelEvalView:
    def __init__(self, name, data_model):
        self.name = name
        self.featrs = data_model.featrs
        self.n_featrs = data_model.n_featrs
        self.c_featrs = data_model.c_featrs

        self.sens_featrs = []
        self.legi_featr = None

        self.gf_cmmts = []

    def update_gf_res(self, evaltr: ModelEvaluator, sens_featrs):
        self.gf_cmmts.clear()

        self.sens_featrs = sens_featrs
        charts = []
        fair_range = evaltr.get_fair_range()
        for i, featr in enumerate(sens_featrs):
            grp_metrivals, cmmts = evaltr.analyze_gf(featr)

            chart = (
                Radar()
                .add_schema(
                    schema=[
                        chopts.RadarIndicatorItem(mtrc, max_=METRIC_UBS[mtrc])
                        for mtrc in METRICS
                    ],
                    splitarea_opt=chopts.SplitAreaOpts(
                        is_show=True, areastyle_opts=chopts.AreaStyleOpts(opacity=1)
                    )
                )
                .add(
                    '公平范围', fair_range,
                    label_opts=chopts.LabelOpts(is_show=False),
                    linestyle_opts=chopts.LineStyleOpts(color='red', width=2)
                )
            )

            for j, grp in enumerate(grp_metrivals.keys()):
                # Compute linear gradient RGB
                alpha = 0.75 * j / (len(grp_metrivals) - 1)
                g, b = round(alpha * 255 + 63), round((0.75 - alpha) * 255 + 63)
                color = f'#{get_rgb_hex(0, g, b)}'

                chart.add(
                    grp, [[grp_metrivals[grp][mtrc] for mtrc in METRICS]],
                    label_opts=chopts.LabelOpts(is_show=False),
                    linestyle_opts=chopts.LineStyleOpts(width=2),
                    areastyle_opts = chopts.AreaStyleOpts(opacity=0.2),
                    color=color
                )
            charts.append(chart)
            self.gf_cmmts.append((i, cmmts))
        return charts

    def update_cgf_res(self, model, data, sens_featrs, legi_featr):
        self.sens_featrs = sens_featrs
        self.legi_featr = legi_featr
        pass


class ModelEvalController:
    insts = {}

    def __init__(self, ip, struct_file, nn_file):
        name, model = upload_model(ip, struct_file, nn_file)
        self.data_model = DataController.insts[ip].model
        self.model_evaltr = ModelEvaluator(
            Predictor(model), self.data_model,
            DataEvaluator(self.data_model)
        )
        self.view = ModelEvalView(name, self.data_model)
        self.charts = {}
        ModelEvalController.insts[ip] = self

    def radar_eval(self, sens_featrs):
        charts = self.view.update_gf_res(self.model_evaltr, sens_featrs)
        for i, chart in enumerate(charts):
            self.charts[f'0{i}'] = chart

    def cgf_eval(self, sens_featr, legi):
        pass

