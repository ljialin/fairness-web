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
from root import PRJROOT


METRICS = ('PLR', 'Acc', 'PPV', 'FPR', 'FNR', 'NPV')
NUM_METRICS = len(METRICS)
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
            predictions = self.model(data[start: min(len(data), start+batch)])
            res += predictions.tolist()
            start += batch
        res = np.array(res)
        if to_binary:
            res = np.where(res > 0.5, 1, 0)
        return res
        # print(self.model)
        # pass


class ModelEvaluator:
    def __init__(self, predictor, data_model, data_evaltr):
        # self.data_model = data_model
        self.data_evaltr = data_evaltr
        self.predictor = predictor

        self.label = data_model.label
        self.label_map = data_model.label_map
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

        self.__global_acc = None
        self.__global_plr = None

    def get_glb_acc(self):
        if self.__global_acc is None:
            ground_truth = [
                self.label_map(item[self.label])
                for item in self.data_with_prediction[self.label]
            ]
            predictions = self.data_with_prediction['binary prediction']
            num_correct = np.where(
                predictions - ground_truth == 0,
                1, 0
            ).sum()
            self.__global_acc = num_correct / len(self.data_with_prediction)
        return self.__global_acc

    def get_grp_acc(self, featr) -> pds.Series:
        frame = self.data_with_prediction[
            [featr, self.label, 'binary prediction']
        ]
        total_cnts = frame[featr].value_counts()
        positive_cnts = (frame
            [frame[self.label] == frame['binary prediction']]
            [featr].value_counts()
        )
        return positive_cnts / total_cnts

    def get_glb_plr(self):
        if self.__global_plr is None:
            self.__global_plr = self.data_evaltr.get_glb_plr(
                self.data_with_prediction,
                ('binary_prediction', 1)
            )
        return self.__global_plr

    def get_grp_plr(self, featr):
        return self.data_evaltr.get_glb_plr(
            self.data_with_prediction, featr,
            ('binary_prediction', 1)
        )

    def get_glb_confus_metrics(self):
        tp_cnt, fp_cnt, fn_cnt, tn_cnt = self.get_glb_plr()
        return self.get_confus_metrics(tp_cnt, fp_cnt, fn_cnt, tn_cnt)

    def get_grp_confus_metrics(self, featr):
        pass
        # for grp in self.data
        # tp_cnt, fp_cnt, fn_cnt, tn_cnt = self.get_glb_plr()
        # return 0, 0, 0, 0

    @staticmethod
    def get_confus_metrics(tp_cnt, fp_cnt, fn_cnt, tn_cnt):
        PPV = tp_cnt / (tp_cnt + fp_cnt)
        FPR = fp_cnt / (fp_cnt + tn_cnt)
        FNR = fn_cnt / (tp_cnt + fn_cnt)
        NPV = tn_cnt / (tn_cnt + fn_cnt)
        return PPV, FPR, FNR, NPV

    def __get_glb_confus(self):
        # Return tp, fp, fn, tn
        return 1, 1, 1, 1

    def __get_grp_confus(self, featr):
        # Return {'<grp_name>': (tp, fp, fn, tn)} for all group of <featr>
        return 0, 0, 0, 0


class ModelEvalView:
    def __init__(self, name):
        self.name = name

    def update_radar(self, evaltr: ModelEvaluator, sens_featrs, theta=0.8):
        charts = []
        for i, featr in enumerate(sens_featrs):
            glb_acc = evaltr.get_glb_acc()
            grp_acc = evaltr.get_grp_acc(featr)
            glb_plr = evaltr.get_glb_plr()
            grp_plr = evaltr.get_grp_plr(featr)

            chart = (
                Radar()
                .add_schema(
                    schema=[
                        chopts.RadarIndicatorItem(name="PLR", max_=2),
                        # chopts.RadarIndicatorItem(name="Well-calibration", min_=-2, max_=2),
                        chopts.RadarIndicatorItem(name="Acc", max_=1),
                        chopts.RadarIndicatorItem(name="PPV", max_=2),
                        chopts.RadarIndicatorItem(name="FPR", max_=2),
                        chopts.RadarIndicatorItem(name="FNR", max_=2),
                        chopts.RadarIndicatorItem(name="NPV", max_=2),
                    ],
                    splitarea_opt=chopts.SplitAreaOpts(
                        is_show=True, areastyle_opts=chopts.AreaStyleOpts(opacity=1)
                    )
                )
                .add(
                    '公平范围', [
                        [glb_plr * theta, glb_acc * theta, 1 * theta,
                         1 * theta, 1 * theta, 1 * theta],      # 下界
                        [glb_plr / theta, glb_acc / theta, 1 / theta,
                         1 / theta, 1 / theta, 1 / theta],      # 上界
                    ]
                )
            )

            groups = grp_plr.index
            grp_metrivals = np.zeros([len(groups), NUM_METRICS])    # rows: metric values of group i
            grp_metrivals[:, 0] = grp_plr.values
            grp_metrivals[:, 1] = grp_acc.values
            grp_metrivals[:, 2] = 1
            grp_metrivals[:, 3] = 1
            grp_metrivals[:, 4] = 1
            grp_metrivals[:, 5] = 1

            for i, grp in enumerate(groups):
                chart.add(
                    grp, grp_metrivals[i],
                    label_opts=chopts.LabelOpts(is_show=False)
                )

        return charts


class ModelEvalController:
    insts = {}

    def __init__(self, ip, struct_file, nn_file):
        name, model = upload_model(ip, struct_file, nn_file)
        self.data_model = DataController.insts[ip].model
        self.model_evaltr = ModelEvaluator(
            Predictor(model), self.data_model,
            DataEvaluator(self.data_model)
        )
        self.view = ModelEvalView(name)
        self.charts = {}
        ModelEvalController.insts[ip] = self

    def radar_eval(self, sens_featrs):
        charts = self.view.update_radar(self.model_evaltr, sens_featrs)
        for i, chart in enumerate(charts):
            self.charts[f'0{i}'] = chart
        # accuracy = self.model_evaltr.accuracy()

    def cgf_eval(self, sens_featr, legi):
        pass

