"""
  @Time : 2021/8/30 15:53 
  @Author : Ziqi Wang
  @File : model_eval.py 
"""

import os
import torch
from werkzeug.utils import secure_filename
from root import PRJROOT


def upload_model(ip, nn_file):
    if nn_file.filename[-4:] not in {'.pth', '.pkl'}:
        return '模型文件必须是.pth或.pkl后缀'
    # path = os.path.join(
    #     PRJROOT + 'models/temp',
    #     ip + secure_filename(nn_file.filename)
    # )
    # nn_file.save(path)
    # model = torch.load(path)
    # os.remove(path)
    return nn_file.filename[:-4], 0



class Predictor:
    def __init__(self, model):
        self.model = model
        pass


class ModelEvalView:
    def __init__(self, name):
        self.name = name


class ModelEvalController:
    insts = {}

    def __init__(self, ip, nn_file):
        name, model = upload_model(ip, nn_file)
        self.predictor = Predictor(model)
        self.view = ModelEvalView(name)
        ModelEvalController.insts[ip] = self
