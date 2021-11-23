"""
  @Time : 2021/10/30 14:10 
  @Author : Ziqi Wang
  @File : algo_cfg.py 
"""

from src.utils import auto_dire
from src.task import Task
from src.mvc.model_upload import init_pop_from_uploaded
from src.algorithm import Algorithm


class AlgoCfgView:
    def __init__(self, data_model):
        self.featrs = data_model.featrs


class AlgoCfg:
    def __init__(self):
        self.acc_metric = 'Accuracy'
        self.fair_metric = 'Individual_fairness'
        self.optimizer = 'SRA'
        self.pop_size = 25
        self.max_gens = 100
        self.sens_featrs = ['age']
        self.models = []

    # def get_models(self):
    #     if len(self.models) >= self.pop_size:
    #         return self.models[:self.pop_size]
    #     else:
    #         lack = self.pop_size - len(self.models)
    #         for _ in range(lack):
    #             pass
    #     pass


class Problem:
    def __init__(self, data_model):
        self.data_model = data_model


class AlgoCfgController:
    instances = {}
    def __init__(self, ip, data_model):
        self.ip = ip
        self.cfg = AlgoCfg()
        self.problem = Problem(data_model)
        self.view = AlgoCfgView(data_model)
        self.task = None
        self.algo = Algorithm()
        self.progress = 0 #100就结束
        self.progress_info = ""
        self.inQ = None
        AlgoCfgController.instances[ip] = self

    def set(self, **kwargs):
        self.cfg.acc_metric = kwargs['acc_metric']
        self.cfg.fair_metric = kwargs['fair_metric']
        self.cfg.pop_size = kwargs['pop_size']
        self.cfg.max_gens = kwargs['max_gens']
        self.cfg.sens_featrs = kwargs['sens_featrs']
        task_id = int(auto_dire('task', 'task_space', fmtr='%04d')[-4:])
        self.task = Task(task_id)
        return task_id

    def add_models(self, struct_file, var_files):
        models = init_pop_from_uploaded(self.task.id, struct_file, var_files, self.cfg.pop_size)
        self.algo.pop = models

class AlgoManager:
    instances = {}
    def __init__(self, ip):
        self.ip = ip
        self.finished_tasks = {} #单个task可以用单个AlgoCfgController来装
        self.running_tasks = {}
        AlgoManager.instances[ip] = self

    def add_task(self, task_id, algoCfgController):
        self.running_tasks[task_id] = algoCfgController