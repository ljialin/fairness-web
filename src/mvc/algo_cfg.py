"""
  @Time : 2021/10/30 14:10 
  @Author : Ziqi Wang
  @File : algo_cfg.py 
"""
import json
import os

import torch

from src.utils import auto_dire
from src.task import Task
from src.mvc.model_upload import init_pop_from_uploaded
from src.algorithm import Algorithm
import numpy as np
from flask_babel import gettext as _
import geatpy as ea


class STATUS:
    INIT = 0
    INIT_PROBLEM = 1
    INIT_POP = 2
    INIT_CONFIG = 3
    RUNNING = 10
    FINISH = 11
    ERROR = 12
    ABORT = 13
    PAUSE = 14
    PAUSED = 15


class AlgoCfgView:
    def __init__(self, data_model):
        self.featrs = []
        self.dataname = ""
        if data_model is not None:
            self.featrs = data_model.featrs
            self.dataname = data_model.name


class AlgoCfg:
    def __init__(self):
        self.acc_metric = 'BCE_loss'
        self.fair_metric = ['Individual_fairness']
        self.optimizer = 'SRA'
        self.pop_size = 50
        self.max_gens = 100
        self.sens_featrs = []

    # def get_models(self):
    #     if len(self.models) >= self.pop_size:
    #         return self.models[:self.pop_size]
    #     else:
    #         lack = self.pop_size - len(self.models)
    #         for _ in range(lack):
    #             pass
    #     pass


# class Problem:
#     def __init__(self, data_model):
#         self.data_model = data_model

class Individual:
    def __init__(self, pop, NNmodel):
        self.pop = pop #二维NP
        self.NNmodel = NNmodel


class AlgoController:
    instances = {}

    def __init__(self, ip, data_model=None):
        self.ip = ip
        self.cfg = AlgoCfg()
        self.view = AlgoCfgView(data_model)
        self.data_model = data_model #None的情况是导入task
        self.task = None
        self.algo = Algorithm()
        self.save_dir = None

        self.status = STATUS.INIT  # 后端传前端
        self.is_abort = 0  # 前端传后端
        self.progress = 0  # 100就结束
        self.progress_info = ""

        self.pops = []  # list套二维np格式，第1个下标选代数，第2个下表选中个体，第3个下标选中个体维度
        self.pops1 = []  # 非支配解
        self.pops2 = []  # 支配解
        self.max_pop = [0, 0]

        AlgoController.instances[ip] = self

    def new_task(self, **kwargs):
        errinfo = None
        self.cfg.acc_metric = kwargs['acc_metric']
        self.cfg.fair_metric = kwargs['fair_metric']
        self.cfg.optimizer = kwargs['optimizer']
        self.max_pop = [0 for i in range(len(self.cfg.fair_metric) + 1)]
        if len(self.cfg.fair_metric) == 0:
            return _("task_error4"), 0
        self.selected_fair = self.cfg.fair_metric[0]

        try:
            self.cfg.pop_size = int(kwargs['pop_size'])
        except:
            return _("task_error1"), 0

        try:
            self.cfg.max_gens = int(kwargs['max_gens'])
        except:
            return _("task_error2"), 0

        self.cfg.sens_featrs = kwargs['sens_featrs']
        # if len(self.cfg.sens_featrs) == 0:
        #     return "需要选择敏感属性", 0
        for fear in self.cfg.sens_featrs:
            if fear in self.data_model.n_featrs:
                return _("task_error3"), 0

        task_id = 'task' + auto_dire('task', 'task_space' + os.sep + str(self.ip), fmtr='%04d')[-4:]
        self.task = Task(task_id)
        self.save_dir = 'task_space/{}/{}/'.format(str(self.ip), task_id)
        return errinfo, task_id

    def __saveConfig(self):
        config = {
            "status": self.status,
            "objectives_class": [self.cfg.acc_metric, self.cfg.fair_metric],
            "optimizer": self.cfg.optimizer,
            "popsize": self.cfg.pop_size,
            "maxgen": self.cfg.max_gens,
            "sensitive_attributions": self.cfg.sens_featrs,
            "model_name": self.view.dataname,
            "progress": self.progress,
            "progress_info": self.progress_info,
            "max_pop": self.max_pop,
        }
        with open(self.save_dir + 'config.json', 'w', encoding='utf8') as f:
            f.write(json.dumps(config, indent=2))

    def loadConfig(self):
        self.save_dir = 'task_space/{}/{}/'.format(self.ip, self.task.id)
        # 读取参数和状态
        with open(self.save_dir + 'config.json', 'r', encoding='utf8') as f:
            text = f.read()
            config = json.loads(text)
            self.status = config["status"] #开始时候读取只会有finish状态的task
            if self.status == STATUS.FINISH:
                self.progress = 100
            self.cfg.acc_metric = config["objectives_class"][0]
            self.cfg.fair_metric = config["objectives_class"][1]
            self.cfg.optimizer = config["optimizer"]
            self.cfg.pop_size = config["popsize"]
            self.cfg.max_gens = config["maxgen"]
            self.cfg.sens_featrs = config["sensitive_attributions"]
            self.view.dataname = config["model_name"]
            self.progress = config["progress"]
            self.progress_info = config["progress_info"]
            self.max_pop = config["max_pop"]
        # 读取种群
        dir = self.save_dir + 'fitness/'
        if not os.path.exists(dir):
            return
        pop_files = os.listdir(dir)
        pop_files = sorted(pop_files, key=lambda x: os.path.getmtime(os.path.join(dir, x)))
        for file in pop_files:
            pop = np.loadtxt(dir + file)
            if len(pop.shape) == 1: #只有一个个体，升维
                pop = np.reshape(pop,(1, -1))
            self.pops.append(pop)
            # 算支配关系
            pop1, pop2 = self.update_dominate_relation(pop)
            self.pops1.append(pop1)
            self.pops2.append(pop2)


    def update_progress(self, status, gen=0, maxgen=1, error_info=""):
        self.status = status
        if status == STATUS.RUNNING:
            self.progress = round(gen * 100 / maxgen, 2)
            self.progress_info = _("progress_info_1").format(gen, maxgen, str(self.progress))
            # if gen == 1:
            for i in range(len(self.max_pop)):
                self.max_pop[i] = max(self.max_pop[i], max(self.pops[-1][:, i]))
        elif status == STATUS.INIT_PROBLEM:
            self.progress_info = _("progress_info_2")
        elif status == STATUS.INIT_POP:
            self.progress_info = _("progress_info_3")
        elif status == STATUS.INIT_CONFIG:
            self.progress_info = _("progress_info_4")
        elif status == STATUS.FINISH:
            self.progress_info = _("progress_info_5")
            algomnger = AlgosManager.instances[self.ip]
            algomnger.running_tasks.pop(self.task.id)
            algomnger.finished_tasks[self.task.id] = self
        elif status == STATUS.ERROR:
            self.progress_info = error_info
        elif status == STATUS.ABORT:
            self.progress_info = _("progress_info_6").format(gen, maxgen, str(self.progress))
            algomnger = AlgosManager.instances[self.ip]
            algomnger.running_tasks.pop(self.task.id)
            algomnger.finished_tasks[self.task.id] = self
        self.__saveConfig()  # 把每个状态写出文件

    # def save_pop(self, pop, NNmodels, size, gen=0):
    #     # 更新支配关系
    #     pop1, pop2, idx1, idx2 = self.update_dominate_relation(pop)
    #     self.pops1.append(pop1)
    #     self.pops2.append(pop2)
    #     new_pop = np.concatenate((pop1, pop2))
    #     idx2 += np.ones(len(idx2), dtype=int)*len(idx1)
    #     new_NNmodels = [None for i in range(size)]
    #     for i in range(len(idx1)):
    #         idx = idx1[i]
    #         new_NNmodels[i] = NNmodels[idx]
    #     for i in range(len(idx1), len(idx2)+len(idx1)):
    #         idx = idx2[i-len(idx1)]
    #         new_NNmodels[i] = NNmodels[idx]
    #
    #     dir = self.get_savepop_dir()
    #     for i in range(size):
    #         torch.save(new_NNmodels[i].state_dict(), dir + 'indiv_{}.pth'.format(str(i+1)))
    #
    #     dir = self.save_dir + 'fitness/'
    #     if not os.path.exists(dir):
    #         os.mkdir(dir)
    #     np.savetxt(dir + 'pop_objs_valid_{}.txt'.format(gen), new_pop)
    #     self.pops.append(new_pop)

    def save_pop(self, pop, NNmodels, size, gen=0):
        NNmodels = np.array(NNmodels)
        pop1, pop2, NNmodels1, NNmodels2 = self.update_dominate_relation(pop, NNmodels)
        self.pops1.append(pop1)
        self.pops2.append(pop2)
        new_pop = np.concatenate((pop1, pop2))
        new_NNmodels = np.concatenate((NNmodels1, NNmodels2))

        dir = self.get_savepop_dir()
        for i in range(size):
            torch.save(new_NNmodels[i].state_dict(), dir + 'indiv_{}.pth'.format(str(i+1)))

        dir = self.save_dir + 'fitness/'
        if not os.path.exists(dir):
            os.mkdir(dir)
        np.savetxt(dir + 'pop_objs_valid_{}.txt'.format(gen), new_pop)
        self.pops.append(new_pop)

    def update_dominate_relation(self, pop, NNmodels=None):
        [level, _] = ea.ndsortDED(pop)
        pop1 = pop[np.where(level == 1)]
        idx1 = pop1.argsort(axis=0)[:,0]
        pop1 = pop1[idx1]

        pop2 = pop[np.where(level != 1)]
        idx2 = pop2.argsort(axis=0)[:, 0]
        pop2 = pop2[idx2]

        if NNmodels is not None:
            NNmodels1 = NNmodels[np.where(level == 1)]
            NNmodels1 = NNmodels1[idx1]
            NNmodels2 = NNmodels[np.where(level != 1)]
            NNmodels2 = NNmodels2[idx2]
            return pop1, pop2, NNmodels1, NNmodels2

        return pop1, pop2


    def add_models(self, struct_file, var_files):
        models = init_pop_from_uploaded(self.task.id, struct_file, var_files, self.cfg.pop_size)
        self.algo.pop = models

    def get_savepop_dir(self):
        dir = self.save_dir + 'net/'
        if not os.path.exists(dir):
            os.mkdir(dir)
        return dir


class AlgosManager:
    instances = {}

    def __init__(self, ip):
        self.ip = ip
        self.finished_tasks = {}  # 单个task可以用单个AlgoCfgController来装
        self.running_tasks = {}
        self.task_space = 'task_space/{}/'.format(str(self.ip))

        self.__loadTasks()
        AlgosManager.instances[ip] = self

        # 读取所有已经完成的task

    def __loadTasks(self):
        try: #新IP会找不到路径
            task_ids = os.listdir(self.task_space)
        except:
            return
        for task_id in task_ids:
            ctrlr = AlgoController(self.ip)
            ctrlr.task = Task(task_id)
            ctrlr.loadConfig()
            status = ctrlr.status
            if status == STATUS.FINISH:
                self.finished_tasks[task_id] = ctrlr
            else:
                self.running_tasks[task_id] = ctrlr
                #这个分支目前看来不会发生，因为服务器每次关闭后默认所有任务是结束状态

    def add_task(self, task_id, algoCfgController):
        if self.finished_tasks.get(task_id) is None:
            self.running_tasks[task_id] = algoCfgController

    def get_task(self, task_id): #要从两个集合中找，看看在哪
        if self.running_tasks.get(task_id) is not None:
            return self.running_tasks.get(task_id)
        if self.finished_tasks.get(task_id) is not None:
            return self.finished_tasks.get(task_id)
        return None
