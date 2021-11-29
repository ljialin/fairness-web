"""
  @Time : 2021/10/30 14:10 
  @Author : Ziqi Wang
  @File : algo_cfg.py 
"""
import json
import os

from src.utils import auto_dire
from src.task import Task
from src.mvc.model_upload import init_pop_from_uploaded
from src.algorithm import Algorithm
import numpy as np


class STATUS:
    INIT = 0
    INIT_PROBLEM = 1
    INIT_POP = 2
    INIT_CONFIG = 3
    RUNNING = 10
    FINISH = 11
    ERROR = 12


class AlgoCfgView:
    def __init__(self, data_model):
        self.featrs = []
        self.dataname = ""
        if data_model is not None:
            self.featrs = data_model.featrs
            self.dataname = data_model.name


class AlgoCfg:
    def __init__(self):
        self.acc_metric = 'Accuracy'
        self.fair_metric = 'Individual_fairness'
        self.optimizer = 'SRA'
        self.pop_size = 25
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

        self.status = STATUS.INIT
        self.progress = 0  # 100就结束
        self.progress_info = ""

        self.pops = []  # list套二维np格式，第1个下标选代数，第2个下表选中个体，第3个下标选中个体维度
        self.max_pop = [0, 0]

        AlgoController.instances[ip] = self

    def new_task(self, **kwargs):
        errinfo = None
        self.cfg.acc_metric = kwargs['acc_metric']
        self.cfg.fair_metric = kwargs['fair_metric']

        try:
            self.cfg.pop_size = int(kwargs['pop_size'])
        except:
            return "种群大小必须是数字！", 0

        try:
            self.cfg.max_gens = int(kwargs['max_gens'])
        except:
            return "代数必须是数字！", 0

        self.cfg.sens_featrs = kwargs['sens_featrs']
        if len(self.cfg.sens_featrs) == 0:
            return "需要选择敏感属性", 0

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
        pop_files = os.listdir(self.save_dir)
        pop_files.pop(0)
        for file in pop_files:
            pop = np.loadtxt(self.save_dir + file)
            self.pops.append(pop)


    def update_progress(self, status, gen=0, maxgen=1, error_info=""):
        self.status = status
        if status == STATUS.RUNNING:
            self.progress = round(gen * 100 / maxgen, 2)
            self.progress_info = '演化中... 代数{}/{}({}%)'.format(gen, maxgen, str(self.progress))
            if gen == 1:
                for i in range(len(self.max_pop)):
                    self.max_pop[i] = max(self.max_pop[i], max(self.pops[-1][:, i]))
        elif status == STATUS.INIT_PROBLEM:
            self.progress_info = '正在实例化问题对象...'
        elif status == STATUS.INIT_POP:
            self.progress_info = '正在初始化种群...'
        elif status == STATUS.INIT_CONFIG:
            self.progress_info = '初始化参数配置...'
        elif status == STATUS.FINISH:
            self.progress_info = '演化完成！'
            algomnger = AlgosManager.instances[self.ip]
            algomnger.running_tasks.pop(self.task.id)
            algomnger.finished_tasks[self.task.id] = self
        elif status == STATUS.ERROR:
            self.progress_info = error_info
        self.__saveConfig()  # 把每个状态写出文件

    def update_pop(self, pop, gen=0):
        np.savetxt(self.save_dir + 'pop_objs_valid{}.txt'.format(gen), pop)
        self.pops.append(pop)


    def add_models(self, struct_file, var_files):
        models = init_pop_from_uploaded(self.task.id, struct_file, var_files, self.cfg.pop_size)
        self.algo.pop = models


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