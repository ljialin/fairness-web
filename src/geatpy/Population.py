# -*- coding: utf-8 -*-
import os
import numpy as np
import geatpy as ea
from moeas4NN.nets import Population_NN, weights_init
import torch
import time
import copy
from torch import nn
from mvc.model_eval import IndividualNet2


def weight_init(m):
    if isinstance(m, nn.Linear):
        # nn.init.xavier_normal_(m.weight)
        # nn.init.constant_(m.bias, 0)

        # uniform distribution
        nn.init.kaiming_uniform_(m, mode='fan_in', nonlinearity='relu')

        # normal distribution
        # nn.init.kaiming_normal_(w, mode='fan_out', nonlinearity='relu')
    # 也可以判断是否为conv2d，使用相应的初始化方式
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
     # 是否为批归一化层
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

class Population:
    """
Population : class - 种群类

描述:
    种群类是用来存储种群相关信息的一个类。

属性:
    sizes    : int   - 种群规模，即种群的个体数目。

    ChromNum : int   - 染色体的数目，即每个个体有多少条染色体。

    Encoding : str   - 染色体编码方式，
                       'BG':二进制/格雷编码；
                       'RI':实整数编码，即实数和整数的混合编码；
                       'P':排列编码。
                       相关概念：术语“实值编码”包含实整数编码和排列编码，
                       它们共同的特点是染色体不需要解码即可直接表示对应的决策变量。
                       "实整数"指的是种群染色体既包含实数的小数，也包含实数的整数。
                       特殊用法：
                       设置Encoding=None，此时种群类的Field,Chrom成员属性将被设置为None，
                       种群将不携带与染色体直接相关的信息，可以减少不必要的数据存储，
                       这种用法可以在只想统计非染色体直接相关的信息时使用，
                       尤其可以在多种群进化优化过程中对个体进行统一的适应度评价时使用。

    Field    : array - 译码矩阵，可以是FieldD或FieldDR（详见Geatpy数据结构）。

    Chrom    : array - 种群染色体矩阵，每一行对应一个个体的一条染色体。

    Lind     : int   - 种群染色体长度。

    ObjV     : array - 种群目标函数值矩阵，每一行对应一个个体的目标函数值，每一列对应一个目标。

    FitnV    : array - 种群个体适应度列向量，每个元素对应一个个体的适应度，最小适应度为0。

    CV       : array - CV(Constraint Violation Value)是用来定量描述违反约束条件程度的矩阵，每行对应一个个体，每列对应一个约束。
                       注意：当没有设置约束条件时，CV设置为None。

    Phen     : array - 种群表现型矩阵（即种群各染色体解码后所代表的决策变量所组成的矩阵）。

函数:
    详见源码。

"""

    def __init__(self, Encoding, Field, NIND, Chrom=None, ObjV=None, ObjV_train=None, ObjV_valid=None, ObjV_ensemble=None, ObjV_test=None,
                 FitnV=None, CV=None, Phen=None, isNN = 0, n_feature=108, n_hidden=100, n_output=1, parameters={},
                 logits=None, pred_label_train=None, pred_label_valid=None, pred_label_ensemble=None, pred_label_test=None,
                 pred_logits_train=None, pred_logits_valid=None, pred_logits_ensemble=None, pred_logits_test=None, info_id=None, family_list=None, is_ensemble=False):

        """
        描述: 种群类的构造函数，用于实例化种群对象，例如：
             import geatpy as ea
             population = ea.Population(Encoding, Field, NIND)，
             NIND为所需要的个体数。
             此时得到的population还没被真正初始化，仅仅是完成种群对象的实例化。
             该构造函数必须传入Chrom，才算是完成种群真正的初始化。
             一开始可以只传入Encoding, Field以及NIND来完成种群对象的实例化，
             其他属性可以后面再通过计算进行赋值。
             另外还可以利用ea.Population(Encoding, Field, 0)来创建一个“空种群”,即不含任何个体的种群对象。
             parameters: 算法的参数
        """

        if isinstance(NIND, int) and NIND >= 0:
            self.sizes = NIND
        else:
            raise RuntimeError('error in Population: Size error. (种群规模设置有误，必须为非负整数。)')
        self.ChromNum = 1
        self.isNN = isNN
        if Chrom is not None:
            if isinstance(Chrom[0], torch.nn.Module):
                self.isNN = 1
        # -------- ZQQ - begin -----------
        if self.isNN == 1:
            self.Encoding = 'NN'
        else:
            self.Encoding = Encoding
        # -------- ZQQ - end -----------

        if Encoding is None:
            self.Field = None
            self.Chrom = None
        else:
            self.Field = Field.copy()
            self.Chrom = copy.deepcopy(Chrom) if Chrom is not None else None

        # -------- ZQQ - begin -----------
        if self.isNN == 0:
            self.Lind = Chrom.shape[1] if Chrom is not None else 0
        elif self.isNN == 1:
            if Chrom is not None:
                self.Lind = 0
                with torch.no_grad():
                    for name, param in self.Chrom[0].named_parameters():
                        self.Lind += 1  # 计算染色体的长度
            else:
                self.Lind = 0
        # -------- ZQQ - end -----------

        self.ObjV = ObjV.copy() if ObjV is not None else None
        self.ObjV_train = ObjV_train.copy() if ObjV_train is not None else None
        self.ObjV_valid = ObjV_valid.copy() if ObjV_valid is not None else None
        self.ObjV_ensemble = ObjV_ensemble.copy() if ObjV_ensemble is not None else None
        self.ObjV_test = ObjV_test.copy() if ObjV_test is not None else None
        self.FitnV = FitnV.copy() if FitnV is not None else None
        self.CV = CV.copy() if CV is not None else None
        self.Phen = Phen.copy() if Phen is not None else None
        self.logits = logits.copy() if logits is not None else None # 在kfold后的网络在test的平均值
        self.pred_label_train = pred_label_train.copy() if pred_label_train is not None else None
        self.pred_label_valid = pred_label_valid.copy() if pred_label_valid is not None else None
        self.pred_label_ensemble = pred_label_ensemble.copy() if pred_label_ensemble is not None else None
        self.pred_label_test = pred_label_test.copy() if pred_label_test is not None else None
        self.pred_logits_train = pred_logits_train.copy() if pred_logits_train is not None else None
        self.pred_logits_valid = pred_logits_valid.copy() if pred_logits_valid is not None else None
        self.pred_logits_ensemble = pred_logits_ensemble.copy() if pred_logits_ensemble is not None else None
        self.pred_logits_test = pred_logits_test.copy() if pred_logits_test is not None else None
        self.n_feature = n_feature
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.start_time = time.strftime("%Y-%m-%d-%H-%M", time.localtime(time.time()))
        self.parameters = parameters
        self.info_id = info_id
        self.family_list = family_list
        self.is_ensemble = is_ensemble

    def initChrom(self, NIND=None, dataname='ricci', dropout=0):

        """
        描述: 初始化种群染色体矩阵。

        输入参数:
            NIND : int - (可选参数)用于修改种群规模。
                         当其不缺省时，种群在初始化染色体矩阵前会把种群规模调整为NIND。

        输出参数:
            无输出参数。

        """

        if NIND is not None:
            self.sizes = NIND  # 重新设置种群规模
        if self.isNN == 0:

            self.Chrom = ea.crtpc(self.Encoding, self.sizes, self.Field)  # 生成染色体矩阵
            self.Lind = self.Chrom.shape[1]  # 计算染色体的长度
            self.ObjV = None
            self.FitnV = None
            self.CV = None
        elif self.isNN == 1:
            # -------- ZQQ - begin -----------
            self.ObjV = None
            self.ObjV_train = None
            self.ObjV_valid = None
            self.ObjV_ensemble = None
            self.ObjV_test = None
            self.FitnV = None
            self.CV = None
            # self.logits = self.logits

            population = []
            pop_ids = []
            family_list = []
            for i in range(self.sizes):
                # pop = copy.deepcopy(IndividualNet(self.n_feature, self.n_hidden, self.n_output, name=dataname, dropout=dropout))
                pop = copy.deepcopy(IndividualNet2(self.n_feature, self.n_hidden, self.n_output, dropout=dropout))
                pop.apply(weights_init)
                population.append(pop)
                pop_ids.append(i)
                family_list.append([])
            self.Chrom = copy.deepcopy(population)
            self.Lind = 0
            self.info_id = pop_ids
            self.family_list = family_list
            with torch.no_grad():
                for name, param in self.Chrom[0].named_parameters():
                    self.Lind += 1  # 计算染色体的长度
            print()
            # -------- ZQQ - end -----------

    def decoding(self):

        """
        描述: 种群染色体解码。

        """

        if self.Encoding == 'BG':  # 此时Field实际上为FieldD
            Phen = ea.bs2ri(self.Chrom, self.Field)  # 把二进制/格雷码转化为实整数
        elif self.Encoding == 'RI' or self.Encoding == 'P':
            Phen = self.Chrom.copy()
        # -------- ZQQ - begin -----------
        elif self.Encoding == 'NN':
            Phen = copy.deepcopy(self.Chrom)
        # -------- ZQQ - end -----------
        else:
            raise RuntimeError(
                'error in Population.decoding: Encoding must be ''BG'' or ''RI'' or ''P''. (编码设置有误，解码时Encoding必须为''BG'', ''RI'' 或 ''P''。)')
        return Phen

    def copy(self):

        """
        copy : function - 种群的复制
        用法:
            假设pop是一个种群矩阵，那么：pop1 = pop.copy()即可完成对pop种群的复制。

        """

        return Population(self.Encoding,
                          self.Field,
                          self.sizes,
                          copy.deepcopy(self.Chrom),
                          self.ObjV,
                          self.ObjV_train,
                          self.ObjV_valid,
                          self.ObjV_ensemble,
                          self.ObjV_test,
                          self.FitnV,
                          self.CV,
                          self.Phen,
                          parameters=self.parameters if self.parameters is not None and self.parameters is not None else None,
                          logits=self.logits,
                          pred_label_train=self.pred_label_train,
                          pred_label_valid=self.pred_label_valid,
                          pred_label_ensemble=self.pred_label_ensemble,
                          pred_label_test=self.pred_label_test,
                          pred_logits_train=self.pred_logits_train,
                          pred_logits_valid=self.pred_logits_valid,
                          pred_logits_ensemble=self.pred_logits_ensemble,
                          pred_logits_test=self.pred_logits_test,
                          info_id=self.info_id,
                          family_list=copy.deepcopy(self.family_list)

                          )

    def __getitem__(self, index):

        """
        描述: 种群的切片，即根据index下标向量选出种群中相应的个体组成一个新的种群。

        用法: 假设pop是一个包含多于2个个体的种群矩阵，那么：
             pop1 = pop[[0,1]]即可得到由pop种群的第1、2个个体组成的种群。

        注意: index必须为一个slice或者为一个Numpy array类型的行向量或者为一个list类型的列表或者为一个整数，
             该函数不对传入的index参数的合法性进行更详细的检查。

        """

        # 计算切片后的长度以及对index进行格式处理
        if not isinstance(index, (slice, np.ndarray, list, int, np.int32, np.int64)):
            raise RuntimeError(
                'error in Population: index must be an integer, a 1-D list, or a 1-D array. (index必须是一个整数，一维的列表或者一维的向量。)')

        if isinstance(index, slice):
            NIND = (index.stop - (index.start if index.start is not None else 0)) // (
                index.step if index.step is not None else 1)
            index_array = index
        else:
            index_array = np.array(index).reshape(-1)
            if index_array.dtype == bool:
                NIND = int(np.sum(index_array))
            else:
                NIND = len(index_array)
            if len(index_array) == 0:
                index_array = []

        if self.Encoding is None:
            NewChrom = None
        else:
            if self.Chrom is None:
                raise RuntimeError('error in Population: Chrom is None. (种群染色体矩阵未初始化。)')

            if self.isNN == 0:
                NewChrom = self.Chrom[index_array]
                Phen_zqq = self.Phen[index_array]
            # -------- ZQQ - end -----------
            elif self.isNN == 1:
                sum_count = len(self.Chrom)
                NewChrom = []
                for i in index_array:
                    temp = copy.deepcopy(self.Chrom[i])
                    NewChrom.append(temp)
                Phen_zqq = []
                for i in index_array:
                    Phen_zqq.append(self.Phen[i])
                infos = []
                for i in index_array:
                    infos.append(self.info_id[i])
                family_lists = []
                for i in index_array:
                    family_lists.append(copy.deepcopy(self.family_list[i]))
            # -------- ZQQ - end -----------
        return Population(self.Encoding,
                          self.Field,
                          NIND,
                          NewChrom,
                          self.ObjV[index_array] if self.ObjV is not None else None,
                          self.ObjV_train[index_array] if self.ObjV_train is not None else None,
                          self.ObjV_valid[index_array] if self.ObjV_valid is not None else None,
                          self.ObjV_ensemble[index_array] if self.ObjV_ensemble is not None else None,
                          self.ObjV_test[index_array] if self.ObjV_test is not None else None,
                          self.FitnV[index_array] if self.FitnV is not None else None,
                          self.CV[index_array] if self.CV is not None else None,
                          Phen_zqq if self.Phen is not None else None,
                          isNN=self.isNN,
                          parameters=self.parameters if self.parameters is not None and self.parameters is not None else None,
                          logits=self.logits[index_array] if self.logits is not None else None,
                          pred_label_train=self.pred_label_train[index_array] if self.pred_label_train is not None else None,
                          pred_label_valid=self.pred_label_valid[index_array] if self.pred_label_valid is not None else None,
                          pred_label_ensemble = self.pred_label_ensemble[index_array] if self.pred_label_ensemble is not None else None,
                          pred_label_test=self.pred_label_test[index_array] if self.pred_label_test is not None else None,
                          pred_logits_train=self.pred_logits_train[index_array] if self.pred_logits_train is not None else None,
                          pred_logits_valid=self.pred_logits_valid[index_array] if self.pred_logits_valid is not None else None,
                          pred_logits_ensemble=self.pred_logits_ensemble[index_array] if self.pred_logits_ensemble is not None else None,
                          pred_logits_test=self.pred_logits_test[index_array] if self.pred_logits_test is not None else None,
                          info_id=infos if self.info_id is not None else None,
                          family_list=family_lists if self.family_list is not None else None
                          )

    def shuffle(self):

        """
        shuffle : function - 打乱种群个体的个体顺序
        用法: 假设pop是一个种群矩阵，那么，pop.shuffle()即可完成对pop种群个体顺序的打乱。

        """

        shuff = np.arange(self.sizes)
        np.random.shuffle(shuff)  # 打乱顺序
        if self.Encoding is None:
            self.Chrom = None
        else:
            if self.Chrom is None:
                raise RuntimeError('error in Population: Chrom is None. (种群染色体矩阵未初始化。)')
            self.Chrom = copy.deepcopy(self.Chrom[shuff, :])
        self.ObjV = self.ObjV[shuff, :] if self.ObjV is not None else None
        self.ObjV_train = self.ObjV_train[shuff, :] if self.ObjV_train is not None else None
        self.ObjV_valid = self.ObjV_valid[shuff, :] if self.ObjV_valid is not None else None
        self.ObjV_ensemble = self.ObjV_ensemble[shuff, :] if self.ObjV_ensemble is not None else None
        self.ObjV_test = self.ObjV_test[shuff, :] if self.ObjV_test is not None else None
        self.FitnV = self.FitnV[shuff] if self.FitnV is not None else None
        self.CV = self.CV[shuff, :] if self.CV is not None else None
        self.Phen = self.Phen[shuff, :] if self.Phen is not None else None
        self.logits = self.logits[shuff, :] if self.logits is not None else None
        self.pred_label_train = self.pred_label_train[shuff, :] if self.pred_label_train is not None else None
        self.pred_label_valid = self.pred_label_valid[shuff, :] if self.pred_label_valid is not None else None
        self.pred_label_ensemble = self.pred_label_ensemble[shuff, :] if self.pred_label_ensemble is not None else None
        self.pred_label_test = self.pred_label_test[shuff, :] if self.pred_label_test is not None else None
        self.pred_logits_train = self.pred_logits_train[shuff, :] if self.pred_logits_train is not None else None
        self.pred_logits_valid = self.pred_logits_valid[shuff, :] if self.pred_logits_valid is not None else None
        self.pred_logits_ensemble = self.pred_logits_ensemble[shuff, :] if self.pred_logits_ensemble is not None else None
        self.pred_logits_test = self.pred_logits_test[shuff, :] if self.pred_logits_test is not None else None
        self.info_id = self.info_id[shuff, :] if self.info_id is not None else None
        self.family_list = self.family_list[shuff, :] if self.family_list is not None else None

    def __setitem__(self, index, pop):  # 种群个体赋值（种群个体替换）

        """
        描述: 种群个体的赋值
        用法: 假设pop是一个包含多于2个个体的种群矩阵，pop1是另一个包含2个个体的种群矩阵，那么
             pop[[0,1]] = pop1，即可完成将pop种群的第1、2个个体赋值为pop1种群的个体。
        注意: index必须为一个slice或者为一个Numpy array类型的行向量或者为一个list类型的列表或者为一个整数，
             该函数不对传入的index参数的合法性进行更详细的检查。
             此外，进行种群个体替换后，该函数不会对适应度进行主动重置，
             如果因个体替换而需要重新对所有个体的适应度进行评价，则需要手写代码更新种群的适应度。

        """

        # 对index进行格式处理
        if not isinstance(index, (slice, np.ndarray, list, int, np.int32, np.int64)):
            raise RuntimeError(
                'error in Population: index must be an integer, a 1-D list, or a 1-D array. (index必须是一个整数，一维的列表或者一维的向量。)')
        if isinstance(index, slice):
            index_array = index
        else:
            index_array = np.array(index).reshape(-1)
            if len(index_array) == 0:
                index_array = []
        if self.Encoding is not None:
            if self.Encoding != pop.Encoding:
                raise RuntimeError('error in Population: Encoding disagree. (两种群染色体的编码方式必须一致。)')
            if np.all(self.Field == pop.Field) == False:
                raise RuntimeError('error in Population: Field disagree. (两者的译码矩阵必须一致。)')
            if self.Chrom is None:
                raise RuntimeError('error in Population: Chrom is None. (种群染色体矩阵未初始化。)')
            self.Chrom[index_array] = copy.deepcopy(pop.Chrom)
        if self.ObjV is not None:
            if pop.ObjV is None:
                raise RuntimeError('error in Population: ObjV disagree. (两者的目标函数值矩阵必须要么同时为None要么同时不为None。)')
            self.ObjV[index_array] = pop.ObjV
        if self.FitnV is not None:
            if pop.FitnV is None:
                raise RuntimeError('error in Population: FitnV disagree. (两者的适应度列向量必须要么同时为None要么同时不为None。)')
            self.FitnV[index_array] = pop.FitnV

        if self.logits is not None:
            if pop.logits is None:
                raise RuntimeError('error in Population: logits disagree. (两者的logits向量必须要么同时为None要么同时不为None。)')
            self.logits[index_array] = pop.logits

        if self.pred_label_train is not None:
            if pop.pred_label_train is None:
                raise RuntimeError('error in Population: pred_label_train disagree. (两者的logits向量必须要么同时为None要么同时不为None。)')
            self.pred_label_train[index_array] = pop.pred_label_train

        if self.pred_label_valid is not None:
            if pop.pred_label_valid is None:
                raise RuntimeError('error in Population: pred_label_valid disagree. (两者的logits向量必须要么同时为None要么同时不为None。)')
            self.pred_label_valid[index_array] = pop.pred_label_valid

        if self.pred_label_ensemble is not None:
            if pop.pred_label_ensemble is None:
                raise RuntimeError('error in Population: pred_label_ensemble disagree. (两者的logits向量必须要么同时为None要么同时不为None。)')
            self.pred_label_ensemble[index_array] = pop.pred_label_ensemble

        if self.pred_label_test is not None:
            if pop.pred_label_test is None:
                raise RuntimeError('error in Population: logits disagree. (两者的logits向量必须要么同时为None要么同时不为None。)')
            self.pred_label_test[index_array] = pop.pred_label_test

        if self.pred_logits_train is not None:
            if pop.pred_logits_train is None:
                raise RuntimeError('error in Population: pred_label_train disagree. (两者的logits向量必须要么同时为None要么同时不为None。)')
            self.pred_logits_train[index_array] = pop.pred_logits_train

        if self.pred_logits_valid is not None:
            if pop.pred_logits_valid is None:
                raise RuntimeError('error in Population: pred_logits_valid disagree. (两者的logits向量必须要么同时为None要么同时不为None。)')
            self.pred_logits_valid[index_array] = pop.pred_logits_valid

        if self.pred_logits_ensemble is not None:
            if pop.pred_logits_ensemble is None:
                raise RuntimeError('error in Population: pred_logits_ensemble disagree. (两者的logits向量必须要么同时为None要么同时不为None。)')
            self.pred_logits_ensemble[index_array] = pop.pred_logits_ensemble

        if self.pred_logits_test is not None:
            if pop.pred_logits_test is None:
                raise RuntimeError('error in Population: logits disagree. (两者的logits向量必须要么同时为None要么同时不为None。)')
            self.pred_logits_test[index_array] = pop.pred_logits_test

        if self.CV is not None:
            if pop.CV is None:
                raise RuntimeError('error in Population: CV disagree. (两者的违反约束程度矩阵必须要么同时为None要么同时不为None。)')
            self.CV[index_array] = pop.CV
        if self.Phen is not None:
            if pop.Phen is None:
                raise RuntimeError('error in Population: Phen disagree. (两者的表现型矩阵必须要么同时为None要么同时不为None。)')
            self.Phen[index_array] = pop.Phen
        self.sizes = self.Phen.shape[0]  # 更新种群规模

    def __add__(self, pop):

        """
        描述: 种群个体合并。

        用法: 假设pop1, pop2是两个种群，它们的个体数可以相等也可以不相等，此时
             pop = pop1 + pop2，即可完成对pop1和pop2两个种群个体的合并。

        注意：
            进行种群合并后，该函数不会对适应度进行主动重置，
            如果因种群合并而需要重新对所有个体的适应度进行评价，则需要手写代码更新种群的适应度。

        """

        if self.Encoding is None:
            NewChrom = None
        else:
            if self.Encoding != pop.Encoding:
                raise RuntimeError('error in Population: Encoding disagree. (两种群染色体的编码方式必须一致。)')
            if self.Chrom is None or pop.Chrom is None:
                raise RuntimeError('error in Population: Chrom is None. (种群染色体矩阵未初始化。)')
            if np.all(self.Field == pop.Field) == False:
                raise RuntimeError('error in Population: Field disagree. (两者的译码矩阵必须一致。)')
            if self.isNN == 0:
                temp1 = copy.deepcopy(self.Chrom)
                temp2 = copy.deepcopy(pop.Chrom)
                NewChrom = np.vstack([temp1, temp2])
                NewPhen = np.vstack([self.Phen, pop.Phen])
            elif self.isNN == 1:
                temp1 = copy.deepcopy(self.Chrom)
                temp2 = copy.deepcopy(pop.Chrom)
                NewChrom = np.transpose(np.hstack([temp1, temp2]))
                NewPhen = np.transpose(np.hstack([self.Phen, pop.Phen]))
                infos = np.transpose(np.hstack([self.info_id, pop.info_id]))
                family_lists = self.family_list+pop.family_list
        NIND = self.sizes + pop.sizes  # 得到合并种群的个体数
        return Population(self.Encoding,
                          self.Field,
                          NIND,
                          NewChrom,
                          np.vstack([self.ObjV, pop.ObjV]) if self.ObjV is not None and pop.ObjV is not None else None,
                          np.vstack([self.ObjV_train, pop.ObjV_train]) if self.ObjV_train is not None and pop.ObjV_train is not None else None,
                          np.vstack([self.ObjV_valid, pop.ObjV_valid]) if self.ObjV_valid is not None and pop.ObjV_valid is not None else None,
                          np.vstack([self.ObjV_ensemble, pop.ObjV_ensemble]) if self.ObjV_ensemble is not None and pop.ObjV_ensemble is not None else None,
                          np.vstack([self.ObjV_test, pop.ObjV_test]) if self.ObjV_test is not None and pop.ObjV_test is not None else None,
                          np.vstack(
                              [self.FitnV, pop.FitnV]) if self.FitnV is not None and pop.FitnV is not None else None,
                          np.vstack([self.CV, pop.CV]) if self.CV is not None and pop.CV is not None else None,
                          NewPhen if self.Phen is not None and pop.Phen is not None else None,
                          parameters=self.parameters if self.parameters is not None and pop.parameters is not None else None,
                          logits=np.vstack(
                              [self.logits, pop.logits]) if self.logits is not None and pop.logits is not None else None,
                          pred_label_train=np.vstack(
                              [self.pred_label_train,
                               pop.pred_label_train]) if self.pred_label_train is not None and pop.pred_label_train is not None else None,
                          pred_label_valid=np.vstack(
                              [self.pred_label_valid,
                               pop.pred_label_valid]) if self.pred_label_valid is not None and pop.pred_label_valid is not None else None,
                          pred_label_ensemble=np.vstack(
                              [self.pred_label_ensemble,
                               pop.pred_label_ensemble]) if self.pred_label_ensemble is not None and pop.pred_label_ensemble is not None else None,
                          pred_label_test=np.vstack(
                              [self.pred_label_test,
                               pop.pred_label_test]) if self.pred_label_test is not None and pop.pred_label_test is not None else None,
                          pred_logits_train = np.vstack(
                                [self.pred_logits_train,
                                 pop.pred_logits_train]) if self.pred_logits_train is not None and pop.pred_logits_train is not None else None,
                          pred_logits_valid = np.vstack(
                                [self.pred_logits_valid,
                                 pop.pred_logits_valid]) if self.pred_logits_valid is not None and pop.pred_logits_valid is not None else None,
                          pred_logits_ensemble = np.vstack(
                                [self.pred_logits_ensemble,
                                 pop.pred_logits_ensemble]) if self.pred_logits_ensemble is not None and pop.pred_logits_ensemble is not None else None,
                          pred_logits_test = np.vstack(
                                [self.pred_logits_test,
                                 pop.pred_logits_test]) if self.pred_logits_test is not None and pop.pred_logits_test is not None else None,
                          info_id=infos if self.info_id is not None and pop.info_id is not None else None,
                          family_list=family_lists if self.family_list is not None and pop.family_list is not None else None,
                          )

    def __len__(self):

        """
        描述: 计算种群规模。

        用法: 假设pop是一个种群，那么len(pop)即可得到该种群的个体数。
             实际上，种群规模也可以通过pop.sizes得到。

        """

        return self.sizes

    def save_network(self, gen, save_dir, network=None):
        if network is None:
            count_NN = len(self.Chrom)
            network = copy.deepcopy(self.Chrom)
        else:
            count_NN = len(network)
        for idx in range(count_NN):
            NN = network[idx]
            save_filename = 'nets/gen%d_net%s.pth' % (gen, idx)
            save_path = os.path.join(save_dir, save_filename)
            torch.save(NN, save_path)

    # def save_pop(self, dir):  # NNmodel=population.Chrom
    #     NNmodels = copy.deepcopy(self.Chrom)
    #     for i in range(self.sizes):
    #         torch.save(NNmodels[i].state_dict(), dir + 'indiv_{}.pth'.format(str(i+1)))

    def save(self, dirName='Result', Gen=0, NNmodel=None, Res_metrics=None,
             All_objs_train=None, All_objs_valid=None, All_objs_ensemble=None, All_objs_test=None, true_y=None, poplogits=None, runtime=0):
        return 0
        # if self.sizes > 0:
        #     if self.isNN == 0:
        #         if not os.path.exists(dirName):
        #             os.makedirs(dirName)
        #         with open(dirName + '/Encoding.txt', 'w') as file:
        #             file.write(str(self.Encoding))
        #             file.close()
        #         if self.Encoding is not None:
        #             np.savetxt(dirName + '/Field.csv', self.Field, delimiter=',')
        #             np.savetxt(dirName + '/Chrom.csv', self.Chrom, delimiter=',')
        #         if self.ObjV is not None:
        #             np.savetxt(dirName + '/ObjV.csv', self.ObjV, delimiter=',')
        #         if self.FitnV is not None:
        #             np.savetxt(dirName + '/FitnV.csv', self.FitnV, delimiter=',')
        #         if self.CV is not None:
        #             np.savetxt(dirName + '/CV.csv', self.CV, delimiter=',')
        #         if self.Phen is not None:
        #             np.savetxt(dirName + '/Phen.csv', self.Phen, delimiter=',')
        #         print('种群信息导出完毕。')
        #     elif self.isNN == 1:
        #         if not os.path.exists(dirName):
        #             os.makedirs(dirName)
        #         # with open(dirName + '/Encoding.txt', 'w') as file:
        #         #     file.write(str(self.Encoding))
        #         #     file.close()
        #         if self.FitnV is not None:
        #             save_filename = '/others/Gen%d_FitnV.csv' % Gen
        #             np.savetxt(dirName + save_filename, self.FitnV, delimiter=',')
        #
        #         if poplogits is not None:
        #             save_filename = '/Gen%d_logits.csv' % Gen
        #
        #             with open(dirName + save_filename, 'a+') as file:
        #                 logits = np.array(poplogits)
        #                 true_y = true_y.reshape(1, -1)
        #                 line = str(runtime) + ','.join(str(x) for x in true_y[0]) + '\n'
        #                 file.write(line)
        #                 for rows in range(logits.shape[0]):
        #                     popobj = logits[rows, :]
        #                     line = ','.join(str(x) for x in popobj) + '\n'
        #                     file.write(line)
        #                 file.close()
        #
        #         # if poplogits is not None:
        #         #     save_filename = '/Gen%d_logits.csv' % Gen
        #         #
        #         #     with open(dirName + save_filename, 'a+') as file:
        #         #         logits = np.array(poplogits)
        #         #         true_y = true_y.reshape(1, -1)
        #         #         line = str(runtime) + ','.join(str(x) for x in true_y[0]) + '\n'
        #         #         file.write(line)
        #         #         for rows in range(logits.shape[0]):
        #         #             popobj = logits[rows, :]
        #         #             line = ','.join(str(x) for x in popobj) + '\n'
        #         #             file.write(line)
        #         #         file.close()
        #
        #         # if NNmodel is not None:
        #         #     self.save_network(Gen, dirName, network=NNmodel)
        #
        #         if All_objs_train is not None:
        #             save_filename = '/allobjs/ALL_Objs_train_gen%d_sofar.csv' % Gen
        #             record_gen = list(All_objs_train.keys())
        #             with open(dirName + save_filename, 'a+') as file:
        #                 for gen in record_gen:
        #                     popobj = All_objs_train[gen]
        #                     for i in range(popobj.shape[0]):
        #                         line = gen + ','
        #                         line += ','.join(str(x) for x in popobj[i, :]) + '\n'
        #                         file.write(line)
        #                 file.close()
        #
        #         if All_objs_valid is not None:
        #             save_filename = '/allobjs/ALL_Objs_valid_gen%d_sofar.csv' % Gen
        #             record_gen = list(All_objs_valid.keys())
        #             with open(dirName + save_filename, 'a+') as file:
        #                 for gen in record_gen:
        #                     popobj = All_objs_valid[gen]
        #                     for i in range(popobj.shape[0]):
        #                         line = gen + ','
        #                         line += ','.join(str(x) for x in popobj[i, :]) + '\n'
        #                         file.write(line)
        #                 file.close()
        #
        #         if All_objs_ensemble is not None:
        #             save_filename = '/allobjs/ALL_Objs_ensemble_gen%d_sofar.csv' % Gen
        #             record_gen = list(All_objs_ensemble.keys())
        #             with open(dirName + save_filename, 'a+') as file:
        #                 for gen in record_gen:
        #                     popobj = All_objs_ensemble[gen]
        #                     for i in range(popobj.shape[0]):
        #                         line = gen + ','
        #                         line += ','.join(str(x) for x in popobj[i, :]) + '\n'
        #                         file.write(line)
        #                 file.close()
        #
        #         if All_objs_test is not None:
        #             save_filename = '/allobjs/ALL_Objs_test_gen%d_sofar.csv' % Gen
        #             record_gen = list(All_objs_test.keys())
        #             with open(dirName + save_filename, 'a+') as file:
        #                 for gen in record_gen:
        #                     popobj = All_objs_test[gen]
        #                     for i in range(popobj.shape[0]):
        #                         line = gen + ','
        #                         line += ','.join(str(x) for x in popobj[i, :]) + '\n'
        #                         file.write(line)
        #                 file.close()

                # if poplogits is not None:
                #     save_filename = '/poplogits.csv'
                #     record_gen = list(poplogits.keys())
                #     with open(dirName + save_filename, 'a+') as file:
                #         for gen in record_gen:
                #             popobj = poplogits[gen]
                #             for i in range(popobj.shape[0]):
                #                 line = gen + ','
                #                 line += ','.join(str(x) for x in popobj[i, :]) + '\n'
                #                 file.write(line)
                #         file.close()
                    #
                    # all_sensitive_attributes = list(self.Metric['0']['0'].keys())
                    # for gen in record_gen:
                    #     nowmetric = self.Metric[gen]
                    #     for sens in all_sensitive_attributes:
                    #         all_colums_name = list(nowmetric[str(0)][all_sensitive_attributes[0]].keys())
                    #         line = ",sensitive attributes,"
                    #         # 写下metric的名字
                    #         for colum in all_colums_name:
                    #             line = line + colum + ','
                    #         file.write(line + "\n")
                    #
                    #         for idx in range(len(nowmetric)):
                    #             info = nowmetric[str(idx)]
                    #             line = "individual " + str(idx) + "," + sens
                    #             for colum in all_colums_name:
                    #                 line = line + ',' + str(info[sens][colum])
                    #             file.write(line + "\n")
                    #         file.write("\n")
                    #     file.close()

                # if Res_metrics is not None:
                #     save_filename = '/Gen%d_metric.csv' % gen
                #     all_sensitive_attributes = list(Res_metrics[str(0)].keys())
                #     with open(dirName + save_filename, 'a+') as file:
                #         for sens in all_sensitive_attributes:
                #             all_colums_name = list(Res_metrics[str(0)][all_sensitive_attributes[0]].keys())
                #             line = ",sensitive attributes,"
                #             # 写下metric的名字
                #             for colum in all_colums_name:
                #                 line = line + colum + ','
                #             file.write(line + "\n")
                #
                #             for idx in range(len(Res_metrics)):
                #                 info = Res_metrics[str(idx)]
                #                 line = "individual " + str(idx) + "," + sens
                #                 for colum in all_colums_name:
                #                     line = line + ',' + str(info[sens][colum])
                #                 file.write(line + "\n")
                #             file.write("\n")
                #         file.close()
                # print('种群信息导出完毕。')
    # moeas4NN added
    def printPare(self, test_org, record_parameter):
        dirName = self.parameters['dirName']
        start_time = self.parameters['start_time']
        if not os.path.exists(dirName + start_time):
            os.makedirs(dirName + start_time)
            os.makedirs(dirName + start_time + '/fulltrain')
            os.makedirs(dirName + start_time + '/nets')
            os.makedirs(dirName + start_time + '/allobjs')
            os.makedirs(dirName + start_time + '/others')
            os.makedirs(dirName + start_time + '/detect')
            os.makedirs(dirName + start_time + '/img')

        with open(dirName + start_time + '/Parameters.txt', 'a+') as file:
            for name in self.parameters:
                if name in record_parameter:
                    strname = name + ' : ' + str(self.parameters[name]) + '\n'
                    file.write(strname)
            file.close()

        test_org.to_csv(dirName + start_time + '/testdata.csv', index=None)


    # moeas4NN added
    def setisNN(self, isNN):
        self.isNN = isNN
        self.Encoding = 'NN'

    def set_indiv_logjts(self, logits):
        self.logits = logits
        print('set logit vector')
        return Population(self.Encoding,
                          self.Field,
                          self.sizes,
                          copy.deepcopy(self.Chrom),
                          self.ObjV,
                          self.ObjV_train,
                          self.ObjV_valid,
                          self.ObjV_ensemble,
                          self.ObjV_test,
                          self.FitnV,
                          self.CV,
                          self.Phen,
                          parameters=self.parameters if self.parameters is not None and self.parameters is not None else None,
                          logits=self.logits,
                          pred_label_train=self.pred_label_train,
                          pred_label_valid=self.pred_label_valid,
                          pred_label_ensemble=self.pred_label_ensemble,
                          pred_label_test=self.pred_label_test,
                          pred_logits_train=self.pred_logits_train,
                          pred_logits_valid=self.pred_logits_valid,
                          pred_logits_ensemble=self.pred_logits_ensemble)



    def get_indiv_logjts(self):
        return self.logits
