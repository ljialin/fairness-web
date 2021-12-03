# -*- coding: utf-8 -*-
import numpy as np
import geatpy as ea  # 导入geatpy库
from sys import path as paths
from os import path
from mvc.algo_cfg import STATUS
import torch
import time
import os
import copy

# from geatpy.plot_demo import plot_decision_boundary2
paths.append(path.split(path.split(path.realpath(__file__))[0])[0])


def save_model(population, gen, filepath='nets/gen%d_net%s.pth', start_time=None):
    for idx in range(len(population)):
        NN = population.Chrom[idx]
        save_filename = filepath % (gen, idx)
        save_path = os.path.join('Result/' + start_time, save_filename)
        # torch.save(NN, save_path)
        torch.save(NN.state_dict(), save_path)


class D_net(torch.nn.Module):
    """
    Adversary model as described in Zhang et al., Mitigating Unwanted Biases with Adversarial Learning.

    Args:
        input_dim (int): Number of input dimensions
        protected_dim (int): Number of dimensions for the protected variable
    """

    def __init__(self, input_dim, protected_dim, equality_of_odds=True):
        super(D_net, self).__init__()
        self.c = torch.nn.Parameter(torch.ones(1 * protected_dim))
        if equality_of_odds:
            self.no_targets = 3
        else:
            self.no_targets = 1
        self.w2 = torch.nn.init.xavier_uniform_(
            torch.nn.Parameter(torch.empty(self.no_targets * input_dim, 1 * protected_dim)))
        self.b = torch.nn.Parameter(torch.zeros(1 * protected_dim))
        if equality_of_odds:
            self.name = 'equality of odds'
        else:
            self.name = 'demographic parity'

    def forward(self, logits, targets):
        s = torch.sigmoid((1 + torch.abs(self.c)) * logits)
        if self.no_targets == 3:
            z_hat = torch.cat((s, s * targets, s * (1 - targets)), dim=1) @ self.w2 + self.b
        else:
            z_hat = s @ self.w2 + self.b
        return z_hat, torch.sigmoid(z_hat)


class D_net2(torch.nn.Module):
    """
    Adversary model as described in Zhang et al., Mitigating Unwanted Biases with Adversarial Learning.

    Args:
        input_dim (int): Number of input dimensions
        protected_dim (int): Number of dimensions for the protected variable
    """

    def __init__(self, input_dim, protected_dim, equality_of_odds=True):
        super(D_net2, self).__init__()
        self.c = torch.nn.Parameter(torch.ones(1 * protected_dim))
        self.hidden_nodes = 10
        if equality_of_odds:
            self.no_targets = 3
        else:
            self.no_targets = 1
        self.w2 = torch.nn.init.xavier_uniform_(
            torch.nn.Parameter(torch.empty(self.no_targets * input_dim, 1 * self.hidden_nodes)))
        self.b = torch.nn.Parameter(torch.zeros(1 * self.hidden_nodes))
        self.hidden = torch.nn.Linear(self.hidden_nodes, protected_dim)
        if equality_of_odds:
            self.name = 'equality of odds'
        else:
            self.name = 'demographic parity'

    def forward(self, logits, targets):
        s = torch.sigmoid((1 + torch.abs(self.c)) * logits)
        if self.no_targets == 3:
            z_hat = torch.cat((s, s * targets, s * (1 - targets)), dim=1) @ self.w2 + self.b
        else:
            z_hat = s @ self.w2 + self.b

        z_hat = self.hidden(z_hat)
        return z_hat, torch.sigmoid(z_hat)


def k_tournament(k, fitness, num):
    selection_idxs = []
    fitness = fitness.reshape(1, -1)
    N = len(fitness[0])

    for i in range(num):
        cmps = np.random.randint(0, N, k)
        fits = fitness[0, cmps]
        best_idx = np.argmax(fits)
        selection_idxs.append(cmps[best_idx])

    return selection_idxs


class moea_templet_more_objectives(ea.MoeaAlgorithm):
    """
moea_NSGA2_templet : class - 多目标进化NSGA-II算法模板
    
算法描述:
    采用NSGA-II进行多目标优化，算法详见参考文献[1]。

参考文献:
    [1] Deb K , Pratap A , Agarwal S , et al. A fast and elitist multiobjective 
    genetic algorithm: NSGA-II[J]. IEEE Transactions on Evolutionary 
    Computation, 2002, 6(2):0-197.

    """

    def __init__(self, problem, start_time, population, muta_mu=0, muta_var=0.001, objectives=None, kfold=0,
                 calculmetric=20, ctrlr=None,
                 run_id=0, use_GAN=False, dropout=0, MOEAs=6, mutation_p=0.2, crossover_p=0.8, lr_decay_factor=0.99,
                 record_parameter=None, is_ensemble=False):
        ea.MoeaAlgorithm.__init__(self, problem, population)  # 先调用父类构造方法
        if objectives is None:
            objectives = ['BCE_loss', 'Individual_fairness']
        # if population.ChromNum != 1:
        #     raise RuntimeError('传入的种群对象必须是单染色体的种群类型。')
        # self.name = 'NSGA2'
        if self.problem.M < 10:
            self.ndSort = ea.ndsortESS  # 采用ENS_SS进行非支配排序
        else:
            self.ndSort = ea.ndsortTNS  # 高维目标采用T_ENS进行非支配排序，速度一般会比ENS_SS要快
        # self.selFunc = 'tour'  # 选择方式，采用锦标赛选择
        # if population.Encoding == 'P':
        #     self.recOper = ea.Xovpmx(XOVR=1)  # 生成部分匹配交叉算子对象
        #     self.mutOper = ea.Mutinv(Pm=1)  # 生成逆转变异算子对象
        # elif population.Encoding == 'BG':
        #     self.recOper = ea.Xovud(XOVR=1)  # 生成均匀交叉算子对象
        #     self.mutOper = ea.Mutbin(Pm=None)  # 生成二进制变异算子对象，Pm设置为None时，具体数值取变异算子中Pm的默认值
        # elif population.Encoding == 'RI':
        #     self.recOper = ea.Recsbx(XOVR=1, n=20)  # 生成模拟二进制交叉算子对象
        #     self.mutOper = ea.Mutpolyn(Pm=1 / self.problem.Dim, DisI=20)  # 生成多项式变异算子对象
        # # -------- ZQQ - begin -----------
        # elif population.Encoding == 'NN':
        self.recOper = ea.Crossover_NN(crossover_p)
        self.mutOper = ea.Mutation_NN(mu=muta_mu, var=muta_var, p=mutation_p)
        # # -------- ZQQ - end -----------
        # else:
        #     raise RuntimeError('编码方式必须为''BG''、''RI''、''P''或''NN''.')
        self.start_time = start_time
        self.dirName = 'Result/' + self.start_time
        self.objectives_class = objectives
        self.kfold = kfold
        self.calculmetric = calculmetric
        self.run_id = run_id  # 就是前端的task_id
        self.use_GAN = use_GAN
        self.dropout = dropout
        # self.MOEAs = MOEAs
        self.MOEAs = {'SRA': 6, "NSGA-II": 4}[ctrlr.cfg.optimizer]
        self.mutation_p = mutation_p
        self.crossover_p = crossover_p
        self.muta_mu = muta_mu
        self.muta_var_org = muta_var
        self.muta_var = muta_var
        self.lr_decay_factor = lr_decay_factor
        self.record_parameter = record_parameter
        self.is_ensemble = is_ensemble
        self.ctrlr = ctrlr

    def reinsertion(self, population, offspring, NUM, isNN=1):

        """
        描述:
            重插入个体产生新一代种群（采用父子合并选择的策略）。
            NUM为所需要保留到下一代的个体数目。
            注：这里对原版NSGA-II进行等价的修改：先按帕累托分级和拥挤距离来计算出种群个体的适应度，
            然后调用dup选择算子(详见help(ea.dup))来根据适应度从大到小的顺序选择出个体保留到下一代。
            这跟原版NSGA-II的选择方法所得的结果是完全一样的。
        """

        # 父子两代合并
        population = population + offspring
        population.setisNN(1)
        # 选择个体保留到下一代
        # [levels, criLevel] = self.ndSort(population.ObjV, NUM, None, population.CV,
        #                                  self.problem.maxormins)  # 对NUM个个体进行非支配分层
        # dis = ea.crowdis(population.ObjV, levels)  # 计算拥挤距离
        # population.FitnV[:, 0] = np.argsort(np.lexsort(np.array([dis, -levels])), kind='mergesort')  # 计算适应度
        # chooseFlag = ea.selecting('dup', population.FitnV, NUM)  # 调用低级选择算子dup进行基于适应度排序的选择，保留NUM个个体
        if self.MOEAs == 1:
            # orginal SDE-MOEA
            ObjV = copy.deepcopy(population.ObjV)
            chooseFlag = ea.SDE_env_selection1(ObjV, NUM)
        elif self.MOEAs == 2:
            # SDE-MOEA + extreme
            ObjV = copy.deepcopy(population.ObjV)
            chooseFlag = ea.SDE_env_selection2(ObjV, NUM)
        elif self.MOEAs == 3:
            # SDE-MOEA + extreme + uniformity
            ObjV = copy.deepcopy(population.ObjV)
            chooseFlag = ea.SDE_env_selection3(ObjV, NUM)
        elif self.MOEAs == 4:
            # NSGA--II
            [levels, criLevel] = self.ndSort(population.ObjV, NUM, None, population.CV,
                                             self.problem.maxormins)  # 对NUM个个体进行非支配分层
            dis = ea.crowdis(population.ObjV, levels)  # 计算拥挤距离
            population.FitnV[:, 0] = np.argsort(np.lexsort(np.array([dis, -levels])), kind='mergesort')  # 计算适应度
            chooseFlag = ea.selecting('dup', population.FitnV, NUM)  # 调用低级选择算子dup进行基于适应度排序的选择，保留NUM个个体
        elif self.MOEAs == 5:
            # single-objective
            objs = copy.deepcopy(population.ObjV)
            population.FitnV[:, 0] = -objs[:, 0]  # 计算适应度
            chooseFlag_sort1 = np.argsort(objs[:, 0])
            chooseFlag_sort = np.argsort(chooseFlag_sort1)
            chooseFlag_idx = np.where(chooseFlag_sort < NUM)
            chooseFlag = chooseFlag_idx[0]
        elif self.MOEAs == 6:
            # SRA
            ObjV = copy.deepcopy(population.ObjV)
            chooseFlag = ea.SRA_env_selection(ObjV, NUM)
        elif self.MOEAs == 7:
            # SRA + extreme
            [levels, criLevel] = self.ndSort(population.ObjV, NUM, None, population.CV,
                                             self.problem.maxormins)  # 对NUM个个体进行非支配分层
            ObjV = copy.deepcopy(population.ObjV)
            chooseFlag = ea.SRA_env_selection2(ObjV, NUM, levels)
        elif self.MOEAs == 8:
            # SDE-MOEA + extreme + randomselection
            ObjV = copy.deepcopy(population.ObjV)
            chooseFlag = ea.SDE_env_selection2(ObjV, NUM)
        elif self.MOEAs == 9:
            # SDE-MOEA + extreme + uniformity + randomselection
            ObjV = copy.deepcopy(population.ObjV)
            chooseFlag = ea.SDE_env_selection3(ObjV, NUM)
        else:
            ObjV = copy.deepcopy(population.ObjV)
            [levels, criLevel] = self.ndSort(population.ObjV, NUM, None, population.CV,
                                             self.problem.maxormins)  # 对NUM个个体进行非支配分层
            chooseFlag = ea.SRA_env_selection3(ObjV, NUM, levels)

        return population[chooseFlag], chooseFlag

    def update_passtime(self):
        self.passTime += time.time() - self.timeSlot

    def update_timeslot(self):
        self.timeSlot = time.time()

    def run(self, prophetPop=None):  # prophetPop为先知种群（即包含先验知识的种群）
        # True -> equality_of_odds
        # False -> demographic
        # Adversity_EO = D_net(1, 1, True)
        # Adversity_EO2 = D_net(1, 1, True)
        # Adversity_DE = D_net(1, 1, False)
        # Adversity_DE2 = D_net(1, 1, False)

        Adversity_EO = D_net2(1, 1, True)
        Adversity_EO2 = D_net2(1, 1, True)
        Adversity_DE = D_net2(1, 1, False)
        Adversity_DE2 = D_net2(1, 1, False)

        adv_models = []
        adv_models.append(Adversity_EO)
        adv_models.append(Adversity_DE)

        # self.population.printPare(self.problem.test_org, self.record_parameter)
        # self.population.printPare(self.problem.test_org)
        # ==========================初始化配置===========================
        population = self.population
        # cmp_population = population.copy()
        # self.train_nets(cmp_population, Gen=0, epoch=1, iscal_metric=1)
        NIND = population.sizes
        self.problem.do_pre()
        self.initialization(is_ensemble=self.is_ensemble)  # 初始化算法模板的一些动态参数
        # self.initialization()  # 初始化算法模板的一些动态参数
        # ===========================准备进化============================

        # 这里可以把个体传入
        population.initChrom(dataname=self.problem.dataname, dropout=self.dropout)  # 初始化种群染色体矩阵

        # self.problem.model_test1(population, use_GAN=self.use_GAN, adv_model_EO=Adversity_EO, adv_model_DE=Adversity_DE, dirName='Result/' + self.start_time, problem=self.problem)

        # self.problem.model_test2(population, use_GAN=self.use_GAN, adv_models=adv_models,
        #                          dirName='Result/' + self.start_time, problem=self.problem)
        # return self.finishing(population)

        # self.problem.model_test4(population, dirName='Result/' + self.start_time, problem=self.problem)

        self.update_passtime()
        pop_idx_count = 0
        for pop_idx in range(len(population)):
            population.info_id[pop_idx] = pop_idx_count
            pop_idx_count += 1
            population.family_list[pop_idx].append(-1)
        # population.save(dirName='Result/' + self.start_time, Gen=-1, NNmodel=population.Chrom)  # 打印最开始初始化时的网络参数
        population.save_pop(self.ctrlr.get_savepop_dir())  # 保存神经网络的权值
        self.update_timeslot()
        gen = 1
        if self.use_GAN:
            if np.random.random() < 0:
                self.call_aimFunc_GAN(population, kfold=self.kfold, adversary=Adversity_EO, gen=0,
                                      dirName='Result/' + self.start_time)  # 求进化后个体的目标函数值
                print('run GAN')
            else:
                self.call_aimFunc(population, kfold=self.kfold, gen=gen,
                                  dirName='Result/' + self.start_time)  # 计算种群的目标函数值 计算目标值时，kfold!=0时不修改原网络参数，只是试探性的kfold后计算，kfold=0时会修改值
                print('run no GAN')
        else:
            self.call_aimFunc(population, kfold=self.kfold, gen=gen, dirName='Result/' + self.start_time,
                              lr_decay_factor=self.lr_decay_factor)  # 计算种群的目标函数值 计算目标值时，kfold!=0时不修改原网络参数，只是试探性的kfold后计算，kfold=0时会修改值
        # return self.finishing(population)
        # self.train_nets(population, Gen=-1, epoch=1, iscal_metric=1, changeNets=0,
        #                 problem=self.problem, runtime=self.passTime)
        ##################################
        # return self.finishing(population)
        # # 插入先验知识（注意：这里不会对先知种群prophetPop的合法性进行检查，故应确保prophetPop是一个种群类且拥有合法的Chrom、ObjV、Phen等属性）
        # if prophetPop is not None:
        #     population = (prophetPop + population)[:NIND]  # 插入先知种群
        #################################
        if self.MOEAs == 4 | self.MOEAs == 5:
            [levels, criLevel] = self.ndSort(population.ObjV, NIND, None, population.CV,
                                             self.problem.maxormins)  # 对NIND个个体进行非支配分层
            population.FitnV = (1 / levels).reshape(-1, 1)  # 直接根据levels来计算初代个体的适应度

        # self.detect_differneces(population, gen, datatype='test', problem=self.problem)  # 记录
        # plot_decision_boundary2(population, self.problem, dirName='Result/' + self.start_time, gen=gen,
        #                         ndSort=self.ndSort)
        # np.savetxt('Result/' + self.start_time + '/detect/pred_label_valid_gen{}.txt'.format(gen),
        #            population.pred_label_valid)
        # np.savetxt('Result/' + self.start_time + '/detect/pred_label_train_gen{}.txt'.format(gen),
        #            population.pred_label_train)
        # np.savetxt('Result/' + self.start_time + '/detect/pred_label_test_gen{}.txt'.format(gen),
        #            population.pred_label_test)

        # self.update_passtime()
        # # plot_decision_boundary2(population, self.problem, dirName='Result/' + self.start_time, gen=gen, ndSort=self.ndSort)
        # np.savetxt('Result/' + self.start_time + '/detect/pred_label_valid_gen{}.txt'.format(gen), population.pred_logits_valid)
        # np.savetxt('Result/' + self.start_time + '/detect/pred_label_train_gen{}.txt'.format(gen), population.pred_logits_train)
        # np.savetxt('Result/' + self.start_time + '/detect/pred_label_test_gen{}.txt'.format(gen), population.pred_logits_test)
        # self.update_timeslot()
        self.update_passtime()
        self.add_gen2info(population, gen)
        self.update_timeslot()
        Archive = copy.deepcopy(population)

        self.ctrlr.save_fitness(population.ObjV_valid, gen=gen)
        if self.handle_status(gen):
            return self.finishing(population)
        population.FitnV = np.ones([len(population), 1])
        # ===========================开始进化============================
        while not self.terminated(population):

            self.muta_var = self.muta_var_org - gen * (self.muta_var_org * 0.9) / self.MAXGEN
            self.mutOper = ea.Mutation_NN(mu=self.muta_mu, var=self.muta_var, p=self.mutation_p)
            gen += 1
            # print('Gen', gen)
            # #################选择个体参与进化############################################################################
            #
            K_num = 2
            if "Individual_fairness" in self.problem.objectives_class or "Group_fairness" in self.problem.objectives_class:
                MOEA_sel_num = NIND - K_num
            else:
                MOEA_sel_num = NIND - K_num
            if self.MOEAs == 1:
                # orginal SDE-MOEA
                # better_parents = ea.selecting(self.selFunc, ea.SDE_parent_selection1(population.ObjV), MOEA_sel_num)
                better_parents = k_tournament(2, ea.SDE_parent_selection1(population.ObjV), MOEA_sel_num)
            elif self.MOEAs == 2:
                # SDE-MOEA + extreme
                # better_parents = ea.selecting(self.selFunc, ea.SDE_parent_selection2(population.ObjV), MOEA_sel_num)
                better_parents = k_tournament(2, SDE_parent_selection2(population.ObjV), MOEA_sel_num)
            elif self.MOEAs == 3:
                # SDE-MOEA + extreme + uniformity
                # better_parents = ea.selecting(self.selFunc, ea.SDE_parent_selection3(population.ObjV), MOEA_sel_num)
                better_parents = k_tournament(2, ea.SDE_parent_selection3(population.ObjV), MOEA_sel_num)
            elif self.MOEAs == 4:
                # NSGA-II
                # better_parents = ea.selecting(self.selFunc, population.FitnV, MOEA_sel_num)
                better_parents = k_tournament(2, population.FitnV, MOEA_sel_num)
            elif self.MOEAs == 5:
                # single-objective
                acc_obj = -population.ObjV[:, 0]
                # better_parents = ea.selecting(self.selFunc, acc_obj.reshape(-1, 1), MOEA_sel_num)
                better_parents = k_tournament(2, acc_obj.reshape(-1, 1), MOEA_sel_num)
            elif self.MOEAs == 6:
                # SRA
                better_parents = list(np.random.randint(0, NIND, MOEA_sel_num))
            elif self.MOEAs == 7:
                # SRA + extreme
                better_parents = list(np.random.randint(0, NIND, MOEA_sel_num))
            elif self.MOEAs == 8:
                # SDE-MOEA + extreme + randomselection
                # better_parents = ea.selecting(self.selFunc, ea.SDE_parent_selection2(population.ObjV), MOEA_sel_num)
                better_parents = list(np.random.randint(0, NIND, MOEA_sel_num))
            elif self.MOEAs == 9:
                # SDE-MOEA + extreme + uniformity + randomselection
                # better_parents = ea.selecting(self.selFunc, ea.SDE_parent_selection2(population.ObjV), MOEA_sel_num)
                better_parents = list(np.random.randint(0, NIND, MOEA_sel_num))
            else:
                better_parents = list(np.random.randint(0, NIND, MOEA_sel_num))

            # make sure the individuals with the extreme(best) of each objective can be selected to generate offspring
            extreme = np.argmin(population.ObjV, axis=0)
            # for te in extreme:
            #     better_parents.append(te)

            offspring_better = population[better_parents].copy()
            offsprint_extrme = population[extreme].copy()
            offsprint_extrme_temp = population[extreme].copy()
            offsprint_extrme_temp2 = population[extreme].copy()
            # ##########################################################################################################

            # #################对选出的个体进行进化操作######################################################################
            # use_crossover = True
            # if any(offsprint_extrme_temp.ObjV[:, 1] < 0.065):
            #     for rep in range(100):
            #         if use_crossover:
            #             for i in offsprint_extrme_temp2.ObjV:
            #                 print(i[0], i[1])
            #             offsprint_extrme_temp2.Chrom[0:1] = self.recOper.do([offsprint_extrme_temp.Chrom[0]], [offsprint_extrme_temp.Chrom[1]], rep/100)
            #             self.call_aimFunc(offsprint_extrme_temp2, kfold=self.kfold, gen=gen, dirName='Result/' + self.start_time, loss_type=['individual', 'individual'], train_net=0)
            #
            #     offsprint_extrme_temp2 = population[extreme].copy()
            #     for rep in range(100):
            #         if use_crossover:
            #             for i in offsprint_extrme_temp2.ObjV:
            #                 print(i[0], i[1])
            #             offsprint_extrme_temp2.Chrom[0:1] = self.recOper.do([offsprint_extrme_temp.Chrom[0]], [offsprint_extrme_temp.Chrom[1]], rep/100)
            #             self.call_aimFunc(offsprint_extrme_temp2, kfold=self.kfold, gen=gen, dirName='Result/' + self.start_time, loss_type=['Error', 'Error'], train_net=1)
            #
            #     offsprint_extrme_temp2 = population[extreme].copy()
            #     for rep in range(100):
            #         if use_crossover:
            #             for i in offsprint_extrme_temp2.ObjV:
            #                 print(i[0], i[1])
            #             offsprint_extrme_temp2.Chrom[0:1] = self.recOper.do([offsprint_extrme_temp.Chrom[0]], [offsprint_extrme_temp.Chrom[1]], rep/100)
            #             self.call_aimFunc(offsprint_extrme_temp2, kfold=self.kfold, gen=gen, dirName='Result/' + self.start_time, loss_type=['individual', 'individual'], train_net=1)
            #
            #

            if self.crossover_p > 0:
                offspring_better.Chrom[0:np.int(np.floor(MOEA_sel_num / 2) * 2)] = self.recOper.do(
                    offspring_better.Chrom[0:np.int(np.floor(MOEA_sel_num / 2))],
                    offspring_better.Chrom[np.int(np.floor(MOEA_sel_num / 2)):np.int(np.floor(MOEA_sel_num / 2) * 2)],
                    np.random.uniform(0, 1, 1))  # 重组
            offspring_better.Chrom = self.mutOper.do(offspring_better.Chrom)  # 变异
            self.call_aimFunc(offspring_better, kfold=self.kfold, gen=gen, dirName='Result/' + self.start_time,
                              loss_type=-1, train_net=1, lr_decay_factor=self.lr_decay_factor)
            offspring = offspring_better.copy()

            if gen < self.MAXGEN and len(self.problem.objectives_class) > 1:
                if np.random.random() < 0.2:
                    offsprint_extrme.Chrom = self.mutOper.do(offsprint_extrme.Chrom)  # 变异
                for k in range(K_num):
                    # if np.random.random() < 0.5:
                    # offsprint_extrme.Chrom = self.mutOper.do(offsprint_extrme.Chrom)  # 变异
                    if "Individual_fairness" in self.problem.objectives_class or "Group_fairness" in self.problem.objectives_class:
                        self.call_aimFunc(offsprint_extrme, kfold=self.kfold, gen=gen,
                                          dirName='Result/' + self.start_time, loss_type=self.problem.objectives_class,
                                          lr_decay_factor=self.lr_decay_factor)
                    else:
                        self.call_aimFunc(offsprint_extrme[0], kfold=self.kfold, gen=gen,
                                          dirName='Result/' + self.start_time,
                                          loss_type=self.problem.objectives_class[0],
                                          lr_decay_factor=self.lr_decay_factor)
                    offspring = offspring + offsprint_extrme[0]

            record_process_MOEAs = 0
            if record_process_MOEAs == 1:
                self.update_passtime()
                # print better_parents
                np.savetxt('Result/' + self.start_time + '/detect/better_parents_gen{}.txt'.format(gen), better_parents)

                # print offspring
                np.savetxt('Result/' + self.start_time + '/detect/offspring_test_gen{}.txt'.format(gen),
                           offspring.ObjV_test)
                np.savetxt('Result/' + self.start_time + '/detect/offspring_valid_gen{}.txt'.format(gen),
                           offspring.ObjV_valid)
                np.savetxt('Result/' + self.start_time + '/detect/offspring_train_gen{}.txt'.format(gen),
                           offspring.ObjV_train)

                # print old population
                np.savetxt('Result/' + self.start_time + '/detect/oldpop_test_gen{}.txt'.format(gen),
                           population.ObjV_test)
                np.savetxt('Result/' + self.start_time + '/detect/oldpop_valid_gen{}.txt'.format(gen),
                           population.ObjV_valid)
                np.savetxt('Result/' + self.start_time + '/detect/oldpop_train_gen{}.txt'.format(gen),
                           population.ObjV_train)
                self.update_timeslot()
            # #################重插入生成新一代种群#########################################################################
            population, chooseidx = self.reinsertion(population, offspring, NIND, isNN=1)

            # Archive = Archive + population
            # [levels, criLevel] = self.ndSort(Archive.ObjV, len(Archive), None, Archive.CV,
            #                                  self.problem.maxormins)  # 对NUM个个体进行非支配分层
            # Archive = Archive[np.where(levels == 1)[0]]
            # ##########################################################################################################

            if record_process_MOEAs == 1:
                self.update_passtime()
                # print new population
                np.savetxt('Result/' + self.start_time + '/detect/newpop_test_gen{}.txt'.format(gen),
                           population.ObjV_test)
                np.savetxt('Result/' + self.start_time + '/detect/newpop_valid_gen{}.txt'.format(gen),
                           population.ObjV_valid)
                np.savetxt('Result/' + self.start_time + '/detect/newpop_train_gen{}.txt'.format(gen),
                           population.ObjV_train)

                # print better individuals
                np.savetxt('Result/' + self.start_time + '/detect/better_individuals_gen{}.txt'.format(gen), chooseidx)
                # self.q.put('演化中... ({}%)'.format(str(round(gen * 100 / self.MAXGEN, 2))))
                self.update_timeslot()

            # if np.mod(gen, 10) == 0:
            #     self.update_passtime()
            #     # plot_decision_boundary2(population, self.problem, dirName='Result/' + self.start_time, gen=gen, ndSort=self.ndSort)
            #     np.savetxt('Result/'+self.start_time+'/detect/pred_label_valid_gen{}.txt'.format(gen), population.pred_logits_valid)
            #     np.savetxt('Result/'+self.start_time+'/detect/pred_label_train_gen{}.txt'.format(gen), population.pred_logits_train)
            #     np.savetxt('Result/'+self.start_time+'/detect/pred_label_test_gen{}.txt'.format(gen), population.pred_logits_test)
            #     self.update_timeslot()
            self.update_passtime()
            self.add_gen2info(population, gen)
            self.update_timeslot()
            # print(population.ObjV_valid)

            self.ctrlr.save_fitness(population.ObjV_valid, gen=gen) #保存适应度
            population.save_pop(self.ctrlr.get_savepop_dir()) # 保存神经网络
            if self.handle_status(gen):
                return self.finishing(population)

            if np.mod(gen, self.calculmetric) == 0 or gen == 1:
                self.update_passtime()
                population.save(dirName='Result/' + self.start_time, Gen=gen, NNmodel=population,
                                All_objs_train=self.all_objetives_train,
                                All_objs_valid=self.all_objetives_valid,
                                All_objs_test=self.all_objetives_test,
                                All_objs_ensemble=self.all_objetives_ensemble,
                                true_y=np.array(self.problem.test_y),
                                runtime=self.passTime)

                # save_model(population, gen, filepath='nets/gen%d_net%s.pth', start_time=self.start_time)

                # save_model(Archive, gen, filepath='nets/archive_gen%d_net%s.pth', start_time=self.start_time)

                # for idx in range(NIND):
                #     NN = population.Chrom[idx]
                #     save_filename = 'nets/gen%d_net%s.pth' % (gen, idx)
                #     save_path = os.path.join('Result/' + self.start_time, save_filename)
                #     # torch.save(NN, save_path)
                #     torch.save(NN.state_dict(), save_path)

                now_time = time.strftime("%d %H.%M:%S", time.localtime(time.time()))
                # self.detect_differneces(population, gen, datatype='test', problem=self.problem)  # 记录
                print(now_time, " Run ID ", self.run_id, ", Gen :", gen, ", runtime:", str(self.passTime))
                # Res_metrics = self.get_metric(population)
                # np.savetxt('Result/' + self.start_time + '/detect/passtime_gen{}.txt'.format(gen), np.array([self.passTime]))
                ####
                # record info of population
                # record the predict probility
                # np.savetxt('Result/' + self.start_time + '/detect/pop_logits_train{}.txt'.format(gen),
                #            population.pred_logits_train)
                # np.savetxt('Result/' + self.start_time + '/detect/pop_logits_valid{}.txt'.format(gen),
                #            population.pred_logits_valid)
                # if self.is_ensemble:
                #     np.savetxt('Result/' + self.start_time + '/detect/pop_logits_ensemble{}.txt'.format(gen),
                #                population.pred_logits_ensemble)
                # np.savetxt('Result/' + self.start_time + '/detect/pop_logits_test{}.txt'.format(gen),
                #            population.pred_logits_test)
                #
                # np.savetxt('Result/' + self.start_time + '/detect/popobj_train{}.txt'.format(gen),
                #            population.ObjV_train)
                # np.savetxt('Result/' + self.start_time + '/detect/popobj_valid{}.txt'.format(gen),
                #            population.ObjV_valid)
                # if self.is_ensemble:
                #     np.savetxt('Result/' + self.start_time + '/detect/popobj_ensemble{}.txt'.format(gen),
                #                population.ObjV_ensemble)
                # np.savetxt('Result/' + self.start_time + '/detect/popobj_test{}.txt'.format(gen),
                #            population.ObjV_test)

                ####
                # # record info of Archive
                # np.savetxt('Result/' + self.start_time + '/detect/Archive_logits_train{}.txt'.format(gen),
                #            Archive.pred_logits_train)
                # np.savetxt('Result/' + self.start_time + '/detect/Archive_logits_valid{}.txt'.format(gen),
                #            Archive.pred_logits_valid)
                # if self.is_ensemble:
                #     np.savetxt('Result/' + self.start_time + '/detect/Archive_logits_ensemble{}.txt'.format(gen),
                #                Archive.pred_logits_ensemble)
                # np.savetxt('Result/' + self.start_time + '/detect/Archive_logits_test{}.txt'.format(gen),
                #            Archive.pred_logits_test)
                #
                # np.savetxt('Result/' + self.start_time + '/detect/Archiveobj_train{}.txt'.format(gen),
                #            Archive.ObjV_train)
                # np.savetxt('Result/' + self.start_time + '/detect/Archiveobj_valid{}.txt'.format(gen),
                #            Archive.ObjV_valid)
                # if self.is_ensemble:
                #     np.savetxt('Result/' + self.start_time + '/detect/Archiveobj_ensemble{}.txt'.format(gen),
                #                Archive.ObjV_ensemble)
                # np.savetxt('Result/' + self.start_time + '/detect/Archiveobj_test{}.txt'.format(gen),
                #            Archive.ObjV_test)

                self.update_timeslot()

        # np.savetxt('Result/' + self.start_time + '/detect/passtime.txt', np.array([self.passTime]))
        population.save_pop(self.ctrlr.get_savepop_dir())
        # population.save(dirName='Result/' + self.start_time, Gen=gen, NNmodel=population,
        #                 All_objs_train=self.all_objetives_train,
        #                 All_objs_valid=self.all_objetives_valid,
        #                 All_objs_test=self.all_objetives_test,
        #                 true_y=np.array(self.problem.test_y),
        #                 runtime=self.passTime)

        self.ctrlr.update_progress(STATUS.FINISH)
        print("Run ID ", self.run_id, "finished!")
        return self.finishing(population)  # 调用finishing完成后续工作并返回结果

    def handle_status(self, gen):
        print('curr_status', self.ctrlr.status)
        if self.ctrlr.status == STATUS.ABORT:
            self.ctrlr.update_progress(STATUS.FINISH, gen=gen, maxgen=self.MAXGEN)
            return 1
        elif self.ctrlr.status == STATUS.PAUSE:
            self.ctrlr.status = STATUS.PAUSED
            while self.ctrlr.status != STATUS.RUNNING:
                time.sleep(1)
            self.ctrlr.update_progress(STATUS.RUNNING, gen=gen, maxgen=self.MAXGEN)
        else:
            self.ctrlr.update_progress(STATUS.RUNNING, gen=gen, maxgen=self.MAXGEN)
        return 0
