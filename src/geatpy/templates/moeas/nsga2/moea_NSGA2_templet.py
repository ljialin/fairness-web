# -*- coding: utf-8 -*-
import numpy as np
import geatpy as ea  # 导入geatpy库
from sys import path as paths
from os import path
import torch
import time
paths.append(path.split(path.split(path.realpath(__file__))[0])[0])


class moea_NSGA2_templet(ea.MoeaAlgorithm):
    """
moea_NSGA2_templet : class - 多目标进化NSGA-II算法模板
    
算法描述:
    采用NSGA-II进行多目标优化，算法详见参考文献[1]。

参考文献:
    [1] Deb K , Pratap A , Agarwal S , et al. A fast and elitist multiobjective 
    genetic algorithm: NSGA-II[J]. IEEE Transactions on Evolutionary 
    Computation, 2002, 6(2):0-197.

    """

    def __init__(self, problem, start_time, population, muta_mu=0, muta_var=0.001, objectives=None, kfold=0, calculmetric=20,
                 run_id=0):
        ea.MoeaAlgorithm.__init__(self, problem, population)  # 先调用父类构造方法
        if objectives is None:
            objectives = ['individual', 'group']
        if population.ChromNum != 1:
            raise RuntimeError('传入的种群对象必须是单染色体的种群类型。')
        self.name = 'NSGA2'
        if self.problem.M < 10:
            self.ndSort = ea.ndsortESS  # 采用ENS_SS进行非支配排序
        else:
            self.ndSort = ea.ndsortTNS  # 高维目标采用T_ENS进行非支配排序，速度一般会比ENS_SS要快
        self.selFunc = 'tour'  # 选择方式，采用锦标赛选择
        if population.Encoding == 'P':
            self.recOper = ea.Xovpmx(XOVR=1)  # 生成部分匹配交叉算子对象
            self.mutOper = ea.Mutinv(Pm=1)  # 生成逆转变异算子对象
        elif population.Encoding == 'BG':
            self.recOper = ea.Xovud(XOVR=1)  # 生成均匀交叉算子对象
            self.mutOper = ea.Mutbin(Pm=None)  # 生成二进制变异算子对象，Pm设置为None时，具体数值取变异算子中Pm的默认值
        elif population.Encoding == 'RI':
            self.recOper = ea.Recsbx(XOVR=1, n=20)  # 生成模拟二进制交叉算子对象
            self.mutOper = ea.Mutpolyn(Pm=1 / self.problem.Dim, DisI=20)  # 生成多项式变异算子对象
        # -------- ZQQ - begin -----------
        elif population.Encoding == 'NN':
            self.recOper = 0
            self.mutOper = ea.Mutation_NN(mu=muta_mu, var=muta_var)
        # -------- ZQQ - end -----------
        else:
            raise RuntimeError('编码方式必须为''BG''、''RI''、''P''或''NN''.')
        self.start_time = start_time
        self.dirName = 'Result/'+self.start_time
        self.objectives_class = objectives
        self.kfold = kfold
        self.calculmetric = calculmetric
        self.run_id = run_id

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
        [levels, criLevel] = self.ndSort(population.ObjV, NUM, None, population.CV,
                                         self.problem.maxormins)  # 对NUM个个体进行非支配分层
        dis = ea.crowdis(population.ObjV, levels)  # 计算拥挤距离
        population.FitnV[:, 0] = np.argsort(np.lexsort(np.array([dis, -levels])), kind='mergesort')  # 计算适应度
        chooseFlag = ea.selecting('dup', population.FitnV, NUM)  # 调用低级选择算子dup进行基于适应度排序的选择，保留NUM个个体
        # if isNN == 1:
        return population[chooseFlag], chooseFlag
        # elif isNN == 1:
        #     res = []
        #     for i in chooseFlag:
        #         res.append(population[i])
        #     return res

    def run(self, prophetPop=None):  # prophetPop为先知种群（即包含先验知识的种群）
        self.population.printPare(self.problem.test_org)
        # ==========================初始化配置===========================
        population = self.population
        # cmp_population = population.copy()
        # self.train_nets(cmp_population, Gen=0, epoch=1, iscal_metric=1)
        NIND = population.sizes
        self.initialization()  # 初始化算法模板的一些动态参数
        # ===========================准备进化============================
        population.initChrom(dataname=self.problem.dataname)  # 初始化种群染色体矩阵
        # population.save(dirName='Result/' + self.start_time, Gen=-1, NNmodel=population.Chrom)  # 打印最开始初始化时的网络参数

        self.call_aimFunc(population, kfold=self.kfold)  # 计算种群的目标函数值 计算目标值时，kfold!=0时不修改原网络参数，只是试探性的kfold后计算，kfold=0时会修改值
        self.train_nets(population, Gen=-1, epoch=1, iscal_metric=1, changeNets=0,
                        problem=self.problem, runtime=self.passTime)

        # 插入先验知识（注意：这里不会对先知种群prophetPop的合法性进行检查，故应确保prophetPop是一个种群类且拥有合法的Chrom、ObjV、Phen等属性）
        if prophetPop is not None:
            population = (prophetPop + population)[:NIND]  # 插入先知种群
        [levels, criLevel] = self.ndSort(population.ObjV, NIND, None, population.CV,
                                         self.problem.maxormins)  # 对NIND个个体进行非支配分层
        population.FitnV = (1 / levels).reshape(-1, 1)  # 直接根据levels来计算初代个体的适应度
        gen = 0
        flag_again = 0
        self.detect_differneces(population, gen, datatype='test', problem=self.problem)  # 记录
        # ===========================开始进化============================
        while not self.terminated(population):

            self.add_gen2info(population)
            gen += 1
            # print('Gen', gen)
            flag_again = 0
            # 选择个体参与进化
            offspring = population[ea.selecting(self.selFunc, population.FitnV, NIND)]
            # 对选出的个体进行进化操作
            # offspring.Chrom = self.recOper.do(offspring.Chrom)  # 重组
            offspring.Chrom = self.mutOper.do(offspring.Chrom)  # 变异
            self.call_aimFunc(offspring, kfold=self.kfold)  # 求进化后个体的目标函数值
            population, chooseidx = self.reinsertion(population, offspring, NIND, isNN=1)  # 重插入生成新一代种群

            if np.mod(gen, self.calculmetric) == 0:
                # population.save(dirName='Result/' + self.start_time, Gen=gen, NNmodel=population,
                #                 All_objs_train=self.all_objetives_train,
                #                 All_objs_valid=self.all_objetives_valid,
                #                 All_objs_test=self.all_objetives_test,
                #                 true_y=np.array(self.problem.test_y),
                #                 runtime=self.passTime)
                self.train_nets(population, Gen=gen, epoch=1, iscal_metric=1, changeNets=0,
                                problem=self.problem, runtime=self.passTime)  # 虚拟在所有train data上训练，但不改变网络的结构
                flag_again = 1
                now_time = time.strftime("%d %H.%M:%S", time.localtime(time.time()))
                self.detect_differneces(population, gen, datatype='test', problem=self.problem)  # 记录
                print(now_time, " Run ID ", self.run_id, ", Gen :", gen, ", runtime:", str(self.passTime))
                # Res_metrics = self.get_metric(population)
                print(population.ObjV)
        # self.problem.test_data
        if flag_again == 0:
            self.train_nets(population, Gen=gen, epoch=1, iscal_metric=1, problem=self.problem, runtime=self.passTime)
        # population.save(dirName='Result/' + self.start_time, Gen=gen, NNmodel=population,
        #                 All_objs_train=self.all_objetives_train,
        #                 All_objs_valid=self.all_objetives_valid,
        #                 All_objs_test=self.all_objetives_test,
        #                 true_y=np.array(self.problem.test_y),
        #                 runtime=self.passTime)

        print("Run ID ", self.run_id, "finished!")
        return self.finishing(population)  # 调用finishing完成后续工作并返回结果
