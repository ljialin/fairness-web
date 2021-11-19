# -*- coding: utf-8 -*-
import geatpy as ea
from geatpy.NNProblem import NNProblem_new  # 导入自定义问题接口
from geatpy.Population import Population
import time
import os
import numpy as np
import random
import torch
import sys


def run(parameters):
    start_time = parameters['start_time']
    print('The time is ', start_time)
    """===============================实例化问题对象============================"""
    problem = NNProblem_new(M=len(parameters['objectives_class']), learning_rate=parameters['learning_rate'],
                             batch_size=parameters['batch_size'],
                             sensitive_attributions=parameters['sensitive_attributions'],
                             epoches=parameters['epoches'], dataname=parameters['dataname'],
                             objectives_class=parameters['objectives_class'],
                             dirname='Result/' + parameters['start_time'],
                             seed_split_traintest=parameters['seed_split_traintest']
                             )  # 生成问题对象
    """==================================种群设置==============================="""
    Encoding = parameters['Encoding']  # 编码方式 --> 假的，其实是NN
    NIND = parameters['NIND']  # 种群规模
    Field = ea.crtfld('BG', problem.varTypes, problem.ranges, problem.borders,
                      [10] * len(problem.varTypes))  # 创建区域描述器
    population = Population(Encoding, Field, NIND, isNN=parameters['isNN'], n_feature=problem.getFeature(),
                               n_hidden=parameters['n_hidden'],
                               n_output=parameters['n_output'],
                               parameters=parameters,
                               logits=np.zeros([NIND, problem.test_data.shape[0]]))  # 实例化种群对象（此时种群还没被初始化，仅仅是完成种群对象的实例化）
    """================================算法参数设置============================="""
    myAlgorithm = ea.moea_templet_more_objectives(problem=problem, start_time=start_time,
                                  population=population,
                                  muta_mu=parameters['muta_mu'],
                                  muta_var=parameters['muta_var'],
                                  kfold=parameters['kfold'],
                                  calculmetric=parameters['logMetric'],
                                  run_id=parameters['run_id'])  # 实例化一个算法模板对象
    # myAlgorithm.mutOper.Pm = 0.2  # 修改变异算子的变异概率
    # myAlgorithm.recOper.XOVR = 0.9  # 修改交叉算子的交叉概率
    myAlgorithm.MAXGEN = parameters['MAXGEN']  # 最大进化代数
    myAlgorithm.logTras = parameters['logTras']  # 设置每多少代记录日志，若设置成0则表示不记录日志
    myAlgorithm.verbose = parameters['verbose']  # 设置是否打印输出日志信息
    myAlgorithm.drawing = parameters['drawing']  # 设置绘图方式（0：不绘图；1：绘制结果图；2：绘制目标空间过程动画；3：绘制决策空间过程动画）
    """==========================调用算法模板进行种群进化=========================
    调用run执行算法模板，得到帕累托最优解集NDSet以及最后一代种群。NDSet是一个种群类Population的对象。
    NDSet.ObjV为最优解个体的目标函数值；NDSet.Phen为对应的决策变量值。
    详见Population.py中关于种群类的定义。
    """
    [NDSet, population] = myAlgorithm.run()  # 执行算法模板，得到非支配种群以及最后一代种群
    NDSet.save(dirName=parameters['dirName'] + start_time)  # 把非支配种群的信息保存到文件中
    """==================================输出结果=============================="""
    print('run time: %s s' % myAlgorithm.passTime)
    return NDSet


def interface4flask(dataname='german', sensitive_attributions=None, objectives_class=None,
                    popsize=50, MAXGEN=500, optimizer="NSGA2", pid=0):
    if objectives_class is None:
        objectives_class = ['BCE_loss', 'Individual_fairness']
    if sensitive_attributions is None:
        sensitive_attributions = ['sex', 'race']

    n_hidden = [128]  # 默认一层隐藏层，宽度为n_hidden   np.array([64, 32])

    # 网络中的参数
    learning_rate = 0.01  # 机器学习中训练网络的学习率
    batch_size = 512  # 机器学习中的batch尺寸

    n_output = 1  # adult数据集中是二分类问题，因此为1
    epoches = 1  # 初始化或变异后再训练的epoch
    muta_mu = 0  # 网络变异正态噪音的均值
    muta_var = 0.01  # 网络变异正态噪音的方差
    kfold = 0  # 影响NNprobelm.py里面的流程， 计算objective时，需要将train data的kfold的值
    # 如果 kfold为大于0的数，则是最新版的算法流程
    # 如果 kfold为0不做cv，所有的train data只训练一次，并且保存此参数，用训练所得到的结果作为目标值
    preserve_sens_in_net = 0  # 在训练时，如果为1，则在训练网络的数据集的属性中包含敏感属性，反之

    # 多目标优化算法参数
    Encoding = 'BG'  # 无所谓，因为原来算法需要一个参数，实际并不影响进程
    popsize = popsize  # 种群大小
    isNN = 1  # 决定编码方式为 NN
    MAXGEN = MAXGEN  # 最大进化代数

    # 算法其他参数
    # run_id = 0 现在用来做前端任务的编号
    logTras = 10  # 设置每多少代记录日志，若设置成0则表示不记录日志，只是打印，并不是储存
    verbose = True  # 设置是否打印输出日志信息
    drawing = 1  # 设置绘图方式（0：不绘图；1：绘制结果图；2：绘制目标空间过程动画；3：绘制决策空间过程动画）
    logMetric = 20  # np.floor(MAXGEN/50)                        # 设置是否在每一代记录metric值，0代表不记录，如为3则为每隔3代记录一次
    dirName = 'Result//'  # 存储文件的路径
    start_time = time.strftime \
        ("%Y-%m-%d-%H-%M-%S",
         time.localtime(time.time()))  # 时间 决定文件夹的名字，保证唯一性
    seed_split_traintest = 1234
    parameters = {'start_time': start_time, 'learning_rate': learning_rate, 'batch_size': batch_size,
                  'n_hidden': n_hidden,
                  'n_output': n_output, 'epoches': epoches, 'muta_mu': muta_mu, 'muta_var': muta_var,
                  'Encoding': Encoding, 'NIND': popsize, 'isNN': isNN, 'MAXGEN': MAXGEN,
                  'logTras': logTras, 'verbose': verbose, 'drawing': drawing, 'dirName': dirName,
                  'dataname': dataname, 'sensitive_attributions': sensitive_attributions,
                  'logMetric': logMetric, 'objectives_class': objectives_class, 'kfold': kfold,
                  'preserve_sens_in_net': preserve_sens_in_net, 'seed_split_traintest': seed_split_traintest,
                  'run_id': pid}

    print(parameters)
    run(parameters)


if __name__ == '__main__':

    try:
        run_ids = sys.argv[1]
    except:
        run_ids = 25

    run_ids = int(run_ids)
    for run_id in range(run_ids, run_ids+1): #+80
        print(run_id)

        # for 确定 compas muta_var=0.01， 从2021-03-17-20-13-53开始
        if (run_id >= -1) & (run_id <= 24):
            objectives_class = ['MSE', 'individual', 'group']  # 选用 'l2' 'MSE', 'individual', 'group'中的一个或多个作为MOEA中的目标函数值
        elif (run_id >= 25) & (run_id <= 50): # 2021-03-18-12-38-47 ------ 2021-03-19-03-49-25
            objectives_class = ['MSE', 'group']  # 选用 'l2' 'MSE', 'individual', 'group'中的一个或多个作为MOEA中的目标函数值
        elif (run_id >= 51) & (run_id <= 77):
            objectives_class = ['MSE', 'individual']  # 选用 'l2' 'MSE', 'individual', 'group'中的一个或多个作为MOEA中的目标函数值
        else:
            continue

        n_hidden = [128]  # 默认一层隐藏层，宽度为n_hidden   np.array([64, 32])

        # 网络中的参数
        learning_rate = 0.01  # 机器学习中训练网络的学习率
        batch_size = 512  # 机器学习中的batch尺寸

        n_output = 1  # adult数据集中是二分类问题，因此为1
        epoches = 1  # 初始化或变异后再训练的epoch
        muta_mu = 0  # 网络变异正态噪音的均值
        muta_var = 0.01  # 网络变异正态噪音的方差
        kfold = 0  # 影响NNprobelm.py里面的流程， 计算objective时，需要将train data的kfold的值
        # 如果 kfold为大于0的数，则是最新版的算法流程
        # 如果 kfold为0不做cv，所有的train data只训练一次，并且保存此参数，用训练所得到的结果作为目标值
        preserve_sens_in_net = 0  # 在训练时，如果为1，则在训练网络的数据集的属性中包含敏感属性，反之

        # 多目标优化算法参数
        Encoding = 'BG'  # 无所谓，因为原来算法需要一个参数，实际并不影响进程
        popsize = 50  # 种群大小
        isNN = 1  # 决定编码方式为 NN
        MAXGEN = 500  # 最大进化代数


        # 算法其他参数
        logTras = 10  # 设置每多少代记录日志，若设置成0则表示不记录日志，只是打印，并不是储存
        verbose = True  # 设置是否打印输出日志信息
        drawing = 1  # 设置绘图方式（0：不绘图；1：绘制结果图；2：绘制目标空间过程动画；3：绘制决策空间过程动画）
        logMetric = 20  # np.floor(MAXGEN/50)                        # 设置是否在每一代记录metric值，0代表不记录，如为3则为每隔3代记录一次
        dirName = 'Result//'  # 存储文件的路径
        start_time = time.strftime \
            ("%Y-%m-%d-%H-%M-%S",
             time.localtime(time.time()))  # 时间 决定文件夹的名字，保证唯一性
        seed_split_traintest = 1234

        # sys.argv[1] -> run_id
        # sys.argv[2] -> randomsee
        # randomseed = 1234 + randomseed*123
        # random.seed(randomseed)
        # torch.manual_seed(randomseed)
        # torch.cuda.manual_seed(randomseed)
        # np.random.seed(randomseed)

        # 数据集信息
        # ricci : 'Race'
        # german : 'sex', 'age'
        # propublica-recidivism: 'sex', 'race'
        # adult : 'sex', 'race'
        # dataname = 'german'  # 数据集名字
        # sensitive_attributions = ['sex', 'age']  # 决定 group fairness 的计算，e.g. ['gender']、['race']、['gender', 'race']

        dataname = 'propublica-recidivism'  # 数据集名字
        sensitive_attributions = ['sex', 'race']  # 决定 group fairness 的计算，e.g. ['gender']、['race']、['gender', 'race']
        parameters = {'start_time': start_time, 'learning_rate': learning_rate, 'batch_size': batch_size,
                      'n_hidden': n_hidden,
                      'n_output': n_output, 'epoches': epoches, 'muta_mu': muta_mu, 'muta_var': muta_var,
                      'Encoding': Encoding, 'NIND': popsize, 'isNN': isNN, 'MAXGEN': MAXGEN,
                      'logTras': logTras, 'verbose': verbose, 'drawing': drawing, 'dirName': dirName,
                      'dataname': dataname, 'sensitive_attributions': sensitive_attributions,
                      'logMetric': logMetric, 'objectives_class': objectives_class, 'kfold': kfold,
                      'preserve_sens_in_net': preserve_sens_in_net, 'seed_split_traintest': seed_split_traintest,
                      'run_id': run_id}

        print(parameters)
        run(parameters)
