# -*- coding: utf-8 -*-
import numpy as np
import geatpy as ea
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from geatpy.zqq.load_data import load_data
from sklearn.model_selection import KFold
import time
import os
import copy
import matplotlib.pyplot as plt
# from geatpy.plot_demo import plot_decision_boundary, plot_decision_boundary4
from zqq.Mutation_NN import Mutation_NN
from scipy.optimize import minimize
# from geatpy.MGD_utils import steep_direct_cost, steep_direc_cost_deriv, make_constraints
import torch.nn.functional as F
from itertools import product
from scipy.spatial.distance import pdist, squareform


def get_label(logits):
    pred_label = logits
    pred_label[np.where(pred_label >= 0.5)] = 1
    pred_label[np.where(pred_label < 0.5)] = 0
    pred_label = pred_label.reshape(1, logits.shape[0] * logits.shape[1])
    pred_label = pred_label.reshape(1, -1)
    return pred_label


def get_gen_grads(model, loss_, retain_graph):
    grads = torch.autograd.grad(outputs=loss_, inputs=model.parameters(), retain_graph=retain_graph)
    # model.zero_grad()
    for params_grads in grads:

        try:
            grads_ = torch.cat([grads_, params_grads.view(-1)], 0)
        except:
            grads_ = params_grads.view(-1)
    norm_value = grads_.norm()
    return grads_ / grads_.norm(), norm_value  # origin
    # return grads_, norm_value


def concat_grad(model):
    """
    Concatenates the gradients of a model to form a single parameter vector tensor.

    Args:
        model (nn.Module): PyTorch model object

    Returns: A single vector tensor with the model gradients concatenated
    """
    g = None
    for name, param in model.named_parameters():
        grad = param.grad
        if "bias" in name:
            grad = grad.unsqueeze(dim=0)  # 增加一维
        if g is None:
            g = param.grad.view(1, -1)
        else:
            if len(grad.shape) < 2:
                grad = grad.unsqueeze(dim=0)
            else:
                grad = grad.view(1, -1)
            g = torch.cat((g, grad), dim=1)
    return g.squeeze(dim=0)


def replace_grad(model, grad):
    """
    Replaces the gradients of the model with the specified gradient vector tensor.

    Args:
        model (nn.Module): PyTorch model object
        grad (Tensor): Vector of concatenated gradients

    Returns: None
    """
    start = 0
    for name, param in model.named_parameters():
        numel = param.numel()
        param.grad.data = grad[start:start + numel].view(param.grad.shape)
        start += numel


def project_grad(x, v):
    """
    Performs a projection of one vector on another.

    Args:
        x (Tensor): Vector to project
        v (Tensor): Vector on which projection is required

    Returns: Tensor containing the projected vector
    """
    norm_v = v / (torch.norm(v) + torch.finfo(torch.float32).tiny)
    proj_grad = torch.dot(x, norm_v) * v  # origin
    # proj_grad = torch.dot(x, norm_v) * norm_v  # new
    return proj_grad


def forward_full(dataloader, predictor, adversary, criterion, device, optimizer_P=None, optimizer_A=None,
                 train=False, alpha=0.3):
    """
    Performs one epoch of training/evaluation on the data

    Args:
        dataloader (DataLoader): Dataloader for the data
        predictor (nn.Module): Predictor model
        adversary (nn.Module): Adversary model
        criterion (func): Loss criterion
        device (torch.device): Device on which to train/evaluate
        dataset_name (str): Name of the dataset
        optimizer_P (Optimizer): Optimizer for the predictor
        optimizer_A (Optimizer): Optimizer for the adversary
        train (bool): True for training mode, False for evaluation
        alpha (float): Value of hyperparameter alpha

    Returns: Metrics from training/evaluation

    """
    labels_dict = {'true': [], 'pred': []}
    protected_dict = {'true': [], 'pred': []}
    losses_P, losses_A = [], []

    for i, (x, y, z) in enumerate(dataloader):

        x = x.to(device)
        true_y_label = y.to(device)
        true_z_label = z.to(device)

        # Forward step through predictor
        pred_y_logit, pred_y_prob = predictor(x)

        if train is False:
            if i == 0:
                prediction_probs = pred_y_prob.cpu().detach().numpy()
            else:
                prediction_probs = np.concatenate((prediction_probs, pred_y_prob.cpu().detach().numpy()), axis=0)

        # Compute loss with respect to predictor
        loss_P = criterion(pred_y_prob, true_y_label)
        losses_P.append(loss_P.item())

        # Store the true labels and the predictions
        labels_dict['true'].extend(y.squeeze().cpu().numpy().tolist())
        pred_y = (pred_y_prob > 0.5).int().squeeze(dim=1).cpu().numpy().tolist()
        labels_dict['pred'].extend(pred_y)

        protected_dict['true'].extend(z.squeeze().cpu().numpy().tolist())

        if adversary is not None:
            # Forward step through adversary
            pred_z_logit, pred_z_prob = adversary(pred_y_logit, true_y_label)

            # Compute loss with respect to adversary
            loss_A = criterion(pred_z_prob, true_z_label)
            losses_A.append(loss_A.item())

            pred_z = (pred_z_prob > 0.5).float().squeeze(dim=1).cpu().numpy().tolist()
            protected_dict['pred'].extend(pred_z)

        if train:
            if adversary is not None:
                # Reset gradients of adversary and predictor
                optimizer_A.zero_grad()
                optimizer_P.zero_grad()
                # Compute gradients of adversary loss
                loss_A.backward(retain_graph=True)
                # Concatenate gradients of adversary loss with respect to the predictor
                grad_w_La = concat_grad(predictor)

            # Reset gradients of predictor
            optimizer_P.zero_grad()

            # Compute gradients of predictor loss
            loss_P.backward()

            if adversary is not None:
                # Concatenate gradients of predictor loss with respect to the predictor
                grad_w_Lp = concat_grad(predictor)
                # Project gradients of the predictor
                proj_grad = project_grad(grad_w_Lp, grad_w_La)
                # Modify and replace the gradient of the predictor
                alph1 = 1
                alph2 = 18
                grad_w_Lp = grad_w_Lp - alph1 * proj_grad - alpha * grad_w_La
                replace_grad(predictor, grad_w_Lp)

            # Update predictor weights
            optimizer_P.step()

            if adversary is not None:
                # Update adversary weights
                optimizer_A.step()


def generalized_entropy_index(benefits, alpha):
    # The method is from "Unified Approach to Quantifying Algorithmic Unfairness:
    # Measuring Individual & Group Unfairness via Inequality Indices"

    mu = torch.mean(benefits)
    individual_fitness = torch.mean(torch.pow(benefits / mu, alpha) - 1) / ((alpha - 1) * alpha)
    return individual_fitness


def get_average(group_values, plan):
    if plan == 1:
        values = 0.0
        count = 0
        num_group = len(group_values)
        for i in range(num_group):
            if i == (num_group - 1):
                break
            for j in range(i+1, num_group):
                values += (torch.abs(group_values[i] - group_values[j]))
                count += 1
        if count == 0:
            return torch.zeros(1)[0]
        else:
            return values/count

    else:
        values = torch.zeros(1)[0]
        count = 0
        num_group = len(group_values)
        for i in range(num_group):
            if i == (num_group - 1):
                break
            for j in range(i + 1, num_group):
                values = torch.max(values, torch.abs(group_values[i] - group_values[j]))
                # values.append(torch.abs(group_values[i] - group_values[j]))
                count += 1
        if count == 0:
            return torch.zeros(1)[0]
        else:
            return values


def get_L_regularization(individual, L):
    l2_regularization = 0.0
    for param in individual.parameters():
        l2_regularization += torch.norm(param, L)  # L2 正则化
    return l2_regularization


def calcul_all_fairness(data, logits, truelabel, sensitive_attributions, alpha):
    # The method is from "Unified Approach to Quantifying Algorithmic Unfairness:
    # Measuring Individual & Group Unfairness via Inequality Indices"
    # a few differences: "logits - truelabel + 1" instead on "pred_label - truelabel + 1"
    sum_num = logits.shape[0] * logits.shape[1]

    benefits = logits - truelabel + 1  # new version in section 3.1
    benefits_mean = torch.mean(benefits)

    attribution = data.columns
    Within_fairness = 0.0
    Group_fairness = 0.0
    FPRs_logit = []
    FNRs_logit = []
    DPs_logit = []
    PPs_logit = []

    group_dict = {}

    for sens in sensitive_attributions:
        temp = []
        for attr in attribution:
            temp1 = sens + '_'
            if temp1 in attr:
                temp.append(attr)
        group_dict.update({sens: temp})

    group_attr = []
    for sens in sensitive_attributions:
        group_attr.append(group_dict[sens])
    for item in product(*eval(str(group_attr))):
        group = item
        flag = np.ones([1, sum_num]) == np.ones([1, sum_num])
        for g in group:
            flag = flag & data[g]
        g_num = np.sum(flag)
        if g_num != 0:
            g_idx = np.array(np.where(flag)).reshape([1, g_num])

            g_benefits = benefits[g_idx].reshape([1, g_num])
            g_fairness = generalized_entropy_index(g_benefits, alpha)
            g_benefits_mean = torch.mean(g_benefits)
            g_individual_fairness = (g_num / sum_num) * (torch.pow(g_benefits_mean / benefits_mean, alpha)) * g_fairness
            g_group_fairness = (g_num / (sum_num * (alpha - 1) * alpha)) * (
                    torch.pow(g_benefits_mean / benefits_mean, alpha) - 1)
            Within_fairness += g_individual_fairness
            Group_fairness += g_group_fairness

            # g_logits = logits[g_idx]
            # g_truelabel = truelabel[g_idx]
            # g_predlabel_flag = np.array(np.where(np.array(g_logits[:, 0].clone().detach()) > 0.4))
            # g_predlabel_num = g_predlabel_flag.reshape(1, -1).shape[1]
            # g_truelabel_flag = np.array(np.where(g_truelabel[:, 0].clone().detach()))
            # plan = 2
            # if plan == 1:
            #     # original paper : FNNC: Achieving Fairness through Neural Networks
            #     # calculate based on the logits
            #     if torch.sum(1 - g_truelabel) > 0:
            #         FPR_logit = torch.sum(g_logits * (1 - g_truelabel)) / g_num
            #         FPRs_logit.append(FPR_logit)
            #
            #     if torch.sum(g_truelabel) > 0:
            #         FNR_logit = torch.sum((1 - g_logits) * g_truelabel) / g_num
            #         FNRs_logit.append(FNR_logit)
            #
            #     DP_logit = torch.sum(g_logits) / g_num
            #     DPs_logit.append(DP_logit)
            #
            #     if g_num > 0:
            #         PP_logit = torch.sum(g_logits * g_truelabel) / g_num
            #         PPs_logit.append(PP_logit)
            #
            # else:
            #     # modified "FNNC: Achieving Fairness through Neural Networks"
            #     # calculate based on the logits
            #     if torch.sum(1 - g_truelabel) > 0:
            #         FPR_logit = torch.sum(g_logits * (1 - g_truelabel)) / torch.sum(1 - g_truelabel)
            #         FPRs_logit.append(FPR_logit)
            #
            #     if torch.sum(g_truelabel) > 0:
            #         FNR_logit = torch.sum((1 - g_logits) * g_truelabel) / torch.sum(g_truelabel)
            #         FNRs_logit.append(FNR_logit)
            #
            #     DP_logit = torch.sum(g_logits) / g_num
            #     DPs_logit.append(DP_logit)
            #
            #     if g_truelabel_flag.shape[1]:
            #         # PP_logit = torch.sum(g_logits[g_predlabel_flag] * g_truelabel[g_predlabel_flag]) / g_predlabel_num
            #         PP_logit = torch.sum(g_logits * g_truelabel) / torch.sum(g_logits)
            #         PPs_logit.append(PP_logit)

    # # Demographic Parity
    # DP_value = get_average(DPs_logit, 2)
    #
    # # FPR
    # FPR_value = get_average(FPRs_logit, 2)
    #
    # # FNR
    # FNR_value = get_average(FNRs_logit, 2)
    #
    # # Predictive parity
    # PP_value = get_average(PPs_logit, 1)
    #
    # Individual unfairness = within-group + between-group
    Individual_fairness = generalized_entropy_index(benefits, alpha)

    # Group unfairness = between-group

    Group_losses = {"Individual_fairness": Individual_fairness, "Group_fairness": Group_fairness}

    return Group_losses


def get_obj_vals(Group_infos, obj_classes):
    res = np.zeros([1, len(obj_classes)])
    for idx, obj_name in enumerate(obj_classes):
        res[0, idx] = Group_infos[obj_name]
        # print(Group_infos)
    return res


class NNProblem_new(ea.Problem):  # 继承Problem父类
    def __init__(self, M=None, learning_rate=0.01, batch_size=500, sensitive_attributions=None, dataModel=None,
                 epoches=2, dataname='ricci', objectives_class=None, dirname=None, preserve_sens_in_net=0,
                 seed_split_traintest=2021, weight_decay=1e-1, cal_obj_plan=6, GAN_alpha=0.3, L_regul=0.8,
                 start_time=0, sel_obj_plan=1, obj_is_logits=1, is_ensemble=False):
        if objectives_class is None:
            objectives_class = ['BCE_loss', 'Individual_fairness']
            M = len(objectives_class)
        if sensitive_attributions is None:
            if dataname == 'ricci':
                sensitive_attributions = ['Race']
            elif dataname == 'adult':
                sensitive_attributions = ['sex']
            elif dataname == 'german':
                sensitive_attributions = ['sex']
            else:
                print('There is no dataset called ', dataname)
        name = 'FairnessProblem'  # 初始化name（函数名称，可以随意设置）
        Dim = 5  # 初始化Dim（决策变量维数）
        maxormins = [1] * M  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        varTypes = [1] * Dim  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        lb = [0] * Dim  # 决策变量下界
        ub = [10] * Dim  # 决策变量上界
        lbin = [1] * Dim  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1] * Dim  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)
        # [self.train_data, self.train_data_norm, self.test_data, self.test_data_norm, self.train_label,
        #  self.test_label, self.train_y, self.test_y, self.positive_y] = load_data(dataname, test_size=0.25)
        self.preserve_sens_in_net = preserve_sens_in_net
        self.seed_split_traintest = seed_split_traintest
        self.weight_decay = weight_decay
        self.dataModel = dataModel
        DATA = load_data(dataModel, preserve_sens_in_net=preserve_sens_in_net, sensitive_attributions=sensitive_attributions)
        self.train_data = DATA['train_data']
        self.train_data_norm = DATA['train_data_norm']
        self.train_label = DATA['train_label']
        self.train_y = DATA['train_y']

        self.valid_data = DATA['valid_data']
        self.valid_data_norm = DATA['valid_data_norm']
        self.valid_label = DATA['valid_label']
        self.valid_y = DATA['valid_y']

        if is_ensemble:
            self.ensemble_data = DATA['ensemble_data']
            self.ensemble_data_norm = DATA['ensemble_data_norm']
            self.ensemble_label = DATA['ensemble_label']
            self.ensemble_y = DATA['ensemble_y']

            self.ensemble_data = DATA['ensemble_data']
            self.ensemble_data_norm = DATA['ensemble_data_norm']
            self.ensemble_label = DATA['ensemble_label']
            self.ensemble_y = DATA['ensemble_y']

            self.ensemble_org = DATA['ensemble_org']
            self.num_ensemble = self.ensemble_org.shape[0]

        self.test_data = DATA['test_data']
        self.test_data_norm = DATA['test_data_norm']
        self.test_label = DATA['test_label']
        self.test_y = DATA['test_y']

        self.data_org = DATA['org_data']
        self.train_org = DATA['train_org']
        self.valid_org = DATA['valid_org']
        self.test_org = DATA['test_org']

        self.positive_class_name = DATA['positive_class_name']
        self.positive_class = DATA['positive_class']

        self.Groups_info = DATA['Groups_info']
        self.privileged_class_names = DATA['privileged_class_names']
        self.num_train = self.train_org.shape[0]
        self.num_valid = self.valid_org.shape[0]
        self.num_test = self.test_org.shape[0]

        self.is_ensemble=is_ensemble

        # self.sens_flag = DATA['sens_flag']
        # self.sens_flag_name = DATA['sens_flag_name']

        # ricci : Race
        # adult : sex, race
        # german : sex, age
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.sensitive_attributions = sensitive_attributions
        self.epoches = epoches
        self.M = M
        self.num_features = self.train_data_norm.shape[1]
        self.dataname = dataname
        # self.dataset_obj = dataset_obj
        self.objectives_class = objectives_class
        self.dirname = 'zqq/' + dirname
        self.cal_sens_name = [sensitive_attributions[0]]
        self.cal_obj_plan = cal_obj_plan
        self.GAN_alpha = GAN_alpha
        self.L_regul = L_regul
        self.DATA = DATA
        self.ran_flag = 0
        self.start_time = start_time
        self.sel_obj_plan = sel_obj_plan
        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None
        self.more_info_num = None
        self.use_gpu = False

        self.x_train = None
        self.y_train = None
        self.y_train = None

        self.x_test = None
        self.y_test = None
        self.y_test = None

        self.x_valid = None
        self.y_valid = None
        self.y_valid = None

        self.obj_is_logits = obj_is_logits
        self.dist_mat = {}
        self.base_num = 10

    def do_pre(self):
        # calculate the distance among the data
        dist_mat_train = pdist(self.train_data_norm, 'euclidean')
        dist_mat_train = dist_mat_train.reshape(1, -1)
        max_dist_train = np.max(dist_mat_train)
        dist_mat_train = dist_mat_train / max_dist_train

        dist_mat_valid = pdist(self.valid_data_norm, 'euclidean')
        dist_mat_valid = dist_mat_valid.reshape(1, -1)
        max_dist_valid = np.max(dist_mat_valid)
        dist_mat_valid = dist_mat_valid / max_dist_valid

        dist_mat_test = pdist(self.test_data_norm, 'euclidean')
        dist_mat_test = dist_mat_test.reshape(1, -1)
        max_dist_test = np.max(dist_mat_test)
        dist_mat_test = dist_mat_test / max_dist_test

        self.dist_mat = {"dist_mat_train": dist_mat_train, "dist_mat_valid": dist_mat_valid,
                         "dist_mat_test": dist_mat_test}
        if self.is_ensemble:
            dist_mat_ensemble = pdist(self.ensemble_data_norm, 'euclidean')
            dist_mat_ensemble = dist_mat_ensemble.reshape(1, -1)
            max_dist_ensemble = np.max(dist_mat_ensemble)
            dist_mat_ensemble = dist_mat_ensemble / max_dist_ensemble
            self.dist_mat.update({"dist_mat_ensemble": dist_mat_ensemble})
            self.x_ensemble = torch.Tensor(self.ensemble_data_norm)
            self.y_ensemble = torch.Tensor(self.ensemble_y)
            self.y_ensemble = self.y_ensemble.view(self.y_ensemble.shape[0], 1)

        self.x_train = torch.Tensor(self.train_data_norm)
        self.y_train = torch.Tensor(self.train_y)
        self.y_train = self.y_train.view(self.y_train.shape[0], 1)

        self.x_test = torch.Tensor(self.test_data_norm)
        self.y_test = torch.Tensor(self.test_y)
        self.y_test = self.y_test.view(self.y_test.shape[0], 1)

        self.x_valid = torch.Tensor(self.valid_data_norm)
        self.y_valid = torch.Tensor(self.valid_y)
        self.y_valid = self.y_valid.view(self.y_valid.shape[0], 1)

        sens_attr = self.cal_sens_name
        group_dicts = self.Groups_info['group_dict_train']
        s_labels = group_dicts[sens_attr[0]][0]

        sens_idxs_train = self.Groups_info['sens_idxs_train']
        s_labels_train = sens_idxs_train[s_labels][0]
        S_train = np.zeros([1, self.num_train])
        S_train[0, s_labels_train] = 1
        s_train = torch.Tensor(S_train)

        sens_idxs_valid = self.Groups_info['sens_idxs_valid']
        s_labels_valid = sens_idxs_valid[s_labels][0]
        S_valid = np.zeros([1, self.num_valid])
        S_valid[0, s_labels_valid] = 1
        s_valid = torch.Tensor(S_valid)

        sens_idxs_test = self.Groups_info['sens_idxs_test']
        sens_idxs_name_test = self.Groups_info['sens_idxs_name_test']
        s_labels_test = sens_idxs_test[s_labels][0]
        S_test = np.zeros([1, self.num_test])
        S_test[0, s_labels_test] = 1
        s_test = torch.Tensor(S_test)

        train_idx = torch.arange(start=0, end=self.train_data_norm.shape[0], step=1)
        valid_idx = torch.arange(start=0, end=self.valid_data_norm.shape[0], step=1)
        test_idx = torch.arange(start=0, end=self.test_data_norm.shape[0], step=1)

        train = TensorDataset(self.x_train, self.y_train, s_train[0], train_idx)
        test = TensorDataset(self.x_test, self.y_test, s_test[0], test_idx)
        valid = TensorDataset(self.x_valid, self.y_valid, s_valid[0], valid_idx)

        self.train_loader = DataLoader(train, batch_size=self.batch_size, shuffle=True)
        self.valid_loader = DataLoader(valid, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(test, batch_size=self.batch_size, shuffle=True)

        if self.is_ensemble:
            sens_idxs_ensemble = self.Groups_info['sens_idxs_ensemble']
            sens_idxs_name_ensemble = self.Groups_info['sens_idxs_name_ensemble']
            s_labels_ensemble = sens_idxs_ensemble[s_labels][0]
            S_ensemble = np.zeros([1, self.num_ensemble])
            S_ensemble[0, s_labels_ensemble] = 1
            s_ensemble = torch.Tensor(S_ensemble)
            ensemble_idx = torch.arange(start=0, end=self.ensemble_data_norm.shape[0], step=1)
            ensemble = TensorDataset(self.x_ensemble, self.y_ensemble, s_ensemble[0], ensemble_idx)
            self.ensemble_loader = DataLoader(ensemble, batch_size=self.batch_size, shuffle=True)

        #############
        sens_attr = self.cal_sens_name
        group_dicts = self.Groups_info['group_dict_train']
        s_labels = group_dicts[sens_attr[0]][0]

        sens_idxs_train = self.Groups_info['sens_idxs_train']
        s_labels_train = sens_idxs_train[s_labels][0]
        S_train = np.zeros([1, self.num_train])
        S_train[0, s_labels_train] = 1

        sens_idxs_valid = self.Groups_info['sens_idxs_valid']
        s_labels_valid = sens_idxs_valid[s_labels][0]
        S_valid = np.zeros([1, self.num_valid])
        S_valid[0, s_labels_valid] = 1

        sens_idxs_test = self.Groups_info['sens_idxs_test']
        s_labels_test = sens_idxs_test[s_labels][0]
        S_test = np.zeros([1, self.num_test])
        S_test[0, s_labels_test] = 1

        # np.savetxt('Result/' + self.start_time + '/detect/Sens_train.txt', S_train)
        # np.savetxt('Result/' + self.start_time + '/detect/Sens_valid.txt', S_valid)
        # np.savetxt('Result/' + self.start_time + '/detect/Sens_test.txt', S_test)

        ############
        self.use_gpu = torch.cuda.is_available()
        self.use_gpu = False
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Groups_info = ea.Cal_objectives(self.train_data, self.train_data_norm,
        #                                 np.array(self.train_y).reshape(1, -1),
        #                                 np.array(self.train_y).reshape(1, -1), self.sensitive_attributions,
        #                                 2, self.Groups_info['sens_idxs_test'],
        #                                 plan=self.cal_obj_plan,
        #                                 dis=np.array(self.train_y).reshape(1, -1),
        #                                 dist_mat=dist_mat_train,
        #                                 obj_names=self.objectives_class)

        # self.more_info_num = Groups_info["addition_num"]

        # record sensitive info
        # if len(self.Groups_info['group_dict_train']) > 1:
        #     with open('Result/' + self.start_time + '/detect/train_sensitive_name.txt', 'a+') as file:
        #         for sens_name in self.Groups_info['sens_idxs_train']:
        #             if '+' in sens_name:
        #                 file.write(sens_name + ',')
        #         file.close()
        #     with open('Result/' + self.start_time + '/detect/valid_sensitive_name.txt', 'a+') as file:
        #         for sens_name in self.Groups_info['sens_idxs_valid']:
        #             if '+' in sens_name:
        #                 file.write(sens_name + ',')
        #         file.close()
        #
        #     with open('Result/' + self.start_time + '/detect/test_sensitive_name.txt', 'a+') as file:
        #         for sens_name in self.Groups_info['sens_idxs_test']:
        #             if '+' in sens_name:
        #                 file.write(sens_name + ',')
        #         file.close()
        #
        #     idx_sens_train = np.zeros([1, self.num_train])
        #     idx_sens_valid = np.zeros([1, self.num_valid])
        #     idx_sens_test = np.zeros([1, self.num_test])
        #
        #     idx_count = 0
        #     for sens_name in self.Groups_info['sens_idxs_train']:
        #         if '+' in sens_name:
        #             idx_sens_train[0][self.Groups_info['sens_idxs_train'][sens_name][0]] = idx_count
        #             idx_count += 1
        #     np.savetxt('Result/' + self.start_time + '/detect/train_idxs_sensitive.txt', idx_sens_train)
        #
        #     idx_count = 0
        #     for sens_name in self.Groups_info['sens_idxs_valid']:
        #         if '+' in sens_name:
        #             idx_sens_valid[0][self.Groups_info['sens_idxs_valid'][sens_name][0]] = idx_count
        #             idx_count += 1
        #     np.savetxt('Result/' + self.start_time + '/detect/valid_idxs_sensitive.txt', idx_sens_valid)
        #
        #     idx_count = 0
        #     for sens_name in self.Groups_info['sens_idxs_test']:
        #         if '+' in sens_name:
        #             idx_sens_test[0][self.Groups_info['sens_idxs_test'][sens_name][0]] = idx_count
        #             idx_count += 1
        #     np.savetxt('Result/' + self.start_time + '/detect/test_idxs_sensitive.txt', idx_sens_test)
        #
        #     if self.is_ensemble:
        #         with open('Result/' + self.start_time + '/detect/ensemble_sensitive_name.txt', 'a+') as file:
        #             for sens_name in self.Groups_info['sens_idxs_ensemble']:
        #                 if '+' in sens_name:
        #                     file.write(sens_name + ',')
        #             file.close()
        #         idx_sens_ensemble = np.zeros([1, self.num_ensemble])
        #         idx_count = 0
        #         for sens_name in self.Groups_info['sens_idxs_ensemble']:
        #             if '+' in sens_name:
        #                 idx_sens_ensemble[0][self.Groups_info['sens_idxs_ensemble'][sens_name][0]] = idx_count
        #                 idx_count += 1
        #         np.savetxt('Result/' + self.start_time + '/detect/ensemble_idxs_sensitive.txt', idx_sens_ensemble)
        #         np.savetxt('Result/' + self.start_time + '/detect/ensemble_truelabel.txt', np.array(self.ensemble_y))
        #
        # else:
        #
        #     with open('Result/' + self.start_time + '/detect/train_sensitive_name.txt', 'a+') as file:
        #         for sens_name in self.Groups_info['sens_idxs_train']:
        #             file.write(sens_name + ',')
        #         file.close()
        #     with open('Result/' + self.start_time + '/detect/valid_sensitive_name.txt', 'a+') as file:
        #         for sens_name in self.Groups_info['sens_idxs_valid']:
        #             file.write(sens_name + ',')
        #         file.close()
        #
        #     with open('Result/' + self.start_time + '/detect/test_sensitive_name.txt', 'a+') as file:
        #         for sens_name in self.Groups_info['sens_idxs_test']:
        #             file.write(sens_name + ',')
        #         file.close()
        #
        #     idx_sens_train = np.zeros([1, self.num_train])
        #     idx_sens_valid = np.zeros([1, self.num_valid])
        #     idx_sens_test = np.zeros([1, self.num_test])
        #
        #     idx_count = 0
        #     for sens_name in self.Groups_info['sens_idxs_train']:
        #         idx_sens_train[0][self.Groups_info['sens_idxs_train'][sens_name][0]] = idx_count
        #         idx_count += 1
        #     np.savetxt('Result/' + self.start_time + '/detect/train_idxs_sensitive.txt', idx_sens_train)
        #
        #     idx_count = 0
        #     for sens_name in self.Groups_info['sens_idxs_valid']:
        #         idx_sens_valid[0][self.Groups_info['sens_idxs_valid'][sens_name][0]] = idx_count
        #         idx_count += 1
        #     np.savetxt('Result/' + self.start_time + '/detect/valid_idxs_sensitive.txt', idx_sens_valid)
        #
        #     idx_count = 0
        #     for sens_name in self.Groups_info['sens_idxs_test']:
        #         idx_sens_test[0][self.Groups_info['sens_idxs_test'][sens_name][0]] = idx_count
        #         idx_count += 1
        #     np.savetxt('Result/' + self.start_time + '/detect/test_idxs_sensitive.txt', idx_sens_test)
        #
        #     if self.is_ensemble:
        #         with open('Result/' + self.start_time + '/detect/ensemble_sensitive_name.txt', 'a+') as file:
        #             for sens_name in self.Groups_info['sens_idxs_ensemble']:
        #                 file.write(sens_name + ',')
        #             file.close()
        #         idx_sens_ensemble = np.zeros([1, self.num_ensemble])
        #         idx_count = 0
        #         for sens_name in self.Groups_info['sens_idxs_ensemble']:
        #             idx_sens_ensemble[0][self.Groups_info['sens_idxs_ensemble'][sens_name][0]] = idx_count
        #             idx_count += 1
        #         np.savetxt('Result/' + self.start_time + '/detect/ensemble_idxs_sensitive.txt', idx_sens_ensemble)
        #         np.savetxt('Result/' + self.start_time + '/detect/ensemble_truelabel.txt', np.array(self.ensemble_y))
        #
        # np.savetxt('Result/' + self.start_time + '/detect/train_truelabel.txt', np.array(self.train_y))
        # np.savetxt('Result/' + self.start_time + '/detect/valid_truelabel.txt', np.array(self.valid_y))
        # np.savetxt('Result/' + self.start_time + '/detect/test_truelabel.txt', np.array(self.test_y))

    def getFeature(self):
        return self.num_features

    def aimFunc(self, pop, kfold=0, gen=0, dirName=None, loss_type=-1, train_net=1, lr_decay_factor=0.99):  # 目标函数
        # kfold = 0 : 全部的train训练model
        # kfold！= 0 : 将train进行kfold并 k 为输入的数值
        start_time = time.time()
        test_flag = 0
        if test_flag == 1:
            if self.ran_flag == 0:
                Groups_info = self.Groups_info
                sens_attr = self.cal_sens_name
                group_dicts = self.Groups_info['group_dict_train']
                s_labels = group_dicts[sens_attr[0]][0]

                sens_idxs_train = self.Groups_info['sens_idxs_train']
                s_labels_train = sens_idxs_train[s_labels][0]
                S_train = np.zeros([1, self.num_train])
                S_train[0, s_labels_train] = 1

                sens_idxs_valid = self.Groups_info['sens_idxs_valid']
                s_labels_valid = sens_idxs_valid[s_labels][0]
                S_valid = np.zeros([1, self.num_valid])
                S_valid[0, s_labels_valid] = 1

                sens_idxs_test = self.Groups_info['sens_idxs_test']
                s_labels_test = sens_idxs_test[s_labels][0]
                S_test = np.zeros([1, self.num_test])
                S_test[0, s_labels_test] = 1

                np.savetxt('Result/' + self.start_time + '/detect/Sens_train.txt', S_train)
                np.savetxt('Result/' + self.start_time + '/detect/Sens_valid.txt', S_valid)
                np.savetxt('Result/' + self.start_time + '/detect/Sens_test.txt', S_test)

        self.ran_flag = 1
        is_recordmore = 0
        popsize = len(pop)
        pred_label_train = np.zeros([popsize, self.num_train])
        pred_label_valid = np.zeros([popsize, self.num_valid])
        pred_label_test = np.zeros([popsize, self.num_test])
        if self.is_ensemble:
            pred_label_ensemble = np.zeros([popsize, self.num_ensemble])

        pred_logits_train = np.zeros([popsize, self.num_train])
        pred_logits_valid = np.zeros([popsize, self.num_valid])
        pred_logits_test = np.zeros([popsize, self.num_test])
        if self.is_ensemble:
            pred_logits_ensemble = np.zeros([popsize, self.num_ensemble])

        # 这种情况下会修改pop中个体网络的权重值
        # -------- ZQQ - begin -----------

        if test_flag == 1:
            self.use_gpu = torch.cuda.is_available()
            self.use_gpu = False
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            Groups_info = ea.Cal_objectives(self.train_data, self.train_data_norm,
                                              np.array(self.train_y).reshape(1, -1),
                                              np.array(self.train_y).reshape(1, -1), self.cal_sens_name,
                                              2, self.Groups_info['sens_idxs_test'],
                                              plan=self.cal_obj_plan,
                                              dis=np.array(self.train_y).reshape(1, -1),
                                            obj_names=self.objectives_class)

            self.more_info_num = Groups_info["addition_num"]

        # 依次，accuracy Error L2 Individual Group
        if is_recordmore == 0:
            AllObj_valid = np.zeros([popsize, len(self.objectives_class)])
            AllObj_test = np.zeros([popsize, len(self.objectives_class)])
            AllObj_train = np.zeros([popsize, len(self.objectives_class)])
            AllObj_ensemble = np.zeros([popsize, len(self.objectives_class)])
        else:
            AllObj_valid = np.zeros([popsize, self.base_num + self.more_info_num])
            AllObj_test = np.zeros([popsize, self.base_num + self.more_info_num])
            AllObj_train = np.zeros([popsize, self.base_num + self.more_info_num])
            AllObj_ensemble = np.zeros([popsize, self.base_num + self.more_info_num])

        pop_logits_test = np.zeros([popsize, self.test_data_norm.shape[0]])
        for idx in range(popsize):

            individual = pop.Chrom[idx]  # 只是引用，不是复制，还会修改pop.Chrom 网络的值
            # individual = copy.deepcopy(pop.Chrom[idx])  # 不是引用，是复制，不会修改pop.Chrom 网络的值

            if self.use_gpu:
                individual.cuda()
            if test_flag == 1:
                x_train = torch.Tensor(self.train_data_norm)
                y_train = torch.Tensor(np.array(self.train_y))
                y_train = y_train.view(y_train.shape[0], 1)

                x_test = torch.Tensor(self.test_data_norm)
                y_test = torch.Tensor(np.array(self.test_y))
                y_test = y_test.view(y_test.shape[0], 1)

                x_valid = torch.Tensor(self.valid_data_norm)
                y_valid = torch.Tensor(self.valid_y)
                y_valid = y_valid.view(y_valid.shape[0], 1)

                sens_attr = self.cal_sens_name
                group_dicts = self.Groups_info['group_dict_train']
                s_labels = group_dicts[sens_attr[0]][0]

                sens_idxs_train = self.Groups_info['sens_idxs_train']
                s_labels_train = sens_idxs_train[s_labels][0]
                S_train = np.zeros([1, self.num_train])
                S_train[0, s_labels_train] = 1
                s_train = torch.Tensor(S_train)

                sens_idxs_valid = self.Groups_info['sens_idxs_valid']
                s_labels_valid = sens_idxs_valid[s_labels][0]
                S_valid = np.zeros([1, self.num_valid])
                S_valid[0, s_labels_valid] = 1
                s_valid = torch.Tensor(S_valid)

                sens_idxs_test = self.Groups_info['sens_idxs_test']
                sens_idxs_name_test = self.Groups_info['sens_idxs_name_test']
                s_labels_test = sens_idxs_test[s_labels][0]
                S_test = np.zeros([1, self.num_test])
                S_test[0, s_labels_test] = 1
                s_test = torch.Tensor(S_test)

                train_idx = torch.arange(start=0, end=self.train_data_norm.shape[0], step=1)
                valid_idx = torch.arange(start=0, end=self.valid_data_norm.shape[0], step=1)
                test_idx = torch.arange(start=0, end=self.test_data_norm.shape[0], step=1)

                train = TensorDataset(x_train, y_train, s_train[0], train_idx)
                test = TensorDataset(x_test, y_test, s_test[0], test_idx)
                valid = TensorDataset(x_valid, y_valid, s_valid[0], valid_idx)

                train_loader = DataLoader(train, batch_size=self.batch_size, shuffle=True)
                test_loader = DataLoader(test, batch_size=self.batch_size, shuffle=True)
                valid_loader = DataLoader(valid, batch_size=self.batch_size, shuffle=True)

            # optimizer = torch.optim.Adam(individual.parameters(), lr=self.learning_rate,
            #                              weight_decay=self.weight_decay)
            lr = self.learning_rate
            try:
                if pop.ObjV[idx][1] < 0.09:
                    lr = 0.001
            except:
                zqq = 1
            # lr_now = np.max([self.learning_rate * np.power(lr_decay_factor, gen), 0.004])
            lr_now = self.learning_rate
            optimizer = torch.optim.SGD(individual.parameters(), lr=lr_now, momentum=0.9,
                                        weight_decay=self.weight_decay)

            loss_fn = torch.nn.BCEWithLogitsLoss()
            if self.use_gpu:
                loss_fn.cuda()

            if np.random.random() < train_net or loss_type != -1:

                epoch_now = self.epoches

                for epoch in range(epoch_now):
                    individual.train()

                    for i, (x_batch, y_batch, s_batch, data_idx) in enumerate(self.train_loader):
                        if self.use_gpu:
                            x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
                        y_logits, y_pred = individual(x_batch)

                        loss_acc = loss_fn(y_logits, y_batch)
                        if "Individual_fairness" in self.objectives_class or "Group_fairness" in self.objectives_class:
                            group_losses = calcul_all_fairness(self.train_data.loc[data_idx.detach()], y_pred,
                                                           y_batch, self.sensitive_attributions, 2)
                            if np.random.random() > 0.5:
                                loss = loss_acc
                            else:
                                loss = group_losses[self.objectives_class[1]]
                        else:
                            loss = loss_acc

                        if loss_type != -1:
                            if loss_type[idx] == 'Error':
                                loss = loss_acc
                            elif loss_type[idx] == 'Individual_fairness':
                                loss = group_losses['Individual_fairness']
                            elif loss_type[idx] == 'Group_fairness':
                                loss = group_losses['Group_fairness']
                            else:
                                loss = loss_acc

                        optimizer.zero_grad()  # clear gradients for next train
                        loss.backward()  # -> accumulates the gradient (by addition) for each parameter
                        optimizer.step()  # -> update weights and biases

            with torch.no_grad():
                individual.eval()
                # if self.use_gpu:
                #     individual.cuda()
                #     x_test, x_train, x_valid = x_test.cuda(), x_train.cuda(), x_valid.cuda()
                if self.use_gpu:
                    individual.cpu()

                l2_regularization = get_L_regularization(individual, 2)

                # in the test data
                logit_temp, pred_sigmoid_temp = individual(self.x_test)
                logits_test = np.array(pred_sigmoid_temp.detach())
                pred_label_test[idx][:] = get_label(logits_test.reshape(1, -1).copy())
                pop_logits_test[idx][:] = logits_test.reshape(1, -1)
                pred_logits_test[idx][:] = logits_test.reshape(1, -1)
                Groups_info_test = ea.Cal_objectives(
                    self.test_data,
                    self.test_data_norm,
                    logits_test, self.y_test,
                    self.sensitive_attributions,
                    2, self.Groups_info['sens_idxs_test'],
                    plan=self.cal_obj_plan,
                    dis=logit_temp,
                    obj_is_logits=self.obj_is_logits,
                    dist_mat = self.dist_mat["dist_mat_test"],
                    obj_names=self.objectives_class)

                # in the train data
                logit_temp, pred_sigmoid_temp = individual(self.x_train)
                logits_train = np.array(pred_sigmoid_temp.detach())
                pred_label_train[idx][:] = get_label(logits_train.reshape(1, -1).copy())
                pred_logits_train[idx][:] = logits_train.reshape(1, -1)
                Groups_info_train = ea.Cal_objectives(self.train_data,
                                                self.train_data_norm,
                                                logits_train, self.y_train,
                                                self.sensitive_attributions,
                                                2, self.Groups_info['sens_idxs_train'],
                                                plan=self.cal_obj_plan,
                                                dis=logit_temp,
                                                obj_is_logits=self.obj_is_logits,
                                                dist_mat = self.dist_mat["dist_mat_train"],
                                                obj_names=self.objectives_class)

                # in the validation data
                logit_temp, pred_sigmoid_temp = individual(self.x_valid)
                logits_valid = np.array(pred_sigmoid_temp.detach())
                pred_label_valid[idx][:] = get_label(logits_valid.reshape(1, -1).copy())
                pred_logits_valid[idx][:] = logits_valid.reshape(1, -1)
                Groups_info_valid = ea.Cal_objectives(self.valid_data,
                                                    self.valid_data_norm,
                                                    logits_valid, self.y_valid,
                                                    self.sensitive_attributions,
                                                    2, self.Groups_info['sens_idxs_valid'],
                                                    plan=self.cal_obj_plan,
                                                    dis=logit_temp,
                                                    obj_is_logits=self.obj_is_logits,
                                                    dist_mat = self.dist_mat["dist_mat_valid"],
                                                    obj_names=self.objectives_class)

                if self.is_ensemble:
                    # in the ensemble data
                    logit_temp, pred_sigmoid_temp = individual(self.x_ensemble)
                    logits_ensemble = np.array(pred_sigmoid_temp.detach())
                    pred_label_ensemble[idx][:] = get_label(logits_ensemble.reshape(1, -1).copy())
                    pred_logits_ensemble[idx][:] = logits_ensemble.reshape(1, -1)
                    Groups_info_ensemble = ea.Cal_objectives(self.ensemble_data,
                                                          self.ensemble_data_norm,
                                                          logits_ensemble, self.y_ensemble,
                                                          self.sensitive_attributions,
                                                          2, self.Groups_info['sens_idxs_ensemble'],
                                                          plan=self.cal_obj_plan,
                                                          dis=logit_temp,
                                                          obj_is_logits=self.obj_is_logits,
                                                          dist_mat=self.dist_mat["dist_mat_ensemble"],
                                                          obj_names=self.objectives_class)

                # print('all is ok')
                ####################################################################################################
                # # in test data
                # print('The information in test data: ')
                # print('  accuracy: %.4f, MSE: %.4f individual fairness: %.5f, group fairness: %.5f\n'
                #       % (accuracy_test, accuracy_loss_test, individual_fairness_test, group_fairness_test))
                #
                # ####################################################################################################
                # # in train data
                # print('The formation in train data: ')
                # print('  accuracy: %.4f, MSE: %.4f, individual fairness: %.5f, group fairness: %.5f\n'
                #       % (accuracy_train, accuracy_loss_train, individual_fairness_train, group_fairness_train))
                ####################################################################################################

                # Groups_info.append(Groups_train)
                #
                if is_recordmore == 0:
                    AllObj_train[idx][:] = get_obj_vals(Groups_info_train, self.objectives_class)[0]
                    AllObj_test[idx][:] = get_obj_vals(Groups_info_test, self.objectives_class)[0]
                    AllObj_valid[idx][:] = get_obj_vals(Groups_info_valid, self.objectives_class)[0]
                    if self.is_ensemble:
                        AllObj_ensemble[idx][:] = get_obj_vals(Groups_info_ensemble, self.objectives_class)[0]
                else:
                    temp = np.zeros([1, self.base_num + self.more_info_num])
                    temp[0][0:self.base_num] = [accuracy_train, accuracy_loss_train, l2_regularization,
                                    individual_fairness_train, group_fairness_train, Demographic_parity_train, FPR_train, FNR_train, Predictive_parity_train, Dwork_value_train]
                    temp[0][self.base_num:] = more_vals_train
                    AllObj_train[idx][:] = temp

                    temp = np.zeros([1, self.base_num + self.more_info_num])
                    temp[0][0:self.base_num] = [accuracy_test, accuracy_loss_test, l2_regularization, individual_fairness_test,
                                    group_fairness_test, Demographic_parity_test, FPR_test, FNR_test, Predictive_parity_test, Dwork_value_test]
                    temp[0][self.base_num:] = more_vals_test
                    AllObj_test[idx][:] = temp

                    temp = np.zeros([1, self.base_num + self.more_info_num])
                    temp[0][0:self.base_num] = [accuracy_valid, accuracy_loss_valid, l2_regularization,
                                    individual_fairness_valid, group_fairness_valid, Demographic_parity_valid, FPR_valid, FNR_valid, Predictive_parity_valid, Dwork_value_valid]
                    temp[0][self.base_num:] = more_vals_valid
                    AllObj_valid[idx][:] = temp

                    if self.is_ensemble:
                        temp = np.zeros([1, self.base_num + self.more_info_num])
                        temp[0][0:self.base_num] = [accuracy_ensemble, accuracy_loss_ensemble, l2_regularization,
                                                    individual_fairness_ensemble, group_fairness_ensemble,
                                                    Demographic_parity_ensemble, FPR_ensemble, FNR_ensemble,
                                                    Predictive_parity_ensemble, Dwork_value_ensemble]
                        temp[0][self.base_num:] = more_vals_ensemble
                        AllObj_ensemble[idx][:] = temp

        pop.CV = np.zeros([popsize, 1])

        pop.ObjV = AllObj_valid

        pop.ObjV_train = AllObj_train
        pop.ObjV_valid = AllObj_valid
        pop.ObjV_test = AllObj_test
        pop.ObjV_ensemble = AllObj_ensemble if self.is_ensemble else None

        pop.pred_label_train = pred_label_train
        pop.pred_label_valid = pred_label_valid
        pop.pred_label_test = pred_label_test
        pop.pred_label_ensemble = pred_label_ensemble if self.is_ensemble else None

        pop.pred_logits_train = pred_logits_train
        pop.pred_logits_valid = pred_logits_valid
        pop.pred_logits_test = pred_logits_test
        pop.pred_logits_ensemble = pred_logits_ensemble if self.is_ensemble else None

        return AllObj_train, AllObj_valid, AllObj_test

    def model_test1(self, pop, kfold=0, gen=0, dirName=None, use_GAN=False, adv_model_DE=None, adv_model_EO=None,
                    problem=None, ndSort=None):  # 目标函数
        # 只对一个adv
        # kfold = 0 : 全部的train训练model
        # kfold！= 0 : 将train进行kfold并 k 为输入的数值
        start_time = time.time()
        if self.ran_flag == 0:
            Groups_info = self.Groups_info
            sens_attr = self.cal_sens_name
            group_dicts = Groups_info['group_dict_train']
            s_labels = group_dicts[sens_attr[0]][0]

            sens_idxs_train = Groups_info['sens_idxs_train']
            s_labels_train = sens_idxs_train[s_labels][0]
            S_train = np.zeros([1, self.num_train])
            S_train[0, s_labels_train] = 1

            sens_idxs_valid = Groups_info['sens_idxs_valid']
            s_labels_valid = sens_idxs_valid[s_labels][0]
            S_valid = np.zeros([1, self.num_valid])
            S_valid[0, s_labels_valid] = 1

            sens_idxs_test = Groups_info['sens_idxs_test']
            s_labels_test = sens_idxs_test[s_labels][0]
            S_test = np.zeros([1, self.num_test])
            S_test[0, s_labels_test] = 1

            np.savetxt('Result/' + self.start_time + '/detect/Sens_train.txt', S_train)
            np.savetxt('Result/' + self.start_time + '/detect/Sens_valid.txt', S_valid)
            np.savetxt('Result/' + self.start_time + '/detect/Sens_test.txt', S_test)

        self.ran_flag = 1
        is_recordmore = 1
        popsize = len(pop)
        pred_label_train = np.zeros([popsize, self.num_train])
        pred_label_valid = np.zeros([popsize, self.num_valid])
        pred_label_test = np.zeros([popsize, self.num_test])

        pred_logits_train = np.zeros([popsize, self.num_train])
        pred_logits_valid = np.zeros([popsize, self.num_valid])
        pred_logits_test = np.zeros([popsize, self.num_test])

        # 这种情况下会修改pop中个体网络的权重值
        # -------- ZQQ - begin -----------

        use_gpu = torch.cuda.is_available()
        use_gpu = False
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _, _, _, _, _, more_vals_test = ea.Cal_objectives(self.train_data, self.train_data_norm,
                                                          np.array(self.train_y).reshape(1, -1),
                                                          np.array(self.train_y).reshape(1, -1), self.cal_sens_name,
                                                          2, self.Groups_info['sens_idxs_test'], plan=self.cal_obj_plan,
                                                          dis=np.array(self.train_y).reshape(1, -1))
        self.more_info_num = len(more_vals_test)

        # 依次，accuracy Error L2 Individual Group
        if is_recordmore == 0:
            AllObj_valid = np.zeros([popsize, 5])
            AllObj_test = np.zeros([popsize, 5])
            AllObj_train = np.zeros([popsize, 5])
        else:
            AllObj_valid = np.zeros([popsize, 5 + self.more_info_num])
            AllObj_test = np.zeros([popsize, 5 + self.more_info_num])
            AllObj_train = np.zeros([popsize, 5 + self.more_info_num])

        pop_logits_test = np.zeros([popsize, self.test_data_norm.shape[0]])
        record_loss = np.zeros([popsize, self.epoches * 10])
        record_test = np.zeros([popsize, self.epoches * 10])
        # Groups_info = []

        # for ti in range(10):
        #     print(ti)
        loss_indi = np.zeros([popsize, self.epoches])
        test_indi = np.zeros([popsize, self.epoches])
        mse_indi = np.zeros([popsize, self.epoches])
        unfairness_indi = np.zeros([popsize, self.epoches])
        mutator = ea.Mutation_NN2(mu=0., var=0.01)
        for idx in range(popsize):
            use_GAN = True
            # individual = pop.Chrom[idx]  # 只是引用，不是复制，还会修改pop.Chrom 网络的值
            pop_pop = copy.deepcopy(pop)
            individual = copy.deepcopy(pop.Chrom[idx])  # 不是引用，是复制，不会修改pop.Chrom 网络的值
            adversary_EO = copy.deepcopy(adv_model_EO)
            adversary_DE = copy.deepcopy(adv_model_DE)

            x_train = torch.Tensor(self.train_data_norm)
            y_train = torch.Tensor(np.array(self.train_y))
            y_train = y_train.view(y_train.shape[0], 1)

            x_test = torch.Tensor(self.test_data_norm)
            y_test = torch.Tensor(np.array(self.test_y))
            y_test = y_test.view(y_test.shape[0], 1)

            x_valid = torch.Tensor(self.valid_data_norm)
            y_valid = torch.Tensor(self.valid_y)
            y_valid = y_valid.view(y_valid.shape[0], 1)

            Groups_info = self.Groups_info

            sens_attr = self.cal_sens_name
            group_dicts = Groups_info['group_dict_train']
            s_labels = group_dicts[sens_attr[0]][0]
            # print(s_labels)

            sens_idxs_train = Groups_info['sens_idxs_train']
            s_labels_train = sens_idxs_train[s_labels][0]
            S_train = np.zeros([1, self.num_train])
            S_train[0, s_labels_train] = 1
            s_train = torch.Tensor(S_train)

            sens_idxs_valid = Groups_info['sens_idxs_valid']
            s_labels_valid = sens_idxs_valid[s_labels][0]
            S_valid = np.zeros([1, self.num_valid])
            S_valid[0, s_labels_valid] = 1

            sens_idxs_test = Groups_info['sens_idxs_test']
            sens_idxs_name_test = Groups_info['sens_idxs_name_test']
            s_labels_test = sens_idxs_test[s_labels][0]
            S_test = np.zeros([1, self.num_test])
            S_test[0, s_labels_test] = 1
            s_test = torch.Tensor(S_test)

            loss_fn = torch.nn.BCELoss()
            if self.use_gpu:
                loss_fn.cuda()

            if self.use_gpu:
                individual.cuda()
                if use_GAN:
                    adversary_EO.cuda()
                    adversary_DE.cuda()
            pop_optimizer = torch.optim.Adam(individual.parameters(), lr=self.learning_rate,
                                             weight_decay=self.weight_decay)

            if use_GAN:
                if adversary_EO.no_targets == 1:
                    adv_optimizer_EO = torch.optim.Adam(adversary_EO.parameters(), lr=self.learning_rate)
                else:
                    adv_optimizer_EO = torch.optim.Adam(adversary_EO.parameters(), lr=self.learning_rate)

                if adversary_DE.no_targets == 1:
                    adv_optimizer_DE = torch.optim.Adam(adversary_DE.parameters(), lr=self.learning_rate)
                else:
                    adv_optimizer_DE = torch.optim.Adam(adversary_DE.parameters(), lr=self.learning_rate)

            train = TensorDataset(x_train, y_train, s_train[0])
            train_loader = DataLoader(train, batch_size=self.batch_size, shuffle=True)
            test = TensorDataset(x_test, y_test, s_test[0])
            test_loader = DataLoader(test, batch_size=self.batch_size, shuffle=True)
            use_GAN = False
            self.GAN_alpha = 0.3
            for epoch in range(self.epoches):
                individual.train()
                avg_loss_test = 0.0
                avg_loss_train = 0.0

                if epoch > -1:
                    use_GAN = True
                    self.GAN_alpha = 20

                for i, (x_batch, y_batch, s_batch) in enumerate(train_loader):
                    if self.use_gpu:
                        x_batch, y_batch, s_batch = x_batch.cuda(), y_batch.cuda(), s_batch.cuda()
                    y_logits, y_pred = individual(x_batch)
                    loss_P = loss_fn(y_pred, y_batch)
                    avg_loss_train += loss_P.item() / len(train_loader)
                    # calculate L2 regular
                    # reg_loss = None
                    # for param in individual.parameters():
                    #     if reg_loss is None:
                    #         reg_loss = 0.5 * torch.sum(param ** 2)
                    #     else:
                    #         reg_loss = reg_loss + 0.5 * param.norm(2) ** 2
                    # loss_P += self.L_regul * reg_loss
                    if use_GAN:
                        pred_z_logit_EO, pred_z_prob_EO = adversary_EO(y_logits, y_batch)
                        loss_A_EO = loss_fn(pred_z_prob_EO, s_batch.reshape(-1, 1))
                        adv_optimizer_EO.zero_grad()
                        pop_optimizer.zero_grad()
                        loss_A_EO.backward(retain_graph=True)

                        # grads_w_La = get_gen_grads(individual, loss_A_EO, True)
                        # grads_w_Lp = get_gen_grads(individual, loss_P, True)

                        # grad_w_La = concat_grad(individual)
                        # proba = np.random.rand(2)
                        # proba /= np.sum(proba)
                        # grads_list = []
                        # grads_list.append(grads_w_La.cpu().detach().numpy())
                        # grads_list.append(grads_w_Lp.cpu().detach().numpy())
                        # constraints, bounds = make_constraints(2)
                        # grads_list = np.asarray(grads_list).T
                        # result = minimize(steep_direct_cost, proba, args=grads_list, jac=steep_direc_cost_deriv,
                        #                   bounds=bounds, constraints=constraints, method='SLSQP',
                        #                   options={'ftol': 1e-9, 'disp': False})
                        # proba = result.x
                        # print(proba)
                    pop_optimizer.zero_grad()
                    loss_P.backward()
                    if use_GAN:
                        grad_w_Lp = concat_grad(individual)
                        proj_grad = project_grad(grad_w_Lp, grad_w_La)
                        alph1 = 0
                        grad_w_Lp = grad_w_Lp - alph1 * proj_grad - self.GAN_alpha * grad_w_La

                        replace_grad(individual, grad_w_Lp)
                    pop_optimizer.step()
                    if use_GAN:
                        adv_optimizer_EO.step()

                with torch.no_grad():
                    individual.eval()
                    for i, (x_batch, y_batch, s_batch) in enumerate(test_loader):
                        if self.use_gpu:
                            x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
                        _, y_pred = individual(x_batch)
                        loss_test = loss_fn(y_pred, y_batch)
                        avg_loss_test += loss_test.item() / len(test_loader)
                    test_indi[idx, epoch] = avg_loss_test
                    loss_indi[idx, epoch] = avg_loss_train
                    if np.mod(epoch, 50000) == 0:
                        if epoch == 0:
                            continue
                        individual = mutator.do(individual)
                        # for name, param in individual.named_parameters():
                        #     weighs = np.array(param.detach())
                        #     # print(name)
                        #     # print('  before: ', weighs)
                        #     weighs += np.random.normal(loc=0.0, scale=0.01, size=param.shape)
                        #     individual.state_dict()[name].data.copy_(torch.Tensor(weighs))
                    individual.eval()
                    if self.use_gpu:
                        individual.cpu()
                        # if np.mod(gen, 20) == 0:
                        #     plot_decision_boundary(individual, x_train, y_train, 1-S_train[0], np.array([gen, idx]), dirName=dirName)

                    l2_regularization = 0.0
                    for param in individual.parameters():
                        l2_regularization += torch.norm(param, 2)  # L2 正则化

                    logit_temp, pred_sigmoid_temp = individual(x_test)
                    logits_test = np.array(pred_sigmoid_temp.detach())
                    accuracy_test, accuracy_loss_test, individual_fairness_test, group_fairness_test, Groups_test, more_vals_test = ea.Cal_objectives(
                        self.test_data,
                        self.test_data_norm,
                        logits_test, y_test,
                        self.cal_sens_name,
                        2, self.Groups_info['sens_idxs_test'],
                        plan=self.cal_obj_plan,
                        dis=logit_temp)

                    logit_temp, pred_sigmoid_temp = individual(x_train)
                    logits_train = np.array(pred_sigmoid_temp.detach())
                    accuracy_train, accuracy_loss_train, individual_fairness_train, group_fairness_train, Groups_train, more_vals_train = ea.Cal_objectives(
                        self.train_data,
                        self.train_data_norm,
                        logits_train, y_train,
                        self.cal_sens_name,
                        2, self.Groups_info['sens_idxs_train'],
                        plan=self.cal_obj_plan,
                        dis=logit_temp)

                    logit_temp, pred_sigmoid_temp = individual(x_valid)
                    logits_valid = np.array(pred_sigmoid_temp.detach())
                    accuracy_valid, accuracy_loss_valid, individual_fairness_valid, group_fairness_valid, Groups_valid, more_vals_valid = ea.Cal_objectives(
                        self.valid_data,
                        self.valid_data_norm,
                        logits_valid, y_valid,
                        self.cal_sens_name,
                        2, self.Groups_info['sens_idxs_valid'],
                        plan=self.cal_obj_plan,
                        dis=logit_temp)
                    pop_pop.ObjV_train = AllObj_train
                    pop_pop.ObjV_valid = AllObj_valid
                    pop_pop.ObjV_test = AllObj_test
                    if is_recordmore == 0:
                        AllObj_train[idx][:] = np.array(
                            [accuracy_train, accuracy_loss_train, l2_regularization, individual_fairness_train,
                             group_fairness_train])
                        AllObj_test[idx][:] = np.array(
                            [accuracy_test, accuracy_loss_test, l2_regularization, individual_fairness_test,
                             group_fairness_test])
                        AllObj_valid[idx][:] = np.array(
                            [accuracy_valid, accuracy_loss_valid, l2_regularization, individual_fairness_valid,
                             group_fairness_valid])
                    else:
                        temp = np.zeros([1, 5 + self.more_info_num])
                        temp[0][0:5] = [accuracy_train, accuracy_loss_train, l2_regularization,
                                        individual_fairness_train, group_fairness_train]
                        temp[0][5:] = more_vals_train
                        AllObj_train[idx][:] = temp

                        temp = np.zeros([1, 5 + self.more_info_num])
                        temp[0][0:5] = [accuracy_test, accuracy_loss_test, l2_regularization, individual_fairness_test,
                                        group_fairness_test]
                        temp[0][5:] = more_vals_test
                        AllObj_test[idx][:] = temp

                        temp = np.zeros([1, 5 + self.more_info_num])
                        temp[0][0:5] = [accuracy_valid, accuracy_loss_valid, l2_regularization,
                                        individual_fairness_valid, group_fairness_valid]
                        temp[0][5:] = more_vals_valid
                        AllObj_valid[idx][:] = temp
                    pop_pop.ObjV = np.zeros([popsize, len(self.objectives_class)])
                    pop_pop.ObjV_train[idx][:] = AllObj_train[idx][:]
                    pop_pop.ObjV_valid[idx][:] = AllObj_valid[idx][:]
                    pop_pop.ObjV_test[idx][:] = AllObj_test[idx][:]
                    pop_pop.Chrom[idx] = individual
                    plot_decision_boundary4(pop_pop, problem, dirName=dirName, epoch=epoch, idx=idx)
                    mse_indi[idx][epoch] = accuracy_loss_test
                    unfairness_indi[idx][epoch] = individual_fairness_test

        with torch.no_grad():
            # plt.plot(np.array(range(self.epoches)), np.mean(loss_indi, axis=0), label='train')
            # plt.plot(np.array(range(self.epoches)), np.mean(test_indi, axis=0), label='test')
            # plt.legend(loc='best', fontsize=15)
            np.savetxt(
                "geatpy/data/TwoGaussian_largerer0_alpha{}_2terms_mse".format(self.GAN_alpha).replace('.', '') + '.txt',
                mse_indi)
            np.savetxt("geatpy/data/TwoGaussian_largerer0_alpha{}_2terms_fairness".format(self.GAN_alpha).replace('.',
                                                                                                                  '') + '.txt',
                       unfairness_indi)
            # plt.show()

            # with torch.no_grad():
            #
            #     # if self.use_gpu:
            #     #     individual.cuda()
            #     #     x_test, x_train, x_valid = x_test.cuda(), x_train.cuda(), x_valid.cuda()
            #     if self.use_gpu:
            #         individual.cpu()
            #     # if np.mod(gen, 50) == 0:
            #     #     plot_decision_boundary(individual, x_train, y_train, 1 - S_train[0], np.array([gen, idx]),
            #     #                            dirName=dirName)
            #     l2_regularization = 0.0
            #     for param in individual.parameters():
            #         # l1_regularization += torch.norm(param, 1)  # L1正则化
            #         # print(param)
            #         l2_regularization += torch.norm(param, 2)  # L2 正则化
            #     logit_temp, pred_sigmoid_temp = individual(x_test)
            #     logits_test = np.array(pred_sigmoid_temp.detach())
            #     pred_label_test[idx][:] = get_label(logits_test.reshape(1, -1).copy())
            #     pop_logits_test[idx][:] = logits_test.reshape(1, -1)
            #     pred_logits_test[idx][:] = logits_test.reshape(1, -1)
            #     accuracy_test, accuracy_loss_test, individual_fairness_test, group_fairness_test, Groups_test, more_vals_test = ea.Cal_objectives(
            #         self.test_data,
            #         self.test_data_norm,
            #         logits_test, y_test,
            #         self.cal_sens_name,
            #         2, self.Groups_info['sens_idxs_test'],
            #         plan=self.cal_obj_plan,
            #     dis=logit_temp)
            #
            #     logit_temp, pred_sigmoid_temp = individual(x_train)
            #     logits_train = np.array(pred_sigmoid_temp.detach())
            #     pred_label_train[idx][:] = get_label(logits_train.reshape(1, -1).copy())
            #     pred_logits_train[idx][:] = logits_train.reshape(1, -1)
            #     accuracy_train, accuracy_loss_train, individual_fairness_train, group_fairness_train, Groups_train, more_vals_train = ea.Cal_objectives(
            #         self.train_data,
            #         self.train_data_norm,
            #         logits_train, y_train,
            #         self.cal_sens_name,
            #         2, self.Groups_info['sens_idxs_train'],
            #     plan=self.cal_obj_plan,
            #     dis=logit_temp)
            #
            #     logit_temp, pred_sigmoid_temp = individual(x_valid)
            #     logits_valid = np.array(pred_sigmoid_temp.detach())
            #     pred_label_valid[idx][:] = get_label(logits_valid.reshape(1, -1).copy())
            #     pred_logits_valid[idx][:] = logits_valid.reshape(1, -1)
            #     accuracy_valid, accuracy_loss_valid, individual_fairness_valid, group_fairness_valid, Groups_valid, more_vals_valid = ea.Cal_objectives(
            #         self.valid_data,
            #         self.valid_data_norm,
            #         logits_valid, y_valid,
            #         self.cal_sens_name,
            #         2, self.Groups_info['sens_idxs_valid'],
            #     plan=self.cal_obj_plan,
            #     dis=logit_temp)
            #
            #     # print('all is ok')
            #     ####################################################################################################
            #     # # in test data
            #     # print('The information in test data: ')
            #     # print('  accuracy: %.4f, MSE: %.4f individual fairness: %.5f, group fairness: %.5f\n'
            #     #       % (accuracy_test, accuracy_loss_test, individual_fairness_test, group_fairness_test))
            #     #
            #     # ####################################################################################################
            #     # # in train data
            #     # print('The formation in train data: ')
            #     # print('  accuracy: %.4f, MSE: %.4f, individual fairness: %.5f, group fairness: %.5f\n'
            #     #       % (accuracy_train, accuracy_loss_train, individual_fairness_train, group_fairness_train))
            #     ####################################################################################################
            #
            #     # Groups_info.append(Groups_train)
            #     #
            #     if is_recordmore == 0:
            #         AllObj_train[idx][:] = np.array(
            #             [accuracy_train, accuracy_loss_train, l2_regularization, individual_fairness_train,
            #              group_fairness_train])
            #         AllObj_test[idx][:] = np.array(
            #             [accuracy_test, accuracy_loss_test, l2_regularization, individual_fairness_test,
            #              group_fairness_test])
            #         AllObj_valid[idx][:] = np.array(
            #             [accuracy_valid, accuracy_loss_valid, l2_regularization, individual_fairness_valid,
            #              group_fairness_valid])
            #     else:
            #         temp = np.zeros([1, 5 + self.more_info_num])
            #         temp[0][0:5] = [accuracy_train, accuracy_loss_train, l2_regularization,
            #                         individual_fairness_train, group_fairness_train]
            #         temp[0][5:] = more_vals_train
            #         AllObj_train[idx][:] = temp
            #
            #         temp = np.zeros([1, 5 + self.more_info_num])
            #         temp[0][0:5] = [accuracy_test, accuracy_loss_test, l2_regularization, individual_fairness_test,
            #                         group_fairness_test]
            #         temp[0][5:] = more_vals_test
            #         AllObj_test[idx][:] = temp
            #
            #         temp = np.zeros([1, 5 + self.more_info_num])
            #         temp[0][0:5] = [accuracy_valid, accuracy_loss_valid, l2_regularization,
            #                         individual_fairness_valid, group_fairness_valid]
            #         temp[0][5:] = more_vals_valid
            #         AllObj_valid[idx][:] = temp
            #
            #     # AllObj_test[idx][:] = np.array([accuracy_loss_test, individual_fairness_train, group_fairness_train])
            #     # PopObj[idx][:] = np.array([individual_fairness, group_fairness])

        # mean_loss = np.mean(record_loss, axis=0)
        # mean_acc = np.mean(record_test, axis=0)
        # np.savetxt("compas_wd2_loss_lr0001.txt", record_loss)
        # np.savetxt("compas_wd2_acc_lr0001.txt", record_test)
        # plt.plot(np.array(range(self.epoches*10)), mean_loss)
        # plt.plot(np.array(range(self.epoches*10)), mean_acc)
        # end_time = time.time()
        # print('cost: ', end_time-start_time)
        # plt.show()
        # print('draw')

        pop.CV = np.zeros([popsize, 1])
        PopObj = AllObj_valid.copy()
        delete_list = []
        if 'accuracy' not in self.objectives_class:
            delete_list.append(0)
        if 'l2' not in self.objectives_class:
            delete_list.append(2)
        if 'Error' not in self.objectives_class:
            delete_list.append(1)
        if 'individual' not in self.objectives_class:
            delete_list.append(3)
        if 'group' not in self.objectives_class:
            delete_list.append(4)
        if is_recordmore == 1:
            for i in range(5, 5 + self.more_info_num):
                delete_list.append(i)
        pop.ObjV = np.delete(PopObj, delete_list, 1)  # 把求得的目标函数值赋值给种群pop的ObjV.
        if 'accuracy' in self.objectives_class:
            pop.ObjV[:, 0] = 1 - pop.ObjV[:, 0]
        pop.ObjV_train = AllObj_train
        pop.ObjV_valid = AllObj_valid
        pop.ObjV_test = AllObj_test
        pop.pred_label_train = pred_label_train
        pop.pred_label_valid = pred_label_valid
        pop.pred_label_test = pred_label_test
        pop.pred_logits_train = pred_logits_train
        pop.pred_logits_valid = pred_logits_valid
        pop.pred_logits_test = pred_logits_test

        endtime = time.time()
        # if self.use_gpu:
        #     print('use gpu')
        # else:
        #     print('not use gpu')
        # print('calculate objectives run time:', endtime - start_time)

        return AllObj_train, AllObj_valid, AllObj_test

    def model_test2(self, pop, kfold=0, gen=0, dirName=None, use_GAN=False, adv_models=None, problem=None,
                    ndSort=None):  # 目标函数
        # 针对多个adverasrys
        # 具体的，每一个epoch的batch下：先更新adversarial models 再更新predictors
        # kfold = 0 : 全部的train训练model
        # kfold！= 0 : 将train进行kfold并 k 为输入的数值
        start_time = time.time()
        if self.ran_flag == 0:
            Groups_info = self.Groups_info
            sens_attr = self.cal_sens_name
            group_dicts = Groups_info['group_dict_train']
            s_labels = group_dicts[sens_attr[0]][0]

            sens_idxs_train = Groups_info['sens_idxs_train']
            s_labels_train = sens_idxs_train[s_labels][0]
            S_train = np.zeros([1, self.num_train])
            S_train[0, s_labels_train] = 1

            sens_idxs_valid = Groups_info['sens_idxs_valid']
            s_labels_valid = sens_idxs_valid[s_labels][0]
            S_valid = np.zeros([1, self.num_valid])
            S_valid[0, s_labels_valid] = 1

            sens_idxs_test = Groups_info['sens_idxs_test']
            s_labels_test = sens_idxs_test[s_labels][0]
            S_test = np.zeros([1, self.num_test])
            S_test[0, s_labels_test] = 1

            np.savetxt('Result/' + self.start_time + '/detect/Sens_train.txt', S_train)
            np.savetxt('Result/' + self.start_time + '/detect/Sens_valid.txt', S_valid)
            np.savetxt('Result/' + self.start_time + '/detect/Sens_test.txt', S_test)

        self.ran_flag = 1
        is_recordmore = 1
        popsize = len(pop)
        pred_label_train = np.zeros([popsize, self.num_train])
        pred_label_valid = np.zeros([popsize, self.num_valid])
        pred_label_test = np.zeros([popsize, self.num_test])

        pred_logits_train = np.zeros([popsize, self.num_train])
        pred_logits_valid = np.zeros([popsize, self.num_valid])
        pred_logits_test = np.zeros([popsize, self.num_test])

        # 这种情况下会修改pop中个体网络的权重值
        # -------- ZQQ - begin -----------

        self.use_gpu = torch.cuda.is_available()
        self.use_gpu = False
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _, _, _, _, _, more_vals_test = ea.Cal_objectives(self.train_data, self.train_data_norm,
                                                          np.array(self.train_y).reshape(1, -1),
                                                          np.array(self.train_y).reshape(1, -1), self.cal_sens_name,
                                                          2, self.Groups_info['sens_idxs_test'], plan=self.cal_obj_plan,
                                                          dis=np.array(self.train_y).reshape(1, -1))
        self.more_info_num = len(more_vals_test)

        # 依次，accuracy MSE L2 Individual Group
        if is_recordmore == 0:
            AllObj_valid = np.zeros([popsize, 5])
            AllObj_test = np.zeros([popsize, 5])
            AllObj_train = np.zeros([popsize, 5])
        else:
            AllObj_valid = np.zeros([popsize, 5 + self.more_info_num])
            AllObj_test = np.zeros([popsize, 5 + self.more_info_num])
            AllObj_train = np.zeros([popsize, 5 + self.more_info_num])

        pop_logits_test = np.zeros([popsize, self.test_data_norm.shape[0]])
        record_loss = np.zeros([popsize, self.epoches * 10])
        record_test = np.zeros([popsize, self.epoches * 10])
        # Groups_info = []

        # for ti in range(10):
        #     print(ti)
        loss_indi = np.zeros([popsize, self.epoches])
        test_indi = np.zeros([popsize, self.epoches])
        mse_indi = np.zeros([popsize, self.epoches])
        unfairness_indi = np.zeros([popsize, self.epoches])
        mutator = ea.Mutation_NN2(mu=0., var=0.01)

        x_train = torch.Tensor(self.train_data_norm)
        y_train = torch.Tensor(np.array(self.train_y))
        y_train = y_train.view(y_train.shape[0], 1)

        x_test = torch.Tensor(self.test_data_norm)
        y_test = torch.Tensor(np.array(self.test_y))
        y_test = y_test.view(y_test.shape[0], 1)

        x_valid = torch.Tensor(self.valid_data_norm)
        y_valid = torch.Tensor(self.valid_y)
        y_valid = y_valid.view(y_valid.shape[0], 1)

        Groups_info = self.Groups_info

        sens_attr = self.cal_sens_name
        group_dicts = Groups_info['group_dict_train']
        s_labels = group_dicts[sens_attr[0]][0]

        sens_idxs_train = Groups_info['sens_idxs_train']
        s_labels_train = sens_idxs_train[s_labels][0]
        S_train = np.zeros([1, self.num_train])
        S_train[0, s_labels_train] = 1
        s_train = torch.Tensor(S_train)

        sens_idxs_valid = Groups_info['sens_idxs_valid']
        s_labels_valid = sens_idxs_valid[s_labels][0]
        S_valid = np.zeros([1, self.num_valid])
        S_valid[0, s_labels_valid] = 1
        s_valid = torch.Tensor(S_valid)

        sens_idxs_test = Groups_info['sens_idxs_test']
        sens_idxs_name_test = Groups_info['sens_idxs_name_test']
        s_labels_test = sens_idxs_test[s_labels][0]
        S_test = np.zeros([1, self.num_test])
        S_test[0, s_labels_test] = 1
        s_test = torch.Tensor(S_test)

        train = TensorDataset(x_train, y_train, s_train[0])
        train_loader = DataLoader(train, batch_size=self.batch_size, shuffle=True)
        test = TensorDataset(x_test, y_test, s_test[0])
        test_loader = DataLoader(test, batch_size=self.batch_size, shuffle=True)
        valid = TensorDataset(x_valid, y_valid, s_valid[0])
        valid_loader = DataLoader(valid, batch_size=self.batch_size, shuffle=True)

        proba_adv = np.random.random(len(adv_models) + 1)
        proba_adv /= np.sum(proba_adv)
        constraints, bounds = make_constraints(len(adv_models) + 1)
        ###########################################################################################################################

        # get predictor optimizers
        predictor_optmizers = []
        predictor_models = pop.Chrom
        for i in range(len(pop)):
            predictor_model = predictor_models[i]
            predictor_optmizer = torch.optim.Adam(predictor_model.parameters(), lr=self.learning_rate)
            predictor_optmizers.append(predictor_optmizer)

        # get individual optimizers
        adversary_optimizers = []
        for i in range(len(adv_models)):
            adversary_model = adv_models[i]
            if adversary_model.no_targets == 1:
                # for DE
                adv_optimizer = torch.optim.Adam(adversary_model.parameters(), lr=self.learning_rate)
            else:
                # for EO
                adv_optimizer = torch.optim.Adam(adversary_model.parameters(), lr=self.learning_rate)
            adversary_optimizers.append(adv_optimizer)

        loss_BCE_pred = torch.nn.BCELoss()
        loss_BCE_adv = torch.nn.BCELoss()

        pop_pop = copy.deepcopy(pop)

        for epoch in range(self.epoches):
            for idx, (x_batch, y_batch, s_batch) in enumerate(train_loader):

                # update adversarial models
                # for each adversarial model
                iteration_update_adv = 3
                for it in range(iteration_update_adv):
                    for advserary, advserary_optimizer in zip(adv_models, adversary_optimizers):
                        losses_adverary = []
                        losses_adv_float = []
                        advserary.zero_grad()
                        # get all losses based on predictors
                        for individual in predictor_models:
                            individual.zero_grad()
                            indiv_logits, indiv_pred = individual(x_batch)
                            pred_z_logit, pred_z_prob = advserary(indiv_logits, y_batch)

                            loss_A = loss_BCE_adv(pred_z_prob, s_batch.reshape(-1, 1))
                            losses_adverary.append(loss_A)
                            losses_adv_float.append(loss_A.item())

                        losses_adv_floats = torch.FloatTensor(losses_adv_float)
                        gman_aplha = 0.8
                        proba_indiv = torch.nn.functional.softmax(gman_aplha * losses_adv_floats,
                                                                  dim=0).detach().cpu().numpy()
                        # proba_indivs = np.zeros_like(proba_indiv)
                        # proba_indivs[0] = 1
                        # proba_indiv = np.array([0.5, -0.5, 0])
                        loss_advs = 0
                        for loss_weight in zip(losses_adverary, proba_indiv):
                            loss_advs += loss_weight[0] * float(loss_weight[1])

                        advserary.zero_grad()
                        loss_advs.backward()
                        advserary_optimizer.step()

                # update predictor models
                iteration_update_pred = 1
                for it in range(iteration_update_pred):
                    for individual, individual_optimizer in zip(predictor_models, predictor_optmizers):
                        losses_individual = []
                        lossed_indiv_float = []
                        individual.zero_grad()
                        y_logits, y_pred = individual(x_batch)

                        loss_indiv = loss_BCE_pred(y_pred, y_batch)
                        # avg_loss_train += loss_indiv.item() / len(train_loader)
                        grads_pred, norm_value = get_gen_grads(individual, loss_indiv, retain_graph=True)
                        grads_list = []
                        losses_list = []
                        grads_list.append(grads_pred.cpu().detach().numpy())
                        losses_list.append(loss_indiv)
                        norm_values = []
                        norm_values.append(norm_value.item())
                        for advserary, advserary_optimizer in zip(adv_models, adversary_optimizers):
                            advserary.zero_grad()
                            pred_z_logit, pred_z_prob = advserary(y_logits, y_batch)
                            loss_adv = loss_BCE_adv(pred_z_prob, s_batch.reshape(-1, 1))
                            losses_list.append(loss_adv)
                            # noted there is a negative  "-" for grads of advserary
                            grad, norm_value = get_gen_grads(individual, loss_adv, retain_graph=True)
                            norm_values.append(norm_value.item())
                            grads_list.append(-grad.cpu().detach().numpy())

                        grads_list = np.asarray(grads_list).T
                        # grads_list: contains all the grads with normalization

                        result = minimize(steep_direct_cost, proba_adv, args=grads_list, jac=steep_direc_cost_deriv,
                                          bounds=bounds, constraints=constraints, method='SLSQP',
                                          options={'ftol': 1e-9, 'disp': False})
                        proba_adv = result.x
                        # print(proba_adv)

                        for p in range(proba_adv.shape[0]):
                            if p > 0:
                                proba_adv[p] *= -1
                        individual_optimizer.zero_grad()
                        loss_P_all = .0
                        proba_advs = np.array([1, 0, -1])
                        # proba_advs = proba_adv
                        # print(proba_advs)
                        for loss_weight in zip(losses_list, proba_advs):
                            loss_P_all += loss_weight[0] * float(loss_weight[1])
                        loss_P_all.backward()
                        individual_optimizer.step()

            avg_loss_test = .0

            for idx, individual in enumerate(predictor_models):
                if idx > 0:
                    continue
                with torch.no_grad():
                    individual.eval()
                    for i, (x_batch, y_batch, s_batch) in enumerate(test_loader):
                        if self.use_gpu:
                            x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
                        _, y_pred = individual(x_batch)
                        loss_test = loss_BCE_pred(y_pred, y_batch)
                        avg_loss_test += loss_test.item() / len(test_loader)
                    test_indi[idx, epoch] = avg_loss_test
                    # loss_indi[idx, epoch] = avg_loss_train
                    if np.mod(epoch, 50000) == 0:
                        is_mutation = 1
                        if epoch > 0:
                            if is_mutation == 1:
                                individual = mutator.do(individual)

                        # for name, param in individual.named_parameters():
                        #     weighs = np.array(param.detach())
                        #     # print(name)
                        #     # print('  before: ', weighs)
                        #     weighs += np.random.normal(loc=0.0, scale=0.01, size=param.shape)
                        #     individual.state_dict()[name].data.copy_(torch.Tensor(weighs))

                        # if np.mod(gen, 20) == 0:
                        #     plot_decision_boundary(individual, x_train, y_train, 1-S_train[0], np.array([gen, idx]), dirName=dirName)

                    l2_regularization = 0.0
                    for param in individual.parameters():
                        l2_regularization += torch.norm(param, 2)  # L2 正则化

                    logit_temp, pred_sigmoid_temp = individual(x_test)
                    logits_test = np.array(pred_sigmoid_temp.detach())
                    accuracy_test, accuracy_loss_test, individual_fairness_test, group_fairness_test, Groups_test, more_vals_test = ea.Cal_objectives(
                        self.test_data,
                        self.test_data_norm,
                        logits_test, y_test,
                        self.cal_sens_name,
                        2, self.Groups_info['sens_idxs_test'],
                        plan=self.cal_obj_plan,
                        dis=logit_temp)

                    logit_temp, pred_sigmoid_temp = individual(x_train)
                    logits_train = np.array(pred_sigmoid_temp.detach())
                    accuracy_train, accuracy_loss_train, individual_fairness_train, group_fairness_train, Groups_train, more_vals_train = ea.Cal_objectives(
                        self.train_data,
                        self.train_data_norm,
                        logits_train, y_train,
                        self.cal_sens_name,
                        2, self.Groups_info['sens_idxs_train'],
                        plan=self.cal_obj_plan,
                        dis=logit_temp)

                    logit_temp, pred_sigmoid_temp = individual(x_valid)
                    logits_valid = np.array(pred_sigmoid_temp.detach())
                    accuracy_valid, accuracy_loss_valid, individual_fairness_valid, group_fairness_valid, Groups_valid, more_vals_valid = ea.Cal_objectives(
                        self.valid_data,
                        self.valid_data_norm,
                        logits_valid, y_valid,
                        self.cal_sens_name,
                        2, self.Groups_info['sens_idxs_valid'],
                        plan=self.cal_obj_plan,
                        dis=logit_temp)
                    pop_pop.ObjV_train = AllObj_train
                    pop_pop.ObjV_valid = AllObj_valid
                    pop_pop.ObjV_test = AllObj_test
                    if is_recordmore == 0:
                        AllObj_train[idx][:] = np.array(
                            [accuracy_train, accuracy_loss_train, l2_regularization, individual_fairness_train,
                             group_fairness_train])
                        AllObj_test[idx][:] = np.array(
                            [accuracy_test, accuracy_loss_test, l2_regularization, individual_fairness_test,
                             group_fairness_test])
                        AllObj_valid[idx][:] = np.array(
                            [accuracy_valid, accuracy_loss_valid, l2_regularization, individual_fairness_valid,
                             group_fairness_valid])
                    else:
                        temp = np.zeros([1, 5 + self.more_info_num])
                        temp[0][0:5] = [accuracy_train, accuracy_loss_train, l2_regularization,
                                        individual_fairness_train, group_fairness_train]
                        temp[0][5:] = more_vals_train
                        AllObj_train[idx][:] = temp

                        temp = np.zeros([1, 5 + self.more_info_num])
                        temp[0][0:5] = [accuracy_test, accuracy_loss_test, l2_regularization, individual_fairness_test,
                                        group_fairness_test]
                        temp[0][5:] = more_vals_test
                        AllObj_test[idx][:] = temp

                        temp = np.zeros([1, 5 + self.more_info_num])
                        temp[0][0:5] = [accuracy_valid, accuracy_loss_valid, l2_regularization,
                                        individual_fairness_valid, group_fairness_valid]
                        temp[0][5:] = more_vals_valid
                        AllObj_valid[idx][:] = temp
                    pop_pop.ObjV = np.zeros([popsize, len(self.objectives_class)])
                    pop_pop.ObjV_train[idx][:] = AllObj_train[idx][:]
                    pop_pop.ObjV_valid[idx][:] = AllObj_valid[idx][:]
                    pop_pop.ObjV_test[idx][:] = AllObj_test[idx][:]
                    pop_pop.Chrom[idx] = individual
                    # plot_decision_boundary4(pop_pop, problem, dirName=dirName, epoch=epoch, idx=idx)
                    mse_indi[idx][epoch] = accuracy_loss_test
                    unfairness_indi[idx][epoch] = individual_fairness_test

        ###########################################################################################################################
        with torch.no_grad():
            # plt.plot(np.array(range(self.epoches)), np.mean(loss_indi, axis=0), label='train')
            # plt.plot(np.array(range(self.epoches)), np.mean(test_indi, axis=0), label='test')
            # plt.legend(loc='best', fontsize=15)
            np.savetxt(
                "geatpy/data/TwoGaussian_largerer0_alpha{}_2terms_mse".format(self.GAN_alpha).replace('.', '') + '.txt',
                mse_indi)
            np.savetxt("geatpy/data/TwoGaussian_largerer0_alpha{}_2terms_fairness".format(self.GAN_alpha).replace('.',
                                                                                                                  '') + '.txt',
                       unfairness_indi)

        pop.CV = np.zeros([popsize, 1])
        PopObj = AllObj_valid.copy()
        delete_list = []
        if 'accuracy' not in self.objectives_class:
            delete_list.append(0)
        if 'l2' not in self.objectives_class:
            delete_list.append(2)
        if 'Error' not in self.objectives_class:
            delete_list.append(1)
        if 'individual' not in self.objectives_class:
            delete_list.append(3)
        if 'group' not in self.objectives_class:
            delete_list.append(4)
        if is_recordmore == 1:
            for i in range(5, 5 + self.more_info_num):
                delete_list.append(i)
        pop.ObjV = np.delete(PopObj, delete_list, 1)  # 把求得的目标函数值赋值给种群pop的ObjV.
        if 'accuracy' in self.objectives_class:
            pop.ObjV[:, 0] = 1 - pop.ObjV[:, 0]
        pop.ObjV_train = AllObj_train
        pop.ObjV_valid = AllObj_valid
        pop.ObjV_test = AllObj_test
        pop.pred_label_train = pred_label_train
        pop.pred_label_valid = pred_label_valid
        pop.pred_label_test = pred_label_test
        pop.pred_logits_train = pred_logits_train
        pop.pred_logits_valid = pred_logits_valid
        pop.pred_logits_test = pred_logits_test

        endtime = time.time()
        # if self.use_gpu:
        #     print('use gpu')
        # else:
        #     print('not use gpu')
        # print('calculate objectives run time:', endtime - start_time)

        return AllObj_train, AllObj_valid, AllObj_test

    def model_test3(self, pop, kfold=0, gen=0, dirName=None, use_GAN=False, adv_models=None, problem=None,
                    ndSort=None):  # 目标函数
        # 针对多个adverasrys
        # 特别的，针对两个种群拆开
        # kfold = 0 : 全部的train训练model
        # kfold！= 0 : 将train进行kfold并 k 为输入的数值
        start_time = time.time()
        if self.ran_flag == 0:
            Groups_info = self.Groups_info
            sens_attr = self.cal_sens_name
            group_dicts = Groups_info['group_dict_train']
            s_labels = group_dicts[sens_attr[0]][0]

            sens_idxs_train = Groups_info['sens_idxs_train']
            s_labels_train = sens_idxs_train[s_labels][0]
            S_train = np.zeros([1, self.num_train])
            S_train[0, s_labels_train] = 1

            sens_idxs_valid = Groups_info['sens_idxs_valid']
            s_labels_valid = sens_idxs_valid[s_labels][0]
            S_valid = np.zeros([1, self.num_valid])
            S_valid[0, s_labels_valid] = 1

            sens_idxs_test = Groups_info['sens_idxs_test']
            s_labels_test = sens_idxs_test[s_labels][0]
            S_test = np.zeros([1, self.num_test])
            S_test[0, s_labels_test] = 1

            np.savetxt('Result/' + self.start_time + '/detect/Sens_train.txt', S_train)
            np.savetxt('Result/' + self.start_time + '/detect/Sens_valid.txt', S_valid)
            np.savetxt('Result/' + self.start_time + '/detect/Sens_test.txt', S_test)

        self.ran_flag = 1
        is_recordmore = 1
        popsize = len(pop)
        pred_label_train = np.zeros([popsize, self.num_train])
        pred_label_valid = np.zeros([popsize, self.num_valid])
        pred_label_test = np.zeros([popsize, self.num_test])

        pred_logits_train = np.zeros([popsize, self.num_train])
        pred_logits_valid = np.zeros([popsize, self.num_valid])
        pred_logits_test = np.zeros([popsize, self.num_test])

        # 这种情况下会修改pop中个体网络的权重值
        # -------- ZQQ - begin -----------

        self.use_gpu = torch.cuda.is_available()
        self.use_gpu = False
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _, _, _, _, _, more_vals_test = ea.Cal_objectives(self.train_data, self.train_data_norm,
                                                          np.array(self.train_y).reshape(1, -1),
                                                          np.array(self.train_y).reshape(1, -1), self.cal_sens_name,
                                                          2, self.Groups_info['sens_idxs_test'], plan=self.cal_obj_plan,
                                                          dis=np.array(self.train_y).reshape(1, -1))
        self.more_info_num = len(more_vals_test)

        # 依次，accuracy MSE L2 Individual Group
        if is_recordmore == 0:
            AllObj_valid = np.zeros([popsize, 5])
            AllObj_test = np.zeros([popsize, 5])
            AllObj_train = np.zeros([popsize, 5])
        else:
            AllObj_valid = np.zeros([popsize, 5 + self.more_info_num])
            AllObj_test = np.zeros([popsize, 5 + self.more_info_num])
            AllObj_train = np.zeros([popsize, 5 + self.more_info_num])

        pop_logits_test = np.zeros([popsize, self.test_data_norm.shape[0]])
        record_loss = np.zeros([popsize, self.epoches * 10])
        record_test = np.zeros([popsize, self.epoches * 10])
        # Groups_info = []

        # for ti in range(10):
        #     print(ti)
        loss_indi = np.zeros([popsize, self.epoches])
        test_indi = np.zeros([popsize, self.epoches])
        mse_indi = np.zeros([popsize, self.epoches])
        unfairness_indi = np.zeros([popsize, self.epoches])
        mutator = ea.Mutation_NN2(mu=0., var=0.01)

        x_train = torch.Tensor(self.train_data_norm)
        y_train = torch.Tensor(np.array(self.train_y))
        y_train = y_train.view(y_train.shape[0], 1)

        x_test = torch.Tensor(self.test_data_norm)
        y_test = torch.Tensor(np.array(self.test_y))
        y_test = y_test.view(y_test.shape[0], 1)

        x_valid = torch.Tensor(self.valid_data_norm)
        y_valid = torch.Tensor(self.valid_y)
        y_valid = y_valid.view(y_valid.shape[0], 1)

        Groups_info = self.Groups_info

        sens_attr = self.cal_sens_name
        group_dicts = Groups_info['group_dict_train']
        s_labels = group_dicts[sens_attr[0]][0]

        sens_idxs_train = Groups_info['sens_idxs_train']
        s_labels_train = sens_idxs_train[s_labels][0]
        S_train = np.zeros([1, self.num_train])
        S_train[0, s_labels_train] = 1
        s_train = torch.Tensor(S_train)

        sens_idxs_valid = Groups_info['sens_idxs_valid']
        s_labels_valid = sens_idxs_valid[s_labels][0]
        S_valid = np.zeros([1, self.num_valid])
        S_valid[0, s_labels_valid] = 1

        sens_idxs_test = Groups_info['sens_idxs_test']
        sens_idxs_name_test = Groups_info['sens_idxs_name_test']
        s_labels_test = sens_idxs_test[s_labels][0]
        S_test = np.zeros([1, self.num_test])
        S_test[0, s_labels_test] = 1
        s_test = torch.Tensor(S_test)

        ###########################################################################################################################
        # update adversarial models
        for idx in range(popsize):

            individual = copy.deepcopy(pop.Chrom[idx])  # 不是引用，是复制，不会修改pop.Chrom 网络的值
            adversary_models = adv_models

            loss_fn = torch.nn.BCELoss()
            if self.use_gpu:
                loss_fn.cuda()

            if self.use_gpu:
                individual.cuda()
                if use_GAN:
                    for adversary_model in adversary_models:
                        adversary_model.cuda()

            pop_optimizer = torch.optim.Adam(individual.parameters(), lr=self.learning_rate,
                                             weight_decay=self.weight_decay)

            if use_GAN:
                adv_optimizers = []
                for adversary_model in adversary_models:
                    if adversary_model.no_targets == 1:
                        adv_optimizer = torch.optim.Adam(adversary_model.parameters(), lr=self.learning_rate)
                    else:
                        adv_optimizer = torch.optim.Adam(adversary_model.parameters(), lr=self.learning_rate)
                    adv_optimizers.append(adv_optimizer)

            train = TensorDataset(x_train, y_train, s_train[0])
            train_loader = DataLoader(train, batch_size=self.batch_size, shuffle=True)
            test = TensorDataset(x_test, y_test, s_test[0])
            test_loader = DataLoader(test, batch_size=self.batch_size, shuffle=True)
            use_GAN = False
            self.GAN_alpha = 0.3
            proba = np.random.rand(len(adversary_models) + 1)
            constraints, bounds = make_constraints(len(adversary_models) + 1)
        ###########################################################################################################################

        # update predictor models
        for idx in range(popsize):
            use_GAN = True
            # individual = pop.Chrom[idx]  # 只是引用，不是复制，还会修改pop.Chrom 网络的值
            pop_pop = copy.deepcopy(pop)
            individual = copy.deepcopy(pop.Chrom[idx])  # 不是引用，是复制，不会修改pop.Chrom 网络的值
            adversary_models = copy.deepcopy(adv_models)

            loss_fn = torch.nn.BCELoss()
            if self.use_gpu:
                loss_fn.cuda()

            if self.use_gpu:
                individual.cuda()
                if use_GAN:
                    for adversary_model in adversary_models:
                        adversary_model.cuda()

            pop_optimizer = torch.optim.Adam(individual.parameters(), lr=self.learning_rate,
                                             weight_decay=self.weight_decay)

            if use_GAN:
                adv_optimizers = []
                for adversary_model in adversary_models:
                    if adversary_model.no_targets == 1:
                        adv_optimizer = torch.optim.Adam(adversary_model.parameters(), lr=self.learning_rate)
                    else:
                        adv_optimizer = torch.optim.Adam(adversary_model.parameters(), lr=self.learning_rate)
                    adv_optimizers.append(adv_optimizer)

            train = TensorDataset(x_train, y_train, s_train[0])
            train_loader = DataLoader(train, batch_size=self.batch_size, shuffle=True)
            test = TensorDataset(x_test, y_test, s_test[0])
            test_loader = DataLoader(test, batch_size=self.batch_size, shuffle=True)
            use_GAN = False
            self.GAN_alpha = 0.3
            proba = np.random.rand(len(adversary_models) + 1)
            constraints, bounds = make_constraints(len(adversary_models) + 1)
            for epoch in range(self.epoches):
                individual.train()
                avg_loss_test = 0.0
                avg_loss_train = 0.0

                if epoch > -1:
                    use_GAN = True
                    self.GAN_alpha = 20

                # updata individul models
                for i, (x_batch, y_batch, s_batch) in enumerate(train_loader):
                    if self.use_gpu:
                        x_batch, y_batch, s_batch = x_batch.cuda(), y_batch.cuda(), s_batch.cuda()
                    individual.zero_grad()
                    y_logits, y_pred = individual(x_batch)

                    loss_P = loss_fn(y_pred, y_batch)
                    avg_loss_train += loss_P.item() / len(train_loader)
                    grads_pred = get_gen_grads(individual, loss_P, retain_graph=True).cpu().detach().numpy()
                    # calculate L2 regular
                    # reg_loss = None
                    # for param in individual.parameters():
                    #     if reg_loss is None:
                    #         reg_loss = 0.5 * torch.sum(param ** 2)
                    #     else:
                    #         reg_loss = reg_loss + 0.5 * param.norm(2) ** 2
                    # loss_P += self.L_regul * reg_loss
                    grads_list = []
                    losses_list = []
                    grads_list.append(grads_pred)
                    if use_GAN:
                        for adv in adv_models:
                            adv.zero_grad()
                            pred_z_logit, pred_z_prob = adv(y_logits, y_batch)
                            loss = loss_fn(pred_z_prob, s_batch.reshape(-1, 1))
                            losses_list.append(loss)
                            grads_list.append(
                                -get_gen_grads(individual, loss, retain_graph=True).cpu().detach().numpy())

                        grads_list = np.asarray(grads_list).T
                        proba /= np.sum(proba)
                        result = minimize(steep_direct_cost, proba, args=grads_list, jac=steep_direc_cost_deriv,
                                          bounds=bounds, constraints=constraints, method='SLSQP',
                                          options={'ftol': 1e-9, 'disp': False})
                        proba = result.x
                        print(proba)
                        for p in range(proba.shape[0]):
                            if p > 0:
                                proba[p] *= -1
                    pop_optimizer.zero_grad()
                    loss_P_all = .0
                    for loss_weight in zip(losses_list, proba):
                        loss_P_all += loss_weight[0] * float(loss_weight[1])
                    loss_P_all.backward()
                    pop_optimizer.step()

                    # if use_GAN:
                    #     pred_z_logit_EO, pred_z_prob_EO = adversary_EO(y_logits, y_batch)
                    #     loss_A_EO = loss_fn(pred_z_prob_EO, s_batch.reshape(-1, 1))
                    #     adv_optimizer_EO.zero_grad()
                    #     pop_optimizer.zero_grad()
                    #     loss_A_EO.backward(retain_graph=True)
                    #
                    #     grads_w_La = get_gen_grads(individual, loss_A_EO, True)
                    #     grads_w_Lp = get_gen_grads(individual, loss_P, True)
                    #
                    #     grad_w_La = concat_grad(individual)
                    #     proba = np.random.rand(2)
                    #     proba /= np.sum(proba)
                    #     grads_list = []
                    #     grads_list.append(grads_w_La.cpu().detach().numpy())
                    #     grads_list.append(grads_w_Lp.cpu().detach().numpy())
                    #     constraints, bounds = make_constraints(2)
                    #     grads_list = np.asarray(grads_list).T
                    #     result = minimize(steep_direct_cost, proba, args=grads_list, jac=steep_direc_cost_deriv,
                    #                       bounds=bounds, constraints=constraints, method='SLSQP',
                    #                       options={'ftol': 1e-9, 'disp': False})
                    #     proba = result.x
                    #     print(proba)
                    # pop_optimizer.zero_grad()
                    # loss_P.backward()
                    # if use_GAN:
                    #     grad_w_Lp = concat_grad(individual)
                    #     proj_grad = project_grad(grad_w_Lp, grad_w_La)
                    #     alph1 = 0
                    #     grad_w_Lp = grad_w_Lp - alph1 * proj_grad - self.GAN_alpha * grad_w_La
                    #
                    #     replace_grad(individual, grad_w_Lp)
                    # pop_optimizer.step()
                    # if use_GAN:
                    #     adv_optimizer_EO.step()

                with torch.no_grad():
                    individual.eval()
                    for i, (x_batch, y_batch, s_batch) in enumerate(test_loader):
                        if self.use_gpu:
                            x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
                        _, y_pred = individual(x_batch)
                        loss_test = loss_fn(y_pred, y_batch)
                        avg_loss_test += loss_test.item() / len(test_loader)
                    test_indi[idx, epoch] = avg_loss_test
                    loss_indi[idx, epoch] = avg_loss_train
                    if np.mod(epoch, 50000) == 0:
                        if epoch == 0:
                            continue
                        individual = mutator.do(individual)
                        # for name, param in individual.named_parameters():
                        #     weighs = np.array(param.detach())
                        #     # print(name)
                        #     # print('  before: ', weighs)
                        #     weighs += np.random.normal(loc=0.0, scale=0.01, size=param.shape)
                        #     individual.state_dict()[name].data.copy_(torch.Tensor(weighs))
                    individual.eval()
                    if self.use_gpu:
                        individual.cpu()
                        # if np.mod(gen, 20) == 0:
                        #     plot_decision_boundary(individual, x_train, y_train, 1-S_train[0], np.array([gen, idx]), dirName=dirName)

                    l2_regularization = 0.0
                    for param in individual.parameters():
                        l2_regularization += torch.norm(param, 2)  # L2 正则化

                    logit_temp, pred_sigmoid_temp = individual(x_test)
                    logits_test = np.array(pred_sigmoid_temp.detach())
                    accuracy_test, accuracy_loss_test, individual_fairness_test, group_fairness_test, Groups_test, more_vals_test = ea.Cal_objectives(
                        self.test_data,
                        self.test_data_norm,
                        logits_test, y_test,
                        self.cal_sens_name,
                        2, self.Groups_info['sens_idxs_test'],
                        plan=self.cal_obj_plan,
                        dis=logit_temp)

                    logit_temp, pred_sigmoid_temp = individual(x_train)
                    logits_train = np.array(pred_sigmoid_temp.detach())
                    accuracy_train, accuracy_loss_train, individual_fairness_train, group_fairness_train, Groups_train, more_vals_train = ea.Cal_objectives(
                        self.train_data,
                        self.train_data_norm,
                        logits_train, y_train,
                        self.cal_sens_name,
                        2, self.Groups_info['sens_idxs_train'],
                        plan=self.cal_obj_plan,
                        dis=logit_temp)

                    logit_temp, pred_sigmoid_temp = individual(x_valid)
                    logits_valid = np.array(pred_sigmoid_temp.detach())
                    accuracy_valid, accuracy_loss_valid, individual_fairness_valid, group_fairness_valid, Groups_valid, more_vals_valid = ea.Cal_objectives(
                        self.valid_data,
                        self.valid_data_norm,
                        logits_valid, y_valid,
                        self.cal_sens_name,
                        2, self.Groups_info['sens_idxs_valid'],
                        plan=self.cal_obj_plan,
                        dis=logit_temp)
                    pop_pop.ObjV_train = AllObj_train
                    pop_pop.ObjV_valid = AllObj_valid
                    pop_pop.ObjV_test = AllObj_test
                    if is_recordmore == 0:
                        AllObj_train[idx][:] = np.array(
                            [accuracy_train, accuracy_loss_train, l2_regularization, individual_fairness_train,
                             group_fairness_train])
                        AllObj_test[idx][:] = np.array(
                            [accuracy_test, accuracy_loss_test, l2_regularization, individual_fairness_test,
                             group_fairness_test])
                        AllObj_valid[idx][:] = np.array(
                            [accuracy_valid, accuracy_loss_valid, l2_regularization, individual_fairness_valid,
                             group_fairness_valid])
                    else:
                        temp = np.zeros([1, 5 + self.more_info_num])
                        temp[0][0:5] = [accuracy_train, accuracy_loss_train, l2_regularization,
                                        individual_fairness_train, group_fairness_train]
                        temp[0][5:] = more_vals_train
                        AllObj_train[idx][:] = temp

                        temp = np.zeros([1, 5 + self.more_info_num])
                        temp[0][0:5] = [accuracy_test, accuracy_loss_test, l2_regularization, individual_fairness_test,
                                        group_fairness_test]
                        temp[0][5:] = more_vals_test
                        AllObj_test[idx][:] = temp

                        temp = np.zeros([1, 5 + self.more_info_num])
                        temp[0][0:5] = [accuracy_valid, accuracy_loss_valid, l2_regularization,
                                        individual_fairness_valid, group_fairness_valid]
                        temp[0][5:] = more_vals_valid
                        AllObj_valid[idx][:] = temp
                    pop_pop.ObjV = np.zeros([popsize, len(self.objectives_class)])
                    pop_pop.ObjV_train[idx][:] = AllObj_train[idx][:]
                    pop_pop.ObjV_valid[idx][:] = AllObj_valid[idx][:]
                    pop_pop.ObjV_test[idx][:] = AllObj_test[idx][:]
                    pop_pop.Chrom[idx] = individual
                    plot_decision_boundary4(pop_pop, problem, dirName=dirName, epoch=epoch, idx=idx)
                    mse_indi[idx][epoch] = accuracy_loss_test
                    unfairness_indi[idx][epoch] = individual_fairness_test

        with torch.no_grad():
            # plt.plot(np.array(range(self.epoches)), np.mean(loss_indi, axis=0), label='train')
            # plt.plot(np.array(range(self.epoches)), np.mean(test_indi, axis=0), label='test')
            # plt.legend(loc='best', fontsize=15)
            np.savetxt(
                "geatpy/data/TwoGaussian_largerer0_alpha{}_2terms_mse".format(self.GAN_alpha).replace('.', '') + '.txt',
                mse_indi)
            np.savetxt("geatpy/data/TwoGaussian_largerer0_alpha{}_2terms_fairness".format(self.GAN_alpha).replace('.',
                                                                                                                  '') + '.txt',
                       unfairness_indi)
            # plt.show()

            # with torch.no_grad():
            #
            #     # if self.use_gpu:
            #     #     individual.cuda()
            #     #     x_test, x_train, x_valid = x_test.cuda(), x_train.cuda(), x_valid.cuda()
            #     if self.use_gpu:
            #         individual.cpu()
            #     # if np.mod(gen, 50) == 0:
            #     #     plot_decision_boundary(individual, x_train, y_train, 1 - S_train[0], np.array([gen, idx]),
            #     #                            dirName=dirName)
            #     l2_regularization = 0.0
            #     for param in individual.parameters():
            #         # l1_regularization += torch.norm(param, 1)  # L1正则化
            #         # print(param)
            #         l2_regularization += torch.norm(param, 2)  # L2 正则化
            #     logit_temp, pred_sigmoid_temp = individual(x_test)
            #     logits_test = np.array(pred_sigmoid_temp.detach())
            #     pred_label_test[idx][:] = get_label(logits_test.reshape(1, -1).copy())
            #     pop_logits_test[idx][:] = logits_test.reshape(1, -1)
            #     pred_logits_test[idx][:] = logits_test.reshape(1, -1)
            #     accuracy_test, accuracy_loss_test, individual_fairness_test, group_fairness_test, Groups_test, more_vals_test = ea.Cal_objectives(
            #         self.test_data,
            #         self.test_data_norm,
            #         logits_test, y_test,
            #         self.cal_sens_name,
            #         2, self.Groups_info['sens_idxs_test'],
            #         plan=self.cal_obj_plan,
            #     dis=logit_temp)
            #
            #     logit_temp, pred_sigmoid_temp = individual(x_train)
            #     logits_train = np.array(pred_sigmoid_temp.detach())
            #     pred_label_train[idx][:] = get_label(logits_train.reshape(1, -1).copy())
            #     pred_logits_train[idx][:] = logits_train.reshape(1, -1)
            #     accuracy_train, accuracy_loss_train, individual_fairness_train, group_fairness_train, Groups_train, more_vals_train = ea.Cal_objectives(
            #         self.train_data,
            #         self.train_data_norm,
            #         logits_train, y_train,
            #         self.cal_sens_name,
            #         2, self.Groups_info['sens_idxs_train'],
            #     plan=self.cal_obj_plan,
            #     dis=logit_temp)
            #
            #     logit_temp, pred_sigmoid_temp = individual(x_valid)
            #     logits_valid = np.array(pred_sigmoid_temp.detach())
            #     pred_label_valid[idx][:] = get_label(logits_valid.reshape(1, -1).copy())
            #     pred_logits_valid[idx][:] = logits_valid.reshape(1, -1)
            #     accuracy_valid, accuracy_loss_valid, individual_fairness_valid, group_fairness_valid, Groups_valid, more_vals_valid = ea.Cal_objectives(
            #         self.valid_data,
            #         self.valid_data_norm,
            #         logits_valid, y_valid,
            #         self.cal_sens_name,
            #         2, self.Groups_info['sens_idxs_valid'],
            #     plan=self.cal_obj_plan,
            #     dis=logit_temp)
            #
            #     # print('all is ok')
            #     ####################################################################################################
            #     # # in test data
            #     # print('The information in test data: ')
            #     # print('  accuracy: %.4f, MSE: %.4f individual fairness: %.5f, group fairness: %.5f\n'
            #     #       % (accuracy_test, accuracy_loss_test, individual_fairness_test, group_fairness_test))
            #     #
            #     # ####################################################################################################
            #     # # in train data
            #     # print('The formation in train data: ')
            #     # print('  accuracy: %.4f, MSE: %.4f, individual fairness: %.5f, group fairness: %.5f\n'
            #     #       % (accuracy_train, accuracy_loss_train, individual_fairness_train, group_fairness_train))
            #     ####################################################################################################
            #
            #     # Groups_info.append(Groups_train)
            #     #
            #     if is_recordmore == 0:
            #         AllObj_train[idx][:] = np.array(
            #             [accuracy_train, accuracy_loss_train, l2_regularization, individual_fairness_train,
            #              group_fairness_train])
            #         AllObj_test[idx][:] = np.array(
            #             [accuracy_test, accuracy_loss_test, l2_regularization, individual_fairness_test,
            #              group_fairness_test])
            #         AllObj_valid[idx][:] = np.array(
            #             [accuracy_valid, accuracy_loss_valid, l2_regularization, individual_fairness_valid,
            #              group_fairness_valid])
            #     else:
            #         temp = np.zeros([1, 5 + self.more_info_num])
            #         temp[0][0:5] = [accuracy_train, accuracy_loss_train, l2_regularization,
            #                         individual_fairness_train, group_fairness_train]
            #         temp[0][5:] = more_vals_train
            #         AllObj_train[idx][:] = temp
            #
            #         temp = np.zeros([1, 5 + self.more_info_num])
            #         temp[0][0:5] = [accuracy_test, accuracy_loss_test, l2_regularization, individual_fairness_test,
            #                         group_fairness_test]
            #         temp[0][5:] = more_vals_test
            #         AllObj_test[idx][:] = temp
            #
            #         temp = np.zeros([1, 5 + self.more_info_num])
            #         temp[0][0:5] = [accuracy_valid, accuracy_loss_valid, l2_regularization,
            #                         individual_fairness_valid, group_fairness_valid]
            #         temp[0][5:] = more_vals_valid
            #         AllObj_valid[idx][:] = temp
            #
            #     # AllObj_test[idx][:] = np.array([accuracy_loss_test, individual_fairness_train, group_fairness_train])
            #     # PopObj[idx][:] = np.array([individual_fairness, group_fairness])

        # mean_loss = np.mean(record_loss, axis=0)
        # mean_acc = np.mean(record_test, axis=0)
        # np.savetxt("compas_wd2_loss_lr0001.txt", record_loss)
        # np.savetxt("compas_wd2_acc_lr0001.txt", record_test)
        # plt.plot(np.array(range(self.epoches*10)), mean_loss)
        # plt.plot(np.array(range(self.epoches*10)), mean_acc)
        # end_time = time.time()
        # print('cost: ', end_time-start_time)
        # plt.show()
        # print('draw')

        pop.CV = np.zeros([popsize, 1])
        PopObj = AllObj_valid.copy()
        delete_list = []
        if 'accuracy' not in self.objectives_class:
            delete_list.append(0)
        if 'l2' not in self.objectives_class:
            delete_list.append(2)
        if 'Error' not in self.objectives_class:
            delete_list.append(1)
        if 'individual' not in self.objectives_class:
            delete_list.append(3)
        if 'group' not in self.objectives_class:
            delete_list.append(4)
        if is_recordmore == 1:
            for i in range(5, 5 + self.more_info_num):
                delete_list.append(i)
        pop.ObjV = np.delete(PopObj, delete_list, 1)  # 把求得的目标函数值赋值给种群pop的ObjV.
        if 'accuracy' in self.objectives_class:
            pop.ObjV[:, 0] = 1 - pop.ObjV[:, 0]
        pop.ObjV_train = AllObj_train
        pop.ObjV_valid = AllObj_valid
        pop.ObjV_test = AllObj_test
        pop.pred_label_train = pred_label_train
        pop.pred_label_valid = pred_label_valid
        pop.pred_label_test = pred_label_test
        pop.pred_logits_train = pred_logits_train
        pop.pred_logits_valid = pred_logits_valid
        pop.pred_logits_test = pred_logits_test

        endtime = time.time()
        # if self.use_gpu:
        #     print('use gpu')
        # else:
        #     print('not use gpu')
        # print('calculate objectives run time:', endtime - start_time)

        return AllObj_train, AllObj_valid, AllObj_test

    def model_test4(self, pop, dirName=None, problem=None, ndSort=None):  # 目标函数
        # method of paper "The Fairness-Accuracy Pareto Front"
        # kfold = 0 : 全部的train训练model
        # kfold！= 0 : 将train进行kfold并 k 为输入的数值
        start_time = time.time()
        if self.ran_flag == 0:
            Groups_info = self.Groups_info
            sens_attr = self.cal_sens_name
            group_dicts = Groups_info['group_dict_train']
            s_labels = group_dicts[sens_attr[0]][0]

            sens_idxs_train = Groups_info['sens_idxs_train']
            s_labels_train = sens_idxs_train[s_labels][0]
            S_train = np.zeros([1, self.num_train])
            S_train[0, s_labels_train] = 1

            sens_idxs_valid = Groups_info['sens_idxs_valid']
            s_labels_valid = sens_idxs_valid[s_labels][0]
            S_valid = np.zeros([1, self.num_valid])
            S_valid[0, s_labels_valid] = 1

            sens_idxs_test = Groups_info['sens_idxs_test']
            s_labels_test = sens_idxs_test[s_labels][0]
            S_test = np.zeros([1, self.num_test])
            S_test[0, s_labels_test] = 1

            np.savetxt('Result/' + self.start_time + '/detect/Sens_train.txt', S_train)
            np.savetxt('Result/' + self.start_time + '/detect/Sens_valid.txt', S_valid)
            np.savetxt('Result/' + self.start_time + '/detect/Sens_test.txt', S_test)

        self.ran_flag = 1
        is_recordmore = 1
        popsize = len(pop)
        pred_label_train = np.zeros([popsize, self.num_train])
        pred_label_valid = np.zeros([popsize, self.num_valid])
        pred_label_test = np.zeros([popsize, self.num_test])

        pred_logits_train = np.zeros([popsize, self.num_train])
        pred_logits_valid = np.zeros([popsize, self.num_valid])
        pred_logits_test = np.zeros([popsize, self.num_test])

        # 这种情况下会修改pop中个体网络的权重值
        # -------- ZQQ - begin -----------

        self.use_gpu = torch.cuda.is_available()
        self.use_gpu = False
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _, _, _, _, _, more_vals_test = ea.Cal_objectives(self.train_data, self.train_data_norm,
                                                          np.array(self.train_y).reshape(1, -1),
                                                          np.array(self.train_y).reshape(1, -1), self.cal_sens_name,
                                                          2, self.Groups_info['sens_idxs_test'], plan=self.cal_obj_plan,
                                                          dis=np.array(self.train_y).reshape(1, -1))
        self.more_info_num = len(more_vals_test)

        # 依次，accuracy MSE L2 Individual Group
        if is_recordmore == 0:
            AllObj_valid = np.zeros([popsize, 5])
            AllObj_test = np.zeros([popsize, 5])
            AllObj_train = np.zeros([popsize, 5])
        else:
            AllObj_valid = np.zeros([popsize, 5 + self.more_info_num])
            AllObj_test = np.zeros([popsize, 5 + self.more_info_num])
            AllObj_train = np.zeros([popsize, 5 + self.more_info_num])

        pop_logits_test = np.zeros([popsize, self.test_data_norm.shape[0]])
        record_loss = np.zeros([popsize, self.epoches * 10])
        record_test = np.zeros([popsize, self.epoches * 10])

        if self.dataname == 'adult':
            learning_rate, weight_decay, batch_size, epoches, hidden_nodes = 0.001, 0, 1000, 200, 256
            acc_upper, acc_lower = 0.65156, 0.2234
            indiv_upper, indiv_lower = 0.056, 0.031
            grou_upper, grou_lower = 0.0036, 0.0022
            # lamda = lamda * 0.5

        elif self.dataname == 'german':
            learning_rate, weight_decay, batch_size, epoches, hidden_nodes = 0.001, 0, 1000, 200, 128
            acc_upper, acc_lower = 0.72156, 0.6197
            indiv_upper, indiv_lower = 0.176, 0.06
            grou_upper, grou_lower = 0.0058, 0.0014

        elif self.dataname == 'propublica-recidivism':
            learning_rate, weight_decay, batch_size, epoches, hidden_nodes = 0.001, 0, 1000, 200, 128
            acc_upper, acc_lower = 0.65156, 0.2097
            indiv_upper, indiv_lower = 0.056, 0.026
            grou_upper, grou_lower = 0.004, 0.0028
            # lamda = lamda * 0.5

        elif 'two-gaussiansnew' in self.dataname:
            learning_rate, weight_decay, batch_size, epoches, hidden_nodes = 0.01, 1e-1, 512, 100, 256
            acc_upper, acc_lower = 0.69426, 0.5054
            indiv_upper, indiv_lower = 0.1118, 0.07
            grou_upper, grou_lower = 0.0029, 0.00023

        else:
            learning_rate, weight_decay, batch_size, epoches, hidden_nodes = 0.01, 1e-1, 512, 100, 256
            acc_upper, acc_lower = 0.69426, 0.5054
            indiv_upper, indiv_lower = 0.1118, 0.07
            grou_upper, grou_lower = 0.0029, 0.00023

        lamda = 0.5

        mse_train = np.zeros([popsize, self.epoches])
        mse_valid = np.zeros([popsize, self.epoches])
        mse_test = np.zeros([popsize, self.epoches])

        indiv_train = np.zeros([popsize, self.epoches])
        indiv_valid = np.zeros([popsize, self.epoches])
        indiv_test = np.zeros([popsize, self.epoches])

        grp_train = np.zeros([popsize, self.epoches])
        grp_valid = np.zeros([popsize, self.epoches])
        grp_test = np.zeros([popsize, self.epoches])

        mutator = ea.Mutation_NN2(mu=0., var=0.01)

        pop.ObjV_train = AllObj_train
        pop.ObjV_valid = AllObj_valid
        pop.ObjV_test = AllObj_test

        plan = 4
        is_mutation = 0
        for idx in range(popsize):
            use_GAN = False

            individual = pop.Chrom[idx]  # 只是引用，不是复制，还会修改pop.Chrom 网络的值
            # pop_pop = copy.deepcopy(pop)

            x_train = torch.Tensor(self.train_data_norm)
            y_train = torch.Tensor(np.array(self.train_y))
            y_train = y_train.view(y_train.shape[0], 1)

            x_test = torch.Tensor(self.test_data_norm)
            y_test = torch.Tensor(np.array(self.test_y))
            y_test = y_test.view(y_test.shape[0], 1)

            x_valid = torch.Tensor(self.valid_data_norm)
            y_valid = torch.Tensor(self.valid_y)
            y_valid = y_valid.view(y_valid.shape[0], 1)

            Groups_info = self.Groups_info

            sens_attr = self.cal_sens_name
            group_dicts = Groups_info['group_dict_train']
            s_labels = group_dicts[sens_attr[0]][0]

            sens_idxs_train = Groups_info['sens_idxs_train']
            s_labels_train = sens_idxs_train[s_labels][0]
            S_train = np.zeros([1, self.num_train])
            S_train[0, s_labels_train] = 1
            s_train = torch.Tensor(S_train)

            sens_idxs_valid = Groups_info['sens_idxs_valid']
            s_labels_valid = sens_idxs_valid[s_labels][0]
            S_valid = np.zeros([1, self.num_valid])
            S_valid[0, s_labels_valid] = 1
            s_valid = torch.Tensor(S_valid)

            sens_idxs_test = Groups_info['sens_idxs_test']
            sens_idxs_name_test = Groups_info['sens_idxs_name_test']
            s_labels_test = sens_idxs_test[s_labels][0]
            S_test = np.zeros([1, self.num_test])
            S_test[0, s_labels_test] = 1
            s_test = torch.Tensor(S_test)
            s_test = torch.Tensor(s_test)

            loss_fn = torch.nn.BCELoss()
            # loss_fn = torch.nn.MSELoss()

            if self.use_gpu:
                loss_fn.cuda()

            # pop_optimizer = torch.optim.Adam(individual.parameters(), lr=self.learning_rate,
            #                                  weight_decay=self.weight_decay)
            pop_optimizer = torch.optim.SGD(individual.parameters(), lr=self.learning_rate,
                                            weight_decay=self.weight_decay, momentum=0.9)

            train_idx = torch.arange(start=0, end=self.train_data_norm.shape[0], step=1)
            valid_idx = torch.arange(start=0, end=self.valid_data_norm.shape[0], step=1)
            test_idx = torch.arange(start=0, end=self.test_data_norm.shape[0], step=1)

            train = TensorDataset(x_train, y_train, s_train[0], train_idx)
            test = TensorDataset(x_test, y_test, s_test[0], test_idx)
            valid = TensorDataset(x_valid, y_valid, s_valid[0], valid_idx)

            train_loader = DataLoader(train, batch_size=self.batch_size, shuffle=True)
            test_loader = DataLoader(test, batch_size=self.batch_size, shuffle=True)
            valid_loader = DataLoader(valid, batch_size=self.batch_size, shuffle=True)

            mse_loss_train = 0.0
            mse_loss_test = 0.0

            min_acc = 1.0
            min_indiv = 1.0
            min_gr = 1.0
            for epoch in range(self.epoches):
                individual.train()

                for i, (x_batch, y_batch, s_batch, data_idx) in enumerate(train_loader):
                    if self.use_gpu:
                        x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
                    _, y_pred = individual(x_batch)
                    loss_acc = loss_fn(y_pred, y_batch)

                    loss_indiv, loss_group = calcul_all_fairness(self.train_data.loc[data_idx.detach()], y_pred,
                                                                 y_batch, self.sensitive_attributions, 2)

                    min_acc = min(min_acc, loss_acc.detach().numpy())
                    min_indiv = min(min_indiv, loss_indiv.detach().numpy())
                    min_gr = min(min_gr, loss_group.detach().numpy())

                    if plan == 1:
                        # calculate individual unfairness
                        loss_acc = (loss_acc - acc_lower) / (acc_upper - acc_lower)
                        loss_indiv = (loss_indiv - indiv_lower) / (indiv_upper - indiv_lower)
                        loss = torch.max((1 - lamda) * loss_acc, lamda * loss_indiv)
                        # loss = torch.sum((1 - lamda) * loss_acc, lamda * loss_indiv)

                    elif plan == 2:
                        # calculate group unfairness
                        loss_acc = (loss_acc - acc_lower) / (acc_upper - acc_lower)
                        loss_group = (loss_group - grou_lower) / (grou_upper - grou_lower)
                        loss = torch.max((1 - lamda) * loss_acc, lamda * loss_group)
                        # loss = torch.sum((1 - lamda) * loss_acc, lamda * loss_group,1)

                    elif plan == 3:
                        # only mse
                        loss = loss_acc

                    elif plan == 4:
                        # only individual unfairness
                        loss = loss_indiv

                    else:
                        # only group unfairness
                        loss = loss_group

                    mse_loss_train += loss.item() / len(train_loader)
                    pop_optimizer.zero_grad()  # clear gradients for next train
                    loss.backward()  # -> accumulates the gradient (by addition) for each parameter
                    pop_optimizer.step()  # -> update weights and biases

                with torch.no_grad():
                    individual.eval()
                    for i, (x_batch, y_batch, s_batch, data_idx) in enumerate(test_loader):
                        if self.use_gpu:
                            x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
                        _, y_pred = individual(x_batch)
                        loss_test = loss_fn(y_pred, y_batch)
                        mse_loss_test += loss_test.item() / len(test_loader)
                    # if np.mod(epoch, 50000) == 0:
                    #     if epoch == 0:
                    #         continue
                    #     individual = mutator.do(individual)
                    # for name, param in individual.named_parameters():
                    #     weighs = np.array(param.detach())
                    #     # print(name)
                    #     # print('  before: ', weighs)
                    #     weighs += np.random.normal(loc=0.0, scale=0.01, size=param.shape)
                    #     individual.state_dict()[name].data.copy_(torch.Tensor(weighs))
                    if is_mutation == 1:
                        if epoch > 0:
                            individual = mutator.do(individual)

                    if self.use_gpu:
                        individual.cpu()
                        # if np.mod(gen, 20) == 0:
                        #     plot_decision_boundary(individual, x_train, y_train, 1-S_train[0], np.array([gen, idx]), dirName=dirName)

                    l2_regularization = 0.0
                    for param in individual.parameters():
                        l2_regularization += torch.norm(param, 2)  # L2 正则化

                    logit_temp, pred_sigmoid_temp = individual(x_test)
                    logits_test = np.array(pred_sigmoid_temp.detach())
                    accuracy_test, accuracy_loss_test, individual_fairness_test, group_fairness_test, Groups_test, more_vals_test = ea.Cal_objectives(
                        self.test_data,
                        self.test_data_norm,
                        logits_test, y_test,
                        self.sensitive_attributions,
                        2, self.Groups_info['sens_idxs_test'],
                        plan=self.cal_obj_plan,
                        dis=logit_temp)

                    logit_temp, pred_sigmoid_temp = individual(x_train)
                    logits_train = np.array(pred_sigmoid_temp.detach())
                    accuracy_train, accuracy_loss_train, individual_fairness_train, group_fairness_train, Groups_train, more_vals_train = ea.Cal_objectives(
                        self.train_data,
                        self.train_data_norm,
                        logits_train, y_train,
                        self.sensitive_attributions,
                        2, self.Groups_info['sens_idxs_train'],
                        plan=self.cal_obj_plan,
                        dis=logit_temp)

                    logit_temp, pred_sigmoid_temp = individual(x_valid)
                    logits_valid = np.array(pred_sigmoid_temp.detach())
                    accuracy_valid, accuracy_loss_valid, individual_fairness_valid, group_fairness_valid, Groups_valid, more_vals_valid = ea.Cal_objectives(
                        self.valid_data,
                        self.valid_data_norm,
                        logits_valid, y_valid,
                        self.sensitive_attributions,
                        2, self.Groups_info['sens_idxs_valid'],
                        plan=self.cal_obj_plan,
                        dis=logit_temp)
                    print(epoch, ': ', accuracy_train, accuracy_valid, accuracy_test)

                    mse_train[idx, epoch] = accuracy_loss_train
                    mse_valid[idx, epoch] = accuracy_loss_valid
                    mse_test[idx, epoch] = accuracy_loss_test

                    indiv_train[idx, epoch] = individual_fairness_train
                    indiv_valid[idx, epoch] = individual_fairness_valid
                    indiv_test[idx, epoch] = individual_fairness_test

                    grp_train[idx, epoch] = group_fairness_train
                    grp_valid[idx, epoch] = group_fairness_valid
                    grp_test[idx, epoch] = group_fairness_test

                    if is_recordmore == 0:
                        AllObj_train[idx][:] = np.array(
                            [accuracy_train, accuracy_loss_train, l2_regularization, individual_fairness_train,
                             group_fairness_train])
                        AllObj_test[idx][:] = np.array(
                            [accuracy_test, accuracy_loss_test, l2_regularization, individual_fairness_test,
                             group_fairness_test])
                        AllObj_valid[idx][:] = np.array(
                            [accuracy_valid, accuracy_loss_valid, l2_regularization, individual_fairness_valid,
                             group_fairness_valid])
                    else:
                        temp = np.zeros([1, 5 + self.more_info_num])
                        temp[0][0:5] = [accuracy_train, accuracy_loss_train, l2_regularization,
                                        individual_fairness_train, group_fairness_train]
                        temp[0][5:] = more_vals_train
                        AllObj_train[idx][:] = temp

                        temp = np.zeros([1, 5 + self.more_info_num])
                        temp[0][0:5] = [accuracy_test, accuracy_loss_test, l2_regularization, individual_fairness_test,
                                        group_fairness_test]
                        temp[0][5:] = more_vals_test
                        AllObj_test[idx][:] = temp

                        temp = np.zeros([1, 5 + self.more_info_num])
                        temp[0][0:5] = [accuracy_valid, accuracy_loss_valid, l2_regularization,
                                        individual_fairness_valid, group_fairness_valid]
                        temp[0][5:] = more_vals_valid
                        AllObj_valid[idx][:] = temp
                    pop.ObjV = np.zeros([popsize, len(self.objectives_class)])
                    pop.ObjV_train[idx][:] = AllObj_train[idx][:]
                    pop.ObjV_valid[idx][:] = AllObj_valid[idx][:]
                    pop.ObjV_test[idx][:] = AllObj_test[idx][:]
                    pop.Chrom[idx] = individual
                    # plot_decision_boundary4(pop_pop, problem, dirName=dirName, epoch=epoch, idx=idx)
        nodes_str = ''

        for k in individual.n_hidden:
            nodes_str = nodes_str + str(k)
        weight_decay_str = str(self.weight_decay)
        weight_decay_str = weight_decay_str.replace('.', '')
        dropout_str = str(individual.dropout_value)
        dropout_str = dropout_str.replace('.', '')
        lr = str(self.learning_rate)
        lr = lr.replace('.', '')


        with torch.no_grad():

            if plan == 3:
                plt.plot(np.array(range(self.epoches)), np.mean(mse_train, axis=0), label='train')
                plt.plot(np.array(range(self.epoches)), np.mean(mse_valid, axis=0), label='valid')
                plt.plot(np.array(range(self.epoches)), np.mean(mse_test, axis=0), label='test')
                plt.legend(loc='best', fontsize=15)
            elif plan == 4:
                plt.plot(np.array(range(self.epoches)), np.mean(indiv_train, axis=0), label='train')
                plt.plot(np.array(range(self.epoches)), np.mean(indiv_valid, axis=0), label='valid')
                plt.plot(np.array(range(self.epoches)), np.mean(indiv_test, axis=0), label='test')
                plt.legend(loc='best', fontsize=15)
            else:
                plt.plot(np.array(range(self.epoches)), np.mean(grp_train, axis=0), label='train')
                plt.plot(np.array(range(self.epoches)), np.mean(grp_valid, axis=0), label='valid')
                plt.plot(np.array(range(self.epoches)), np.mean(grp_test, axis=0), label='test')
                plt.legend(loc='best', fontsize=15)

            # only mse


            # np.savetxt("geatpy/data/{}_train_plan{}_lr{}_dr{}_n{}_Gaus{}_mse_".format(self.dataname, plan, lr,
            #                                                              dropout_str, nodes_str, is_mutation).replace('.',
            #                                                                                              '') + '.txt',
            #            mse_train)
            #
            # np.savetxt(
            #     "geatpy/data/{}_test_plan{}_lr{}_dr{}_n{}_Gaus{}_mse_".format(self.dataname, plan, lr, dropout_str,
            #                                                      nodes_str, is_mutation).replace('.', '') + '.txt',
            #     mse_valid)
            #
            # np.savetxt("geatpy/data/{}_valid_plan{}_lr{}_dr{}_n{}_Gaus{}_mse_".format(self.dataname, plan, lr,
            #                                                              dropout_str, nodes_str, is_mutation).replace('.',
            #                                                                                              '') + '.txt',
            #            mse_test)
            #
            #
            # # only individual unfairness
            #
            #
            # np.savetxt("geatpy/data/{}_train_plan{}_lr{}_dr{}_n{}_Gaus{}_inid_".format(self.dataname, plan, lr,
            #                                                              dropout_str, nodes_str, is_mutation).replace('.',
            #                                                                                              '') + '.txt',
            #            indiv_train)
            # np.savetxt(
            #     "geatpy/data/{}_test_plan{}_lr{}_dr{}_n{}_Gaus{}_inid_".format(self.dataname, plan, lr, dropout_str,
            #                                                      nodes_str, is_mutation).replace('.', '') + '.txt',
            #     indiv_test)
            # np.savetxt("geatpy/data/{}_valid_plan{}_lr{}_dr{}_n{}_Gaus{}_inid_".format(self.dataname, plan, lr,
            #                                                              dropout_str, nodes_str, is_mutation).replace('.',
            #                                                                                              '') + '.txt',
            #            indiv_valid)
            #
            #
            # # only group unfairness
            #
            #
            # np.savetxt(
            #     "geatpy/data/{}_train_plan{}_lr{}_dr{}_n{}_Gaus{}_gr_".format(self.dataname, plan, lr,
            #                                                             dropout_str, nodes_str, is_mutation).replace('.',
            #                                                                                             '') + '.txt',
            #     grp_train)
            # np.savetxt(
            #     "geatpy/data/{}_test_plan{}_lr{}_dr{}_n{}_Gaus{}_gr_".format(self.dataname, plan, lr,
            #                                                            dropout_str,
            #                                                            nodes_str, is_mutation).replace('.', '') + '.txt',
            #     grp_test)
            # np.savetxt(
            #     "geatpy/data/{}_valid_plan{}_lr{}_dr{}_n{}_Gaus{}_gr_".format(self.dataname, plan, lr,
            #                                                             dropout_str, nodes_str, is_mutation).replace('.',
            #                                                                                             '') + '.txt',
            #     grp_valid)

            plt.show()

            # with torch.no_grad():
            #
            #     # if self.use_gpu:
            #     #     individual.cuda()
            #     #     x_test, x_train, x_valid = x_test.cuda(), x_train.cuda(), x_valid.cuda()
            #     if self.use_gpu:
            #         individual.cpu()
            #     # if np.mod(gen, 50) == 0:
            #     #     plot_decision_boundary(individual, x_train, y_train, 1 - S_train[0], np.array([gen, idx]),
            #     #                            dirName=dirName)
            #     l2_regularization = 0.0
            #     for param in individual.parameters():
            #         # l1_regularization += torch.norm(param, 1)  # L1正则化
            #         # print(param)
            #         l2_regularization += torch.norm(param, 2)  # L2 正则化
            #     logit_temp, pred_sigmoid_temp = individual(x_test)
            #     logits_test = np.array(pred_sigmoid_temp.detach())
            #     pred_label_test[idx][:] = get_label(logits_test.reshape(1, -1).copy())
            #     pop_logits_test[idx][:] = logits_test.reshape(1, -1)
            #     pred_logits_test[idx][:] = logits_test.reshape(1, -1)
            #     accuracy_test, accuracy_loss_test, individual_fairness_test, group_fairness_test, Groups_test, more_vals_test = ea.Cal_objectives(
            #         self.test_data,
            #         self.test_data_norm,
            #         logits_test, y_test,
            #         self.cal_sens_name,
            #         2, self.Groups_info['sens_idxs_test'],
            #         plan=self.cal_obj_plan,
            #     dis=logit_temp)
            #
            #     logit_temp, pred_sigmoid_temp = individual(x_train)
            #     logits_train = np.array(pred_sigmoid_temp.detach())
            #     pred_label_train[idx][:] = get_label(logits_train.reshape(1, -1).copy())
            #     pred_logits_train[idx][:] = logits_train.reshape(1, -1)
            #     accuracy_train, accuracy_loss_train, individual_fairness_train, group_fairness_train, Groups_train, more_vals_train = ea.Cal_objectives(
            #         self.train_data,
            #         self.train_data_norm,
            #         logits_train, y_train,
            #         self.cal_sens_name,
            #         2, self.Groups_info['sens_idxs_train'],
            #     plan=self.cal_obj_plan,
            #     dis=logit_temp)
            #
            #     logit_temp, pred_sigmoid_temp = individual(x_valid)
            #     logits_valid = np.array(pred_sigmoid_temp.detach())
            #     pred_label_valid[idx][:] = get_label(logits_valid.reshape(1, -1).copy())
            #     pred_logits_valid[idx][:] = logits_valid.reshape(1, -1)
            #     accuracy_valid, accuracy_loss_valid, individual_fairness_valid, group_fairness_valid, Groups_valid, more_vals_valid = ea.Cal_objectives(
            #         self.valid_data,
            #         self.valid_data_norm,
            #         logits_valid, y_valid,
            #         self.cal_sens_name,
            #         2, self.Groups_info['sens_idxs_valid'],
            #     plan=self.cal_obj_plan,
            #     dis=logit_temp)
            #
            #     # print('all is ok')
            #     ####################################################################################################
            #     # # in test data
            #     # print('The information in test data: ')
            #     # print('  accuracy: %.4f, MSE: %.4f individual fairness: %.5f, group fairness: %.5f\n'
            #     #       % (accuracy_test, accuracy_loss_test, individual_fairness_test, group_fairness_test))
            #     #
            #     # ####################################################################################################
            #     # # in train data
            #     # print('The formation in train data: ')
            #     # print('  accuracy: %.4f, MSE: %.4f, individual fairness: %.5f, group fairness: %.5f\n'
            #     #       % (accuracy_train, accuracy_loss_train, individual_fairness_train, group_fairness_train))
            #     ####################################################################################################
            #
            #     # Groups_info.append(Groups_train)
            #     #
            #     if is_recordmore == 0:
            #         AllObj_train[idx][:] = np.array(
            #             [accuracy_train, accuracy_loss_train, l2_regularization, individual_fairness_train,
            #              group_fairness_train])
            #         AllObj_test[idx][:] = np.array(
            #             [accuracy_test, accuracy_loss_test, l2_regularization, individual_fairness_test,
            #              group_fairness_test])
            #         AllObj_valid[idx][:] = np.array(
            #             [accuracy_valid, accuracy_loss_valid, l2_regularization, individual_fairness_valid,
            #              group_fairness_valid])
            #     else:
            #         temp = np.zeros([1, 5 + self.more_info_num])
            #         temp[0][0:5] = [accuracy_train, accuracy_loss_train, l2_regularization,
            #                         individual_fairness_train, group_fairness_train]
            #         temp[0][5:] = more_vals_train
            #         AllObj_train[idx][:] = temp
            #
            #         temp = np.zeros([1, 5 + self.more_info_num])
            #         temp[0][0:5] = [accuracy_test, accuracy_loss_test, l2_regularization, individual_fairness_test,
            #                         group_fairness_test]
            #         temp[0][5:] = more_vals_test
            #         AllObj_test[idx][:] = temp
            #
            #         temp = np.zeros([1, 5 + self.more_info_num])
            #         temp[0][0:5] = [accuracy_valid, accuracy_loss_valid, l2_regularization,
            #                         individual_fairness_valid, group_fairness_valid]
            #         temp[0][5:] = more_vals_valid
            #         AllObj_valid[idx][:] = temp
            #
            #     # AllObj_test[idx][:] = np.array([accuracy_loss_test, individual_fairness_train, group_fairness_train])
            #     # PopObj[idx][:] = np.array([individual_fairness, group_fairness])

        # mean_loss = np.mean(record_loss, axis=0)
        # mean_acc = np.mean(record_test, axis=0)
        # np.savetxt("compas_wd2_loss_lr0001.txt", record_loss)
        # np.savetxt("compas_wd2_acc_lr0001.txt", record_test)
        # plt.plot(np.array(range(self.epoches*10)), mean_loss)
        # plt.plot(np.array(range(self.epoches*10)), mean_acc)
        # end_time = time.time()
        # print('cost: ', end_time-start_time)
        # plt.show()
        # print('draw')

        pop.CV = np.zeros([popsize, 1])
        PopObj = AllObj_valid.copy()
        delete_list = []
        if 'accuracy' not in self.objectives_class:
            delete_list.append(0)
        if 'l2' not in self.objectives_class:
            delete_list.append(2)
        if 'Error' not in self.objectives_class:
            delete_list.append(1)
        if 'individual' not in self.objectives_class:
            delete_list.append(3)
        if 'group' not in self.objectives_class:
            delete_list.append(4)
        if is_recordmore == 1:
            for i in range(5, 5 + self.more_info_num):
                delete_list.append(i)
        pop.ObjV = np.delete(PopObj, delete_list, 1)  # 把求得的目标函数值赋值给种群pop的ObjV.
        if 'accuracy' in self.objectives_class:
            pop.ObjV[:, 0] = 1 - pop.ObjV[:, 0]
        pop.ObjV_train = AllObj_train
        pop.ObjV_valid = AllObj_valid
        pop.ObjV_test = AllObj_test
        pop.pred_label_train = pred_label_train
        pop.pred_label_valid = pred_label_valid
        pop.pred_label_test = pred_label_test
        pop.pred_logits_train = pred_logits_train
        pop.pred_logits_valid = pred_logits_valid
        pop.pred_logits_test = pred_logits_test

        endtime = time.time()
        # if self.use_gpu:
        #     print('use gpu')
        # else:
        #     print('not use gpu')
        # print('calculate objectives run time:', endtime - start_time)

        return AllObj_train, AllObj_valid, AllObj_test

    def train_nets(self, pop, epoches):
        # 将pop中的所有网络在train data上训练epoch遍，
        # 1. 会修改网络权重
        # 2. 返回在test与train上的四个值
        begin_time = time.time()
        popsize = len(pop)
        self.use_gpu = torch.cuda.is_available()
        self.use_gpu = False
        AllObj_test = np.zeros([popsize, 4])
        AllObj_train = np.zeros([popsize, 4])
        pop_logits_test = np.zeros([popsize, self.test_data_norm.shape[0]])
        for idx in range(popsize):
            individual = pop.Chrom[idx]  # 只是引用，不是复制，还会修改pop.Chrom 网络的值
            # individual = copy.deepcopy(pop.Chrom[idx])  # 不是引用，是复制，不会修改pop.Chrom 网络的值
            x_train = torch.Tensor(self.train_data_norm)
            y_train = torch.Tensor(np.array(self.train_y)).view(self.train_data.shape[0], 1)
            if self.use_gpu:
                individual.cuda()
            x_test = torch.Tensor(self.test_data_norm)
            y_test = torch.Tensor(np.array(self.test_y)).view(self.test_data_norm.shape[0], 1)

            optimizer = torch.optim.Adam(individual.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            loss_fn = torch.nn.BCEWithLogitsLoss()  # Combined with the sigmoid
            if self.use_gpu:
                loss_fn.cuda()

            train = TensorDataset(x_train, y_train)
            train_loader = DataLoader(train, batch_size=self.batch_size, shuffle=True)
            for epoch in range(epoches):
                individual.train()
                avg_loss = 0.
                for i, (x_batch, y_batch) in enumerate(train_loader):
                    if self.use_gpu:
                        x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
                    y_pred = individual(x_batch)
                    loss = loss_fn(y_pred, y_batch)

                    optimizer.zero_grad()  # clear gradients for next train
                    loss.backward()  # -> accumulates the gradient (by addition) for each parameter
                    optimizer.step()  # -> update weights and biases
                    avg_loss += loss.item() / len(train_loader)

            with torch.no_grad():
                if self.use_gpu:
                    individual.cpu()
                pop_logits_test[idx][:] = ea.sigmoid(np.array(individual(x_test).detach())).reshape(1, -1)

                logits_test = ea.sigmoid(np.array(individual(x_test).detach()))
                accuracy_test, accuracy_loss_test, individual_fairness_test, group_fairness_test, Groups_test = ea.Cal_objectives(
                    self.test_data,
                    self.test_data_norm,
                    logits_test, y_test,
                    self.cal_sens_name,
                    2, plan=self.cal_obj_plan)

                logits_train = ea.sigmoid(np.array(individual(x_train).detach()))
                accuracy_train, accuracy_loss_train, individual_fairness_train, group_fairness_train, Groups_train = ea.Cal_objectives(
                    self.train_data,
                    self.train_data_norm,
                    logits_train, y_train,
                    self.cal_sens_name,
                    2, plan=self.cal_obj_plan)
                ####################################################################################################
                # # in test data
                # print('The information in test data: ')
                # print('  accuracy: %.4f, MSE: %.4f individual fairness: %.5f, group fairness: %.5f\n'
                #       % (accuracy_test, accuracy_loss_test, individual_fairness_test, group_fairness_test))
                #
                # ####################################################################################################
                # # in train data
                # print('The formation in train data: ')
                # print('  accuracy: %.4f, MSE: %.4f, individual fairness: %.5f, group fairness: %.5f\n'
                #       % (accuracy_train, accuracy_loss_train, individual_fairness_train, group_fairness_train))
                ####################################################################################################
                AllObj_train[idx][:] = np.array(
                    [accuracy_train, accuracy_loss_train, individual_fairness_train, group_fairness_train])
                AllObj_test[idx][:] = np.array(
                    [accuracy_test, accuracy_loss_test, individual_fairness_test, group_fairness_test])
        end_time = time.time()
        # print('full train time: ', end_time-begin_time)
        return AllObj_train, AllObj_test, pop_logits_test

    def aimFunc_GAN(self, pop, kfold, Adversary, gen, dirName=None, use_GAN=False):
        # kfold = 0 : 全部的train训练model
        # kfold！= 0 : 将train进行kfold并 k 为输入的数值
        start_time = time.time()
        if self.ran_flag == 0:
            Groups_info = self.Groups_info
            sens_attr = self.cal_sens_name
            group_dicts = Groups_info['group_dict_train']
            s_labels = group_dicts[sens_attr[0]][0]

            sens_idxs_train = Groups_info['sens_idxs_train']
            s_labels_train = sens_idxs_train[s_labels][0]
            S_train = np.zeros([1, self.num_train])
            S_train[0, s_labels_train] = 1

            sens_idxs_valid = Groups_info['sens_idxs_valid']
            s_labels_valid = sens_idxs_valid[s_labels][0]
            S_valid = np.zeros([1, self.num_valid])
            S_valid[0, s_labels_valid] = 1

            sens_idxs_test = Groups_info['sens_idxs_test']
            s_labels_test = sens_idxs_test[s_labels][0]
            S_test = np.zeros([1, self.num_test])
            S_test[0, s_labels_test] = 1

            np.savetxt('Result/' + self.start_time + '/detect/Sens_train.txt', S_train)
            np.savetxt('Result/' + self.start_time + '/detect/Sens_valid.txt', S_valid)
            np.savetxt('Result/' + self.start_time + '/detect/Sens_test.txt', S_test)

        self.ran_flag = 1
        is_recordmore = 1
        popsize = len(pop)
        pred_label_train = np.zeros([popsize, self.num_train])
        pred_label_valid = np.zeros([popsize, self.num_valid])
        pred_label_test = np.zeros([popsize, self.num_test])

        pred_logits_train = np.zeros([popsize, self.num_train])
        pred_logits_valid = np.zeros([popsize, self.num_valid])
        pred_logits_test = np.zeros([popsize, self.num_test])

        Groups_info = self.Groups_info

        sens_attr = self.cal_sens_name
        group_dicts = Groups_info['group_dict_train']
        s_labels = group_dicts[sens_attr[0]][0]
        # print(s_labels)

        sens_idxs_train = Groups_info['sens_idxs_train']
        s_labels_train = sens_idxs_train[s_labels][0]
        S_train = np.zeros([1, self.num_train])
        S_train[0, s_labels_train] = 1

        sens_idxs_valid = Groups_info['sens_idxs_valid']
        s_labels_valid = sens_idxs_valid[s_labels][0]
        S_valid = np.zeros([1, self.num_valid])
        S_valid[0, s_labels_valid] = 1

        sens_idxs_test = Groups_info['sens_idxs_test']
        sens_idxs_name_test = Groups_info['sens_idxs_name_test']
        s_labels_test = sens_idxs_test[s_labels][0]
        S_test = np.zeros([1, self.num_test])
        S_test[0, s_labels_test] = 1

        self.use_gpu = torch.cuda.is_available()
        # self.use_gpu = False
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _, _, _, _, _, more_vals_test = ea.Cal_objectives(self.train_data, self.train_data_norm,
                                                          np.array(self.train_y).reshape(1, -1),
                                                          np.array(self.train_y).reshape(1, -1),
                                                          self.cal_sens_name, 2,
                                                          self.Groups_info['sens_idxs_test'], plan=self.cal_obj_plan,
                                                          dis=np.array(self.train_y).reshape(1, -1))
        self.more_info_num = len(more_vals_test)
        # 依次，accuracy MSE L2 Individual Group
        if is_recordmore == 0:
            AllObj_valid = np.zeros([popsize, 5])
            AllObj_test = np.zeros([popsize, 5])
            AllObj_train = np.zeros([popsize, 5])
        else:
            AllObj_valid = np.zeros([popsize, 5 + self.more_info_num])
            AllObj_test = np.zeros([popsize, 5 + self.more_info_num])
            AllObj_train = np.zeros([popsize, 5 + self.more_info_num])

        pop_logits_test = np.zeros([popsize, self.test_data_norm.shape[0]])
        # record_loss = np.zeros([popsize, self.epoches*10])
        # record_test = np.zeros([popsize, self.epoches*10])
        # Groups_info = []

        # for ti in range(10):
        #     print(ti)
        for idx in range(popsize):
            # adversary = copy.deepcopy(Adversary)
            adversary = Adversary
            individual = pop.Chrom[idx]  # 只是引用，不是复制，还会修改pop.Chrom 网络的值
            # individual = copy.deepcopy(pop.Chrom[idx])  # 不是引用，是复制，不会修改pop.Chrom 网络的值

            x_train = torch.Tensor(self.train_data_norm)
            y_train = torch.Tensor(np.array(self.train_y)).view(-1, 1)
            s_train = torch.Tensor(S_train[0]).view(-1, 1)

            x_test = torch.Tensor(self.test_data_norm)
            y_test = torch.Tensor(np.array(self.test_y)).view(-1, 1)
            s_test = torch.Tensor(S_test[0])

            x_valid = torch.Tensor(self.valid_data_norm)
            y_valid = torch.Tensor(self.valid_y).view(-1, 1)
            s_valid = torch.Tensor(S_valid[0])

            # adversary.to(device)
            # predictor.to(device)
            if self.use_gpu:
                individual.cuda()
                if use_GAN:
                    adversary.cuda()
            pop_optimizer = torch.optim.Adam(individual.parameters(), lr=self.learning_rate)
            if use_GAN:
                if adversary.no_targets == 1:
                    adv_optimizer = torch.optim.Adam(adversary.parameters(), lr=self.learning_rate)
                else:
                    adv_optimizer = torch.optim.Adam(adversary.parameters(), lr=self.learning_rate)
            loss_fn = torch.nn.BCELoss()

            train = TensorDataset(x_train, y_train, s_train)
            train_loader = DataLoader(train, batch_size=self.batch_size, shuffle=True)
            for epoch in range(self.epoches):
                individual.train()
                # forward_full(train_loader, predictor, adversary, criterion, device, optimizer_P=optimizer_P,
                #              optimizer_A=optimizer_A, train=True, alpha=0.3)
                for i, (x_batch, y_batch, s_batch) in enumerate(train_loader):
                    if self.use_gpu:
                        x_batch, y_batch, s_batch = x_batch.cuda(), y_batch.cuda(), s_batch.cuda()
                    y_logits, y_pred = individual(x_batch)
                    loss_P = loss_fn(y_pred, y_batch)

                    # calculate L2 regular
                    # reg_loss = None
                    # for param in individual.parameters():
                    #     if reg_loss is None:
                    #         reg_loss = 0.5 * torch.sum(param ** 2)
                    #     else:
                    #         reg_loss = reg_loss + 0.5 * param.norm(2) ** 2
                    # loss_P += self.L_regul * reg_loss
                    if use_GAN:
                        pred_z_logit, pred_z_prob = adversary(y_logits, y_batch)
                        loss_A = loss_fn(pred_z_prob, s_batch.reshape(-1, 1))
                        adv_optimizer.zero_grad()
                        pop_optimizer.zero_grad()
                        loss_A.backward(retain_graph=True)
                        grad_w_La = concat_grad(individual)
                    pop_optimizer.zero_grad()
                    loss_P.backward()
                    if use_GAN:
                        grad_w_Lp = concat_grad(individual)
                        proj_grad = project_grad(grad_w_Lp, grad_w_La)
                        alph1 = 1
                        grad_w_Lp = grad_w_Lp - alph1 * proj_grad - self.GAN_alpha * grad_w_La
                        replace_grad(individual, grad_w_Lp)
                    pop_optimizer.step()
                    if use_GAN:
                        adv_optimizer.step()

            with torch.no_grad():
                individual.eval()
                if self.use_gpu:
                    individual.cpu()
                    # if np.mod(gen, 20) == 0:
                    #     plot_decision_boundary(individual, x_train, y_train, 1-S_train[0], np.array([gen, idx]), dirName=dirName)

                l2_regularization = 0.0
                for param in individual.parameters():
                    l2_regularization += torch.norm(param, 2)  # L2 正则化

                logit_temp, pred_sigmoid_temp = individual(x_test)
                logits_test = np.array(pred_sigmoid_temp.detach())
                pred_label_test[idx][:] = get_label(logits_test.reshape(1, -1).copy())
                pop_logits_test[idx][:] = logits_test.reshape(1, -1)
                pred_logits_test[idx][:] = logits_test.reshape(1, -1)
                accuracy_test, accuracy_loss_test, individual_fairness_test, group_fairness_test, Groups_test, more_vals_test = ea.Cal_objectives(
                    self.test_data,
                    self.test_data_norm,
                    logits_test, y_test,
                    self.cal_sens_name,
                    2, self.Groups_info['sens_idxs_test'],
                    plan=self.cal_obj_plan,
                    dis=logit_temp)

                logit_temp, pred_sigmoid_temp = individual(x_train)
                logits_train = np.array(pred_sigmoid_temp.detach())
                pred_label_train[idx][:] = get_label(logits_train.reshape(1, -1).copy())
                pred_logits_train[idx][:] = logits_train.reshape(1, -1)
                accuracy_train, accuracy_loss_train, individual_fairness_train, group_fairness_train, Groups_train, more_vals_train = ea.Cal_objectives(
                    self.train_data,
                    self.train_data_norm,
                    logits_train, y_train,
                    self.cal_sens_name,
                    2, self.Groups_info['sens_idxs_train'],
                    plan=self.cal_obj_plan,
                    dis=logit_temp)

                logit_temp, pred_sigmoid_temp = individual(x_valid)
                logits_valid = np.array(pred_sigmoid_temp.detach())
                pred_label_valid[idx][:] = get_label(logits_valid.reshape(1, -1).copy())
                pred_logits_valid[idx][:] = logits_valid.reshape(1, -1)
                accuracy_valid, accuracy_loss_valid, individual_fairness_valid, group_fairness_valid, Groups_valid, more_vals_valid = ea.Cal_objectives(
                    self.valid_data,
                    self.valid_data_norm,
                    logits_valid, y_valid,
                    self.cal_sens_name,
                    2, self.Groups_info['sens_idxs_valid'],
                    plan=self.cal_obj_plan,
                    dis=logit_temp)
                # print('all is ok')
                ####################################################################################################
                # # in test data
                # print('The information in test data: ')
                # print('  accuracy: %.4f, MSE: %.4f individual fairness: %.5f, group fairness: %.5f\n'
                #       % (accuracy_test, accuracy_loss_test, individual_fairness_test, group_fairness_test))
                #
                # ####################################################################################################
                # # in train data
                # print('The formation in train data: ')
                # print('  accuracy: %.4f, MSE: %.4f, individual fairness: %.5f, group fairness: %.5f\n'
                #       % (accuracy_train, accuracy_loss_train, individual_fairness_train, group_fairness_train))
                ####################################################################################################

                # Groups_info.append(Groups_train)
                #
                if is_recordmore == 0:
                    AllObj_train[idx][:] = np.array(
                        [accuracy_train, accuracy_loss_train, l2_regularization, individual_fairness_train,
                         group_fairness_train])
                    AllObj_test[idx][:] = np.array(
                        [accuracy_test, accuracy_loss_test, l2_regularization, individual_fairness_test,
                         group_fairness_test])
                    AllObj_valid[idx][:] = np.array(
                        [accuracy_valid, accuracy_loss_valid, l2_regularization, individual_fairness_valid,
                         group_fairness_valid])
                else:
                    temp = np.zeros([1, 5 + self.more_info_num])
                    temp[0][0:5] = [accuracy_train, accuracy_loss_train, l2_regularization,
                                    individual_fairness_train, group_fairness_train]
                    temp[0][5:] = more_vals_train
                    AllObj_train[idx][:] = temp

                    temp = np.zeros([1, 5 + self.more_info_num])
                    temp[0][0:5] = [accuracy_test, accuracy_loss_test, l2_regularization, individual_fairness_test,
                                    group_fairness_test]
                    temp[0][5:] = more_vals_test
                    AllObj_test[idx][:] = temp

                    temp = np.zeros([1, 5 + self.more_info_num])
                    temp[0][0:5] = [accuracy_valid, accuracy_loss_valid, l2_regularization,
                                    individual_fairness_valid, group_fairness_valid]
                    temp[0][5:] = more_vals_valid
                    AllObj_valid[idx][:] = temp

                # AllObj_test[idx][:] = np.array([accuracy_loss_test, individual_fairness_train, group_fairness_train])
                # PopObj[idx][:] = np.array([individual_fairness, group_fairness])

        # mean_loss = np.mean(record_loss, axis=0)
        # mean_acc = np.mean(record_test, axis=0)
        # np.savetxt("compas_wd2_loss_lr0001.txt", record_loss)
        # np.savetxt("compas_wd2_acc_lr0001.txt", record_test)
        # plt.plot(np.array(range(self.epoches*10)), mean_loss)
        # plt.plot(np.array(range(self.epoches*10)), mean_acc)
        # end_time = time.time()
        # print('cost: ', end_time-start_time)
        # plt.show()
        # print('draw')

        pop.CV = np.zeros([popsize, 1])
        PopObj = AllObj_valid.copy()
        delete_list = []
        if 'accuracy' not in self.objectives_class:
            delete_list.append(0)
        if 'l2' not in self.objectives_class:
            delete_list.append(2)
        if 'Error' not in self.objectives_class:
            delete_list.append(1)
        if 'individual' not in self.objectives_class:
            delete_list.append(3)
        if 'group' not in self.objectives_class:
            delete_list.append(4)
        if is_recordmore == 1:
            for i in range(5, 5 + self.more_info_num):
                delete_list.append(i)
        pop.ObjV = np.delete(PopObj, delete_list, 1)  # 把求得的目标函数值赋值给种群pop的ObjV.
        if 'accuracy' in self.objectives_class:
            pop.ObjV[:, 0] = 1 - pop.ObjV[:, 0]

        pop.ObjV = np.delete(PopObj, delete_list, 1)  # 把求得的目标函数值赋值给种群pop的ObjV
        pop.ObjV_train = AllObj_train
        pop.ObjV_valid = AllObj_valid
        pop.ObjV_test = AllObj_test
        pop.pred_label_train = pred_label_train
        pop.pred_label_valid = pred_label_valid
        pop.pred_label_test = pred_label_test
        pop.pred_logits_train = pred_logits_train
        pop.pred_logits_valid = pred_logits_valid
        pop.pred_logits_test = pred_logits_test

        endtime = time.time()
        # if self.use_gpu:
        #     print('use gpu')
        # else:
        #     print('not use gpu')
        # print('calculate objectives run time:', endtime - start_time)

        return AllObj_train, AllObj_valid, AllObj_test
