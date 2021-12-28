import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from itertools import product
import itertools
# from GroupInfo import GroupInfo, GroupsInfo
import geatpy as ea
from sklearn.metrics import accuracy_score, roc_auc_score
import copy
from sklearn.metrics import log_loss
from scipy.spatial.distance import pdist, squareform


def Cal_AUC(logits, truelabel, group):
    sum_num = logits.shape[0] * logits.shape[1]

    pred_label = get_label(logits.copy())
    accuracy = np.sum(pred_label == truelabel) / sum_num
    MSE = np.mean(np.power(logits - truelabel, 2))

    group_names = group.keys()
    vals = []
    for group_name in group_names:
        idx = group[group_name]
        if np.all(truelabel[0][idx[0]] == truelabel[0][idx[0]][0]):
            # Only one class present in y_true, ROC AUC score is not defined in that case.
            val = -1
        else:
            val = roc_auc_score(truelabel[0][idx[0]], logits[0][idx[0]])
            vals.append(val)

    vals = np.array(vals)
    vals_mean = np.mean(vals)
    vals_vars = np.mean(np.power(vals - vals_mean, 2))
    vals_max = np.max(vals)
    vals_min = np.min(vals)

    return accuracy, MSE, vals_vars, 1 - vals_mean, vals


def DemographicParity_EqualizedOdds3(data, logits, truelabel, sensitive_attributions):
    # 计算的组区分方式为重叠式：1) female, male; 2) youth, old
    sum_num = logits.shape[0] * logits.shape[1]

    pred_label = get_label(logits.copy())
    accuracy = np.sum(pred_label == truelabel) / sum_num
    MSE = np.mean(np.power(logits - truelabel, 2))

    attribution = data.columns
    DemographicParity = 0.0
    EqualizedOdds = 0.0
    group_dict = {}
    Groups = []

    for sens in sensitive_attributions:
        temp = []
        for attr in attribution:
            temp1 = sens + '_'
            if temp1 in attr:
                temp.append(attr)
        group_dict.update({sens: temp})

    group_attr = []
    groups_total_inD = []  # 计算 Demographic Parity 的每个组的个数
    groups_pred1_inD = []  # 计算 Demographic Parity 中每个组预测为1的个数

    groups_total_is1_inE = []  # 计算 Equalized Odds 中每个组真实为1的个数
    groups_total_is0_inE = []  # 计算 Equalized Odds 中每个组真实为0的个数
    groups_true1_pred1_inE = []  # 计算 Equalized Odds 中每个组真实为1，预测为1的个数
    groups_true0_pred1_inE = []  # 计算 Equalized Odds 中每个组预测为0，预测为1的个数
    groups_pred1val_inD = []
    groups_total_is1val_inE = []
    groups_total_is0val_inE = []
    groups_true1_pred1val_inE = []
    groups_true0_pred1val_inE = []

    for sens in sensitive_attributions:
        group = group_dict[sens]
        groups_total_inD_temp = []
        groups_pred1val_inD_temp = []
        groups_pred1_inD_temp = []
        groups_total_is1_inE_temp = []
        groups_total_is0_inE_temp = []
        groups_true1_pred1_inE_temp = []
        groups_true0_pred1_inE_temp = []
        groups_total_is1val_inE_temp = []
        groups_total_is0val_inE_temp = []
        groups_true1_pred1val_inE_temp = []
        groups_true0_pred1val_inE_temp = []
        for g in group:
            flag = np.ones([1, sum_num]) == np.ones([1, sum_num])
            flag = flag & data[g]
            g_num = np.sum(flag)
            if g_num != 0:
                flag_pred1 = flag & (pred_label[0, :] == 1)  # 小组g中，被预测为1的flag

                flag_true1 = flag & (truelabel == 1)  # 小组g中，真实标签为1的flag
                flag_true0 = flag & (truelabel == 0)  # 小组g中，真实标签为0的flag

                flag_true1_pred1 = flag_true1 & (pred_label == 1)  # 小组g中，真实标签为1，预测也为1的flag
                flag_true0_pred1 = flag_true0 & (pred_label == 1)  # 小组g中，真实标签为0，预测为1的flag

                # record Demographic Parity
                groups_total_inD_temp.append(np.sum(flag))  # 记录当前这个group的个数
                groups_pred1val_inD_temp.append(
                    np.sum(np.power(logits[0, np.where(flag_pred1.values)], 2)))  # 记录当前这个group的预测为1的logits
                groups_pred1_inD_temp.append(np.sum(flag_pred1))  # 记录当前这个group的预测为1的个数

                # record Equalized Odds
                groups_total_is1_inE_temp.append(np.sum(flag_true1))
                groups_total_is0_inE_temp.append(np.sum(flag_true0))
                groups_true1_pred1_inE_temp.append(np.sum(flag_true1_pred1))
                groups_true0_pred1_inE_temp.append(np.sum(flag_true0_pred1))

                groups_total_is1val_inE_temp.append(np.sum(np.power(logits[0, np.where(flag_true1.values)], 2)))
                groups_total_is0val_inE_temp.append(np.sum(np.power(logits[0, np.where(flag_true0.values)], 2)))
                groups_true1_pred1val_inE_temp.append(np.sum(np.power(logits[0, np.where(flag_true1_pred1.values)], 2)))
                groups_true0_pred1val_inE_temp.append(np.sum(np.power(logits[0, np.where(flag_true0_pred1.values)], 2)))

        groups_total_inD.append(groups_total_inD_temp)
        groups_pred1val_inD.append(groups_pred1val_inD_temp)
        groups_pred1_inD.append(groups_pred1_inD_temp)

        # record Equalized Odds
        groups_total_is1_inE.append(groups_total_is1_inE_temp)
        groups_total_is0_inE.append(groups_total_is0_inE_temp)
        groups_true1_pred1_inE.append(groups_true1_pred1_inE_temp)
        groups_true0_pred1_inE.append(groups_true0_pred1_inE_temp)

        groups_total_is1val_inE.append(groups_total_is1val_inE_temp)
        groups_total_is0val_inE.append(groups_total_is0val_inE_temp)
        groups_true1_pred1val_inE.append(groups_true1_pred1val_inE_temp)
        groups_true0_pred1val_inE.append(groups_true0_pred1val_inE_temp)

    groups_total_inD = np.array(groups_total_inD)
    groups_pred1_inD = np.array(groups_pred1_inD)
    groups_pred1val_inD = np.array(groups_pred1val_inD)

    groups_total_is1_inE = np.array(groups_total_is1_inE)
    groups_total_is0_inE = np.array(groups_total_is0_inE)
    groups_true1_pred1_inE = np.array(groups_true1_pred1_inE)
    groups_true0_pred1_inE = np.array(groups_true0_pred1_inE)

    groups_total_is1val_inE = np.array(groups_total_is1val_inE)
    groups_total_is0val_inE = np.array(groups_total_is0val_inE)
    groups_true1_pred1val_inE = np.array(groups_true1_pred1val_inE)
    groups_true0_pred1val_inE = np.array(groups_true0_pred1val_inE)

    is_propor = 0
    # calculate Demographic Parity
    pro_D = groups_pred1_inD / groups_total_inD
    # pro_D = groups_pred1val_inD / groups_total_inD
    mean_D = np.mean(pro_D, axis=1)
    proportial = groups_total_inD / np.sum(groups_total_inD)
    if is_propor == 1:
        DemographicParity = np.sum(np.abs(pro_D - mean_D.reshape(-1, 1)) * proportial)
    else:
        DemographicParity = np.sum(np.abs(pro_D - mean_D).reshape(-1, 1))

    # record Equalized Odds
    pro_E1 = groups_true1_pred1_inE / groups_total_is1_inE
    # pro_E1 = groups_true1_pred1val_inE / groups_total_is1_inE
    mean_E1 = np.mean(pro_E1, axis=1)
    proportial1 = groups_total_is1_inE / np.sum(groups_total_is1_inE)
    if is_propor == 1:
        EqualizedOdds1 = np.sum(np.abs(pro_E1 - mean_E1.reshape(-1, 1)) * proportial1)
    else:
        EqualizedOdds1 = np.sum(np.abs(pro_E1 - mean_E1.reshape(-1, 1)))

    pro_E2 = groups_true0_pred1_inE / groups_total_is0_inE
    # pro_E2 = groups_true0_pred1val_inE / groups_total_is0_inE
    mean_E2 = np.mean(pro_E2, axis=1)
    proportial2 = groups_total_is0_inE / np.sum(groups_total_is0_inE)
    if is_propor == 1:
        EqualizedOdds2 = np.sum(np.abs(pro_E2 - mean_E2.reshape(-1, 1)) * proportial2)
    else:
        EqualizedOdds2 = np.sum(np.abs(pro_E2 - mean_E2.reshape(-1, 1)))

    EqualizedOdds = (EqualizedOdds1 + EqualizedOdds2) / 2

    Groups_info = []
    return accuracy, MSE, DemographicParity * 100, EqualizedOdds * 100, Groups_info


def DemographicParity_EqualizedOdds2(data, logits, truelabel, sensitive_attributions):
    # 计算的组区分方式为非重叠式：1) female+youth, 2) male+youth, 3) female+old, 4) female+youth
    sum_num = logits.shape[0] * logits.shape[1]

    pred_label = get_label(logits.copy())
    accuracy = np.sum(pred_label == truelabel) / sum_num
    MSE = np.mean(np.power(logits - truelabel, 2))

    attribution = data.columns
    DemographicParity = 0.0
    EqualizedOdds = 0.0
    group_dict = {}
    Groups = []

    for sens in C:
        temp = []
        for attr in attribution:
            temp1 = sens + '_'
            if temp1 in attr:
                temp.append(attr)
        group_dict.update({sens: temp})

    group_attr = []
    groups_total_inD = []  # 计算 Demographic Parity 的每个组的个数
    groups_pred1_inD = []  # 计算 Demographic Parity 中每个组预测为1的个数

    groups_total_is1_inE = []  # 计算 Equalized Odds 中每个组真实为1的个数
    groups_total_is0_inE = []  # 计算 Equalized Odds 中每个组真实为0的个数
    groups_true1_pred1_inE = []  # 计算 Equalized Odds 中每个组真实为1，预测为1的个数
    groups_true0_pred1_inE = []  # 计算 Equalized Odds 中每个组预测为0，预测为1的个数
    groups_pred1val_inD = []
    groups_total_is1val_inE = []
    groups_total_is0val_inE = []
    groups_true1_pred1val_inE = []
    groups_true0_pred1val_inE = []

    for sens in sensitive_attributions:
        group_attr.append(group_dict[sens])
    for item in product(*eval(str(group_attr))):
        group = item
        flag = np.ones([1, sum_num]) == np.ones([1, sum_num])
        for g in group:
            flag = flag & data[g]
        g_num = np.sum(flag)
        if g_num != 0:
            flag_pred1 = flag & (pred_label[0, :] == 1)  # 小组g中，被预测为1的flag

            flag_true1 = flag & (truelabel == 1)  # 小组g中，真实标签为1的flag
            flag_true0 = flag & (truelabel == 0)  # 小组g中，真实标签为0的flag

            flag_true1_pred1 = flag_true1 & (pred_label == 1)  # 小组g中，真实标签为1，预测也为1的flag
            flag_true0_pred1 = flag_true0 & (pred_label == 1)  # 小组g中，真实标签为0，预测为1的flag

            # record Demographic Parity
            groups_total_inD.append(np.sum(flag))  # 记录当前这个group的个数
            groups_pred1val_inD.append(
                np.sum(np.power(logits[0, np.where(flag_pred1.values)], 2)))  # 记录当前这个group的预测为1的logits
            groups_pred1_inD.append(np.sum(flag_pred1))  # 记录当前这个group的预测为1的个数

            # record Equalized Odds
            groups_total_is1_inE.append(np.sum(flag_true1))
            groups_total_is0_inE.append(np.sum(flag_true0))
            groups_true1_pred1_inE.append(np.sum(flag_true1_pred1))
            groups_true0_pred1_inE.append(np.sum(flag_true0_pred1))

            groups_total_is1val_inE.append(np.sum(np.power(logits[0, np.where(flag_true1.values)], 2)))
            groups_total_is0val_inE.append(np.sum(np.power(logits[0, np.where(flag_true0.values)], 2)))
            groups_true1_pred1val_inE.append(np.sum(np.power(logits[0, np.where(flag_true1_pred1.values)], 2)))
            groups_true0_pred1val_inE.append(np.sum(np.power(logits[0, np.where(flag_true0_pred1.values)], 2)))

    groups_total_inD = np.array(groups_total_inD)
    groups_pred1_inD = np.array(groups_pred1_inD)
    groups_pred1val_inD = np.array(groups_pred1val_inD)

    groups_total_is1_inE = np.array(groups_total_is1_inE)
    groups_total_is0_inE = np.array(groups_total_is0_inE)
    groups_true1_pred1_inE = np.array(groups_true1_pred1_inE)
    groups_true0_pred1_inE = np.array(groups_true0_pred1_inE)

    groups_total_is1val_inE = np.array(groups_total_is1val_inE)
    groups_total_is0val_inE = np.array(groups_total_is0val_inE)
    groups_true1_pred1val_inE = np.array(groups_true1_pred1val_inE)
    groups_true0_pred1val_inE = np.array(groups_true0_pred1val_inE)

    is_propor = 0
    is_logits = 1
    # calculate Demographic Parity
    if is_logits == 1:
        pro_D = groups_pred1val_inD / groups_total_inD
    else:
        pro_D = groups_pred1_inD / groups_total_inD
    mean_D = np.mean(pro_D)
    proportial = groups_total_inD / np.sum(groups_total_inD)
    if is_propor == 1:
        DemographicParity = np.sum(np.abs(pro_D - mean_D) * proportial)
    else:
        DemographicParity = np.sum(np.abs(pro_D - mean_D))

    # record Equalized Odds
    if is_logits == 1:
        pro_E1 = groups_true1_pred1val_inE / groups_total_is1_inE
    else:
        pro_E1 = groups_true1_pred1_inE / groups_total_is1_inE
    mean_E1 = np.mean(pro_E1)
    proportial1 = groups_total_is1_inE / np.sum(groups_total_is1_inE)
    if is_propor == 1:
        EqualizedOdds1 = np.sum(np.abs(pro_E1 - mean_E1) * proportial1)
    else:
        EqualizedOdds1 = np.sum(np.abs(pro_E1 - mean_E1))
    if is_logits == 1:
        pro_E2 = groups_true0_pred1val_inE / groups_total_is0_inE
    else:
        pro_E2 = groups_true0_pred1_inE / groups_total_is0_inE

    mean_E2 = np.mean(pro_E2)
    proportial2 = groups_total_is0_inE / np.sum(groups_total_is0_inE)
    if is_propor == 1:
        EqualizedOdds2 = np.sum(np.abs(pro_E2 - mean_E2) * proportial2)
    else:
        EqualizedOdds2 = np.sum(np.abs(pro_E2 - mean_E2))

    EqualizedOdds = (EqualizedOdds1 + EqualizedOdds2) / 2

    Groups_info = []
    return accuracy, MSE, DemographicParity, EqualizedOdds, Groups_info, np.hstack([pro_D, pro_E1, pro_E2])


def DemographicParity_EqualizedOdds4(data, logits, truelabel, sensitive_attributions):
    # 与 DemographicParity_EqualizedOdds2 相同
    # 计算的组区分方式为非重叠式：1) female+youth, 2) male+youth, 3) female+old, 4) female+youth
    sum_num = logits.shape[0] * logits.shape[1]

    pred_label = get_label(logits.copy())
    accuracy = np.sum(pred_label == truelabel) / sum_num
    MSE = np.mean(np.power(logits - truelabel, 2))

    attribution = data.columns
    DemographicParity = 0.0
    EqualizedOdds = 0.0
    group_dict = {}
    Groups = []

    for sens in sensitive_attributions:
        temp = []
        for attr in attribution:
            temp1 = sens + '_'
            if temp1 in attr:
                temp.append(attr)
        group_dict.update({sens: temp})

    group_attr = []
    groups_total_inD = []  # 计算 Demographic Parity 的每个组的个数
    groups_pred1_inD = []  # 计算 Demographic Parity 中每个组预测为1的个数

    groups_total_is1_inE = []  # 计算 Equalized Odds 中每个组真实为1的个数
    groups_total_is0_inE = []  # 计算 Equalized Odds 中每个组真实为0的个数
    groups_true1_pred1_inE = []  # 计算 Equalized Odds 中每个组真实为1，预测为1的个数
    groups_true0_pred1_inE = []  # 计算 Equalized Odds 中每个组预测为0，预测为1的个数
    groups_true1_pred0_inE = []
    groups_true0_pred0_inE = []
    groups_pred1val_inD = []
    groups_total_is1val_inE = []
    groups_total_is0val_inE = []
    groups_true1_pred1val_inE = []
    groups_true0_pred1val_inE = []
    groups_true1_pred0val_inE = []
    groups_true0_pred0val_inE = []


    for sens in sensitive_attributions:
        group_attr.append(group_dict[sens])
    for item in product(*eval(str(group_attr))):
        group = item
        flag = np.ones([1, sum_num]) == np.ones([1, sum_num])
        for g in group:
            flag = flag & data[g]
        g_num = np.sum(flag)
        if g_num != 0 and g_num > 10:
            flag_pred1 = flag & (pred_label[0, :] == 1)  # 小组g中，被预测为1的flag

            flag_true1 = flag & (truelabel == 1)  # 小组g中，真实标签为1的flag
            flag_true0 = flag & (truelabel == 0)  # 小组g中，真实标签为0的flag

            flag_true1_pred1 = flag_true1 & (pred_label == 1)  # 小组g中，真实标签为1，预测也为1的flag
            flag_true0_pred1 = flag_true0 & (pred_label == 1)  # 小组g中，真实标签为0，预测为1的flag
            flag_true1_pred0 = flag_true1 & (pred_label == 0)  # 小组g中，真实标签为1，预测为0的flag
            flag_true0_pred0 = flag_true0 & (pred_label == 0)  # 小组g中，真实标签为1，预测为0的flag

            # record Demographic Parity
            groups_total_inD.append(np.sum(flag))  # 记录当前这个group的个数
            groups_pred1val_inD.append(
                np.sum(np.power(logits[0, np.where(flag_pred1.values)], 2)))  # 记录当前这个group的预测为1的logits
            groups_pred1_inD.append(np.sum(flag_pred1))  # 记录当前这个group的预测为1的个数

            # record Equalized Odds
            groups_total_is1_inE.append(np.sum(flag_true1))
            groups_total_is0_inE.append(np.sum(flag_true0))
            groups_true1_pred1_inE.append(np.sum(flag_true1_pred1))
            groups_true0_pred1_inE.append(np.sum(flag_true0_pred1))
            groups_true1_pred0_inE.append(np.sum(flag_true1_pred0))
            groups_true0_pred0_inE.append(np.sum(flag_true0_pred0))

            groups_total_is1val_inE.append(np.sum(np.power(logits[0, np.where(flag_true1.values)], 2)))
            groups_total_is0val_inE.append(np.sum(np.power(logits[0, np.where(flag_true0.values)], 2)))
            groups_true1_pred1val_inE.append(np.sum(np.power(logits[0, np.where(flag_true1_pred1.values)], 2)))
            groups_true0_pred1val_inE.append(np.sum(np.power(logits[0, np.where(flag_true0_pred1.values)], 2)))
            groups_true1_pred0val_inE.append(np.sum(np.power(logits[0, np.where(flag_true1_pred0.values)], 2)))
            groups_true0_pred0val_inE.append(np.sum(np.power(logits[0, np.where(flag_true0_pred0.values)], 2)))

    groups_total_inD = np.array(groups_total_inD)
    groups_pred1_inD = np.array(groups_pred1_inD)
    groups_pred1val_inD = np.array(groups_pred1val_inD)

    groups_total_is1_inE = np.array(groups_total_is1_inE)
    groups_total_is0_inE = np.array(groups_total_is0_inE)
    groups_true1_pred1_inE = np.array(groups_true1_pred1_inE)
    groups_true0_pred1_inE = np.array(groups_true0_pred1_inE)
    groups_true1_pred0_inE = np.array(groups_true1_pred0_inE)
    groups_true0_pred0_inE = np.array(groups_true0_pred0_inE)

    groups_total_is1val_inE = np.array(groups_total_is1val_inE)
    groups_total_is0val_inE = np.array(groups_total_is0val_inE)
    groups_true1_pred1val_inE = np.array(groups_true1_pred1val_inE)
    groups_true0_pred1val_inE = np.array(groups_true0_pred1val_inE)

    is_propor = 0
    is_logits = 0
    # calculate Demographic Parity
    if is_logits == 1:
        pro_D = groups_pred1val_inD / groups_total_inD
    else:
        pro_D = groups_pred1_inD / groups_total_inD
    mean_D = np.mean(pro_D)
    proportial = groups_total_inD / np.sum(groups_total_inD)
    if is_propor == 1:
        DemographicParity = np.sum(np.abs(pro_D - mean_D) * proportial)
    else:
        DemographicParity = np.sum(np.abs(pro_D - mean_D))

    # record Equalized Odds
    if is_logits == 1:
        pro_E1 = groups_true1_pred1val_inE / groups_total_is1_inE
    else:
        pro_E1 = groups_true1_pred1_inE / groups_total_is1_inE   # P( d=1 | y=1, z )
    mean_E1 = np.mean(pro_E1)
    proportial1 = groups_total_is1_inE / np.sum(groups_total_is1_inE)
    if is_propor == 1:
        EqualizedOdds1 = np.sum(np.abs(pro_E1 - mean_E1) * proportial1)
    else:
        EqualizedOdds1 = np.sum(np.abs(pro_E1 - mean_E1))

    if is_logits == 1:
        pro_E2 = groups_true0_pred1val_inE / groups_total_is0_inE
    else:
        pro_E2 = groups_true0_pred1_inE / groups_total_is0_inE  # P( d=1 | y=0, z )
        pro_E3 = groups_true1_pred0_inE / groups_total_is1_inE  # P( d=0 | y=1, z )
        pro_E4 = groups_true0_pred0_inE / groups_total_is0_inE  # P( d=0 | y=0, z )

    mean_E2 = np.mean(pro_E2)
    proportial2 = groups_total_is0_inE / np.sum(groups_total_is0_inE)
    if is_propor == 1:
        EqualizedOdds2 = np.sum(np.abs(pro_E2 - mean_E2) * proportial2)
    else:
        EqualizedOdds2 = np.sum(np.abs(pro_E2 - mean_E2))

    EqualizedOdds = (EqualizedOdds1 + EqualizedOdds2) / 2

    Groups_info = []
    print(np.hstack([accuracy, pro_D, pro_E1, pro_E2]))
    return accuracy, MSE, DemographicParity, EqualizedOdds, Groups_info, np.hstack([pro_D, pro_E1, pro_E2])


def DemographicParity_EqualizedOdds(data, logits, truelabel, sensitive_attributions):
    sum_num = logits.shape[0] * logits.shape[1]

    pred_label = get_label(logits.copy())
    accuracy = np.sum(pred_label == truelabel) / sum_num
    MSE = np.mean(np.power(logits - truelabel, 2))

    attribution = data.columns
    DemographicParity = 0.0
    EqualizedOdds = 0.0
    group_dict = {}
    Groups = []

    for sens in sensitive_attributions:
        temp = []
        for attr in attribution:
            temp1 = sens + '_'
            if temp1 in attr:
                temp.append(attr)
        group_dict.update({sens: temp})

    group_attr = []
    groups_total_inD = []  # 计算 Demographic Parity 的每个组的个数
    groups_pred1_inD = []  # 计算 Demographic Parity 中每个组预测为1的个数

    groups_total_is1_inE = []  # 计算 Equalized Odds 中每个组真实为1的个数
    groups_total_is0_inE = []  # 计算 Equalized Odds 中每个组真实为0的个数
    groups_true1_pred1_inE = []  # 计算 Equalized Odds 中每个组真实为1，预测为1的个数
    groups_true0_pred1_inE = []  # 计算 Equalized Odds 中每个组预测为0，预测为1的个数

    for sens in sensitive_attributions:
        group_attr.append(group_dict[sens])
    for item in product(*eval(str(group_attr))):
        group = item
        flag = np.ones([1, sum_num]) == np.ones([1, sum_num])
        for g in group:
            flag = flag & data[g]
        g_num = np.sum(flag)
        if g_num != 0:
            flag_pred1 = flag & (pred_label[0, :] == 1)  # 小组g中，被预测为1的flag

            flag_true1 = flag & (truelabel == 1)  # 小组g中，真实标签为1的flag
            flag_true0 = flag & (truelabel == 0)  # 小组g中，真实标签为0的flag

            flag_true1_pred1 = flag_true1 & (pred_label == 1)  # 小组g中，真实标签为1，预测也为1的flag
            flag_true0_pred1 = flag_true0 & (pred_label == 1)  # 小组g中，真实标签为0，预测为1的flag

            # record Demographic Parity
            groups_total_inD.append(np.sum(flag))
            groups_pred1_inD.append(np.sum(flag_pred1))

            # record Equalized Odds
            groups_total_is1_inE.append(np.sum(flag_true1))
            groups_total_is0_inE.append(np.sum(flag_true0))
            groups_true1_pred1_inE.append(np.sum(flag_true1_pred1))
            groups_true0_pred1_inE.append(np.sum(flag_true0_pred1))
    groups_total_inD = np.array(groups_total_inD)
    groups_pred1_inD = np.array(groups_pred1_inD)
    groups_total_is1_inE = np.array(groups_total_is1_inE)
    groups_total_is0_inE = np.array(groups_total_is0_inE)
    groups_true1_pred1_inE = np.array(groups_true1_pred1_inE)
    groups_true0_pred1_inE = np.array(groups_true0_pred1_inE)

    is_propor = 0
    # calculate Demographic Parity
    pro_D = groups_pred1_inD / groups_total_inD
    mean_D = np.mean(pro_D)
    proportial = groups_total_inD / np.sum(groups_total_inD)
    if is_propor == 1:
        DemographicParity = np.sum(np.abs(pro_D - mean_D) * proportial)
    else:
        DemographicParity = np.sum(np.abs(pro_D - mean_D))

    # record Equalized Odds
    pro_E1 = groups_true1_pred1_inE / groups_total_is1_inE
    mean_E1 = np.mean(pro_E1)
    proportial1 = groups_total_is1_inE / np.sum(groups_total_is1_inE)
    if is_propor == 1:
        EqualizedOdds1 = np.sum(np.abs(pro_E1 - mean_E1) * proportial1)
    else:
        EqualizedOdds1 = np.sum(np.abs(pro_E1 - mean_E1))

    pro_E2 = groups_true0_pred1_inE / groups_total_is0_inE
    mean_E2 = np.mean(pro_E2)
    proportial2 = groups_total_is0_inE / np.sum(groups_total_is0_inE)
    if is_propor == 1:
        EqualizedOdds2 = np.sum(np.abs(pro_E2 - mean_E2) * proportial2)
    else:
        EqualizedOdds2 = np.sum(np.abs(pro_E2 - mean_E2))

    EqualizedOdds = (EqualizedOdds1 + EqualizedOdds2) / 2

    Groups_info = []
    return accuracy, MSE, DemographicParity * 100, EqualizedOdds * 100, Groups_info


def change2array(listt):
    return np.array(listt)


def DP_EO_DI(data, logits, truelabel, sensitive_attributions):
    # FNNC: Achieving Fairness through Neural Networks
    # DP: Demographic Parity
    # EO: Equalized Odds
    # DI: Disparate Impact
    # 计算的组区分方式为非重叠式：1) female+youth, 2) male+youth, 3) female+old, 4) female+youth
    sum_num = logits.shape[0] * logits.shape[1]

    pred_label = get_label(logits.copy())
    accuracy = np.sum(pred_label == truelabel) / sum_num
    MSE = np.mean(np.power(logits - truelabel, 2))

    attribution = data.columns
    DemographicParity = 0.0
    EqualizedOdds = 0.0
    group_dict = {}
    Groups = []

    for sens in sensitive_attributions:
        temp = []
        for attr in attribution:
            temp1 = sens + '_'
            if temp1 in attr:
                temp.append(attr)
        group_dict.update({sens: temp})

    group_attr = []
    groups_num = []  # 计算 Demographic Parity 的每个组的个数
    FPR = []
    FNR = []
    DP = []
    group_attr = []
    groups_total_inD = []  # 计算 Demographic Parity 的每个组的个数
    groups_pred1_inD = []  # 计算 Demographic Parity 中每个组预测为1的个数

    groups_total_is1_inE = []  # 计算 Equalized Odds 中每个组真实为1的个数
    groups_total_is0_inE = []  # 计算 Equalized Odds 中每个组真实为0的个数
    groups_true1_pred1_inE = []  # 计算 Equalized Odds 中每个组真实为1，预测为1的个数
    groups_true0_pred1_inE = []  # 计算 Equalized Odds 中每个组预测为0，预测为1的个数
    groups_total_is1val_inE = []
    groups_total_is0val_inE = []
    groups_true1_pred1val_inE = []
    groups_true0_pred1val_inE = []

    for sens in sensitive_attributions:
        group_attr.append(group_dict[sens])
    for item in product(*eval(str(group_attr))):
        group = item
        flag = np.ones([1, sum_num]) == np.ones([1, sum_num])
        for g in group:
            flag = flag & data[g]
        g_num = np.sum(flag)
        if g_num != 0:
            groups_num.append(np.sum(flag))  # 记录当前这个group的个数

            # record Demographic Parity
            DP.append(np.sum(logits[0, np.where(flag)]))

            # record Equalized Odds
            FPR.append(np.sum(logits[0, np.where(flag)] * (1 - truelabel[0, np.where(flag)])))
            FNR.append(np.sum((1 - logits[0, np.where(flag)]) * (truelabel[0, np.where(flag)])))

            flag_pred1 = flag & (pred_label[0, :] == 1)  # 小组g中，被预测为1的flag

            ##################################################
            # record metrics
            flag_true1 = flag & (truelabel == 1)  # 小组g中，真实标签为1的flag
            flag_true0 = flag & (truelabel == 0)  # 小组g中，真实标签为0的flag

            flag_true1_pred1 = flag_true1 & (pred_label == 1)  # 小组g中，真实标签为1，预测也为1的flag
            flag_true0_pred1 = flag_true0 & (pred_label == 1)  # 小组g中，真实标签为0，预测为1的flag

            # record Demographic Parity
            groups_total_inD.append(np.sum(flag))  # 记录当前这个group的个数
            groups_pred1_inD.append(np.sum(flag_pred1))  # 记录当前这个group的预测为1的个数

            # record Equalized Odds
            groups_total_is1_inE.append(np.sum(flag_true1))
            groups_total_is0_inE.append(np.sum(flag_true0))
            groups_true1_pred1_inE.append(np.sum(flag_true1_pred1))
            groups_true0_pred1_inE.append(np.sum(flag_true0_pred1))

    DP = change2array(DP.copy())
    groups_num = change2array(groups_num.copy())
    FPR = change2array(FPR.copy())
    FNR = change2array(FNR.copy())

    # calculate Demographic Parity value
    DP_01 = DP / groups_num
    mean_D = np.mean(DP_01)
    DP_value = np.sum(np.abs(DP_01 - mean_D))

    # record Equalized Odds
    FPR_01 = FPR / groups_num
    fpr_value = np.sum(np.abs(FPR_01 - np.mean(FPR_01)))
    FNR_01 = FNR / groups_num
    fnr_value = np.sum(np.abs(FNR_01 - np.mean(FNR_01)))
    EO_values = fpr_value + fnr_value
    # EO_values = fnr_value
    # fpr_value to groups_true0_pred1_inE pro_E2
    # fnr_value to groups_true1_pred1_inE pro_E1
    Groups_info = []

    ##################################################
    # record metrics
    groups_total_inD = np.array(groups_total_inD)
    groups_pred1_inD = np.array(groups_pred1_inD)

    groups_total_is1_inE = np.array(groups_total_is1_inE)
    groups_total_is0_inE = np.array(groups_total_is0_inE)
    groups_true1_pred1_inE = np.array(groups_true1_pred1_inE)
    groups_true0_pred1_inE = np.array(groups_true0_pred1_inE)

    # calculate Demographic Parity
    pro_D = groups_pred1_inD / groups_total_inD

    # record Equalized Odds

    pro_E1 = groups_true1_pred1_inE / groups_total_is1_inE
    pro_E2 = groups_true0_pred1_inE / groups_total_is0_inE

    mean_D = np.mean(pro_D)
    mean_E1 = np.mean(pro_E1)
    mean_E2 = np.mean(pro_E2)

    DemographicParity = np.sum(np.abs(pro_D - mean_D))
    EqualizedOdds1 = np.sum(np.abs(pro_E1 - mean_E1))
    EqualizedOdds2 = np.sum(np.abs(pro_E2 - mean_E2))
    EqualizedOdds = (EqualizedOdds1 + EqualizedOdds2) / 2

    # DP_value = DP_value * (1 - np.exp(-5 * DemographicParity))
    # EO_values = EO_values * (1 - np.exp(-5 * EqualizedOdds))

    return accuracy, MSE, DP_value, EO_values, Groups_info, np.hstack([pro_D, pro_E1, pro_E2])


def DP_EO_DI2(data, dis, logits, truelabel, sensitive_attributions):
    # FNNC: Achieving Fairness through Neural Networks
    # DP: Demographic Parity
    # EO: Equalized Odds
    # DI: Disparate Impact
    # 计算的组区分方式为非重叠式：1) female+youth, 2) male+youth, 3) female+old, 4) female+youth
    # 与 1 不同的地方是将logits替换为 ln(logits)-ln(1-logits)
    dis = np.array(dis)
    sum_num = dis.shape[0] * dis.shape[1]

    pred_label = get_label(logits.copy())
    accuracy = np.sum(pred_label == truelabel) / sum_num
    MSE = np.mean(np.power(logits - truelabel, 2))

    attribution = data.columns
    DemographicParity = 0.0
    EqualizedOdds = 0.0
    group_dict = {}
    Groups = []
    dis = np.abs(np.array(dis.reshape(1, -1)))

    for sens in sensitive_attributions:
        temp = []
        for attr in attribution:
            temp1 = sens + '_'
            if temp1 in attr:
                temp.append(attr)
        group_dict.update({sens: temp})

    group_attr = []
    groups_num = []  # 计算 Demographic Parity 的每个组的个数
    FPR = []
    FNR = []
    DP = []
    group_attr = []
    groups_total_inD = []  # 计算 Demographic Parity 的每个组的个数
    groups_pred1_inD = []  # 计算 Demographic Parity 中每个组预测为1的个数

    groups_total_is1_inE = []  # 计算 Equalized Odds 中每个组真实为1的个数
    groups_total_is0_inE = []  # 计算 Equalized Odds 中每个组真实为0的个数
    groups_true1_pred1_inE = []  # 计算 Equalized Odds 中每个组真实为1，预测为1的个数
    groups_true0_pred1_inE = []  # 计算 Equalized Odds 中每个组预测为0，预测为1的个数
    groups_total_is1val_inE = []
    groups_total_is0val_inE = []
    groups_true1_pred1val_inE = []
    groups_true0_pred1val_inE = []

    for sens in sensitive_attributions:
        group_attr.append(group_dict[sens])
    for item in product(*eval(str(group_attr))):
        group = item
        flag = np.ones([1, sum_num]) == np.ones([1, sum_num])
        for g in group:
            flag = flag & data[g]
        g_num = np.sum(flag)
        if g_num != 0:
            groups_num.append(np.sum(flag))  # 记录当前这个group的个数

            # record Demographic Parity
            DP.append(np.sum(dis[0, np.where(flag)]))

            # record Equalized Odds
            FPR.append(np.sum(dis[0, np.where(flag)] * (1 - truelabel[0, np.where(flag)])))
            FNR.append(np.sum((1 - dis[0, np.where(flag)]) * (truelabel[0, np.where(flag)])))

            flag_pred1 = flag & (pred_label[0, :] == 1)  # 小组g中，被预测为1的flag

            ##################################################
            # record metrics
            flag_true1 = flag & (truelabel == 1)  # 小组g中，真实标签为1的flag
            flag_true0 = flag & (truelabel == 0)  # 小组g中，真实标签为0的flag

            flag_true1_pred1 = flag_true1 & (pred_label == 1)  # 小组g中，真实标签为1，预测也为1的flag
            flag_true0_pred1 = flag_true0 & (pred_label == 1)  # 小组g中，真实标签为0，预测为1的flag

            # record Demographic Parity
            groups_total_inD.append(np.sum(flag))  # 记录当前这个group的个数
            groups_pred1_inD.append(np.sum(flag_pred1))  # 记录当前这个group的预测为1的个数

            # record Equalized Odds
            groups_total_is1_inE.append(np.sum(flag_true1))
            groups_total_is0_inE.append(np.sum(flag_true0))
            groups_true1_pred1_inE.append(np.sum(flag_true1_pred1))
            groups_true0_pred1_inE.append(np.sum(flag_true0_pred1))

    DP = change2array(DP.copy())
    groups_num = change2array(groups_num.copy())
    FPR = change2array(FPR.copy())
    FNR = change2array(FNR.copy())

    # calculate Demographic Parity value
    DP_01 = DP / groups_num
    mean_D = np.mean(DP_01)
    DP_value = np.sum(np.abs(DP_01 - mean_D))

    # record Equalized Odds
    FPR_01 = FPR / groups_num
    fpr_value = np.sum(np.abs(FPR_01 - np.mean(FPR_01)))
    FNR_01 = FNR / groups_num
    fnr_value = np.sum(np.abs(FNR_01 - np.mean(FNR_01)))
    EO_values = fpr_value + fnr_value
    # EO_values = fnr_value
    # fpr_value to groups_true0_pred1_inE pro_E2
    # fnr_value to groups_true1_pred1_inE pro_E1
    Groups_info = []

    ##################################################
    # record metrics
    groups_total_inD = np.array(groups_total_inD)
    groups_pred1_inD = np.array(groups_pred1_inD)

    groups_total_is1_inE = np.array(groups_total_is1_inE)
    groups_total_is0_inE = np.array(groups_total_is0_inE)
    groups_true1_pred1_inE = np.array(groups_true1_pred1_inE)
    groups_true0_pred1_inE = np.array(groups_true0_pred1_inE)

    # calculate Demographic Parity
    pro_D = groups_pred1_inD / groups_total_inD

    # record Equalized Odds

    pro_E1 = groups_true1_pred1_inE / groups_total_is1_inE
    pro_E2 = groups_true0_pred1_inE / groups_total_is0_inE

    mean_D = np.mean(pro_D)
    mean_E1 = np.mean(pro_E1)
    mean_E2 = np.mean(pro_E2)

    DemographicParity = np.sum(np.abs(pro_D - mean_D))
    EqualizedOdds1 = np.sum(np.abs(pro_E1 - mean_E1))
    EqualizedOdds2 = np.sum(np.abs(pro_E2 - mean_E2))
    EqualizedOdds = (EqualizedOdds1 + EqualizedOdds2) / 2

    # DP_value = DP_value * (1 - np.exp(-5 * DemographicParity))
    # EO_values = EO_values * (1 - np.exp(-5 * EqualizedOdds))

    return accuracy, MSE, DP_value, EO_values, Groups_info, np.hstack([pro_D, pro_E1, pro_E2])


def get_label(logits):
    pred_label = logits
    pred_label[np.where(pred_label >= 0.5)] = 1
    pred_label[np.where(pred_label < 0.5)] = 0
    pred_label = pred_label.reshape(1, logits.shape[0] * logits.shape[1])
    pred_label = pred_label.reshape(1, -1)
    return pred_label


def calcul_indivi(benefits, alpha):
    # The method is from "Unified Approach to Quantifying Algorithmic Unfairness:
    # Measuring Individual & Group Unfairness via Inequality Indices"

    num = benefits.shape[0] * benefits.shape[1]
    mu = np.mean(benefits)
    # individual_fitness = np.sum(np.power(benefits/mu, alpha)-1)/(num*(alpha-1)*alpha)
    if mu == 0:
        individual_fitness = np.sum(np.power(1, alpha) - 1) / (num * (alpha - 1) * alpha)
    else:
        individual_fitness = np.sum(np.power(benefits / mu, alpha) - 1) / (num * (alpha - 1) * alpha)
    return individual_fitness


def get_average(group_values, plan):
    if plan == 1:
        values = []
        num_group = len(group_values)
        if num_group > 1:
            for i in range(num_group):
                if i == (num_group - 1):
                    break
                for j in range(i+1, num_group):
                    values.append(np.abs(group_values[i] - group_values[j]))

            return np.mean(values)
        else:
            return 0
    else:
        values = 0.0
        num_group = len(group_values)
        if num_group > 1:
            for i in range(num_group):
                if i == (num_group - 1):
                    break
                for j in range(i + 1, num_group):
                    values = np.max([values, np.abs(group_values[i] - group_values[j])])

            return values
        else:
            return 0


def get_obj(group_values, plan):
    Group_values = copy.deepcopy(group_values)
    if plan == 1:
        # calculate the difference
        values = []
        num_group = len(group_values)
        if num_group > 1:
            for i in range(num_group):
                if i == (num_group - 1):
                    break
                for j in range(i+1, num_group):
                    values.append(np.abs(group_values[i] - group_values[j]))

            return 0.5 * (np.mean(values) + np.max(values))

        elif num_group == 1:
            return 1

        else:
            return 0
    else:
        # calculate the ratio
        values = []
        num_group = len(group_values)
        if num_group > 1:
            for i in range(num_group):
                if i == (num_group - 1):
                    break
                for j in range(i + 1, num_group):
                    if group_values[j] == 0 and group_values[i] == 0:
                        values.append(1)
                    elif group_values[j] == 0 and group_values[i] != 0:
                        values.append(0)
                    elif group_values[j] != 0 and group_values[i] == 0:
                        values.append(0)
                    else:
                        values.append(np.min([(group_values[j]/group_values[i]), (group_values[i]/group_values[j])]))
            return 0.5 * (1 - np.mean(values) + 1 - np.min(values))

        elif num_group == 1:
            return 1

        else:
            return 0


def calculate_similar_dist(dist_mat, y):
    dist_mat = dist_mat.reshape(1, -1)
    y = y.reshape(1, -1)

    y_diff = pdist(y.T, 'cityblock')

    Dwork_value = y_diff - dist_mat

    flag = Dwork_value < 0
    Dwork_value[flag] = 0

    Dwork_value = np.mean(Dwork_value)
    return Dwork_value


def calcul_all_fairness(data, logits, truelabel, sensitive_attributions, alpha):
    # The method is from "Unified Approach to Quantifying Algorithmic Unfairness:
    # Measuring Individual & Group Unfairness via Inequality Indices"
    # a few differences: "logits - truelabel + 1" instead on "pred_label - truelabel + 1"
    sum_num = logits.shape[0] * logits.shape[1]

    pred_label = get_label(logits.copy())
    # benefits = pred_label - truelabel + 1   # original version
    benefits = logits - truelabel + 1  # new version in section 3.1
    benefits_mean = np.mean(benefits)
    accuracy = np.sum(pred_label == truelabel) / sum_num

    attribution = data.columns
    Individual_fairness = 0.0
    Group_fairness = 0.0
    group_dict = {}
    Groups = []

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
            g_logits = logits[0, g_idx].reshape([1, g_num])
            g_pred_label = pred_label[0, g_idx].reshape([1, g_num])
            g_true_label = truelabel[0, g_idx].reshape([1, g_num])
            g_benefits = benefits[0, g_idx].reshape([1, g_num])
            g_fairness = calcul_indivi(g_benefits, alpha)
            g_benefits_mean = np.mean(g_benefits)
            g_individual_fairness = (g_num / sum_num) * (np.power(g_benefits_mean / benefits_mean, alpha)) * g_fairness
            g_group_fairness = (g_num / (sum_num * (alpha - 1) * alpha)) * (
                    np.power(g_benefits_mean / benefits_mean, alpha) - 1)
            Individual_fairness += g_individual_fairness
            Group_fairness += g_group_fairness
            g = ea.GroupInfo(group, g_group_fairness, g_individual_fairness, g_true_label, g_pred_label, g_logits,
                             sum_num)
            # Groups.append(g)
        # print('In the', item, ':')
        # print('Accuracy: %.4f, Individual fairness: %.4f, Group fairness: %.4f' % (g.getAccuracy(), g_individual_fairness
        #       , g_group_fairness))
    accuracy_loss = np.mean(np.power(logits - truelabel, 2))
    # BCE
    accuracy_loss = log_loss(truelabel.reshape(-1, 1), np.hstack([1 - logits.reshape(-1, 1), logits.reshape(-1, 1)]))

    Groups_info = ea.GroupsInfo(Groups)
    return accuracy, accuracy_loss, Individual_fairness, Group_fairness, Groups_info


def calcul_all_fairness_new(data, logits, truelabel, sensitive_attributions, alpha):
    # The method is from "Unified Approach to Quantifying Algorithmic Unfairness:
    # Measuring Individual & Group Unfairness via Inequality Indices"
    # a few differences: "logits - truelabel + 1" instead on "pred_label - truelabel + 1"

    logits = logits.astype(np.float64)
    truelabel = truelabel.astype(np.float64)

    logits = logits.reshape(1, -1)
    truelabel = truelabel.reshape(1, -1)

    pred_label = get_label(logits.copy())
    # benefits = pred_label - truelabel + 1   # original version
    benefits = logits - truelabel + 1  # new version in section 3.1
    benefits_group = copy.deepcopy(benefits)

    attribution = data.columns

    group_dict = {}
    Groups = []

    for sens in sensitive_attributions:
        temp = []
        for attr in attribution:
            temp1 = sens + '_'
            if temp1 in attr:
                temp.append(attr)
        group_dict.update({sens: temp})

    group_attr = []
    check_gmuns = []
    for sens in sensitive_attributions:
        group_attr.append(group_dict[sens])
    for item in product(*eval(str(group_attr))):
        group = item
        flag = np.ones([1, truelabel.shape[1]]) == np.ones([1, truelabel.shape[1]])
        for g in group:
            flag = flag & data[g]
        g_num = np.sum(flag)
        if g_num != 0:
            check_gmuns.append(g_num)
            g_idx = np.array(np.where(flag)).reshape([1, -1])[0]
            benefits_group[0, g_idx] = np.mean(benefits[0, g_idx])

    # accuracy
    accuracy = np.mean(pred_label == truelabel)

    # MSE loss
    MSE_loss = np.mean(np.power(logits - truelabel, 2))

    # BCE loss
    isnan_values = np.sum(np.isnan(logits))
    if isnan_values == 0:
        BCE_loss = log_loss(truelabel.reshape(-1, 1), np.hstack([1 - logits.reshape(-1, 1), logits.reshape(-1, 1)]))
    else:
        logits_temp = copy.deepcopy(logits)
        isnan_idx = np.array(np.where(np.isnan(logits[0]))).reshape([1, -1])[0]
        logits_temp[0, isnan_idx] = 0.5
        BCE_loss = log_loss(truelabel.reshape(-1, 1), np.hstack([1 - logits_temp.reshape(-1, 1), logits_temp.reshape(-1, 1)]))


    # Individual unfairness = within-group + between-group
    Individual_fairness = generalized_entropy_index(benefits, alpha)

    # Group unfairness = between-group
    Group_fairness = generalized_entropy_index(benefits_group, alpha)

    # Within-group
    Within_g_fairness = Individual_fairness - Group_fairness

    Groups_info = {}

    Groups_info.update({"Individual_fairness": Individual_fairness})
    Groups_info.update({"BCE_loss": BCE_loss})
    Groups_info.update({"Group_fairness": Group_fairness})
    Groups_info.update({"Accuracy": accuracy})
    return accuracy, BCE_loss, Individual_fairness, Group_fairness, Groups_info


def calcul_all_fairness_new2(data, data_norm, logits, truelabel, sensitive_attributions, alpha, obj_is_logits, dist_mat):
    # The method is from "Unified Approach to Quantifying Algorithmic Unfairness:
    # Measuring Individual & Group Unfairness via Inequality Indices"
    # a few differences: "logits - truelabel + 1" instead on "pred_label - truelabel + 1"
    # # add other fairness metrics

    logits = logits.astype(np.float64)
    truelabel = truelabel.astype(np.float64)

    logits = logits.reshape(1, -1)
    truelabel = truelabel.reshape(1, -1)

    pred_label = get_label(logits.copy())
    # benefits = pred_label - truelabel + 1   # original version
    benefits = logits - truelabel + 1  # new version in section 3.1
    benefits_group = copy.deepcopy(benefits)

    attribution = data.columns

    group_dict = {}
    Groups = []

    FPRs_logit = []
    FNRs_logit = []
    DPs_logit = []
    PPs_logit = []
    FPRs = []
    FNRs = []
    DPs = []
    PPs = []

    for sens in sensitive_attributions:
        temp = []
        for attr in attribution:
            temp1 = sens + '_'
            if temp1 in attr:
                temp.append(attr)
        group_dict.update({sens: temp})

    group_attr = []
    check_gmuns = []
    for sens in sensitive_attributions:
        group_attr.append(group_dict[sens])
    for item in product(*eval(str(group_attr))):
        group = item
        flag = np.ones([1, truelabel.shape[1]]) == np.ones([1, truelabel.shape[1]])
        for g in group:
            flag = flag & data[g]
        g_num = np.sum(flag)
        if g_num != 0:
            check_gmuns.append(g_num)
            g_idx = np.array(np.where(flag)).reshape([1, -1])[0]
            benefits_group[0, g_idx] = np.mean(benefits[0, g_idx])
            g_logits = logits[0, g_idx]
            g_truelabel = truelabel[0, g_idx]
            g_predlabel = pred_label[0, g_idx]
            g_predlabel_flag = np.where(g_predlabel)
            plan = 2
            if plan == 1:
                # original paper : FNNC: Achieving Fairness through Neural Networks
                # calculate based on the logits
                FPR_logit = np.sum(g_logits * (1 - g_truelabel)) / g_num
                FNR_logit = np.sum((1 - g_logits) * g_truelabel) / g_num
                DP_logit = np.sum(g_logits) / g_num
                PP_logit = np.sum(g_logits * g_truelabel) / g_num

                # calculate based on the predictive labels
                FPR = np.sum(g_predlabel * (1 - g_truelabel)) / g_num
                FNR = np.sum((1 - g_predlabel) * g_truelabel) / g_num
                DP = np.sum(g_predlabel) / g_num
                PP = np.sum(g_predlabel * g_truelabel) / g_num
            else:
                # modified "FNNC: Achieving Fairness through Neural Networks"
                # calculate based on the logits
                if np.sum(1 - g_truelabel) > 0:
                    FPR_logit = np.sum(g_logits * (1 - g_truelabel)) / np.sum(1 - g_truelabel)
                    FPRs_logit.append(FPR_logit)
                if np.sum(g_truelabel) > 0:
                    FNR_logit = np.sum((1 - g_logits) * g_truelabel) / np.sum(g_truelabel)
                    FNRs_logit.append(FNR_logit)
                DP_logit = np.sum(g_logits) / g_num
                DPs_logit.append(DP_logit)
                if np.sum(g_predlabel) > 0:
                    PP_logit = np.sum(g_logits[g_predlabel_flag] * g_truelabel[g_predlabel_flag]) / np.sum(g_predlabel)
                    PPs_logit.append(PP_logit)

                # calculate based on the predictive labels
                if np.sum(1 - g_truelabel) > 0:
                    FPR = np.sum(g_predlabel * (1 - g_truelabel)) / np.sum(1 - g_truelabel)
                    FPRs.append(FPR)
                if np.sum(g_truelabel) > 0:
                    FNR = np.sum((1 - g_predlabel) * g_truelabel) / np.sum(g_truelabel)
                    FNRs.append(FNR)
                DP = np.sum(g_predlabel) / g_num
                DPs.append(DP)
                if np.sum(g_predlabel) > 0:
                    PP = np.sum(g_predlabel[g_predlabel_flag] * g_truelabel[g_predlabel_flag]) / np.sum(g_predlabel)
                    PPs.append(PP)
    # accuracy
    accuracy = np.mean(pred_label == truelabel)

    # MSE loss
    MSE_loss = np.mean(np.power(logits - truelabel, 2))

    # BCE loss
    BCE_loss = log_loss(truelabel.reshape(-1, 1), np.hstack([1 - logits.reshape(-1, 1), logits.reshape(-1, 1)]))

    # Individual unfairness = within-group + between-group
    Individual_fairness = generalized_entropy_index(benefits, alpha)

    # Group unfairness = between-group
    Group_fairness = generalized_entropy_index(benefits_group, alpha)

    # Within-group
    Within_g_fairness = Individual_fairness - Group_fairness

    # Demographic Parity
    DP_value = get_average(DPs_logit, 2)
    DPs_true = get_average(DPs, 2)

    # FPR
    FPR_value = get_average(FPRs_logit, 2)
    FPRs_true = get_average(FPRs, 2)

    # FNR
    FNR_value = get_average(FNRs_logit, 2)
    FNRs_true = get_average(FNRs, 2)

    # Predictive parity
    PP_value = get_average(PPs_logit, 2)
    PPs_true = get_average(PPs, 2)

    # all the true metric
    more_vals = np.array([DPs_true, FPRs_true, FNRs_true, PPs_true])

    # Average violation of Dwork et al.'s pairwise constraints
    Dwork_value = calculate_similar_dist(dist_mat, pred_label)

    if obj_is_logits == 1:
        Groups_info = {"accuracy": accuracy, "MSE_loss": MSE_loss, "BCE_loss": BCE_loss, "Individual_fairness": Individual_fairness,
                       "Group_fairness": Group_fairness, "Demographic_parity": DP_value, "FPR": FPR_value,
                       "FNR": FNR_value, "Predictive_parity": PP_value,
                       "DPs_logit": DPs_logit, "FPRs_logit": FPRs_logit, "FNRs_logit": FNRs_logit, "PPs_logit": PPs_logit,
                       "DPs": DPs, "FPRs": FPRs, "FNRs": FNRs, "PPs": PPs,
                       "DPs_true": DPs_true, "FPRs_true": FPRs_true, "FNRs_true": FNRs_true, "PPs_true": PPs_true,
                       'addition_num': 4, "more_vals": more_vals, "Dwork_value": Dwork_value}
    else:
        Groups_info = {"accuracy": accuracy, "MSE_loss": MSE_loss, "BCE_loss": BCE_loss,
                       "Individual_fairness": Individual_fairness,
                       "Group_fairness": Group_fairness, "Demographic_parity": DPs_true, "FPR": FPRs_true,
                       "FNR": FNRs_true, "Predictive_parity": PPs_true,
                       "DPs_logit": DPs_logit, "FPRs_logit": FPRs_logit, "FNRs_logit": FNRs_logit,
                       "PPs_logit": PPs_logit,
                       "DPs": DPs, "FPRs": FPRs, "FNRs": FNRs, "PPs": PPs,
                       "DPs_true": DPs_true, "FPRs_true": FPRs_true, "FNRs_true": FNRs_true, "PPs_true": PPs_true,
                       'addition_num': 4, "more_vals": more_vals, "Dwork_value": Dwork_value}

    return Groups_info


def calcul_all_fairness_new3(data, data_norm, logits, truelabel, sensitive_attributions, alpha, obj_is_logits, dist_mat, obj_names):
    # The method is from "Unified Approach to Quantifying Algorithmic Unfairness:
    # Measuring Individual & Group Unfairness via Inequality Indices"
    # a few differences: "logits - truelabel + 1" instead on "pred_label - truelabel + 1"
    # # add other fairness metrics

    logits = logits.reshape(1, -1).astype(np.float64)
    truelabel = truelabel.astype(np.float64).reshape(1, -1)
    pred_label = get_label(logits.copy())

    benefits = logits - truelabel + 1  # new version in section 3.1
    benefits_group = copy.deepcopy(benefits)

    total_num = logits.shape[1]

    attribution = data.columns
    group_attr = []
    check_gmuns = []
    group_dict = {}

    Disparate_impact = []
    Calibration_Neg = []
    Predictive_parity = []
    Discovery_ratio = []
    Discovery_diff = []
    Predictive_equality = []
    FPR_ratio = []
    Equal_opportunity = []
    Equalized_odds1 = []
    Equalized_odds2 = []
    Average_odd_diff = []
    Conditional_use_accuracy_equality1 = []
    Conditional_use_accuracy_equality2 = []
    Overall_accuracy = []
    Error_ratio = []
    Error_diff = []
    Statistical_parity = []
    FOR_ratio = []
    FOR_diff = []
    FPR_ratio = []
    FNR_ratio = []
    FNR_diff = []

    for sens in sensitive_attributions:
        temp = []
        for attr in attribution:
            temp1 = sens + '_'
            if temp1 in attr:
                temp.append(attr)
        group_dict.update({sens: temp})

    for sens in sensitive_attributions:
        group_attr.append(group_dict[sens])
    for item in product(*eval(str(group_attr))):
        group = item
        flag = np.ones([1, truelabel.shape[1]]) == np.ones([1, truelabel.shape[1]])
        for g in group:
            flag = flag & data[g]
        g_num = np.sum(flag)
        if g_num != 0:
            check_gmuns.append(g_num)
            g_idx = np.array(np.where(flag)).reshape([1, -1])[0]
            benefits_group[0, g_idx] = np.mean(benefits[0, g_idx])
            # g_logits = logits[0, g_idx]
            g_truelabel = truelabel[0, g_idx]
            g_predlabel = pred_label[0, g_idx]

            # P(d=1 | g)
            # Disparate Impact  or  Statistical Parity
            if "Disparate_impact" in obj_names or "Statistical_parity" in obj_names:
                Disparate_impact.append(np.sum(g_predlabel) / g_num)
                Statistical_parity = Disparate_impact

            # P(y=d | g)
            # Overall accuracy
            if "Overall_accuracy" in obj_names:
                Overall_accuracy.append(np.sum(g_truelabel * g_predlabel) / g_num)

            # P(y != d, g)
            # Error ratio   or   Error diff
            if "Error_ratio" in obj_names or "Error_diff" in obj_names:
                Error_ratio.append(np.sum(g_truelabel != g_predlabel) / total_num)
                Error_diff = Error_ratio

            # P(y=1 | d=1, g)
            # Predictive parity
            if "Predictive_parity" in obj_names:
                if np.sum(g_predlabel) > 0:
                    Predictive_parity.append(np.sum(g_truelabel * g_predlabel) / np.sum(g_predlabel))

            # P(y=0 | d=1, g)
            # Discovery ratio  or   Discovery diff
            if "Discovery_ratio" in obj_names or "Discovery_diff" in obj_names :
                if np.sum(g_predlabel) > 0:
                    Discovery_ratio.append((np.sum((1-g_truelabel) * g_predlabel) / np.sum(g_predlabel)))
                    Discovery_diff = Discovery_ratio

            # P(y=1 | d=0, g)
            # Calibration-   or   FOR ratio    or   FOR diff
            if "Calibration_neg" in obj_names or "FOR_ratio" in obj_names or "FOR_diff" in obj_names:
                if np.sum(1-g_predlabel) > 0:
                    Calibration_Neg.append(np.sum(g_truelabel * (1-g_predlabel)) / np.sum(1-g_predlabel))
                    FOR_ratio = Calibration_Neg
                    FOR_diff = Calibration_Neg

            # P(d=1 | y=0, g)
            # Predictive equality   or   FPR ratio
            if "Predictive_equality" in obj_names or "FPR_ratio" in obj_names:
                if np.sum(1-g_truelabel) > 0:
                    Predictive_equality.append(np.sum(g_predlabel * (1-g_truelabel)) / np.sum(1-g_truelabel))
                    FPR_ratio = Predictive_equality

            # P(d=1 | y=1, g)
            # Equal opportunity
            if "Equal_opportunity" in obj_names:
                if np.sum(g_truelabel) > 0:
                    Equal_opportunity.append(np.sum(g_predlabel * g_truelabel) / np.sum(g_truelabel))

            # P(d=0 | y=1, g)
            # FNR ratio    or    FNR diff
            if "FNR_ratio" in obj_names or "FNR_diff" in obj_names:
                if np.sum(g_truelabel) > 0:
                    FNR_ratio.append(np.sum((1-g_predlabel) * g_truelabel) / np.sum(g_truelabel))
                    FNR_diff = FNR_ratio

            # P(d=1 | y=0, g) and P(d=1 | y=1, g)
            # Equalized odds
            if "Equalized_odds" in obj_names:
                if np.sum(g_truelabel) > 0:
                    Equalized_odds1.append(np.sum(g_predlabel * g_truelabel) / np.sum(g_truelabel))
                if np.sum(1-g_truelabel) > 0:
                    Equalized_odds2.append(np.sum(g_predlabel * (1-g_truelabel)) / np.sum(1-g_truelabel))

            # Conditional use accuracy equality
            if "Conditional_use_accuracy_equality" in obj_names:
                if np.sum(g_predlabel) > 0:
                    Conditional_use_accuracy_equality1.append(np.sum(g_truelabel * g_predlabel) / np.sum(g_predlabel))
                if np.sum(1-g_predlabel) > 0:
                    Conditional_use_accuracy_equality2.append(np.sum((1-g_truelabel) * (1-g_predlabel)) / np.sum(1-g_predlabel))

            # P(d=1 | y=0, g) + P(d=1 | y=1, g)
            # Average odd difference
            if "Average_odd_diff" in obj_names:
                if np.sum(g_truelabel) > 0 and np.sum(1 - g_truelabel) > 0:
                    Average_odd_diff.append((np.sum(g_predlabel * g_truelabel) / np.sum(g_truelabel)) + (
                                np.sum(g_predlabel * (1 - g_truelabel)) / np.sum(1 - g_truelabel)))

    Groups_info = {}
    if "Accuracy" in obj_names:
        Groups_info.update({"Accuracy": 1-np.mean(pred_label == truelabel)})

    if "Misclassification" in obj_names:
        Groups_info.update({"Misclassification": 1-np.mean(pred_label == truelabel)})

    # MSE loss
    if "MSE_loss" in obj_names:
        MSE_loss = np.mean(np.power(logits - truelabel, 2))
        Groups_info.update({"MSE_loss": MSE_loss})

    # BCE loss
    if "BCE_loss" in obj_names:
        BCE_loss = log_loss(truelabel.reshape(-1, 1), np.hstack([1 - logits.reshape(-1, 1), logits.reshape(-1, 1)]))
        Groups_info.update({"BCE_loss": BCE_loss})

    # Individual unfairness = within-group + between-group
    if "Individual_fairness" in obj_names:
        Individual_fairness_val = generalized_entropy_index(benefits, alpha)
        Groups_info.update({"Individual_fairness": Individual_fairness_val})

    # Group unfairness = between-group
    if "Group_fairness" in obj_names:
        Group_fairness_val = generalized_entropy_index(benefits_group, alpha)
        Groups_info.update({"Group_fairness": Group_fairness_val})

    # Within-group
    # Within_g_fairness = Individual_fairness - Group_fairness

    # Disparate impact
    if "Disparate_impact" in obj_names:
        Disparate_impact_val = get_obj(Disparate_impact, 2)
        Groups_info.update({"Disparate_impact": Disparate_impact_val})

    # Statistical parity
    if "Statistical_parity" in obj_names:
        Statistical_parity_val = get_obj(Statistical_parity, 1)
        Groups_info.update({"Statistical_parity": Statistical_parity_val})

    # Overall accuracy
    if "Overall_accuracy" in obj_names:
        Overall_accuracy_val = get_obj(Overall_accuracy, 1)
        Groups_info.update({"Overall_accuracy": Overall_accuracy_val})

    # Error ratio
    if "Error_ratio" in obj_names:
        Error_ratio_val = get_obj(Error_ratio, 2)
        Groups_info.update({"Error_ratio": Error_ratio_val})

    # Error diff
    if "Error_diff" in obj_names:
        Error_diff_val = get_obj(Error_diff, 1)
        Groups_info.update({"Error_diff": Error_diff_val})

    # Predictive parity
    if "Predictive_parity" in obj_names:
        Predictive_parity_val = get_obj(Predictive_parity, 1)
        Groups_info.update({"Predictive_parity": Predictive_parity_val})

    # Discovery ratio
    if "Discovery_ratio" in obj_names:
        Discovery_ratio_val = get_obj(Discovery_ratio, 2)
        Groups_info.update({"Discovery_ratio": Discovery_ratio_val})

    # Discovery diff
    if "Discovery_diff" in obj_names:
        Discovery_diff_val = get_obj(Discovery_diff, 1)
        Groups_info.update({"Discovery_diff": Discovery_diff_val})

    # Calibration Neg
    if "Calibration_neg" in obj_names:
        Calibration_Neg_val = get_obj(Calibration_Neg, 1)
        Groups_info.update({"Calibration_neg": Calibration_Neg_val})

    # FOR ratio
    if "FOR_ratio" in obj_names:
        FOR_ratio_val = get_obj(FOR_ratio, 2)
        Groups_info.update({"FOR_ratio": FOR_ratio_val})

    # FOR diff
    if "FOR_diff" in obj_names:
        FOR_diff_val = get_obj(FOR_diff, 1)
        Groups_info.update({"FOR_diff": FOR_diff_val})

    # Predictive equality
    if "Predictive_equality" in obj_names:
        Predictive_equality_val = get_obj(Predictive_equality, 1)
        Groups_info.update({"Predictive_equality": Predictive_equality_val})

    # FPR ratio
    if "FPR_ratio" in obj_names:
        FPR_ratio_val = get_obj(FPR_ratio, 2)
        Groups_info.update({"FPR_ratio": FPR_ratio_val})

    # Equal opportunity
    if "Equal_opportunity" in obj_names:
        Equal_opportunity_val = get_obj(Equal_opportunity, 1)
        Groups_info.update({"Equal_opportunity": Equal_opportunity_val})

    # FNR ratio
    if "FNR_ratio" in obj_names:
        FNR_ratio_val = get_obj(FNR_ratio, 2)
        Groups_info.update({"FNR_ratio": FNR_ratio_val})

    # FNR diff
    if "FNR_diff" in obj_names:
        FNR_diff_val = get_obj(FNR_diff, 1)
        Groups_info.update({"FNR_diff": FNR_diff_val})

    # Equalized odds
    if "Equalized_odds" in obj_names:
        Equalized_odds_val = 0.5 * (get_obj(Equalized_odds1, 1) + get_obj(Equalized_odds2, 1))
        Groups_info.update({"Equalized_odds": Equalized_odds_val})

    # Conditional use accuracy equality
    if "Conditional_use_accuracy_equality" in obj_names:
        Conditional_use_accuracy_equality_val = 0.5 * (
                    get_obj(Conditional_use_accuracy_equality1, 1) + get_obj(Conditional_use_accuracy_equality2, 1))
        Groups_info.update({"Conditional_use_accuracy_equality": Conditional_use_accuracy_equality_val})

    # Average odd difference
    if "Average_odd_diff" in obj_names:
        Average_odd_diff_val = 0.5 * get_obj(Average_odd_diff, 1)
        Groups_info.update({"Average_odd_diff": Average_odd_diff_val})

    # Average violation of Dwork et al.'s pairwise constraints
    if "Fairness_through_awareness" in obj_names:
        Dwork_value = calculate_similar_dist(dist_mat, pred_label)
        Groups_info.update({"Fairness_through_awareness": Dwork_value})

    return Groups_info


def generalized_entropy_index(b, alpha):
    # https://github.com/Trusted-AI/AIF360/blob/master/aif360/metrics/classification_metric.py#L664
    # pred_label = get_label(b.copy())
    # benefits = pred_label - truelabel + 1  # original version
    # benefits = logits - truelabel + 1  # new version in section 3.1
    # b = benefits
    if alpha == 1:
        # moving the b inside the log allows for 0 values
        return np.mean(np.log((b / np.mean(b)) ** b) / np.mean(b))
    elif alpha == 0:
        return -np.mean(np.log(b / np.mean(b)) / np.mean(b))
    else:
        return np.mean((b / np.mean(b)) ** alpha - 1) / (alpha * (alpha - 1))


def between_all_groups_generalized_entropy_index(data, logits, truelabel, sensitive_attributions, alpha):
    # https://github.com/Trusted-AI/AIF360/blob/master/aif360/metrics/classification_metric.py#L700
    # The method is from "Unified Approach to Quantifying Algorithmic Unfairness:
    # Measuring Individual & Group Unfairness via Inequality Indices"
    sum_num = logits.shape[0] * logits.shape[1]

    pred_label = get_label(logits.copy())
    # benefits = pred_label - truelabel + 1   # original version
    benefits = logits - truelabel + 1  # new version in section 3.1
    attribution = data.columns
    group_dict = {}
    b = np.zeros(sum_num, dtype=np.float64)

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
            b[g_idx] = np.mean(benefits[0, g_idx].reshape([1, g_num]))

    return calcul_indivi(b.reshape([1, -1]), alpha)


def Cal_objectives(data, data_norm, logits, truelabel, sensitive_attributions, alpha, group=None, plan=6, dis=None, obj_is_logits=1, dist_mat=None, obj_names=None):
    sum_num = logits.shape[0] * logits.shape[1]
    logits = np.array(logits).reshape([1, sum_num])
    truelabel = np.array(truelabel).reshape([1, sum_num])
    pred_label = get_label(logits.copy())
    # print(generalized_entropy_index(logits, truelabel, alpha))
    # print( between_all_groups_generalized_entropy_index(data, logits, truelabel, sensitive_attributions, alpha))
    if plan == 1:
        # accuracy, accuracy_loss, Individual_fairness, Group_fairness, Groups_info = calcul_all_fairness(data,
        #                                                                                                 logits,
        #                                                                                                 truelabel,
        #                                                                                                 sensitive_attributions,
        #                                                                                                 alpha)

        accuracy, accuracy_loss, Individual_fairness, Group_fairness, Groups_info = calcul_all_fairness_new(data,
                                                                                                            logits,
                                                                                                            truelabel,
                                                                                                            sensitive_attributions,
                                                                                                            alpha)

        AUC_vals = []
    elif plan == 2:
        accuracy, accuracy_loss, Individual_fairness, Group_fairness, Groups_info, metr_info = DP_EO_DI(
            data,
            logits,
            truelabel,
            sensitive_attributions)
        AUC_vals = metr_info

    elif plan == 3:
        accuracy, accuracy_loss, Individual_fairness, Group_fairness, Groups_info, metr_info = DemographicParity_EqualizedOdds4(
            data,
            logits,
            truelabel,
            sensitive_attributions)
        AUC_vals = metr_info
    elif plan == 4:
        accuracy, accuracy_loss, Individual_fairness, Group_fairness, Groups_info, metr_info = DP_EO_DI2(
            data,
            dis,
            logits,
            truelabel,
            sensitive_attributions)
        AUC_vals = metr_info

    elif plan == 5:
        Groups_info = calcul_all_fairness_new2(data,
                                               data_norm,
                                                logits,
                                                truelabel,
                                                sensitive_attributions,
                                                alpha, obj_is_logits, dist_mat)
        AUC_vals = []

    elif plan == 6:
        Groups_info = calcul_all_fairness_new3(data,
                                               data_norm,
                                                logits,
                                                truelabel,
                                                sensitive_attributions,
                                                alpha, obj_is_logits, dist_mat, obj_names)
        AUC_vals = []

    else:
        # accuracy, MSE, vals_vars, 1-vals_min, vals
        accuracy, accuracy_loss, AUC_vals_vars, AUC_neg_vals_min, AUC_vals = Cal_AUC(logits, truelabel, group)
        Individual_fairness = AUC_vals_vars
        Group_fairness = AUC_neg_vals_min
        Groups_info = []

    return Groups_info
