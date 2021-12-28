import geatpy as ea
import numpy as np
from itertools import product


def detect_allgroups(data, dataset):
    # 记录class：flag，如 [white:[1,0,1,1], balck:[1,1,..], female_white:[1,0,1,1], female_balck:[1,1,..].......]
    attribution = data.columns
    sensitive_attributions = dataset.get_sensitive_attributes()
    group_dict = {}
    sens_flag_name = []
    num = data.shape[0]
    sens_flag = {}
    # sens_flag_name = []
    for sens in sensitive_attributions:
        temp = []
        for attr in attribution:
            temp1 = sens + '_'
            if temp1 in attr:
                temp.append(attr)
                sens_flag.update({attr: np.array(np.where(data[attr])).reshape(1, -1)})
                sens_flag_name.append(attr)

        group_dict.update({sens: temp})

    group_attr = []
    for sens in sensitive_attributions:
        group_attr.append(group_dict[sens])

    for item in product(*eval(str(group_attr))):
        group = item
        flag = np.ones([1, num]) == np.ones([1, num])
        for g in group:
            flag = flag & data[g]
        g_num = np.sum(flag)
        if g_num != 0:
            name = ("+").join(str(x) for x in group)
            g_idx = np.array(np.where(flag)).reshape([1, g_num])
            sens_flag.update({name: g_idx})
            sens_flag_name.append(name)
    return sens_flag, sens_flag_name


def detect_groups(data, dataset):
    # 记录class：flag，如 [white:[1,0,1,1], balck:[1,1,..], female_white:[1,0,1,1], female_balck:[1,1,..].......]
    attribution = data.columns
    sensitive_attributions = dataset.get_sensitive_attributes()
    group_dict = {}
    sens_flag_name = []
    num = data.shape[0]
    sens_flag = {}
    # sens_flag_name = []
    for sens in sensitive_attributions:
        temp = []
        for attr in attribution:
            temp1 = sens+'_'
            if temp1 in attr:
                temp.append(attr)
                # sens_flag.update({attr: np.array(np.where(data[attr])).reshape(1, -1)})
                # sens_flag_name.append(attr)

        group_dict.update({sens: temp})

    group_attr = []
    for sens in sensitive_attributions:
        group_attr.append(group_dict[sens])

    for item in product(*eval(str(group_attr))):
        group = item
        flag = np.ones([1, num]) == np.ones([1, num])
        for g in group:
            flag = flag & data[g]
        g_num = np.sum(flag)
        if g_num != 0:
            name = "+".join(str(x) for x in group)
            g_idx = np.array(np.where(flag)).reshape([1, g_num])
            sens_flag.update({name: g_idx})
            sens_flag_name.append(name)
    return sens_flag, sens_flag_name
