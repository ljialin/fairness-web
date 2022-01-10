import numpy as np


# 可以传入特定群组的标签或者整体的标签
# label必须经过01处理，1为正样本，0为负样本
def get_confus_vals(pred_label, true_label):
    pred_label = np.array(pred_label)
    true_label = np.array(true_label)

    res = {
        'TP': np.sum((pred_label * true_label)),
        'FP': np.sum((pred_label * (1-true_label))),
        'FN': np.sum(((1-pred_label) * true_label)),
        'TN': np.sum(((1-pred_label) * (1-true_label))),
    }

    return res


def compute_metrics(TP, FP, FN, TN):
    total = TP + FP + FN + TN + 1e-8
    res = {
        'ACC': (TP + TN) / total,
        'PLR': (TP + FP) / total,
        'ERR': (FP + FN) / total,
        'PPV': TP / (TP + FP + 1e-8),
        'NPV': TN / (TN + FN + 1e-8),
        'FPR': FP / (FP + TN + 1e-8),
        'FNR': FN / (TP + FN + 1e-8),
        'FDR': 1 - TP / (TP + FP + 1e-8),  # 1-PPV
        'FOR': 1 - TN / (TN + FN + 1e-8),  # 1-NPV
        'TNR': 1 - FP / (FP + TN + 1e-8),  # 1-FPR
        'TPR': 1 - FP / (FP + TN + 1e-8),  # 1-FNR
    }
    return res


def get_metrics(pred_label, true_label):
    return compute_metrics(**get_confus_vals(pred_label, true_label))


def get_obj(group_values, plan):
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
            return 0.5 * (1 - np.mean(values) + 1 - np.max(values))

        elif num_group == 1:
            return 1

        else:
            return 0


# metric_vals = {"male":{"metric1": 0.5, "metric2": 0.3}, "female":{}}
def cal_fairness_score(metric_vals: dict):
    group_name = list(metric_vals.keys())
    metrics = list(metric_vals[group_name[0]].keys())
    scores = {}
    for metric in metrics:
        grp_value = []
        for grp in group_name:
            grp_value.append(metric_vals[grp][metric])
        scores[metric] = 1 - get_obj(grp_value, 1)
    return scores
