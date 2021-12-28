from geatpy.moeas4NN.metrics.Metric import Metric
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np

class AUC(Metric):
    def __init__(self):
        Metric.__init__(self)
        self.name = 'AUC'

    def calc(self, actual, predicted, dict_of_sensitive_lists, single_sensitive_name,
             unprotected_vals, positive_pred, problem, logits):
        sens_flag = problem.sens_flag
        sens_flag_name = problem.sens_flag_name
        res = {}
        true_y = np.array(problem.test_y)
        logits = np.array(logits)
        for class_name in sens_flag_name:
            idx = sens_flag[class_name]
            if np.all(true_y[idx[0]] == true_y[idx[0]][0]):
                # Only one class present in y_true, ROC AUC score is not defined in that case.
                val = -1
            else:
                val = roc_auc_score(true_y[idx[0]], logits[idx[0]])
            nam = class_name+"(AUC)"
            res.update({nam: val})

        return res
