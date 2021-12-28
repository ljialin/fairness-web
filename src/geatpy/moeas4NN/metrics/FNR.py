from geatpy.moeas4NN.metrics.Metric import Metric
from geatpy.moeas4NN.metrics.TPR import TPR

class FNR(Metric):
    def __init__(self):
        Metric.__init__(self)
        self.name = 'FNR'

    def calc(self, actual, predicted, dict_of_sensitive_lists, single_sensitive_name,
             unprotected_vals, positive_pred, problem, logits):
        tpr = TPR()
        tpr_val = tpr.calc(actual, predicted, dict_of_sensitive_lists, single_sensitive_name,
                           unprotected_vals, positive_pred, problem, logits)
        return 1 - tpr_val
