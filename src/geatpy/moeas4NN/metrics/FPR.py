from geatpy.moeas4NN.metrics.Metric import Metric
from geatpy.moeas4NN.metrics.TNR import TNR

class FPR(Metric):
    def __init__(self):
        Metric.__init__(self)
        self.name = 'FPR'

    def calc(self, actual, predicted, dict_of_sensitive_lists, single_sensitive_name,
             unprotected_vals, positive_pred, problem, logits):
        tnr = TNR()
        tnr_val = tnr.calc(actual, predicted, dict_of_sensitive_lists, single_sensitive_name,
                           unprotected_vals, positive_pred, problem, logits)
        return 1 - tnr_val
