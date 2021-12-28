import numpy

from geatpy.moeas4NN.metrics.Accuracy import Accuracy
from geatpy.moeas4NN.metrics.BCR import BCR
from geatpy.moeas4NN.metrics.CalibrationNeg import CalibrationNeg
from geatpy.moeas4NN.metrics.CalibrationPos import CalibrationPos
from geatpy.moeas4NN.metrics.CV import CV
from geatpy.moeas4NN.metrics.DIAvgAll import DIAvgAll
from geatpy.moeas4NN.metrics.DIBinary import DIBinary
from geatpy.moeas4NN.metrics.EqOppo_fn_diff import EqOppo_fn_diff
from geatpy.moeas4NN.metrics.EqOppo_fn_ratio import EqOppo_fn_ratio
from geatpy.moeas4NN.metrics.EqOppo_fp_diff import EqOppo_fp_diff
from geatpy.moeas4NN.metrics.EqOppo_fp_ratio import EqOppo_fp_ratio
from geatpy.moeas4NN.metrics.FNR import FNR
from geatpy.moeas4NN.metrics.FPR import FPR
from geatpy.moeas4NN.metrics.MCC import MCC
from geatpy.moeas4NN.metrics.SensitiveMetric import SensitiveMetric
from geatpy.moeas4NN.metrics.TNR import TNR
from geatpy.moeas4NN.metrics.TPR import TPR
from geatpy.moeas4NN.metrics.AUC import AUC

METRICS = [
            Accuracy(), TPR(), TNR(), BCR(), MCC(),  # accuracy metrics
           DIBinary(), DIAvgAll(), CV(),                  # fairness metrics
            # SensitiveMetric(TPR), SensitiveMetric(TNR),
           # AUC()
           # SensitiveMetric(Accuracy), SensitiveMetric(TPR), SensitiveMetric(TNR),
           # SensitiveMetric(FPR), SensitiveMetric(FNR),
           # SensitiveMetric(CalibrationPos), SensitiveMetric(CalibrationNeg)
           ]


def get_metrics(dataset, sensitive_dict, tag):
    """
    Takes a dataset object and a dictionary mapping sensitive attributes to a list of the sensitive
    values seen in the data.  Returns an expanded list of metrics based on the base METRICS.
    """
    metrics = []
    for metric in METRICS:
        metrics += metric.expand_per_dataset(dataset, sensitive_dict, tag)
    return metrics


def add_metric(metric):
    METRICS.append(metric)
