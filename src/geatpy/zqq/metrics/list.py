import numpy

from geatpy.zqq.metrics.Accuracy import Accuracy
from geatpy.zqq.metrics.BCR import BCR
from geatpy.zqq.metrics.CalibrationNeg import CalibrationNeg
from geatpy.zqq.metrics.CalibrationPos import CalibrationPos
from geatpy.zqq.metrics.CV import CV
from geatpy.zqq.metrics.DIAvgAll import DIAvgAll
from geatpy.zqq.metrics.DIBinary import DIBinary
from geatpy.zqq.metrics.EqOppo_fn_diff import EqOppo_fn_diff
from geatpy.zqq.metrics.EqOppo_fn_ratio import EqOppo_fn_ratio
from geatpy.zqq.metrics.EqOppo_fp_diff import EqOppo_fp_diff
from geatpy.zqq.metrics.EqOppo_fp_ratio import EqOppo_fp_ratio
from geatpy.zqq.metrics.FNR import FNR
from geatpy.zqq.metrics.FPR import FPR
from geatpy.zqq.metrics.MCC import MCC
from geatpy.zqq.metrics.SensitiveMetric import SensitiveMetric
from geatpy.zqq.metrics.TNR import TNR
from geatpy.zqq.metrics.TPR import TPR
from geatpy.zqq.metrics.AUC import AUC

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
