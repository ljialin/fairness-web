import random
import math
##############################################################################
import matplotlib.pyplot as plt  # for plotting stuff
import numpy as np
import pandas as pd
import utils as ut
from scipy.stats import multivariate_normal  # generating synthetic data

from geatpy.zqq.data.objects.Data import Data

SEED = 1122334455
random.seed(SEED)

def gen_gaussian_diff_size(mean_in, cov_in, z_val, class_label, n):
    """
    mean_in: mean of the gaussian cluster
    cov_in: covariance matrix
    z_val: sensitive feature value
    class_label: +1 or -1
    n: number of points
    """

    nv = multivariate_normal(mean=mean_in, cov=cov_in)
    X = nv.rvs(n)
    y = np.ones(n, dtype=float) * class_label
    z = np.ones(n, dtype=float) * z_val  # all the points in this cluster get this value of the sensitive attribute

    return nv, X, y, z


def generate_synthetic_data(data_type=1, plot_data=False, n_samples=1000):
    """
        Code for generating the synthetic data.
        We will have two non-sensitive features and one sensitive feature.
        Non sensitive features will be drawn from a 2D gaussian distribution.
        Sensitive feature specifies the demographic group of the data point and can take values 0 and 1.

        The code will generate data such that a classifier optimizing for accuracy will lead to disparate misclassification rates for the two demographic groups.
        You can generate different data configurations using different values for the "data_type" parameter.
    """

    # generate dataset: x1, x2, x3, z, y
    # x1 = uniform[0,1]
    # x2 = uniform[0,1]
    # y = 1, x1 + x2 > 1
    # z = 1, (y=1 & rand < p1) | (y=0 & rand < p1), otherwise 0
    # x3 = 1, (z=1 & rand < p2) | (z=0 & rand < p2), otherwise 0

    p1 = 0.8
    p2 = 0.9
    x1 = np.random.uniform(0, 1, n_samples)
    x2 = np.random.uniform(0, 1, n_samples)
    sum_x12 = x1 + x2
    y = np.array([1 if v > 1 else 0 for v in sum_x12])
    z = y.copy()
    for idx in range(y.shape[0]):
        if np.random.uniform() < p1:
            z[idx] = y[idx]
        else:
            z[idx] = 1 - y[idx]
    x3 = z.copy()
    for idx in range(z.shape[0]):
        if np.random.uniform() < p2:
            x3[idx] = z[idx]
        else:
            x3[idx] = 1 - z[idx]

    # make x3 random # x3 = uniform[0,1]
    x3 = np.random.uniform(0, 1, n_samples)
    x3 = np.array([1 if v > 0.5 else 0 for v in x3])

    X = np.hstack((x1.reshape(-1, 1), x2.reshape(-1, 1), x3.reshape(-1, 1)))
    x_control = z

    # shuffle the data
    perm = list(range(len(X)))
    random.shuffle(perm)
    X = X[perm]
    y = y[perm]
    x_control = x_control[perm]

    return X, y, x_control


class GradeScore(Data):

    def __init__(self, n_samples):
        Data.__init__(self)
        self.dataset_name = 'GradeScore_' + str(n_samples)
        self.class_attr = 'decision'
        self.positive_class_val = 1
        self.sensitive_attrs = ['sensitive-attr']
        self.privileged_class_names = [1]
        self.negative_class_val = 0
        self.categorical_features = ['x3']
        self.features_to_keep = ['x1', 'x2', 'x3', 'sensitive-attr', 'decision'] # x1 与 sensitive相关； x2为随机产生
        self.missing_val_indicators = ['?']
        self.n_samples = n_samples

    def load_raw_dataset(self):

        X, y, x_control = generate_synthetic_data(data_type=3, plot_data=False, n_samples=self.n_samples)
        return pd.DataFrame(data={
            "x1": X[:, 0],
            "x2": X[:, 1],
            "x3": X[:, 2],
            "decision": y,
            "sensitive-attr": x_control,
            })


