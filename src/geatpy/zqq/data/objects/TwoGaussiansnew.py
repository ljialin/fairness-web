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

    # generate these many data points per cluster

    if data_type == 1:

        """
        Generate data such that a classifier optimizing for accuracy will have disparate false positive rates as well as disparate false negative rates for both groups.
        """

        mu1, sigma1 = [2, 5], [[5, 1], [1, 5]]  # z=1, +
        mu2, sigma2 = [2, 5], [[5, 1], [1, 5]]  # z=0, +

        mu3, sigma3 = [-1, 0], [[5, 1], [1, 5]]  # z=1, -
        mu4, sigma4 = [-1, 0], [[5, 1], [1, 5]]  # z=0, -

        nv1, X1, y1, z1 = gen_gaussian_diff_size(mean_in=mu1, cov_in=sigma1, z_val=1, class_label=1, n=int(n_samples * 1))  # z=1, +
        nv2, X2, y2, z2 = gen_gaussian_diff_size(mu2, sigma2, 0, +1, int(n_samples * 1))  # z=0, +
        nv3, X3, y3, z3 = gen_gaussian_diff_size(mu3, sigma3, 1, 0, int(n_samples * 1))  # z=1, -
        nv4, X4, y4, z4 = gen_gaussian_diff_size(mu4, sigma4, 0, 0, int(n_samples * 1))  # z=0, -
        X = np.vstack((X1, X2, X3, X4))
        y = np.hstack((y1, y2, y3, y4))
        x_control = np.hstack((z1, z2, z3, z4))
    elif data_type == 2:

        """
        Fairness beyond disparate treatment
        Case I
        """

        mu1, sigma1 = [2, 3], [[5, 1], [1, 5]]  # z=1, +
        mu2, sigma2 = [2, 0], [[5, 1], [1, 5]]  # z=0, +

        mu3, sigma3 = [-1, 0], [[5, 1], [1, 5]]  # z=1, -
        mu4, sigma4 = [-1, -3], [[5, 1], [1, 5]]  # z=0, -

        nv1, X1, y1, z1 = gen_gaussian_diff_size(mean_in=mu1, cov_in=sigma1, z_val=1, class_label=1,
                                                 n=int(n_samples * 1))  # z=1, +
        nv2, X2, y2, z2 = gen_gaussian_diff_size(mu2, sigma2, 0, +1, int(n_samples * 1))  # z=0, +
        nv3, X3, y3, z3 = gen_gaussian_diff_size(mu3, sigma3, 1, 0, int(n_samples * 1))  # z=1, -
        nv4, X4, y4, z4 = gen_gaussian_diff_size(mu4, sigma4, 0, 0, int(n_samples * 1))  # z=0, -
        X = np.vstack((X1, X2, X3, X4))
        y = np.hstack((y1, y2, y3, y4))
        x_control = np.hstack((z1, z2, z3, z4))

    elif data_type == 3:

        """
        Fairness beyond disparate treatment
        Case II
        """

        mu1, sigma1 = [2, 3], [[5, 2], [2, 5]]  # z=1, +
        mu2, sigma2 = [1, 2], [[5, 2], [2, 5]]  # z=0, +

        mu3, sigma3 = [-5, 0], [[5, 1], [1, 5]]  # z=1, -
        mu4, sigma4 = [0, -1], [[7, 1], [1, 7]]  # z=0, -

        nv1, X1, y1, z1 = gen_gaussian_diff_size(mean_in=mu1, cov_in=sigma1, z_val=1, class_label=1,
                                                 n=int(n_samples * 1))  # z=1, +
        nv2, X2, y2, z2 = gen_gaussian_diff_size(mu2, sigma2, 0, +1, int(n_samples * 1))  # z=0, +
        nv3, X3, y3, z3 = gen_gaussian_diff_size(mu3, sigma3, 1, 0, int(n_samples * 1))  # z=1, -
        nv4, X4, y4, z4 = gen_gaussian_diff_size(mu4, sigma4, 0, 0, int(n_samples * 1))  # z=0, -
        X = np.vstack((X1, X2, X3, X4))
        y = np.hstack((y1, y2, y3, y4))
        x_control = np.hstack((z1, z2, z3, z4))

    elif data_type == 4:
        x1_g1 = np.random.randn(math.floor(n_samples / 2))
        x1_g2 = np.random.randn(math.floor(n_samples / 2)) + 0.5
        z1 = np.full(x1_g1.shape, 0.0)
        z2 = np.full(x1_g2.shape, 1.0)
        x1 = np.concatenate((x1_g1, x1_g2))
        z = np.concatenate((z1, z2))
        a_sorted = sorted(x1)
        threshold = a_sorted[math.floor(n_samples - 1 - n_samples / 2)]
        # There has to be a better way of dong this.
        x2 = np.random.uniform(0, 1, x1.shape[0])
        y = np.array([1 if v > threshold else 0 for v in x1])
        X = np.hstack((x2.reshape(-1, 1), x1.reshape(-1, 1)))
        x_control = z

    elif data_type == 5:

        """
        Fairness beyond disparate treatment
        Case I modified
        """

        mu1, sigma1 = [7, 0], [[5, 1], [1, 5]]  # z=1, +
        # mu2, sigma2 = [7, 0], [[5, 1], [1, 5]]  # z=0, +

        # mu3, sigma3 = [-7, 0], [[5, 1], [1, 5]]  # z=1, -
        mu4, sigma4 = [-7, 0], [[5, 1], [1, 5]]  # z=0, -

        nv1, X1, y1, z1 = gen_gaussian_diff_size(mean_in=mu1, cov_in=sigma1, z_val=1, class_label=1,
                                                 n=int(n_samples * 1))  # z=1, +
        # nv2, X2, y2, z2 = gen_gaussian_diff_size(mu2, sigma2, 0, +1, int(n_samples * 1))  # z=0, +
        # nv3, X3, y3, z3 = gen_gaussian_diff_size(mu3, sigma3, 1, 0, int(n_samples * 1))  # z=1, -
        nv4, X4, y4, z4 = gen_gaussian_diff_size(mu4, sigma4, 0, 0, int(n_samples * 1))  # z=0, -
        X = np.vstack((X1, X4,))
        y = np.hstack((y1, y4))
        x_control = np.hstack((z1, z4))

    else:
        # generate dataset: x1, x2, x3, z, y
        # x1 = uniform[0,2]
        # x2 = uniform[0,2]
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

        x1_g1 = np.random.randn(math.floor(n_samples / 2))
        x1_g2 = np.random.randn(math.floor(n_samples / 2)) + 1
        z1 = np.full(x1_g1.shape, 0.0)
        z2 = np.full(x1_g2.shape, 1.0)
        x1 = np.concatenate((x1_g1, x1_g2))
        z = np.concatenate((z1, z2))
        a_sorted = sorted(x1)
        threshold = a_sorted[math.floor(n_samples - 1 - n_samples / 2)]
        # There has to be a better way of dong this.
        x2 = np.random.uniform(0, 1, x1.shape[0])
        x3 = np.random.uniform(0, 1, x1.shape[0])
        y = np.array([1 if v > threshold else 0 for v in x1])
        X = np.hstack((x2.reshape(-1, 1), x3.reshape(-1, 1)))
        x_control = z

    # merge the clusters


    # shuffle the data
    perm = list(range(len(X)))
    random.shuffle(perm)
    X = X[perm]
    y = y[perm]
    x_control = x_control[perm]

    """ Plot the data """
    if plot_data:
        plt.figure()
        num_to_draw = 800  # we will only draw a small number of points to avoid clutter
        x_draw = X[:num_to_draw]
        y_draw = y[:num_to_draw]
        x_control_draw = x_control[:num_to_draw]

        X_s_0 = x_draw[x_control_draw == 0.0]
        X_s_1 = x_draw[x_control_draw == 1.0]
        y_s_0 = y_draw[x_control_draw == 0.0]
        y_s_1 = y_draw[x_control_draw == 1.0]

        plt.scatter(X_s_0[y_s_0 == 1.0][:, 0], X_s_0[y_s_0 == 1.0][:, 1], color='green', marker='x', s=60, linewidth=2,
                    label="group-0 y=1")
        plt.scatter(X_s_0[y_s_0 == 0.0][:, 0], X_s_0[y_s_0 == 0.0][:, 1], color='red', marker='x', s=60, linewidth=2,
                    label="group-0 y=0")
        plt.scatter(X_s_1[y_s_1 == 1.0][:, 0], X_s_1[y_s_1 == 1.0][:, 1], color='green', marker='o', facecolors='none',
                    s=60, linewidth=2, label="group-1 y=1")
        plt.scatter(X_s_1[y_s_1 == 0.0][:, 0], X_s_1[y_s_1 == 0.0][:, 1], color='red', marker='o', facecolors='none',
                    s=60, linewidth=2, label="group-1 y=0")

        plt.tick_params(axis='x', which='both', bottom='off', top='off',
                        labelbottom='off')  # dont need the ticks to see the data distribution
        plt.tick_params(axis='y', which='both', left='off', right='off', labelleft='off')
        plt.legend(loc=2, fontsize=15)
        # plt.ylim((-8, 12))

        plt.show()

    # x_control = {"s1": x_control}  # all the sensitive features are stored in a dictionary
    # X = ut.add_intercept(X)

    return X, y, x_control


class TwoGaussiansnew(Data):

    def __init__(self, n_samples, datatype):
        Data.__init__(self)
        self.dataset_name = 'two-gaussiansnew_n' + str(n_samples) + '_t' + str(datatype)
        self.class_attr = 'decision'
        self.positive_class_val = 1
        self.sensitive_attrs = ['sensitive-attr']
        self.privileged_class_names = [1]
        self.negative_class_val = 0
        self.categorical_features = []
        self.features_to_keep = ['x1', 'x2', 'sensitive-attr', 'decision'] # x1 与 sensitive相关； x2为随机产生
        self.missing_val_indicators = ['?']
        self.n_samples = n_samples
        self.datatype = datatype

    def load_raw_dataset(self):

        X, y, x_control = generate_synthetic_data(data_type=self.datatype, plot_data=True, n_samples=self.n_samples)
        return pd.DataFrame(data={
            "x1": X[:, 0],
            "x2": X[:, 1],
            "decision": y,
            "sensitive-attr": x_control,
            })


