# from Sample import Sample
import geatpy as ea
from geatpy.zqq.data.objects.Ricci import Ricci
from geatpy.zqq.data.objects.Adult import Adult
from geatpy.zqq.data.objects.German import German
from geatpy.zqq.data.objects.PropublicaRecidivism import PropublicaRecidivism
from geatpy.zqq.data.objects.PropublicaViolentRecidivism import PropublicaViolentRecidivism
from geatpy.zqq.data.objects.TwoGaussians import TwoGaussians
from geatpy.zqq.data.objects.TwoGaussiansnew import TwoGaussiansnew
from geatpy.zqq.data.objects.GradeScore import GradeScore
from geatpy.zqq.data.objects.Bank import Bank
from geatpy.zqq.data.objects.Default import Default
from geatpy.zqq.data.objects.LSAT import LSAT
from geatpy.zqq.data.objects.Dutch import Dutch
from geatpy.zqq.data.objects.Student_mat import Student_mat
from geatpy.zqq.data.objects.Student_por import Student_por

DATASETS = [

    # Synthetic datasets to test effects of class balance:
    #   TwoGaussians(0.1), TwoGaussians(0.2), TwoGaussians(0.3), TwoGaussians(0.4),
    #   TwoGaussians(0.5), TwoGaussians(0.6), TwoGaussians(0.7), TwoGaussians(0.8), TwoGaussians(0.9),

    # Downsampled datasetes to test effects of class and protected class balance:
    #     Sample(Ricci(), num = 1000, prob_pos_class = 0.5, prob_privileged = 0.5, sensitive_attr="Race"),
    #     Sample(Adult(), num = 1000, prob_pos_class = 0.5, prob_privileged = 0.5,
    #     sensitive_attr="race-sex"),
    #     Sample(German(), num = 1000, prob_pos_class = 0.5, prob_privileged = 0.5, sensitive_attr="sex-age"),
    #     Sample(PropublicaRecidivism(), num = 1000, prob_pos_class = 0.5, prob_privileged = 0.5,
    #     sensitive_attr="sex-race"),
    #     Sample(PropublicaViolentRecidivism(), num = 1000, prob_pos_class = 0.5, prob_privileged = 0.5,
    #     sensitive_attr="sex-race")

    # Real datasets:
    Student_mat(),
    Dutch(),
    Ricci(),
    Adult(),
    German(),
    PropublicaRecidivism(),
    Bank(),
    Default(),
    LSAT()
    # TwoGaussiansnew(2500, 2),
    # TwoGaussiansnew(1000, 2),
    # TwoGaussiansnew(2500, 3),
    # TwoGaussiansnew(2500, 4),
    # TwoGaussiansnew(1000, 5),
    # TwoGaussiansnew(5000, 2),

    # TwoGaussiansnew(10000),
    # GradeScore(10000),
    # PropublicaViolentRecidivism()
]


def get_dataset_names():
    names = []
    for dataset in DATASETS:
        names.append(dataset.get_dataset_name())
    return names


def add_dataset(dataset):
    DATASETS.append(dataset)


def get_dataset_by_name(name):
    for ds in DATASETS:
        if ds.get_dataset_name() == name:
            return ds
    raise Exception("No dataset with name %s could be found." % name)
