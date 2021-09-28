"""
  @Time : 2021/9/25 15:26 
  @Author : Ziqi Wang
  @File : fair_analyze.py 
"""
from pyecharts import options as chopts
from pyecharts.charts import Bar

METRICS = ('Acc', 'PLR', 'PPV', 'FPR', 'FNR', 'NPV')
METRIC_UBS = {'Acc': 1, 'PLR': 1, 'PPV': 1, 'FPR': 1, 'FNR': 1, 'NPV': 1}
THRESHOLDS = {'Acc': 0.8, 'PLR': 0.8, 'PPV': 0.8, 'FPR': 0.8, 'FNR': 0.8, 'NPV': 0.8}
PREFER_HIGH = {'Acc': 1, 'PLR': 1, 'PPV': 1, 'FPR': 1, 'FNR': 0, 'NPV': 0}
NUM_METRICS = len(METRICS)
N_SPLIT = 5
assert set(METRICS) == set(METRIC_UBS.keys())

# PLR: Positive Label Rate. Statistical Parity requeirs PLR of each group close to global PLR.
class CgfAnalyzeRes:
    def __init__(self, featr_name):
        self.featr_name = featr_name
        self.sens_groups = []
        self.legi_groups = []
        self.cmmts = []
        self.data = {}
        self.f_range = {}

    def get_chart(self):
        chart = (
            Bar()
            .set_global_opts(
                title_opts=chopts.TitleOpts(title=self.featr_name)
            )
            .add_xaxis(self.legi_groups)
        )
        for sens_grp in self.sens_groups:
            chart.add_yaxis(
                sens_grp,
                [self.data[legi_grp][sens_grp] for legi_grp in self.legi_groups],
                label_opts = chopts.LabelOpts(is_show=False)
            )
        return chart