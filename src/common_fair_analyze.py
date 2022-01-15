"""
  @Time : 2021/9/25 15:26 
  @Author : Ziqi Wang
  @File : common_fair_analyze.py
"""

from pyecharts import options as chopts
from pyecharts.charts import Bar
from flask_babel import gettext as _
from src.utils import get_count_from_series

# METRICS = ('ACC', 'PLR', 'PPV', 'FPR', 'FNR', 'NPV')
METRICS = ('overall accuracy equality', 'statistical parity', 'PPV balance', 'FPR balance', 'FNR balance', 'NPV balance')
METRIC_UBS = {'overall accuracy equality': 1, 'statistical parity': 1, 'PPV balance': 1, 'FPR balance': 1, 'FNR balance': 1, 'NPV balance': 1}
THRESHOLDS = {'overall accuracy equality': 0.8, 'statistical parity': 0.8, 'PPV balance': 0.8, 'FPR balance': 0.8, 'FNR balance': 0.8, 'NPV balance': 0.8}
PREFER_HIGH = {'overall accuracy equality': 1, 'statistical parity': 1, 'PPV balance': 1, 'FPR balance': 1, 'FNR balance': 0, 'NPV balance': 0}
NUM_METRICS = len(METRICS)
N_SPLIT = 5
# assert set(METRICS) == set(METRIC_UBS.keys())

# PLR: Positive Label Rate. statistical parity requeirs PLR of each group close to global PLR.


class CgfAnalyzeRes:
    def __init__(self, sens_featr, legi_featr):
        self.sens_featr = sens_featr
        self.legi_featr = legi_featr
        self.sens_groups = []
        self.legi_groups = []
        self.cmmts = []
        self.data = {}
        self.f_range = {}

    def get_chart(self):
        chart = (
            Bar()
            .set_global_opts(
                yaxis_opts=chopts.AxisOpts(
                    name=_("model_chart_1"),
                    name_gap=10, name_rotate=0,
                    name_textstyle_opts=chopts.TextStyleOpts(
                        align="left"
                    )
                ),
                xaxis_opts=chopts.AxisOpts(
                    # name=f'基于{self.legi_featr}划分的群组',
                    name_location='middle', name_gap=20,
                    axislabel_opts=chopts.LabelOpts(
                        rotate=10,
                        vertical_align='middle',
                        horizontal_align='center',
                        margin=30
                    )
                )
            )
            .add_xaxis(
                [f'{self.legi_featr}={legi_grp}' for legi_grp in self.legi_groups]
            )
        )
        for sens_grp in self.sens_groups:
            chart.add_yaxis(
                "{}={}".format(self.sens_featr,sens_grp),
                # f'{self.sens_featr}={sens_grp}群组',
                [self.data[legi_grp][sens_grp] for legi_grp in self.legi_groups],
                label_opts = chopts.LabelOpts(
                    is_show=False,
                )
            )
        return chart

