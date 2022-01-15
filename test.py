from pyecharts import options as opts
from pyecharts.charts import Scatter
from pyecharts.commons.utils import JsCode
import numpy as np
import geatpy as ea


class Bi:
    def __init__(self, ebb):
        self.ebb = ebb

    def __str__(self):
        return str(self.ebb)

a = np.array([Bi(123), Bi("ASdsada"), Bi(True)])
for each in a:
    print(each)

# pop = np.array([0,1,2,3,4,5,6,7,8,9])
# a = np.array([6,0,8,1,9,4,5,7,3,2])
# print(pop)
# b = pop[a]
# print(b)
# [level, _] = ea.ndsortDED(pop)
# pop1 = pop[np.where(level==1)]
# pop2 = pop[np.where(level!=1)]
# print(level)
# print(pop1)
# print(pop2)



# data = [[13, 12], [16, 43], [16, 86], [26, 46]]
# x = [each[0] for each in data]
# y = [each[1] for each in data]
# print(x)
# print(y)
#
# a = Scatter()
# a.add_xaxis(x)
# a.add_yaxis(
#     "商家A",
#     [each for each in zip(y, x)],
#     label_opts=opts.LabelOpts(is_show=False)
# )
# a.set_global_opts(
#     title_opts=opts.TitleOpts(title="Scatter-多维度数据"),
#     tooltip_opts=opts.TooltipOpts(
#         formatter=JsCode(
#             "function (params) {return '( '+ params.value[2] +' : '+ params.value[1] + ' )';}"
#         )
#     ),
#     visualmap_opts=opts.VisualMapOpts(
#         type_="color", max_=150, min_=20, dimension=1
#     ),
#     xaxis_opts=opts.AxisOpts(type_='value'),
#     yaxis_opts=opts.AxisOpts(type_='value')
# )
# a.render("scatter_multi_dimension.html")




# chart = (Scatter(opts.InitOpts(width="600px", height="600px"))
#          .set_global_opts(xaxis_opts=opts.AxisOpts(name='x-aix',
#                                                    name_location='center',
#                                                    name_gap=20,
#                                                    type_="value"),
#                           yaxis_opts=opts.AxisOpts(name='y-aix',
#                                                    name_gap=20,
#                                                    type_="value"),
#                           title_opts=opts.TitleOpts(title="公平性指标和准确性指标优化结果", subtitle="subtitile"),
#                           )
#          )
# colors = []
# for i in range(10):
#     tmp = 20 - i
#     colors.append(str(hex(50 + i * 20))[2:])
#     x = [tmp + v for v in tmp * np.linspace(-0.3, 0.3, 20)]
#     y = [tmp - v for v in tmp * np.linspace(-0.3, 0.3, 20)]
#     chart.add_xaxis(x)
#     chart.add_yaxis(
#         series_name="",
#         y_axis=[each for each in zip(y, x)],
#         symbol_size=5,
#         symbol=None,
#         is_selected=True,
#         color='#00{}FF'.format(colors[i]),
#         label_opts=opts.LabelOpts(is_show=False)
#     )
# chart.set_global_opts(tooltip_opts=opts.TooltipOpts(
#     formatter=JsCode(
#         "function (params) {return '( '+ params.value[2] +' : '+ params.value[1] + ' )';}"
#     )
# ))
# chart.render("scatter_multi_dimension.html")

