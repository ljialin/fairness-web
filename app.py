"""
  @DATE: 2021/6/03
  @Author: Ziqi Wang
  @File: app.py
"""

import os
from socket import gethostname
from flask import Flask, render_template, request, abort, redirect
from pyecharts.charts import Bar, Radar
from pyecharts import options as opts
from src.mvc.model_eval import ModelEvalController
# from root import HOSTIP
# from src.task import Task
# from src.utils import CustomUnpickler
from src.mvc.data import DataController
from src.mvc.data_eval import DataEvalController


app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    if not request.form:
        return render_template('index.html')
    else:
        if request.form['name'] == 'next':
            ip = request.remote_addr
            target = request.form['tar']
            DataController(ip, target)
            return redirect('/data')

@app.route('/data', methods=['GET', 'POST'])
def data_page():
    ip = request.remote_addr
    ctrlr = DataController.insts[ip]
    if not request.form:
        return render_template('data.html', view=ctrlr.view)
    else:
        errinfo = None
        if request.form['name'] == 'upload-dataset':
            errinfo = ctrlr.upload_dataset(
                request.files['desc'], request.files['data'],
                request.form['keep-or-not'] == 'T'
            )
        elif request.form['name'] == 'select-dataset':
            errinfo = ctrlr.select_dataset(
                request.form['dataset']
            )
        elif request.form['name'] == 'next':
            if ctrlr.model is None:
                errinfo = '必须选择数据集并确认后才能进行下一步'
            else:
                return redirect(ctrlr.target)
        return render_template('data.html', view=ctrlr.view, errinfo=errinfo)

@app.route('/data-eval', methods=['GET', 'POST'])
def data_eval():
    ip = request.remote_addr
    form = request.form
    if not form:
        if DataController.insts[ip] is None:
            return '必须在选择数据集页面选择数据集后才能访问该页面'
        ctrlr = DataEvalController(ip, DataController.insts[ip].model)
    else:
        # print(request.form)
        # print(request.form.getlist('sens-featrs'))
        ctrlr = DataEvalController.insts[ip]
        if form['name'] == 'eval':
            sens_featrs = form.getlist('sens-featrs')
            if form['type'] == '群体公平分析':
                ctrlr.gf_eval(sens_featrs)
            elif form['type'] == '个体公平分析':
                ctrlr.if_eval(sens_featrs)
            elif form['type'] == '条件性群体公平分析':
                ctrlr.cgf_eval(sens_featrs, form['legi_featr'])

    return render_template('data_eval.html', view=ctrlr.view, ip=ip)

@app.route('/model-upload', methods=['GET', 'POST'])
def model_upload():
    ip = request.remote_addr
    if request.files:
        # print(request.files)
        try:
            ModelEvalController(ip, request.files['struct'], request.files['var'])
        except RuntimeError as e:
            return render_template('model_upload.html', errinfo=str(e))
        return redirect('/model-eval')
    return render_template('model_upload.html')

@app.route('/model-eval', methods=['GET', 'POST'])
def model_eval():
    ip = request.remote_addr
    form = request.form
    if ModelEvalController.insts[ip] is None:
        return '必须在选择数据集页面选择数据集，并在上传模型页面上传模型后才能访问该页面'
    ctrlr = ModelEvalController.insts[ip]
    if form:
        if form['name'] == 'eval':
            sens_featrs = form.getlist('sens-featrs')
            if form['type'] == '群体公平分析':
                ctrlr.gf_eval(sens_featrs)
            elif form['type'] == '条件性群体公平分析':
                ctrlr.cgf_eval(sens_featrs, form['legi_featr'])
        pass
    return render_template('model_eval.html', view=ctrlr.view)

@app.route('/model-eval/intro')
def metric_intro():
    ip = request.remote_addr
    ctrlr = ModelEvalController.insts[ip]
    return render_template('metric_intro.html', view=ctrlr.view)

@app.route('/algo-cfg')
def algo_cfg():
    return render_template('algo_cfg.html')

@app.route('/task/<pid>')
def task_page(pid):
    # filepath = PRJROOT + f'metadata/processes/Task-{pid}.pickle'
    # try:
    #     with open(filepath, 'rb') as f:
    #         process = CustomUnpickler(f).load()
    # except FileNotFoundError:
    #     abort(404)

    # kwargs = {
    #     'pid': pid,
    #     'finished': process.finished
    # }
    print(pid)
    if pid == '0002':
        finish = True
    else:
        finish = False
    return render_template('task_page.html', pid=pid, finished=finish)

# 向前端js发送图表数据
@app.route('/data-eval/charts/<cid>')
def data_eval_charts(cid):
    ip = request.remote_addr
    charts = DataEvalController.insts[ip].charts
    return charts[cid].dump_options_with_quotes()

@app.route('/model-eval/charts/<cid>')
def model_eval_charts(cid):
    ip = request.remote_addr
    charts = ModelEvalController.insts[ip].charts
    return charts[cid].dump_options_with_quotes()

####### Hard-encoding charts #######
@app.route('/charts/datagf/<feature>')
def datagf_chart(feature):
    chart = Bar()
    if feature == 'gender':
        chart.add_xaxis(['Male', 'Female'])
        chart.add_yaxis('', [0.96, 1.03])
        chart.set_global_opts(title_opts=opts.TitleOpts(title="Gender"))
    elif feature == 'race':
        chart.add_xaxis(['Yellow', 'Black', 'White'])
        chart.add_yaxis('', [1.1, 0.75, 1.1])
        chart.set_global_opts(title_opts=opts.TitleOpts(title="Race"))
    return chart.dump_options_with_quotes()

@app.route('/charts/datacgf/<feature>')
def datacgf_chart(feature):
    chart = Bar()
    if feature == 'gender':
        chart.add_xaxis(['A', 'B', 'C', 'D', 'F'])
        chart.add_yaxis('Male', [1, 0.96, 1, 1, 1])
        chart.add_yaxis('Female', [1, 1.02, 1, 1, 1])
    elif feature == 'race':
        chart.add_xaxis(['A', 'B', 'C', 'D', 'F'])
        chart.add_yaxis('Yellow', [1, 1, 1, 1, 1])
        chart.add_yaxis('Black', [1, 1, 1, 1, 0.97])
        chart.add_yaxis('White', [1, 1, 1, 1, 1.04])
    return chart.dump_options_with_quotes()

@app.route('/charts/models-radar/<feature>')
def model_radar_chart(feature):
    chart = (
        Radar()
        .add_schema(
            schema=[
                opts.RadarIndicatorItem(name="Statistical Parity", max_=2),
                opts.RadarIndicatorItem(name="Well-calibration", min_=-2, max_=2),
                opts.RadarIndicatorItem(name="Equalized odds", max_=2),
                opts.RadarIndicatorItem(name="PPV", max_=2),
                opts.RadarIndicatorItem(name="FPR", max_=2),
                opts.RadarIndicatorItem(name="FNR", max_=2),
                opts.RadarIndicatorItem(name="NPV", max_=2),
            ],
            splitarea_opt=opts.SplitAreaOpts(
                is_show=True, areastyle_opts=opts.AreaStyleOpts(opacity=1)
            ),
        )
        .add(
            series_name="公平范围",
            data=[
                [0.8, -0.5, 0.8, 0.8, 0.8, 0.8, 0.8],
                [1.25, 0.5, 1.25, 1.25, 1.25, 1.25, 1.25]
            ],
            linestyle_opts=opts.LineStyleOpts(color="red"),
            label_opts=opts.LabelOpts(is_show=False)
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title=f"{feature}"), legend_opts=opts.LegendOpts(),
        )
    )
    if feature == 'gender':
        data = [
            [0.93, -0.521, 1.367, 0.916, 0.88, 1.225, 0.813],
            [1.017, 0.326, 1.082, 1.217, 1.061, 0.839, 1.293]
        ]
        groups = ('male', 'female')
    else:
        data = [
            [1.184, 0.088, 0.776, 0.897, 0.77, 1.219, 0.731],
            [1.285, -0.191, 0.801, 1.344, 1.133, 1.24, 1.13],
            [0.794, -0.03, 1.088, 1.23, 1.074, 1.035, 0.953]
        ]
        groups = ('yellow', 'white', 'black')
    colors = ('blue', 'darkgreen', 'purple')
    for i, group in enumerate(groups):
        chart.add(
            series_name=group, data=[data[i]],
            linestyle_opts=opts.LineStyleOpts(color=colors[i]),
            label_opts=opts.LabelOpts(is_show=False)
        )
    return chart.dump_options_with_quotes()
####################################
# @app.route('')


if __name__ == '__main__':
    app.run()
