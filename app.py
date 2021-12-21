"""
  @DATE: 2021/6/03
  @Author: Ziqi Wang
  @File: app.py
"""
import math
import os
from socket import gethostname

import numpy as np
from flask import Flask, render_template, request, redirect, send_from_directory, jsonify
from pyecharts.charts import Bar, Radar, Scatter
from pyecharts import options as opts
from pyecharts.commons.utils import JsCode
from src.mvc.model_eval import ModelEvalController
from src.mvc.algo_cfg import AlgoController, AlgosManager
# from root import HOSTIP
# from src.task import Task
# from src.utils import CustomUnpickler
from src.mvc.data import DataController
from src.mvc.data_eval import DataEvalController
from mvc.algo_cfg import STATUS
from Fairness_main import interface4flask
from zipfile import ZipFile

url = "127.0.0.1:5000"
port = 5000
app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    ip = request.remote_addr
    if AlgosManager.instances.get(ip) is None:
        AlgosManager(ip)  # 在最初的时候就初始化好AlgoManager

    if not request.form:
        return render_template('index.html', algomnger=AlgosManager.instances[ip])
    else:
        if request.form['name'] == 'next':
            target = request.form['tar']
            # if ip in DataController.insts.keys():
            #     DataController.insts[ip].free()
            # else:
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
                request.files['data'],
                request.form['keep-or-not'] == 'F'
            )
        elif request.form['name'] == 'select-dataset':
            if request.form.get("dataset") is None:
                errinfo = "请选择数据集"
            else:
                errinfo = ctrlr.select_dataset(
                    request.form['dataset']
                )
        elif request.form['name'] == 'next':
            if ctrlr.model is None:
                errinfo = '必须选择数据集并确认后才能进行下一步'
            else:
                return redirect(ctrlr.target)
        return render_template('data.html', view=ctrlr.view, errinfo=errinfo)


@app.route('/data/desc_template')
def download_desc_template():
    return send_from_directory('static/file_templates', 'data.csv', as_attachment=True)


@app.route('/data-eval', methods=['GET', 'POST'])
def data_eval():
    ip = request.remote_addr
    form = request.form
    errinfo = None
    if not form:  # 检测是不是在页面内按按钮之后，还要用这个页面
        if DataController.insts[ip] is None:
            return '必须在选择数据集页面选择数据集后才能访问该页面'
        ctrlr = DataEvalController(ip, DataController.insts[ip].model)
    else:
        # print(request.form)
        # print(request.form.getlist('sens-featrs'))
        ctrlr = DataEvalController.insts[ip]
        if form['name'] == 'eval':
            sens_featrs = form.getlist('sens-featrs')
            legi_featr = form.get('legi-featr')
            if form['type'] == '群体公平分析':
                errinfo = ctrlr.gf_eval(sens_featrs)
            elif form['type'] == '个体公平分析':
                errinfo = ctrlr.if_eval(legi_featr)
            elif form['type'] == '条件性群体公平分析':
                errinfo = ctrlr.cgf_eval(sens_featrs, legi_featr)

    # 这里的ip好像没被用到
    return render_template('data_eval.html', view=ctrlr.view, url=url, errinfo=errinfo)


@app.route('/model-upload', methods=['GET', 'POST'])
def model_upload():
    ip = request.remote_addr
    if request.files:
        # print(request.files)
        try:
            ModelEvalController(ip, request.files['struct'], request.files['var'])
        except RuntimeError as e:
            print(str(e))
            return render_template('model_upload.html', errinfo=str(e))
        return redirect('/model-eval')
    return render_template('model_upload.html')


@app.route('/model/model_def_template')
def download_model_template():
    return send_from_directory('static/file_templates', 'model_defination.py', as_attachment=True)


@app.route('/model-eval', methods=['GET', 'POST'])
def model_eval():
    ip = request.remote_addr
    form = request.form
    if ModelEvalController.insts[ip] is None:
        return '必须在选择数据集页面选择数据集，并在上传模型页面上传模型后才能访问该页面'
    ctrlr = ModelEvalController.insts[ip]
    errinfo = None
    if form:
        if form['name'] == 'eval':
            sens_featrs = form.getlist('sens-featrs')
            if form['type'] == '群体公平分析':
                errinfo = ctrlr.gf_eval(sens_featrs)
            elif form['type'] == '条件性群体公平分析':
                errinfo = ctrlr.cgf_eval(sens_featrs, form['legi-featr'])
    return render_template('model_eval.html', url=url, view=ctrlr.view, errinfo=errinfo)


@app.route('/model-eval/intro')
def metric_intro():
    ip = request.remote_addr
    if ModelEvalController.insts[ip] is None:
        return '必须在选择数据集页面选择数据集，并在上传模型页面上传模型后才能访问该页面'
    ctrlr = ModelEvalController.insts[ip]
    return render_template('metric_intro.html', view=ctrlr.view)


@app.route('/algo-cfg', methods=['GET', 'POST'])
def algo_cfg():
    ip = request.remote_addr
    form = request.form
    print("algo_cfg:form", form)
    errinfo = None
    if not form:  # 检测是不是首次载入这个页面，首次载入应该是被选数据页面重定向过来的
        if DataController.insts.get(ip) is None:
            return '必须在选择数据集页面选择数据集后才能访问该页面'
        else:  # 是第一次，且经过选择数据集这部跳转
            data_model = DataController.insts[ip].model
            ctrlr = AlgoController(ip, data_model)  # 新建一个Controller，默认值写在这个类里面
            return render_template('algo_cfg.html', view=ctrlr.view, cfg=ctrlr.cfg, errinfo=errinfo)
    else:  # 点击了页面中的按钮之后执行的
        ctrlr = AlgoController.instances[ip]
        errinfo, task_id = ctrlr.new_task(
            acc_metric=form['acc_metric'],
            fair_metric=form['fair_metric'],
            optimizer=form['optimizer'],
            pop_size=form['pop_size'],
            max_gens=form['max_gens'],
            sens_featrs=form.getlist('sens-featrs')
        )
        AlgosManager.instances[ip].running_tasks[task_id] = ctrlr
        if errinfo is not None:
            return render_template('algo_cfg.html', view=ctrlr.view, cfg=ctrlr.cfg, errinfo=errinfo)
        if form['type'] == '上传初始化模型':
            return redirect('/algo-cfg/model-upload')
        else:
            # return redirect(f'/task/{task_id:04d}')
            return redirect('/task/{}'.format(task_id))


@app.route('/algo-cfg/model-upload', methods=['GET', 'POST'])
def model_upload_for_algo():
    ip = request.remote_addr
    if request.files:
        try:
            ctrlr = AlgoController.instances[ip]
            ctrlr.add_models(request.files['struct'], request.files.getlist('var'))
            # ModelEvalController(ip, request.files['struct'], request.files['var'])
        except RuntimeError as e:
            return render_template('model_upload_4_algo.html', errinfo=str(e))
        return redirect(f'/task/{ctrlr.task.id:04d}')
    return render_template('model_upload_4_algo.html')


@app.route('/task/<task_id>')
def task_page(task_id):
    print(task_id)
    form = request.form
    ip = request.remote_addr
    algomnger = AlgosManager.instances[ip]
    ctrlr = algomnger.get_task(task_id)
    algoCfg = ctrlr.cfg
    algoView = ctrlr.view

    algomnger.add_task(task_id, ctrlr)  # 在新建任务，跳转到运行页面之前，把这个task加入管理者

    fpops = [] if len(ctrlr.pops) == 0 else np.around(ctrlr.pops[-1], 5).tolist()

    return render_template('task_page.html', pid=task_id, status=ctrlr.status,
                           fpops=fpops, cfg=algoCfg, view=algoView, url=url)


@app.route('/task/<task_id>/intervene')
def abort_task(task_id):
    ip = request.remote_addr
    algomnger = AlgosManager.instances[ip]
    ctrlr = algomnger.get_task(task_id)
    ctrlr.status = STATUS.ABORT
    return jsonify({'res': 0})


@app.route('/task/<task_id>/pause')
def pause_task(task_id):
    ip = request.remote_addr
    algomnger = AlgosManager.instances[ip]
    ctrlr = algomnger.get_task(task_id)
    if ctrlr.status == STATUS.RUNNING: # 确保只有在运行状态才显示按钮
        ctrlr.status = STATUS.PAUSE
    elif ctrlr.status == STATUS.PAUSED:
        ctrlr.status = STATUS.RUNNING
    return jsonify({'res': 0})


@app.route('/task/<task_id>/progress')
def run_task(task_id):
    # JS传参开头带0自动转为八进制的问题,通过task_id前面加a解决
    ip = request.remote_addr
    algomnger = AlgosManager.instances[ip]
    ctrlr = algomnger.get_task(task_id)

    if ctrlr.status != STATUS.INIT: #已经有算法在跑住了（用户直接访问带task_id的页面）
        print(ctrlr.status) #看到状态了前端才会停止刷新
        return jsonify({'progress_info': ctrlr.progress_info,
                        'progress_rate': ctrlr.progress,
                        'progress_status': ctrlr.status})

    interface4flask(ctrlr, task_id)

    fpops = [] if len(ctrlr.pops) == 0 else np.around(ctrlr.pops[-1], 5).tolist()
    return jsonify({'progress_info': ctrlr.progress_info,
                    'progress_rate': ctrlr.progress,
                    'progress_status': ctrlr.status,
                    'pop': fpops})


@app.route('/task/<task_id>/show_progress')
def show_progress(task_id):
    ip = request.remote_addr
    algomnger = AlgosManager.instances[ip]
    ctrlr = algomnger.get_task(task_id)
    fpops = [] if len(ctrlr.pops) == 0 else np.around(ctrlr.pops[-1], 5).tolist()
    # print(ctrlr.status)
    return jsonify({'progress_info': ctrlr.progress_info,
                    'progress_rate': ctrlr.progress,
                    'progress_status': ctrlr.status,
                    'pop': fpops})


@app.route('/task/<task_id>/download_model')
def download_model(task_id):
    ip = request.remote_addr
    dir = "task_space/{}/{}".format(ip, task_id)
    with ZipFile(dir + os.sep + 'net.zip', 'w') as zip:
        dir2 = dir + '/net'
        for file in os.listdir(dir2):
            zip.write(dir2 + os.sep + file, file)
    return send_from_directory(dir, 'net.zip', as_attachment=True)


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


@app.route('/task/<task_id>/chart')
def algo_status_chart(task_id):
    ip = request.remote_addr
    algomnger = AlgosManager.instances[ip]
    ctrlr = algomnger.get_task(task_id)
    pop = ctrlr.pops[-1] if len(ctrlr.pops) > 0 else np.array([[0,0]])

    chart = (Scatter(opts.InitOpts(width="600px", height="600px"))
             .set_global_opts(xaxis_opts=opts.AxisOpts(name=ctrlr.cfg.acc_metric,
                                                       name_location='center',
                                                       name_gap=25,
                                                       max_=float('%.3g' % ctrlr.max_pop[0]),
                                                       type_="value"),
                              yaxis_opts=opts.AxisOpts(name=ctrlr.cfg.fair_metric,
                                                       name_gap=40,
                                                       name_location='center',
                                                       max_=float('%.3g' % ctrlr.max_pop[1]),
                                                       type_="value"),
                              title_opts=opts.TitleOpts(title="公平性指标和准确性指标优化结果"),
                              )
             )

    chart.add_xaxis(list(pop[:,0]))
    chart.add_yaxis(
        series_name="",
        y_axis=[each for each in zip(list(pop[:,1]), list(pop[:,0]))],
        symbol_size=3,
        symbol=None,
        is_selected=True,
        color='#00BBFF',
        label_opts=opts.LabelOpts(is_show=False)
    )

    chart.set_global_opts(tooltip_opts=opts.TooltipOpts(
        formatter=JsCode(
            "function (params) {return '( '+ params.value[2] +' : '+ params.value[1] + ' )';}"
        )
    ))
    return chart.dump_options_with_quotes()


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


# @app.route('/charts/models-radar/<feature>')
# def model_radar_chart(feature):
#     chart = (
#         Radar()
#         .add_schema(
#             schema=[
#                 opts.RadarIndicatorItem(name="Statistical Parity", max_=2),
#                 opts.RadarIndicatorItem(name="Well-calibration", min_=-2, max_=2),
#                 opts.RadarIndicatorItem(name="Equalized odds", max_=2),
#                 opts.RadarIndicatorItem(name="PPV", max_=2),
#                 opts.RadarIndicatorItem(name="FPR", max_=2),
#                 opts.RadarIndicatorItem(name="FNR", max_=2),
#                 opts.RadarIndicatorItem(name="NPV", max_=2),
#             ],
#             splitarea_opt=opts.SplitAreaOpts(
#                 is_show=True, areastyle_opts=opts.AreaStyleOpts(opacity=1)
#             ),
#         )
#         .add(
#             series_name="公平范围",
#             data=[
#                 [0.8, -0.5, 0.8, 0.8, 0.8, 0.8, 0.8],
#                 [1.25, 0.5, 1.25, 1.25, 1.25, 1.25, 1.25]
#             ],
#             linestyle_opts=opts.LineStyleOpts(color="red"),
#             label_opts=opts.LabelOpts(is_show=False)
#         )
#         .set_global_opts(
#             title_opts=opts.TitleOpts(title=f"{feature}"), legend_opts=opts.LegendOpts(),
#         )
#     )
#     if feature == 'gender':
#         data = [
#             [0.93, -0.521, 1.367, 0.916, 0.88, 1.225, 0.813],
#             [1.017, 0.326, 1.082, 1.217, 1.061, 0.839, 1.293]
#         ]
#         groups = ('male', 'female')
#     else:
#         data = [
#             [1.184, 0.088, 0.776, 0.897, 0.77, 1.219, 0.731],
#             [1.285, -0.191, 0.801, 1.344, 1.133, 1.24, 1.13],
#             [0.794, -0.03, 1.088, 1.23, 1.074, 1.035, 0.953]
#         ]
#         groups = ('yellow', 'white', 'black')
#     colors = ('blue', 'darkgreen', 'purple')
#     for i, group in enumerate(groups):
#         chart.add(
#             series_name=group, data=[data[i]],
#             linestyle_opts=opts.LineStyleOpts(color=colors[i]),
#             label_opts=opts.LabelOpts(is_show=False)
#         )
#     return chart.dump_options_with_quotes()
####################################
# @app.route('')


if __name__ == '__main__':
    app.run(port=port)
