"""
@DATE: 2021/6/03
@Author: Ziqi Wang
@File: app.py
"""

import os
from flask import Flask, render_template, request, abort
from src.entities import *
from src.process import Process
from src.utils import CustomUnpickler, FileUtils
from src.pages import UserPageMgr

app = Flask(__name__)

app.config["SECRET_KEY"] = 'fairness'


@app.route('/')
def main_page():
    kwargs = {
        'f_processes': [],
        'r_processes': []
    }

    path = PRJROOT + 'metadata/processes'
    for _, _, files in os.walk(path):
        for fname in files:
            if fname[-7:] != '.pickle':
                continue
            with open(path + '/' + fname, 'rb') as f:
                process = CustomUnpickler(f).load()
                Process.count += 1
            if process.finished:
                kwargs['f_processes'].append(process)
            else:
                kwargs['r_processes'].append(process)
    return render_template('index.html', **kwargs)


@app.route('/data', methods=['POST', 'GET'])
def data_page():
    # print(request.form)
    if not request.form:
        return render_template('data_page.html', page=UserPageMgr.new())
    else:
        errinfo = None
        if request.form['name'] == 'upload-dataset':
            errinfo = UserPageMgr.get().upload_dataset(
                request.files['desc'], request.files['data'], request.form['keep-or-not'] == 'T'
            )
        elif request.form['name'] == 'select-dataset':
            errinfo = UserPageMgr.get().select_dataset(
                request.form['dataset']
            )
            print(errinfo)
        return render_template('data_page.html', page=UserPageMgr.get(), errinfo=errinfo)


@app.route('/create')
def create_page():
    return render_template('create_page.html')


@app.route('/process/<pid>')
def process_page(pid):
    filepath = PRJROOT + f'metadata/processes/process_{pid}.pickle'
    try:
        with open(filepath, 'rb') as f:
            process = CustomUnpickler(f).load()
    except FileNotFoundError:
        abort(404)

    kwargs = {
        'pid': pid,
        'finished': process.finished
    }
    return render_template('process_page.html', **kwargs)


if __name__ == '__main__':
    app.run(debug=True)
