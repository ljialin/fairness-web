import os
import csv
from flask import Flask, render_template, request, abort
from root import PRJROOT
from entities import *

app = Flask(__name__)


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


@app.route('/data')
def data_page():
    kwargs = {
        'datasets':{}
    }
    path = PRJROOT + 'data/'
    datasets = []
    for _, _, files in os.walk(path):
        for fname in files:
            if fname[-4:] != '.csv':
                continue
            kwargs['datasets'][fname[:-4]] = {'discrete'}
    return render_template('data_page.html', **kwargs)


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
