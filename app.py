import os
from flask import Flask, render_template, request
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
         print(fname[-7:])
         if fname[-7:] != '.pickle':
            continue
         with open(path + '/' + fname, 'rb') as f:
            process = CustomUnpickler(f).load()
         if process.finished:
            kwargs['f_processes'].append(process)
         else:
            kwargs['r_processes'].append(process)
   return render_template('index.html', **kwargs)


if __name__ == '__main__':
   app.run(debug = True)
