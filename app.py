from flask import Flask, render_template, request
from entities import *
import os

app = Flask(__name__)

@app.route('/')
def student():
   kwargs = {
      'f_processes': [],
      'r_processes': []
   }

   for _, _, files in os.walk('./metadata/processes'):
      for fname in files:
         print(fname[-7:])
         if fname[-7:] != '.pickle':
            continue
         with open('metadata/processes/' + fname, 'rb') as f:
            process = CustomUnpickler(f).load()
         if process.finished:
            kwargs['f_processes'].append(process)
         else:
            kwargs['r_processes'].append(process)
   return render_template('index.html', **kwargs)


if __name__ == '__main__':
   app.run(debug = True)
