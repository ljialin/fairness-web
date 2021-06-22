import csv
import json
import pickle
from root import PRJROOT

class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(__name__, name)
        except AttributeError:
            return super().find_class(module, name)


class DataTab:
    def __init__(self, name):
        self.name = name
        self.attrs = []
        self.convert_map = {}
        self.load_convert_map()

    def load_convert_map(self):
        with open(PRJROOT + 'data/' + self.name + '.txt', 'r') as f:
            content = f.read().split('\n')
        attrs_readed = False
        cur_attr = ''
        val_cnt = 0
        for line in content:
            line = line.strip()
            if line[:20] == 'DISCRETE ATTRIBUTES:':
                attrs_readed = True
                continue
            if line == '':
                continue
            if not attrs_readed:
                self.attrs.append(line)
            else:
                if line[-1] == ':':
                    cur_attr = line[:-1]
                    self.convert_map[cur_attr] = {}
                    val_cnt = 0
                else:
                    self.convert_map[cur_attr][line] = val_cnt
                    val_cnt += 1

    def get_discrete_attrs(self):
        return [key for key in self.convert_map]

    def get_numberical_attrs(self):
        numberical_attrs = []
        for key in self.attrs:
            if key in self.convert_map:
                continue
            numberical_attrs.append(key)
        return numberical_attrs


class Process:
    # Don't overwrite __getattr__(self, item) function!
    count = 0

    def __init__(self):
        self.finished = False
        self.pid = Process.count
        Process.count += 1

    def get_name(self):
        return 'process_%03d' % self.pid

    def get_strid(self):
        return '%03d' % self.pid

    def __str__(self):
        return 'process_%03d' % self.pid


if __name__ == '__main__':
    datatab = DataTab('german')
    print(datatab.get_discrete_attrs())
    # for key in datatab.convert_map:
    #     print(datatab.convert_map[key])
    # print(datata)
