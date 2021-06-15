import pickle

class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(__name__, name)
        except AttributeError:
            return super().find_class(module, name)


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


