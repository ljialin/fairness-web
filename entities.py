import pickle

class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(__name__, name)
        except AttributeError:
            return super().find_class(module, name)


class Process:
    # Don't overwrite __getattr__(self, item) function!

    def __init__(self, **kwargs):
        self.finished = kwargs['finished']
        self.pid = kwargs['pid']

    def get_name(self):
        return 'process_%03d' % self.pid

    def __str__(self):
        return 'process_%03d' % self.pid


