# """
#   @Time : 2021/8/30 15:46
#   @Author : Ziqi Wang
#   @File : model_upload.py
# """
import importlib
import inspect
import os

import torch
from werkzeug.utils import secure_filename

from root import PRJROOT


def upload_model(ip, struct_file, var_file):
    if struct_file.filename[-3:] != '.py':
        raise RuntimeError('模型结构定义文件必须是.py后缀')
    if var_file.filename[-4:] not in {'.pth', '.pkl'}:
        raise RuntimeError('模型参数文件必须是.pth或.pkl后缀')

    prefix = ip.replace('.', '_') + '__'
    struct_path = os.path.join(
        PRJROOT + 'models/temp',
        prefix + secure_filename(struct_file.filename)
    )
    struct_file.save(struct_path)
    var_path = os.path.join(
        PRJROOT + 'models/temp',
        prefix + secure_filename(var_file.filename)
    )
    var_file.save(var_path)

    module = importlib.import_module(
        'models.temp.' + secure_filename(prefix + struct_file.filename)[:-3]
    )
    funcs = inspect.getmembers(module, inspect.isfunction)

    if len(funcs) != 1:
        raise RuntimeError('模型的类定义文件必须只含一个函数')

    func = funcs[0][1]
    model = func()
    model.load_state_dict(torch.load(var_path))

    return var_file.filename[:-4], model


def get_pop_from_uploaded(ip, struct_file, var_files, n):
    print(type(struct_file))
    # print(var_files.filename)
    if struct_file.filename[-3:] != '.py':
        raise RuntimeError('模型结构定义文件必须是.py后缀')
    for var_file in var_files:
        # print(type(var_file))
        if var_file.filename[-4:] not in {'.pth', '.pkl'}:
            raise RuntimeError('模型参数文件必须是.pth或.pkl后缀')

    prefix = ip.replace('.', '_') + '__'
    struct_path = os.path.join(
        PRJROOT + 'models/temp',
        prefix + secure_filename(struct_file.filename)
    )
    struct_file.save(struct_path)
    var_paths = []
    for i, var_file in enumerate(var_files):
        if i == n:
            break
        var_path = os.path.join(
            PRJROOT + 'models/temp',
            prefix + secure_filename(var_file.filename)
        )
        var_file.save(var_path)
        var_paths.append(var_path)


    module = importlib.import_module(
        'models.temp.' + secure_filename(prefix + struct_file.filename)[:-3]
    )
    funcs = inspect.getmembers(module, inspect.isfunction)

    if len(funcs) != 1:
        raise RuntimeError('模型的类定义文件必须只含一个函数')

    func = funcs[0][1]
    models = []
    for var_path in var_paths:
        model = func()
        model.load_state_dict(torch.load(var_path))
        models.append(model)
    while len(models) < n:
        models.append(func())

    return models
