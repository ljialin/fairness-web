# """
#   @Time : 2021/8/30 15:46
#   @Author : Ziqi Wang
#   @File : model_upload.py
# """

import os
import torch
import inspect
import importlib
from werkzeug.utils import secure_filename
from root import PRJROOT
# from task_space.task0000.model import main

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
    struct_file.save(struct_path)  # 结构定义文件，作为临时文件复制到一个地方
    var_path = os.path.join(
        PRJROOT + 'models/temp',
        prefix + secure_filename(var_file.filename)
    )
    var_file.save(var_path) # 模型文件

    try:
        module = importlib.import_module(
            'models.temp.' + secure_filename(prefix + struct_file.filename)[:-3]
        )
        importlib.reload(module)
        funcs = inspect.getmembers(module, inspect.isfunction)

        if len(funcs) != 1:
            raise RuntimeError('模型的类定义文件必须只含一个函数')

        func = funcs[0][1]
        model = func()  # 数据类型本质上是nn.Module
        state_dict = torch.load(var_path)
        model.load_state_dict(state_dict)
    except Exception as e:
        os.remove(var_path)
        os.remove(struct_path)
        raise RuntimeError('导入模型失败\n错误信息：{}'.format(str(e)))


    return var_file.filename[:-4], model


def init_pop_from_uploaded(task_id, struct_file, var_files, n):
    print(type(struct_file))
    # print(var_files.filename)
    if struct_file.filename[-3:] != '.py':
        raise RuntimeError('模型结构定义文件必须是.py后缀')
    for var_file in var_files:
        # print(type(var_file))
        if var_file.filename[-4:] not in {'.pth', '.pkl'}:
            raise RuntimeError('模型参数文件必须是.pth或.pkl后缀')

    # var_file_suffix =

    space_path = PRJROOT + f'task_space/task{task_id:04d}'
    # struct_path = os.path.join(space_path, secure_filename(struct_file.filename))
    struct_path = os.path.join(space_path, 'model.py')
    struct_file.save(struct_path)
    var_paths = []
    for i, var_file in enumerate(var_files):
        if i == n:
            break
        var_path = os.path.join(
            space_path, f'model_{i}.{var_file.filename[-3:]}'
        )
        var_file.save(var_path)
        var_paths.append(var_path)

    module = importlib.import_module(f'task_space.task{task_id:04d}.model')
    print(inspect.getmembers(module))
    funcs = inspect.getmembers(module, inspect.isfunction)
    # print(funcs)

    if len(funcs) != 1:
        raise RuntimeError('模型的类定义文件必须有且只有一个函数')

    func = funcs[0][1]
    models = []
    for var_path in var_paths:
        model = func()
        model.load_state_dict(torch.load(var_path))
        models.append(model)
    while len(models) < n:
        model = func()
        models.append(model)
        torch.save(
            model.state_dict(),
            space_path + f'/model_{len(models)}.pth'
        )

    return models
