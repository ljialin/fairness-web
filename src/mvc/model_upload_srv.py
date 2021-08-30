# """
#   @Time : 2021/8/30 15:46
#   @Author : Ziqi Wang
#   @File : model_upload_srv.py
# """
#
# import os
# from werkzeug.utils import secure_filename
# from root import PRJROOT
#
#
# def upload_to_mem(model_file):
#     if model_file[-4:] not in {'.pth', '.pkl'}:
#         return '模型文件必须是.pth或.pkl后缀'
#     path = os.path.join(
#         PRJROOT + 'models/temp',
#         secure_filename(model_file.filename)
#     )
#     model_file
#     pass
