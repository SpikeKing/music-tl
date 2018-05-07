#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/5/2

将数据集划分训练和测试
"""
import os
import sys

p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)

from root_dir import ROOT_DIR
from utils.utils import *


def split_train_test(folder_name, n_part=6):
    """
    将数据集分为训练数据和测试数据

    :param folder_name: 文件夹名
    :param n_part: 划分块，1份为测试，其余为训练
    :return: None
    """
    paths, names = traverse_dir_files(folder_name)
    train_dir = os.path.join(folder_name, 'train')
    test_dir = os.path.join(folder_name, 'test')
    mkdir_if_not_exist(train_dir)
    mkdir_if_not_exist(test_dir)

    print "[INFO] 训练总量： %s" % len(paths)
    count = 0
    for path, name in zip(paths, names):
        count += 1
        if count % n_part != 0:
            shutil.move(path, os.path.join(train_dir, name))  # 移动文件
        else:
            shutil.move(path, os.path.join(test_dir, name))

    train_ps, _ = traverse_dir_files(train_dir)
    test_ps, _ = traverse_dir_files(test_dir)
    print "[INFO] 训练数据： %s" % len(train_ps)
    print "[INFO] 测试数据： %s" % len(test_ps)


if __name__ == '__main__':
    data_dir = os.path.join(ROOT_DIR, 'experiments', 'raw_data')
    split_train_test(data_dir)  # 划分训练和测试
