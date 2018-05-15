#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/5/3
"""
import collections
import os
import sys
import time

p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)

from multiprocessing import Pool
from root_dir import ROOT_DIR
from utils.utils import *
from utils.np_utils import *


def get_path_clz(path):
    """
    获取音频类的名称，即原音频和增强音频
    """
    return path.split('/')[-1].split('-')[0].split('.')[0]


def get_path_name(path):
    """
    获取音频的名称
    """
    return path.split('/')[-1].split('-')[0]


def get_label_dict(path_list):
    """
    根据音频路径列表，生成标签字典
    """
    normal_num = 21  # 音频的正常样本数量，小于即认为异常样本

    label_dict = collections.defaultdict(int)

    for path in path_list:
        clz_name = get_path_clz(path)
        label_dict[clz_name] += 1

    for key in label_dict.keys():
        if label_dict[key] < normal_num:
            print '[INFO] 异常类别: %s, 样本数: %s' % (key, label_dict[key])
            label_dict.pop(key, None)  # 删除类别

    label_list = sorted(list(label_dict.keys()))
    res_dict = dict()

    for index, label in enumerate(label_list):
        res_dict[label] = index  # 将类别转换为数字

    return res_dict


def prc_func(path, label):
    """
    子进程，加载数据
    """
    try:
        print '[INFO] ' + path
        features = np.load(path)
        name = get_path_name(path)
        if np.min(features) == np.max(features):
            raise Exception('features error')
        return features, label, name
    except Exception as e:
        print "[INFO] 错误: %s" % str(e)
        return np.array([]), -1, None


def mp_data_merger(data_dir, out_dir, n_prc=40):
    """
    合并数据，数据清洗
    :param data_dir: 文件路径
    :param n_prc: 进程数
    :return: None
    """
    start_time = datetime.now()  # 起始时间
    print "[INFO] 当前时间: %s" % time_2_readable(time.time())

    path_list, _ = traverse_dir_files(data_dir)  # 路径列表
    label_dict = get_label_dict(path_list)
    print "[INFO] 音频总数: %s" % len(path_list)

    results = []
    p = Pool(processes=n_prc)  # 进程数尽量与核数匹配
    count = 0
    for path in path_list:
        clz_name = get_path_clz(path)
        if clz_name not in label_dict:
            print "[INFO] %s 类别样本过少! " % clz_name
            continue
        label = label_dict[clz_name]
        results.append(p.apply_async(prc_func, args=(path, label,)))
        # count += 1
        # if count == 19 * 400:  # 存储指定量的数据，用于测试
        #     break

    count = 0
    f_list, l_list, n_list = [], [], []
    print len(results)
    for i in results:
        features, label, name = i.get()
        if not name or label == -1 or check_error_features(features):  # 检查异常数据
            print "[INFO] 错误数据"
            continue
        f_list.append(features)
        l_list.append(label)
        n_list.append(name)
        count += 1
        if count % 1000 == 0:
            print "\t[INFO] count %s" % count
    print "\t[INFO] count %s" % count

    np.savez(out_dir, f_list=f_list, l_list=l_list, n_list=n_list)
    res_data = np.load(out_dir)
    print "[INFO] 最终数据: %s %s, %s %s, %s %s" % (
        'f_list', res_data['f_list'].shape, 'l_list', res_data['l_list'].shape, 'n_list', res_data['n_list'].shape)
    print "[INFO] 最终特征: %s, %s, %s" % (res_data['f_list'][0], res_data['l_list'][0], res_data['n_list'][0])
    print "[INFO] 结束时间: %s" % time_2_readable(time.time())
    elapsed_time = datetime.now() - start_time  # 终止时间
    print "[INFO] 耗时: %s (秒)" % elapsed_time


def merge_data():
    """
    合并数据，npz格式，f_list是特征矩阵，l_list是标签列表，n_list是名称列表
    """
    train_path = os.path.join(ROOT_DIR, 'experiments', 'npy_data', 'train')
    train_out = os.path.join(ROOT_DIR, "experiments", "data_train_v2.npz")
    mp_data_merger(data_dir=train_path, out_dir=train_out)

    # test_path = os.path.join(ROOT_DIR, 'experiments', 'npy_data', 'test')
    # test_out = os.path.join(ROOT_DIR, "experiments", "data_test_v2.npz")
    # mp_data_merger(data_dir=test_path, out_dir=test_out)


if __name__ == '__main__':
    merge_data()
