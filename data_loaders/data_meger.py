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
    return path.split('/')[-1].split('-')[0].split('.')[0]


def get_path_name(path):
    return path.split('/')[-1].split('-')[0]


def get_label_dict(path_list):
    """
    标签字典
    """
    label_dict = collections.defaultdict(int)
    for path in path_list:
        clz_name = get_path_clz(path)
        label_dict[clz_name] += 1
    for key in label_dict.keys():
        if label_dict[key] < 19:
            print '[INFO] 类别: %s, 样本: %s' % (key, label_dict[key])
            label_dict.pop(key, None)

    label_list = sorted(list(label_dict.keys()))
    res_dict = dict()
    for index, label in enumerate(label_list):
        res_dict[label] = index
    return res_dict


def func(path, label):
    """
    子进程
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


def merge_data(data_path, n_prc=40):
    start_time = datetime.now()  # 起始时间
    print "[INFO] 当前时间: %s" % timestamp_2_readable(time.time())
    path_list, _ = traverse_dir_files(data_path)  # 路径列表
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
        results.append(p.apply_async(func, args=(path, label,)))
        count += 1
        if count == 19 * 400:
            break

    count = 0
    f_list, l_list, n_list = [], [], []
    print len(results)
    for i in results:
        features, label, name = i.get()
        if not name or label == -1 or check_error_features(features):
            print "[INFO] 错误数据"
            continue
        f_list.append(features)
        l_list.append(label)
        n_list.append(name)
        count += 1
        if count % 1000 == 0:
            print "\t[INFO] count %s" % count
    print "\t[INFO] count %s" % count

    o_path = os.path.join(ROOT_DIR, "experiments", "data_train_400.npz")
    np.savez(o_path, f_list=f_list, l_list=l_list, n_list=n_list)
    res_data = np.load(o_path)
    print "[INFO] 最终数据: %s %s, %s %s, %s %s" % (
        'f_list', res_data['f_list'].shape, 'l_list', res_data['l_list'].shape, 'n_list', res_data['n_list'].shape)
    print "[INFO] 最终特征: %s, %s, %s" % (res_data['f_list'][0], res_data['l_list'][0], res_data['n_list'][0])
    print "[INFO] 结束时间: %s" % timestamp_2_readable(time.time())
    elapsed_time = datetime.now() - start_time  # 终止时间
    print "[INFO] 耗时: %s (秒)" % elapsed_time


def check_error_features(features):
    if np.isnan(features).sum() > 0:
        return True
    elif is_same_line(features):
        return True
    return False


if __name__ == '__main__':
    train_path = os.path.join(ROOT_DIR, 'experiments', 'npy_data', 'train')
    merge_data(data_path=train_path)
