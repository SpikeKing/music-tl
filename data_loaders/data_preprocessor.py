#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/4/27

音频特征，每个音频提取32组256维的特征，组成8个通道；

音频位置：/data2/ych/mp3
"""
import collections
import time
import librosa
import numpy as np
import sys
import os

p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)

from multiprocessing import Pool
from pyAudioAnalysis import audioFeatureExtraction
from root_dir import ROOT_DIR
from utils.utils import *


def load_music(path):
    """
    加载音频
    """
    y, sr = librosa.load(path)
    return y, sr


def get_feature(y, sr, dim=256):
    """
    提取音频特征
    """
    f_ws = len(y) / (dim + 4)  # 使用固定的窗口大小, 1000~1001维
    return audioFeatureExtraction.stFeatureExtraction(y, sr, f_ws, f_ws)


def audio_to_feature(path):
    """
    音频转特征
    """
    y, sr = load_music(path)
    features = get_feature(y, sr)
    name = path.split('/')[-1]
    features = features[1:33][:, :256]  # 定向选择特征
    return features, name


def get_label_dict(path_list):
    """
    标签字典
    """
    label_dict = collections.defaultdict(int)
    for path in path_list:
        clz_name = get_path_name(path)
        label_dict[clz_name] += 1
    for key in label_dict.keys():
        if label_dict[key] <= 2:
            print '[INFO] 类别: %s, 样本: %s' % (key, label_dict[key])
            label_dict.pop(key, None)

    label_list = sorted(list(label_dict.keys()))
    res_dict = dict()
    for index, label in enumerate(label_list):
        res_dict[label] = index
    return res_dict


def get_path_name(path):
    return path.split('/')[-1].split('-')[0]


def func(path, label):
    """
    子进程
    """
    try:
        print '[INFO] ' + path
        features, name = audio_to_feature(path)
        if np.min(features) == np.max(features):
            raise Exception('features error')
        return features, label, name
    except Exception as e:
        print "[INFO] 错误: %s" % str(e)
        return np.array([]), -1, None


def run_multiprocess(data_path):
    """
    预处理特征的多进程模型
    """
    start_time = datetime.now()  # 起始时间
    print "[INFO] 当前时间: %s" % timestamp_2_readable(time.time())

    p = Pool(processes=40)  # 进程数尽量与核数匹配
    results = []
    path_list, _ = traverse_dir_files(data_path)  # 路径列表
    label_dict = get_label_dict(path_list)
    print "[INFO] 音频总数: %s" % len(path_list)
    for path in path_list:
        clz_name = get_path_name(path)
        if clz_name not in label_dict:
            print "[INFO] %s 类别样本过少! " % clz_name
            continue
        label = label_dict[clz_name]
        results.append(p.apply_async(func, args=(path, label,)))
    p.close()
    p.join()

    count = 0
    f_list, l_list, n_list = [], [], []
    print len(results)
    for i in results:
        features, label, name = i.get()
        if not name or label == -1:
            print "[INFO] 错误数据"
            continue
        f_list.append(features)
        l_list.append(label)
        n_list.append(name)
        count += 1
        if count % 40 == 0:
            print "\t[INFO] count %s" % count
    print "\t[INFO] count %s" % count
    o_path = os.path.join(ROOT_DIR, "experiments", "music_data_train.npz")
    np.savez(o_path, f_list=f_list, l_list=l_list, n_list=n_list)
    print "[INFO] 写入音频特征完成"
    print "[INFO] 结束时间: %s" % timestamp_2_readable(time.time())
    elapsed_time = datetime.now() - start_time  # 终止时间
    print "[INFO] 总耗时: %s" % str(elapsed_time)
    res_data = np.load(o_path)
    print "[INFO] 最终数据: %s %s, %s %s, %s %s" % (
        'f_list', res_data['f_list'].shape, 'l_list', res_data['l_list'].shape, 'n_list', res_data['n_list'].shape)
    print "[INFO] 最终特征: %s, %s, %s" % (res_data['f_list'][0], res_data['l_list'][0], res_data['n_list'][0])


def run_oneprocess(data_path):
    start_time = datetime.now()  # 起始时间
    print "[INFO] 当前时间: %s" % timestamp_2_readable(time.time())

    results = []
    path_list, _ = traverse_dir_files(data_path)  # 路径列表
    label_dict = get_label_dict(path_list)
    print "[INFO] 音频总数: %s" % len(path_list)
    for path in path_list:
        clz_name = get_path_name(path)
        if clz_name not in label_dict:
            print "[INFO] %s 类别样本过少! " % clz_name
            continue
        label = label_dict[clz_name]
        results.append(func(path, label))

    count = 0
    f_list, l_list, n_list = [], [], []
    print len(results)
    for result in results:
        features, label, name = result
        if not name or label == -1:
            print "[INFO] 错误数据"
            continue
        f_list.append(features)
        l_list.append(label)
        n_list.append(name)
        count += 1
        if count % 40 == 0:
            print "\t[INFO] count %s" % count
    print "\t[INFO] count %s" % count
    o_path = os.path.join(ROOT_DIR, "experiments", "music_data_train.npz")
    np.savez(o_path, f_list=f_list, l_list=l_list, n_list=n_list)
    print "[INFO] 写入音频特征完成"
    print "[INFO] 结束时间: %s" % timestamp_2_readable(time.time())
    elapsed_time = datetime.now() - start_time  # 终止时间
    print "[INFO] 总耗时: %s" % str(elapsed_time)
    res_data = np.load(o_path)
    print "[INFO] 最终数据: %s %s, %s %s, %s %s" % (
        'f_list', res_data['f_list'].shape, 'l_list', res_data['l_list'].shape, 'n_list', res_data['n_list'].shape)
    print "[INFO] 最终特征: %s, %s, %s" % (res_data['f_list'][0], res_data['l_list'][0], res_data['n_list'][0])


def main_test_1():
    """
    主要测试
    """
    path = os.path.join(ROOT_DIR, 'data')
    features, name = audio_to_feature(path)
    print features.shape
    print name


def main_test_2():
    """
    主要测试
    """
    path = os.path.join(ROOT_DIR, '/Users/wang/Desktop/same_music/train')
    # run_multiprocess('/data2/wcl1/data/same_music/train')
    run_oneprocess(path)


if __name__ == '__main__':
    main_test_2()
