#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/5/2
"""

import cv2
import librosa
import numpy as np
import os
import sys

p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)

from multiprocessing.pool import ThreadPool
from pyAudioAnalysis import audioFeatureExtraction
from root_dir import ROOT_DIR
from utils.utils import *


def audio_slice(y, sr, name_id, folder):
    # name = file_path.split('/')[-1]
    # folder = file_path.replace(name, '')
    # print '【INFO] 文件名: %s' % name
    # print '【INFO] 文件夹: %s' % folder
    # y, sr = librosa.load(file_path)  # 读取音频
    n_part = len(y) / 5
    # file1 = os.path.join(folder, name + '.slice_first.mp3')
    # wavfile.write(file1, sr, y[:n_part * 3])  # 写入音频
    file1 = os.path.join(folder, name_id + '.slice_first' + '.npy')
    np.save(file1, get_feature(y[:n_part * 3], sr))

    # file2 = os.path.join(folder, name + '.slice_middle.mp3')
    # wavfile.write(file2, sr, y[n_part * 1:n_part * 4])  # 写入音频
    file2 = os.path.join(folder, name_id + '.slice_middle' + '.npy')
    np.save(file2, get_feature(y[n_part * 1:n_part * 4], sr))

    # file3 = os.path.join(folder, name + '.slice_last.mp3')
    # wavfile.write(file3, sr, y[n_part * 2:])  # 写入音频
    file3 = os.path.join(folder, name_id + '.slice_last' + '.npy')
    np.save(file3, get_feature(y[n_part * 2:], sr))


def audio_roll(y, sr, name_id, folder):
    # name = file_path.split('/')[-1]
    # folder = file_path.replace(name, '')
    # print '【INFO] 文件名: %s' % name
    # print '【INFO] 文件夹: %s' % folder
    # y, sr = librosa.load(file_path)  # 读取音频

    n_part = len(y) / 4

    # file1 = os.path.join(folder, name + '.roll_1.mp3')
    # wavfile.write(file1, sr, np.roll(y, n_part))  # 写入音频
    file1 = os.path.join(folder, name_id + '.roll_1' + '.npy')
    np.save(file1, get_feature(np.roll(y, n_part), sr))

    # file2 = os.path.join(folder, name + '.roll_2.mp3')
    # wavfile.write(file2, sr, np.roll(y, n_part * 2))  # 写入音频
    file2 = os.path.join(folder, name_id + '.roll_2' + '.npy')
    np.save(file2, get_feature(np.roll(y, n_part * 2), sr))

    # file3 = os.path.join(folder, name + '.roll_3.mp3')
    # wavfile.write(file3, sr, np.roll(y, n_part * 3))  # 写入音频
    file3 = os.path.join(folder, name_id + '.roll_3' + '.npy')
    np.save(file3, get_feature(np.roll(y, n_part * 3), sr))


def audio_tune(y, sr, name_id, folder):
    # name = file_path.split('/')[-1]
    # folder = file_path.replace(name, '')
    # print '【INFO] 文件名: %s' % name
    # print '【INFO] 文件夹: %s' % folder
    # y, sr = librosa.load(file_path)  # 读取音频

    tune1 = cv2.resize(y, (1, int(len(y) * 1.05))).squeeze()
    # file1 = os.path.join(folder, name + '.tune_5.mp3')
    # wavfile.write(file1, sr, y_tune)  # 写入音频
    file1 = os.path.join(folder, name_id + '.tune_5' + '.npy')
    np.save(file1, get_feature(tune1, sr))

    tune2 = cv2.resize(y, (1, int(len(y) * 1.10))).squeeze()
    # file2 = os.path.join(folder, name + '.tune_10.mp3')
    # wavfile.write(file2, sr, y_tune)  # 写入音频
    file2 = os.path.join(folder, name_id + '.tune_10' + '.npy')
    np.save(file2, get_feature(tune2, sr))

    tune3 = cv2.resize(y, (1, int(len(y) * 1.15))).squeeze()
    # file3 = os.path.join(folder, name + '.tune_15.mp3')
    # wavfile.write(file3, sr, tune1)  # 写入音频
    file3 = os.path.join(folder, name_id + '.tune_15' + '.npy')
    np.save(file3, get_feature(tune3, sr))


def audio_noise(y, sr, name_id, folder):
    # name = file_path.split('/')[-1]
    # folder = file_path.replace(name, '')
    # print '【INFO] 文件名: %s' % name
    # print '【INFO] 文件夹: %s' % folder
    # y, sr = librosa.load(file_path)  # 读取音频

    np.random.seed(seed=47)
    wn = np.random.randn(len(y))

    yn1 = np.where(y != 0.0, y + 0.01 * wn, 0.0)
    # file1 = os.path.join(folder, name + '.noise_1.mp3')
    # wavfile.write(file1, sr, yn1)  # 写入音频
    file1 = os.path.join(folder, name_id + '.noise_1' + '.npy')
    np.save(file1, get_feature(yn1, sr))

    yn2 = np.where(y != 0.0, y + 0.02 * wn, 0.0)
    # file2 = os.path.join(folder, name + '.noise_2.mp3')
    # wavfile.write(file2, sr, yn2)  # 写入音频
    file2 = os.path.join(folder, name_id + '.noise_2' + '.npy')
    np.save(file2, get_feature(yn2, sr))

    yn3 = np.where(y != 0.0, y + 0.03 * wn, 0.0)
    # file3 = os.path.join(folder, name + '.noise_3.mp3')
    # wavfile.write(file3, sr, yn3)  # 写入音频
    file3 = os.path.join(folder, name_id + '.noise_3' + '.npy')
    np.save(file3, get_feature(yn3, sr))


def audio_high(y, sr, name_id, folder):
    # name = file_path.split('/')[-1]
    # folder = file_path.replace(name, '')
    # print '【INFO] 文件名: %s' % name
    # print '【INFO] 文件夹: %s' % folder
    # y, sr = librosa.load(file_path)  # 读取音频

    yn1 = np.where(y != 0.0, y * 1.05, 0.0)
    # file1 = os.path.join(folder, name + '.high_5.mp3')
    # wavfile.write(file1, sr, yn1)  # 写入音频
    file1 = os.path.join(folder, name_id + '.high_5' + '.npy')
    np.save(file1, get_feature(yn1, sr))

    yn2 = np.where(y != 0.0, y * 1.10, 0.0)
    # file2 = os.path.join(folder, name + '.high_10.mp3')
    # wavfile.write(file2, sr, yn2)  # 写入音频
    file2 = os.path.join(folder, name_id + '.high_10' + '.npy')
    np.save(file2, get_feature(yn2, sr))

    yn3 = np.where(y != 0.0, y * 1.15, 0.0)
    # file3 = os.path.join(folder, name + '.high_15.mp3')
    # wavfile.write(file3, sr, yn3)  # 写入音频
    file3 = os.path.join(folder, name_id + '.high_15' + '.npy')
    np.save(file3, get_feature(yn3, sr))


def audio_low(y, sr, name_id, folder):
    # name = file_path.split('/')[-1]
    # folder = file_path.replace(name, '')
    # print '【INFO] 文件名: %s' % name
    # print '【INFO] 文件夹: %s' % folder
    # y, sr = librosa.load(file_path)  # 读取音频

    yn1 = np.where(y != 0.0, y * 0.95, 0.0)
    # file1 = os.path.join(folder, name + '.low_5.mp3')
    # wavfile.write(file1, sr, yn1)  # 写入音频
    file1 = os.path.join(folder, name_id + '.low_5' + '.npy')
    np.save(file1, get_feature(yn1, sr))

    yn2 = np.where(y != 0.0, y * 0.90, 0.0)
    # file2 = os.path.join(folder, name + '.low_10.mp3')
    # wavfile.write(file2, sr, yn2)  # 写入音频
    file2 = os.path.join(folder, name_id + '.low_10' + '.npy')
    np.save(file2, get_feature(yn2, sr))

    yn3 = np.where(y != 0.0, y * 0.85, 0.0)
    # file3 = os.path.join(folder, name + '.low_15.mp3')
    # wavfile.write(file3, sr, yn3)  # 写入音频
    file3 = os.path.join(folder, name_id + '.low_15' + '.npy')
    np.save(file3, get_feature(yn3, sr))


def save_features(params):
    file_path, name_id, folder = params
    try:
        print '[INFO] ' + name_id
        y, sr = librosa.load(file_path)
        saved_path = os.path.join(folder, name_id + '.npy')
        np.save(saved_path, get_feature(y, sr))

        audio_slice(y, sr, name_id, folder)
        audio_roll(y, sr, name_id, folder)
        audio_tune(y, sr, name_id, folder)
        audio_noise(y, sr, name_id, folder)
        audio_high(y, sr, name_id, folder)
        audio_low(y, sr, name_id, folder)
    except Exception as e:
        print '[Exception] %s' % e


def generate_npy_data(tn=40):
    print "[INFO] 特征提取开始"
    npy_folder = os.path.join(ROOT_DIR, 'experiments', 'npy_data')
    mkdir_if_not_exist(npy_folder)
    npy_train = os.path.join(npy_folder, 'train')
    npy_test = os.path.join(npy_folder, 'test')
    mkdir_if_not_exist(npy_train)
    mkdir_if_not_exist(npy_test)

    raw_train = os.path.join(ROOT_DIR, 'experiments', 'raw_data', 'train')
    paths, names = traverse_dir_files(raw_train)
    param_list = []
    for path, name in zip(paths, names):
        name_id = name.split('_')[0]
        param_list.append((path, name_id, npy_train))
    print "[INFO] 训练数据: %s" % len(param_list)
    pool = ThreadPool(tn)
    pool.map(save_features, param_list)
    pool.close()
    pool.join()

    raw_test = os.path.join(ROOT_DIR, 'experiments', 'raw_data', 'test')
    paths, names = traverse_dir_files(raw_test)
    param_list = []
    for path, name in zip(paths, names):
        name_id = name.split('_')[0]
        param_list.append((path, name_id, npy_test))
    print "[INFO] 测试数据: %s" % len(param_list)
    pool = ThreadPool(tn)
    pool.map(save_features, param_list)
    pool.close()
    pool.join()

    print "[INFO] 特征提取结束"


def get_feature(y, sr, dim=256):
    """
    提取音频特征
    """
    f_ws = len(y) / (dim + 4)  # 使用固定的窗口大小, 1000~1001维
    features = audioFeatureExtraction.stFeatureExtraction(y, sr, f_ws, f_ws)
    features = features[1:33][:, :256]  # 定向选择特征
    return features


if __name__ == '__main__':
    generate_npy_data()
