#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/5/2

音频数据扩充（Data Augment），存储为32*dim的npy格式
"""

import os
import sys
from multiprocessing import Pool

import librosa
import numpy as np

p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)

from utils.np_utils import check_error_features
from pyAudioAnalysis import audioFeatureExtraction
from root_dir import ROOT_DIR
from utils.utils import *


def audio_slice(y, sr, name_id, folder):
    """
    音频剪裁，平分5段，
    1. 剪裁：[0:3]，[1:4]，[2:5]
    2. 子集：[2], [3], [4]
    3. 超集：2倍，3倍

    生成11个扩充数据
    """
    n_part = len(y) / 5

    file1 = os.path.join(folder, name_id + '.slice_first' + '.npy')
    np.save(file1, get_feature(y[:n_part * 3], sr))

    file2 = os.path.join(folder, name_id + '.slice_middle' + '.npy')
    np.save(file2, get_feature(y[n_part * 1:n_part * 4], sr))

    file3 = os.path.join(folder, name_id + '.slice_last' + '.npy')
    np.save(file3, get_feature(y[n_part * 2:], sr))

    file4 = os.path.join(folder, name_id + '.large_2' + '.npy')
    np.save(file4, get_feature(np.tile(y, 2), sr))

    file5 = os.path.join(folder, name_id + '.large_3' + '.npy')
    np.save(file5, get_feature(np.tile(y, 3), sr))

    file8 = os.path.join(folder, name_id + '.tiny_2' + '.npy')
    np.save(file8, get_feature(y[n_part:n_part * 2], sr))

    file9 = os.path.join(folder, name_id + '.tiny_3' + '.npy')
    np.save(file9, get_feature(y[n_part * 2:n_part * 3], sr))

    file10 = os.path.join(folder, name_id + '.tiny_4' + '.npy')
    np.save(file10, get_feature(y[n_part * 3:n_part * 4], sr))


def audio_roll(y, sr, name_id, folder):
    """
    音频旋转，平分4段，旋转分为三种，[2, 0, 1]，[1, 2, 0]

    生成3个扩充数据
    """
    n_part = len(y) / 3

    file1 = os.path.join(folder, name_id + '.roll_1' + '.npy')
    np.save(file1, get_feature(np.roll(y, n_part), sr))

    file2 = os.path.join(folder, name_id + '.roll_2' + '.npy')
    np.save(file2, get_feature(np.roll(y, n_part * 2), sr))


def audio_tune(y, sr, name_id, folder):
    """
    音频调音，调音分为三种，拉长为30%，50%，100%
    """
    y_fast1 = librosa.effects.time_stretch(y, 1.3)
    file1 = os.path.join(folder, name_id + '.fast_3' + '.npy')
    np.save(file1, get_feature(y_fast1, sr))
    # librosa.output.write_wav(os.path.join(folder, name_id + '.fast_3' + '.mp3'), y_fast1, sr)

    y_fast2 = librosa.effects.time_stretch(y, 1.5)
    file2 = os.path.join(folder, name_id + '.fast_5' + '.npy')
    np.save(file2, get_feature(y_fast2, sr))
    # librosa.output.write_wav(os.path.join(folder, name_id + '.fast_5' + '.mp3'), y_fast2, sr)

    y_slow1 = librosa.effects.time_stretch(y, 0.9)
    file4 = os.path.join(folder, name_id + '.slow_1' + '.npy')
    np.save(file4, get_feature(y_slow1, sr))
    # librosa.output.write_wav(os.path.join(folder, name_id + '.slow_1' + '.mp3'), y_slow1, sr)

    y_slow3 = librosa.effects.time_stretch(y, 0.7)
    file6 = os.path.join(folder, name_id + '.slow_3' + '.npy')
    np.save(file6, get_feature(y_slow3, sr))
    # librosa.output.write_wav(os.path.join(folder, name_id + '.slow_3' + '.mp3'), y_slow3, sr)


def audio_noise(y, sr, name_id, folder):
    """
    音频噪声，添加高斯噪声，噪声分为三种，添加1%，2%，3%
    """
    np.random.seed(seed=47)
    wn = np.random.randn(len(y))

    yn2 = np.where(y != 0.0, y + 0.02 * wn, 0.0)
    file2 = os.path.join(folder, name_id + '.noise_2' + '.npy')
    np.save(file2, get_feature(yn2, sr))

    yn3 = np.where(y != 0.0, y + 0.03 * wn, 0.0)
    file3 = os.path.join(folder, name_id + '.noise_3' + '.npy')
    np.save(file3, get_feature(yn3, sr))


def audio_high(y, sr, name_id, folder):
    """
    音频高音，提高音调，高音分为三种，添加5%，10%，15%
    """
    yh1 = librosa.effects.pitch_shift(y, sr, n_steps=2.0)
    file1 = os.path.join(folder, name_id + '.high_2' + '.npy')
    np.save(file1, get_feature(yh1, sr))
    # librosa.output.write_wav(os.path.join(folder, name_id + '.high_2' + '.mp3'), yh1, sr)

    yh2 = librosa.effects.pitch_shift(y, sr, n_steps=4.0)
    file2 = os.path.join(folder, name_id + '.high_4' + '.npy')
    np.save(file2, get_feature(yh2, sr))
    # librosa.output.write_wav(os.path.join(folder, name_id + '.high_4' + '.mp3'), yh2, sr)

    yl1 = librosa.effects.pitch_shift(y, sr, n_steps=-2.0)  # 降调
    file4 = os.path.join(folder, name_id + '.low_2' + '.npy')
    np.save(file4, get_feature(yl1, sr))
    # librosa.output.write_wav(os.path.join(folder, name_id + '.low_2' + '.mp3'), yl1, sr)

    yl2 = librosa.effects.pitch_shift(y, sr, n_steps=-4.0)  # 降调
    file5 = os.path.join(folder, name_id + '.low_4' + '.npy')
    np.save(file5, get_feature(yl2, sr))
    # librosa.output.write_wav(os.path.join(folder, name_id + '.low_4' + '.mp3'), yl2, sr)


def get_feature(y, sr, dim=256):
    """
    计算音频的特征值

    :param y: 音频帧
    :param sr: 音频帧率
    :param dim: 音频特征长度
    :return: (32, sample_bin)
    """
    hop_length = len(y) / (dim + 2) / 64 * 64  # 频率距离需要对于64取模

    # 32维特征值
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13)  # 13dim
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)  # 12dim
    rmse = librosa.feature.rmse(y=y, hop_length=hop_length)  # 1dim
    sp_ce = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)  # 1dim
    sp_cf = librosa.feature.spectral_flatness(y=y, hop_length=hop_length)  # 1dim
    sp_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length)  # 1dim
    zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=hop_length)  # 1dim
    poly = librosa.feature.poly_features(y=y, sr=sr, hop_length=hop_length)  # 2dim

    all_features = np.vstack([mfcc, chroma, rmse, sp_ce, sp_cf, sp_bw, zcr, poly])[:, :dim]
    # print all_features.shape
    return all_features


def get_feature_old(y, sr, dim=256):
    """
    特征属性:
    1-Zero Crossing Rate: 短时平均过零率, 即每帧信号内, 信号过零点的次数, 体现的是频率特性.
    2-Energy: 短时能量, 即每帧信号的平方和, 体现的是信号能量的强弱.
    3-Entropy of Energy: 能量熵, 跟频谱的谱熵（Spectral Entropy）有点类似, 不过它描述的是信号的时域分布情况, 体现的是连续性.
    4-Spectral Centroid: 频谱中心又称为频谱一阶距, 频谱中心的值越小, 表明越多的频谱能量集中在低频范围内,
        如:voice与music相比, 通常spectral centroid较低.
    5-Spectral Spread: 频谱延展度, 又称为频谱二阶中心矩, 它描述了信号在频谱中心周围的分布状况.
    6-Spectral Entropy: 谱熵, 根据熵的特性可以知道, 分布越均匀, 熵越大, 能量熵反应了每一帧信号的均匀程度,
        如说话人频谱由于共振峰存在显得不均匀, 而白噪声的频谱就更加均匀, 借此进行VAD便是应用之一.
    7-Spectral Flux: 频谱通量, 描述的是相邻帧频谱的变化情况.
    8-Spectral Rolloff: 频谱滚降点.
    9~21-MFCCs: 就是大名鼎鼎的梅尔倒谱系数, 这个网上资料非常多, 也是非常重要的音频特征.
    22~33-Chroma Vector: 这个有12个参数, 对应就是12级音阶, 这个在音乐声里可能用的比较多.
    34-Chroma Deviation: 这个就是Chroma Vector的标准方差.
    :return: 音频特征[左通道, 右通道]
    """
    f_ws = len(y) / (dim + 4)  # 加4，避免数据不足
    features = audioFeatureExtraction.stFeatureExtraction(y, sr, f_ws, f_ws)
    features = features[1:33][:, :dim]  # 定向选择特征，（32*dim）
    return features


def generate_augment(params):
    """
    音频增强
    :param params: 参数，[文件路径，音频ID，存储文件夹]
    :return: None
    """
    file_path, name_id, folder = params
    try:
        saved_path = os.path.join(folder, name_id + '.npy')
        if os.path.exists(saved_path):
            print("[INFO] 文件 %s 存在" % name_id)
            return

        y_o, sr = librosa.load(file_path)
        y, _ = librosa.effects.trim(y_o, top_db=40)  # 去掉空白部分

        duration = len(y) / sr
        if duration < 4:  # 过滤小于3秒的音频
            print('[INFO] 音频 %s 过短: %0.4f' % (name_id, duration))
            return

        if not np.any(y):
            print('[Exception] 音频 %s 错误' % name_id)
            return

        features = get_feature(y, sr)
        if check_error_features(features):
            print('[Exception] 音频 %s 错误' % name_id)
            return

        np.save(saved_path, features)  # 存储原文件的npy

        # 20种数据增强
        audio_slice(y, sr, name_id, folder)  # 8个
        audio_roll(y, sr, name_id, folder)  # 2个
        audio_tune(y, sr, name_id, folder)  # 4个
        audio_noise(y, sr, name_id, folder)  # 2个
        audio_high(y, sr, name_id, folder)  # 4个
    except Exception as e:
        print('[Exception] %s' % e)
        return

    print '[INFO] 音频ID ' + name_id
    return


def mp_augment(raw_dir, npy_dir, n_process=40):
    """
    多进程的音频增强
    :param raw_dir: 音频文件文件夹
    :param npy_dir: npy文件的存储文件夹
    :param n_process: 进程数
    :return:
    """
    paths, names = traverse_dir_files(raw_dir)
    p = Pool(processes=n_process)  # 进程数尽量与核数匹配
    print "[INFO] 数据数: %s" % len(paths)
    for path, name in zip(paths, names):
        name_id = name.split('_')[0]
        params = (path, name_id, npy_dir)
        generate_augment(params)
        p.apply_async(generate_augment, args=(params,))
    p.close()
    p.join()


def process_audio_augment():
    """
    音频增强
    """
    print "[INFO] 特征提取开始! "

    npy_folder = os.path.join(ROOT_DIR, 'experiments', 'npy_data_v2')
    mkdir_if_not_exist(npy_folder)
    npy_train = os.path.join(npy_folder, 'train')
    npy_test = os.path.join(npy_folder, 'test')
    mkdir_if_not_exist(npy_train)
    mkdir_if_not_exist(npy_test)

    raw_train = os.path.join(ROOT_DIR, 'experiments', 'raw_data', 'train')
    raw_test = os.path.join(ROOT_DIR, 'experiments', 'raw_data', 'test')

    mp_augment(raw_train, npy_train, n_process=10)
    # mp_augment(raw_test, npy_test, n_process=10)
    n_tr, _ = traverse_dir_files(npy_train)
    n_te, _ = traverse_dir_files(npy_test)
    print('训练数据: %s' % len(n_tr))
    print('测试数据: %s' % len(n_te))

    print "[INFO] 特征提取结束! "


if __name__ == '__main__':
    process_audio_augment()
