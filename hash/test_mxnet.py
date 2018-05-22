#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/5/22
"""
import os
import sys
import librosa
import mxnet as mx
import numpy as np
from datetime import datetime

p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)

from models.triplet_model_mxnet import TripletModelMxnet
from data_loaders.data_augment import get_feature
from root_dir import ROOT_DIR


def test_load_audio(mp3_path):
    num = 100

    print('[统计开始]')
    y_o, sr = None, None
    start_time = datetime.now()  # 起始时间
    for i in range(num):
        y_o, sr = librosa.load(mp3_path)
    elapsed_time = (datetime.now() - start_time).total_seconds()
    tps = float(num) / float(elapsed_time)
    print "[加载音频] Time: %s s, TPS: %0.4f (%s ms)" % (elapsed_time, tps, (1 / tps * 1000))

    start_time = datetime.now()  # 起始时间
    y = None
    for i in range(num):
        y, _ = librosa.effects.trim(y_o, top_db=40)  # 去掉空白部分
    elapsed_time = (datetime.now() - start_time).total_seconds()
    tps = float(num) / float(elapsed_time)
    print "[去掉留白] Time: %s s, TPS: %0.4f (%s ms)" % (elapsed_time, tps, (1 / tps * 1000))

    start_time = datetime.now()  # 起始时间
    features = None
    for i in range(num):
        features = get_feature(y, sr)
    elapsed_time = (datetime.now() - start_time).total_seconds()
    tps = float(num) / float(elapsed_time)
    print "[提取特征] Time: %s s, TPS: %0.4f (%s ms)" % (elapsed_time, tps, (1 / tps * 1000))

    ctx = mx.gpu(0)
    model = TripletModelMxnet.deep_conv_lstm()
    params = os.path.join(ROOT_DIR, "experiments/music_tl_v2/checkpoints", "triplet_loss_model_10_0.9893.params")
    print('[INFO] 模型: %s' % params)
    model.load_params(params, ctx=ctx)

    features = np.reshape(features, (1, 32, 256))
    features = np.transpose(features, [0, 2, 1])
    features = mx.nd.array(features).as_in_context(ctx)
    start_time = datetime.now()  # 起始时间
    for i in range(num):
        model(features)
    elapsed_time = (datetime.now() - start_time).total_seconds()
    tps = float(num) / float(elapsed_time)
    print "[预测] Time: %s s, TPS: %0.4f (%s ms)" % (elapsed_time, tps, (1 / tps * 1000))


if __name__ == '__main__':
    path = os.path.join(ROOT_DIR, 'experiments/raw_data/train', '992488609_15.12.mp3')
    test_load_audio(path)
