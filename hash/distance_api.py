#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/5/8
"""

import os
import sys

import numpy as np

from keras.models import load_model

p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)

import librosa
from datetime import datetime
from root_dir import ROOT_DIR, O_DIM
from data_loaders.data_augment import get_feature
from models.triplet_model import TripletModel

from utils.utils import sort_two_list


class DistanceApi(object):
    def __init__(self):
        self.b_list, self.l_list, self.n_list = self.load_data()
        pass

    @staticmethod
    def load_data():
        file_name = 'data_test_v2.bin.npz'
        print('模型名称: %s' % file_name)
        data_path = os.path.join(ROOT_DIR, 'experiments', file_name)
        data_all = np.load(data_path)
        b_list = data_all['b_list']
        l_list = data_all['l_list']
        n_list = data_all['n_list']
        return b_list, l_list, n_list

    def distance(self, audio_id):
        # 获取索引ID
        i_name = audio_id + '.npy'
        i_id = np.where(self.n_list == i_name)
        i_id = int(i_id[0])  # 索引ID

        print bin(self.b_list[i_id])

        def hamdist_for(data):  # Hamming距离
            return self.hamdist(self.b_list[i_id], data)

        start_time = datetime.now()  # 起始时间
        b_list_dist = [hamdist_for(x) for x in list(self.b_list)]
        elapsed_time = (datetime.now() - start_time).total_seconds()
        run_num = self.b_list.shape[0]
        tps = float(run_num) / float(elapsed_time)
        print "[INFO] Num: %s, Time: %s s, TPS: %0.0f (%s ms)" % (run_num, elapsed_time, tps, (1 / tps * 1000))

        sb_list, sn_list = sort_two_list(list(b_list_dist), list(self.n_list))
        return sb_list[0:20], sn_list[0:20]

    def distance_for_mp3(self, mp3_path):
        model_path = os.path.join(ROOT_DIR, "experiments/music_tl_v2/checkpoints", "triplet_loss_model_35_0.9955.h5")
        model = load_model(model_path, custom_objects={'triplet_loss': TripletModel.triplet_loss})

        # features = np.load('./993238670.npy')
        start_time = datetime.now()  # 起始时间
        y, sr = librosa.load(mp3_path)
        features = get_feature(y, sr)
        features = np.reshape(features, (1, 32, 256))
        features = np.transpose(features, [0, 2, 1])
        elapsed_time = (datetime.now() - start_time).total_seconds()
        tps = float(1.0) / float(elapsed_time)
        print "Time: %s s, TPS: %0.0f (%s ms)" % (elapsed_time, tps, (1 / tps * 1000))

        # file_name = 'data_test.npz'
        # data_path = os.path.join(ROOT_DIR, 'experiments', file_name)
        # data_all = np.load(data_path)
        # X_test2 = data_all['f_list']

        X = {
            'anc_input': features,
            'pos_input': np.zeros(features.shape),
            'neg_input': np.zeros(features.shape)
        }

        res = model.predict(X)
        data_prop = np.squeeze(res[:, :O_DIM])
        oz_arr = np.where(data_prop >= 0.0, 1.0, 0.0).astype(int)
        input_b = self.to_binary(oz_arr)

        print bin(input_b)

        def hamdist_for(o_data):  # Hamming距离
            return self.hamdist(input_b, o_data)

        b_list_dist = [hamdist_for(x) for x in list(self.b_list)]
        sb_list, sn_list = sort_two_list(list(b_list_dist), list(self.n_list))
        return sb_list[0:20], sn_list[0:20]

    @staticmethod
    def hamdist(a, b):
        # 较慢
        # bits=128
        # x = (a ^ b) & ((1 << bits) - 1)
        # ans = 0
        # while x:
        #     ans += 1
        #     x &= x - 1
        # return ans
        return bin(a ^ b).count('1')  # 更快

    @staticmethod
    def to_binary(bit_list):
        out = long(0)  # 必须指定为long，否则存储过少
        # out = 0  # 必须指定为long，否则存储过少
        for bit in bit_list:
            out = (out << 1) | bit
        return out


def test_of_distance():
    da = DistanceApi()
    audio_name = '993238670'
    print('[INFO] 目标音频: %s' % audio_name)
    rb_list, rn_list = da.distance(audio_name)
    print('[INFO] 距离: %s' % rb_list)
    print('[INFO] 相似: %s' % rn_list)


def test_of_mp3():
    mp3_path = os.path.join(ROOT_DIR, 'experiments/raw_data/train', '993238670_18.20.mp3')
    da = DistanceApi()
    print('[INFO] 目标音频: %s' % mp3_path)
    rb_list, rn_list = da.distance_for_mp3(mp3_path)
    print('[INFO] 距离: %s' % rb_list)
    print('[INFO] 相似: %s' % rn_list)


if __name__ == '__main__':
    test_of_distance()
    test_of_mp3()
