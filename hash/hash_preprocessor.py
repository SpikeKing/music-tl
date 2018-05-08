#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/5/8
"""
import os
import sys
import numpy as np

from keras.layers import K
from keras.models import load_model

p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)

from root_dir import ROOT_DIR


class HashPreProcessor(object):
    def __init__(self):
        pass

    def process(self):
        print('[INFO] 转换开始')
        model_path = os.path.join(ROOT_DIR, "experiments/music_tl/checkpoints", "triplet_loss_model_91_0.9989.h5")
        model = load_model(model_path, custom_objects={'triplet_loss': self.triplet_loss})

        file_name = 'data_test.npz'
        data_path = os.path.join(ROOT_DIR, 'experiments', file_name)
        data_all = np.load(data_path)
        X_test = data_all['f_list']
        l_list = data_all['l_list']
        n_list = data_all['n_list']

        def split_num(name):
            name = str(name)
            return len(name.split('.'))  # 长度为2的音频是原始音频

        # 获取原始音频的索引
        n_list_r = np.reshape(n_list, (-1, 1))
        n_list_num = np.apply_along_axis(split_num, axis=1, arr=n_list_r)
        o_indexes = np.where(n_list_num == 2, True, False)  # 原始音频的索引

        n_list = n_list[o_indexes]
        l_list = l_list[o_indexes]
        X_test = X_test[o_indexes]

        print('[INFO] 转换数量: %s' % n_list.shape[0])

        X_test = np.transpose(X_test, [0, 2, 1])

        X = {
            'anc_input': X_test,
            'pos_input': np.zeros(X_test.shape),
            'neg_input': np.zeros(X_test.shape)
        }
        res = model.predict(X)
        data = res[:, :128]
        oz_arr = np.where(data >= 0.0, 1.0, 0.0).astype(int)
        # print np.sum(oz_arr, axis=1)  # 测试分布
        oz_bin = np.apply_along_axis(self.to_binary, axis=1, arr=oz_arr)

        out_path = os.path.join(ROOT_DIR, 'experiments', file_name.replace('.npz', '') + ".bin.npz")
        np.savez(out_path, b_list=oz_bin, l_list=l_list, n_list=n_list)

        print('[INFO] 输出示例: %s %s %s' % (str(oz_bin.shape), bin(oz_bin[0]), oz_bin[0]))
        print('[INFO] 转换结束')

    @staticmethod
    def triplet_loss(y_true, y_pred):
        """
        Triplet Loss的损失函数
        """
        anc, pos, neg = y_pred[:, 0:128], y_pred[:, 128:256], y_pred[:, 256:]

        # 欧式距离
        pos_dist = K.sum(K.square(anc - pos), axis=-1, keepdims=True)
        neg_dist = K.sum(K.square(anc - neg), axis=-1, keepdims=True)
        basic_loss = pos_dist - neg_dist + 10.0

        loss = K.maximum(basic_loss, 0.0)

        print "[INFO] model - triplet_loss shape: %s" % str(loss.shape)
        return loss

    @staticmethod
    def to_binary(bit_list):
        out = 0
        for bit in bit_list:
            out = (out << 1) | bit
        return out


if __name__ == '__main__':
    hpp = HashPreProcessor()
    hpp.process()
