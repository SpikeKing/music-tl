#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/5/8
"""
import os
import sys

import numpy as np
import mxnet as mx

from models.triplet_model_mxnet import TripletModelMxnet

p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)

from root_dir import ROOT_DIR, O_DIM


class HashPreProcessor(object):
    def __init__(self):
        self.model = None
        pass

    def process(self):
        print('[INFO] 转换开始')
        ctx = mx.cpu(0)
        self.model = TripletModelMxnet.deep_conv_lstm()
        params = os.path.join(ROOT_DIR, "experiments/music_tl_v2/checkpoints", "triplet_loss_model_3_0.7750.params")
        self.model.load_params(params, ctx=ctx)

        file_name = 'data_train_v2.npz'
        data_path = os.path.join(ROOT_DIR, 'experiments', file_name)
        data_all = np.load(data_path)
        X_test1 = data_all['f_list']
        l_list1 = data_all['l_list']
        n_list1 = data_all['n_list']

        print('[INFO] X_test1.shape: ' + str(X_test1.shape))

        file_name = 'data_test_v2.npz'
        data_path = os.path.join(ROOT_DIR, 'experiments', file_name)
        data_all = np.load(data_path)
        X_test2 = data_all['f_list']
        l_list2 = data_all['l_list']
        n_list2 = data_all['n_list']
        print('[INFO] X_test2.shape: ' + str(X_test2.shape))

        X_test = np.concatenate((X_test1, X_test2), axis=0)
        l_list = np.concatenate((l_list1, l_list2), axis=0)
        n_list = np.concatenate((n_list1, n_list2), axis=0)

        print('[INFO] X_test.shape: ' + str(X_test.shape))

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
        X_test = mx.nd.array(X_test).as_in_context(ctx)
        print('[INFO] 输入结构: %s' % str(X_test.shape))
        res = self.model(X_test)
        print('[INFO] 输出结构: %s' % str(res.shape))
        data = res.asnumpy()
        oz_arr = np.where(data >= 0.0, 1.0, 0.0).astype(int)
        print oz_arr[0]
        # print np.sum(oz_arr, axis=1)  # 测试分布
        oz_bin = np.apply_along_axis(self.to_binary, axis=1, arr=oz_arr)
        print('[INFO] oz_bin: %s' % oz_bin[0])

        out_path = os.path.join(ROOT_DIR, 'experiments', file_name.replace('.npz', '') + ".bin.mx.npz")
        np.savez(out_path, b_list=oz_bin, l_list=l_list, n_list=n_list)

        print('[INFO] 输出示例: %s %s %s' % (str(oz_bin.shape), bin(oz_bin[0]), oz_bin[0]))
        print('[INFO] 转换结束')

    @staticmethod
    def to_binary(bit_list):
        out = long(0)  # 必须指定为long，否则存储过少
        # out = 0  # 必须指定为long，否则存储过少
        for bit in bit_list:
            out = (out << 1) | bit
        return out


if __name__ == '__main__':
    hpp = HashPreProcessor()
    hpp.process()
