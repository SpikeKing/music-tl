#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/5/8
"""

import os
import sys
import numpy as np

p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)

from datetime import datetime
from root_dir import ROOT_DIR

from utils.utils import sort_two_list


class DistanceApi(object):
    def __init__(self):
        self.b_list, self.l_list, self.n_list = self.load_data()
        pass

    @staticmethod
    def load_data():
        file_name = 'data_test.bin.npz'
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


if __name__ == '__main__':
    da = DistanceApi()
    audio_name = '926738397'
    print('[INFO] 目标音频: %s' % audio_name)
    rb_list, rn_list = da.distance(audio_name)
    print('[INFO] 距离: %s' % rb_list)
    print('[INFO] 相似: %s' % rn_list)
