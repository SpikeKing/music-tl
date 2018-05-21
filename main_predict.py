#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/5/8
"""
import argparse

import sys
import numpy as np

# from hash.distance_api import DistanceApi
from hash.distance_api_mxnet import DistanceApi
from utils.utils import sort_two_list


def main_predict():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '-n', '--name',
        dest='name',
        metavar='name',
        help='add a target name')
    parser.add_argument(
        '-m', '--mp3',
        dest='mp3',
        metavar='mp3',
        help='add a target mp3')
    args = parser.parse_args()
    if len(sys.argv) == 1:
        parser.print_help()
        exit(0)

    audio_name = args.name
    if audio_name:
        print('[INFO] 目标音频: %s' % audio_name)
        da = DistanceApi()
        rb_list, rn_list = da.distance(audio_name)
        rn_list = [x.replace('.npy', '') for x in list(np.squeeze(rn_list))]
        # print('[INFO] 距离: %s' % rb_list)
        # print('[INFO] 相似: %s' % rn_list)
        rb_list, rn_list = sort_two_list(rb_list, rn_list)
        print '[INFO]',
        for rb, rn in zip(rb_list, rn_list):
            print '%s-%s ' % (rb, rn),

    mp3_path = args.mp3
    if mp3_path:
        print('[INFO] 目标音频: %s' % mp3_path)
        da = DistanceApi()
        rb_list, rn_list = da.distance_for_mp3(mp3_path)
        rn_list = [x.replace('.npy', '') for x in list(np.squeeze(rn_list))]

        rb_list, rn_list = sort_two_list(rb_list, rn_list)
        print '[INFO]',
        for rb, rn in zip(rb_list, rn_list):
            print '%s-%s ' % (rb, rn),
        # print('[INFO] 距离: %s' % rb_list)
        # print('[INFO] 相似: %s' % rn_list)


if __name__ == '__main__':
    main_predict()
