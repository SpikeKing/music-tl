#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/5/8
"""
import argparse

import sys

from hash.distance_api import DistanceApi


def main_predict():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '-n', '--name',
        dest='name',
        metavar='name',
        default='None',
        help='add a target name')
    args = parser.parse_args()
    if len(sys.argv) == 1:
        parser.print_help()
        exit(0)

    audio_name = args.name
    print('[INFO] 目标音频: %s' % audio_name)
    da = DistanceApi()
    rb_list, rn_list = da.distance(audio_name)
    print('[INFO] 距离: %s' % rb_list)
    print('[INFO] 相似: %s' % rn_list)


if __name__ == '__main__':
    main_predict()
