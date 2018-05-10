#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/5/10
"""
import matplotlib

matplotlib.use('Agg')

import os
import sys

import numpy as np
import matplotlib.pyplot as plot

p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)

from root_dir import ROOT_DIR
from utils.utils import *


def draw_duration_plot():
    _, name_list = traverse_dir_files(os.path.join(ROOT_DIR, "experiments", "raw_data"))

    dr_list = []
    for name in name_list:
        duration = float(name.replace('.mp3', '').split('_')[1])
        dr_list.append(duration)
    plot_hist(dr_list)


def plot_hist(hist_data):
    """
    绘制直方图
    :param hist_data: 直方图数据
    :return: 展示图
    """
    hist_data = np.asarray(hist_data)
    # sns.set(style='ticks', palette='Set2')
    fig, aux = plot.subplots(ncols=1, nrows=1)
    min_x, max_x = 0, 61
    aux.hist(hist_data, bins=60, range=[min_x, max_x], facecolor='magenta', edgecolor="black", alpha=0.75)
    aux.set_xlabel("sec(%0.2f ~ %0.2f)" % (np.min(hist_data), np.max(hist_data)))
    aux.set_ylabel("num(%s)" % (hist_data.shape[0]))
    aux.set_xlim([min_x, max_x])
    aux.set_xticks(range(min_x, max_x, 5))
    fig.set_size_inches(10, 8)
    plot.savefig(os.path.join(ROOT_DIR, "experiments/music_tl/images", "durations_hist.png"))


if __name__ == '__main__':
    draw_duration_plot()
