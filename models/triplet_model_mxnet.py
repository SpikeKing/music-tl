# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/4/18
"""
import mxnet as mx
import os
from mxnet.gluon.nn import Conv1D, BatchNorm, MaxPool1D, Dropout, Dense, HybridSequential

from bases.model_base import ModelBase
from root_dir import ROOT_DIR
from utils.utils import write_line


class TripletModelMxnet(ModelBase):
    """
    TripletLoss模型
    """

    def __init__(self, config):
        super(TripletModelMxnet, self).__init__(config)
        self.build_model()

    def build_model(self):
        self.model = self.deep_conv_lstm()  # LSTM模型

    @staticmethod
    def deep_conv_lstm():
        """
        共享模型deep_conv_lstm
        :return: 模型
        """
        net_triplet = HybridSequential(prefix='net_')

        kernel_size = 1
        pool_size = 2
        f_act = 'relu'

        with net_triplet.name_scope():
            net_triplet.add(Conv1D(channels=256, kernel_size=kernel_size, activation=f_act))
            net_triplet.add(BatchNorm())
            net_triplet.add(MaxPool1D(pool_size=pool_size))

            net_triplet.add(Conv1D(channels=128, kernel_size=kernel_size, activation=f_act))
            net_triplet.add(BatchNorm())
            net_triplet.add(MaxPool1D(pool_size=pool_size))

            net_triplet.add(Conv1D(channels=64, kernel_size=kernel_size, activation=f_act))
            net_triplet.add(BatchNorm())
            net_triplet.add(MaxPool1D(pool_size=pool_size))

            net_triplet.add(Dense(units=128))

        sym_json = net_triplet(mx.sym.var('data')).tojson()
        json_file = os.path.join(ROOT_DIR, 'experiments', 'sym.json')
        write_line(json_file, sym_json)
        return net_triplet
