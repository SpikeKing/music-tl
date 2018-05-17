# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/4/18
"""
from mxnet.gluon.nn import Sequential, Conv1D, BatchNorm, MaxPool1D, Dropout, HybridSequential
import mxnet as mx
from mxnet.gluon.rnn import LSTM

from bases.model_base import ModelBase
from root_dir import O_DIM


class TripletModelMxnet(ModelBase):
    """
    TripletLoss模型
    """

    MARGIN = 10.0  # 超参

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
        net_triplet = Sequential()

        kernel_size = 1
        pool_size = 2
        dropout_rate = 0.15
        f_act = 'relu'

        with net_triplet.name_scope():
            net_triplet.add(Conv1D(channels=256, kernel_size=kernel_size, activation=f_act))
            net_triplet.add(BatchNorm())
            net_triplet.add(MaxPool1D(pool_size=pool_size))
            net_triplet.add(Dropout(rate=dropout_rate))

            net_triplet.add(Conv1D(channels=128, kernel_size=kernel_size, activation=f_act))
            net_triplet.add(BatchNorm())
            net_triplet.add(MaxPool1D(pool_size=pool_size))
            net_triplet.add(Dropout(rate=dropout_rate))

            net_triplet.add(Conv1D(channels=64, kernel_size=kernel_size, activation=f_act))
            net_triplet.add(BatchNorm())
            net_triplet.add(MaxPool1D(pool_size=pool_size))

            net_triplet.add(LSTM(hidden_size=O_DIM, bidirectional=True))
            net_triplet.add(LSTM(hidden_size=O_DIM, bidirectional=True))
            net_triplet.add(LSTM(hidden_size=O_DIM))

        return net_triplet
