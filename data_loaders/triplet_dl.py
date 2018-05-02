# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/4/18
"""
import os

import numpy as np
from keras.utils import to_categorical

from bases.data_loader_base import DataLoaderBase
from root_dir import ROOT_DIR


class TripletDL(DataLoaderBase):
    def __init__(self, config=None):
        super(TripletDL, self).__init__(config)
        # (self.X_train, self.y_train), (self.X_test, self.y_test) = mnist.load_data()

        # 展平数据，for RNN，感知机
        # self.X_train = self.X_train.reshape((-1, 28 * 28))
        # self.X_test = self.X_test.reshape((-1, 28 * 28))

        # 图片数据，for CNN
        # self.X_train = self.X_train.reshape(self.X_train.shape[0], 28, 28, 1)
        # self.X_test = self.X_test.reshape(self.X_test.shape[0], 28, 28, 1)

        # self.y_train = to_categorical(self.y_train)
        # self.y_test = to_categorical(self.y_test)

        data_path = os.path.join(ROOT_DIR, 'experiments', 'music_data_features.npz')
        data_all = np.load(data_path)
        self.X_train = data_all['f_list']
        self.X_train = np.transpose(self.X_train, [0, 2, 1])
        # self.X_train = np.transpose(self.X_train, [1, 0, 2, 3])
        self.y_train = data_all['l_list']
        self.y_train = to_categorical(self.y_train)

        self.X_test = np.array([])
        self.y_test = np.array([])

        print "[INFO] X_train.shape: %s, y_train.shape: %s" \
              % (str(self.X_train.shape), str(self.y_train.shape))
        # print "[INFO] X_test.shape: %s, y_test.shape: %s" \
        #       % (str(self.X_test.shape), str(self.y_test.shape))

    def get_train_data(self):
        return self.X_train, self.y_train

    def get_test_data(self):
        return self.X_test, self.y_test


if __name__ == '__main__':
    dl = TripletDL()
