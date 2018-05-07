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

        data_path = os.path.join(ROOT_DIR, 'experiments', 'data_train.npz')
        data_all = np.load(data_path)
        self.X_train = data_all['f_list']
        self.X_train = np.transpose(self.X_train, [0, 2, 1])
        self.y_train = data_all['l_list']
        self.y_train = to_categorical(self.y_train)

        data_path = os.path.join(ROOT_DIR, 'experiments', 'data_test_200.npz')
        data_all = np.load(data_path)
        self.X_test = data_all['f_list']
        self.X_test = np.transpose(self.X_test, [0, 2, 1])
        self.y_test = data_all['l_list']
        self.y_test = to_categorical(self.y_test)

        print "[INFO] X_train.shape: %s, y_train.shape: %s" \
              % (str(self.X_train.shape), str(self.y_train.shape))
        print "[INFO] X_test.shape: %s, y_test.shape: %s" \
              % (str(self.X_test.shape), str(self.y_test.shape))

    def get_train_data(self):
        return self.X_train, self.y_train

    def get_test_data(self):
        return self.X_test, self.y_test


if __name__ == '__main__':
    dl = TripletDL()
