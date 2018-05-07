# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/4/18
"""
import os

import tensorflow as tf
from keras import Input, Model
from keras.layers import Dropout, K, Concatenate, Conv1D, BatchNormalization, MaxPooling1D, LSTM
from keras.optimizers import Adam
from keras.utils import plot_model

from bases.model_base import ModelBase


class TripletModel(ModelBase):
    """
    TripletLoss模型
    """

    MARGIN = 10.0  # 超参

    def __init__(self, config):
        super(TripletModel, self).__init__(config)
        self.build_model()

    def build_model(self):
        self.model = self.triplet_loss_model()

    @staticmethod
    def triplet_loss(y_true, y_pred):
        """
        Triplet Loss的损失函数
        """
        anc, pos, neg = y_pred[:, 0:128], y_pred[:, 128:256], y_pred[:, 256:]

        # 欧式距离
        pos_dist = K.sum(K.square(anc - pos), axis=-1, keepdims=True)
        neg_dist = K.sum(K.square(anc - neg), axis=-1, keepdims=True)
        basic_loss = pos_dist - neg_dist + TripletModel.MARGIN

        loss = K.maximum(basic_loss, 0.0)

        print "[INFO] model - triplet_loss shape: %s" % str(loss.shape)
        return loss

    def triplet_loss_model(self):
        anc_input = Input(shape=(256, 32), name='anc_input')  # anchor
        pos_input = Input(shape=(256, 32), name='pos_input')  # positive
        neg_input = Input(shape=(256, 32), name='neg_input')  # negative

        shared_model = self.deep_conv_lstm()  # 共享模型

        # 必须指定GPU，否则出现异常，参考Keras的Device parallelism
        # https://keras.io/getting-started/faq/#how-can-i-run-a-keras-model-on-multiple-gpus
        with tf.device('/gpu:0'):
            anc_out = shared_model(anc_input)
        with tf.device('/gpu:1'):
            pos_out = shared_model(pos_input)
        with tf.device('/gpu:2'):
            neg_out = shared_model(neg_input)

        print "[INFO] anc_out - 锚shape: %s" % str(anc_out.get_shape())
        print "[INFO] pos_out - 正shape: %s" % str(pos_out.get_shape())
        print "[INFO] neg_out - 负shape: %s" % str(neg_out.get_shape())

        with tf.device('/cpu:0'):
            output = Concatenate()([anc_out, pos_out, neg_out])  # 连接
        model = Model(inputs=[anc_input, pos_input, neg_input], outputs=output)

        plot_model(model, to_file=os.path.join(self.config.img_dir, "triplet_loss_model.png"),
                   show_shapes=True)  # 绘制模型图
        model.compile(loss=self.triplet_loss, optimizer=Adam())

        return model

    def deep_conv_lstm(self):
        """
        共享模型deep_conv_lstm
        :return: 模型
        """

        def cnn_lstm_cell(cell_input):
            """
            基于DeepConvLSTM算法, 创建子模型
            :param cell_input: 输入数据
            :return: 子模型
            """
            kernel_size = 1
            pool_size = 2
            dropout_rate = 0.15
            f_act = 'relu'

            sub_model = Conv1D(256, kernel_size, input_shape=(256, 32), activation=f_act, padding='same')(cell_input)
            sub_model = BatchNormalization()(sub_model)
            sub_model = MaxPooling1D(pool_size=pool_size)(sub_model)
            sub_model = Dropout(dropout_rate)(sub_model)
            sub_model = Conv1D(128, kernel_size, activation=f_act, padding='same')(sub_model)
            sub_model = BatchNormalization()(sub_model)
            sub_model = MaxPooling1D(pool_size=pool_size)(sub_model)
            sub_model = Dropout(dropout_rate)(sub_model)
            sub_model = Conv1D(64, kernel_size, activation=f_act, padding='same')(sub_model)
            sub_model = BatchNormalization()(sub_model)
            sub_model = MaxPooling1D(pool_size=pool_size)(sub_model)
            sub_model = LSTM(128, return_sequences=True)(sub_model)
            sub_model = BatchNormalization()(sub_model)
            sub_model = LSTM(128, return_sequences=True)(sub_model)
            sub_model = BatchNormalization()(sub_model)
            sub_model = LSTM(128)(sub_model)
            main_output = Dropout(dropout_rate)(sub_model)

            return main_output

        ins_input = Input(shape=(256, 32))
        model = cnn_lstm_cell(ins_input)  # 合并模型
        output = Dropout(0.4)(model)
        model = Model(ins_input, output)

        plot_model(model, to_file=os.path.join(self.config.img_dir, "sub_model.png"),
                   show_shapes=True)  # 绘制模型图

        return model
