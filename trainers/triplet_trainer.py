# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/4/18
"""
import os
import random
import warnings

import numpy as np
from keras.callbacks import TensorBoard, ModelCheckpoint, Callback
from keras.models import load_model
from keras.utils import to_categorical
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import precision_recall_fscore_support

from bases.trainer_base import TrainerBase
from models.triplet_model import TripletModel
from root_dir import ROOT_DIR
from utils.np_utils import prp_2_oh_array
from utils.utils import mkdir_if_not_exist


class TripletTrainer(TrainerBase):
    def __init__(self, model, data, config):
        super(TripletTrainer, self).__init__(model, data, config)
        self.callbacks = []
        self.loss = []
        self.acc = []
        self.val_loss = []
        self.val_acc = []
        self.init_callbacks()

    def init_callbacks(self):
        train_dir = os.path.join(ROOT_DIR, self.config.tb_dir, "train")
        mkdir_if_not_exist(train_dir)
        self.callbacks.append(
            TensorBoard(
                log_dir=train_dir,
                write_images=True,
                write_graph=True,
            )
        )

        self.callbacks.append(TlMetric())
        # self.callbacks.append(FPRMetric())
        # self.callbacks.append(FPRMetricDetail())

    def train_v2(self):
        x_train = self.data[0][0]
        print x_train.shape
        y_train = self.data[0][1]
        x_train = list(x_train)

        self.model.fit(
            x_train, y_train,
            batch_size=32,
            epochs=10,
            verbose=1,
            callbacks=self.callbacks)

    def train(self):
        x_train = self.data[0][0]
        y_train = np.argmax(self.data[0][1], axis=1)
        x_test = self.data[1][0][:19 * 100]
        y_test = np.argmax(self.data[1][1], axis=1)[:19 * 100]

        self.train_core(x_train, y_train, x_test, y_test)

    def train_core(self, x_train, y_train, x_test, y_test):

        clz_train = len(np.unique(y_train))
        print "[INFO] 训练 - 类别数: %s" % clz_train
        print "[INFO] X_train.shape: %s, y_train.shape: %s" \
              % (str(x_train.shape), str(y_train.shape))

        train_indices = [np.where(y_train == i)[0] for i in sorted(np.unique(y_train))]
        tr_pairs = self.create_pairs(x_train, train_indices, clz_train)
        print('[INFO] tr_pairs.shape: %s' % str(tr_pairs.shape))

        tr_error = np.isnan(tr_pairs).sum()
        print('[INFO] tr_pairs 异常数据数: %s' % tr_error)
        if tr_error > 0:
            raise Exception('[Exception] Nan数据错误!!!')

        clz_test = len(np.unique(y_test))
        print "[INFO] 测试 - 类别数: %s" % clz_test
        print "[INFO] X_test.shape: %s, y_test.shape: %s" \
              % (str(x_test.shape), str(y_test.shape))
        test_indices = [np.where(y_test == i)[0] for i in sorted(np.unique(y_test))]
        print "[INFO] 测试 - 类别数: %s" % test_indices[0]
        te_pairs = self.create_pairs(x_test, test_indices, clz_test)

        print('[INFO] te_pairs.shape: %s' % str(te_pairs.shape))
        te_error = np.isnan(te_pairs).sum()
        print('[INFO] te_pairs 异常数据数: %s' % te_error)
        if te_error > 0:
            raise Exception('[Exception] Nan数据错误!!!')

        anc_ins = tr_pairs[:, 0]
        pos_ins = tr_pairs[:, 1]
        neg_ins = tr_pairs[:, 2]

        print "[INFO] anc_ins: %s" % str(anc_ins.shape)
        print "[INFO] pos_ins: %s" % str(pos_ins.shape)
        print "[INFO] neg_ins: %s" % str(neg_ins.shape)

        X = {
            'anc_input': anc_ins,
            'pos_input': pos_ins,
            'neg_input': neg_ins
        }

        anc_ins_te = te_pairs[:, 0]
        pos_ins_te = te_pairs[:, 1]
        neg_ins_te = te_pairs[:, 2]

        X_te = {
            'anc_input': anc_ins_te,
            'pos_input': pos_ins_te,
            'neg_input': neg_ins_te
        }

        print "[INFO] anc_ins_te: %s" % str(anc_ins_te.shape)
        print "[INFO] pos_ins_te: %s" % str(pos_ins_te.shape)
        print "[INFO] neg_ins_te: %s" % str(neg_ins_te.shape)

        self.model.fit(
            X, np.ones(len(anc_ins)),
            batch_size=self.config.batch_size,
            epochs=self.config.num_epochs,
            # validation_data=[X_te, np.ones(len(anc_ins_te))],
            validation_data=[X_te, np.ones(len(anc_ins_te))],
            verbose=1,
            callbacks=self.callbacks)

        self.model.save(os.path.join(self.config.cp_dir, "triplet_loss_model.h5"))  # 存储模型

        y_pred = self.model.predict(X_te)  # 验证模型
        self.show_acc_facets(y_pred, y_pred.shape[0] / clz_test, clz_test)

    @staticmethod
    def show_acc_facets(anchor, positive, negative, clz_size):
        """
        展示模型的准确率
        :param y_pred: 测试结果数据组
        :param n: 数据长度
        :param clz_size: 类别数
        :return: 打印数据
        """
        print "[INFO] trainer - clz_size: %s" % clz_size
        # print "[INFO] trainer - clz %s" % i
        # final = y_pred
        # anchor, positive, negative = final[:, 0:128], final[:, 128:256], final[:, 256:]

        pos_dist = np.sum(np.square(anchor - positive), axis=-1, keepdims=True)
        neg_dist = np.sum(np.square(anchor - negative), axis=-1, keepdims=True)
        basic_loss = pos_dist - neg_dist
        r_count = np.sum(np.where(basic_loss < 0, 1, 0))
        res_min = np.min(basic_loss)
        res_max = np.max(basic_loss)
        res_avg = np.average(basic_loss)
        res_acc = np.average(float(r_count) / float(len(basic_loss)))

        print "[INFO] min: %s, max: %s, avg: %s, acc: %0.4f%% (%s / %s)" % (
            res_min, res_max, res_avg, res_acc * 100, r_count, len(basic_loss))

    @staticmethod
    def create_pairs(x, digit_indices, num_classes):
        """
        创建正例和负例的Pairs
        :param x: 数据
        :param digit_indices: 不同类别的索引列表
        :param num_classes: 类别
        :return: Triplet Loss 的 Feed 数据
        """

        pairs = []
        clz_samples = 19
        # n = min([len(digit_indices[d]) for d in range(num_classes)]) - 1  # 最小类别数
        n = clz_samples - 1
        print "[INFO] create_pairs - n: %s, num_classes: %s" % (n, num_classes)
        for d in range(num_classes):
            if len(digit_indices[d]) < clz_samples:
                print('[INFO] 去除样本类别: %s' % d)
                continue
            for i in range(n):
                z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
                while True:
                    inc = random.randrange(1, num_classes)
                    dn = (d + inc) % num_classes
                    if len(digit_indices[dn]) >= clz_samples:
                        break
                    # else:
                    #     print('[INFO] 去除样本类别: %s' % dn)
                z3 = digit_indices[dn][i]
                pairs += [[x[z1], x[z2], x[z3]]]
        return np.array(pairs)


class TlMetric(Callback):
    def on_epoch_end(self, batch, logs=None):
        X_te0 = {
            'anc_input': self.validation_data[0],
            'pos_input': self.validation_data[0],
            'neg_input': self.validation_data[0]
        }
        y_pred0 = self.model.predict(X_te0)  # 验证模型
        X_te1 = {
            'anc_input': self.validation_data[1],
            'pos_input': self.validation_data[1],
            'neg_input': self.validation_data[1]
        }
        y_pred1 = self.model.predict(X_te1)  # 验证模型
        X_te2 = {
            'anc_input': self.validation_data[2],
            'pos_input': self.validation_data[2],
            'neg_input': self.validation_data[2]
        }
        y_pred2 = self.model.predict(X_te2)  # 验证模型
        clz_test = len(self.validation_data[0]) / 18

        TripletTrainer.show_acc_facets(
            y_pred0[:, :128], y_pred1[:, :128], y_pred2[:, :128], clz_test)


class FPRMetric(Callback):
    """
    输出F, P, R
    """

    def on_epoch_end(self, batch, logs=None):
        val_x = self.validation_data[0]
        val_y = self.validation_data[1]

        prd_y = prp_2_oh_array(np.asarray(self.model.predict(val_x)))

        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        precision, recall, f_score, _ = precision_recall_fscore_support(
            val_y, prd_y, average='macro')
        print " — val_f1: % 0.4f — val_pre: % 0.4f — val_rec % 0.4f" % (f_score, precision, recall)


class FPRMetricDetail(Callback):
    """
    输出F, P, R
    """

    def on_epoch_end(self, batch, logs=None):
        val_x = self.validation_data[0]
        val_y = self.validation_data[1]

        prd_y = prp_2_oh_array(np.asarray(self.model.predict(val_x)))

        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        precision, recall, f_score, support = precision_recall_fscore_support(val_y, prd_y)

        for p, r, f, s in zip(precision, recall, f_score, support):
            print " — val_f1: % 0.4f — val_pre: % 0.4f — val_rec % 0.4f - ins %s" % (f, p, r, s)
