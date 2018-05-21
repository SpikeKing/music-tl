# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/4/18
"""
import os
import random

import mxnet as mx
import numpy as np
from mxnet import gluon, autograd
from mxnet.gluon.data import dataset, DataLoader

from bases.trainer_base import TrainerBase
from root_dir import ROOT_DIR
from utils.utils import mkdir_if_not_exist, safe_div


class TripletTrainerMxnet(TrainerBase):

    def __init__(self, model, data, config):
        super(TripletTrainerMxnet, self).__init__(model, data, config)
        self.loss = []
        self.acc = []
        self.val_loss = []
        self.val_acc = []

    def init_callbacks(self):
        train_dir = os.path.join(ROOT_DIR, self.config.tb_dir, "train")
        mkdir_if_not_exist(train_dir)

    def train(self):
        x_train = self.data[0][0]
        y_train = np.argmax(self.data[0][1], axis=1)

        # 测试不使用全量数据
        x_test = self.data[1][0]
        y_test = np.argmax(self.data[1][1], axis=1)

        print('[INFO] 原始训练数据: %s, %s' % (str(x_train.shape), str(y_train.shape)))
        print('[INFO] 原始测试数据: %s, %s' % (str(x_test.shape), str(y_test.shape)))

        self.train_core(x_train, y_train, x_test, y_test)

    def train_core(self, x_train, y_train, x_test, y_test):
        ctx = mx.gpu()
        self.model.collect_params().initialize(mx.init.Uniform(scale=0.1), ctx=ctx)
        triplet_loss = gluon.loss.TripletLoss(margin=40)
        trainer_triplet = gluon.Trainer(self.model.collect_params(), 'adam')

        def transform(data_, label_):
            return data_.astype(np.float32), label_.astype(np.float32)

        for epoch in range(self.config.num_epochs):

            print('[INFO] 数据处理中...')
            train_data = DataLoader(
                TripletDataset(rd=x_train, rl=y_train, transform=transform),
                self.config.batch_size, shuffle=True)

            test_data = DataLoader(
                TripletDataset(rd=x_test, rl=y_test, transform=transform),
                self.config.batch_size, shuffle=True)

            print('[INFO] 数据处理完成')

            for i, (data, _) in enumerate(train_data):
                data = data.as_in_context(ctx)
                anc_ins, pos_ins, neg_ins = data[:, 0], data[:, 1], data[:, 2]

                with autograd.record():
                    inter1 = self.model(anc_ins)[:, 63]  # 训练的时候组合
                    inter2 = self.model(pos_ins)[:, 63]
                    inter3 = self.model(neg_ins)[:, 63]
                    loss = triplet_loss(inter1, inter2, inter3)  # TripletLoss
                    loss.backward()

                trainer_triplet.step(self.config.batch_size)
                curr_loss = mx.nd.mean(loss).asscalar()
                print("[INFO] epoch: %s, loss: %s" % (epoch, curr_loss))

            dist_acc = self.evaluate_net(self.model, test_data, ctx)  # 评估epoch的性能
            self.model.save_params(
                os.path.join(ROOT_DIR, 'experiments/music_tl_v2/checkpoints', "triplet_loss_model_%s_%s.params" %
                             (epoch, '%0.4f' % dist_acc)))  # 存储模型

    def test(self):
        ctx = mx.gpu()
        self.model.load_params(
            os.path.join(ROOT_DIR, 'experiments/music_tl_v2/checkpoints', 'triplet_loss_model_15_1.0000.params'),
            ctx=ctx)

        # 测试不使用全量数据
        x_test = self.data[1][0]
        y_test = np.argmax(self.data[1][1], axis=1)

        def transform(data_, label_):
            return data_.astype(np.float32), label_.astype(np.float32)

        test_data = DataLoader(
            TripletDataset(rd=x_test, rl=y_test, transform=transform),
            self.config.batch_size, shuffle=True)
        self.evaluate_net(self.model, test_data, ctx=ctx)  # 评估epoch的性能

    @staticmethod
    def evaluate_net(model, test_data, ctx):
        triplet_loss = gluon.loss.TripletLoss(margin=0)
        sum_correct = 0
        sum_all = 0
        rate = 0.0
        for i, (data, _) in enumerate(test_data):
            data = data.as_in_context(ctx)

            anc_ins, pos_ins, neg_ins = data[:, 0], data[:, 1], data[:, 2]
            inter1 = model(anc_ins)[:, 63]  # 训练的时候组合
            inter2 = model(pos_ins)[:, 63]
            inter3 = model(neg_ins)[:, 63]
            loss = triplet_loss(inter1, inter2, inter3)  # 交叉熵

            loss = loss.asnumpy()
            n_all = loss.shape[0]
            n_correct = np.sum(np.where(loss == 0, 1, 0))

            sum_correct += n_correct
            sum_all += n_all
            rate = safe_div(sum_correct, sum_all)
            print('准确率: %.4f (%s / %s)' % (rate, sum_correct, sum_all))
        return rate


class TripletDataset(dataset.Dataset):
    def __init__(self, rd, rl, transform=None):
        self.__rd = rd  # 原始数据
        self.__rl = rl  # 原始标签
        self._data = None
        self._label = None
        self._transform = transform
        self._get_data()

    def __getitem__(self, idx):
        if self._transform is not None:
            return self._transform(self._data[idx], self._label[idx])
        return self._data[idx], self._label[idx]

    def __len__(self):
        return len(self._label)

    def _get_data(self):
        label_list = np.unique(self.__rl)
        digit_indices = [np.where(self.__rl == i)[0] for i in label_list]
        tl_pairs = self.create_pairs_v2(self.__rd, digit_indices, len(label_list))
        print('[INFO] 完成Triplet数据处理! ')
        self._data = tl_pairs
        self._label = np.ones(tl_pairs.shape[0])

    @staticmethod
    def create_pairs(x, digit_indices, num_classes):
        pairs = []
        n = min([len(digit_indices[d]) for d in range(num_classes)]) - 1  # 最小类别数
        for d in range(num_classes):
            for i in range(n):
                z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
                inc = random.randrange(1, num_classes)
                dn = (d + inc) % num_classes
                z3 = digit_indices[dn][i]
                pairs += [[x[z1], x[z2], x[z3]]]
        return np.asarray(pairs)

    @staticmethod
    def create_pairs_v2(x, digit_indices, num_classes, clz_samples=21, n_loop=1):
        pairs = []
        n = clz_samples - 1
        print "[INFO] create_pairs - n: %s, num_classes: %s" % (n, num_classes)
        for d in range(num_classes):
            if len(digit_indices[d]) < clz_samples:
                print('[INFO] 去除样本类别: %s' % d)
                continue
            for n_i in range(n_loop):  # 多次循环，多组数据
                for i in range(n):
                    np.random.shuffle(digit_indices[d])
                    z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
                    while True:
                        inc = random.randrange(1, num_classes)
                        dn = (d + inc) % num_classes
                        if len(digit_indices[dn]) >= clz_samples:
                            break
                    z3 = digit_indices[dn][i]
                    pairs += [[x[z1], x[z2], x[z3]]]
        return np.asarray(pairs)
