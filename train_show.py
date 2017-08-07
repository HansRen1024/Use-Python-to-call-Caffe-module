#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 21:56:32 2017

@author: hans

"""

import matplotlib.pyplot as plt
import caffe
import numpy as np

caffe.set_device(0)
caffe.set_mode_gpu()
solver = caffe.SGDSolver('doc/solver_lenet.prototxt')

# 下面参数参照solver
max_iter = 10000
display= 100
test_iter = 100
test_interval =500

#初始化
train_loss = np.zeros(max_iter/ display)
test_loss = np.zeros(max_iter/ test_interval)
test_acc = np.zeros(max_iter / test_interval)

_train_loss = 0; _test_loss = 0; _accuracy = 0
for it in range(max_iter):
    solver.step(1)
    _train_loss += solver.net.blobs['loss'].data # 'loss' or 'Softmax1'
    if it % display == 0:
        # 计算平均train loss
        train_loss[it / display] = _train_loss / display
        _train_loss = 0

    if it % test_interval == 0:
        for test_it in range(test_iter):
            solver.test_nets[0].forward()
            _test_loss += solver.test_nets[0].blobs['loss'].data # 'loss' or 'Softmax1'
            _accuracy += solver.test_nets[0].blobs['acc'].data # 'acc' or 'Accuracy1'
        # 计算平均test loss
        test_loss[it / test_interval] = _test_loss / test_iter
        # 计算平均test accuracy
        test_acc[it / test_interval] = _accuracy / test_iter
        _test_loss = 0
        _accuracy = 0

_, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.plot(display * np.arange(len(train_loss)), train_loss, 'g')
ax1.plot(test_interval * np.arange(len(test_loss)), test_loss, 'y')
ax2.plot(test_interval * np.arange(len(test_acc)), test_acc, 'r')

ax1.set_xlabel('iteration')
ax1.set_ylabel('loss')
ax2.set_ylabel('accuracy')
plt.show()