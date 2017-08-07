#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 22:15:05 2017

@author: hans

"""

import caffe
import numpy as np
import matplotlib.pyplot as plt

def show(data, padsize=1, padval=0):
    data = (data - data.min()) / (data.max() - data.min())
    
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
    
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    data = data[:,:,0]
    plt.imshow(data)
    plt.axis('off')
    plt.show()

prototxt='doc/deploy_lenet.prototxt'
caffe_model='models/lenet_iter_10000.caffemodel'
net = caffe.Net(prototxt,caffe_model,caffe.TEST)

for name, param_blob in net.params.items():#查看各层参数规模
    print name + '\t' + str(param_blob[0].data.shape), str(param_blob[1].data.shape)

conv1_param=net.params['conv1'][0].data  #提取参数w, 参数维度为(n, k, h, w)
show(conv1_param.transpose(0, 2, 3, 1)) # 转换参数维度为(n, h, w, k)