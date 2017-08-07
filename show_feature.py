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
    data -= data.min()
    data /= data.max()
    
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
    
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    plt.imshow(data)
    plt.axis('off')

prototxt='doc/deploy_lenet.prototxt'
caffe_model='models/lenet_iter_10000.caffemodel'
mean_file='doc/mnist_mean.npy'

im = caffe.io.load_image('doc/9.jpg')
im = caffe.io.resize_image(im,(28,28,1))

caffe.set_mode_gpu()
net = caffe.Net(prototxt,caffe_model,caffe.TEST)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape}) #data blob 结构（n, k, h, w)
transformer.set_transpose('data', (2, 0, 1)) #改变图片维度顺序，(h, w, k) -> (k, h, w)
transformer.set_mean('data', np.load(mean_file).mean(1).mean(1))
transformer.set_raw_scale('data', 255)

net.blobs['data'].data[...] = transformer.preprocess('data', im)
net.forward()

for name,feature in net.blobs.items(): #查看各层特征规模
    print name + '\t' + str(feature.data.shape)

conv1_data = net.blobs['conv1'].data[0] #提取特征
show(conv1_data)

prob_data = net.blobs['prob'].data[0]
plt.figure()
plt.plot(prob_data)