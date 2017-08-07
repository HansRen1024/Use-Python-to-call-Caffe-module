#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 20:41:03 2017

@author: hans

"""

import caffe
import numpy as np
deploy='.prototxt'
caffe_model='.caffemodel'
img='.jpg'
labels_filename='.txt'
mean_file='.npy'

net = caffe.Net(deploy, caffe_model, caffe.TEST)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape}) #data blob 结构（n, k, w, h)
transformer.set_transpose('data', (2, 0, 1)) #改变图片维度顺序，(w, h, k) -> (k, w, h)
transformer.set_mean('data', np.load(mean_file).mean(1).mean(1))
transformer.set_raw_scale('data', 255)
transformer.set_channel_swap('data', (2, 1, 0)) # RGB -> BGR

im = caffe.io.load_image(img)
# 将处理好的数据放入网络中名为'data'的bolb内，就是放入net预分配的内存中。
net.blobs['data'].data[...] = transformer.preprocess('data', im)

out = net.forward() # 网络结构，模型和数据都已经准备好，无需加参数

labels = np.loadtxt(labels_filename, str, delimiter='\t')
prob = net.blobs['prob'].data[0].flatten()
print prob

# print 'the class is:', labels[prob.argmax()], 'accuracy: ', prob[prob.argmax()] #跟下面两句话功能相同

order = prob.argsort()[-1]
print 'the class is:', labels[order], 'accuracy: ', prob[order]
