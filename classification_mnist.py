#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 20:41:03 2017

@author: hans

"""

import caffe
import numpy as np


deploy='doc/deploy_lenet.prototxt' # 需要修改inout_dim: 1, 1, 28, 28
caffe_model='models/lenet_iter_10000.caffemodel'
img='doc/7.jpg'
labels_filename='doc/words.txt'
labels = np.loadtxt(labels_filename, str, delimiter='\t')
mean_file='doc/mnist_mean.npy'

net = caffe.Net(deploy, caffe_model, caffe.TEST)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape}) #data blob 结构（n, k, h, w)
transformer.set_transpose('data', (2, 0, 1)) #改变图片维度顺序，(h, w, k) -> (k, h, w)
transformer.set_mean('data', np.load(mean_file).mean(1).mean(1))
transformer.set_raw_scale('data', 255) #将像素范围改成缩放到[0,1]
# transformer.set_channel_swap('data', (2, 1, 0)) # mnist单通道不需要转换

im = caffe.io.load_image(img) #加载图片
im = caffe.io.resize_image(im,(28,28,1)) # 修改图片尺寸维度

caffe_in = transformer.preprocess('data', im) #将处理好的数据放入caffe_in
out = net.forward(**{'data': caffe_in}) #将数据放入网络中进行一次前向传播
prob = out['prob'].reshape(10,) # 可以看出网络中blob都是以字典形式存储数据的。

# net.blobs['data'].data[...] = transformer.preprocess('data', im)#与上面功能相同
# net.forward()
# prob = net.blobs['prob'].data[0].flatten()

print prob

# print 'the class is:', labels[prob.argmax()], 'accuracy: ', prob[prob.argmax()] #跟下面两句话功能相同

order = prob.argsort()[-1]
print 'the class is:', labels[order], 'accuracy: ', prob[order]
