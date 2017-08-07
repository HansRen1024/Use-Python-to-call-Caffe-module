#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 20:41:03 2017

@author: hans

http://www.cnblogs.com/denny402/p/5685909.html
"""

import caffe
import numpy as np


deploy='doc/deploy_lenet.prototxt' # 需要修改inout_dim: 1, 1, 28, 28
caffe_model='models/lenet_iter_10000.caffemodel'
img='doc/9.jpg'
labels_filename='doc/words.txt'
mean_file='doc/mnist_mean.npy'
labels = np.loadtxt(labels_filename, str, delimiter='\t')

net = caffe.Net(deploy, caffe_model, caffe.TEST)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape}) #data blob 结构（n, k, h, w)
transformer.set_transpose('data', (2, 0, 1)) #改变图片维度顺序，(h, w, k) -> (k, h, w)
transformer.set_mean('data', np.load(mean_file).mean(1).mean(1))
transformer.set_raw_scale('data', 255)
#transformer.set_channel_swap('data', (2, 1, 0)) # RGB -> BGR

im = caffe.io.load_image(img)
im = caffe.io.resize_image(im,(28,28,1))

#caffe_in = transformer.preprocess('data', im)
#out = net.forward(**{'data': caffe_in})
#prob = out['prob'].reshape(10,)

net.blobs['data'].data[...] = transformer.preprocess('data', im)
net.forward()
prob = net.blobs['prob'].data[0].flatten()
print prob

#print 'the class is:', labels[prob.argmax()], 'accuracy: ', prob[prob.argmax()]

order = prob.argsort()[-1]
print 'the class is:', labels[order], 'accuracy: ', prob[order]