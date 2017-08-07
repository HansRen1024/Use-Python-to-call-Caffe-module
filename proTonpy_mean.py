#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import caffe
import numpy as np

prototxt = 'doc/mnist_mean.binaryproto'
npy = 'doc/mnist_mean.npy'

blob = caffe.proto.caffe_pb2.BlobProto()
data = open(prototxt, 'rb' ).read()
blob.ParseFromString(data)

array = np.array(caffe.io.blobproto_to_array(blob))
mean_npy = array[0]
np.save(npy ,mean_npy)