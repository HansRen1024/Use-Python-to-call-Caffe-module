#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np

npy = 'mean.npy'

mean = np.ones([3,227, 227], dtype=np.float) #256是图像尺寸
mean[0,:,:] = 102.502
mean[1,:,:] = 115.405
mean[2,:,:] = 123.468

np.save(npy, mean)