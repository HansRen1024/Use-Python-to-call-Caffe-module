#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 11:16:29 2017

@author: hans
"""

import caffe

def lenet(lmdb, batch_size, include_acc=False):
    from caffe import layers as L
    from caffe import params as P
    n = caffe.NetSpec()
    n.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb,
                             transform_param=dict(scale=1./255), ntop=2)
    n.conv1 = L.Convolution(n.data, kernel_size=5, num_output=20, weight_filler=dict(type='xavier'))
    n.pool1 = L.Pooling(n.conv1, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.conv2 = L.Convolution(n.pool1, kernel_size=5, num_output=50, weight_filler=dict(type='xavier'))
    n.pool2 = L.Pooling(n.conv2, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.ip1 = L.InnerProduct(n.pool2, num_output=500, weight_filler=dict(type='xavier'))
    n.relu1 = L.ReLU(n.ip1, in_place=True)
    n.ip2 = L.InnerProduct(n.relu1, num_output=10, weight_filler=dict(type='xavier'))
    n.loss = L.SoftmaxWithLoss(n.ip2, n.label)
    if include_acc:
        n.acc = L.Accuracy(n.ip2, n.label)
    return n.to_proto()

def write_lenet():
    with open('./doc/train_lenet.prototxt','w') as f:
        f.write(str(lenet('./doc/mnist_train_lmdb', 64)))
    
    with open('./doc/test_lenet.prototxt', 'w') as f:
        f.write(str(lenet('./doc/mnist_test_lmdb', 100, True)))

def deploy():
    from caffe import layers as L
    from caffe import params as P
    from caffe import to_proto
    conv1 = L.Convolution(bottom='data', kernel_size=5, num_output=20, weight_filler=dict(type='xavier'))
    pool1 = L.Pooling(conv1, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    conv2 = L.Convolution(pool1, kernel_size=5, num_output=50, weight_filler=dict(type='xavier'))
    pool2 = L.Pooling(conv2, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    ip1 = L.InnerProduct(pool2, num_output=500, weight_filler=dict(type='xavier'))
    relu1 = L.ReLU(ip1, in_place=True)
    ip2 = L.InnerProduct(relu1, num_output=10, weight_filler=dict(type='xavier'))
    prob = L.Softmax(ip2)
    return to_proto(prob)

def write_deploy():
    with open('doc/deploy_lenet.prototxt', 'w') as f:
        f.write('name: "Lenet"\n')
        f.write('input: "data"\n')
        f.write('input_dim: 1\n')
        f.write('input_dim: 3\n')
        f.write('input_dim: 28\n')
        f.write('input_dim: 28\n')
        f.write(str(deploy()))
    
def solver_dict():
    solver_file='doc/solver_lenet.prototxt'
    sp={}
    sp['train_net']='"doc/train_lenet.prototxt"'
    sp['test_net']='"doc/test_lenet.prototxt"'
    sp['test_iter']='100'
    sp['test_interval']='500'
    sp['display']='100'
    sp['max_iter']='10000'
    sp['base_lr']='0.01'
    sp['lr_policy']='"inv"'
    sp['gamma']='0.0001'
    sp['power']='0.75'
    sp['momentum']='0.9'
    sp['weight_decay']='0.0005'
    sp['snapshot']='1000'
    sp['snapshot_prefix']='"models/lenet"'
    sp['solver_mode']='GPU'
    sp['solver_type']='SGD'
    sp['device_id']='0'
    
    with open(solver_file, 'w') as f:
        for key, value in sp.items():
            if not(type(value) is str):
                raise TypeError('All solver parameters must be string')
            f.write('%s: %s\n' %(key, value))
            
def solver_caffe():
    from caffe.proto import caffe_pb2
    s = caffe_pb2.SolverParameter()
    solver_file='doc/solver_lenet.prototxt'
    
    s.train_net = 'doc/train_lenet.prototxt'
    s.test_net.append('doc/test_lenet.prototxt')
    s.test_interval = 500
    s.test_iter.append(100)
    s.display = 100
    s.max_iter = 5000
    s.base_lr = 0.01
    s.lr_policy = "inv"
    s.gamma = 0.0001
    s.power = 0.75
    s.momentum = 0.9
    s.weight_decay = 0.0005
    s.snapshot = 5000
    s.snapshot_prefix = "models/lenet"
    s.type = "SGD"
    s.solver_mode = caffe_pb2.SolverParameter.GPU
    
    with open(solver_file, 'w') as f:
        f.write(str(s))

def train():
    caffe.set_device(0)
    caffe.set_mode_gpu()
    solver = caffe.SGDSolver('doc/solver_lenet.prototxt')
    solver.solve()
if __name__ == '__main__':
#    write_lenet()
#    write_deploy()
    solver_dict()
#    solver_caffe()
    train()