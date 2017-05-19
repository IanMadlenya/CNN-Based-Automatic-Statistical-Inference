import caffe
from caffe import layers as L
from caffe import params as P
import os
import sys

def Net(model_type):
    batch_size = 20
    # our version of LeNet: a series of linear and simple nonlinear transformations
    n = caffe.NetSpec()
    DIR = '/home/purduethu/scratch/radon/d/deng106/CNNStatisticalModel/'
    # this part will be changed by our adapter
    train = DIR + "10distribution/data.1.30.30/train-parameters-path.txt"
    test = DIR + "10distribution/data.1.30.30/test-parameters-path.txt"
    w_lr=[{'lr_mult':1},{'lr_mult':2}]
    g = 'gaussian'
    c = 'constant'
    n.data, n.label = L.Data(name="parameters", type="HDF5Data", include={'phase': caffe.TRAIN}, batch_size=batch_size, source=train, ntop=2)
    n.data1, n.label1 = L.Data(name="intercept", type="HDF5Data", top=["data", "label"], include={'phase': caffe.TEST}, batch_size=batch_size, source=test, ntop=2)
    
    kerSize = 5
    padding = 2
    oo1 = 64
    oo2 = 64
    oo3 = 64
    oo4 = 64
    n.conv1 = L.Convolution(n.data, param=w_lr, kernel_size=kerSize, pad=padding, stride=1, num_output=oo1, weight_filler=dict(type=g, std=0.0001), bias_filler=dict(type=c))
    n.relu1 =        L.ReLU(n.conv1, in_place=True)

    n.conv2 = L.Convolution(n.relu1, param=w_lr, kernel_size=kerSize, pad=padding, stride=1,num_output=oo1, weight_filler=dict(type=g, std=0.01), bias_filler=dict(type=c))
    n.relu2 =        L.ReLU(n.conv2, in_place=True)
    n.pool2 =     L.Pooling(n.relu2, kernel_size=3, stride=2, pool=P.Pooling.MAX)

    
    n.conv3 = L.Convolution(n.pool2, param=w_lr, kernel_size=kerSize, pad=padding, stride=1,num_output=oo2, weight_filler=dict(type=g, std=0.01), bias_filler=dict(type=c))
    n.relu3 =        L.ReLU(n.conv3, in_place=True)

    n.conv4 = L.Convolution(n.relu3, param=w_lr, kernel_size=kerSize, pad=padding, stride=1,num_output=oo2, weight_filler=dict(type=g, std=0.01), bias_filler=dict(type=c))
    n.relu4 =        L.ReLU(n.conv4, in_place=True)
    
    n.conv5 = L.Convolution(n.relu4, param=w_lr, kernel_size=kerSize, pad=padding, stride=1,num_output=oo2, weight_filler=dict(type=g, std=0.01), bias_filler=dict(type=c))
    n.relu5 =        L.ReLU(n.conv5, in_place=True)


    n.pool5 =     L.Pooling(n.relu5, kernel_size=3, stride=2, pool=P.Pooling.AVE)
    
    n.ip1 =  L.InnerProduct(n.pool5, param=w_lr, num_output=oo3, weight_filler=dict(type='gaussian',  std=0.1), bias_filler=dict(type='constant'))
    n.ip2 =  L.InnerProduct(n.ip1, param=w_lr, num_output=oo4, weight_filler=dict(type='gaussian',  std=0.1), bias_filler=dict(type='constant'))
    #n.ip3 =  L.InnerProduct(n.ip2, param=w_lr, num_output=32, weight_filler=dict(type='gaussian',  std=0.1), bias_filler=dict(type='constant'))
    if model_type == "classification":
        n.ip3 =     L.InnerProduct(n.ip2, param=w_lr, num_output=50, weight_filler=dict(type='gaussian', std=0.1), bias_filler=dict(type='constant'))
        n.accuracy =    L.Accuracy(n.ip3, n.label, include={'phase': caffe.TEST})
        n.loss = L.SoftmaxWithLoss(n.ip3, n.label)
    else:
        n.ip3 =     L.InnerProduct(n.ip2, param=w_lr, num_output=1, weight_filler=dict(type='gaussian', std=0.1), bias_filler=dict(type='constant'))
        n.loss1 =  L.HuberLoss(n.ip3, n.label, include={'phase': caffe.TRAIN})
        n.loss2 =  L.HuberLoss(n.ip3, n.label, include={'phase': caffe.TEST})
    return n.to_proto()


# parameter part
inputFile = './auto_train.prototxt'
outputFile = './protocol/parameter.baseline'
with open(inputFile, 'w') as f:
    f.write(str(Net('parameter')))

with open(inputFile) as f:
    fout = open(outputFile, 'w')
    for line in f:
        l = line.strip()
        if l.startswith('data_param'):
            fout.write('  hdf5_data_param {\n')
        elif l.startswith('top: "data1"') or l.startswith('top: "label1"'):
            continue
        elif l.startswith('top: "loss'):
            fout.write('  top: "loss"\n')
        elif l.startswith('name: "loss'):
            fout.write('  name: "loss"\n')
        else:
            fout.write(line)

os.system('rm ' + inputFile)

# distribution part
inputFile = './auto_train.prototxt'
outputFile = './protocol/distribution.baseline'
with open(inputFile, 'w') as f:
    f.write(str(Net('classification')))

with open(inputFile) as f:
    fout = open(outputFile, 'w')
    for line in f:
        l = line.strip()
        if l.startswith('data_param'):
            fout.write('  hdf5_data_param {\n')
        elif l.startswith('top: "data1"') or l.startswith('top: "label1"'):
            continue
        elif l.startswith('top: "loss'):
            fout.write('  top: "loss"\n')
        elif l.startswith('name: "loss'):
            fout.write('  name: "loss"\n')
        else:
            fout.write(line)

os.system('rm ' + inputFile)
