#
#   This file demonstrates how to generate a .prototxt file for Caffe consumption 
#   from object definitions in Python. I imagine this might be easier than manually
#   writing prototxt every time; also probably a lot easier to change. 
#

import caffe
from caffe import layers as L
from caffe import params as P


def lenet(lmdb, batch_size):
    # examples/01-learning-lenet.ipynb
    # our version of LeNet: a series of linear and simple nonlinear transformations
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
    return n.to_proto()

def autoencoder():

    n = caffe.NetSpec()

    n.data, n.label=L.Data(batch_size=100, backend=P.Data.LMDB, source='mnist/mnist_train_lmdb',
                            transform_param=dict(scale=1./255), ntop=2)

    n.ip1 = L.InnerProduct(n.data, num_output=784, weight_filler=dict(type='xavier'))
    n.relu1 = L.ReLU(n.ip1, in_place=True)

    n.ip2 = L.InnerProduct(n.relu1, num_output=392, weight_filler=dict(type='xavier'))
    n.relu2 = L.ReLU(n.ip2, in_place=True)

    n.ip3 = L.InnerProduct(n.relu2, num_output=784, weight_filler=dict(type='xavier'))

    n.loss = L.SoftmaxWithLoss(n.ip3, n.data)

    return n.to_proto()

print autoencoder()

# with open('mnist/lenet_auto_train.prototxt', 'w') as f:
#     f.write(str(lenet('mnist/mnist_train_lmdb', 64)))
    
# with open('mnist/lenet_auto_test.prototxt', 'w') as f:
#     f.write(str(lenet('mnist/mnist_test_lmdb', 100)))