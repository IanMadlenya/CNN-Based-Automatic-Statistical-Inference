import caffe
import sys
import pandas as pd
import numpy as np


'''
py predict.py 20 100 distribution  ''
py predict.py 20 400 distribution  1
py predict.py 20 400 distribution  2
py predict.py 20 900 distribution  11
py predict.py 20 900 distribution  12
py predict.py 20 900 distribution  21
py predict.py 20 900 distribution  22
'''

num_dis = sys.argv[1]
num_samples = sys.argv[2]
model_type = sys.argv[3]
if len(sys.argv) == 4:
    rand = '1000'
    gpuid = '0'
else:
    rand = sys.argv[4]
    gpuid = sys.argv[5]

print len(sys.argv)

architecture = 'huber_10_conv_k5_p2_64_64_max_128_128_128_fc_1024_512_share_5_layers_' + num_dis + '_' + num_samples + '_' + model_type + '_' + rand + '_gpu' + gpuid

total_testing_samples = 100 * float(num_dis) # number of feature samples
window_dim = int(np.sqrt(int(num_samples)))

model = './protocol/deploy_large.prototxt_par.' +  num_dis + '.' + num_samples
# used in paper already


if model_type == 'parameter':
    weights = '/home/purduethu/scratch/radon/d/deng106/CNNStatisticalModel/distributions/models/parameter/'
elif model_type == 'joint':
    weights = '/home/purduethu/scratch/radon/d/deng106/CNNStatisticalModel/distributions/models/joint/parameter/'

weights += num_dis + '_dis_' + num_samples + '_dim/'
weights += architecture + '/_iter_200000.caffemodel'




print weights

caffe.set_mode_gpu()
caffe.set_device(1)

net = caffe.Net(model, weights, caffe.TEST)

dir = "/home/purduethu/scratch/radon/d/deng106/CNNStatisticalModel/distributions/input/"
data = np.genfromtxt(dir + 'test_input_' + num_dis + '_' + num_samples + '.txt',delimiter=',')


data = np.reshape(data, (total_testing_samples, 1, window_dim, window_dim))
labels = pd.read_table(dir + 'test_target_' + num_dis + '_' + num_samples + '.txt', delimiter=',',header=None)
distributions = np.array(labels.iloc[:, 0])


net.blobs['data'].data[...] = data
predictions =  net.forward()
name = predictions.keys()[0]


predictions = predictions['ip3'][:,0]

print "Truth is: "
truth = distributions

output = np.array([predictions, truth])

np.savetxt('par_estimate' + num_dis + '_' + num_samples + '_' + model_type, output, fmt='%.4f')
