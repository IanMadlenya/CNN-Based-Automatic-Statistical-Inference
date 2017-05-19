import caffe
import sys
import pandas as pd
import numpy as np


'''
py predict.py 20 100 distribution
'''

num_dis = sys.argv[1]
num_samples = sys.argv[2]
model_type = sys.argv[3]
suffix = ''
if len(sys.argv) == 4:
    rand = '1000'
    gpuid = '0'
else:
    rand = sys.argv[4]
    gpuid = sys.argv[5]


architecture = 'huber_10_conv_k5_p2_64_64_max_128_128_128_fc_1024_512_share_5_layers_' + num_dis + '_' + num_samples + '_' + model_type + '_' + rand + '_gpu' + gpuid


total_testing_samples = 100 * float(num_dis)
'''
if num_dis == '5' or num_dis == '50':
    total_testing_samples = 5000
elif num_dis == '20':
    if num_samples == '100':
        total_testing_samples = 20000
    elif num_samples == '400':
        total_testing_samples = 10000
    else:
        total_testing_samples = 5000
'''
window_dim = int(np.sqrt(int(num_samples)))
model = './protocol/deploy_large.prototxt.' + num_dis + '.'  + num_samples
# used in paper already


if model_type == 'distribution':
    weights = '/home/purduethu/scratch/radon/d/deng106/CNNStatisticalModel/distributions/models/distribution/'
elif model_type == 'joint':
    weights = '/home/purduethu/scratch/radon/d/deng106/CNNStatisticalModel/distributions/models/joint/distribution/'

weights += num_dis + '_dis_' + num_samples + '_dim/'
weights += architecture + '/_iter_200000.caffemodel'




print(weights)

caffe.set_mode_gpu()
caffe.set_device(1)

net = caffe.Net(model, weights, caffe.TEST)

dir = "/home/purduethu/scratch/radon/d/deng106/CNNStatisticalModel/distributions/input/"
data = np.genfromtxt(dir + 'test_input_' + num_dis + '_' + num_samples + '.txt' + suffix,delimiter=',')


data = np.reshape(data, (total_testing_samples, 1, window_dim, window_dim))
labels = pd.read_table(dir + 'test_target_' + num_dis + '_' + num_samples + '.txt' + suffix, delimiter=',',header=None)
distributions = np.array(labels.iloc[:, 1])


net.blobs['data'].data[...] = data
predictions =  net.forward()
name = predictions.keys()[0]


print(predictions)

print("Predictions are: ")

predict = np.argmax(predictions[name], axis=1)
print(np.argmax(predictions[name], axis=1))

print("Truth is: ")
truth = distributions
print(distributions)



from sklearn.metrics import confusion_matrix
#print confusion_matrix(truth, predict)

# when dim is large, use another way to print it
myMat = confusion_matrix(truth, predict)


np.savetxt('confusion_matrix.txt', myMat, fmt='%d', delimiter=',')


#######################################################
top2 = np.argpartition(-predictions[name], 2, axis=1)
result_args = top2[:,:1]
count = 0
for i in range(distributions.size):
    if distributions[i,] in result_args[i,]:
        count = count + 1

print("top 1 accuracy is: ", count * 1.0 /distributions.size, num_dis, num_samples, model_type)
print(count * 1.0 /distributions.size)

#######################################################
top2 = np.argpartition(-predictions[name], 2, axis=1)
result_args = top2[:,:2]
count = 0
for i in range(distributions.size):
    if distributions[i,] in result_args[i,]:
        count = count + 1
print("top 2 accuracy is: ", count * 1.0 /distributions.size, num_dis, num_samples, model_type)
print(count * 1.0 /distributions.size)
