import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import caffe


num_distributions = str(sys.argv[1])
sample_dimension = str(sys.argv[2])
architecture = sys.argv[3]
NO_gpu = int(sys.argv[4])
no_sharing = int(sys.argv[5])
server_name = 'radon'




os.system('python reProtocol_pars.py ' + num_distributions + ' ' + sample_dimension + ' joint ' + architecture)
#caffe.set_mode_cpu() 
caffe.set_mode_gpu() 
caffe.set_device(NO_gpu)

SOLVER = './protocol/dis_' + num_distributions + '_' + sample_dimension + '_dimension.solver_joint_'
PROTO = './protocol/dis_' + num_distributions + '_' + sample_dimension + '_dimension.protocol_'
MODEL = '/home/purduethu/scratch/' + server_name  + '/d/deng106/CNNStatisticalModel/distributions/models/joint/'

max_steps = 200000
steps = 1


layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'ip1', 'ip2']
layers = layers[:no_sharing]

solver_par = caffe.get_solver(SOLVER + 'parameter') 
solver_dis = caffe.get_solver(SOLVER + 'distribution') 


par_bak = {}
dis_bak = {}
save_loop = 20


'''
  Joint head model is very likely to get nan, we cache our data in case nan appears
  Since checking if nan value exists consumes time, we use a gap to check periodically

'''

for i in range(max_steps / steps):
    solver_par.step(steps)
    for name in layers:
        solver_dis.net.params[name][0].data[...] = solver_par.net.params[name][0].data[...]
        solver_dis.net.params[name][1].data[...] = solver_par.net.params[name][1].data[...]

    solver_dis.step(steps)
    for name in layers:
        solver_par.net.params[name][0].data[...] = solver_dis.net.params[name][0].data[...]
        solver_par.net.params[name][1].data[...] = solver_dis.net.params[name][1].data[...]
