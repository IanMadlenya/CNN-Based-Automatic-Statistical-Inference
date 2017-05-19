import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import caffe


num_distributions = str(sys.argv[1])
sample_dimension = str(sys.argv[2])
architecture = sys.argv[3]
NO_gpu = int(sys.argv[4])
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
layers = ['conv1', 'conv2']
#layers = ['conv1', 'conv2', 'conv3']
#layers = ['conv1']

solver_par = caffe.get_solver(SOLVER + 'parameter') 
solver_dis = caffe.get_solver(SOLVER + 'distribution') 


# need to change sys.argv if needed
if len(sys.argv) == 6: # if to continue from an existing model
    model_number = str(sys.argv[5])
    NUM = num_distributions + '_dis_' + sample_dimension + '_dim/' + architecture + '/_iter_' + model_number + '.caffemodel'
    model_par = caffe.Net(PROTO + 'parameter', MODEL + 'parameter/' + NUM, caffe.TRAIN)
    model_dis = caffe.Net(PROTO + 'distribution', MODEL + 'distribution/' + NUM, caffe.TRAIN)
    for name in model_par.params.keys():
        solver_par.net.params[name][0].data[...] = model_par.params[name][0].data[...]
        solver_par.net.params[name][1].data[...] = model_par.params[name][1].data[...]
    for name in model_dis.params.keys():
        solver_dis.net.params[name][0].data[...] = model_dis.params[name][0].data[...]
        solver_dis.net.params[name][1].data[...] = model_dis.params[name][1].data[...]


def get_Max(solver):
    dmax = 0
    for name in solver.net.params.keys():
        dmax = max(dmax, np.max(np.abs(solver.net.params[name][0].data[...])))
    return dmax

par_bak = {}
dis_bak = {}
save_loop = 20


'''
  Joint head model is very likely to get nan, we cache our data in case nan appears
  Since checking if nan value exists consumes time, we use a gap to check periodically

'''

#threshold = np.exp(100) 
for i in range(max_steps / steps):
    solver_par.step(steps)
    #if get_Max(solver_dis) < threshold:
    for name in layers:
        solver_dis.net.params[name][0].data[...] = solver_par.net.params[name][0].data[...]
        solver_dis.net.params[name][1].data[...] = solver_par.net.params[name][1].data[...]

    solver_dis.step(steps)
    #if get_Max(solver_dis) < threshold:
    for name in layers:
        solver_par.net.params[name][0].data[...] = solver_dis.net.params[name][0].data[...]
        solver_par.net.params[name][1].data[...] = solver_dis.net.params[name][1].data[...]
