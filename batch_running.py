#!/usr/bin/python
import os,sys

gpu_id = sys.argv[1]



set_num_dis = [50, 20, 5]
#set_num_dis = [50]
set_num_samples = [900, 400, 100]
#set_num_samples = [100]
#set_num_types = ['joint', 'parameter', 'distribution']
#set_num_types = ['distribution']
set_num_types = ['joint']
#set_num_types = ['parameter']
#set_num_types = ['joint', 'parameter']

starting_rand = 1000
repeat = 3


set_num_dis = map(str, set_num_dis)
set_num_samples = map(str, set_num_samples)


#pre = 'huber_10_conv_k5_p2_64_64_max_128_128_128_fc_1024_512_share_5_layers_'
pre = 'huber_10_conv_k5_p2_64_64_max_k5_p2_64_64_64_ave_fc_64_64_share_2_layers_'
#pre = 'huber_10_conv_k5_p2_64_max_128_128_fc_512_256_share_3_layers_'

for rand in range(starting_rand, starting_rand + repeat):
    rand = str(rand)
    for tys in set_num_types:
        for dis in set_num_dis:
            for sps in set_num_samples:
                cmds = 'sh train_net_qsub.sh ' + dis + ' ' + sps + ' ' + tys
                fileName = pre + dis + '_' + sps + '_' + tys + '_' + rand + '_gpu' + gpu_id
                cmds += ' ' + fileName + ' ' + gpu_id + ' > ./tmp/' + fileName + ' 2>&1'
                #print cmds
                os.system(cmds)
