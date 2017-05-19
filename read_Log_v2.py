import re
import os
import sys

# read all the logs of the same type, and output the result



#FILE = 'conv_64_k9_p4_32_k7_p3_maxpoiling_64_64_128_ave_pooling_fc_64_parameter_'
#FILE = 'conv_32_k9_p4_32_k7_p3_maxpoiling_64_64_64_ave_pooling_fc_64_32_parameter_'
FILE = 'conv_32_k9_p4_32_k9_p4_maxpoiling_64_64_ave_pooling_parameter_'
FILE = 'conv_64_k9_p4_64_k7_p3_maxpoiling_64_64_ave_pooling_fc_64_64_32_'
FILE = 'conv_64_64_maxpoiling_64_64_ave_pooling_fc_64_64_32_'
FILE = 'conv_k5_p2_64_64_max_0.3_drop_k5_p2_64_64_ave_0.3_drop_fc_64_64_32_'

'''
def main(model_type, FILE):
    iter = 0
    fout = open('result/' + FILE + model_type, 'w')

    line = "models\t"
    diff = 500
    maxIter = 200000

    for i in range(0, maxIter + 1, diff):
        line += str(i) + '\t'
    fout.write(line + '\n') # write the first line of the output: models 500 1000 1500 .....


    for distribution_num in [5, 20, 50]:
        for sample_dimension in [100, 400,  900]:        
            distribution_num = str(distribution_num)
            sample_dimension = str(sample_dimension)
    	    line = model_type + '_' + distribution_num + '_' + sample_dimension + '\t'
            f = open('./log/' + FILE + model_type  + '_dis_' + distribution_num  + '_par_' + sample_dimension + '.log')
            for l in f:
                l = l.strip()
                m = re.search('Test net output #0: (accuracy|loss) = ([\d.]+)', l)
                if not m: continue
                measure = m.group(1)
                score = m.group(2)
                iter += 500
                line += score + '\t'
            fout.write(line + '\n')

            line_par = 'joint_head_' + distribution_num + '_' + sample_dimension + '\t'
            line_dis = 'joint_head_' + distribution_num + '_' + sample_dimension + '\t'
            f = open('./log/' + FILE + 'joint_dis_' + distribution_num  + '_par_' + sample_dimension + '.log')
            print('./log/' + FILE + 'joint_dis_' + distribution_num  + '_par_' + sample_dimension + '.log')
            for l in f:
                l = l.strip()
                m = re.search('Test net output #0: (accuracy|loss) = ([\d.]+)', l)
                if not m: continue
                measure = m.group(1)
                score = m.group(2)
                if measure == 'loss':
                    line_par += score + '\t'
                elif measure == 'accuracy':
                    line_dis += score + '\t'
            if model_type == 'parameter':
                fout.write(line_par + '\n')
            if model_type == 'distribution':
                fout.write(line_dis + '\n')

main('distribution', FILE)
main('parameter', FILE)
'''

'''
# get the result for all the test for 50 100
def single(model_type):
    fout = open('./result/test_for_different_joint_structures_under_50_dis_100_sample_' + model_type, 'w')
    line = "models\t"
    diff = 500
    maxIter = 200000
    for i in range(0, maxIter + 1, diff):
        line += str(i) + '\t'
    fout.write(line + '\n') # write the first line of the output: models 500 1000 1500 .....

    for file in os.listdir('/home/deng106/caffeCNN/framework/log/'):
        #if not file.startswith('test_for_architectures'): continue
        m = re.search(r'(.+)(joint|distribution|parameter)_dis_50_par_[41]00.log', file)
        if not m: continue
        title = m.group(1)
        name = file.split('_joint_dis_50_par_100')[0]
        line = name + '\t'
        print name
        f = open('./log/' + file)
        for l in f:
            l = l.strip()
            m = re.search('Test net output #0: (accuracy|loss) = ([\d.]+)', l)
            if not m: continue
            measure = m.group(1)
            score = m.group(2)
            if model_type == 'parameter' and measure == 'loss':
                line += score + '\t'
            elif model_type == 'distribution' and measure == 'accuracy':
                line += score + '\t'
        fout.write(line + '\n')
'''
# get the result for all the test for 50 100
def single(model_type):
    fout = open('./result/test_for_different_joint_structures_under_50_dis_100_sample_' + model_type, 'w')
    line = "models\t"
    diff = 500
    maxIter = 200000
    for i in range(0, maxIter + 1, diff):
        line += str(i) + '\t'
    fout.write(line + '\n') # write the first line of the output: models 500 1000 1500 .....

    for file in os.listdir('/home/purduethu/caffeCNN/framework/log/'):
        #if not file.startswith('test_for_architectures'): continue

        #m = re.search(r'(.+)(joint|distribution|parameter)_dis_50_par_[41]00.log', file)
        #m = re.search(r'.*log.*', file)
        #if not m:
        #    print file
        #    continue
        ##title = m.group(1)
        name = file
        if model_type == "parameter" and file.find("distribution") >= 0: continue
        if model_type == "distribution" and file.find("parameter") >= 0: continue
        #name = file.split('_joint_dis_50_par_100')[0]
        line = name + '\t'
        #if file != "conv_k5_p2_64_64_max_k5_p2_128_128_128_ave_fc_1024_512_share_conv1_trial_3_parameter_dis_20_par_900.log":
        #    continue
        try:
            f = open('./log/' + file)
        except:
            continue
        for l in f:
            l = l.strip()
            m = re.search('Test net output #0: (accuracy|loss) = ([\d.]+)', l)
            if not m: continue
            measure = m.group(1)
            score = m.group(2)
            if model_type == 'parameter' and measure == 'loss':
                line += score + '\t'
            elif model_type == 'distribution' and measure == 'accuracy':
                line += score + '\t'
        #print line
        fout.write(line + '\n')

single('parameter')
single('distribution')

             
        



