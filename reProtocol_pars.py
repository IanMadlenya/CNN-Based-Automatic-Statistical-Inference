import sys
import os

distribution_num = sys.argv[1]
sample_dimension = sys.argv[2]
model_type = sys.argv[3]
architecture = sys.argv[4]
server_name = 'radon'

print(model_type)
print(architecture)



prefix_name = 'protocol/dis_' + distribution_num + "_" + sample_dimension + "_dimension"
BASE = '/home/purduethu/scratch/' + server_name + '/d/deng106/CNNStatisticalModel/distributions/models/'

if model_type != 'joint':
    # generate protocol file for cases of par and dis
    f = open('protocol/' + model_type + '.baseline')
    out = open(prefix_name + '.protocol_' + model_type, 'w')

    dir = '"/home/purduethu/scratch/' + server_name + '/d/deng106/CNNStatisticalModel/distributions/input/'

    phase = "train"
    tag_LastLayer = False

    for l in f:
        line = l.strip()
        if line == "phase: TEST":
            phase = "test"
        if line.startswith('source:'):
            out.write('    source: ' + dir + distribution_num + "_distributions_" + sample_dimension + "_sample_size_" + phase + '-' + model_type + 's-path.txt"\n')
        elif line.startswith('######'):
            tag_LastLayer = True
        elif tag_LastLayer and line.startswith('num_output:'): # work only when we reach the last layer
            out.write('     num_output: ' + distribution_num + '\n')
        elif len(line) != 0:
            if l[-1] == '\n': l = l[:-1]
            out.write(l + '\n')
        else:
            out.write(l)

    # generate solver file
    f = open('protocol/' + model_type + '.solver')
    out = open(prefix_name + '.solver_' + model_type, 'w')
    DIR = '"' + BASE + model_type + '/' + distribution_num + '_dis_' + sample_dimension + '_dim/' + architecture + '/"'
    if os.path.isdir(DIR[1:-1]): # delete double quotion marks
        os.system('rm -r ' + DIR[1:-1]) # delete that directory if it exists
    os.system('mkdir -p ' + DIR[1:-1]) # generate target dir
       


    for l in f:
        line = l.strip()
        if line.startswith("net: "):
            out.write('net: "./protocol/' + 'dis_' + distribution_num + '_' + sample_dimension + '_dimension.protocol_' + model_type + '"\n')
        elif line.startswith("snapshot_prefix:"):
            out.write('snapshot_prefix: ' + DIR + '\n')
        else:
            out.write(line + '\n')
elif model_type == 'joint':
    for inner in ['distribution', 'parameter']:
        f = open('protocol/joint.solver')
        out = open(prefix_name + '.solver_joint_' + inner, 'w')
        DIR = '"' + BASE + 'joint/' + inner + '/' + distribution_num + '_dis_' + sample_dimension + '_dim/' + architecture + '/"'
        if os.path.isdir(DIR[1:-1]): # delete double quotion marks
            os.system('rm -r ' + DIR[1:-1]) # delete that directory if it exists
        os.system('mkdir -p ' + DIR[1:-1]) # generate target dir
        
        for l in f:
            line = l.strip()
            if line.startswith("net: "):
                out.write('net: "./protocol/' + 'dis_' + distribution_num + '_' + sample_dimension + '_dimension.protocol_' + inner + '"\n')
            elif line.startswith("snapshot_prefix:"):
                out.write('snapshot_prefix: ' + DIR + '\n')
            else:
                out.write(line + '\n')
        
        
    
