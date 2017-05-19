import os, sys



'''
py batch_predict.py 100 distribution  '""'
py batch_predict.py 400 distribution 1
py batch_predict.py 400 distribution 2
py batch_predict.py 900 distribution 11
py batch_predict.py 900 distribution 12
py batch_predict.py 900 distribution 21
py batch_predict.py 900 distribution 22
'''

num_samples = sys.argv[1]
types = sys.argv[2]
suffix = sys.argv[3]


os.system('rm tmp.tmp')

for rand in ['1000', '1001', '1002']:
    for gpuid in ['0', '1']:
        pars = ' '.join([num_samples, types, suffix, rand, gpuid])
        print pars
        os.system('python predict.py 20 ' + pars + ' >> tmp.tmp')


os.system('cat tmp.tmp | grep "top 1 accuracy"')
os.system('cat tmp.tmp | grep "top 2 accuracy"')
