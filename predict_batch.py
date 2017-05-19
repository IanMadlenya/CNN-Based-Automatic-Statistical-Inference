import os,sys


'''
for i in ['100', '400', '900']:
    for j in ['5', '20', '50']:
        for t in ['distribution', 'joint']:
            print("**********************************************************", j, i)
            for k in ['1000', '1001', '1002']:
                for gpu in ['0', '1']:
                    pars = ' '.join([j, i, t, k, gpu])
                    os.system('/usr/bin/python predict.py ' + pars)
'''

for i in ['100', '400', '900']:
    for j in ['5', '20', '50']:
        os.system('/usr/bin/python predict_par.py ' + j + ' ' + i + ' joint')
        os.system('/usr/bin/python predict_par.py ' + j + ' ' + i + ' parameter')
