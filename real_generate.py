import numpy as np
import sys, os
from numpy import genfromtxt

dirs = '/home/purduethu/scratch/radon/d/deng106/CNNStatisticalModel/distributions/input/'

data = genfromtxt(dirs + 'communities.data', delimiter=',')

dat = data[:900,5:96].T

data = np.empty([0,900])
for i in range(91):
    if sum(np.isnan(dat[i])) == 0:
        data = np.vstack((data, dat[i])) 
    else:
        print "missing col", i + 1

np.savetxt(dirs + 'input_real.csv', data, delimiter=',', fmt='%.2f')
