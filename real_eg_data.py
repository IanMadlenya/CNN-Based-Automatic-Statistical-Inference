import sys, os
import json
import numpy as np


samples = 900
dir = '/home/purduethu/scratch/radon/d/deng106/CNNStatisticalModel/distributions/input/'

with open(dir + 'stockPrices.json') as data_file:    
    data = json.load(data_file)

tickers = data.keys()

output = open(dir + 'input.real_example.' + str(samples), 'a+')
for ticker in tickers:
    num = 0
    print ticker
    line = []
    for day in sorted(data[ticker].iterkeys()):
        line.append(data[ticker][day])
        num += 1
        if num >= samples: break
    if num != 900: continue
    line = map(str, line)
    tmp = np.array(line)
    print tmp.shape
    line = ','.join(line) + '\n'
    output.write(line)
