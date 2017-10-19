This code is used for the experiments in the paper [R. Zhang, W. Deng, Michael Zhu, Using Deep Neural Networks to Automate Large Scale Statistical Analysis for Big Data Applications ACML 2017](https://arxiv.org/pdf/1708.03027.pdf). 

### Prerequisite
Linux, Python2.7, Caffe, R version > 3.3

### Compile New Layer
Huber Loss Layer in Caffe

### Generate data
```sh
sh main.sh 100 # generate 5, 20, 50 models with 100 sample size
sh main.sh 400 # generate 5, 20, 50 models with 400 sample size
sh main.sh 900 # generate 5, 20, 50 models with 900 sample size
```

### Protocol file in Caffe

The following protocol file is an example to run model selection based K=20, N=100 with large model. 
```
./protocol/dis_20_100_dimension.protocol_distribution
```

### Run experiments
```python
python batch_running.py
```

### Analysis

Log file is saved in the log folder, parsed data in saved in the result folder
