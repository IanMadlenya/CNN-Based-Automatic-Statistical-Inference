
# Prerequisite
Python2.7, caffe, R version > 3.3

# Compile New Layer
Huber Loss Layer in Caffe

# Generate data
```sh
sh main.sh 100 # generate 5, 20, 50 models with 100 sample size
sh main.sh 400 # generate 5, 20, 50 models with 400 sample size
sh main.sh 900 # generate 5, 20, 50 models with 900 sample size
```

# Run experiments
```python
python batch_running.py
```
