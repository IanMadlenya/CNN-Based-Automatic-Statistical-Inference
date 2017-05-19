#!/usr/bin/env sh
#set -e

num_distributions=$1
sample_dimension=$2
model_type=$3
architecture=$4
NOgpu=$5

echo "Number of distributions: " ${num_distributions} 
echo "Sample dimension: " ${sample_dimension}
echo "model type: " ${model_type}
echo "model architecture: " ${architecture}

# generate protocol and solver file based on custom parameters
python reProtocol_pars.py $num_distributions $sample_dimension distribution $architecture radon
python reProtocol_pars.py $num_distributions $sample_dimension parameter $architecture radon
python reProtocol_pars.py $num_distributions $sample_dimension joint $architecture radon

SCRATCH=/home/purduethu/scratch/radon/d/deng106/CNNStatisticalModel/distributions/models/
BASE_DIR=${SCRATCH}${model_type}/${num_distributions}_dis_${sample_dimension}_dim/${architecture}/_iter_
SOLVER=./protocol/dis_${num_distributions}_${sample_dimension}_dimension.solver_${model_type}
CAFFE=/home/purduethu/src/nccl/caffe-0.15.9/build/tools/caffe

# 4 input parameters, start training from scratch
# input format, e.g. 50 900 parameter conv_k5_p2_64_64_max_k5_p2_64_64_ave_fc_64_64_32_
#
# 5 input parameters, continue training from solverstate
# input format, e.g. 50 900 distribution conv_k5_p2_64_64_max_k5_p2_64_64_ave_fc_64_64_32_ 15000
#
# 6 input parameters, continue training using shared weights


if [ "$3" = "joint" ]; then
    echo "Joint layer estimation"
    python shared_conv_robust.py $num_distributions $sample_dimension $architecture $NOgpu
else
    #$CAFFE train -solver $SOLVER
    $CAFFE train -solver $SOLVER -gpu $NOgpu
    #$CAFFE train -solver $SOLVER --gpu=0,1
fi
:<<block
if [ $# == 4 ]; then
    if [ $3 == joint ]; then 
        echo "Joint layer estimation"
        python shared_conv_robust.py $num_distributions $sample_dimension $architecture
    else
        $CAFFE train -solver $SOLVER
    fi
elif [ $# == 5 ]; then
    if [ $3 == joint ]; then
        echo "Joint layer estimation"
        python shared_conv_robust.py $num_distributions $sample_dimension ${4} 0.00005
    else
        SNAPSHOT=${BASE_DIR}${4}.solverstate
        $CAFFE train -solver $SOLVER -snapshot $SNAPSHOT
    fi
elif [ $# == 6 ]; then
    MODEL=${BASE_DIR}0.caffemodel
    $CAFFE train -solver $SOLVER -weights $MODEL
fi
block
