#!/usr/bin/env sh
set -e
# prepare data

r="/usr/bin/Rscript"

sample_number=1000
sample_dimension=$1
factor=1
server=radon

DIR="/home/purduethu/scratch/radon/d/deng106/CNNStatisticalModel/distributions/data/"
newPath=${DIR}${sample_dimension}"_sample_dimension/"
echo $newPath

#:<<block
mkdir -p $newPath
$r genData_v2.R $sample_number $sample_dimension $factor $newPath
#$r tmpData_v2.R $sample_number $sample_dimension $newPath
#block
cat ${newPath}feature_*_[0-4].txt > ${DIR}feature_5_distributions_${sample_dimension}
cat ${newPath}feature_*_1?.txt ${newPath}feature_*_?.txt > ${DIR}feature_20_distributions_$sample_dimension
cat ${newPath}feature_* > ${DIR}feature_50_distributions_${sample_dimension}

cat ${newPath}label_*_[0-4].txt > ${DIR}label_5_distributions_${sample_dimension}
cat ${newPath}label_*_1?.txt ${newPath}label_*_?.txt > ${DIR}label_20_distributions_${sample_dimension}
cat ${newPath}label_* > ${DIR}label_50_distributions_${sample_dimension}


python hf5.py 5 $sample_dimension $server &
python hf5.py 20 $sample_dimension $server &
python hf5.py 50 $sample_dimension $server &

