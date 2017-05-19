GPU=$1
dis=50
sps=100
tp=distribution
pre=conv_k5_p2_64_64_max_k5_p2_64_64_64_ave_fc_64_64_32_share_
pre=conv_k5_p2_64_64_max_k5_p2_128_128_128_ave_fc_1024_512_share_conv1_gpu_
pre=conv_64_max_128_128_fc_512_256_share_3_


sh train_net_qsub.sh $dis $sps $tp ${pre}${dis}_${sps}_${tp}_104_g${GPU} $GPU > ${pre}${dis}_${sps}_${tp}_104_g${GPU} 2>&1
