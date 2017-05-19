
# stat average loss in the last 50 iterations
num_dis=50
num_sample=900
type_model=joint
gpu=0
rand=1000


cat ./tmp/huber_1000_conv_k5_p2_64_64_max_128_128_128_fc_1024_512_share_5_layers_50_900_${type_model}_${rand}_gpu${gpu} | grep "429]     Test net output #0: loss = "  | awk -F "loss\ =\ " '{print $2}' | tail -50 | awk -F "\ " '{print $1}' | awk '{ sum += $1; n++ } END { if (n > 0) print sum / n; }'
