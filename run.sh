#!/usr/bin/env sh

set -e

# generate target newwork
python set_up_net_par_for_dis_par.py
#python set_up_net_par_for_dis_par_multi_convs.py

# submit our tasks, you should better change output logs before do this
python submit_multi_tasks.py
