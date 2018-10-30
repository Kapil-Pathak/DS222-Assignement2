#!/bin/bash
python AsyncSGD_turing.py \
     --ps_hosts=10.24.1.211:2222,10.24.1.212:2223 \
     --worker_hosts=10.24.1.213:2222,10.24.1.214:2222 \
     --job_name=ps --task_index=0 &

python AsyncSGD_turing.py \
     --ps_hosts=10.24.1.211:2222,10.24.1.212:2223 \
     --worker_hosts=10.24.1.213:2222,10.24.1.214:2222 \
     --job_name=ps --task_index=1 &

python AsyncSGD_turing.py \
     --ps_hosts=10.24.1.211:2222,10.24.1.212:2223 \
     --worker_hosts=10.24.1.213:2222,10.24.1.214:2222 \
     --job_name=worker --task_index=0 &

python AsyncSGD_turing.py \
     --ps_hosts=10.24.1.211:2222,10.24.1.212:2223 \
     --worker_hosts=10.24.1.213:2222,10.24.1.214:2222 \
     --job_name=worker --task_index=1 &
