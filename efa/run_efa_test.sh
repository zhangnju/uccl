# !/bin/bash

./autogen.sh && ./configure CUDA_H_PATH=/usr/local/cuda/include/cuda.h && make -j

ib_send_bw -d rdmap110s0 --report_gbits -x 0 -c UD -t 128 -Q 1 -q 32 -l 2 -s 8192 -F --use_cuda=0
ib_send_bw -d rdmap110s0 --report_gbits -x 0 -c UD -t 128 -Q 1 -q 32 -l 2 -s 8192 -F --use_cuda=0 10.0.55.147

./transport_test --logtostderr=1 --clientip=10.0.58.244 --test=bimq
./transport_test --logtostderr=1 --serverip=10.0.50.44 --test=bimq