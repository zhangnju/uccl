#!/bin/bash

# make MPI=1 MPI_HOME=/opt/amazon/openmpi CUDA_HOME=/usr/local/cuda NCCL_HOME=/home/xzhiying/uccl_rdma_zc/nccl/build -j

#SBATCH --job-name=nccl-tests
#SBATCH --output=nccl-tests.%j.out
#SBATCH --error=nccl-tests.err
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --time=00:10:00

LIBNCCL_PATH="/home/xzhiying/nccl/build/lib/libnccl.so"
PLUGIN_PATH="/opt/amazon/ofi-nccl/lib/x86_64-linux-gnu/libnccl-net.so"

# run the test binary
mpirun --bind-to none -np $SLURM_NTASKS \
      -x LD_PRELOAD="${LIBNCCL_PATH} ${PLUGIN_PATH}" \
      -x NCCL_DEBUG=INFO \
      -x NCCL_PROTO=Simple \
      -x NCCL_P2P_DISABLE=1 \
      -x NCCL_SHM_DISABLE=1 \
      -x NCCL_NET_DISABLE=0 \
      -x NCCL_P2P_NET_CHUNKSIZE=524288 \
      -x NCCL_BUFFSIZE=8388608 \
      -x OFI_NCCL_PROTOCOL=SENDRECV \
      /home/xzhiying/nccl-tests/build/alltoall_perf \
      -b 1K -e 4G -f 2 -w 50 -n 100 -c 1 -g 1 -t 1

# alltoall_perf all_reduce_perf all_gather_perf
# -x NCCL_DEBUG=INFO \
# -x NCCL_P2P_DISABLE=1 \
# -x NCCL_SHM_DISABLE=1 \
# -x NCCL_NET_DISABLE=0 \
# -x NCCL_PXN_DISABLE=1 \

# scancel xxx
# salloc -N 1 -n 1 -t 00:30:00