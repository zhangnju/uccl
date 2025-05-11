#!/bin/bash

# Compile NCCL for UCCL
# make src.build -j NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90" && cp src/include/nccl_common.h build/include/
# Compile NCCL-tests for UCCL
# make MPI=1 MPI_HOME=/opt/amazon/openmpi CUDA_HOME=/usr/local/cuda NCCL_HOME=/home/xzhiying/zhongjie_dev/nccl/build -j NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90"

# make MPI=1 MPI_HOME=/opt/amazon/openmpi CUDA_HOME=/usr/local/cuda NCCL_HOME=/home/xzhiying/uccl_rdma_zc/nccl/build -j

#SBATCH --job-name=nccl-tests
##SBATCH --output=nccl-tests.%j.out
#SBATCH --output=nccl-tests.alltoall.out
#SBATCH --error=nccl-tests.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00

ROOT_DIR=/home/xzhiying/zhongjie_dev

NVLINK_OFF=0

TEST=${1:-"uccl"}

if [ "$TEST" != "uccl" ] && [ "$TEST" != "nccl" ]; then
    echo "Error: 'uccl' or 'nccl' only"
    exit 1
fi

if [ "$TEST" = "nccl" ]; then
      LIBNCCL_PATH="/home/xzhiying/nccl/build/lib/libnccl.so"
      NCCL_TESTS_PATH="/home/xzhiying/nccl-tests/build"
      PLUGIN_PATH="/opt/amazon/ofi-nccl/lib/x86_64-linux-gnu/libnccl-net.so"

      NCCL_P2P_NET_CHUNKSIZE=524288
      NCCL_BUFFSIZE=8388608
else
      LIBNCCL_PATH="${ROOT_DIR}/nccl/build/lib/libnccl.so"
      NCCL_TESTS_PATH="${ROOT_DIR}/nccl-tests/build"
      PLUGIN_PATH="${ROOT_DIR}/efa/libnccl-net.so"

      LD_LIBRARY_PATH="/home/xzhiying/local_libs:${LD_LIBRARY_PATH}"

      NCCL_P2P_NET_CHUNKSIZE=131072
      NCCL_BUFFSIZE=1048576
fi

# alltoall_perf all_reduce_perf all_gather_perf
NCCL_TESTS_PATH="${NCCL_TESTS_PATH}/alltoall_perf"

export PATH=/opt/amazon/openmpi/bin:$PATH

# run the test binary
mpirun --bind-to none -np $SLURM_NNODES -N 1 \
      -x LD_LIBRARY_PATH=${LD_LIBRARY_PATH} \
      -x LD_PRELOAD="${LIBNCCL_PATH}" \
      -x NCCL_NET_PLUGIN="${PLUGIN_PATH}" \
      -x NCCL_P2P_NET_CHUNKSIZE=${NCCL_P2P_NET_CHUNKSIZE} \
      -x NCCL_BUFFSIZE=${NCCL_BUFFSIZE} \
      -x NCCL_P2P_DISABLE=${NVLINK_OFF} \
      -x NCCL_SHM_DISABLE=${NVLINK_OFF} \
      -x NCCL_NET_DISABLE=0 \
      ${NCCL_TESTS_PATH} \
      -f 2 -b 1K -e 4G \
      -w 5 -n 10 \
      -c 1 -g 1 -t 1

# -x NCCL_DEBUG=INFO \
# -x NCCL_P2P_DISABLE=1 \
# -x NCCL_SHM_DISABLE=1 \
# -x NCCL_NET_DISABLE=0 \

# scancel xxx
