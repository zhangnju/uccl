#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstring>
#include <unistd.h>
#include <thread>
#include <assert.h>

#include <infiniband/verbs.h>
#include <hip/hip_runtime.h>
#include <glog/logging.h>

//////////////////////////////////////////////////////////////////////
#define DEV_INDEX 0           // rdma0
#define USE_GPU_SRC 0         // Source GPU
#define USE_GPU_DST 0         // Destination GPU  
#define OUTSTNADING_MSG 4     // Number of outstanding messages
#define ITERATIONS 1000       // Number of iterations per size
#define ROCE_NET true        // true: RoCE, false: IB
//////////////////////////////////////////////////////////////////////

#if ROCE_NET
static constexpr uint8_t GID_INDEX = 0;
#else
static constexpr uint8_t GID_INDEX = 3;
#endif
#define PORT_NUM 1
#define MAX_WR 1024
#define BASE_PSN 0x12345

struct metadata {
  uint32_t qpn[2];  // Two QPs for loopback
  uint32_t rkey_src;
  uint32_t rkey_dst;
  union ibv_gid gid;
  uint64_t addr_src;
  uint64_t addr_dst;
  uint16_t lid;
};

struct rdma_context {
  struct ibv_context* ctx;
  struct ibv_pd* pd;
  struct ibv_cq* cq;
  struct ibv_mr* mr_src;  // Source GPU memory
  struct ibv_mr* mr_dst;  // Destination GPU memory

  char* local_buf_src;
  char* local_buf_dst;

  struct ibv_qp* qp[2];  // Two QPs: qp[0] sends to qp[1], qp[1] receives from qp[0]
  struct metadata local_meta;

  union ibv_gid gid;
  struct ibv_device_attr dev_attr;
  struct ibv_port_attr port_attr;
};

static bool force_exit = false;
static uint32_t next_data_offset = 0;

struct ibv_recv_wr wr[MAX_WR];

class RDMAGPUBenchmark {
private:
    rdma_context* rdma_;
    size_t max_buf_size_;
    
    // Performance tracking
    struct perf_stats {
        size_t msg_size;
        double throughput_gbps;
        double latency_us;
        uint64_t iterations;
    };
    
public:
    RDMAGPUBenchmark(size_t max_size = 1ULL * 1024 * 1024 * 1024) 
        : max_buf_size_(max_size) {
        rdma_ = init_rdma();
    }
    
    ~RDMAGPUBenchmark() {
        cleanup();
    }
    
private:
    void create_cq(rdma_context* rdma) {
        CHECK(rdma != nullptr) << "RDMA context is null";
        CHECK(rdma->ctx != nullptr) << "RDMA device context is null";
        
        rdma->cq = ibv_create_cq(rdma->ctx, 16384, nullptr, nullptr, 0);
        CHECK(rdma->cq != nullptr) << "Failed to create CQ";
    }
    
    void create_qp(rdma_context* rdma) {
        CHECK(rdma != nullptr) << "RDMA context is null";
        CHECK(rdma->pd != nullptr) << "Protection domain is null";
        CHECK(rdma->cq != nullptr) << "Completion queue is null";
        
        struct ibv_qp_init_attr attr = {};
        attr.qp_context = nullptr;
        attr.send_cq = rdma->cq;
        attr.recv_cq = rdma->cq;
        attr.qp_type = IBV_QPT_RC;
        attr.cap.max_send_wr = MAX_WR;
        attr.cap.max_recv_wr = MAX_WR;
        attr.cap.max_send_sge = 1;
        attr.cap.max_recv_sge = 1;
        attr.cap.max_inline_data = 0;
        
        // Create two QPs for loopback communication
        rdma->qp[0] = ibv_create_qp(rdma->pd, &attr);
        CHECK(rdma->qp[0] != nullptr) << "Failed to create QP[0]";
        
        rdma->qp[1] = ibv_create_qp(rdma->pd, &attr);
        CHECK(rdma->qp[1] != nullptr) << "Failed to create QP[1]";
        
        struct ibv_qp_attr qp_attr = {};
        qp_attr.qp_state = IBV_QPS_INIT;
        qp_attr.pkey_index = 0;
        qp_attr.port_num = PORT_NUM;
        qp_attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE;
        
        // Initialize both QPs to INIT state
        CHECK_EQ(ibv_modify_qp(rdma->qp[0], &qp_attr,
                               IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS), 0)
            << "Failed to modify QP[0] to INIT";
            
        CHECK_EQ(ibv_modify_qp(rdma->qp[1], &qp_attr,
                               IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS), 0)
            << "Failed to modify QP[1] to INIT";
    }
    
    void modify_qp_rtr() {
        auto& local_meta = rdma_->local_meta;
        struct ibv_qp_attr attr = {};
        int attr_mask = IBV_QP_STATE | IBV_QP_PATH_MTU | IBV_QP_AV | IBV_QP_DEST_QPN |
                        IBV_QP_RQ_PSN | IBV_QP_MIN_RNR_TIMER | IBV_QP_MAX_DEST_RD_ATOMIC;
        
        attr.qp_state = IBV_QPS_RTR;
        attr.path_mtu = rdma_->port_attr.active_mtu;
        
#if ROCE_NET
        attr.ah_attr.is_global = 1;
        attr.ah_attr.grh.dgid = local_meta.gid;
        attr.ah_attr.grh.sgid_index = GID_INDEX;
        attr.ah_attr.grh.hop_limit = 0xff;
        attr.ah_attr.grh.traffic_class = 0;
#else
        attr.ah_attr.is_global = 0;
        attr.ah_attr.dlid = local_meta.lid;
#endif
        
        attr.ah_attr.sl = 0;
        attr.ah_attr.port_num = PORT_NUM;
        attr.rq_psn = BASE_PSN;
        attr.min_rnr_timer = 12;
        attr.max_dest_rd_atomic = 1;
        
        // QP[0] connects to QP[1]
        attr.dest_qp_num = local_meta.qpn[1];
        int ret = ibv_modify_qp(rdma_->qp[0], &attr, attr_mask);
        CHECK_EQ(ret, 0) << "Failed to modify QP[0] to RTR: " << ret;
        
        // QP[1] connects to QP[0] (for bidirectional capability)
        attr.dest_qp_num = local_meta.qpn[0];
        ret = ibv_modify_qp(rdma_->qp[1], &attr, attr_mask);
        CHECK_EQ(ret, 0) << "Failed to modify QP[1] to RTR: " << ret;
    }
    
    void modify_qp_rts() {
        struct ibv_qp_attr attr = {};
        int attr_mask = IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_TIMEOUT |
                        IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_MAX_QP_RD_ATOMIC;
        
        attr.qp_state = IBV_QPS_RTS;
        attr.sq_psn = BASE_PSN;
        attr.timeout = 14;
        attr.retry_cnt = 7;
        attr.rnr_retry = 7;
        attr.max_rd_atomic = 1;
        
        // Move both QPs to RTS state
        int ret = ibv_modify_qp(rdma_->qp[0], &attr, attr_mask);
        CHECK_EQ(ret, 0) << "Failed to modify QP[0] to RTS: " << ret;
        
        ret = ibv_modify_qp(rdma_->qp[1], &attr, attr_mask);
        CHECK_EQ(ret, 0) << "Failed to modify QP[1] to RTS: " << ret;
    }
    
    rdma_context* init_rdma() {
        rdma_ = (rdma_context*)calloc(1, sizeof(rdma_context));
        
        int nb_devices;
        struct ibv_device** dev_list = ibv_get_device_list(&nb_devices);
        CHECK(dev_list != nullptr) << "Failed to get device list";
        
        char device_name[32];
        sprintf(device_name, "rdma%d", DEV_INDEX);
        
        int i;
        for (i = 0; i < nb_devices; i++) {
            if (strcmp(ibv_get_device_name(dev_list[i]), device_name) == 0) {
                break;
            }
        }
        CHECK_LT(i, nb_devices) << "Device " << device_name << " not found";
        
        auto* open_dev = dev_list[i];
        rdma_->ctx = ibv_open_device(open_dev);
        CHECK(rdma_->ctx != nullptr) << "Failed to open device";
        
        ibv_free_device_list(dev_list);
        
        rdma_->pd = ibv_alloc_pd(rdma_->ctx);
        CHECK(rdma_->pd != nullptr) << "Failed to allocate PD";
        
        CHECK_EQ(ibv_query_device(rdma_->ctx, &rdma_->dev_attr), 0) << "Failed to query device";
        CHECK_EQ(ibv_query_port(rdma_->ctx, 1, &rdma_->port_attr), 0) << "Failed to query port";
        CHECK_EQ(rdma_->port_attr.state, IBV_PORT_ACTIVE) << "Port is not active";
        CHECK_EQ(ibv_query_gid(rdma_->ctx, 1, GID_INDEX, &rdma_->gid), 0) << "Failed to query GID";
        
        // Allocate GPU memory
        CHECK_EQ(hipSetDevice(USE_GPU_SRC), hipSuccess) << "Failed to set source GPU";
        CHECK_EQ(hipMalloc((void**)&rdma_->local_buf_src, max_buf_size_), hipSuccess) 
            << "Failed to allocate source GPU memory";
        CHECK_EQ(hipMemset(rdma_->local_buf_src, 0xAB, max_buf_size_), hipSuccess);
        
        CHECK_EQ(hipSetDevice(USE_GPU_DST), hipSuccess) << "Failed to set destination GPU";
        CHECK_EQ(hipMalloc((void**)&rdma_->local_buf_dst, max_buf_size_), hipSuccess) 
            << "Failed to allocate destination GPU memory";
        CHECK_EQ(hipMemset(rdma_->local_buf_dst, 0x00, max_buf_size_), hipSuccess);
        
        // Register GPU memory with RDMA
        rdma_->mr_src = ibv_reg_mr(rdma_->pd, rdma_->local_buf_src, max_buf_size_,
                                  IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                                  IBV_ACCESS_REMOTE_READ | IBV_ACCESS_RELAXED_ORDERING);
        CHECK(rdma_->mr_src != nullptr) << "Failed to register source GPU memory";
        
        rdma_->mr_dst = ibv_reg_mr(rdma_->pd, rdma_->local_buf_dst, max_buf_size_,
                                  IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                                  IBV_ACCESS_REMOTE_READ | IBV_ACCESS_RELAXED_ORDERING);
        CHECK(rdma_->mr_dst != nullptr) << "Failed to register destination GPU memory";
        
        LOG(INFO) << "Source MR: addr=" << (uint64_t)rdma_->mr_src->addr
                  << ", len=" << rdma_->mr_src->length << ", lkey=" << rdma_->mr_src->lkey
                  << ", rkey=" << rdma_->mr_src->rkey;
        LOG(INFO) << "Destination MR: addr=" << (uint64_t)rdma_->mr_dst->addr
                  << ", len=" << rdma_->mr_dst->length << ", lkey=" << rdma_->mr_dst->lkey
                  << ", rkey=" << rdma_->mr_dst->rkey;
        
        create_cq(rdma_);
        create_qp(rdma_);
        
        // Setup local metadata for loopback
        rdma_->local_meta.qpn[0] = rdma_->qp[0]->qp_num;
        rdma_->local_meta.qpn[1] = rdma_->qp[1]->qp_num;
        rdma_->local_meta.rkey_src = rdma_->mr_src->rkey;
        rdma_->local_meta.rkey_dst = rdma_->mr_dst->rkey;
        rdma_->local_meta.addr_src = (uint64_t)rdma_->local_buf_src;
        rdma_->local_meta.addr_dst = (uint64_t)rdma_->local_buf_dst;
        rdma_->local_meta.lid = rdma_->port_attr.lid;
        memcpy(&rdma_->local_meta.gid, &rdma_->gid, sizeof(rdma_->local_meta.gid));
        
        modify_qp_rtr();
        modify_qp_rts();
        
        LOG(INFO) << "RDMA loopback initialized successfully";
        return rdma_;
    }
    
    uint32_t poll_cq(uint32_t expect_opcode) {
        auto total_ops = 0;
        auto* cq = rdma_->cq;
        
        struct ibv_wc wc;
        int ne = ibv_poll_cq(cq, 1, &wc);
        
        while (ne > 0) {
            CHECK_EQ(wc.status, IBV_WC_SUCCESS) << "Failed to poll CQ: " << wc.status;
            CHECK_EQ(wc.opcode, expect_opcode) 
                << "Unexpected opcode: " << wc.opcode;
            total_ops++;
            ne = ibv_poll_cq(cq, 1, &wc);
        }
        
        return total_ops;
    }
    
    void send_message(size_t msg_size) {
        auto* data = rdma_->local_buf_src + next_data_offset;
        auto& local_meta = rdma_->local_meta;
        
        struct ibv_send_wr wr = {};
        struct ibv_sge sge = {};
        
        wr.wr.rdma.remote_addr = local_meta.addr_dst + next_data_offset;
        wr.wr.rdma.rkey = local_meta.rkey_dst;
        
        sge.lkey = rdma_->mr_src->lkey;
        sge.addr = (uint64_t)data;
        sge.length = msg_size;
        
        wr.sg_list = &sge;
        wr.num_sge = 1;
        wr.opcode = IBV_WR_RDMA_WRITE;
        wr.send_flags = IBV_SEND_SIGNALED;
        
        struct ibv_send_wr* bad_wr;
        CHECK_EQ(ibv_post_send(rdma_->qp[0], &wr, &bad_wr), 0) 
            << "Failed to post RDMA WRITE";
        
        next_data_offset = (next_data_offset + msg_size) % (msg_size * OUTSTNADING_MSG);
    }
    
public:
    perf_stats benchmark_size(size_t msg_size) {
        const uint64_t warmup_iters = 100;
        const uint64_t test_iters = ITERATIONS;
        
        LOG(INFO) << "Benchmarking message size: " << msg_size << " bytes";
        
        // Warmup
        const int max_inflight = OUTSTNADING_MSG;
        int inflight = 0;
        
        for (uint64_t i = 0; i < warmup_iters; i++) {
            while (inflight >= max_inflight) {
                auto total_ops = poll_cq(IBV_WC_RDMA_WRITE);
                inflight -= total_ops;
            }
            send_message(msg_size);
            inflight++;
        }
        
        while (inflight > 0) {
            auto total_ops = poll_cq(IBV_WC_RDMA_WRITE);
            inflight -= total_ops;
        }
        
        // Actual benchmark
        auto start = std::chrono::high_resolution_clock::now();
        inflight = 0;
        
        for (uint64_t i = 0; i < test_iters; i++) {
            while (inflight >= max_inflight) {
                auto total_ops = poll_cq(IBV_WC_RDMA_WRITE);
                inflight -= total_ops;
            }
            send_message(msg_size);
            inflight++;
        }
        
        while (inflight > 0) {
            auto total_ops = poll_cq(IBV_WC_RDMA_WRITE);
            inflight -= total_ops;
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_sec = std::chrono::duration<double>(end - start).count();
        
        perf_stats stats;
        stats.msg_size = msg_size;
        stats.iterations = test_iters;
        stats.latency_us = (elapsed_sec * 1e6) / test_iters;
        
        double total_bytes = static_cast<double>(msg_size) * test_iters;
        double throughput_bps = total_bytes / elapsed_sec;
        stats.throughput_gbps = throughput_bps / (1024.0 * 1024.0 * 1024.0);
        
        return stats;
    }
    
    void run_benchmark() {
        std::vector<perf_stats> results;
        
        std::cout << "\n=== RDMA GPU-to-GPU Loopback Benchmark ===" << std::endl;
        std::cout << "GPU" << USE_GPU_SRC << " -> GPU" << USE_GPU_DST 
                  << " via RDMA (rdma" << DEV_INDEX << " NIC)" << std::endl;
        std::cout << std::setw(12) << "Size"
                  << std::setw(15) << "Throughput"
                  << std::setw(15) << "Latency"
                  << std::setw(12) << "Iterations" << std::endl;
        std::cout << std::setw(12) << "(Bytes)"
                  << std::setw(15) << "(GB/s)"
                  << std::setw(15) << "(μs)"
                  << std::setw(12) << "(Count)" << std::endl;
        std::cout << std::string(54, '-') << std::endl;
        
        // Test sizes from 1KB to 1GB with step size of 2
        for (size_t size = 1024; size <= max_buf_size_ && size <= (1ULL << 30); size *= 2) {
            auto stats = benchmark_size(size);
            results.push_back(stats);
            
            std::cout << std::setw(12) << stats.msg_size
                      << std::setw(15) << std::fixed << std::setprecision(2) 
                      << stats.throughput_gbps
                      << std::setw(15) << std::fixed << std::setprecision(2) 
                      << stats.latency_us
                      << std::setw(12) << stats.iterations << std::endl;
        }
        
        // Verify data integrity for the last transfer
        verify_data_integrity(results.back().msg_size);
        
        std::cout << "\nBenchmark completed successfully!" << std::endl;
    }
    
    void verify_data_integrity(size_t size) {
        LOG(INFO) << "Verifying data integrity for size: " << size;
        
        // Copy data from destination GPU to host for verification
        std::vector<uint8_t> host_buf(size);
        CHECK_EQ(hipSetDevice(USE_GPU_DST), hipSuccess);
        CHECK_EQ(hipMemcpy(host_buf.data(), rdma_->local_buf_dst, size, hipMemcpyDeviceToHost), 
                 hipSuccess);
        
        // Check if data matches expected pattern (0xAB)
        for (size_t i = 0; i < size; i++) {
            if (host_buf[i] != 0xAB) {
                LOG(ERROR) << "Data mismatch at byte " << i 
                          << ": expected 0xAB, got 0x" << std::hex << (int)host_buf[i];
                return;
            }
        }
        
        LOG(INFO) << "Data integrity verified successfully!";
    }
    
    void cleanup() {
        if (!rdma_) return;
        
        if (rdma_->qp[0]) ibv_destroy_qp(rdma_->qp[0]);
        if (rdma_->qp[1]) ibv_destroy_qp(rdma_->qp[1]);
        if (rdma_->cq) ibv_destroy_cq(rdma_->cq);
        if (rdma_->mr_src) ibv_dereg_mr(rdma_->mr_src);
        if (rdma_->mr_dst) ibv_dereg_mr(rdma_->mr_dst);
        if (rdma_->pd) ibv_dealloc_pd(rdma_->pd);
        if (rdma_->ctx) ibv_close_device(rdma_->ctx);
        
        if (rdma_->local_buf_src) {
            auto ret = hipSetDevice(USE_GPU_SRC);
            if (ret == hipSuccess) {
                ret = hipFree(rdma_->local_buf_src);
                (void)ret;  // Suppress unused variable warning
            }
        }
        if (rdma_->local_buf_dst) {
            auto ret = hipSetDevice(USE_GPU_DST);
            if (ret == hipSuccess) {
                ret = hipFree(rdma_->local_buf_dst);
                (void)ret;  // Suppress unused variable warning
            }
        }
        
        free(rdma_);
        rdma_ = nullptr;
    }
};

// TO RUN: 
// LD_LIBRARY_PATH=${HOME}/anaconda3/lib:${LD_LIBRARY_PATH} ./rdma_loopback

int main(int argc, char* argv[]) {
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = 1;
    
    try {
        LOG(INFO) << "Starting RDMA GPU-to-GPU loopback benchmark";
        
        RDMAGPUBenchmark benchmark;
        
        LOG(INFO) << "Starting benchmark...";
        benchmark.run_benchmark();
        
        LOG(INFO) << "Benchmark completed successfully";
        
    } catch (const std::exception& e) {
        LOG(ERROR) << "Error: " << e.what();
        return 1;
    }
    
    return 0;
}
