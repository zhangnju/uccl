#include "util_efa.h"

#include "transport_config.h"

namespace uccl {

EFAFactory efa_ctl;

void EFAFactory::Init() {
    for (int pdev_idx = 0; pdev_idx < NUM_DEVICES; pdev_idx++) {
        EFAFactory::InitDev(pdev_idx);
    }
}

void EFAFactory::InitDev(int pdev_idx) {
    struct EFADevice *dev = new struct EFADevice();
    struct ibv_device **device_list;
    struct ibv_context *context;
    struct ibv_device_attr dev_attr;
    struct ibv_port_attr port_attr;
    int i, nb_devices;

    // Check if the device is already initialized.
    DCHECK(efa_ctl.dev_map.find(pdev_idx) == efa_ctl.dev_map.end());

    // Get Infiniband name from pdev_idx.
    DCHECK(util_efa_get_ib_name_from_dev_idx(pdev_idx, dev->ib_name) == 0);

    // Get IP address from pdev_idx.
    DCHECK(util_efa_get_ip_from_dev_idx(pdev_idx, &dev->local_ip_str) == 0);

    // Get the list of RDMA devices.
    device_list = ibv_get_device_list(&nb_devices);
    if (device_list == nullptr || nb_devices == 0) {
        perror("ibv_get_device_list");
        goto error;
    }

    // Find the device by name.
    for (i = 0; i < nb_devices; i++) {
        if (strcmp(ibv_get_device_name(device_list[i]), dev->ib_name) == 0) {
            break;
        }
    }
    if (i == nb_devices) {
        fprintf(stderr, "No device found for %s\n", dev->ib_name);
        goto free_devices;
    }

    LOG(INFO) << "Found device: " << dev->ib_name << " at pdev_idx " << i
              << " with gid_idx " << (uint32_t)EFA_GID_IDX;

    // Open the device.
    memset(&dev_attr, 0, sizeof(dev_attr));
    if ((context = ibv_open_device(device_list[i])) == nullptr) {
        perror("ibv_open_device");
        goto free_devices;
    }

    if (ibv_query_device(context, &dev_attr)) {
        perror("ibv_query_device");
        goto close_device;
    }

    // Currently, we only use one port.
    if (dev_attr.phys_port_cnt != EFA_PORT_NUM /* 1 */) {
        fprintf(stderr, "Only one port is supported\n");
        goto close_device;
    }

    // Port number starts from 1.
    if (ibv_query_port(context, 1, &port_attr)) {
        perror("ibv_query_port");
        goto close_device;
    }

    if (port_attr.state != IBV_PORT_ACTIVE) {
        fprintf(stderr, "Port is not active\n");
        goto close_device;
    }

    if (port_attr.link_layer != IBV_LINK_LAYER_UNSPECIFIED) {
        fprintf(stderr, "EFA link layer is not supported\n");
        goto close_device;
    }

    dev->dev_attr = dev_attr;
    dev->port_attr = port_attr;
    dev->pdev_idx = pdev_idx;
    dev->efa_port_num = EFA_PORT_NUM;
    dev->context = context;

    if (ibv_query_gid(context, EFA_PORT_NUM, EFA_GID_IDX, &dev->gid)) {
        perror("ibv_query_gid");
        goto close_device;
    }

    // Allocate a PD for this device.
    dev->pd = ibv_alloc_pd(context);
    if (dev->pd == nullptr) {
        perror("ibv_alloc_pd");
        goto close_device;
    }

    // Detect DMA-BUF support.
    {
        struct ibv_pd *pd;
        pd = ibv_alloc_pd(context);
        if (pd == nullptr) {
            perror("ibv_alloc_pd");
            goto close_device;
        }
        // Test kernel DMA-BUF support with a dummy call (fd=-1)
        (void)ibv_reg_dmabuf_mr(pd, 0ULL /*offset*/, 0ULL /*len*/,
                                0ULL /*iova*/, -1 /*fd*/, 0 /*flags*/);
        dev->dma_buf_support =
            !((errno == EOPNOTSUPP) || (errno == EPROTONOSUPPORT));
        ibv_dealloc_pd(pd);

        VLOG(5) << "DMA-BUF support: " << dev->dma_buf_support;
    }

    // Detect hardware timestamp support.
    {
        struct ibv_cq_init_attr_ex cq_ex_attr;
        cq_ex_attr.cqe = 1024;
        cq_ex_attr.cq_context = nullptr;
        cq_ex_attr.channel = nullptr;
        cq_ex_attr.comp_vector = 0;
        cq_ex_attr.wc_flags =
            IBV_WC_EX_WITH_COMPLETION_TIMESTAMP;  // Timestamp support.
        cq_ex_attr.comp_mask = 0;
        cq_ex_attr.flags = 0;

        auto send_cq_ex_ = ibv_create_cq_ex(context, &cq_ex_attr);
        if (send_cq_ex_ == nullptr) {
            VLOG(5) << "HW timestamp not supported";
        } else {
            VLOG(5) << "HW timestamp supported";
            ibv_destroy_cq(ibv_cq_ex_to_cq(send_cq_ex_));
        }
    }

    efa_ctl.dev_map.insert({pdev_idx, dev});
    return;

close_device:
    ibv_close_device(context);
free_devices:
    ibv_free_device_list(device_list);
error:
    throw std::runtime_error("Failed to initialize EFAFactory");
}

EFASocket *EFAFactory::CreateSocket(int gpu_idx, int pdev_idx, int socket_idx) {
    std::lock_guard<std::mutex> lock(efa_ctl.socket_q_lock_);
    // Yang: magic here---we need to keep QPN allocated for each socket
    // continuous, so that they can be evenly distirbuted to differnet EFA
    // microcores; otherwise, interleaved QPNs for different sockets will cause
    // unstable performance.
    auto socket = new EFASocket(gpu_idx, pdev_idx, socket_idx);
    efa_ctl.socket_q_.push_back(socket);
    return socket;
}

struct EFADevice *EFAFactory::GetEFADevice(int pdev_idx) {
    auto dev_iter = efa_ctl.dev_map.find(pdev_idx);
    DCHECK(dev_iter != efa_ctl.dev_map.end()) << "pdev_idx " << pdev_idx;
    auto *dev = dev_iter->second;
    return dev;
}

void EFAFactory::Shutdown() {
    std::lock_guard<std::mutex> lock(efa_ctl.socket_q_lock_);
    for (auto socket : efa_ctl.socket_q_) {
        delete socket;
    }
    efa_ctl.socket_q_.clear();

    efa_ctl.dev_map.clear();
}

EFASocket::EFASocket(int gpu_idx, int pdev_idx, int socket_idx)
    : gpu_idx_(gpu_idx), dev_idx_(pdev_idx), socket_idx_(socket_idx) {
    LOG(INFO) << "[EFA] creating gpu_idx " << gpu_idx << " dev_idx_ "
              << dev_idx_ << " socket_idx " << socket_idx;

    memset(deficit_cnt_recv_wrs_, 0, sizeof(deficit_cnt_recv_wrs_));
    memset(deficit_cnt_recv_wrs_for_ctrl_, 0,
           sizeof(deficit_cnt_recv_wrs_for_ctrl_));

    auto *factory_dev = EFAFactory::GetEFADevice(pdev_idx);
    context_ = factory_dev->context;
    pd_ = factory_dev->pd;
    gid_ = factory_dev->gid;

    // Allocate memory for packet headers.
    void *pkt_hdr_buf_ =
        mmap(nullptr, PktHdrBuffPool::kNumPktHdr * PktHdrBuffPool::kPktHdrSize,
             PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    DCHECK(pkt_hdr_buf_ != nullptr) << "aligned_alloc failed";
    auto *pkt_hdr_mr_ =
        ibv_reg_mr(pd_, pkt_hdr_buf_,
                   PktHdrBuffPool::kNumPktHdr * PktHdrBuffPool::kPktHdrSize,
                   IBV_ACCESS_LOCAL_WRITE);
    DCHECK(pkt_hdr_mr_ != nullptr) << "ibv_reg_mr failed";
    pkt_hdr_pool_ = new PktHdrBuffPool(pkt_hdr_mr_);

    auto old_gpu_idx = 0;
    auto ret = cudaGetDevice(&old_gpu_idx);
    CHECK(ret == cudaSuccess) << "cudaGetDevice failed ";
    ret = cudaSetDevice(gpu_idx);
    CHECK(ret == cudaSuccess) << "cudaSetDevice failed " << old_gpu_idx << " " << gpu_idx;

    #if defined(USE_SRD) && defined(SRD_RDMA_WRITE)
    // No need to allocate memory for packet data as use SRD_RDMA_WRITE
    #else
    // Allocate memory for packet data.
    void *pkt_data_buf_ = nullptr;
    auto cuda_ret =
        cudaMalloc(&pkt_data_buf_, PktDataBuffPool::kNumPktData *
                                       PktDataBuffPool::kPktDataSize);
    DCHECK(cuda_ret == cudaSuccess) << "cudaMalloc failed";
    LOG(INFO) << "[EFA] gpu_idx " << gpu_idx << " pkt_data_buf_ " << std::hex
              << pkt_data_buf_ << " to "
              << pkt_data_buf_ + PktDataBuffPool::kNumPktData *
                                     PktDataBuffPool::kPktDataSize;
    auto *pkt_data_mr_ =
        ibv_reg_mr(pd_, pkt_data_buf_,
                   PktDataBuffPool::kNumPktData * PktDataBuffPool::kPktDataSize,
                   IBV_ACCESS_LOCAL_WRITE);
    DCHECK(pkt_data_mr_ != nullptr) << "ibv_reg_mr failed";
    pkt_data_pool_ = new PktDataBuffPool(pkt_data_mr_);
    #endif

    // Allocate memory for frame desc.
    frame_desc_pool_ = new FrameDescBuffPool();

    // Create completion queue.
    #if defined(USE_SRD) && defined(SRD_RDMA_WRITE)
    struct ibv_cq_init_attr_ex init_attr_ex = {
        .cqe = kMaxCqeTotal,
        .cq_context = nullptr,
        .channel = nullptr,
        .comp_vector = 0,
        /* EFA requires these values for wc_flags and comp_mask.
         * See `efa_create_cq_ex` in rdma-core.
         */
        .wc_flags = IBV_WC_STANDARD_FLAGS,
        .comp_mask = 0,
    };
    send_cq_ = ibv_create_cq_ex(context_, &init_attr_ex);
    recv_cq_ = ibv_create_cq_ex(context_, &init_attr_ex);
    #else
    send_cq_ = ibv_create_cq(context_, kMaxCqeTotal, nullptr, nullptr, 0);
    recv_cq_ = ibv_create_cq(context_, kMaxCqeTotal, nullptr, nullptr, 0);
    #endif
    DCHECK(send_cq_ && recv_cq_) << "Failed to allocate send/recv_cq_";

    auto create_qp_func = &EFASocket::create_qp;

#ifdef USE_SRD
    #ifdef SRD_RDMA_WRITE
    create_qp_func = &EFASocket::create_srd_qp_ex;
    #else
    create_qp_func = &EFASocket::create_srd_qp;
    #endif
#endif

    // Create send/recv QPs.
    for (int i = 0; i < kMaxSrcDstQP; i++) {
        #if defined(USE_SRD) && defined(SRD_RDMA_WRITE)
        qp_list_[i] =
            (this->*create_qp_func)(ibv_cq_ex_to_cq(send_cq_), ibv_cq_ex_to_cq(recv_cq_), kMaxSendWr, kMaxRecvWr);
        post_recv_rdma_write_wrs(i);
        qp_num_to_idx_[qp_list_[i]->qp_num] = i;
        #else
        qp_list_[i] =
            (this->*create_qp_func)(send_cq_, recv_cq_, kMaxSendWr, kMaxRecvWr);
        post_recv_wrs(kMaxRecvWr, i);
        #endif
    }

    // Create QP for ACK packets.
    ctrl_cq_ = ibv_create_cq(context_, kMaxCqeTotal, nullptr, nullptr, 0);
    DCHECK(ctrl_cq_) << "Failed to allocate ctrl CQ";

#ifdef USE_SRD_FOR_CTRL
    create_qp_func = &EFASocket::create_srd_qp;
#endif
    for (int i = 0; i < kMaxSrcDstQPCtrl; i++) {
        ctrl_qp_list_[i] = (this->*create_qp_func)(
            ctrl_cq_, ctrl_cq_, kMaxSendRecvWrForCtrl, kMaxSendRecvWrForCtrl);
        post_recv_wrs_for_ctrl(kMaxSendRecvWrForCtrl, i);
    }

    ret = cudaSetDevice(old_gpu_idx);
    CHECK(ret == cudaSuccess) << "cudaSetDevice failed ";
}

// Create and configure a UD QP
struct ibv_qp *EFASocket::create_qp(struct ibv_cq *send_cq,
                                    struct ibv_cq *recv_cq,
                                    uint32_t send_cq_size,
                                    uint32_t recv_cq_size) {
    struct ibv_qp_init_attr qp_attr = {};
    qp_attr.send_cq = send_cq;
    qp_attr.recv_cq = recv_cq;
    qp_attr.cap.max_send_wr = send_cq_size;
    qp_attr.cap.max_recv_wr = recv_cq_size;
    qp_attr.cap.max_send_sge = 2;
    qp_attr.cap.max_recv_sge = 2;

    qp_attr.qp_type = IBV_QPT_UD;
    struct ibv_qp *qp = ibv_create_qp(pd_, &qp_attr);

    if (!qp) {
        perror("Failed to create QP");
        exit(1);
    }

    struct ibv_qp_attr attr = {};
    attr.qp_state = IBV_QPS_INIT;
    attr.pkey_index = 0;
    attr.port_num = EFA_PORT_NUM;
    attr.qkey = QKEY;
    if (ibv_modify_qp(
            qp, &attr,
            IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_QKEY)) {
        perror("Failed to modify QP to INIT");
        exit(1);
    }

    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTR;
    if (ibv_modify_qp(qp, &attr, IBV_QP_STATE)) {
        perror("Failed to modify QP to RTR");
        exit(1);
    }

    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTS;
    attr.sq_psn = SQ_PSN;  // Set initial Send Queue PSN
    if (ibv_modify_qp(qp, &attr, IBV_QP_STATE | IBV_QP_SQ_PSN)) {
        perror("Failed to modify QP to RTS");
        exit(1);
    }

    return qp;
}

/// @ref efa_qp_create(), efa_base_ep_construct_ibv_qp_init_attr_ex()
struct ibv_qp *EFASocket::create_srd_qp_ex(struct ibv_cq *send_cq,
                                        struct ibv_cq *recv_cq,
                                        uint32_t send_cq_size,
                                        uint32_t recv_cq_size)
{
    struct ibv_qp_init_attr_ex qp_attr_ex = {};
    struct efadv_qp_init_attr efa_attr = {};
    
    qp_attr_ex.comp_mask = IBV_QP_INIT_ATTR_PD | IBV_QP_INIT_ATTR_SEND_OPS_FLAGS;

    qp_attr_ex.send_ops_flags = IBV_QP_EX_WITH_RDMA_WRITE_WITH_IMM;

    qp_attr_ex.cap.max_send_wr = send_cq_size;
    qp_attr_ex.cap.max_recv_wr = recv_cq_size; 
    qp_attr_ex.cap.max_send_sge = 1;
    qp_attr_ex.cap.max_recv_sge = 1;
	qp_attr_ex.cap.max_inline_data = 0;

	qp_attr_ex.pd = pd_;
	qp_attr_ex.qp_context = context_;
	qp_attr_ex.sq_sig_all = 1;

	qp_attr_ex.send_cq = send_cq;
	qp_attr_ex.recv_cq = recv_cq;

    qp_attr_ex.qp_type = IBV_QPT_DRIVER;

    efa_attr.driver_qp_type = EFADV_QP_DRIVER_TYPE_SRD;
    #define EFA_QP_LOW_LATENCY_SERVICE_LEVEL 8
    efa_attr.sl = EFA_QP_LOW_LATENCY_SERVICE_LEVEL;
    efa_attr.flags = 0;
    // If set, Receive WRs will not be consumed for RDMA write with imm.
    // Zhongjie: if not set, sender cannot poll the completion of RDMA write with imm.
    efa_attr.flags |= EFADV_QP_FLAGS_UNSOLICITED_WRITE_RECV;

    struct ibv_qp *qp = efadv_create_qp_ex(
			context_, &qp_attr_ex, &efa_attr,
			sizeof(struct efadv_qp_init_attr));

    if (!qp) {
        perror("Failed to create QP");
        exit(1);
    }

    struct ibv_qp_attr attr = {};
    attr.qp_state = IBV_QPS_INIT;
    attr.pkey_index = 0;
    attr.port_num = EFA_PORT_NUM;
    attr.qkey = QKEY;
    if (ibv_modify_qp(
            qp, &attr,
            IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_QKEY)) {
        perror("Failed to modify QP to INIT");
        exit(1);
    }

    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTR;
    if (ibv_modify_qp(qp, &attr, IBV_QP_STATE)) {
        perror("Failed to modify QP to RTR");
        exit(1);
    }

    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTS;
    attr.sq_psn = SQ_PSN;  // Set initial Send Queue PSN
    if (ibv_modify_qp(qp, &attr, IBV_QP_STATE | IBV_QP_SQ_PSN)) {
        perror("Failed to modify QP to RTS");
        exit(1);
    }

    return qp;
}

struct ibv_qp *EFASocket::create_srd_qp(struct ibv_cq *send_cq,
                                        struct ibv_cq *recv_cq,
                                        uint32_t send_cq_size,
                                        uint32_t recv_cq_size) {
    struct ibv_qp_init_attr qp_attr = {};
    qp_attr.send_cq = send_cq;
    qp_attr.recv_cq = recv_cq;
    qp_attr.cap.max_send_wr = send_cq_size;
    qp_attr.cap.max_recv_wr = recv_cq_size;
    qp_attr.cap.max_send_sge = 2;
    qp_attr.cap.max_recv_sge = 2;

    qp_attr.qp_type = IBV_QPT_DRIVER;
    struct ibv_qp *qp = qp =
        efadv_create_driver_qp(pd_, &qp_attr, EFADV_QP_DRIVER_TYPE_SRD);

    if (!qp) {
        perror("Failed to create QP");
        exit(1);
    }

    struct ibv_qp_attr attr = {};
    attr.qp_state = IBV_QPS_INIT;
    attr.pkey_index = 0;
    attr.port_num = EFA_PORT_NUM;
    attr.qkey = QKEY;
    if (ibv_modify_qp(
            qp, &attr,
            IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_QKEY)) {
        perror("Failed to modify QP to INIT");
        exit(1);
    }

    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTR;
    if (ibv_modify_qp(qp, &attr, IBV_QP_STATE)) {
        perror("Failed to modify QP to RTR");
        exit(1);
    }

    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTS;
    attr.sq_psn = SQ_PSN;  // Set initial Send Queue PSN
    if (ibv_modify_qp(qp, &attr, IBV_QP_STATE | IBV_QP_SQ_PSN)) {
        perror("Failed to modify QP to RTS");
        exit(1);
    }

    return qp;
}

uint32_t EFASocket::post_send_wr(FrameDesc *frame, uint16_t src_qp_idx) {
    struct ibv_sge sge[2] = {{0}, {0}};
    struct ibv_send_wr send_wr = {0}, *bad_send_wr;
    struct ibv_qp *src_qp;

    auto dest_ah = frame->get_dest_ah();
    auto dest_qpn = frame->get_dest_qpn();

    if (frame->get_pkt_data_len() == 0) {
        // This is a ack packet.
        sge[0] = {frame->get_pkt_hdr_addr(), frame->get_pkt_hdr_len(),
                  get_pkt_hdr_lkey()};
        send_wr.num_sge = 1;

        DCHECK(frame->get_src_qp_idx() == UINT16_MAX);
        frame->set_src_qp_idx(src_qp_idx);
        src_qp = ctrl_qp_list_[src_qp_idx];

        ctrl_send_queue_wrs_++;
    } else {
        // This is a data packet.
        sge[0] = {frame->get_pkt_hdr_addr(), frame->get_pkt_hdr_len(),
                  get_pkt_hdr_lkey()};
        sge[1] = {frame->get_pkt_data_addr(), frame->get_pkt_data_len(),
                  frame->get_pkt_data_lkey_tx()};
        send_wr.num_sge = 2;

        DCHECK(frame->get_src_qp_idx() == UINT16_MAX);
        frame->set_src_qp_idx(src_qp_idx);
        src_qp = qp_list_[src_qp_idx];

        send_queue_wrs_++;
        send_queue_wrs_per_qp_[src_qp_idx]++;
    }

    send_wr.wr_id = (uint64_t)frame;
    send_wr.opcode = IBV_WR_SEND;
    send_wr.sg_list = sge;
    send_wr.wr.ud.ah = dest_ah;
    send_wr.wr.ud.remote_qpn = dest_qpn;
    send_wr.wr.ud.remote_qkey = QKEY;
    send_wr.send_flags = IBV_SEND_SIGNALED;

    if (ibv_post_send(src_qp, &send_wr, &bad_send_wr)) {
        perror("Server: Failed to post send");
        exit(1);
    }

    out_packets_++;
    out_bytes_ += frame->get_pkt_hdr_len() + frame->get_pkt_data_len();

    return 1;
}

uint32_t EFASocket::post_rdma_write_wrs(std::vector<FrameDesc *> &frames,
                                        uint16_t src_qp_idx) {
    
    int ret = 0;
    auto *qp = qp_list_[src_qp_idx];
    auto *qpx = ibv_qp_to_qp_ex(qp);

    int i = 0;
    ibv_wr_start(qpx);

    for (auto *frame : frames) {
        
        auto dest_ah = frame->get_dest_ah();
        auto dest_qpn = frame->get_dest_qpn();

        DCHECK(frame->get_src_qp_idx() == UINT16_MAX);
        frame->set_src_qp_idx(src_qp_idx);

        qpx->wr_id = (uint64_t)frame;
        qpx->comp_mask = 0;
        qpx->wr_flags = IBV_SEND_SIGNALED;

        ibv_wr_rdma_write_imm(qpx, frame->get_remote_rkey(), frame->get_remote_addr(), htobe32(frame->get_imm_data()));
        ibv_wr_set_sge(qpx, frame->get_pkt_data_lkey_tx(), frame->get_pkt_data_addr(), frame->get_pkt_data_len());
        
        // printf("ibv_wr_rdma_write_imm, rkey: %d, addr: %lu, imm: %d\n", frame->get_remote_rkey(), frame->get_remote_addr(), frame->get_imm_data());
        // printf("ibv_wr_set_sge, lkey: %d, addr: %lu, len: %d\n", frame->get_pkt_data_lkey_tx(), frame->get_pkt_data_addr(), frame->get_pkt_data_len());

        ibv_wr_set_ud_addr(qpx, frame->get_dest_ah(), frame->get_dest_qpn(), QKEY);

        if (i++ == kMaxChainedWr - 1) {
            ret = ibv_wr_complete(qpx);
            DCHECK(ret == 0) << "ibv_wr_complete failed" << ret;
            i = 0;
            ibv_wr_start(qpx);
        }

        send_queue_wrs_++;
        send_queue_wrs_per_qp_[src_qp_idx]++;
        
        out_packets_++;
        out_bytes_ += frame->get_pkt_data_len();
    }

    ret = ibv_wr_complete(qpx);
    DCHECK(ret == 0) << "ibv_wr_complete failed" << ret;

    return frames.size();
}

uint32_t EFASocket::post_send_wrs(std::vector<FrameDesc *> &frames,
                                  uint16_t src_qp_idx) {
    int i = 0;
    auto *wr_head = &send_wr_vec_[0];

    for (auto *frame : frames) {
        auto dest_ah = frame->get_dest_ah();
        auto dest_qpn = frame->get_dest_qpn();
        DCHECK(frame->get_src_qp_idx() == UINT16_MAX);
        frame->set_src_qp_idx(src_qp_idx);
        VLOG(3) << "post_send_wrs i " << i << " src_qp_idx " << src_qp_idx;

        auto *sge = send_sge_vec_[i % kMaxChainedWr];
        auto *wr = &send_wr_vec_[i % kMaxChainedWr];

        // This is a data packet.
        sge[0] = {frame->get_pkt_hdr_addr(), frame->get_pkt_hdr_len(),
                  get_pkt_hdr_lkey()};
        sge[1] = {frame->get_pkt_data_addr(), frame->get_pkt_data_len(),
                  frame->get_pkt_data_lkey_tx()};

        wr->num_sge = 2;
        wr->wr_id = (uint64_t)frame;
        wr->opcode = IBV_WR_SEND;
        wr->sg_list = sge;
        wr->wr.ud.ah = dest_ah;
        wr->wr.ud.remote_qpn = dest_qpn;
        wr->wr.ud.remote_qkey = QKEY;
        wr->send_flags = IBV_SEND_SIGNALED;

        bool is_last = (i + 1) % kMaxChainedWr == 0 || (i + 1) == frames.size();
        wr->next = is_last ? nullptr : &send_wr_vec_[(i + 1) % kMaxChainedWr];

        if (is_last) {
            struct ibv_send_wr *bad_send_wr;
            if (ibv_post_send(qp_list_[src_qp_idx], wr_head, &bad_send_wr)) {
                DCHECK(false)
                    << "[util_efa]: Failed to post send wrs send_queue_wrs_ "
                    << send_queue_wrs_ << " send_queue_wrs_per_qp_[src_qp_idx] "
                    << send_queue_wrs_per_qp_[src_qp_idx] << " frames.size() "
                    << frames.size();
            }
            if (i + 1 != frames.size()) {
                wr_head = &send_wr_vec_[(i + 1) % kMaxChainedWr];
            }
        }

        send_queue_wrs_++;
        send_queue_wrs_per_qp_[src_qp_idx]++;
        i++;

        out_packets_++;
        out_bytes_ += frame->get_pkt_hdr_len() + frame->get_pkt_data_len();
    }

    return frames.size();
}

uint32_t EFASocket::post_send_wrs_for_ctrl(std::vector<FrameDesc *> &frames,
                                           uint16_t src_qp_idx) {
    int i = 0;
    auto *wr_head = &send_wr_vec_[0];

    for (auto *frame : frames) {
        auto dest_ah = frame->get_dest_ah();
        auto dest_qpn = frame->get_dest_qpn();
        DCHECK(frame->get_pkt_data_len() == 0);
        DCHECK(frame->get_src_qp_idx() == UINT16_MAX);
        frame->set_src_qp_idx(src_qp_idx);  // indicating ctrl qp

        auto *sge = send_sge_vec_[i % kMaxChainedWr];
        auto *wr = &send_wr_vec_[i % kMaxChainedWr];

        // This is a ack packet.
        sge[0] = {frame->get_pkt_hdr_addr(), frame->get_pkt_hdr_len(),
                  get_pkt_hdr_lkey()};

        wr->num_sge = 1;
        wr->wr_id = (uint64_t)frame;
        wr->opcode = IBV_WR_SEND;
        wr->sg_list = sge;
        wr->wr.ud.ah = dest_ah;
        wr->wr.ud.remote_qpn = dest_qpn;
        wr->wr.ud.remote_qkey = QKEY;
        wr->send_flags = IBV_SEND_SIGNALED;

        bool is_last = (i + 1) % kMaxChainedWr == 0 || (i + 1) == frames.size();
        wr->next = is_last ? nullptr : &send_wr_vec_[(i + 1) % kMaxChainedWr];

        if (is_last) {
            struct ibv_send_wr *bad_send_wr;
            if (ibv_post_send(ctrl_qp_list_[src_qp_idx], wr_head,
                              &bad_send_wr)) {
                DCHECK(false) << "[util_efa]: Failed to post send wrs for ctrl "
                                 "ctrl_send_queue_wrs_ "
                              << ctrl_send_queue_wrs_;
            }
            if (i + 1 != frames.size()) {
                wr_head = &send_wr_vec_[(i + 1) % kMaxChainedWr];
            }
        }

        ctrl_send_queue_wrs_++;
        i++;

        out_packets_++;
        out_bytes_ += frame->get_pkt_hdr_len() + frame->get_pkt_data_len();
    }

    return frames.size();
}

void EFASocket::post_recv_rdma_write_wrs(uint16_t qp_idx)
{
    DCHECK(qp_idx < kMaxSrcDstQP);
    auto *qp = qp_list_[qp_idx];

    auto *wr = &recv_wr_vec_[0];

    wr->num_sge = 0;
    wr->sg_list = nullptr;
    wr->next = nullptr;

    struct ibv_recv_wr *bad_wr;
    // Post receive buffer
    if (ibv_post_recv(qp, wr, &bad_wr)) {
        perror("post_recv_rdma_write_wrs: Failed to post recv");
        exit(1);
    }

    recv_queue_wrs_ += 1;
}

void EFASocket::post_recv_wrs(uint32_t budget, uint16_t qp_idx) {
    DCHECK(qp_idx < kMaxSrcDstQP);
    auto &deficit_cnt = deficit_cnt_recv_wrs_[qp_idx];
    deficit_cnt += budget;
    if (deficit_cnt < kMaxRecvWrDeficit) return;

    int ret;
    uint64_t pkt_hdr_buf, pkt_data_buf, frame_desc_buf;
    auto *qp = qp_list_[qp_idx];

    auto *wr_head = &recv_wr_vec_[0];
    for (int i = 0; i < deficit_cnt; i++) {
        ret = pkt_hdr_pool_->alloc_buff(&pkt_hdr_buf);
        ret |= pkt_data_pool_->alloc_buff(&pkt_data_buf);
        ret |= frame_desc_pool_->alloc_buff(&frame_desc_buf);
        DCHECK(ret == 0) << pkt_hdr_pool_->avail_slots() << " "
                         << pkt_data_pool_->avail_slots() << " "
                         << frame_desc_pool_->avail_slots();

        auto *frame_desc = FrameDesc::Create(
            frame_desc_buf, pkt_hdr_buf, EFA_UD_ADDITION + kUcclPktHdrLen,
            pkt_data_buf, kUcclPktDataMaxLen, get_pkt_data_lkey(), 0);
        frame_desc->set_src_qp_idx(qp_idx);

        auto *sge = recv_sge_vec_[i % kMaxChainedWr];
        auto *wr = &recv_wr_vec_[i % kMaxChainedWr];

        // recv size does not need to exactly match send size. But we need limit
        // the hdr sge to exactly match hdr size, so that we can split hdr and
        // data between GPU and CPU.
        sge[0] = {(uintptr_t)pkt_hdr_buf, EFA_UD_ADDITION + kUcclPktHdrLen,
                  get_pkt_hdr_lkey()};
        sge[1] = {(uintptr_t)pkt_data_buf, kUcclPktDataMaxLen,
                  get_pkt_data_lkey()};

        wr->wr_id = (uint64_t)frame_desc;
        wr->num_sge = 2;
        wr->sg_list = sge;

        bool is_last = (i + 1) % kMaxChainedWr == 0 || (i + 1) == deficit_cnt;
        wr->next = is_last ? nullptr : &recv_wr_vec_[(i + 1) % kMaxChainedWr];

        if (is_last) {
            struct ibv_recv_wr *bad_wr;
            // Post receive buffer
            if (ibv_post_recv(qp, wr_head, &bad_wr)) {
                perror("post_recv_wrs: Failed to post recv");
                exit(1);
            }
            if (i + 1 != deficit_cnt)
                wr_head = &recv_wr_vec_[(i + 1) % kMaxChainedWr];
        }
    }

    recv_queue_wrs_ += deficit_cnt;
    deficit_cnt = 0;
}

void EFASocket::post_recv_wrs_for_ctrl(uint32_t budget, uint16_t qp_idx) {
    auto &deficit_cnt = deficit_cnt_recv_wrs_for_ctrl_[qp_idx];
    deficit_cnt += budget;
    VLOG(3) << "deficit_cnt_recv_wrs_for_ctrl_ deficit_cnt " << deficit_cnt;
    if (deficit_cnt < kMaxRecvWrDeficit) return;

    int ret;
    uint64_t pkt_hdr_buf, frame_desc_buf;
    auto *qp = ctrl_qp_list_[qp_idx];

    auto *wr_head = &recv_wr_vec_[0];
    for (int i = 0; i < deficit_cnt; i++) {
        ret = pkt_hdr_pool_->alloc_buff(&pkt_hdr_buf);
        ret |= frame_desc_pool_->alloc_buff(&frame_desc_buf);
        DCHECK(ret == 0);

        auto *frame_desc = FrameDesc::Create(
            frame_desc_buf, pkt_hdr_buf,
            EFA_UD_ADDITION + kUcclPktHdrLen + kUcclSackHdrLen, 0, 0, 0, 0);
        frame_desc->set_src_qp_idx(qp_idx);

        auto *sge = recv_sge_vec_[i % kMaxChainedWr];
        auto *wr = &recv_wr_vec_[i % kMaxChainedWr];

        // recv size does not need to exactly match send size.
        sge[0] = {(uintptr_t)pkt_hdr_buf,
                  EFA_UD_ADDITION + kUcclPktHdrLen + kUcclSackHdrLen,
                  get_pkt_hdr_lkey()};

        wr->wr_id = (uint64_t)frame_desc;
        wr->num_sge = 1;
        wr->sg_list = sge;

        bool is_last = (i + 1) % kMaxChainedWr == 0 || (i + 1) == deficit_cnt;
        wr->next = is_last ? nullptr : &recv_wr_vec_[(i + 1) % kMaxChainedWr];

        if (is_last) {
            struct ibv_recv_wr *bad_wr;
            // Post receive buffer
            if (ibv_post_recv(qp, wr_head, &bad_wr)) {
                perror("post_recv_wrs_for_ctrl: Failed to post recv");
                exit(1);
            }
            if (i + 1 != deficit_cnt)
                wr_head = &recv_wr_vec_[(i + 1) % kMaxChainedWr];
        }
    }

    deficit_cnt = 0;
}

#if defined(USE_SRD) && defined(SRD_RDMA_WRITE)
std::vector<FrameDesc *> EFASocket::poll_send_cq() {
    std::vector<FrameDesc *> frames;

    // Don't bother polling CQ.
    if (send_queue_wrs_ == 0) return frames;

    auto budget_use = 0;

    auto *cq_ex = send_cq_;
    struct ibv_poll_cq_attr poll_cq_attr = {.comp_mask = 0};
    auto ret = ibv_start_poll(cq_ex, &poll_cq_attr);

    if (ret) return frames;

    while (1) {
        auto now = rdtsc();
        DCHECK(cq_ex->status == IBV_WC_SUCCESS) << "cq_ex->status: " << cq_ex->status;
        auto opcode = ibv_wc_read_opcode(cq_ex);
        DCHECK(opcode == IBV_WC_RDMA_WRITE) << "opcode: " << opcode;

        auto *frame = (FrameDesc *)(cq_ex->wr_id);
        DCHECK(frame != nullptr);
        auto src_qp_idx = frame->get_src_qp_idx();
        send_queue_wrs_per_qp_[src_qp_idx]--;
        send_queue_wrs_--;

        frame->set_cpe_time_tsc(now);

        // printf("poll_send_cq: frame %p\n", frame);

        #if defined(USE_SRD) && !defined(SRD_USE_ACK)
        frames.push_back(frame);
        #endif
        
        // Exit if we have enough frames.
        if (budget_use >= 1.25 * kMaxPollBatch) break;
        // Exit if we have no more cqes.
        ret = ibv_next_poll(cq_ex);
        if (ret) break;
    }

    ibv_end_poll(cq_ex);

    return frames;
}
std::vector<FrameDesc *> EFASocket::poll_send_cq(uint32_t budget) {
    std::vector<FrameDesc *> frames;

    // Don't bother polling CQ.
    if (send_queue_wrs_ == 0) return frames;

    auto budget_use = 0;

    auto *cq_ex = send_cq_;
    struct ibv_poll_cq_attr poll_cq_attr = {.comp_mask = 0};
    auto ret = ibv_start_poll(cq_ex, &poll_cq_attr);

    if (ret) return frames;

    while (1) {
        auto now = rdtsc();
        DCHECK(cq_ex->status == IBV_WC_SUCCESS);
        auto opcode = ibv_wc_read_opcode(cq_ex);
        DCHECK(opcode == IBV_WC_RDMA_WRITE);

        auto *frame = (FrameDesc *)(cq_ex->wr_id);
        DCHECK(frame != nullptr);
        auto src_qp_idx = frame->get_src_qp_idx();
        send_queue_wrs_per_qp_[src_qp_idx]--;
        send_queue_wrs_--;

        frame->set_cpe_time_tsc(now);

        #if defined(USE_SRD) && !defined(SRD_USE_ACK)
        frames.push_back(frame);
        #endif
        
        // Exit if we have enough frames.
        if (budget_use >= budget) break;
        // Exit if we have no more cqes.
        ret = ibv_next_poll(cq_ex);
        if (ret) break;
    }

    ibv_end_poll(cq_ex);

    return frames;
}

#else
std::vector<FrameDesc *> EFASocket::poll_send_cq() {
    std::vector<FrameDesc *> frames;

    // Don't bother polling CQ.
    if (send_queue_wrs_ == 0) return frames;

    // We give a higher budget to poll_send_cq because its work is very light.
    int ne = ibv_poll_cq(send_cq_, 1.25 * kMaxPollBatch, wc_);
    DCHECK(ne >= 0) << "poll_send_cq ibv_poll_cq error";
    
    auto now = rdtsc();
    for (int i = 0; i < ne; i++) {
        // Check the completion status.
        DCHECK(wc_[i].status == IBV_WC_SUCCESS)
            << "poll_send_cq: completion error: "
            << ibv_wc_status_str(wc_[i].status);

        #if defined(USE_SRD) && defined(SRD_RDMA_WRITE)
        DCHECK(wc_[i].opcode == IBV_WC_RDMA_WRITE);
        #else
        DCHECK(wc_[i].opcode == IBV_WC_SEND);
        #endif

        auto *frame = (FrameDesc *)wc_[i].wr_id;
        auto src_qp_idx = frame->get_src_qp_idx();
        send_queue_wrs_per_qp_[src_qp_idx]--;
        send_queue_wrs_--;

        frame->set_cpe_time_tsc(now);

        #if defined(USE_SRD) && !defined(SRD_USE_ACK)
        frames.push_back(frame);
        #endif
    }

    return frames;
}
std::vector<FrameDesc *> EFASocket::poll_send_cq(uint32_t budget) {
    std::vector<FrameDesc *> frames;

    // Don't bother polling CQ.
    if (send_queue_wrs_ == 0) return frames;

    while (frames.size() < budget) {
        auto now = rdtsc();
        int ne = ibv_poll_cq(send_cq_, kMaxPollBatch, wc_);
        DCHECK(ne >= 0) << "poll_send_cq ibv_poll_cq error";

        for (int i = 0; i < ne; i++) {
            // Check the completion status.
            DCHECK(wc_[i].status == IBV_WC_SUCCESS)
                << "poll_send_cq: completion error: "
                << ibv_wc_status_str(wc_[i].status);

            DCHECK(wc_[i].opcode == IBV_WC_SEND);

            auto *frame = (FrameDesc *)wc_[i].wr_id;
            auto src_qp_idx = frame->get_src_qp_idx();
            send_queue_wrs_per_qp_[src_qp_idx]--;

            frame->set_cpe_time_tsc(now);
            frames.push_back(frame);
        }

        // TODO(yang): do we need to do anything smarter?
        if (ne < kMaxPollBatch) break;
    }

    send_queue_wrs_ -= frames.size();
    return frames;
}
#endif

#if defined(USE_SRD) && defined(SRD_RDMA_WRITE)
std::vector<FrameDesc *> EFASocket::poll_recv_cq(uint32_t budget) {
    
    std::vector<FrameDesc *> frames;
    struct ibv_poll_cq_attr poll_cq_attr = {.comp_mask = 0};
    auto *cq_ex = recv_cq_;
    auto ret = ibv_start_poll(cq_ex, &poll_cq_attr);
    if (ret) return frames;
    while (1) {
        auto now = rdtsc();
        DCHECK(cq_ex->status == IBV_WC_SUCCESS) 
            << "poll_recv_cq_ex: completion error: "
            << ibv_wc_status_str(cq_ex->status);
        
        auto opcode = ibv_wc_read_opcode(cq_ex);
        DCHECK(opcode == IBV_WC_RECV_RDMA_WITH_IMM) << "opcode: " << opcode;

        auto imm_data = ibv_wc_read_imm_data(cq_ex);
        auto byte_len = ibv_wc_read_byte_len(cq_ex);
        
        uint64_t frame_buf;
        ret = frame_desc_pool_->alloc_buff(&frame_buf);
        DCHECK(ret == 0)  << frame_desc_pool_->avail_slots();
        auto *frame = FrameDesc::Create(
            frame_buf, 0, 0, 0, 0, 0, 0);
        frame->set_src_qp_idx(qp_num_to_idx_[ibv_wc_read_qp_num(cq_ex)]);

        frame->set_cpe_time_tsc(now);

        frame->set_imm_data(be32toh(imm_data));
        frame->set_pkt_data_len(byte_len);
        frame->set_msg_flags(0);
        frame->mark_write();

        frames.push_back(frame);

        in_packets_++;
        in_bytes_ += byte_len;

        // Exit if we have enough frames.
        if (frames.size() >= budget) break;
        // Exit if we have no more cqes.
        ret = ibv_next_poll(cq_ex);
        if (ret) break;
    }

    ibv_end_poll(cq_ex);

    return frames;
}
#else
std::vector<FrameDesc *> EFASocket::poll_recv_cq(uint32_t budget) {
    std::vector<FrameDesc *> frames;

    while (frames.size() < budget) {
        auto now = rdtsc();
        int ne = ibv_poll_cq(recv_cq_, kMaxPollBatch, wc_);
        DCHECK(ne >= 0) << "poll_recv_cq ibv_poll_cq error";

        for (int i = 0; i < ne; i++) {
            // Check the completion status.
            DCHECK(wc_[i].status == IBV_WC_SUCCESS)
                << "poll_recv_cq: completion error: "
                << ibv_wc_status_str(wc_[i].status);

            DCHECK(wc_[i].opcode == IBV_WC_RECV);

            auto *frame = (FrameDesc *)wc_[i].wr_id;
            frame->set_cpe_time_tsc(now);

            frames.push_back(frame);

            auto src_qp_idx = frame->get_src_qp_idx();

            post_recv_wrs(1, src_qp_idx);

            in_packets_++;
            in_bytes_ += wc_[i].byte_len;
        }

        // TODO(yang): do we need to do anything smarter?
        if (ne < kMaxPollBatch) break;
    }

    recv_queue_wrs_ -= frames.size();
    return frames;
}
#endif

std::tuple<std::vector<FrameDesc *>, uint32_t> EFASocket::poll_ctrl_cq(
    uint32_t budget) {
    std::vector<FrameDesc *> recv_frames;
    uint32_t polled_send_acks = 0;

    while (recv_frames.size() + polled_send_acks < budget) {
        auto now = rdtsc();
        int ne = ibv_poll_cq(ctrl_cq_, kMaxPollBatch, wc_);
        DCHECK(ne >= 0) << "recv_acks ibv_poll_cq error";

        for (int i = 0; i < ne; i++) {
            // Check the completion status.
            DCHECK(wc_[i].status == IBV_WC_SUCCESS)
                << "recv_acks: completion error: "
                << ibv_wc_status_str(wc_[i].status);

            FrameDesc *frame = (FrameDesc *)wc_[i].wr_id;
            frame->set_cpe_time_tsc(now);

            if (wc_[i].opcode == IBV_WC_RECV) {

                #if defined(USE_SRD) && defined(SRD_RDMA_WRITE)
                frame->set_msg_flags(0);
                frame->mark_write_ctrl();
                #endif

                recv_frames.push_back(frame);

                auto src_qp_idx = frame->get_src_qp_idx();
                post_recv_wrs_for_ctrl(1, src_qp_idx);

                in_packets_++;
                in_bytes_ += wc_[i].byte_len;

            } else if (wc_[i].opcode == IBV_WC_SEND) {
                auto pkt_hdr_addr = frame->get_pkt_hdr_addr();

                push_pkt_hdr(pkt_hdr_addr);
                push_frame_desc((uint64_t)frame);

                polled_send_acks++;
                ctrl_send_queue_wrs_--;
            } else {
                DCHECK(false) << "Wrong wc_[i].opcode: " << wc_[i].opcode;
            }
        }

        // TODO(yang): do we need to do anything smarter?
        if (ne < kMaxPollBatch) break;
    }

    return {recv_frames, polled_send_acks};
}

std::string EFASocket::to_string() {
    std::string s;
    s += Format(
        "\n\t\t[EFA] free pkt hdr: %u, free pkt data: %u, free frame desc: "
        "%u, unpolled tx pkts: %u, fill queue entries: %u",
        pkt_hdr_pool_->avail_slots(), pkt_data_pool_->avail_slots(),
        frame_desc_pool_->avail_slots(), send_queue_wrs_, recv_queue_wrs_);
    auto now = std::chrono::high_resolution_clock::now();
    auto elapsed =
        std::chrono::duration_cast<std::chrono::microseconds>(now - last_stat_)
            .count();
    last_stat_ = now;

    auto out_packets_rate = (double)out_packets_.load() / elapsed;
    auto out_bytes_rate = (double)out_bytes_.load() / elapsed / 1000 * 8;
    auto in_packets_rate = (double)in_packets_.load() / elapsed;
    auto in_bytes_rate = (double)in_bytes_.load() / elapsed / 1000 * 8;
    out_packets_ = 0;
    out_bytes_ = 0;
    in_packets_ = 0;
    in_bytes_ = 0;

    s += Format(
        "\n\t\t      total in: %lf Mpps, %lf Gbps; total out: %lf "
        "Mpps, %lf Gbps",
        in_packets_rate, in_bytes_rate, out_packets_rate, out_bytes_rate);
    return s;
}

void EFASocket::shutdown() {
    // pull_completion_queue to make sure all frames are tx successfully.
    while (send_queue_wrs_) {
        poll_send_cq(kMaxPollBatch);
    }
}

EFASocket::~EFASocket() {
    delete pkt_hdr_pool_;
    delete pkt_data_pool_;
    delete frame_desc_pool_;
}
}  // namespace uccl
