#include <arpa/inet.h>
#include <cuda_runtime.h>
#include <infiniband/efadv.h>
#include <infiniband/verbs.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#define DEVICE_NAME "rdmap16s27"  // Change to your RDMA device
#define GID_INDEX 0
#define PORT_NUM 1
#define QKEY 0x12345
#define MTU 8928
#define MSG_SIZE (MTU)
#define TCP_PORT 12345  // Port for exchanging QPNs & GIDs
#define USE_GDR 1

struct rdma_context {
    struct ibv_context *ctx;
    struct ibv_pd *pd;
    struct ibv_cq *cq;
    struct ibv_qp *qp;
    struct ibv_mr *mr;
    struct ibv_ah *ah;
    char *local_buf;
    uint32_t remote_rkey;
    uint64_t remote_addr; 
};

size_t align_size(size_t size, size_t alignment) {
    return (size + alignment - 1) & ~(alignment - 1);
}

// Retrieve GID based on gid_index
void get_gid(struct rdma_context *rdma, int gid_index, union ibv_gid *gid) {
    if (ibv_query_gid(rdma->ctx, PORT_NUM, gid_index, gid)) {
        perror("Failed to query GID");
        exit(1);
    }
    printf("GID[%d]: %s\n", gid_index,
           inet_ntoa(*(struct in_addr *)&gid->raw[8]));
}

// Create and configure a UD QP
struct ibv_qp *create_qp(struct rdma_context *rdma) {

    struct ibv_qp_init_attr_ex qp_attr_ex = {};
    struct efadv_qp_init_attr efa_attr = {};
    
    efa_attr.driver_qp_type = EFADV_QP_DRIVER_TYPE_SRD;
    
    qp_attr_ex.comp_mask = IBV_QP_INIT_ATTR_PD | IBV_QP_INIT_ATTR_SEND_OPS_FLAGS;

    qp_attr_ex.send_ops_flags = IBV_QP_EX_WITH_RDMA_WRITE_WITH_IMM;

    qp_attr_ex.cap.max_send_wr = 256;
    qp_attr_ex.cap.max_recv_wr = 256; 
    qp_attr_ex.cap.max_send_sge = 1;
    qp_attr_ex.cap.max_recv_sge = 1;

	qp_attr_ex.cap.max_inline_data = 0;
	qp_attr_ex.pd = rdma->pd;
	qp_attr_ex.qp_context = rdma->ctx;
	qp_attr_ex.sq_sig_all = 1;

	qp_attr_ex.send_cq = rdma->cq;
	qp_attr_ex.recv_cq = rdma->cq;

    qp_attr_ex.qp_type = IBV_QPT_DRIVER;

    struct ibv_qp *qp = efadv_create_qp_ex(rdma->ctx, &qp_attr_ex, &efa_attr,
                                           sizeof(struct efadv_qp_init_attr));

    if (!qp) {
        perror("Failed to create QP");
        exit(1);
    }

    struct ibv_qp_attr attr = {};
    attr.qp_state = IBV_QPS_INIT;
    attr.pkey_index = 0;
    attr.port_num = PORT_NUM;
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
    attr.sq_psn = 0x12345;  // Set initial Send Queue PSN
    if (ibv_modify_qp(qp, &attr, IBV_QP_STATE | IBV_QP_SQ_PSN)) {
        perror("Failed to modify QP to RTS");
        exit(1);
    }

    return qp;
}

// Create AH using specific GID index
struct ibv_ah *create_ah(struct rdma_context *rdma, int gid_index,
                         union ibv_gid remote_gid) {
    struct ibv_ah_attr ah_attr = {};

    ah_attr.is_global = 1;  // Enable Global Routing Header (GRH)
    ah_attr.port_num = PORT_NUM;
    ah_attr.grh.sgid_index = gid_index;  // Use selected GID index
    ah_attr.grh.dgid = remote_gid;       // Destination GID
    ah_attr.grh.flow_label = 0;
    ah_attr.grh.hop_limit = 255;
    ah_attr.grh.traffic_class = 0;

    struct ibv_ah *ah = ibv_create_ah(rdma->pd, &ah_attr);
    if (!ah) {
        perror("Failed to create AH");
        exit(1);
    }
    return ah;
}

struct metadata {
    uint32_t qpn;
    union ibv_gid gid;
    uint32_t rkey;
    uint64_t addr;
};

// Exchange QPNs and GIDs via TCP
void exchange_qpns(const char *peer_ip, metadata* local_meta, metadata* remote_meta) {
    int sock;
    struct sockaddr_in addr;
    char mode = peer_ip ? 'c' : 's';

    sock = socket(AF_INET, SOCK_STREAM, 0);
    int opt = 1;
    setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &opt,
               sizeof(opt));  // Avoid port conflicts

    addr.sin_family = AF_INET;
    addr.sin_port = htons(TCP_PORT);
    addr.sin_addr.s_addr = peer_ip ? inet_addr(peer_ip) : INADDR_ANY;

    if (mode == 's') {
        printf("Server waiting for connection...\n");
        bind(sock, (struct sockaddr *)&addr, sizeof(addr));
        listen(sock, 10);
        sock = accept(sock, NULL, NULL);  // Blocks if no client
        printf("Server accepted connection\n");
    } else {
        printf("Client attempting connection...\n");
        int attempts = 5;
        while (connect(sock, (struct sockaddr *)&addr, sizeof(addr)) < 0 &&
               attempts--) {
            perror("Connect failed, retrying...");
            sleep(1);
        }
        if (attempts == 0) {
            perror("Failed to connect after retries");
            exit(1);
        }
        printf("Client connected\n");
    }

    // Set receive timeout to avoid blocking
    struct timeval timeout = {5, 0};  // 5 seconds timeout
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));

    // Send local metadata
    if (send(sock, local_meta, sizeof(*local_meta), 0) <= 0)
        perror("send() failed");

    // Receive remote metadata
    if (recv(sock, remote_meta, sizeof(*remote_meta), 0) <= 0)
        perror("recv() timeout");

    close(sock);
    printf("QPNs and GIDs exchanged\n");
}

// Initialize RDMA resources
struct rdma_context *init_rdma() {
    struct rdma_context *rdma =
        (struct rdma_context *)calloc(1, sizeof(struct rdma_context));

    struct ibv_device **dev_list = ibv_get_device_list(NULL);
    rdma->ctx = ibv_open_device(dev_list[0]);
    ibv_free_device_list(dev_list);
    if (!rdma->ctx) {
        perror("Failed to open device");
        exit(1);
    }

    rdma->pd = ibv_alloc_pd(rdma->ctx);
    rdma->cq = ibv_create_cq(rdma->ctx, 1024, NULL, NULL, 0);
    if (!rdma->pd || !rdma->cq) {
        perror("Failed to allocate PD or CQ");
        exit(1);
    }

// Register memory regions
#if USE_GDR == 0
    rdma->buf1 = (char *)aligned_alloc(
        4096, align_size(4096, MSG_SIZE ));
    rdma->mr1 = ibv_reg_mr(rdma->pd, rdma->buf1, MSG_SIZE ,
                           IBV_ACCESS_LOCAL_WRITE);
#else
    if (cudaMalloc(&rdma->local_buf, MSG_SIZE ) != cudaSuccess) {
        perror("Failed to allocate GPU memory");
        exit(1);
    }
    rdma->mr = ibv_reg_mr(rdma->pd, rdma->local_buf, MSG_SIZE ,
                           IBV_ACCESS_LOCAL_WRITE);
#endif

    if (!rdma->mr) {
        perror("Failed to register memory regions");
        exit(1);
    }

    rdma->qp = create_qp(rdma);
    return rdma;
}

// Server: Post a receive and poll CQ
void run_server(struct rdma_context *rdma, int gid_index) {
    metadata local_meta, remote_meta;
    local_meta.qpn = rdma->qp->qp_num;
    local_meta.rkey = rdma->mr->rkey;
    local_meta.addr = (uint64_t)rdma->local_buf;
    get_gid(rdma, gid_index, &local_meta.gid);

    exchange_qpns(NULL, &local_meta, &remote_meta);

    // Post receive buffer
    struct ibv_sge sge[1] = {
        {(uintptr_t)rdma->local_buf, MSG_SIZE , rdma->mr->lkey}};

    struct ibv_recv_wr wr = {}, *bad_wr;
    wr.wr_id = 1;
    wr.num_sge = 0;
    wr.sg_list = nullptr;

    if (ibv_post_recv(rdma->qp, &wr, &bad_wr)) {
        perror("Failed to post recv");
        exit(1);
    }

    struct ibv_wc wc;
    printf("Server waiting for message...\n");
    while (ibv_poll_cq(rdma->cq, 1, &wc) < 1);

    // Only the first message is attached a hdr.
#if USE_GDR == 0
    printf("Server received: %s\n", rdma->local_buf);
#else
    char *h_data = (char *)malloc(MSG_SIZE);
    cudaMemcpy(h_data, rdma->local_buf, MSG_SIZE ,
               cudaMemcpyDeviceToHost);
    printf("Server received: %s\n", h_data);
    free(h_data);
#endif
}

// Client: Send message
void run_client(struct rdma_context *rdma, const char *server_ip,
                int gid_index) {
    metadata local_meta, remote_meta;
    local_meta.qpn = rdma->qp->qp_num;
    local_meta.rkey = rdma->mr->rkey;
    local_meta.addr = (uint64_t)rdma->local_buf;
    get_gid(rdma, gid_index, &local_meta.gid);

    exchange_qpns(server_ip, &local_meta, &remote_meta);

    sleep(1);  // Wait for server to post receive

    rdma->ah = create_ah(rdma, gid_index, remote_meta.gid);

    // prepare message
#if USE_GDR == 0
    strcpy(rdma->buf1, "Hello World!");
#else
    char *h_data = (char *)malloc(MSG_SIZE);
    strcpy(h_data, "Hello World!");
    cudaMemcpy(rdma->local_buf, h_data, MSG_SIZE, cudaMemcpyHostToDevice);
#endif

    auto* qpx = ibv_qp_to_qp_ex(rdma->qp);
    ibv_wr_start(qpx);

    qpx->wr_id = 1;
    qpx->wr_flags = IBV_SEND_SIGNALED;

    ibv_wr_rdma_write_imm(qpx, remote_meta.rkey, remote_meta.addr, 0xdeadbeef);
    ibv_wr_set_sge(qpx, rdma->mr->lkey, (uintptr_t)rdma->local_buf,
                     MSG_SIZE);
    ibv_wr_set_ud_addr(qpx, rdma->ah, remote_meta.qpn, QKEY);

    if (ibv_wr_complete(qpx)) {
        printf("ibv_wr_complete failed\n");
    }

    printf("Client: Message sent!\n");

    struct ibv_wc wc;
    printf("Client poll message completion...\n");
    while (ibv_poll_cq(rdma->cq, 1, &wc) < 1);

#if USE_GDR == 0
    printf("Client sent: %s\n", rdma->local_buf);
#else
    memset(h_data, 0, MSG_SIZE);
    cudaMemcpy(h_data, rdma->local_buf, MSG_SIZE, cudaMemcpyDeviceToHost);
    printf("Client sent: %s\n", h_data);
    free(h_data);
#endif
}

int main(int argc, char *argv[]) {
    struct rdma_context *rdma = init_rdma();

    if (argc == 2)
        run_client(rdma, argv[1], GID_INDEX);
    else
        run_server(rdma, GID_INDEX);

    return 0;
}
