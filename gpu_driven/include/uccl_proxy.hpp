#pragma once
#include "bench_utils.hpp"
#include "proxy.hpp"
#include "ring_buffer.cuh"
#include <atomic>
#include <memory>
#include <string>
#include <thread>

class PeerCopyManager;

struct PeerMeta {
  int rank;
  uintptr_t ptr;
  size_t nbytes;
  std::string ip;
};

class UcclProxy {
  friend class PeerCopyManager;

 public:
  UcclProxy(uintptr_t rb_addr, int block_idx, uintptr_t gpu_buffer_addr,
            size_t total_size, int rank, std::string const& peer_ip = {});
  ~UcclProxy();

  void start_sender();
  void start_remote();
  void start_local();
  void start_dual();
  void stop();
  uintptr_t rb_addr() const noexcept { return rb_; }
  int block_idx() const noexcept { return block_idx_; }
  void* gpu_buffer_addr() const noexcept { return gpu_buffer_addr_; }
  void set_peers_meta(std::vector<PeerMeta> const& peers);

 private:
  enum class Mode { None, Sender, Remote, Local, Dual };
  void start(Mode m);

  std::string peer_ip_storage_;
  std::unique_ptr<Proxy> proxy_;
  std::thread thread_;
  Mode mode_;
  std::atomic<bool> running_;
  uintptr_t rb_;
  int block_idx_;
  void* gpu_buffer_addr_;
  std::vector<PeerMeta> peers_;
  int local_rank_;
};
