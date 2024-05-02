// cucorr library
// Copyright (c) 2023 cucorr authors
// Distributed under the BSD 3-Clause License license.
// (See accompanying file LICENSE)
// SPDX-License-Identifier: BSD-3-Clause

#include "CucorrRuntime.h"

#include "RuntimeInterface.h"
#include "analysis/KernelModel.h"
#include "support/Logger.h"
#include "TSan_External.h"

#include <iostream>
// #include <llvm/Support/Compiler.h>
#include <llvm/Support/Compiler.h>
#include <map>


namespace cucorr::runtime {
  
#define CUDA_DEFAULT_STREAM (CudaStream)nullptr
using CudaStream = const void*;
using TsanFiber = void*;
using CudaEvent = const void*;

struct PtrAttribute {
  AccessState state{AccessState::kRW};
  bool is_ptr{false};
};

PtrAttribute access_cast_back(short cb_value) {
  const short access = (cb_value >> 1);
  const bool ptr     = cb_value & 1;
  return PtrAttribute{AccessState{access}, ptr};
}

struct PointerAccess {
  size_t alloc_size{0};
  AccessState mode{AccessState::kRW};
};

class Runtime {
  std::map<CudaStream, TsanFiber> streams_;
  std::map<CudaEvent, CudaStream> events_;
  TsanFiber cpu_fiber_;
  TsanFiber curr_fiber_;
  bool init_ = false;


 public:
  static Runtime& get() {
    static Runtime run_t;
    if (!run_t.init_){
      run_t.cpu_fiber_ = TsanGetCurrentFiber();
      run_t.curr_fiber_ = run_t.cpu_fiber_;

      //default '0' cuda stream
      {
        run_t.register_stream(CUDA_DEFAULT_STREAM); 
      }
      
      run_t.init_ = true;
    }
    return run_t;
  }

  Runtime(const Runtime&) = delete;

  void operator=(const Runtime&) = delete;

  void happens_before(){
    TsanHappensBefore(curr_fiber_);
  }

  void switch_to_cpu(){
    // without synchronization
    TsanSwitchToFiber(cpu_fiber_, 1);
    curr_fiber_ = cpu_fiber_;
  }

  void register_stream(CudaStream stream){
    auto search_result = streams_.find(stream);
    assert(search_result == streams_.end() && "Registered stream twice");
    TsanFiber fiber = TsanCreateFiber(0);
    TsanSetFiberName(fiber, "cuda_stream");
    streams_.insert({stream, fiber});
  }

  void switch_to_stream(CudaStream stream){
    auto search_result = streams_.find(stream);
    assert(search_result != streams_.end() && "Tried using stream that wasnt created prior");
    TsanSwitchToFiber(search_result->second, 0);
    curr_fiber_ = search_result->second;
  }

  void happens_after_all_streams(){
    for(auto [_, fiber]: streams_){
      TsanHappensAfter(fiber);
    }
  }

  void happens_after_stream(CudaStream stream){
    auto search_result = streams_.find(stream);
    assert(search_result != streams_.end() && "Tried using stream that wasnt created prior");
    TsanHappensAfter(search_result->second);
  }

  
  void record_event(CudaEvent event, CudaStream stream){
    events_[event] = stream;
  }

  void sync_event(CudaEvent event){
    auto search_result = events_.find(event);
    assert(search_result != events_.end() && "Tried using event that wasnt registered prior");
    happens_after_stream(events_[event]);
  }

 private:
  Runtime() = default;

  ~Runtime() = default;
};

}  // namespace cucorr::runtime

using namespace cucorr::runtime;

void _cucorr_kernel_register(const void* ptr, short mode, CudaStream stream) {
}

void _cucorr_kernel_register_n(void*** kernel_args, short* modes, int n, CudaStream stream) {
  auto& runtime = Runtime::get();
  runtime.switch_to_stream(stream);
  for (int i = 0; i < n; ++i) {
    const auto mode = cucorr::runtime::access_cast_back(modes[i]);
    if (!mode.is_ptr) {
      continue;
    }
    size_t alloc_size{0};
    auto ptr          = *kernel_args[i];
    auto query_status = typeart_get_type_length(ptr, &alloc_size);
    if (query_status != TYPEART_OK) {
      LOG_ERROR("Querying allocation length failed. Code: " << int(query_status))
      continue;
    }
    if (mode.state == cucorr::AccessState::kRW || mode.state == cucorr::AccessState::kWritten){
      TsanMemoryWritePC(ptr, alloc_size, __builtin_return_address(0));
      // TsanMemoryWrite(ptr, alloc_size);
    }else if (mode.state == cucorr::AccessState::kRead){
      TsanMemoryReadPC(ptr, alloc_size, __builtin_return_address(0));
      // TsanMemoryRead(ptr, alloc_size);
    }
  }
  runtime.happens_before();
  runtime.switch_to_cpu();
}

void _cucorr_sync_device(){
    auto& runtime = Runtime::get();
    runtime.happens_after_all_streams();
    LOG_DEBUG("SyncDevice");
}
void _cucorr_event_record(CudaEvent event, CudaStream stream){
    Runtime::get().record_event(event, stream);
    LOG_DEBUG("EventRecord");
}
void _cucorr_sync_stream(CudaStream stream){
    auto& runtime = Runtime::get();
    runtime.happens_after_stream(stream);
    LOG_DEBUG("SyncStream");
}
void _cucorr_sync_event(CudaEvent event){
    Runtime::get().sync_event(event);
    LOG_DEBUG("SyncEvent");
}
void _cucorr_create_event(CudaEvent*){
    LOG_DEBUG("CreateEvent");
}
void _cucorr_create_stream(CudaStream* stream){
    Runtime::get().register_stream(*stream);
    LOG_DEBUG("CreateEvent");
}
