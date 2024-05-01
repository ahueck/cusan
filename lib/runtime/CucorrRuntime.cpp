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
  std::map<const void*, PointerAccess> access_map_;
  std::map<const void*, void*> streams_;
  std::map<const void*, const void*> events_;
  void* cpu_fiber_;
  void* curr_fiber_;
  bool init_ = false;


 public:
  static Runtime& get() {
    static Runtime run_t;
    if (!run_t.init_){
      run_t.cpu_fiber_ = TsanGetCurrentFiber();
      run_t.curr_fiber_ = run_t.cpu_fiber_;
      run_t.init_ = true;
    }
    return run_t;
  }

  Runtime(const Runtime&) = delete;

  void operator=(const Runtime&) = delete;

  void emplace_pointer_access(const void* ptr, short attribute) {
    size_t alloc_size{0};
    const auto mode   = access_cast_back(attribute);
    auto query_status = typeart_get_type_length(ptr, &alloc_size);
    if (query_status != TYPEART_OK) {
      LOG_ERROR("Querying allocation length failed. Code: " << int(query_status))
    }
    const auto emplace_token = access_map_.emplace(ptr, PointerAccess{alloc_size, mode.state});
    if (emplace_token.second) {
      LOG_TRACE(emplace_token.first->first << " of size=" << alloc_size
                                           << " with access=" << access_state_string(emplace_token.first->second.mode))
    }
  }

  void happens_before(){
    TsanHappensBefore(curr_fiber_);
  }

  void switch_to_cpu(){
    // without synchronization
    TsanSwitchToFiber(cpu_fiber_, 1);
    curr_fiber_ = cpu_fiber_;
  }

  void switch_to_stream(const void* stream){
    void* fiber;
    auto search_result = streams_.find(stream);
    if (search_result != streams_.end()){
      fiber = search_result->second;
    }else{
      fiber = TsanCreateFiber(0);
      TsanSetFiberName(fiber, "cuda_stream");
      streams_.insert({stream, fiber});
    }
    TsanSwitchToFiber(fiber, 0);
    curr_fiber_ = fiber;
  }

  void happens_after_all_streams(){
    for(auto [_, fiber]: streams_){
      TsanHappensAfter(fiber);
    }
  }

  void happens_after_stream(const void* stream){
    void* fiber;
    auto search_result = streams_.find(stream);
    if (search_result != streams_.end()){
      fiber = search_result->second;
    }else{
      fiber = TsanCreateFiber(0);
      TsanSetFiberName(fiber, "cuda_stream");
      streams_.insert({stream, fiber});
    }
    TsanHappensAfter(fiber);
  }

  
  void record_event(const void* event, const void* stream){
    events_[event] = stream;
  }

  void sync_event(const void* event){
    if (events_.find(event) != events_.end()){
      happens_after_stream(events_[event]);
    }
  }

 private:
  Runtime() = default;

  ~Runtime() = default;
};

}  // namespace cucorr::runtime

void _cucorr_kernel_register(const void* ptr, short mode, const void* stream) {
  cucorr::runtime::Runtime::get().emplace_pointer_access(ptr, mode);
}

void _cucorr_kernel_register_n(void*** kernel_args, short* modes, int n, const void* stream) {
  auto& runtime = cucorr::runtime::Runtime::get();
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
    LOG_DEBUG(ptr << " with length " << alloc_size << " and mode " << cucorr::access_state_string(mode.state))
  }
  runtime.happens_before();
  runtime.switch_to_cpu();
}

void _cucorr_sync_device(){
    auto& runtime = cucorr::runtime::Runtime::get();
    runtime.happens_after_all_streams();
    LOG_DEBUG("SyncDevice");
}
void _cucorr_event_record(const void* event, const void* stream){
    cucorr::runtime::Runtime::get().record_event(event, stream);
    LOG_DEBUG("EventRecord");
}
void _cucorr_sync_stream(const void* stream){
    auto& runtime = cucorr::runtime::Runtime::get();
    runtime.happens_after_stream(stream);
    LOG_DEBUG("SyncStream");
}
void _cucorr_sync_event(const void* event){
    cucorr::runtime::Runtime::get().sync_event(event);
    LOG_DEBUG("SyncEvent");
}
