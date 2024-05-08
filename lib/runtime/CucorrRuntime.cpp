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
#include <map>


namespace cucorr::runtime {
struct Stream{
  RawStream handle;
  constexpr explicit Stream(const void* h = nullptr): handle(h){}
  constexpr bool operator<(const Stream& rhs) const {
      return this->handle < rhs.handle;
  }
}; 

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
  std::map<Stream, TsanFiber> streams_;
  std::map<Event, Stream> events_;
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
        run_t.register_stream(Stream()); 
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

  void register_stream(Stream stream){
    auto search_result = streams_.find(stream);
    assert(search_result == streams_.end() && "Registered stream twice");
    TsanFiber fiber = TsanCreateFiber(0);
    TsanSetFiberName(fiber, "cuda_stream");
    streams_.insert({stream, fiber});
  }

  void switch_to_stream(Stream stream){
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

  void happens_after_stream(Stream stream){
    auto search_result = streams_.find(stream);
    assert(search_result != streams_.end() && "Tried using stream that wasnt created prior");
    TsanHappensAfter(search_result->second);
  }

  
  void record_event(Event event, Stream stream){
    events_[event] = stream;
  }

  void sync_event(Event event){
    auto search_result = events_.find(event);
    assert(search_result != events_.end() && "Tried using event that wasnt recorded to prior");
    happens_after_stream(events_[event]);
  }

 private:
  Runtime() = default;

  ~Runtime() = default;
};

}  // namespace cucorr::runtime

using namespace cucorr::runtime;

void _cucorr_kernel_register(void*** kernel_args, short* modes, int n, RawStream stream) {
  auto& runtime = Runtime::get();
  runtime.switch_to_stream(Stream(stream));
  for (int i = 0; i < n; ++i) {
    const auto mode = cucorr::runtime::access_cast_back(modes[i]);
    if (!mode.is_ptr) {
      continue;
    }
    size_t alloc_size{0};
    auto *ptr          = *kernel_args[i];
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
void _cucorr_event_record(Event event, RawStream stream){
    Runtime::get().record_event(event, Stream(stream));
    LOG_DEBUG("EventRecord");
}
void _cucorr_sync_stream(RawStream stream){
    auto& runtime = Runtime::get();
    runtime.happens_after_stream(Stream(stream));
    LOG_DEBUG("SyncStream");
}
void _cucorr_sync_event(Event event){
    Runtime::get().sync_event(event);
    LOG_DEBUG("SyncEvent");
}
void _cucorr_create_event(Event*){
    LOG_DEBUG("CreateEvent");
}
void _cucorr_create_stream(RawStream* stream){
    Runtime::get().register_stream(Stream(*stream));
    LOG_DEBUG("CreateEvent");
}
void _cucorr_memcpy ( void* target, const void* from, size_t size, cucorr_MemcpyKind){
  LOG_DEBUG("MemCpy sync");
  //auto& r = Runtime::get();
  TsanMemoryReadPC(from, size, __builtin_return_address(0));
  TsanMemoryWritePC(target, size, __builtin_return_address(0));
}
void _cucorr_memset( void* target, int, size_t size){
  LOG_DEBUG("Memset sync");
  //auto& r = Runtime::get();
  TsanMemoryWritePC(target, size, __builtin_return_address(0));
}
void _cucorr_memcpy_async ( void* target, const void* from, size_t size, cucorr_MemcpyKind, RawStream stream){
  LOG_DEBUG("MemCpy async");
  auto& r = Runtime::get();
  r.switch_to_stream(Stream(stream));
  TsanMemoryReadPC(from, size, __builtin_return_address(0));
  TsanMemoryWritePC(target, size, __builtin_return_address(0));
  r.happens_before();
  r.switch_to_cpu();
}
void _cucorr_memset_async ( void* target, int, size_t size, RawStream stream){
  LOG_DEBUG("Memset async");
  auto& r = Runtime::get();
  r.switch_to_stream(Stream(stream));
  TsanMemoryWritePC(target, size, __builtin_return_address(0));
  r.happens_before();
  r.switch_to_cpu();
}