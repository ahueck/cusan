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
constexpr bool DEBUG_PRINT = true;



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
    if constexpr (DEBUG_PRINT){
      llvm::errs() << "[cucorr]    Record event: "<< event << " stream:" << stream.handle << "\n";
    }
    events_[event] = stream;
  }

  //Sync the event on the current stream
  void sync_event(Event event){
    auto search_result = events_.find(event);
    assert(search_result != events_.end() && "Tried using event that wasnt recorded to prior");
    if constexpr (DEBUG_PRINT){
      llvm::errs() << "[cucorr]    Sync event: "<< event << " recorded on stream:" << events_[event].handle << "\n";
    }
    happens_after_stream(events_[event]);
  }

 private:
  Runtime() = default;

  ~Runtime() = default;
};

}  // namespace cucorr::runtime

using namespace cucorr::runtime;

void _cucorr_kernel_register(void*** kernel_args, short* modes, int n, RawStream stream) {
  if constexpr (DEBUG_PRINT){
      llvm::errs() << "[cucorr]Kernel Register with " << n << " Args and on stream:" << stream << "\n";
  }
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
      if constexpr (DEBUG_PRINT){
        llvm::errs() << "[cucorr]    Write to " << ptr << " with size " << alloc_size << "\n";
      }
      TsanMemoryWritePC(ptr, alloc_size, __builtin_return_address(0));
    }else if (mode.state == cucorr::AccessState::kRead){
      if constexpr (DEBUG_PRINT){
        llvm::errs() << "[cucorr]    Read from " << ptr << " with size " << alloc_size << "\n";
      }
      TsanMemoryReadPC(ptr, alloc_size, __builtin_return_address(0));
    }
  }
  runtime.happens_before();
  runtime.switch_to_cpu();
}

void _cucorr_sync_device(){
    if constexpr (DEBUG_PRINT){
        llvm::errs() << "[cucorr]Sync Device\n";
    }
    auto& runtime = Runtime::get();
    runtime.happens_after_all_streams();

}
void _cucorr_event_record(Event event, RawStream stream){
    if constexpr (DEBUG_PRINT){
        llvm::errs() << "[cucorr]Event Record\n";
    }
    Runtime::get().record_event(event, Stream(stream));
}
void _cucorr_sync_stream(RawStream stream){
    if constexpr (DEBUG_PRINT){
        llvm::errs() << "[cucorr]Sync Stream" << stream << "\n";
    }
    auto& runtime = Runtime::get();
    runtime.happens_after_stream(Stream(stream));
}
void _cucorr_sync_event(Event event){
    if constexpr (DEBUG_PRINT){
        llvm::errs() << "[cucorr]Sync Event" << event << "\n";
    }
    Runtime::get().sync_event(event);
}
void _cucorr_create_event(Event*){
}
void _cucorr_create_stream(RawStream* stream){
    Runtime::get().register_stream(Stream(*stream));
}
void _cucorr_memcpy ( void* target, const void* from, size_t size, cucorr_MemcpyKind){
  if constexpr (DEBUG_PRINT){
        llvm::errs() << "[cucorr]Memcpy\n";
  }
  TsanMemoryReadPC(from, size, __builtin_return_address(0));
  TsanMemoryWritePC(target, size, __builtin_return_address(0));
}
void _cucorr_memset( void* target, int, size_t size){
  if constexpr (DEBUG_PRINT){
        llvm::errs() << "[cucorr]Memset\n";
  }
  TsanMemoryWritePC(target, size, __builtin_return_address(0));
}
void _cucorr_memcpy_async ( void* target, const void* from, size_t size, cucorr_MemcpyKind, RawStream stream){
  if constexpr (DEBUG_PRINT){
        llvm::errs() << "[cucorr]MemcpyAsync\n";
  }
  auto& r = Runtime::get();
  r.switch_to_stream(Stream(stream));
  TsanMemoryReadPC(from, size, __builtin_return_address(0));
  TsanMemoryWritePC(target, size, __builtin_return_address(0));
  r.happens_before();
  r.switch_to_cpu();
}
void _cucorr_memset_async ( void* target, int, size_t size, RawStream stream){
  if constexpr (DEBUG_PRINT){
        llvm::errs() << "[cucorr]MemsetAsync\n";
  }
  auto& r = Runtime::get();
  r.switch_to_stream(Stream(stream));
  TsanMemoryWritePC(target, size, __builtin_return_address(0));
  r.happens_before();
  r.switch_to_cpu();
}
void _cucorr_stream_wait_event(RawStream stream, Event event, unsigned int flags){
  if constexpr (DEBUG_PRINT){
        llvm::errs() << "[cucorr]StreamWaitEvent stream:" << stream << " on event:" << event << "\n";
  }
  auto& r = Runtime::get();
  r.switch_to_stream(Stream(stream));
  r.sync_event(event);
  r.happens_before();
  r.switch_to_cpu();
}