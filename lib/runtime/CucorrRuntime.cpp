// cucorr library
// Copyright (c) 2023 cucorr authors
// Distributed under the BSD 3-Clause License license.
// (See accompanying file LICENSE)
// SPDX-License-Identifier: BSD-3-Clause

#include "CucorrRuntime.h"
// clang-format off
#include "RuntimeInterface.h"
#include "analysis/KernelModel.h"
#include "support/Logger.h"
#include "TSan_External.h"
// clang-format on
#include <iostream>
#include <map>

namespace cucorr::runtime {
constexpr bool DEBUG_PRINT = false;

struct Stream {
  RawStream handle;

  // blocks the default stream till its done
  bool isBlocking;

  constexpr explicit Stream(const void* h = nullptr, bool isBlocking = true) : handle(h), isBlocking(isBlocking) {
  }
  constexpr bool operator<(const Stream& rhs) const {
    return this->handle < rhs.handle;
  }
  [[nodiscard]] constexpr bool isDefaultStream() const {
    return handle == nullptr;
  }
};

struct AllocationInfo {
  size_t size;
  bool is_pinned;
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
  std::map<const void*, AllocationInfo> allocations_;
  std::map<Stream, TsanFiber> streams_;
  std::map<Event, Stream> events_;
  TsanFiber cpu_fiber_;
  TsanFiber curr_fiber_;
  bool init_ = false;

 public:
  static Runtime& get() {
    static Runtime run_t;
    if (!run_t.init_) {
      run_t.cpu_fiber_  = TsanGetCurrentFiber();
      run_t.curr_fiber_ = run_t.cpu_fiber_;

      // default '0' cuda stream
      { run_t.register_stream(Stream()); }

      run_t.init_ = true;
    }
    return run_t;
  }

  Runtime(const Runtime&) = delete;

  void operator=(const Runtime&) = delete;

  void happens_before() {
    TsanHappensBefore(curr_fiber_);
  }

  void switch_to_cpu() {
    // without synchronization
    TsanSwitchToFiber(cpu_fiber_, 1);
    curr_fiber_ = cpu_fiber_;
  }

  void register_stream(Stream stream) {
    auto search_result = streams_.find(stream);
    assert(search_result == streams_.end() && "Registered stream twice");
    TsanFiber fiber = TsanCreateFiber(0);
    TsanSetFiberName(fiber, "cuda_stream");
    streams_.insert({stream, fiber});
  }

  void switch_to_stream(Stream stream) {
    if constexpr (DEBUG_PRINT) {
      llvm::errs() << "[cucorr]    Switching to stream: " << stream.handle << "\n";
    }
    auto search_result = streams_.find(stream);
    assert(search_result != streams_.end() && "Tried using stream that wasnt created prior");
    TsanSwitchToFiber(search_result->second, 0);
    if (search_result->first.isDefaultStream()) {
      // then we are on the default stream and as such want to synchronize behind all other streams
      // unless they are nonBlocking
      for (auto& [s, sync_var] : streams_) {
        if (s.isBlocking) {
          TsanHappensAfter(sync_var);
        }
      }
    }
    curr_fiber_ = search_result->second;
  }

  void happens_after_all_streams() {
    for (auto [_, fiber] : streams_) {
      TsanHappensAfter(fiber);
    }
  }

  void happens_after_stream(Stream stream) {
    auto search_result = streams_.find(stream);
    assert(search_result != streams_.end() && "Tried using stream that wasnt created prior");
    TsanHappensAfter(search_result->second);
  }

  void record_event(Event event, Stream stream) {
    if constexpr (DEBUG_PRINT) {
      llvm::errs() << "[cucorr]    Record event: " << event << " stream:" << stream.handle << "\n";
    }
    events_[event] = stream;
  }

  // Sync the event on the current stream
  void sync_event(Event event) {
    auto search_result = events_.find(event);
    assert(search_result != events_.end() && "Tried using event that wasnt recorded to prior");
    if constexpr (DEBUG_PRINT) {
      llvm::errs() << "[cucorr]    Sync event: " << event << " recorded on stream:" << events_[event].handle << "\n";
    }
    happens_after_stream(events_[event]);
  }

  void insert_allocation(void* ptr, AllocationInfo info) {
    assert(allocations_.find(ptr) == allocations_.end() && "Registered an allocation multiple times");
    allocations_[ptr] = info;
  }

  void free_allocation(void* ptr) {
    assert(allocations_.find(ptr) != allocations_.end() && "Tried to delete a non existant allocation");
    allocations_.erase(ptr);
  }

  AllocationInfo* get_allocation_info(const void* ptr) {
    auto res = allocations_.find(ptr);
    if (res == allocations_.end()) {
      //fallback find if it lies within a region
      //for(auto [alloc_ptr, alloc_info]: allocations_){
      //  if(alloc_ptr < ptr && ((const char*)alloc_ptr) + alloc_info.size > ptr){
      //    return &allocations_[ptr];
      //  }
      //}
      return nullptr;
    }
    return &res->second;
  }

 private:
  Runtime() = default;

  ~Runtime() = default;
};

cucorr_MemcpyKind infer_memcpy_direction(const void* target, const void* from);

}  // namespace cucorr::runtime

using namespace cucorr::runtime;

void _cucorr_kernel_register(void*** kernel_args, short* modes, int n, RawStream stream) {
  if constexpr (DEBUG_PRINT) {
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
    auto* ptr         = *kernel_args[i];
    auto query_status = typeart_get_type_length(ptr, &alloc_size);
    if (query_status != TYPEART_OK) {
      LOG_ERROR("Querying allocation length failed. Code: " << int(query_status))
      continue;
    }
    if (mode.state == cucorr::AccessState::kRW || mode.state == cucorr::AccessState::kWritten) {
      if constexpr (DEBUG_PRINT) {
        llvm::errs() << "[cucorr]    Write to " << ptr << " with size " << alloc_size << "\n";
      }
      TsanMemoryWritePC(ptr, alloc_size, __builtin_return_address(0));
    } else if (mode.state == cucorr::AccessState::kRead) {
      if constexpr (DEBUG_PRINT) {
        llvm::errs() << "[cucorr]    Read from " << ptr << " with size " << alloc_size << "\n";
      }
      TsanMemoryReadPC(ptr, alloc_size, __builtin_return_address(0));
    }
  }
  runtime.happens_before();
  runtime.switch_to_cpu();
}

void _cucorr_sync_device() {
  if constexpr (DEBUG_PRINT) {
    llvm::errs() << "[cucorr]Sync Device\n";
  }
  auto& runtime = Runtime::get();
  runtime.happens_after_all_streams();
}

void _cucorr_event_record(Event event, RawStream stream) {
  if constexpr (DEBUG_PRINT) {
    llvm::errs() << "[cucorr]Event Record\n";
  }
  Runtime::get().record_event(event, Stream(stream));
}

void _cucorr_sync_stream(RawStream stream) {
  if constexpr (DEBUG_PRINT) {
    llvm::errs() << "[cucorr]Sync Stream" << stream << "\n";
  }
  auto& runtime = Runtime::get();
  runtime.happens_after_stream(Stream(stream));
}

void _cucorr_sync_event(Event event) {
  if constexpr (DEBUG_PRINT) {
    llvm::errs() << "[cucorr]Sync Event" << event << "\n";
  }
  Runtime::get().sync_event(event);
}

void _cucorr_create_event(Event*) {
}

void _cucorr_create_stream(RawStream* stream) {
  Runtime::get().register_stream(Stream(*stream));
}

void _cucorr_memcpy(void* target, const void* from, size_t count, cucorr_MemcpyKind kind) {
  // NOTE: atleast for cuda non async memcpy is beheaving like on the default stream
  // https://forums.developer.nvidia.com/t/is-cudamemcpyasync-cudastreamsynchronize-on-default-stream-equal-to-cudamemcpy-non-async/108853/5
  if constexpr (DEBUG_PRINT) {
    llvm::errs() << "[cucorr]Memcpy\n";
  }

  if (kind == cucorr_MemcpyDefault) {
    kind = infer_memcpy_direction(target, from);
  }

  auto& r = Runtime::get();
  if (kind == cucorr_MemcpyDeviceToDevice) {
    // 4. For transfers from device memory to device memory, no host-side synchronization is performed.
    r.switch_to_stream(Stream());
    TsanMemoryReadPC(from, count, __builtin_return_address(0));
    TsanMemoryWritePC(target, count, __builtin_return_address(0));
    r.happens_before();
    r.switch_to_cpu();
  } else if (kind == cucorr_MemcpyDeviceToHost) {
    // 3. For transfers from device to either pageable or pinned host memory, the function returns only once the copy
    // has completed.
    r.switch_to_stream(Stream());
    TsanMemoryReadPC(from, count, __builtin_return_address(0));
    TsanMemoryWritePC(target, count, __builtin_return_address(0));
    r.happens_before();
    r.switch_to_cpu();
    r.happens_after_stream(Stream());
  } else if (kind == cucorr_MemcpyHostToDevice) {
    // 1. For transfers from pageable host memory to device memory, a stream sync is performed before the copy is
    // initiated.
    auto* alloc_info = r.get_allocation_info(from);
    // if we couldnt find alloc info we just assume the worst and dont sync
    if (alloc_info && !alloc_info->is_pinned) {
      r.happens_after_stream(Stream());
    }
    //   The function will return once the pageable buffer has been copied to the staging memory for DMA transfer to
    //   device memory
    TsanMemoryReadPC(from, count, __builtin_return_address(0));
    r.switch_to_stream(Stream());
    TsanMemoryWritePC(target, count, __builtin_return_address(0));
    r.happens_before();
    r.switch_to_cpu();
    r.happens_after_stream(Stream());
  } else if (kind == cucorr_MemcpyHostToHost) {
    // 5. For transfers from any host memory to any host memory, the function is fully synchronous with respect to the
    // host.
    TsanMemoryReadPC(from, count, __builtin_return_address(0));
    TsanMemoryWritePC(target, count, __builtin_return_address(0));
  } else {
    assert(false && "Should be unreachable");
  }
}

void _cucorr_memset(void* target, int, size_t count) {
  // The cudaMemset functions are asynchronous with respect to the host except when the target memory is pinned host
  // memory.
  if constexpr (DEBUG_PRINT) {
    llvm::errs() << "[cucorr]Memset\n";
  }
  auto& r = Runtime::get();
  r.switch_to_stream(Stream());
  TsanMemoryWritePC(target, count, __builtin_return_address(0));
  r.happens_before();
  r.switch_to_cpu();
  // r.happens_after_stream(Stream());
}

void _cucorr_memcpy_async(void* target, const void* from, size_t count, cucorr_MemcpyKind kind, RawStream stream) {
  if constexpr (DEBUG_PRINT) {
    llvm::errs() << "[cucorr]MemcpyAsync\n";
  }
  if (kind == cucorr_MemcpyHostToHost) {
    // 2. For transfers from any host memory to any host memory, the function is fully synchronous with respect to the
    // host.
    TsanMemoryReadPC(from, count, __builtin_return_address(0));
    TsanMemoryWritePC(target, count, __builtin_return_address(0));
  } else {
    // 1. For transfers between device memory and pageable host memory, the function *might* be synchronous with respect
    // to host.
    // 2. If pageable memory must first be staged to pinned memory, the driver *may* synchronize with the stream and
    // stage the copy into pinned memory.
    // 4. For all other transfers, the function should be fully asynchronous.
    auto& r = Runtime::get();
    r.switch_to_stream(Stream(stream));
    TsanMemoryReadPC(from, count, __builtin_return_address(0));
    TsanMemoryWritePC(target, count, __builtin_return_address(0));
    r.happens_before();
    r.switch_to_cpu();
  }
}

void _cucorr_memset_async(void* target, int, size_t count, RawStream stream) {
  // The Async versions are always asynchronous with respect to the host.
  if constexpr (DEBUG_PRINT) {
    llvm::errs() << "[cucorr]MemsetAsync\n";
  }
  auto& r = Runtime::get();
  r.switch_to_stream(Stream(stream));
  TsanMemoryWritePC(target, count, __builtin_return_address(0));
  r.happens_before();
  r.switch_to_cpu();
}

void _cucorr_stream_wait_event(RawStream stream, Event event, unsigned int flags) {
  if constexpr (DEBUG_PRINT) {
    llvm::errs() << "[cucorr]StreamWaitEvent stream:" << stream << " on event:" << event << "\n";
  }
  auto& r = Runtime::get();
  r.switch_to_stream(Stream(stream));
  r.sync_event(event);
  r.happens_before();
  r.switch_to_cpu();
}

void _cucorr_host_alloc(void** ptr, size_t size, unsigned int) {
  // atleast based of this presentation and some comments in the cuda forums this syncs the whole devic
  //  https://developer.download.nvidia.com/CUDA/training/StreamsAndConcurrencyWebinar.pdf
  if constexpr (DEBUG_PRINT) {
    llvm::errs() << "[cucorr]host alloc -> implicit device Device\n";
  }
  auto& runtime = Runtime::get();
  runtime.happens_after_all_streams();

  runtime.insert_allocation(*ptr, AllocationInfo{size, true});
}

void _cucorr_host_free(void* ptr) {
  if constexpr (DEBUG_PRINT) {
    llvm::errs() << "[cucorr]host free\n";
  }
  auto& runtime = Runtime::get();
  runtime.free_allocation(ptr);
}

void _cucorr_host_register(void* ptr, size_t size, unsigned int flags) {
  if constexpr (DEBUG_PRINT) {
    llvm::errs() << "[cucorr]host register\n";
  }
  auto& runtime = Runtime::get();
  runtime.insert_allocation(ptr, AllocationInfo{size, true});
}
void _cucorr_host_unregister(void* ptr) {
  if constexpr (DEBUG_PRINT) {
    llvm::errs() << "[cucorr]host unregister\n";
  }
  auto& runtime = Runtime::get();
  runtime.free_allocation(ptr);
}