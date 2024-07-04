// cucorr library
// Copyright (c) 2023 cucorr authors
// Distributed under the BSD 3-Clause License license.
// (See accompanying file LICENSE)
// SPDX-License-Identifier: BSD-3-Clause

#define CUCORR_LOG_LEVEL 3
#include "CucorrRuntime.h"
// clang-format off
#include "RuntimeInterface.h"
#include "analysis/KernelModel.h"
#include "support/Logger.h"
#include "TSan_External.h"
#include "StatsCounter.h"
#include "support/Table.h"
// clang-format on
#include <cstddef>
#include <iostream>
#include <map>

namespace cucorr::runtime {

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
  bool is_pinned  = false;
  bool is_managed = false;
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
  Recorder stats_recorder;

 public:
  static Runtime& get() {
    static Runtime run_t;
    if (!run_t.init_) {
#ifdef CUCORR_FIBERPOOL
      TsanFiberPoolInit();
#endif
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
    stats_recorder.inc_TsanHappensBefore();
  }

  void switch_to_cpu() {
    // without synchronization
    TsanSwitchToFiber(cpu_fiber_, 1);
    stats_recorder.inc_TsanSwitchToFiber();
    curr_fiber_ = cpu_fiber_;
  }

  void register_stream(Stream stream) {
    auto search_result = streams_.find(stream);
    assert(search_result == streams_.end() && "Registered stream twice");
    TsanFiber fiber = TsanCreateFiber(0);
    stats_recorder.inc_TsanCreateFiber();
    TsanSetFiberName(fiber, "cuda_stream");
    streams_.insert({stream, fiber});
  }

  void switch_to_stream(Stream stream) {
    LOG_TRACE("[cucorr]    Switching to stream: " << stream.handle)
    auto search_result = streams_.find(stream);
    assert(search_result != streams_.end() && "Tried using stream that wasnt created prior");
    TsanSwitchToFiber(search_result->second, 0);
    stats_recorder.inc_TsanSwitchToFiber();
    if (search_result->first.isDefaultStream()) {
      // then we are on the default stream and as such want to synchronize behind all other streams
      // unless they are nonBlocking
      for (auto& [s, sync_var] : streams_) {
        if (s.isBlocking) {
          TsanHappensAfter(sync_var);
          stats_recorder.inc_TsanHappensAfter();
        }
      }
    }
    curr_fiber_ = search_result->second;
  }

  void happens_after_all_streams() {
    for (auto [_, fiber] : streams_) {
      TsanHappensAfter(fiber);
      stats_recorder.inc_TsanHappensAfter();
    }
  }

  void happens_after_stream(Stream stream) {
    auto search_result = streams_.find(stream);
    assert(search_result != streams_.end() && "Tried using stream that wasnt created prior");
    TsanHappensAfter(search_result->second);
    stats_recorder.inc_TsanHappensAfter();
  }

  void record_event(Event event, Stream stream) {
    LOG_TRACE("[cucorr]    Record event: " << event << " stream:" << stream.handle);
    events_[event] = stream;
  }

  // Sync the event on the current stream
  void sync_event(Event event) {
    auto search_result = events_.find(event);
    assert(search_result != events_.end() && "Tried using event that wasnt recorded to prior");
    LOG_TRACE("[cucorr]    Sync event: " << event << " recorded on stream:" << events_[event].handle)
    happens_after_stream(events_[event]);
  }

  void insert_allocation(void* ptr, AllocationInfo info) {
    assert(allocations_.find(ptr) == allocations_.end() && "Registered an allocation multiple times");
    allocations_[ptr] = info;
  }

  void free_allocation(void* ptr, bool must_exist = true) {
    bool found = allocations_.find(ptr) != allocations_.end();
    if (must_exist) {
      assert(found && "Tried to delete a non existant allocation");
    }
    if (found) {
      allocations_.erase(ptr);
    }
  }

  AllocationInfo* get_allocation_info(const void* ptr) {
    auto res = allocations_.find(ptr);
    if (res == allocations_.end()) {
      // fallback find if it lies within a region
      // for(auto [alloc_ptr, alloc_info]: allocations_){
      //   if(alloc_ptr < ptr && ((const char*)alloc_ptr) + alloc_info.size > ptr){
      //     return &allocations_[ptr];
      //   }
      // }
      return nullptr;
    }
    return &res->second;
  }

 private:
  Runtime() = default;

  ~Runtime() {
#undef cucorr_stat_handle
#define cucorr_stat_handle(name) table.put(Row::make(#name, stats_recorder.get_##name()));
#if CUCORR_SOFTCOUNTER
    Table table{"Cucorr runtime statistics"};
    CUCORR_CUDA_EVENT_LIST
#include "TsanEvents.inc"
    table.print(std::cout);
#endif
#undef cucorr_stat_handle
#undef CUCORR_CUDA_EVENT_LIST

#ifdef CUCORR_FIBERPOOL
    TsanFiberPoolFini();
#endif
  }
};

cucorr_MemcpyKind infer_memcpy_direction(const void* target, const void* from);

}  // namespace cucorr::runtime

using namespace cucorr::runtime;

void _cucorr_kernel_register(void** kernel_args, short* modes, int n, RawStream stream) {
  LOG_TRACE("[cucorr]Kernel Register with " << n << " Args and on stream:" << stream)
  auto& runtime = Runtime::get();
  runtime.stats_recorder.inc_kernel_register_calls();
  runtime.switch_to_stream(Stream(stream));
  for (int i = 0; i < n; ++i) {
    const auto mode = cucorr::runtime::access_cast_back(modes[i]);
    if (!mode.is_ptr) {
      continue;
    }
    size_t alloc_size{0};
    int alloc_id{0};
    auto* ptr         = kernel_args[i];
    auto query_status = typeart_get_type(ptr, &alloc_id, &alloc_size);
    if (query_status != TYPEART_OK) {
      LOG_TRACE("Querying allocation length failed. Code: " << int(query_status))
      continue;
    }
    const auto bytes_for_type = typeart_get_type_size(alloc_id);
    const auto total_bytes    = bytes_for_type * alloc_size;

    if (mode.state == cucorr::AccessState::kRW || mode.state == cucorr::AccessState::kWritten) {
      LOG_TRACE("[cucorr]    Write to " << ptr << " with size " << total_bytes)
      TsanMemoryWritePC(ptr, total_bytes, __builtin_return_address(0));
      runtime.stats_recorder.inc_TsanMemoryWrite();
    } else if (mode.state == cucorr::AccessState::kRead) {
      LOG_TRACE("[cucorr]    Read from " << ptr << " with size " << total_bytes)
      TsanMemoryReadPC(ptr, total_bytes, __builtin_return_address(0));
      runtime.stats_recorder.inc_TsanMemoryRead();
    }
  }

  runtime.happens_before();
  runtime.switch_to_cpu();
}

void _cucorr_sync_device() {
  LOG_TRACE("[cucorr]Sync Device\n")
  auto& runtime = Runtime::get();
  runtime.stats_recorder.inc_sync_device_calls();
  runtime.happens_after_all_streams();
}

void _cucorr_event_record(Event event, RawStream stream) {
  LOG_TRACE("[cucorr]Event Record")
  auto& runtime = Runtime::get();
  runtime.stats_recorder.inc_event_record_calls();
  runtime.record_event(event, Stream(stream));
}

void _cucorr_sync_stream(RawStream stream) {
  LOG_TRACE("[cucorr]Sync Stream" << stream)
  auto& runtime = Runtime::get();
  runtime.stats_recorder.inc_sync_stream_calls();
  runtime.happens_after_stream(Stream(stream));
}

void _cucorr_sync_event(Event event) {
  LOG_TRACE("[cucorr]Sync Event" << event)
  auto& runtime = Runtime::get();
  runtime.stats_recorder.inc_sync_event_calls();
  runtime.sync_event(event);
}

void _cucorr_create_event(Event*) {
  LOG_TRACE("[cucorr]create event")
  auto& runtime = Runtime::get();
  runtime.stats_recorder.inc_create_event_calls();
}

void _cucorr_create_stream(RawStream* stream) {
  LOG_TRACE("[cucorr]create stream")
  auto& runtime = Runtime::get();
  runtime.stats_recorder.inc_create_stream_calls();
  runtime.register_stream(Stream(*stream));
}

void _cucorr_memcpy(void* target, const void* from, size_t count, cucorr_MemcpyKind kind) {
  // NOTE: atleast for cuda non async memcpy is beheaving like on the default stream
  // https://forums.developer.nvidia.com/t/is-cudamemcpyasync-cudastreamsynchronize-on-default-stream-equal-to-cudamemcpy-non-async/108853/5
  LOG_TRACE("[cucorr]Memcpy " << count << " bytes from:" << from << " to:" << target)

  if (kind == cucorr_MemcpyDefault) {
    kind = infer_memcpy_direction(target, from);
  }

  auto& runtime = Runtime::get();
  runtime.stats_recorder.inc_memcpy_calls();
  if (kind == cucorr_MemcpyDeviceToDevice) {
    // 4. For transfers from device memory to device memory, no host-side synchronization is performed.
    runtime.switch_to_stream(Stream());
    TsanMemoryReadPC(from, count, __builtin_return_address(0));
    runtime.stats_recorder.inc_TsanMemoryRead();
    TsanMemoryWritePC(target, count, __builtin_return_address(0));
    runtime.happens_before();
    runtime.switch_to_cpu();
  } else if (kind == cucorr_MemcpyDeviceToHost) {
    // 3. For transfers from device to either pageable or pinned host memory, the function returns only once the copy
    // has completed.
    runtime.switch_to_stream(Stream());
    TsanMemoryReadPC(from, count, __builtin_return_address(0));
    runtime.stats_recorder.inc_TsanMemoryRead();
    TsanMemoryWritePC(target, count, __builtin_return_address(0));
    runtime.stats_recorder.inc_TsanMemoryWrite();
    runtime.happens_before();
    runtime.switch_to_cpu();
    runtime.happens_after_stream(Stream());
  } else if (kind == cucorr_MemcpyHostToDevice) {
    // 1. For transfers from pageable host memory to device memory, a stream sync is performed before the copy is
    // initiated.
    auto* alloc_info = runtime.get_allocation_info(from);
    // if we couldnt find alloc info we just assume the worst and dont sync
    if (alloc_info && !alloc_info->is_pinned) {
      runtime.happens_after_stream(Stream());
    }
    //   The function will return once the pageable buffer has been copied to the staging memory for DMA transfer to
    //   device memory
    TsanMemoryReadPC(from, count, __builtin_return_address(0));
    runtime.stats_recorder.inc_TsanMemoryRead();
    runtime.switch_to_stream(Stream());
    TsanMemoryWritePC(target, count, __builtin_return_address(0));
    runtime.stats_recorder.inc_TsanMemoryWrite();
    runtime.happens_before();
    runtime.switch_to_cpu();
    runtime.happens_after_stream(Stream());
  } else if (kind == cucorr_MemcpyHostToHost) {
    // 5. For transfers from any host memory to any host memory, the function is fully synchronous with respect to the
    // host.
    TsanMemoryReadPC(from, count, __builtin_return_address(0));
    runtime.stats_recorder.inc_TsanMemoryRead();
    TsanMemoryWritePC(target, count, __builtin_return_address(0));
    runtime.stats_recorder.inc_TsanMemoryWrite();
  } else {
    assert(false && "Should be unreachable");
  }
}

void _cucorr_memset(void* target, int, size_t count) {
  // The cudaMemset functions are asynchronous with respect to the host except when the target memory is pinned host
  // memory.
  LOG_TRACE("[cucorr]Memset " << count << " bytes to:" << target)
  auto& runtime = Runtime::get();
  runtime.stats_recorder.inc_memset_calls();
  runtime.switch_to_stream(Stream());
  LOG_TRACE("[cucorr]    "
            << "Write to " << target << " with size: " << count)
  TsanMemoryWritePC(target, count, __builtin_return_address(0));
  runtime.stats_recorder.inc_TsanMemoryWrite();
  runtime.happens_before();
  runtime.switch_to_cpu();

  auto* alloc_info = runtime.get_allocation_info(target);
  // if we couldnt find alloc info we just assume the worst and dont sync
  if (alloc_info && (alloc_info->is_pinned || alloc_info->is_managed)) {
    LOG_TRACE("[cucorr]    "
              << "Memset is synced")
    runtime.happens_after_stream(Stream());
  } else {
    LOG_TRACE("[cucorr]    "
              << "Memset is not synced")
    if (!alloc_info) {
      LOG_DEBUG("[cucorr]    Failed to get alloc info " << target);
    } else {
      LOG_TRACE("[cucorr]    " << alloc_info->is_pinned << " " << alloc_info->is_managed)
    }
  }

  // r.happens_after_stream(Stream());
}

void _cucorr_memcpy_async(void* target, const void* from, size_t count, cucorr_MemcpyKind kind, RawStream stream) {
  LOG_TRACE("[cucorr]MemcpyAsync" << count << " bytes to:" << target)
  auto& runtime = Runtime::get();
  runtime.stats_recorder.inc_memcpy_async_calls();
  if (kind == cucorr_MemcpyHostToHost) {
    // 2. For transfers from any host memory to any host memory, the function is fully synchronous with respect to the
    // host.
    TsanMemoryReadPC(from, count, __builtin_return_address(0));
    runtime.stats_recorder.inc_TsanMemoryRead();
    TsanMemoryWritePC(target, count, __builtin_return_address(0));
    runtime.stats_recorder.inc_TsanMemoryWrite();
  } else {
    // 1. For transfers between device memory and pageable host memory, the function *might* be synchronous with respect
    // to host.
    // 2. If pageable memory must first be staged to pinned memory, the driver *may* synchronize with the stream and
    // stage the copy into pinned memory.
    // 4. For all other transfers, the function should be fully asynchronous.

    runtime.switch_to_stream(Stream(stream));
    TsanMemoryReadPC(from, count, __builtin_return_address(0));
    runtime.stats_recorder.inc_TsanMemoryRead();
    TsanMemoryWritePC(target, count, __builtin_return_address(0));
    runtime.stats_recorder.inc_TsanMemoryWrite();
    runtime.happens_before();
    runtime.switch_to_cpu();
  }
}

void _cucorr_memset_async(void* target, int, size_t count, RawStream stream) {
  // The Async versions are always asynchronous with respect to the host.
  LOG_TRACE("[cucorr]MemsetAsync" << count << " bytes to:" << target)
  auto& runtime = Runtime::get();
  runtime.stats_recorder.inc_memset_async_calls();
  runtime.switch_to_stream(Stream(stream));
  TsanMemoryWritePC(target, count, __builtin_return_address(0));
  runtime.stats_recorder.inc_TsanMemoryWrite();
  runtime.happens_before();
  runtime.switch_to_cpu();
}

void _cucorr_stream_wait_event(RawStream stream, Event event, unsigned int flags) {
  LOG_TRACE("[cucorr]StreamWaitEvent stream:" << stream << " on event:" << event)
  auto& runtime = Runtime::get();
  runtime.stats_recorder.inc_stream_wait_event_calls();
  runtime.switch_to_stream(Stream(stream));
  runtime.sync_event(event);
  runtime.happens_before();
  runtime.switch_to_cpu();
}

void _cucorr_host_alloc(void** ptr, size_t size, unsigned int) {
  // atleast based of this presentation and some comments in the cuda forums this syncs the whole devic
  //  https://developer.download.nvidia.com/CUDA/training/StreamsAndConcurrencyWebinar.pdf
  LOG_TRACE("[cucorr]host alloc " << *ptr << " with size " << size << " -> implicit device sync")
  auto& runtime = Runtime::get();
  runtime.stats_recorder.inc_host_alloc_calls();
  runtime.happens_after_all_streams();

  runtime.insert_allocation(*ptr, AllocationInfo{size, true, false});
}

void _cucorr_host_free(void* ptr) {
  LOG_TRACE("[cucorr]host free")
  auto& runtime = Runtime::get();
  runtime.stats_recorder.inc_host_free_calls();
  runtime.free_allocation(ptr);
}

void _cucorr_host_register(void* ptr, size_t size, unsigned int) {
  LOG_TRACE("[cucorr]host register " << ptr << " with size:" << size);
  auto& runtime = Runtime::get();
  runtime.stats_recorder.inc_host_register_calls();
  runtime.insert_allocation(ptr, AllocationInfo{size, true, false});
}
void _cucorr_host_unregister(void* ptr) {
  LOG_TRACE("[cucorr]host unregister " << ptr);
  auto& runtime = Runtime::get();
  runtime.stats_recorder.inc_host_unregister_calls();
  runtime.free_allocation(ptr);
}

void _cucorr_managed_alloc(void** ptr, size_t size, unsigned int) {
  LOG_TRACE("[cucorr]Managed host alloc " << *ptr << " with size " << size << " -> implicit device sync")
  auto& runtime = Runtime::get();
  runtime.stats_recorder.inc_managed_alloc_calls();
  runtime.happens_after_all_streams();
  runtime.insert_allocation(*ptr, AllocationInfo{size, false, true});
}

void _cucorr_device_alloc(void** ptr, size_t size) {
  // implicit syncs device
  // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#stream-ordered-memory-allocator
  LOG_TRACE("[cucorr]Device alloc " << *ptr << " with size " << size << " -> implicit device sync")
  auto& runtime = Runtime::get();
  runtime.stats_recorder.inc_device_alloc_calls();
  runtime.switch_to_stream(Stream());
  runtime.switch_to_cpu();
}
void _cucorr_device_free(void* ptr) {
  // implicit syncs device
  // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#stream-ordered-memory-allocator
  LOG_TRACE("[cucorr]Device free " << ptr << " -> TODO maybe implicit device sync")
  auto& runtime = Runtime::get();
  runtime.stats_recorder.inc_device_free_calls();
  runtime.happens_after_all_streams();
}

// TODO: get rid of cudaSpecifc check for cudaSuccess 0
void _cucorr_stream_query(RawStream stream, unsigned int err) {
  LOG_TRACE("[cucorr] Stream query " << stream << " -> " << err)
  auto& runtime = Runtime::get();
  runtime.stats_recorder.inc_stream_query_calls();
  if (err == 0) {
    LOG_TRACE("[cucorr]    syncing")

    runtime.happens_after_stream(Stream{stream});
  }
}

// TODO: get rid of cudaSpecifc check for cudaSuccess 0
void _cucorr_event_query(Event event, unsigned int err) {
  LOG_TRACE("[cucorr] Event query " << event << " -> " << err)
  auto& runtime = Runtime::get();
  runtime.stats_recorder.inc_event_query_calls();
  if (err == 0) {
    LOG_TRACE("[cucorr]    syncing")
    runtime.sync_event(event);
  }
}
