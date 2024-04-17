// cucorr library
// Copyright (c) 2023 cucorr authors
// Distributed under the BSD 3-Clause License license.
// (See accompanying file LICENSE)
// SPDX-License-Identifier: BSD-3-Clause

#include "CucorrRuntime.h"

#include "RuntimeInterface.h"
#include "analysis/KernelModel.h"
#include "support/Logger.h"

#include <iostream>
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
  std::map<const void*, PointerAccess> access_map;

 public:
  static Runtime& get() {
    static Runtime run_t;
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
    const auto emplace_token = access_map.emplace(ptr, PointerAccess{alloc_size, mode.state});
    if (emplace_token.second) {
      LOG_TRACE(emplace_token.first->first << " of size=" << alloc_size
                                           << " with access=" << access_state_string(emplace_token.first->second.mode))
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
    LOG_DEBUG(ptr << " with length " << alloc_size << " and mode " << cucorr::access_state_string(mode.state))
  }
}

void _cucorr_sync_device(){
    LOG_DEBUG("SyncDevice");
}
void _cucorr_event_record(const void* event, const void* stream){
    LOG_DEBUG("EventRecord");
}
void _cucorr_sync_stream(const void* stream){
    LOG_DEBUG("SyncStream");
}
void _cucorr_sync_event(const void* event){
    LOG_DEBUG("SyncEvent");
}
