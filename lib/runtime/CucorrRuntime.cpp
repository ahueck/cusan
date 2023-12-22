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

  void emplace_pointer_access(const void* ptr, short mode) {
    size_t alloc_size{0};
    auto query_status = typeart_get_type_length(ptr, &alloc_size);
    if (query_status != TYPEART_OK) {
      LOG_ERROR("Querying allocation length failed. Code: " << int(query_status))
    }
    const auto emplace_token = access_map.emplace(ptr, PointerAccess{alloc_size, AccessState{mode}});
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

void _cucorr_register_pointer(const void* ptr, short mode) {
  cucorr::runtime::Runtime::get().emplace_pointer_access(ptr, mode);
}