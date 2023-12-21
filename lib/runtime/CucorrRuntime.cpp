// cucorr library
// Copyright (c) 2023 cucorr authors
// Distributed under the BSD 3-Clause License license.
// (See accompanying file LICENSE)
// SPDX-License-Identifier: BSD-3-Clause

#include "CucorrRuntime.h"

#include <iostream>
#include <map>

namespace cucorr::runtime {

enum class Access : int { Read = 1, Write = 1 << 1, RW = Read | Write };

struct PointerAccess {
  Access mode{Access::RW};
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
    const auto emplace_token = access_map.emplace(ptr, PointerAccess{Access{mode}});
    if (emplace_token.second) {
      std::cerr << emplace_token.first->first << " mode= " << static_cast<int>(emplace_token.first->second.mode)
                << "\n";
    }
  }

 private:
  Runtime() = default;

  ~Runtime() = default;
};

}  // namespace cucorr::runtime

void _cucorr_register_pointer(const void* ptr, short mode) {
  std::cerr << "Callback called \n";
  cucorr::runtime::Runtime::get().emplace_pointer_access(ptr, mode);
}