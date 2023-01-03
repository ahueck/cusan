// cucorr library
// Copyright (c) 2023 cucorr authors
// Distributed under the BSD 3-Clause License license.
// (See accompanying file LICENSE)
// SPDX-License-Identifier: BSD-3-Clause

#include "CucorrRuntime.h"

namespace cucorr::runtime {

class Runtime {
public:
  static Runtime &get() {
    static Runtime run_t;
    return run_t;
  }

  Runtime(const Runtime &) = delete;

  void operator=(const Runtime &) = delete;

private:
  Runtime() = default;

  ~Runtime() = default;
};

} // namespace cucorr::runtime
