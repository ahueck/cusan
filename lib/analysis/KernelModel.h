// cusan library
// Copyright (c) 2023-2024 cusan authors
// Distributed under the BSD 3-Clause License license.
// (See accompanying file LICENSE)
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CUSAN_KERNELMODEL_H
#define CUSAN_KERNELMODEL_H

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>
#include <string_view>

namespace cusan {

enum class AccessState : short { kWritten = 1, kRead = 1 << 1, kNone = 1 << 2, kRW = kRead | kWritten };

constexpr AccessState mergeAccessState(AccessState a, AccessState b) {
  if (a == AccessState::kNone) {
    return b;
  }
  if (b == AccessState::kNone) {
    return a;
  }
  return (AccessState)((short)a | (short)b);
}

inline constexpr const char* access_state_string(AccessState state) {
  switch (state) {
    case AccessState::kWritten:
      return "Write";
    case AccessState::kRead:
      return "Read";
    case AccessState::kRW:
      return "ReadWrite";
    case AccessState::kNone:
      return "None";
    default:
      return "";
  }
}

struct FunctionSubArg {
  llvm::Optional<llvm::Value*> value{nullptr};
  llvm::SmallVector<int32_t> indices;  // gep and loads needed to get the argument from 'actual' args
  bool is_pointer{false};
  AccessState state{AccessState::kRW};
};

struct FunctionArg {
  llvm::Optional<llvm::Value*> value{nullptr};
  unsigned arg_pos{0};
  bool is_pointer{false};
  llvm::SmallVector<FunctionSubArg> subargs;
};

struct KernelModel {
  llvm::Optional<const llvm::Function*> kernel{nullptr};
  std::string kernel_name{};
  llvm::SmallVector<FunctionArg, 4> args{};
};

struct ModelHandler {
  std::vector<KernelModel> models;
  bool insert(const KernelModel&);
};

llvm::raw_ostream& operator<<(llvm::raw_ostream&, const ModelHandler&);
llvm::raw_ostream& operator<<(llvm::raw_ostream&, const KernelModel&);
llvm::raw_ostream& operator<<(llvm::raw_ostream&, const FunctionArg&);
llvm::raw_ostream& operator<<(llvm::raw_ostream&, const FunctionSubArg&);

namespace io {
[[nodiscard]] llvm::ErrorOr<bool> store(const ModelHandler& kernel_db, std::string_view file);
[[nodiscard]] llvm::ErrorOr<bool> load(ModelHandler& kernel_db, std::string_view file);
}  // namespace io

}  // namespace cusan

#endif  // CUSAN_KERNELMODEL_H
