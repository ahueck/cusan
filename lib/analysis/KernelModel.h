//
// Created by ahueck on 08.01.23.
//

#ifndef CUCORR_KERNELMODEL_H
#define CUCORR_KERNELMODEL_H

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>
#include <string_view>

namespace cucorr {

struct FunctionArg {
  enum State { kWritten = 1, kRead = 1 << 1, kNone = 1 << 2, kRW = kRead | kWritten };
  llvm::Optional<const llvm::Argument*> arg{nullptr};
  unsigned arg_pos{0};
  bool is_pointer{false};
  State state{kRW};
};

struct KernelModel {
  llvm::Optional<const llvm::Function*> kernel{nullptr};
  std::string kernel_name{};
  //  unsigned kernel_id{0};
  llvm::SmallVector<FunctionArg, 4> args{};
};

struct ModelHandler {
  std::vector<KernelModel> models;
  bool insert(const KernelModel&);
};

llvm::raw_ostream& operator<<(llvm::raw_ostream&, const ModelHandler&);
llvm::raw_ostream& operator<<(llvm::raw_ostream&, const KernelModel&);
llvm::raw_ostream& operator<<(llvm::raw_ostream&, const FunctionArg&);

namespace io {
[[nodiscard]] llvm::ErrorOr<bool> store(const ModelHandler& kernel_db, std::string_view file);
[[nodiscard]] llvm::ErrorOr<bool> load(ModelHandler& kernel_db, std::string_view file);
}  // namespace io

}  // namespace cucorr

#endif  // CUCORR_KERNELMODEL_H
