//
// Created by ahueck on 08.01.23.
//

#ifndef CUCORR_KERNELMEMORY_H
#define CUCORR_KERNELMEMORY_H

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>

namespace cucorr {

struct FunctionArg {
  llvm::Argument* arg{nullptr};
  bool is_pointer{false};
  bool is_written{false};
};

struct KernelModel {
  llvm::Function* kernel{nullptr};
  llvm::SmallVector<FunctionArg, 4> args{};
};

llvm::Optional<KernelModel> analyze(llvm::Function*);

llvm::raw_ostream& operator<<(llvm::raw_ostream&, const KernelModel&);

}  // namespace cucorr

#endif  // CUCORR_KERNELMEMORY_H
