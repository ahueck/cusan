//
// Created by ahueck on 08.01.23.
//

#include "KernelMemory.h"

#include "support/CudaUtil.h"
#include "support/Util.h"

namespace cucorr {

llvm::Optional<KernelModel> analyze(llvm::Function* f) {
  if (!cuda::is_kernel(f)) {
    return llvm::None;
  }
  return KernelModel{f};
}

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const KernelModel& model) {
  os << "Kernel: " << util::try_demangle(*model.kernel) << "\n";
  const auto& vec = model.args;
  if (vec.empty()) {
    os << "args = []";
    return os;
  }
  const auto* begin = std::begin(vec);
  os << "args = " << begin;
  std::for_each(std::next(begin), std::end(vec), [&](const auto& value) {
    os << ", ";
    os << value;
  });
//  os << "]";
  return os;
}

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const FunctionArg& arg) {
  os << "[" << arg.arg << ", ptr: " << static_cast<int>(arg.is_pointer) << ", rw: " << static_cast<int>(arg.is_written) << "]";
  return os;
}

}  // namespace cucorr
