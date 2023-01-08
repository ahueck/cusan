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
  return os;
}

}  // namespace cucorr
