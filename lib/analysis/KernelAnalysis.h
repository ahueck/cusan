#ifndef CUSAN_KERNELANALYSIS_H
#define CUSAN_KERNELANALYSIS_H

#include "KernelModel.h"

#include <optional>
#include <string_view>

namespace llvm {
class Function;
}

namespace cusan {

namespace device {
std::optional<KernelModel> analyze_device_kernel(llvm::Function*);
}

namespace host {
std::optional<KernelModel> kernel_model_for_stub(llvm::Function*, const ModelHandler&);
}

}  // namespace cusan

#endif  // CUSAN_KERNELANALYSIS_H
