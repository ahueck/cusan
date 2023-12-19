#ifndef CUCORR_KERNELANALYSIS_H
#define CUCORR_KERNELANALYSIS_H

#include "KernelModel.h"

#include <optional>
#include <string_view>

namespace llvm {
class Function;
}

namespace cucorr {

namespace device {
std::optional<KernelModel> analyze_device_kernel(llvm::Function*);
}

namespace host {
std::optional<KernelModel> kernel_model_for_stub(llvm::Function*, const ModelHandler&);
}

}

#endif  // CUCORR_KERNELANALYSIS_H
