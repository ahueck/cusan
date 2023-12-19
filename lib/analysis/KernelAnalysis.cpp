//
// Created by ahueck on 05.07.23.
//

#include "KernelAnalysis.h"

#include "support/CudaUtil.h"
#include "support/Util.h"

#include <llvm/Transforms/IPO/Attributor.h>

namespace cucorr {

namespace device {
inline FunctionArg::State state(const llvm::AAMemoryBehavior& mem) {
  if (mem.isAssumedReadNone()) {
    return FunctionArg::State::kNone;
  }
  if (mem.isAssumedReadOnly()) {
    return FunctionArg::State::kRead;
  }
  if (mem.isAssumedWriteOnly()) {
    return FunctionArg::State::kWritten;
  }
  return FunctionArg::kRW;
}

std::optional<KernelModel> info_with_attributor(llvm::Function* kernel) {
  using namespace llvm;

  auto* module = kernel->getParent();
  AnalysisGetter ag;
  SetVector<Function*> functions;
  for (auto& module_f : module->functions()) {
    functions.insert(&module_f);
  }
  CallGraphUpdater cg_updater;
  BumpPtrAllocator allocator;
  InformationCache info_cache(*module, ag, allocator, /* CGSCC */ nullptr);

  Attributor attrib(functions, info_cache, cg_updater);

  llvm::SmallVector<FunctionArg, 4> args{};
  for (Argument const& arg : kernel->args()) {
    IRPosition const arg_pos = IRPosition::argument(arg);
    if (arg.getType()->isPointerTy()) {
      const auto& mem_behavior = attrib.getOrCreateAAFor<AAMemoryBehavior>(arg_pos);
      const FunctionArg kernel_arg{&arg, arg.getArgNo(), true, state(mem_behavior)};
      args.emplace_back(kernel_arg);
    } else {
      const FunctionArg kernel_arg{&arg, arg.getArgNo(), false, FunctionArg::kRW};
      args.emplace_back(kernel_arg);
    }
  }

  KernelModel model{kernel, std::string{kernel->getName()}, args};
  return model;
}

std::optional<KernelModel> analyze_device_kernel(llvm::Function* f) {
  if (!cuda::is_kernel(f)) {
    return {};
  }
  using namespace llvm;
  const auto kernel_model = info_with_attributor(f);
  return kernel_model;
}

}  // namespace device

namespace host {

std::optional<KernelModel> kernel_model_for_stub(llvm::Function* f, const ModelHandler& models){
  return {};
}

}

namespace io {

}

}  // namespace cucorr