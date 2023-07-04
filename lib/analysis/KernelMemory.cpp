//
// Created by ahueck on 08.01.23.
//

#include "KernelMemory.h"

#include "support/CudaUtil.h"
#include "support/Util.h"

#include <llvm/Transforms/IPO/Attributor.h>

namespace cucorr {


inline FunctionArg::State state(const llvm::AAMemoryBehavior& mem)  {
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


llvm::Optional<KernelModel> info_with_attributor(llvm::Function* kernel){
  using namespace llvm;

  auto* module = kernel->getParent();
  AnalysisGetter ag;
  SetVector<Function*> functions;
  for(auto& module_f : module->functions()) {
    functions.insert(&module_f);
  }
  CallGraphUpdater cg_updater;
  BumpPtrAllocator allocator;
  InformationCache info_cache(*module, ag, allocator, /* CGSCC */ nullptr);

  Attributor attrib(functions, info_cache, cg_updater);

  //  llvm::Attributor attrib = std::move(make_attrib(f));

  //  attrib.identifyDefaultAbstractAttributes(*f);
  //  attrib.run();

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

  KernelModel model{kernel, std::string{kernel->getName()}, 0, args};
  return model;
}

llvm::Optional<KernelModel> analyze(llvm::Function* f) {
  if (!cuda::is_kernel(f)) {
    return llvm::None;
  }
  using namespace llvm;

//  auto* module = f->getParent();
//  AnalysisGetter ag;
//  SetVector<Function*> functions;
//  for(auto& module_f : module->functions()) {
//    functions.insert(&module_f);
//  }
//  CallGraphUpdater cg_updater;
//  BumpPtrAllocator allocator;
//  InformationCache info_cache(*module, ag, allocator, /* CGSCC */ nullptr);
//
//  Attributor attrib(functions, info_cache, cg_updater);
//
//  //  llvm::Attributor attrib = std::move(make_attrib(f));
//
////  attrib.identifyDefaultAbstractAttributes(*f);
////  attrib.run();
//
//  for (Argument const& arg : f->args()) {
//    IRPosition const arg_pos = IRPosition::argument(arg);
//    if (arg.getType()->isPointerTy()) {
//      // "readnone/readonly/writeonly/..."
//      llvm::dbgs() << arg << ":\n";
//      const auto& mem_behavior = attrib.getOrCreateAAFor<AAMemoryBehavior>(arg_pos);
////      attrib.run();
//      mem_behavior.dump();
//      mem_behavior.printWithDeps(llvm::dbgs());
//      llvm::dbgs() << mem_behavior.getAsStr() << "\n\n";
//    }
//  }
//  // AAMemoryBehavior
  const auto kernel_model = info_with_attributor(f);
//  if(kernel_model) {
//    llvm::dbgs() << kernel_model.getValue() << "\n";
//  }
  return kernel_model;
}

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const KernelModel& model) {
  os << "Kernel: " << util::try_demangle(model.kernel_name) << "\n";
  const auto& vec = model.args;
  if (vec.empty()) {
    os << "args = []";
    return os;
  }
  const auto* begin = std::begin(vec);
  os << "args = " << *begin;
  std::for_each(std::next(begin), std::end(vec), [&](const auto& value) {
    os << ", ";
    os << value;
  });
  //  os << "]";
  return os;
}

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const FunctionArg::State& arg) {
  switch(arg){
    case FunctionArg::kWritten:
      os << "Write";
      break;
    case FunctionArg::kRead:
      os << "Read";
      break;
    case FunctionArg::kNone:
      os << "None";
      break;
    case FunctionArg::kRW:
      os << "R/W";
      break;
  }
  return os;
}

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const FunctionArg& arg) {
  os << "[";
  if(arg.arg.hasValue()) {
    os << *arg.arg.getValue();
  } else {
    os << "<null>";
  }
  os << ", pos: " << arg.arg_pos;
  os << ", ptr: " << static_cast<int>(arg.is_pointer) << ", rw: " << arg.state << "]";
  return os;
}

}  // namespace cucorr
