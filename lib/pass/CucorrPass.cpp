// cucorr library
// Copyright (c) 2023 cucorr authors
// Distributed under the BSD 3-Clause License license.
// (See accompanying file LICENSE)
// SPDX-License-Identifier: BSD-3-Clause

#include "CucorrPass.h"

#include "CommandLine.h"
#include "analysis/KernelAnalysis.h"
#include "support/CudaUtil.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"

#include <llvm/IR/IRBuilder.h>

using namespace llvm;

namespace cucorr {

namespace callback {

struct FunctionDecl {
  struct CucorrFunction {
    const std::string name;
    llvm::FunctionCallee f{nullptr};
  };

  CucorrFunction cucorr_register_access{"_cucorr_register_pointer"};

  void initialize(Module& m) {
    using namespace llvm;
    auto& c = m.getContext();

    const auto add_optimizer_attributes = [&](auto& arg) {
      arg.addAttr(Attribute::NoCapture);
      arg.addAttr(Attribute::ReadOnly);
    };

    const auto make_function = [&](auto& f_struct, auto f_type) {
      auto func_callee = m.getOrInsertFunction(f_struct.name, f_type);
      f_struct.f       = func_callee;
      if (auto f = dyn_cast<Function>(f_struct.f.getCallee())) {
        f->setLinkage(GlobalValue::ExternalLinkage);
        auto& first_param = *(f->arg_begin());
        if (first_param.getType()->isPointerTy()) {
          add_optimizer_attributes(first_param);
        }
      }
    };

    Type* arg_types_cucorr_register[] = {Type::getInt8PtrTy(c), Type::getInt16Ty(c)};
    make_function(cucorr_register_access, FunctionType::get(Type::getVoidTy(c), arg_types_cucorr_register, false));
  }
};

}  // namespace callback

namespace analysis {

llvm::StringSet kCudaKernelInvokes{{"cudaLaunchKernel"}};

using KernelArgInfo = cucorr::FunctionArg;

struct KernelInvokeData {
  llvm::CallBase* call{nullptr};
  llvm::SmallVector<KernelArgInfo, 4> args{};
  llvm::Value* cu_stream{nullptr};
};

using KernelInvokeDataVec = llvm::SmallVector<KernelInvokeData, 4>;

struct CudaKernelInvokeCollector : public llvm::InstVisitor<CudaKernelInvokeCollector> {
  KernelInvokeDataVec invokes_;
  KernelModel model;

  CudaKernelInvokeCollector(const KernelModel& current_stub_model) : model(current_stub_model) {
  }

  void visitCallBase(llvm::CallBase& cb) {
    if (auto* f = cb.getCalledFunction()) {
      if (kCudaKernelInvokes.contains(f->getName())) {
        auto* cu_stream_handle = std::prev(cb.arg_end())->get();
        auto kernel_args       = extract_kernel_args_for(*cb.getFunction());
        invokes_.emplace_back(KernelInvokeData{&cb, kernel_args, cu_stream_handle});
      }
    }
  }

 private:
  llvm::SmallVector<KernelArgInfo, 4> extract_kernel_args_for(Function& stub) {
    llvm::SmallVector<KernelArgInfo, 4> arg_info;
    for (auto arg : llvm::enumerate(stub.args())) {
      const auto index = arg.index();
      assert(index < model.args.size() && "More stub args than in model data.");
      auto model_arg = model.args[index];
      model_arg.arg  = &arg.value();
      arg_info.emplace_back(model_arg);
    }
    return arg_info;
  }
};

}  // namespace analysis

namespace transform {

struct KernelInvokeTransformer {
  Function* f_;
  callback::FunctionDecl* decls_;

  KernelInvokeTransformer(Function* f, callback::FunctionDecl* decls) : f_(f), decls_(decls) {
  }

  bool handleReadWriteMapping(const analysis::KernelInvokeData& data) {
    using namespace llvm;
    const auto call_inst = data.call;
    const bool is_invoke = llvm::isa<llvm::InvokeInst>(call_inst);

    auto target_callback = decls_->cucorr_register_access;

    auto access_encoding = [&](FunctionArg::State access) -> short {
      switch (access) {
        case FunctionArg::State::kRead:
          return 1;
        case FunctionArg::State::kWritten:
          return 2;
        case FunctionArg::State::kRW:
          return 4;
        case FunctionArg::State::kNone:
          return 8;
      }
      return 16;
    };

    // normal callinst:
    IRBuilder<> irb(call_inst->getPrevNode());
    for (const auto& arg : data.args) {
      if (arg.is_pointer) {
        auto* pointer     = const_cast<Value*>(dyn_cast<Value>(arg.arg.getValue()));
        auto* void_ptr    = irb.CreateBitOrPointerCast(pointer, irb.getInt8PtrTy());
        const auto access = access_encoding(arg.state);

        if (access > 4) {
          continue;
        }
        auto const_access_val = irb.getInt16(access);
        llvm::errs() << "Insert " << *void_ptr << " access " << *const_access_val << "\n";
        Value* args_cucorr_register[] = {void_ptr, const_access_val};
        irb.CreateCall(target_callback.f, args_cucorr_register);
      }
    }
    return true;
  }
};

}  // namespace transform

class CucorrPass : public llvm::PassInfoMixin<CucorrPass> {
  cucorr::ModelHandler kernel_models;
  callback::FunctionDecl cucorr_decls;

 public:
  llvm::PreservedAnalyses run(llvm::Module&, llvm::ModuleAnalysisManager&);

  bool runOnModule(llvm::Module&);

  bool runOnFunc(llvm::Function&);

  bool runOnKernelFunc(llvm::Function&);
};

class LegacyCucorrPass : public llvm::ModulePass {
 private:
  CucorrPass pass_impl_;

 public:
  static char ID;  // NOLINT

  LegacyCucorrPass() : ModulePass(ID){};

  bool runOnModule(llvm::Module& module) override;

  ~LegacyCucorrPass() override = default;
};

bool LegacyCucorrPass::runOnModule(llvm::Module& module) {
  const auto modified = pass_impl_.runOnModule(module);
  return modified;
}

llvm::PreservedAnalyses CucorrPass::run(llvm::Module& module, llvm::ModuleAnalysisManager&) {
  const auto changed = runOnModule(module);
  return changed ? llvm::PreservedAnalyses::none() : llvm::PreservedAnalyses::all();
}

bool CucorrPass::runOnModule(llvm::Module& module) {
  cucorr_decls.initialize(module);
  const auto kernel_models_file = [&]() {
    if (cl_cucorr_kernel_file.getNumOccurrences()) {
      return cl_cucorr_kernel_file.getValue();
    }

    const auto* data_file = getenv("CUCORR_KERNEL_DATA_FILE");
    if (data_file) {
      return std::string{data_file};
    }

    for (llvm::DICompileUnit* cu : module.debug_compile_units()) {
      if (!cu->getFilename().empty()) {
        return std::string{cu->getFilename()} + "-data.yaml";
      }
    }
    return std::string{"cucorr-kernel.yaml"};
  }();

  llvm::errs() << "Using model data file " << kernel_models_file << "\n";
  const auto result       = io::load(this->kernel_models, kernel_models_file);
  const auto changed      = llvm::count_if(module.functions(), [&](auto& func) {
                         if (cuda::is_kernel(&func)) {
                           return runOnKernelFunc(func);
                         }
                         return runOnFunc(func);
                       }) > 1;
  const auto store_result = io::store(this->kernel_models, kernel_models_file);
  return changed;
}

bool CucorrPass::runOnKernelFunc(llvm::Function& function) {
  if (function.isDeclaration()) {
    return false;
  }
  auto data = device::analyze_device_kernel(&function);
  if (data) {
    if (!cl_cucorr_quiet.getValue()) {
      llvm::errs() << "[Device] " << data.value() << "\n";
    }
    this->kernel_models.insert(data.value());
  }

  return false;
}

bool CucorrPass::runOnFunc(llvm::Function& function) {
  auto data_for_host = host::kernel_model_for_stub(&function, this->kernel_models);
  if (!data_for_host) {
    return false;
  }
  llvm::errs() << "[Host] " << data_for_host.value() << "\n";

  analysis::CudaKernelInvokeCollector visitor{data_for_host.value()};
  visitor.visit(function);
  const auto kernel_invoke_data = visitor.invokes_;

  if (kernel_invoke_data.empty()) {
    return false;
  }

  transform::KernelInvokeTransformer transformer{&function, &cucorr_decls};
  for (const auto& data : kernel_invoke_data) {
    transformer.handleReadWriteMapping(data);
  }

  return true;
}

}  // namespace cucorr

#define DEBUG_TYPE "cucorr-pass"

//.....................
// New PM
//.....................
llvm::PassPluginLibraryInfo getCucorrPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "cucorr", LLVM_VERSION_STRING, [](PassBuilder& pass_builder) {
            pass_builder.registerPipelineParsingCallback(
                [](StringRef name, ModulePassManager& module_pm, ArrayRef<PassBuilder::PipelineElement>) {
                  if (name == "cucorr") {
                    module_pm.addPass(cucorr::CucorrPass());
                    return true;
                  }
                  return false;
                });
          }};
}

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo llvmGetPassPluginInfo() {
  return getCucorrPassPluginInfo();
}

//.....................
// Old PM
//.....................
char cucorr::LegacyCucorrPass::ID = 0;  // NOLINT

static RegisterPass<cucorr::LegacyCucorrPass> x("cucorr", "Cucorr Pass");  // NOLINT

ModulePass* createCucorrPass() {
  return new cucorr::LegacyCucorrPass();
}

extern "C" void AddCucorrPass(LLVMPassManagerRef pass_manager) {
  unwrap(pass_manager)->add(createCucorrPass());
}
