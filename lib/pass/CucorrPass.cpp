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
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"

using namespace llvm;

namespace cucorr {

class CucorrPass : public llvm::PassInfoMixin<CucorrPass> {
  cucorr::ModelHandler kernel_models;

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
  const auto kernel_models_file = [&]() {
    if (cl_cucorr_kernel_file.getNumOccurrences()) {
      return cl_cucorr_kernel_file.getValue();
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
      //      llvm::errs() << "[Device] " << data.value() << "\n";
    }
    this->kernel_models.insert(data.value());
  }

  return false;
}

bool CucorrPass::runOnFunc(llvm::Function& function) {
  auto data_for_host = host::kernel_model_for_stub(&function, this->kernel_models);
  if (data_for_host) {
    llvm::errs() << "[Host] " << data_for_host.value() << "\n";
  }

  return false;
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
