//
// Created by ahueck on 08.01.23.
//

#ifndef CUCORR_CUDAUTIL_H
#define CUCORR_CUDAUTIL_H

#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"

namespace cucorr::cuda {

//bool AA::isGPU(const Module &M) {
//  Triple T(M.getTargetTriple());
//  return T.isAMDGPU() || T.isNVPTX();
//}

bool is_kernel(const llvm::Function* function) {
  const auto* module   = function->getParent();
  const auto* named_md = module->getNamedMetadata("nvvm.annotations");
  if (named_md == nullptr) {
    return false;
  }

  const auto any = llvm::any_of(named_md->operands(), [&](auto* operand) {
    const auto* md_value = llvm::dyn_cast_or_null<llvm::ValueAsMetadata>(operand->getOperand(0).get());
    if (md_value == nullptr) {
      return false;
    }
    return llvm::dyn_cast<llvm::Function>(md_value->getValue()) == function;
  });

  return any;
}

}  // namespace cucorr::cuda

#endif  // CUCORR_CUDAUTIL_H
