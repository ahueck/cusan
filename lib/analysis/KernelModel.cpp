//
// Created by ahueck on 08.01.23.
//

#include "KernelModel.h"

#include "support/CudaUtil.h"
#include "support/Util.h"

namespace cusan {

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

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const AccessState& arg) {
  os << access_state_string(arg);
  return os;
}

bool ModelHandler::insert(const cusan::KernelModel& model) {
  auto result =
      llvm::find_if(models, [&model](const auto& model_) { return model.kernel_name == model_.kernel_name; });

  if (result == std::end(models)) {
    models.emplace_back(model);
    return true;
  }

  return false;
}
llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const FunctionSubArg& arg) {
  os << "[";
  if (arg.value.hasValue()) {
    os << *arg.value.getValue();
  } else {
    os << "<null>";
  }
  if (!arg.indices.empty()) {
    os << ", indicies:[";
    for(auto index: arg.indices){
      os << index << ", ";
    }
    os << "]";
  } else {
    os << ", indicies:[]";
  }
  os << ", ptr: " << static_cast<int>(arg.is_pointer) << ", rw: " << arg.state << "]";
  return os;
}
llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const FunctionArg& arg) {
  os << "[";
  if (arg.value.hasValue()) {
    os << *arg.value.getValue();
  } else {
    os << "<null>";
  }
  os << ", subArgs: [";
  for(const auto& arg: arg.subargs){
    os << arg;
  }
  os << "]";
  os << ", ptr: " << static_cast<int>(arg.is_pointer) << ", pos: " << arg.arg_pos << "]";
  return os;
}




llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const ModelHandler& arg) {
  const auto& models = arg.models;
  if (models.empty()) {
    os << "<[ ]>";
    return os;
  }

  auto begin = std::begin(models);
  os << "<[" << *begin;
  std::for_each(std::next(begin), std::end(models), [&](const auto& model) {
    os << ", ";
    os << model;
  });
  os << "]>";
  return os;
}

}  // namespace cusan
