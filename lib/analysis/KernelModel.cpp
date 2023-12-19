//
// Created by ahueck on 08.01.23.
//

#include "KernelModel.h"
#include "support/CudaUtil.h"
#include "support/Util.h"

namespace cucorr {

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

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const ModelHandler& arg) {
  const auto& models = arg.models;
  if(models.empty()){
    os << "<[ ]>";
    return os;
  }

  auto begin = std::begin(models);
  os << "<[" << *begin;
  std::for_each(std::next(begin), std::end(models), [&](const auto& model){
    os << ", ";
    os << model;
  });
  os << "]>";
  return os;
}

}  // namespace cucorr
