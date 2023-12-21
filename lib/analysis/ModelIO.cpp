//
// Created by ahueck on 19.12.23.
//

#include "KernelModel.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/YAMLParser.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <system_error>
#include <vector>

namespace compat {
auto open_flag() {
#if LLVM_VERSION_MAJOR < 13
  return llvm::sys::fs::OpenFlags::F_Text;
#else
  return llvm::sys::fs::OpenFlags::OF_Text;
#endif
}
}  // namespace compat

template <>
struct llvm::yaml::ScalarTraits<cucorr::FunctionArg::State> {
  static void output(const cucorr::FunctionArg::State& value, void*, llvm::raw_ostream& out) {
    switch (value) {
      case cucorr::FunctionArg::kWritten:
        out << "Write";
        break;
      case cucorr::FunctionArg::kRead:
        out << "Read";
        break;
      case cucorr::FunctionArg::kNone:
        out << "None";
        break;
      case cucorr::FunctionArg::kRW:
        out << "ReadWrite";
        break;
    }
  }

  static llvm::StringRef input(llvm::StringRef scalar, void*, cucorr::FunctionArg::State& value) {
    value = llvm::StringSwitch<cucorr::FunctionArg::State>(scalar)
                .Case("Write", cucorr::FunctionArg::State::kWritten)
                .Case("None", cucorr::FunctionArg::State::kNone)
                .Case("Read", cucorr::FunctionArg::State::kRead)
                .Default(cucorr::FunctionArg::State::kRW);
    return StringRef();
  }

  // Determine if this scalar needs quotes.
  static QuotingType mustQuote(StringRef) {
    return QuotingType::None;
  }
};

template <>
struct llvm::yaml::MappingTraits<cucorr::FunctionArg> {
  static void mapping(IO& io, cucorr::FunctionArg& info) {
    if (!io.outputting()) {
      info.arg = llvm::None;
    }
    io.mapRequired("position", info.arg_pos);
    io.mapRequired("access", info.state);
    io.mapRequired("pointer", info.is_pointer);
  }
};

LLVM_YAML_IS_SEQUENCE_VECTOR(cucorr::FunctionArg)

template <>
struct llvm::yaml::MappingTraits<cucorr::KernelModel> {
  static void mapping(IO& io, cucorr::KernelModel& info) {
    if (!io.outputting()) {
      info.kernel = llvm::None;
    }
    io.mapRequired("name", info.kernel_name);
    io.mapRequired("args", info.args);
    //    info.kernel = {};
  }
};

LLVM_YAML_IS_SEQUENCE_VECTOR(cucorr::KernelModel)

namespace cucorr::io {
[[nodiscard]] llvm::ErrorOr<bool> store(const ModelHandler& kernel_db, std::string_view file) {
  using namespace llvm;

  std::error_code error;
  raw_fd_ostream oss(StringRef(file), error, compat::open_flag());

  if (oss.has_error()) {
    return error;
  }

  auto types = kernel_db.models;
  yaml::Output out(oss);
  if (!types.empty()) {
    out << types;
  } else {
    out.beginDocuments();
    out.endDocuments();
  }

  return true;
}

[[nodiscard]] llvm::ErrorOr<bool> load(ModelHandler& kernel_db, std::string_view file) {
  using namespace llvm;
  ErrorOr<std::unique_ptr<MemoryBuffer>> memBuffer = MemoryBuffer::getFile(file.data());

  if (std::error_code error = memBuffer.getError(); error) {
    return error;
  }

  kernel_db.models.clear();

  yaml::Input in(memBuffer.get()->getMemBufferRef());
  std::vector<KernelModel> models;
  in >> models;

  kernel_db.models = std::move(models);

  return !in.error();
}

}  // namespace cucorr::io