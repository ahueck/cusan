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
struct llvm::yaml::ScalarTraits<cucorr::AccessState> {
  static void output(const cucorr::AccessState& value, void*, llvm::raw_ostream& out) {
    out << cucorr::access_state_string(value);
  }

  static llvm::StringRef input(llvm::StringRef scalar, void*, cucorr::AccessState& value) {
    // FIXME keep stringliteral and enum value in sync, see cucorr::access_state_string
    value = llvm::StringSwitch<cucorr::AccessState>(scalar)
                .Case("Write", cucorr::AccessState::kWritten)
                .Case("None", cucorr::AccessState::kNone)
                .Case("Read", cucorr::AccessState::kRead)
                .Default(cucorr::AccessState::kRW);
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
    io.mapRequired("indexes", info.indices);
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
    io.mapRequired("n_args", info.n_args);
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