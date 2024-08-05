//
// Created by ahueck on 08.01.23.
//

#ifndef CUSAN_UTIL_H
#define CUSAN_UTIL_H

#include "llvm/Demangle/Demangle.h"

#include <string>

namespace cusan::util {

// template <typename... Strings>
// bool starts_with_any_of(llvm::StringRef lhs, Strings&&... rhs) {
//   return !lhs.empty() && ((lhs.startswith(std::forward<Strings>(rhs))) || ...);
// }

template <typename... Strings>
bool starts_with_any_of(const std::string& lhs, Strings&&... rhs) {
  const auto starts_with = [](const std::string& str, std::string_view prefix) { return str.rfind(prefix, 0) == 0; };
  return !lhs.empty() && ((starts_with(lhs, std::forward<Strings>(rhs))) || ...);
}

template <typename String>
inline std::string demangle(String&& s) {
  std::string name = std::string{s};
  auto demangle    = llvm::itaniumDemangle(name.data(), nullptr, nullptr, nullptr);
  if (demangle && !std::string(demangle).empty()) {
    return {demangle};
  }
  return name;
}

template <typename T>
inline std::string try_demangle(const T& site) {
  if constexpr (std::is_same_v<T, llvm::Function>) {
    return demangle(site.getName());
  } else {
    return demangle(site);
  }
}

}  // namespace cusan::util

#endif  // CUSAN_UTIL_H
