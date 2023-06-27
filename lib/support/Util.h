//
// Created by ahueck on 08.01.23.
//

#ifndef CUCORR_UTIL_H
#define CUCORR_UTIL_H

#include "llvm/Demangle/Demangle.h"

#include <string>

namespace cucorr::util {

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

}  // namespace cucorr::util

#endif  // CUCORR_UTIL_H
