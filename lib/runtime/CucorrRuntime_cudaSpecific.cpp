#include "CucorrRuntime.h"

#include <cassert>
#include <cuda_runtime_api.h>

namespace cucorr::runtime {
cucorr_MemcpyKind infer_memcpy_direction(const void* target, const void* from) {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  assert(prop.unifiedAddressing && "Can only use default direction for memcpy when Unified memory is supported.");

  cudaPointerAttributes target_attribs;
  cudaPointerGetAttributes(&target_attribs, target);
  cudaPointerAttributes from_attribs;
  cudaPointerGetAttributes(&from_attribs, target);
  bool targetIsHostMem = target_attribs.type == cudaMemoryType::cudaMemoryTypeUnregistered ||
                         target_attribs.type == cudaMemoryType::cudaMemoryTypeHost;
  bool fromIsHostMem = target_attribs.type == cudaMemoryType::cudaMemoryTypeUnregistered ||
                       target_attribs.type == cudaMemoryType::cudaMemoryTypeHost;

  if (!fromIsHostMem && !targetIsHostMem) {
    return cucorr_MemcpyDeviceToDevice;
  }
  if (!fromIsHostMem && targetIsHostMem) {
    return cucorr_MemcpyDeviceToHost;
  }
  if (fromIsHostMem && !targetIsHostMem) {
    return cucorr_MemcpyHostToDevice;
  }
  if (fromIsHostMem && targetIsHostMem) {
    return cucorr_MemcpyHostToHost;
  }
}
}  // namespace cucorr::runtime