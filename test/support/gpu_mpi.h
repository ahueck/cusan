#ifndef CUSAN_GPUAWAREMPI_H
#define CUSAN_GPUAWAREMPI_H

// clang-format off
#include <mpi.h>
#include <mpi-ext.h>
#include <stdbool.h>
#include <stdio.h>
// clang-format on

#ifdef __cplusplus
extern "C" {
#endif
inline void print_gpu_aware_mpi() {
  printf("Compile time check:\n");
#if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
  printf("This MPI library has CUDA-aware support.\n");
#elif defined(MPIX_CUDA_AWARE_SUPPORT) && !MPIX_CUDA_AWARE_SUPPORT
  printf("This MPI library does not have CUDA-aware support.\n");
#else
  printf("This MPI library cannot determine if there is CUDA-aware support.\n");
#endif /* MPIX_CUDA_AWARE_SUPPORT */
  printf("Run time check:\n");
#if defined(MPIX_CUDA_AWARE_SUPPORT)
  if (1 == MPIX_Query_cuda_support()) {
    printf("This MPI library has CUDA-aware support.\n");
  } else {
    printf("This MPI library does not have CUDA-aware support.\n");
  }
#else  /* !defined(MPIX_CUDA_AWARE_SUPPORT) */
  printf("This MPI library cannot determine if there is CUDA-aware support.\n");
#endif /* MPIX_CUDA_AWARE_SUPPORT */
}

inline bool has_gpu_aware_mpi() {
#if defined(MPIX_CUDA_AWARE_SUPPORT)
  return 1 == MPIX_Query_cuda_support();
#endif /* MPIX_CUDA_AWARE_SUPPORT */
  return false;
}
#ifdef __cplusplus
}
#endif

#endif  // CUSAN_GPUAWAREMPI_H
