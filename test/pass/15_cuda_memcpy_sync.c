// clang-format off
// RUN: %wrapper-cxx %tsan-compile-flags -O2 -g %s -x cuda -gencode arch=compute_70,code=sm_70 -o %cucorr_test_dir/%basename_t.exe
// RUN: %tsan-options %cucorr_test_dir/%basename_t.exe 2>&1 | %filecheck %s

// clang-format on

// CHECK-NOT: data race
// CHECK-NOT: [Error] sync

#include <cstdio>
#include <cuda_runtime.h>


int main() {
  const int size            = 256;
  int* d_data;  // Unified Memory pointer

  // Allocate Unified Memory
  cudaMallocManaged(&d_data, size * sizeof(int));
  cudaMemset(d_data, 32, size * sizeof(int));
  cudaDeviceSynchronize();

  cudaMemset(d_data, 0, size * sizeof(int));

  for (int i = 0; i < size; i++) {
    if (d_data[i] != 0) {
      printf("[Error] sync\n");
      break;
    }
  }
  cudaFree(d_data);

  return 0;
}
