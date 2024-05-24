// clang-format off
// RUN: %wrapper-cxx %tsan-compile-flags -O2 -g %s -x cuda -gencode arch=compute_70,code=sm_70 -o %cucorr_test_dir/%basename_t.exe
// RUN: %tsan-options %cucorr_test_dir/%basename_t.exe 2>&1 | %filecheck --allow-empty %s

// clang-format on

// CHECK-NOT: data race
// CHECK-NOT: [Error] sync
#include <cuda_runtime.h>
#include <stdio.h>

int main(int argc, char* argv[]) {
  const int size            = 1<<26;//268mb
  int* h_data = (int*)malloc(size * sizeof(int));
  memset(h_data, 0, size*sizeof(int));
  //just to check default
  cudaMemcpy(h_data, h_data, size * sizeof(int), cudaMemcpyDefault);
  for (int i = 0; i < size; i++) {
    const int buf_v = h_data[i];
    if (buf_v != 0) {
      printf("[Error] sync\n");
      break;
    }
  }
  free(h_data);
  return 0;
}
