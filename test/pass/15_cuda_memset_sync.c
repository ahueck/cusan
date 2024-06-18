// clang-format off
// RUN: %wrapper-cxx %tsan-compile-flags -O2 -g %s -x cuda -gencode arch=compute_70,code=sm_70 -o %cucorr_test_dir/%basename_t.exe
// RUN: %tsan-options %cucorr_test_dir/%basename_t.exe 2>&1 | %filecheck %s

// RUN: %wrapper-cxx %tsan-compile-flags -DCUCORR_SYNC -O2 -g %s -x cuda -gencode arch=compute_70,code=sm_70 -o %cucorr_test_dir/%basename_t-sync.exe
// RUN: %tsan-options %cucorr_test_dir/%basename_t-sync.exe 2>&1 | %filecheck %s --allow-empty --check-prefix CHECK-SYNC

// clang-format on

// CHECK-DAG: data race
// CHECK-DAG: [Error] sync

// CHECK-SYNC-NOT: data race
// CHECK-SYNC-NOT: [Error] sync

#include <cstdio>
#include <cuda_runtime.h>

__global__ void write_kernel(int* arr, const int N) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
#if __CUDA_ARCH__ >= 700
  for (int i = 0; i < tid; i++) {
    __nanosleep(1000000U);
  }
#else
  printf(">>> __CUDA_ARCH__ !\n");
#endif
  if (tid < N) {
    arr[tid] = (tid + 1);
  }
}

int main() {
  const int size            = 256;
  const int threadsPerBlock = size;
  const int blocksPerGrid   = (size + threadsPerBlock - 1) / threadsPerBlock;
  int* d_data;  // Unified Memory pointer
  int* fake_data;

  // Allocate Unified Memory
  cudaMallocManaged(&d_data, size * sizeof(int));
  cudaMallocManaged(&fake_data, size * sizeof(int));
  cudaMemset(d_data, 0, size * sizeof(int));

  write_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, size);
#ifdef CUCORR_SYNC
  // write to fake_data to cause device sync but dont override d_data
  cudaMemset(fake_data, 0, size * sizeof(int));
#endif

  for (int i = 0; i < size; i++) {
    if (d_data[i] == 0) {
      printf("[Error] sync\n");
      break;
    }
  }
  cudaFree(fake_data);
  cudaFree(d_data);

  return 0;
}
