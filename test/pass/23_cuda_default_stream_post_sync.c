// clang-format off
// RUN: %wrapper-cxx %tsan-compile-flags -O2 -g %s -x cuda -gencode arch=compute_70,code=sm_70 -o %cusan_test_dir/%basename_t.exe
// RUN: %tsan-options %cusan_test_dir/%basename_t.exe 2>&1 | %filecheck %s -DFILENAME=%s

// RUN: %wrapper-cxx %tsan-compile-flags -DCUSAN_SYNC -O2 -g %s -x cuda -gencode arch=compute_70,code=sm_70 -o %cusan_test_dir/%basename_t-sync.exe
// RUN: %tsan-options %cusan_test_dir/%basename_t-sync.exe 2>&1 | %filecheck %s  -DFILENAME=%s --allow-empty --check-prefix CHECK-SYNC
// clang-format on

// CHECK-DAG: data race
// CHECK-DAG: [Error] sync

// CHECK-SYNC-NOT: data race
// CHECK-SYNC-NOT: [Error] sync

#include "../support/gpu_mpi.h"

#include <unistd.h>

__global__ void write_kernel_delay(int* arr, const int N, int value, const unsigned int delay) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
#if __CUDA_ARCH__ >= 700
  for (int i = 0; i < tid; i++) {
    __nanosleep(delay);
  }
#else
  printf(">>> __CUDA_ARCH__ !\n");
#endif
  if (tid < N) {
    arr[tid] = value;
  }
}

int main(int argc, char* argv[]) {
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  const int size            = 512;
  const int threadsPerBlock = size;
  const int blocksPerGrid   = (size + threadsPerBlock - 1) / threadsPerBlock;

  int* managed_data;
  cudaMallocManaged(&managed_data, size * sizeof(int));
  cudaMemset(managed_data, 0, size * sizeof(int));

  int* d_data2;
  cudaMalloc(&d_data2, size * sizeof(int));
  cudaDeviceSynchronize();

  write_kernel_delay<<<blocksPerGrid, threadsPerBlock, 0, 0>>>(managed_data, size, 128, 9999999);
  write_kernel_delay<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_data2, size, 0, 1);

#ifdef CUSAN_SYNC
  cudaStreamSynchronize(stream);
#endif
  for (int i = 0; i < size; i++) {
    if (managed_data[i] == 0) {
      printf("[Error] sync %i %i\n", managed_data[i], i);
      break;
    }
  }

  cudaStreamDestroy(stream);
  cudaFree(managed_data);
  cudaFree(d_data2);
  return 0;
}
