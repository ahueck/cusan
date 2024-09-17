// clang-format off

// RUN: %apply %s -strip-debug --cusan-kernel-data=%t.yaml --show_host_ir -x cuda --cuda-gpu-arch=sm_72 2>&1 | %filecheck %s  -DFILENAME=%s --allow-empty --check-prefix CHECK-LLVM-IR
// clang-format on


// CHECK-LLVM-IR: {{call|invoke}} i32 @cudaStreamCreateWithFlags
// CHECK-LLVM-IR: {{call|invoke}} void @_cusan_create_stream
// CHECK-LLVM-IR: {{call|invoke}} i32 @cudaMemset
// CHECK-LLVM-IR: {{call|invoke}} void @_cusan_memset
// CHECK-LLVM-IR: {{call|invoke}} i32 @cudaStreamSynchronize
// CHECK-LLVM-IR: {{call|invoke}} void @_cusan_sync_stream
// CHECK-LLVM-IR: {{call|invoke}} i32 @cudaStreamDestroy
// CHECK-LLVM-IR: {{call|invoke}} i32 @cudaFree
// CHECK-LLVM-IR: {{call|invoke}} void @_cusan_device_free

#include "../../support/gpu_mpi.h"

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
#ifdef CUSAN_SYNC
  cudaStreamCreate(&stream);
#else
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
#endif

  const int size            = 512;
  const int threadsPerBlock = size;
  const int blocksPerGrid   = (size + threadsPerBlock - 1) / threadsPerBlock;

  int* managed_data;
  cudaMallocManaged(&managed_data, size * sizeof(int));
  cudaMemset(managed_data, 0, size * sizeof(int));

  write_kernel_delay<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(managed_data, size, 128, 9999999);
  cudaStreamSynchronize(0);

  for (int i = 0; i < size; i++) {
    if (managed_data[i] == 0) {
      printf("[Error] sync %i %i\n", managed_data[i], i);
      break;
    }
  }

  cudaStreamDestroy(stream);
  cudaFree(managed_data);
  return 0;
}
