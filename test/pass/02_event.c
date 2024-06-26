// clang-format off
// RUN: %wrapper-cc %tsan-compile-flags -O2 -g %s -x cuda -gencode arch=compute_70,code=sm_70 -o %cucorr_test_dir/%basename_t.exe
// RUN: %cucorr_ldpreload %tsan-options %cucorr_test_dir/%basename_t.exe 2>&1 | %filecheck %s

// RUN: %wrapper-cc %tsan-compile-flags -DCUCORR_SYNC -O2 -g %s -x cuda -gencode arch=compute_70,code=sm_70 -o %cucorr_test_dir/%basename_t-sync.exe
// RUN: %cucorr_ldpreload %tsan-options %cucorr_test_dir/%basename_t-sync.exe 2>&1 | %filecheck %s --allow-empty --check-prefix CHECK-SYNC

// RUN: %apply %s --cucorr-kernel-data=%t.yaml --show_host_ir -x cuda --cuda-gpu-arch=sm_72 2>&1 | %filecheck %s  -DFILENAME=%s --allow-empty --check-prefix CHECK-LLVM-IR

// clang-format on

// CHECK-DAG: data race
// CHECK-DAG: [Error] sync

// CHECK-SYNC-NOT: data race
// CHECK-SYNC-NOT: [Error] sync

// CHECK-LLVM-IR: invoke i32 @cudaEventCreate
// CHECK-LLVM-IR: {{call|invoke}} void @_cucorr_create_event

// CHECK-LLVM-IR: invoke i32 @cudaEventRecord
// CHECK-LLVM-IR: {{call|invoke}} void @_cucorr_event_record

#include <cuda_runtime.h>
#include <stdio.h>

__global__ void kernel(int* data) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
#if __CUDA_ARCH__ >= 700
  for (int i = 0; i < tid; i++) {
    __nanosleep(1000000U);
  }
#else
  printf(">>> __CUDA_ARCH__ !\n");
#endif
  data[tid] = (tid + 1);
}

int main() {
  const int size            = 256;
  const int threadsPerBlock = 256;
  const int blocksPerGrid   = (size + threadsPerBlock - 1) / threadsPerBlock;

  int* d_data;  // Unified Memory pointer

  // Allocate Unified Memory
  cudaMallocManaged(&d_data, size * sizeof(int));
  cudaMemset(d_data, 0, size * sizeof(int));

  cudaEvent_t endEvent;
  cudaEventCreate(&endEvent);
  kernel<<<blocksPerGrid, threadsPerBlock>>>(d_data);
  cudaEventRecord(endEvent);

#ifdef CUCORR_SYNC
  // Wait for the end event to complete (alternative to querying)
  cudaEventSynchronize(endEvent);
#endif

  for (int i = 0; i < size; i++) {
    if (d_data[i] < 1) {
      printf("[Error] sync\n");
      break;
    }
  }

  cudaEventDestroy(endEvent);
  cudaFree(d_data);

  return 0;
}
