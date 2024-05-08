// clang-format off
// RUN: %wrapper-mpicxx %tsan-compile-flags -O2 -g %s -x cuda -gencode arch=compute_70,code=sm_70 -o %cucorr_test_dir/%basename_t.exe
// RUN: %tsan-options %mpi-exec -n 1 %cucorr_test_dir/%basename_t.exe 2>&1 | %filecheck %s -DFILENAME=%s

// RUN: %wrapper-mpicxx %tsan-compile-flags -DCUCORR_SYNC -O2 -g %s -x cuda -gencode arch=compute_70,code=sm_70 -o %cucorr_test_dir/%basename_t-sync.exe
// RUN: %tsan-options %mpi-exec -n 1 %cucorr_test_dir/%basename_t-sync.exe 2>&1 | %filecheck %s  -DFILENAME=%s --allow-empty --check-prefix CHECK-SYNC

// RUN: %apply %s --cucorr-kernel-data=%t.yaml --show_host_ir -x cuda --cuda-gpu-arch=sm_72 > test_out.ll
// RUN: %apply %s --cucorr-kernel-data=%t.yaml --show_host_ir -x cuda --cuda-gpu-arch=sm_72 2>&1 | %filecheck %s  -DFILENAME=%s --allow-empty --check-prefix CHECK-LLVM-IR

// clang-format on

// CHECK-DAG: data race

// CHECK-SYNC-NOT: data race

// CHECK-LLVM-IR: cudaMemcpy
// CHECK-LLVM-IR: _cucorr_memcpy
// CHECK-LLVM-IR: cudaStreamCreate
// CHECK-LLVM-IR: _cucorr_create_stream 
// CHECK-LLVM-IR: cudaStreamCreate
// CHECK-LLVM-IR: _cucorr_create_stream 
// CHECK-LLVM-IR: cudaEventCreate
// CHECK-LLVM-IR: _cucorr_create_event
// CHECK-LLVM-IR: cudaEventRecord
// CHECK-LLVM-IR: _cucorr_event_record 
// CHECK-LLVM-IR: cudaStreamSynchronize
// CHECK-LLVM-IR: _cucorr_sync_stream
// CHECK-LLVM-IR: cudaMemcpy
// CHECK-LLVM-IR: _cucorr_memcpy



#include "../support/gpu_mpi.h"

#include <unistd.h>

#define MUST_DEBUG 1
#include "TSan_External.h"

__global__ void writing_kernel(float* arr, const int N, float value) { 
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < N) {
    arr[tid] = (float)tid + value;
  }
}

__global__ void reading_kernel(float* res, const float* read, const int N, float value) { 
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < N) {
    res[tid] = read[tid]+value;
  }
}

int main(int argc, char* argv[]) {
  if (!has_gpu_aware_mpi()) {
    printf("This example is designed for CUDA-aware MPI. Exiting.\n");
    return 1;
  }
  const int size            = 512;
  const int threadsPerBlock = size;
  const int blocksPerGrid   = (size + threadsPerBlock - 1) / threadsPerBlock;

  float* h_data = (float*)malloc(size * sizeof(float));
  // Allocate device memory
  float *d_data;
  float *res_data;
  cudaMalloc(&res_data, size * sizeof(float));
  cudaMalloc(&d_data, size * sizeof(float));
  
  // Copy host memory to device
  cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
  
  // Create CUDA streams
  cudaStream_t stream1, stream2;
  cudaStreamCreate(&stream1);
  cudaStreamCreate(&stream2);
  // Create an event
  cudaEvent_t event;
  cudaEventCreate(&event);
  // Launch first kernel in stream1
  writing_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(d_data, size, 5.0f);
  
  // Record event after kernel in stream1
  cudaEventRecord(event, stream1);
  // Make stream2 wait for the event
#ifdef CUCORR_SYNC
  cudaStreamWaitEvent(stream2, event, 0);
#endif

  // Launch second kernel in stream2
  reading_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream2>>>(res_data, d_data, size, 10.0f);

  // Wait for stream2 to finish
  cudaStreamSynchronize(stream2);

  // Copy data back to host
  cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);

  cudaStreamDestroy ( stream2 );
  cudaStreamDestroy ( stream1 );
  cudaEventDestroy(event);
  cudaFree(d_data);
  free(h_data);
  return 0;
}
