// clang-format off

// RUN: %apply %s -strip-debug --cusan-kernel-data=%t.yaml --show_host_ir -x cuda --cuda-gpu-arch=sm_72 2>&1 | %filecheck %s  -DFILENAME=%s --allow-empty --check-prefix CHECK-LLVM-IR

// clang-format on


// CHECK-LLVM-IR: invoke i32 @cudaStreamCreate
// CHECK-LLVM-IR: invoke void @_cusan_create_stream
// CHECK-LLVM-IR: invoke i32 @cudaEventCreate
// CHECK-LLVM-IR: invoke void @_cusan_create_event
// CHECK-LLVM-IR: invoke i32 @cudaMemset
// CHECK-LLVM-IR: invoke void @_cusan_memset
// CHECK-LLVM-IR: invoke i32 @cudaEventRecord
// CHECK-LLVM-IR: invoke void @_cusan_event_record
// CHECK-LLVM-IR: invoke i32 @cudaFree
// CHECK-LLVM-IR: invoke void @_cusan_device_free
// CHECK-LLVM-IR: invoke i32 @cudaStreamDestroy
// CHECK-LLVM-IR: invoke i32 @cudaEventDestroy


#include "../../support/gpu_mpi.h"

#include <cstdio>

__global__ void write_kernel_delay(int* arr, const int N, const unsigned int delay) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
#if __CUDA_ARCH__ >= 700
  for (int i = 0; i < tid; i++) {
    __nanosleep(delay);
  }
#else
  printf(">>> __CUDA_ARCH__ !\n");
#endif
  if (tid < N) {
    arr[tid] = (tid + 1);
  }
}

int main(int argc, char* argv[]) {
  if (!has_gpu_aware_mpi()) {
    printf("This example is designed for CUDA-aware MPI. Exiting.\n");
    return 1;
  }

  const int size            = 256;
  const int threadsPerBlock = size;
  const int blocksPerGrid   = (size + threadsPerBlock - 1) / threadsPerBlock;

  MPI_Init(&argc, &argv);
  int world_size, world_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  if (world_size != 2) {
    printf("This example is designed for 2 MPI processes. Exiting.\n");
    MPI_Finalize();
    return 1;
  }

  int* managed_data;
  cudaMallocManaged(&managed_data, size * sizeof(int));
  cudaStream_t stream1;
  cudaStreamCreate(&stream1);
  cudaEvent_t event1;
  cudaEventCreate(&event1);

  if (world_rank == 0) {
    cudaMemset(managed_data, 0, size * sizeof(int));
    write_kernel_delay<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(managed_data, size, 1316134912);
    cudaEventRecord(event1, stream1);
#ifdef CUSAN_SYNC
    while (cudaEventQuery(event1) != cudaSuccess) {
    }
#endif
    MPI_Send(managed_data, size, MPI_INT, 1, 0, MPI_COMM_WORLD);
  } else if (world_rank == 1) {
    MPI_Recv(managed_data, size, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    for (int i = 0; i < size; i++) {
      if (managed_data[i] == 0) {
        printf("[Error] sync %i\n", managed_data[i]);
        break;
      }
    }
  }

  cudaFree(managed_data);
  cudaStreamDestroy(stream1);
  cudaEventDestroy(event1);
  MPI_Finalize();
  return 0;
}
