// clang-format off
// RUN: %wrapper-mpicxx %tsan-compile-flags -O2 -g %s -x cuda -gencode arch=compute_70,code=sm_70 -o %cucorr_test_dir/%basename_t.exe
// RUN: %tsan-options %mpi-exec -n 2 %cucorr_test_dir/%basename_t.exe 2>&1 | %filecheck %s

// RUN: %apply %s --cucorr-kernel-data=%t.yaml --show_host_ir -x cuda --cuda-gpu-arch=sm_72 2>&1 | %filecheck %s  -DFILENAME=%s --allow-empty --check-prefix CHECK-LLVM-IR
// clang-format on

// CHECK-NOT: data race
// CHECK-NOT: [Error] sync

// CHECK-LLVM-IR: invoke i32 @cudaStreamCreate
// CHECK-LLVM-IR: call void @_cucorr_create_stream
// CHECK-LLVM-IR: invoke i32 @cudaMemset(i8* {{.*}}[[mset_target:%[0-9a-z]+]],
// CHECK-LLVM-IR: call void @_cucorr_memset(i8* {{.*}}[[mset_target]],
// CHECK-LLVM-IR: invoke i32 @cudaDeviceSynchronize 
// CHECK-LLVM-IR: call void @_cucorr_sync_device 
// CHECK-LLVM-IR: invoke i32 @cudaMemcpy(i8* {{.*}}[[mcpy_target:%[0-9a-z]+]], i8* {{.*}}[[mcpy_from:%[0-9a-z]+]],
// CHECK-LLVM-IR: call void @_cucorr_memcpy(i8* {{.*}}[[mcpy_target]], i8* {{.*}}[[mcpy_from]],
// CHECK-LLVM-IR: invoke i32 @cudaMemcpyAsync(i8* {{.*}}[[mcpyasy_target:%[0-9a-z]+]], i8* {{.*}}[[mcpyasy_from:%[0-9a-z]+]],
// CHECK-LLVM-IR: call void @_cucorr_memcpy_async(i8* {{.*}}[[mcpyasy_target]], i8* {{.*}}[[mcpyasy_from]],
// CHECK-LLVM-IR: invoke i32 @cudaStreamSynchronize
// CHECK-LLVM-IR: call void @_cucorr_sync_stream

// Tsan sometimes crashes with this test it seems 
// FLAKYPASS: *
// ALLOW_RETRIES: 5

#include "../support/gpu_mpi.h"

int main(int argc, char* argv[]) {
  if (!has_gpu_aware_mpi()) {
    printf("This example is designed for CUDA-aware MPI. Exiting.\n");
    return 1;
  }

  const int size            = 1<<26;//268mb
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

  int* h_data = (int*)malloc(size * sizeof(int));
  memset(h_data, 0, size*sizeof(int));
  
  int* d_data;
  cudaMalloc(&d_data, size * sizeof(int));

  cudaStream_t extraStream;
  cudaStreamCreate(&extraStream);

  if (world_rank == 0) {
    cudaMemset(d_data, 255, size * sizeof(int));
    cudaDeviceSynchronize();
    cudaMemcpy(d_data, h_data, size * sizeof(int), cudaMemcpyHostToDevice);
    MPI_Send(d_data, size, MPI_INT, 1, 0, MPI_COMM_WORLD);
  } else if (world_rank == 1) {
    MPI_Recv(d_data, size, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  if (world_rank == 1) {
    //to make sure it doesnt wait for the previous memcpy on default stream we start in another one
    cudaMemcpyAsync(h_data, d_data, size * sizeof(int), cudaMemcpyDeviceToHost, extraStream);
    cudaStreamSynchronize(extraStream);
    for (int i = 0; i < size; i++) {
      const int buf_v = h_data[i];
      if (buf_v != 0) {
        printf("[Error] sync\n");
        break;
      }
    }
    
  }
  free(h_data);
  cudaFree(d_data);
  MPI_Finalize();
  return 0;
}
