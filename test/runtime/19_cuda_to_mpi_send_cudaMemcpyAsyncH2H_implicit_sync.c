// clang-format off
// RUN: %wrapper-mpicxx %tsan-compile-flags -O1 -g %s -x cuda -gencode arch=compute_70,code=sm_70 -o %cusan_test_dir/%basename_t.exe
// RUN: %cusan_ldpreload %tsan-options %mpi-exec -n 2 %cusan_test_dir/%basename_t.exe 2>&1 | %filecheck %s --allow-empty

// RUN: %wrapper-mpicxx %tsan-compile-flags -DCUSAN_SYNC -O1 -g %s -x cuda -gencode arch=compute_70,code=sm_70 -o %cusan_test_dir/%basename_t-sync.exe
// RUN: %cusan_ldpreload %tsan-options %mpi-exec -n 2 %cusan_test_dir/%basename_t-sync.exe 2>&1 | %filecheck %s --allow-empty --check-prefix CHECK-SYNC

// clang-format on

// CHECK-DAG: data race
// CHECK-DAG: [Error] sync

// CHECK-SYNC-NOT: data race
// CHECK-SYNC-NOT: [Error] sync



#include "../support/gpu_mpi.h"

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

  int* data;
  int* h_data  = (int*)malloc(sizeof(int));
  int* h_data2 = (int*)malloc(sizeof(int));
  int* h_data3 = (int*)malloc(size * sizeof(int));
  cudaStream_t stream1;
  cudaStreamCreate(&stream1);

  cudaMalloc(&data, size * sizeof(int));
  cudaMemset(data, 0, size * sizeof(int));

  cudaDeviceSynchronize();

  if (world_rank == 0) {
    write_kernel_delay<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(data, size, 1316134912);
#ifdef CUSAN_SYNC
    cudaMemcpy(h_data, h_data2, sizeof(int), cudaMemcpyHostToHost);
#endif
    MPI_Send(data, size, MPI_INT, 1, 0, MPI_COMM_WORLD);
  } else if (world_rank == 1) {
    MPI_Recv(data, size, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    cudaMemcpyAsync(h_data3, data, size * sizeof(int), cudaMemcpyDefault, stream1);
    cudaStreamSynchronize(stream1);
    for (int i = 0; i < size; i++) {
      if (h_data3[i] == 0) {
        printf("[Error] sync %i\n", h_data3[i]);
        break;
      }
    }
  }

  free(h_data);
  free(h_data2);
  free(h_data3);
  cudaFree(data);
  cudaStreamDestroy(stream1);
  MPI_Finalize();
  return 0;
}
