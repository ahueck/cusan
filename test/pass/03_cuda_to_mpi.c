// RUN: %wrapper-mpicxx -O2 -g %s -x cuda -gencode arch=compute_70,code=sm_70 -o %s.exe
// RUN: %mpi-exec -n 2 %s.exe 2>&1 | %filecheck %s

// RUN: %wrapper-mpicxx -DCUCORR_SYNC -O2 -g %s -x cuda -gencode arch=compute_70,code=sm_70 -o %s-synced.exe
// RUN: %mpi-exec -n 2 %s-synced.exe 2>&1 | %filecheck %s --allow-empty --check-prefix CHECK-SYNC

// CHECK: [Error] sync

// CHECK-SYNC-NOT: [Error] sync

#include <mpi.h>
#include <stdio.h>

#include <mpi-ext.h> /* Needed for CUDA-aware check */

__global__ void kernel(int* arr, const int N) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < N) {
#if __CUDA_ARCH__ >= 700
    for (int i = 0; i < tid; i++) {
      __nanosleep(1000000U);
    }
#else
    printf(">>> __CUDA_ARCH__ !\n");
#endif
    arr[tid] = (tid+1);
  }
}

int main(int argc, char* argv[]) {
  printf("Compile time check:\n");
#if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
  printf("This MPI library has CUDA-aware support.\n", MPIX_CUDA_AWARE_SUPPORT);
#elif defined(MPIX_CUDA_AWARE_SUPPORT) && !MPIX_CUDA_AWARE_SUPPORT
  printf("This MPI library does not have CUDA-aware support.\n");
#else
  printf("This MPI library cannot determine if there is CUDA-aware support.\n");
#endif /* MPIX_CUDA_AWARE_SUPPORT */
  printf("Run time check:\n");
#if defined(MPIX_CUDA_AWARE_SUPPORT)
  if (1 == MPIX_Query_cuda_support()) {
    printf("This MPI library has CUDA-aware support.\n");
  } else {
    printf("This MPI library does not have CUDA-aware support.\n");
  }
#else  /* !defined(MPIX_CUDA_AWARE_SUPPORT) */
  printf("This MPI library cannot determine if there is CUDA-aware support.\n");
#endif /* MPIX_CUDA_AWARE_SUPPORT */

  const int size            = 512;
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

  int* d_data;
  cudaMalloc(&d_data, size * sizeof(int));

  if (world_rank == 0) {
    kernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, size);
#ifdef CUCORR_SYNC
    cudaDeviceSynchronize(); // FIXME: uncomment for correct execution
#endif
    MPI_Send(d_data, size, MPI_INT, 1, 0, MPI_COMM_WORLD);
  } else if (world_rank == 1) {
    MPI_Recv(d_data, size, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  if (world_rank == 1) {
    int* h_data = (int*)malloc(size * sizeof(int));
    cudaMemcpy(h_data, d_data, size * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < size; i++) {
      const int buf_v = h_data[i];
      if(buf_v == 0){
        printf("[Error] sync\n");
      }
//      printf("buf[%d] = %d (r%d)\n", i, buf_v, world_rank);
    }
    free(h_data);
  }

  cudaFree(d_data);
  MPI_Finalize();
  return 0;
}
