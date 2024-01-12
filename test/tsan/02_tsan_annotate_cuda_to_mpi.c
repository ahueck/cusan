// clang-format off
// TODO: Fix segfault when program terminates.

// RUN: %wrapper-mpicxx %tsan-compile-flags -O2 -g %s -x cuda -gencode arch=compute_70,code=sm_70 -o %cucorr_test_dir/%basename_t.exe
// RUN: %tsan-options %mpi-exec -n 2 %cucorr_test_dir/%basename_t.exe 2>&1 | %filecheck %s -DFILENAME=%s

// RUN: %wrapper-mpicxx %tsan-compile-flags -DCUCORR_SYNC -O2 -g %s -x cuda -gencode arch=compute_70,code=sm_70 -o %cucorr_test_dir/%basename_t-sync.exe
// RUN: %tsan-options %mpi-exec -n 2 %cucorr_test_dir/%basename_t-sync.exe 2>&1 | %filecheck %s  -DFILENAME=%s --allow-empty --check-prefix CHECK-SYNC
// clang-format on

// CHECK-DAG: [Error] sync
// CHECK-DAG: data race

// CHECK-SYNC-NOT: [Error] sync
// CHECK-SYNC-NOT: data race

#include "../support/gpu_mpi.h"

#include <unistd.h>

#define MUST_DEBUG 1
#include "TSan_External.h"

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
    arr[tid] = (tid + 1);
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
    // get main fiber, create CUDA fiber
    int dummy;
    void* mycurrfiber = __tsan_get_current_fiber();
    void* mycudafiber = __tsan_create_fiber(0);
    __tsan_set_fiber_name(mycudafiber, "CUDA");

    // switch to fiber *with* synchronization
    __tsan_switch_to_fiber(mycudafiber, 0);

    // actual kernel call
    kernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, size);

    // assume worst case: memory write to d_data of "size"
    TsanMemoryWrite(d_data, size * sizeof(int));  // CHECK-DAG: [[FILENAME]]:[[@LINE]]

    // write in sync clock for later synchronization
    TsanHappensBefore(&dummy);  // release

    // switch to main fiber *without* synchronization, since the CUDA memory access is ongoing
    __tsan_switch_to_fiber(mycurrfiber, 1);

#ifdef CUCORR_SYNC
    cudaDeviceSynchronize();  // FIXME: uncomment for correct execution
    // if we synchronize with cudaDeviceSynchronize, we can be sure that the CUDA memory access is completed here
    TsanHappensAfter(&dummy);  // aquire
#endif

    MPI_Send(d_data, size, MPI_INT, 1, 0, MPI_COMM_WORLD);
    // annotate local buffer read of MPI_Send as memory read in main thread
    TsanMemoryRead(d_data, size * sizeof(int));  // CHECK-DAG: [[FILENAME]]:[[@LINE]]

  } else if (world_rank == 1) {
    MPI_Recv(d_data, size, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    // annotate local buffer write of MPI_Recv as memory write in main thread
    TsanMemoryWrite(d_data, size * sizeof(int));
  }

  if (world_rank == 1) {
    int* h_data = (int*)malloc(size * sizeof(int));
    cudaMemcpy(h_data, d_data, size * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < size; i++) {
      const int buf_v = h_data[i];
      if (buf_v == 0) {
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
