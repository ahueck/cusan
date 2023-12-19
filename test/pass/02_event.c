// RUN: %wrapper-cc -O1 -g %s -x cuda -gencode arch=compute_70,code=sm_70 -o %s.exe
// RUN: %s.exe 2>&1 | %filecheck %s

// RUN: %wrapper-cc -DCUCORR_SYNC -O2 -g %s -x cuda -gencode arch=compute_70,code=sm_70 -o %s-synced.exe
// RUN: %s-synced.exe 2>&1 | %filecheck %s --allow-empty --check-prefix CHECK-SYNC

// CHECK: [Error] sync

// CHECK-SYNC-NOT: [Error] sync

#include <cuda_runtime.h>
#include <stdio.h>

__global__ void kernel(int* data)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    data[tid] = (tid+1);
}

int main()
{
    const int size = 256;
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    int* d_data;  // Unified Memory pointer

    // Allocate Unified Memory
    cudaMallocManaged(&d_data, size * sizeof(int));

    cudaEvent_t endEvent;
    cudaEventCreate(&endEvent);
    kernel<<<blocksPerGrid, threadsPerBlock>>>(d_data);
    cudaEventRecord(endEvent);

#ifdef CUCORR_SYNC
    // Wait for the end event to complete (alternative to querying)
    cudaEventSynchronize(endEvent);
#endif

    for (int i = 0; i < size; i++) {
      if(d_data[i] < 1) {
        printf("[Error] sync\n");
      }
//      printf("d_data[%d] = %d\n", i, d_data[i]);
    }

    cudaEventDestroy(endEvent);
    cudaFree(d_data);

    return 0;
}

