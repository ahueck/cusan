// clang-format off
// RUN: %wrapper-cxx %tsan-compile-flags -O2 -g %s -x cuda -gencode arch=compute_70,code=sm_70 -o %cucorr_test_dir/%basename_t.exe
// RUN: %tsan-options %cucorr_test_dir/%basename_t.exe 2>&1 | %filecheck %s -DFILENAME=%s

//// RUN: %wrapper-cxx %tsan-compile-flags -DCUCORR_SYNC -O2 -g %s -x cuda -gencode arch=compute_70,code=sm_70 -o %cucorr_test_dir/%basename_t-sync.exe
//// RUN: %tsan-options %cucorr_test_dir/%basename_t-sync.exe 2>&1 | %filecheck %s  -DFILENAME=%s --allow-empty --check-prefix CHECK-SYNC
// clang-format on

// CHECK-NOT: data race
// CHECK-NOT: [Error] sync

//// CHECK-SYNC-NOT: data race
//// CHECK-SYNC-NOT: [Error] sync

#include <cstdio>
#include <mpi.h>
#include <cuda_runtime.h>

#define SENDER 0
#define RECIEVER 1
#define CHUNKS 4
#define SIZE 1024
// Example size, you can adjust as needed

__global__ void computation_kernel(double* buf, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        buf[idx] = 1.0; // Example computation
    }
}

void computation_on_GPU(double* dev_buf, cudaStream_t kernel_stream) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (SIZE + threadsPerBlock - 1) / threadsPerBlock;
    computation_kernel<<<blocksPerGrid, threadsPerBlock, 0, kernel_stream>>>(dev_buf, SIZE);
}

void more_computation_on_GPU(double* dev_buf) {
    // Placeholder for additional GPU computations
    // Launch more kernels or perform other GPU tasks here
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if (world_size != 2) {
        printf("[Error] This example is designed for 2 MPI processes. Exiting.\n");
        MPI_Finalize();
        return 1;
    }

    double *dev_buf, *host_buf;
    cudaStream_t kernel_stream, streams[CHUNKS];
    cudaMalloc(&dev_buf, SIZE * sizeof(double));
    cudaMallocHost(&host_buf, SIZE * sizeof(double));

    // Create CUDA streams
    cudaStreamCreate(&kernel_stream);
    for (int j = 0; j < CHUNKS; j++) {
        cudaStreamCreate(&streams[j]);
    }

    if (world_rank == SENDER) { /* sender */
        computation_on_GPU(dev_buf, kernel_stream);

        // Explicit GPU sync between GPU streams
        cudaStreamSynchronize(kernel_stream);

        // Calculate chunk size and offset
        int chunk_size = SIZE / CHUNKS;
        for (int j = 0; j < CHUNKS; j++) {
            int offset = j * chunk_size;
            cudaMemcpyAsync(host_buf + offset, dev_buf + offset,
                            chunk_size * sizeof(double), cudaMemcpyDeviceToHost, streams[j]);
        }

        for (int j = 0; j < CHUNKS; j++) {
            // Explicit GPU sync before MPI
            cudaStreamSynchronize(streams[j]);
            int offset = j * chunk_size;
            MPI_Request req;
            MPI_Isend(host_buf + offset, chunk_size, MPI_DOUBLE, RECIEVER, 0, MPI_COMM_WORLD, &req);
        }

        // Explicit MPI sync before GPU kernel
        MPI_Barrier(MPI_COMM_WORLD);

        more_computation_on_GPU(dev_buf);

    } else if (world_rank == RECIEVER) { /* receiver */
        // Calculate chunk size and offset
        int chunk_size = SIZE / CHUNKS;
        MPI_Request requests[CHUNKS];
        for (int j = 0; j < CHUNKS; j++) {
            int offset = j * chunk_size;
            MPI_Irecv(host_buf + offset, chunk_size, MPI_DOUBLE, SENDER, 0, MPI_COMM_WORLD, &requests[j]);
        }

        MPI_Waitall(CHUNKS, requests, MPI_STATUSES_IGNORE);

        // Use the received data (host_buf) on the GPU or CPU as needed
        // Example: Copy received data to the GPU
        cudaMemcpy(dev_buf, host_buf, SIZE * sizeof(double), cudaMemcpyHostToDevice);

        more_computation_on_GPU(dev_buf);
    }

    // Cleanup
    cudaFree(dev_buf);
    cudaFreeHost(host_buf);
    cudaStreamDestroy(kernel_stream);
    for (int j = 0; j < CHUNKS; j++) {
        cudaStreamDestroy(streams[j]);
    }

    MPI_Finalize();
    return 0;
}