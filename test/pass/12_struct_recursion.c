// clang-format off
// RUN: %wrapper-cxx %tsan-compile-flags -O2 -g %s -x cuda -gencode arch=compute_70,code=sm_70 -o %cucorr_test_dir/%basename_t.exe
// RUN: %tsan-options %cucorr_test_dir/%basename_t.exe 2>&1 | %filecheck %s

// RUN: %wrapper-cxx %tsan-compile-flags -DCUCORR_SYNC -O2 -g %s -x cuda -gencode arch=compute_70,code=sm_70 -o %cucorr_test_dir/%basename_t-sync.exe
// RUN: %tsan-options %cucorr_test_dir/%basename_t-sync.exe 2>&1 | %filecheck %s --allow-empty --check-prefix CHECK-SYNC

// RUN: %apply %s --cucorr-kernel-data=%t.yaml --show_host_ir -x cuda --cuda-gpu-arch=sm_72 > test_out.ll


// CHECK-DAG: data race

// CHECK-SYNC-NOT: data race



#include "../support/gpu_mpi.h"

struct BufferStorage2{
  int* buff;
};

struct BufferStorage{
  BufferStorage2 buff1;
  BufferStorage2 buff2;
};


__global__ void kernel1(BufferStorage storage, const int N) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < N) {
    storage.buff1.buff[tid] = tid*32;
  }
}
__global__ void kernel2(BufferStorage storage, const int N) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < N) {
    storage.buff2.buff[tid] = tid*32;
  }
}

__global__ void kernel3(BufferStorage storage, const int N) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < N) {
    storage.buff2.buff[tid] = tid*32;
  }
}


int main(int argc, char* argv[]) {
  const int size            = 512;
  const int threadsPerBlock = size;
  const int blocksPerGrid   = (size + threadsPerBlock - 1) / threadsPerBlock;

  BufferStorage buffStor;
  cudaMalloc(&buffStor.buff1.buff, size * sizeof(int));
  cudaMalloc(&buffStor.buff2.buff, size * sizeof(int));

  cudaStream_t stream1, stream2;
  cudaStreamCreate(&stream1);
  cudaStreamCreate(&stream2);

  kernel1<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(buffStor, size);
  kernel3<<<blocksPerGrid, threadsPerBlock, 0, stream2>>>(buffStor, size);
  kernel2<<<blocksPerGrid, threadsPerBlock, 0, stream2>>>(buffStor, size);
#ifdef CUCORR_SYNC
  cudaDeviceSynchronize();
#endif
  kernel3<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(buffStor, size);

  cudaDeviceSynchronize();

  cudaStreamDestroy ( stream2 );
  cudaStreamDestroy ( stream1 );
  cudaFree(buffStor.buff1.buff);
  cudaFree(buffStor.buff2.buff);
  return 0;
}
