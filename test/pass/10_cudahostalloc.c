// clang-format off
// RUN: %wrapper-cxx %tsan-compile-flags -O2 -g %s -x cuda -gencode arch=compute_70,code=sm_70 -o %cucorr_test_dir/%basename_t.exe
// RUN: %tsan-options %cucorr_test_dir/%basename_t.exe 2>&1 | %filecheck --allow-empty %s

// RUN: %apply %s --cucorr-kernel-data=%t.yaml --show_host_ir -x cuda --cuda-gpu-arch=sm_72 2>&1 | %filecheck %s  -DFILENAME=%s --allow-empty --check-prefix CHECK-LLVM-IR
// clang-format on

// CHECK-NOT: data race
// CHECK-NOT: [Error] sync

// CHECK-LLVM-IR: @main(i32 noundef %0, i8** noundef %1)
// HECK-LLVM-IR: cudaMallocHost
// HECK-LLVM-IR: _cucorr_host_alloc
// CHECK-LLVM-IR: cudaHostAlloc
// CHECK-LLVM-IR: _cucorr_host_alloc
// HECK-LLVM-IR: cudaFreeHost
// HECK-LLVM-IR: _cucorr_host_free
// CHECK-LLVM-IR: cudaFreeHost
// CHECK-LLVM-IR: _cucorr_host_free

#include <cuda_runtime.h>
#include <stdio.h>

int main(int argc, char* argv[]) {
  const int size            = 512;
  //int* h_data1;
  //cudaMallocHost(&h_data1, size*sizeof(int));
  int* h_data2;
  cudaHostAlloc(&h_data2, size*sizeof(int), cudaHostAllocDefault);

  //cudaFreeHost(h_data1);
  cudaFreeHost(h_data2);
  return 0;
}
