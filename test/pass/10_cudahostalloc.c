// clang-format off
// RUN: %wrapper-cxx %tsan-compile-flags -O2 -g %s -x cuda -gencode arch=compute_70,code=sm_70 -o %cucorr_test_dir/%basename_t.exe
// RUN: %tsan-options %cucorr_test_dir/%basename_t.exe 2>&1 | %filecheck --allow-empty %s

// RUN: %apply %s --cucorr-kernel-data=%t.yaml --show_host_ir -x cuda --cuda-gpu-arch=sm_72 2>&1 | %filecheck %s  -DFILENAME=%s --allow-empty --check-prefix CHECK-LLVM-IR
// clang-format on

// CHECK-NOT: data race
// CHECK-NOT: [Error] sync

// CHECK-LLVM-IR: @main(i32 noundef %0, i8** noundef %1)
// CHECK-LLVM-IR: invoke i32 @cudaMallocHost
// CHECK-LLVM-IR: call void @_cucorr_host_alloc
// CHECK-LLVM-IR: invoke noundef i32 @_ZL13cudaHostAllocIiE9cudaErrorPPT_mj
// CHECK-LLVM-IR: invoke i32 @cudaFreeHost({{.*}}[[free_ptr1:%[0-9a-z]+]])
// CHECK-LLVM-IR: call void @_cucorr_host_free({{.*}}[[free_ptr1]])
// CHECK-LLVM-IR: invoke i32 @cudaFreeHost({{.*}}[[free_ptr2:%[0-9a-z]+]])
// CHECK-LLVM-IR: call void @_cucorr_host_free({{.*}}[[free_ptr2]])

// CHECK-LLVM-IR: _ZL13cudaHostAllocIiE9cudaErrorPPT_mj
// CHECK-LLVM-IR: invoke i32 @cudaHostAlloc({{.*}}[[host_alloc_ptr:%[0-9a-z]+]])
// CHECK-LLVM-IR: call void @_cucorr_host_alloc({{.*}}[[host_alloc_ptr]])

#include <cuda_runtime.h>
#include <stdio.h>

int main(int argc, char* argv[]) {
  const int size            = 512;
  int* h_data1;
  cudaMallocHost((void**)&h_data1, size*sizeof(int));
  int* h_data2;
  cudaHostAlloc(&h_data2, size*sizeof(int), cudaHostAllocDefault);
  cudaFreeHost(h_data1);
  cudaFreeHost(h_data2);
  return 0;
}
