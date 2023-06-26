// RUN: TYPEART_WRAPPER_EMIT_IR=1 %wrapper-cc -O1 -g %s -x cuda -gencode arch=compute_50,code=sm_50 -o %s.exe

#include <stdio.h>
__global__ void axpy(float a, float* x, float* y) {
  y[threadIdx.x] = a * x[threadIdx.x];
}

int main(int argc, char* argv[]) {
  const int kDataLen = 4;

  float a                = 2.0f;
  float host_x[kDataLen] = {1.0f, 2.0f, 3.0f, 4.0f};
  float host_y[kDataLen];

  float* device_x;
  float* device_y;
  cudaMalloc((void**)&device_x, kDataLen * sizeof(float));
  cudaMalloc((void**)&device_y, kDataLen * sizeof(float));

  cudaMemcpy(device_x, host_x, kDataLen * sizeof(float), cudaMemcpyHostToDevice);

  axpy<<<1, kDataLen>>>(a, device_x, device_y);

  cudaDeviceSynchronize();
  cudaMemcpy(host_y, device_y, kDataLen * sizeof(float), cudaMemcpyDeviceToHost);

  for (int i = 0; i < kDataLen; ++i) {
    printf("y[%i] = %f\n",i, host_y[i]);
  }

  cudaDeviceReset();
  return 0;
}