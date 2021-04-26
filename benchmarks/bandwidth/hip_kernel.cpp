#include "kernel.h"

void copyData(float *To, float *From, size_t NumElements, size_t blocks, size_t threads, void *stream) {
  hipLaunchKernelGGL(kernel_copyData, dim3(blocks), dim3(threads), 0, 0, To, From, NumElements);
}

__global__ void kernel_copyData(float *To, float *From, size_t NumElements) {
  int Idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (Idx < NumElements) {
    To[Idx] = From[Idx];
  }
}
