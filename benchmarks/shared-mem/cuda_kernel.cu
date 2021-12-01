#include "kernel.h"
#include <iostream>
#include <cuda.h>

#include "interfaces/cuda/Internals.h"

#define WARP_SIZE 32


// https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
__device__ void copy(float *in, float *out) {
  uint64_t tid = threadIdx.x;

  #pragma unroll
  for (size_t r{0}; r < REPEATS; ++r) {
    asm volatile ("{\t\n"
                  ".reg .f32 data;\n\t"
                  ".reg .u64 r0;\n\t"
                  ".reg .u64 r1;\n\t"
                  "cvta.to.shared.u64 r0, %0;\n\t"
                  "cvta.to.shared.u64 r1, %1;\n\t"
                  "ld.shared.f32 data, [r1];\n\t"
                  "st.shared.f32 [r0], data;\n\t"
                  "}" :: "l"(out + tid), "l"(in + tid) : "memory");

  }
}


__global__ void kernel_shrMemBW(uint32_t *clocks, float* scratch, float magicNumber) {
  __shared__ float in[WARP_SIZE];
  __shared__ float out[WARP_SIZE];

  uint64_t tid = threadIdx.x;
  in[tid] = magicNumber;
  asm volatile ("bar.warp.sync  0xffffffff;\n\t" ::);

  // warm up
  copy(in, out);
  asm volatile ("bar.warp.sync  0xffffffff;\n\t" ::);


  // measure
  uint32_t start{0};
  uint32_t end{0};
  asm volatile ("mov.u32 %0, %%clock;" : "=r"(start) :: "memory");
  copy(in, out);
  asm volatile ("bar.warp.sync  0xffffffff;\n\t" ::);
  asm volatile ("mov.u32 %0, %%clock;" : "=r"(end) :: "memory");

  scratch[tid] = out[tid];
  if (tid == 0) {
    clocks[START] = start;
    clocks[END] = end;
  }
}


void shrMemBW(uint32_t *clocks, float* scratch, float magicNumber) {
  dim3 block(WARP_SIZE, 1, 1);
  dim3 grid(1, 1, 1);
  kernel_shrMemBW<<<grid, block>>>(clocks, scratch, magicNumber); CHECK_ERR;

}


double getPeakGPUFrequency(int deviceId) {
  int peakClock{0};
  cudaDeviceGetAttribute(&peakClock, cudaDevAttrClockRate, deviceId); CHECK_ERR;
  // Note: cure returns frequency in kHz
  return static_cast<double>(peakClock * 1000);
}


long long getNumTransferedBytes() {
  constexpr long long numReadWritePerInteration{2};
  return static_cast<long long>(numReadWritePerInteration * WARP_SIZE * REPEATS * sizeof(float));
}


size_t getLaneSize() {
  return WARP_SIZE;
};

