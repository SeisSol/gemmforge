
#include <iostream>
#include <cuda.h>
#include <cstring>

#define CHECK_ERR checkErr(__FILE__,__LINE__)

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* const func, const char* const file,
           const int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        // We don't exit when we encounter CUDA errors in this example.
        // std::exit(EXIT_FAILURE);
    }
}

std::string PrevFile = "";
int PrevLine = 0;

void checkErr(const std::string &File, int Line) {
#ifndef NDEBUG
      cudaError_t Error = cudaGetLastError();
      if (Error != cudaSuccess) {
        std::cout << std::endl << File
                  << ", line " << Line
                  << ": " << cudaGetErrorString(Error)
                  << " (" << Error << ")"
                  << std::endl;

        if (PrevLine > 0)
          std::cout << "Previous CUDA call:" << std::endl
                    << PrevFile << ", line " << PrevLine << std::endl;
        throw;
      }
      PrevFile = File;
      PrevLine = Line;
#endif
}

// Dense x Dense Kernel
__global__ void 
__launch_bounds__(64)
 kernel_sgemm_T_NT_m9_n9_k56_lda9_ldb9_ldc56_alpha_1_beta_1_sss_0b51dc0(const float * A, int A_extraOffset, const float * B, int B_extraOffset, float * C, int C_extraOffset, unsigned numElements, unsigned* flags) {
  unsigned batchID = (threadIdx.y + blockDim.y * blockIdx.x);
  if (batchID < numElements) {
    bool isFlagsProvided = (flags != nullptr);
    bool allowed = isFlagsProvided ? static_cast<bool>(flags[batchID]) : true;
    if (allowed) {
      const float * const __restrict__ glb_A = &A[batchID * 504 + 0 + A_extraOffset];
      const float * const __restrict__ glb_B = &B[batchID * 81 + 0 + B_extraOffset];
      float * const __restrict__ glb_C = &C[batchID * 504 + 0 + C_extraOffset];
      float reg0[9] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
      __shared__  __align__(8) float totalShrMem[612];
      float * localShrMem0 = &totalShrMem[612 * threadIdx.y];

      float* shrRegion0 = &localShrMem0[0];
      // using ExtendedTransposePatchLoader
      {
        int index;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
          index = threadIdx.x + i * 64;
          shrRegion0[(index % 9) * 59 + index / 9] = glb_A[threadIdx.x + i * 64];
        }
        if (threadIdx.x < 19) {
          index = threadIdx.x + 512;
          shrRegion0[(index % 9) * 59 + index / 9] = glb_A[threadIdx.x + 512];
        }
      }

      float* shrRegion1 = &localShrMem0[531];
      // using ExtendedPatchLoader
      {
        shrRegion1[threadIdx.x + 0] = glb_B[threadIdx.x + 0];
        if (threadIdx.x < 17) {
          shrRegion1[threadIdx.x + 64] = glb_B[threadIdx.x + 64];
        }
      }
      __syncthreads();
      if (threadIdx.x < 56) {
        float value;

        for (int k = 0; k < 9; ++k) {
          value = shrRegion0[threadIdx.x + k * 59];

          #pragma unroll
          for (int n = 0; n < 9; ++n) {
            reg0[n] += value * shrRegion1[k + 9 * n];
          }
        }
      }
      if (threadIdx.x < 56) {
        #pragma unroll
        for (int n = 0; n < 9; ++n) {
          glb_C[threadIdx.x + 56 * n] = reg0[n] + glb_C[threadIdx.x + 56 * n];
        }
      }
    }
  }
}


// Dense x Sparse Kernel
__global__ void 
__launch_bounds__(64)
 kernel_sgemm_T_NT_m9_n9_k56_lda9_ldb9_ldc56_alpha_1_beta_1_sss_b15c35b(const float * A, int A_extraOffset, const float * B, int B_extraOffset, float * C, int C_extraOffset, unsigned numElements, unsigned* flags) {
  unsigned batchID = (threadIdx.y + blockDim.y * blockIdx.x);
  if (batchID < numElements) {
    bool isFlagsProvided = (flags != nullptr);
    bool allowed = isFlagsProvided ? static_cast<bool>(flags[batchID]) : true;
    if (allowed) {
      const float * const __restrict__ glb_A = &A[batchID * 504 + 0 + A_extraOffset];
      const float * const __restrict__ glb_B = &B[batchID * 9 + 0 + B_extraOffset];
      float * const __restrict__ glb_C = &C[batchID * 504 + 0 + C_extraOffset];
      float reg0[9] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
      __shared__  __align__(8) float totalShrMem[540];
      float * localShrMem0 = &totalShrMem[540 * threadIdx.y];

      float* shrRegion0 = &localShrMem0[0];
      // using ExtendedTransposePatchLoader
      {
        int index;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
          index = threadIdx.x + i * 64;
          shrRegion0[(index % 9) * 59 + index / 9] = glb_A[threadIdx.x + i * 64];
        }
        if (threadIdx.x < 19) {
          index = threadIdx.x + 512;
          shrRegion0[(index % 9) * 59 + index / 9] = glb_A[threadIdx.x + 512];
        }
      }

      float* shrRegion1 = &localShrMem0[531];
      // using ExtendedPatchLoader
      {
        if (threadIdx.x < 9) {
          shrRegion1[threadIdx.x + 0] = glb_B[threadIdx.x + 0];
        }
      }
      __syncthreads();
      if (threadIdx.x < 56) {
        float value;

        value = shrRegion0[threadIdx.x + 0 * 59];

        // Mul begin col 0
        reg0[1] += value * shrRegion1[0];
        // Mul end col 1

        value = shrRegion0[threadIdx.x + 1 * 59];

        // Mul begin col 1
        reg0[1] += value * shrRegion1[1];
        // Mul end col 1

        value = shrRegion0[threadIdx.x + 2 * 59];

        // Mul begin col 2
        reg0[1] += value * shrRegion1[2];
        // Mul end col 1

        value = shrRegion0[threadIdx.x + 3 * 59];

        // Mul begin col 3
        reg0[1] += value * shrRegion1[3];
        // Mul end col 1

        value = shrRegion0[threadIdx.x + 4 * 59];

        // Mul begin col 4
        reg0[1] += value * shrRegion1[4];
        // Mul end col 1

        value = shrRegion0[threadIdx.x + 5 * 59];

        // Mul begin col 5
        reg0[1] += value * shrRegion1[5];
        // Mul end col 1

        value = shrRegion0[threadIdx.x + 6 * 59];

        // Mul begin col 6
        reg0[1] += value * shrRegion1[6];
        // Mul end col 1

        value = shrRegion0[threadIdx.x + 7 * 59];

        // Mul begin col 7
        reg0[1] += value * shrRegion1[7];
        // Mul end col 1

        value = shrRegion0[threadIdx.x + 8 * 59];

        // Mul begin col 8
        reg0[1] += value * shrRegion1[8];
        // Mul end col 1

      }
      if (threadIdx.x < 56) {
        #pragma unroll
        for (int n = 0; n < 9; ++n) {
          glb_C[threadIdx.x + 56 * n] = reg0[n] + glb_C[threadIdx.x + 56 * n];
        }
      }
    }
  }
}


// Dense x Dense Kernel Launcher
void sgemm_T_NT_m9_n9_k56_lda9_ldb9_ldc56_alpha_1_beta_1_sss_0b51dc0(const float * A, int A_extraOffset, const float * B, int B_extraOffset, float * C, int C_extraOffset, unsigned numElements, unsigned* flags, void* streamPtr) {
  dim3 block(64, 1, 1);
  dim3 grid((numElements + 1 - 1) / 1, 1, 1);
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_sgemm_T_NT_m9_n9_k56_lda9_ldb9_ldc56_alpha_1_beta_1_sss_0b51dc0<<<grid,block,0,stream>>>(A, A_extraOffset, B, B_extraOffset, C, C_extraOffset, numElements, flags);
  CHECK_ERR;
}


// Dense x Sparse Kernel Launcher
void sgemm_T_NT_m9_n9_k56_lda9_ldb9_ldc56_alpha_1_beta_1_sss_b15c35b(const float * A, int A_extraOffset, const float * B, int B_extraOffset, float * C, int C_extraOffset, unsigned numElements, unsigned* flags, void* streamPtr) {
  dim3 block(64, 1, 1);
  dim3 grid((numElements + 1 - 1) / 1, 1, 1);
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_sgemm_T_NT_m9_n9_k56_lda9_ldb9_ldc56_alpha_1_beta_1_sss_b15c35b<<<grid,block,0,stream>>>(A, A_extraOffset, B, B_extraOffset, C, C_extraOffset, numElements, flags);
  CHECK_ERR;
}



int main(){
  float A[9*56] = {1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.};
  float B_sparse[9] = {4., 4., 4., 4., 4., 4., 4., 4., 4.};
  float B_dense[9 * 9] = {0., 0., 0., 0., 0., 0., 0., 0., 0., 4., 4., 4., 4., 4., 4., 4., 4., 4.,
 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
 0., 0., 0., 0., 0., 0., 0., 0., 0.};
  float C[56*9] = {0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1};
  float R1[56*9];
  float R2[56*9];

  float *A_dev = nullptr;
  float *B_sparse_dev = nullptr;
  float *B_dense_dev = nullptr;
  float *C1_dev = nullptr;
  float *C2_dev = nullptr;

  cudaMalloc((void **)&A_dev, sizeof(float) * 9 * 56); CHECK_ERR;
  cudaMalloc((void **)&B_sparse_dev, sizeof(float) * 9); CHECK_ERR;
  cudaMalloc((void **)&B_dense_dev, sizeof(float) * 9 * 9); CHECK_ERR;
  cudaMalloc((void **)&C1_dev, sizeof(float) * 56 * 9); CHECK_ERR;
  cudaMalloc((void **)&C2_dev, sizeof(float) * 56 * 9); CHECK_ERR;

  cudaMemcpy((void *)A_dev, (void *)A, sizeof(float) * 9 * 56, cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void *)B_sparse_dev, (void *)B_sparse, sizeof(float) *  9, cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void *)B_dense_dev, (void *)B_dense, sizeof(float) *  9 * 9, cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void *)C1_dev, (void *)C, sizeof(float) * 56 * 9, cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void *)C2_dev, (void *)C, sizeof(float) * 56 * 9, cudaMemcpyHostToDevice); CHECK_ERR;

  // Dense x Dense Matrix Mult
   sgemm_T_NT_m9_n9_k56_lda9_ldb9_ldc56_alpha_1_beta_1_sss_0b51dc0(A_dev, 0, B_dense_dev, 0, C1_dev, 0, 1, nullptr, nullptr);
  cudaDeviceSynchronize();
  cudaMemcpy(R1, C1_dev, sizeof(float)*56*9, cudaMemcpyDeviceToHost);

  // Dense x Sparse Matrix Mult
   sgemm_T_NT_m9_n9_k56_lda9_ldb9_ldc56_alpha_1_beta_1_sss_b15c35b(A_dev, 0, B_sparse_dev, 0, C2_dev, 0, 1, nullptr, nullptr);
  cudaDeviceSynchronize();
  cudaMemcpy(R2, C2_dev, sizeof(float)*56*9, cudaMemcpyDeviceToHost);

  std::cout << "[";
  for (int ii = 0; ii < 56*9 -1; ii++){
    std::cout << R1[ii] << ", ";
  }
  std::cout << R1[56*9 -1] << "]" << std::endl;
  std::cout << "[";
  for (int ii = 0; ii < 56*9 - 1; ii++){
    std::cout << R2[ii] << ", ";
  }
  std::cout << R2[56*9 -1] << "]" << std::endl;
  for (int i = 0; i < 56*9; i++){
    if (R1[i] != R2[i]) {
    throw std::runtime_error("Transposed Dense x  Dense and Transposed Dense x  Sparse Matrix Mismatch in Multiplication!");
    }
  }
  std::cout << "Transposed Dense x  Dense and Transposed Dense x  Sparse Matrix Multiplications Match!" << std::endl;
}
