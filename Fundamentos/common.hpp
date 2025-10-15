#pragma once
#include <cstdio>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do {                                        \
  cudaError_t err = (call);                                          \
  if (err != cudaSuccess) {                                          \
    std::fprintf(stderr, "CUDA error %s:%d: %s\n",                   \
                 __FILE__, __LINE__, cudaGetErrorString(err));       \
    std::exit(EXIT_FAILURE);                                         \
  }                                                                  \
} while(0)

struct CudaTimer {
  cudaEvent_t start_, stop_;
  CudaTimer() { CUDA_CHECK(cudaEventCreate(&start_)); CUDA_CHECK(cudaEventCreate(&stop_)); }
  ~CudaTimer(){ cudaEventDestroy(start_); cudaEventDestroy(stop_); }
  void start() { CUDA_CHECK(cudaEventRecord(start_)); }
  float stop_ms(){ CUDA_CHECK(cudaEventRecord(stop_)); CUDA_CHECK(cudaEventSynchronize(stop_));
                   float ms=0; CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_)); return ms; }
};
