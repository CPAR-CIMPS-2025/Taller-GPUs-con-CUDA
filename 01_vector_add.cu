#include <cstdio>
#include <vector>
#include <random>
#include "common.hpp"

__global__ void vecAdd(const float* __restrict__ A,
                       const float* __restrict__ B,
                       float* __restrict__ C,
                       int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) C[i] = A[i] + B[i];
}

int main(int argc, char** argv) {
  int N = (argc > 1) ? std::atoi(argv[1]) : (1<<24); // ~16M elementos
  size_t bytes = N * sizeof(float);
  printf("N = %d (%.2f MB por vector)\n", N, bytes / (1024.0*1024.0));

  // Host
  std::vector<float> hA(N), hB(N), hC(N);
  std::mt19937 rng(123);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);
  for (int i=0;i<N;++i){ hA[i]=dist(rng); hB[i]=dist(rng); }

  // Device
  float *dA, *dB, *dC;
  CUDA_CHECK(cudaMalloc(&dA, bytes));
  CUDA_CHECK(cudaMalloc(&dB, bytes));
  CUDA_CHECK(cudaMalloc(&dC, bytes));

  CUDA_CHECK(cudaMemcpy(dA, hA.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB.data(), bytes, cudaMemcpyHostToDevice));

  // Configuración de ejecución
  int block = 256;                           // múltiplo de 32 (warp)
  int grid  = (N + block - 1) / block;

  CudaTimer timer;
  timer.start();
  vecAdd<<<grid, block>>>(dA, dB, dC, N);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  float ms = timer.stop_ms();

  CUDA_CHECK(cudaMemcpy(hC.data(), dC, bytes, cudaMemcpyDeviceToHost));

  // Verificación simple
  for (int i=0;i<5;++i) {
    printf("C[%d] = %f (A+B=%f)\n", i, hC[i], hA[i]+hB[i]);
  }

  double gb = 3.0 * bytes / 1e9; // A,B leído + C escrito
  double bw = gb / (ms/1000.0);
  printf("Kernel time: %.3f ms, BW efectivo: %.2f GB/s\n", ms, bw);

  cudaFree(dA); cudaFree(dB); cudaFree(dC);
  return 0;
}
