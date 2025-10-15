#include <cstdio>
#include <vector>
#include <random>
#include <numeric>
#include "common.hpp"

__global__ void blockReduceSum(const float* __restrict__ data,
                               float* __restrict__ partial,
                               int N) {
  extern __shared__ float smem[]; // tamaño pasado en <<<... , ..., shmem>>>
  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x * 2 + threadIdx.x; // 2 elementos por hilo
  float x = 0.f;
  if (i < N) x += data[i];
  if (i + blockDim.x < N) x += data[i + blockDim.x];
  smem[tid] = x;
  __syncthreads();

  // Reducción en potencias de 2
  for (int s = blockDim.x/2; s > 0; s >>= 1) {
    if (tid < s) smem[tid] += smem[tid + s];
    __syncthreads();
  }

  if (tid == 0) partial[blockIdx.x] = smem[0];
}

int main(int argc, char** argv) {
  int N = (argc > 1) ? std::atoi(argv[1]) : (1<<24);
  size_t bytes = N * sizeof(float);
  std::vector<float> h(N);
  std::mt19937 rng(123);
  std::uniform_real_distribution<float> dist(0.f, 1.f);
  for (int i=0;i<N;++i) h[i]=dist(rng);

  float *d_in, *d_partial;
  CUDA_CHECK(cudaMalloc(&d_in, bytes));
  CUDA_CHECK(cudaMemcpy(d_in, h.data(), bytes, cudaMemcpyHostToDevice));

  int block = 256;
  int grid  = (N + block*2 - 1) / (block*2); // 2 elems por hilo
  CUDA_CHECK(cudaMalloc(&d_partial, grid * sizeof(float)));

  CudaTimer timer;
  timer.start();
  blockReduceSum<<<grid, block, block*sizeof(float)>>>(d_in, d_partial, N);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  float ms1 = timer.stop_ms();

  // Segunda fase en CPU (o lanzar más kernels hasta un solo valor)
  std::vector<float> h_partial(grid);
  CUDA_CHECK(cudaMemcpy(h_partial.data(), d_partial, grid*sizeof(float), cudaMemcpyDeviceToHost));
  float gpu_sum = std::accumulate(h_partial.begin(), h_partial.end(), 0.0f);

  // Comprobación con CPU
  double ref = std::accumulate(h.begin(), h.end(), 0.0);
  printf("GPU sum = %.6f | CPU sum = %.6f | err = %.6f\n",
         gpu_sum, (float)ref, fabs(gpu_sum - (float)ref));
  printf("Phase1 kernel time: %.3f ms (%d blocks)\n", ms1, grid);

  cudaFree(d_in); cudaFree(d_partial);
  return 0;
}
