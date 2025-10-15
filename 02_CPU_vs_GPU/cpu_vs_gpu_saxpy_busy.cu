// cpu_vs_gpu_saxpy_busy.cu
// Comparación CPU (multi-hilo) vs GPU (SAXPY) + kernel "busy" con duración ajustable.
// Compilar: nvcc -O3 -std=c++17 cpu_vs_gpu_saxpy_busy.cu -o compare
// Ejecutar (ejemplos):
//   ./compare                 # valores por defecto
//   ./compare --N 33554432 --repeats 50 --threads 8 --busy_ms 5000

#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <thread>
#include <random>
#include <chrono>
#include <cstring>
#include <atomic>
#include <cmath>

#define CUDA_CHECK(x) do { \
  cudaError_t err = (x); \
  if (err != cudaSuccess) { \
    std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    std::exit(EXIT_FAILURE); \
  } \
} while(0)

__global__ void saxpy_kernel(int n, float a, const float* __restrict__ x, float* __restrict__ y) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = a * x[i] + y[i];
}

// Kernel ocupado (flops en bucle) por un tiempo objetivo en clocks de SM
__global__ void busy_kernel(unsigned long long target_clocks) {
  unsigned long long start = clock64();
  float x = threadIdx.x * 0.001f + 1.0f;
  float y = blockIdx.x   * 0.002f + 2.0f;
  while ((clock64() - start) < target_clocks) {
    // Algunas operaciones para mantener ALUs ocupadas
    x = fmaf(x, 1.0001f, 0.0001f);
    y = fmaf(y, 0.9999f, 0.0002f);
    // Evitar ser optimizado
    if (x > 1e10f) x = 1.0f;
    if (y > 1e10f) y = 2.0f;
  }
}

// SAXPY CPU multi-hilo
void saxpy_cpu_range(int start, int end, float a, const float* x, float* y) {
  for (int i = start; i < end; ++i) {
    y[i] = a * x[i] + y[i];
  }
}

// Cronómetro
struct Timer {
  std::chrono::high_resolution_clock::time_point t0;
  void tic() { t0 = std::chrono::high_resolution_clock::now(); }
  double toc_ms() const {
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
  }
};

// Parseo muy simple de argumentos
struct Args {
  int N = 1<<24;          // elementos (~16M)
  int repeats = 20;       // número de veces que se repite SAXPY
  int threads = std::thread::hardware_concurrency() ? std::thread::hardware_concurrency() : 8;
  int busy_ms = 3000;     // duración del kernel busy para mostrar utilización GPU
};
Args parse_args(int argc, char** argv){
  Args a;
  for (int i=1;i<argc;++i){
    if (std::strcmp(argv[i],"--N")==0 && i+1<argc) a.N = std::atoi(argv[++i]);
    else if (std::strcmp(argv[i],"--repeats")==0 && i+1<argc) a.repeats = std::atoi(argv[++i]);
    else if (std::strcmp(argv[i],"--threads")==0 && i+1<argc) a.threads = std::atoi(argv[++i]);
    else if (std::strcmp(argv[i],"--busy_ms")==0 && i+1<argc) a.busy_ms = std::atoi(argv[++i]);
    else if (std::strcmp(argv[i],"--help")==0) {
      std::printf("Uso: %s [--N int] [--repeats int] [--threads int] [--busy_ms int]\n", argv[0]);
      std::exit(0);
    }
  }
  return a;
}

int main(int argc, char** argv) {
  Args args = parse_args(argc, argv);
  std::printf("Config: N=%d, repeats=%d, cpu_threads=%d, busy_ms=%d\n",
              args.N, args.repeats, args.threads, args.busy_ms);

  // Datos host
  std::vector<float> hx(args.N), hy(args.N), hy_ref(args.N);
  std::mt19937 rng(123);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);
  for (int i=0;i<args.N;++i){ hx[i]=dist(rng); hy[i]=dist(rng); }
  hy_ref = hy; // copia para CPU

  const float a = 2.5f;

  // -------- CPU SAXPY (multi-hilo) --------
  Timer tcpu;
  tcpu.tic();
  for (int r=0; r<args.repeats; ++r) {
    std::vector<std::thread> pool;
    int chunk = (args.N + args.threads - 1) / args.threads;
    for (int t=0; t<args.threads; ++t){
      int s = t*chunk;
      int e = std::min(args.N, s+chunk);
      if (s < e) pool.emplace_back(saxpy_cpu_range, s, e, a, hx.data(), hy_ref.data());
    }
    for (auto& th : pool) th.join();
  }
  double cpu_ms = tcpu.toc_ms();
  std::printf("[CPU] SAXPY %d repeticiones: %.2f ms\n", args.repeats, cpu_ms);

  // -------- GPU SAXPY --------
  float *dx=nullptr, *dy=nullptr;
  size_t bytes = size_t(args.N) * sizeof(float);
  CUDA_CHECK(cudaMalloc(&dx, bytes));
  CUDA_CHECK(cudaMalloc(&dy, bytes));
  CUDA_CHECK(cudaMemcpy(dx, hx.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dy, hy.data(), bytes, cudaMemcpyHostToDevice));

  int block = 256;
  int grid  = (args.N + block - 1) / block;

  // Warm-up
  saxpy_kernel<<<grid, block>>>(args.N, a, dx, dy);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t evs, eve;
  CUDA_CHECK(cudaEventCreate(&evs));
  CUDA_CHECK(cudaEventCreate(&eve));
  CUDA_CHECK(cudaEventRecord(evs));
  for (int r=0; r<args.repeats; ++r) {
    saxpy_kernel<<<grid, block>>>(args.N, a, dx, dy);
  }
  CUDA_CHECK(cudaEventRecord(eve));
  CUDA_CHECK(cudaEventSynchronize(eve));
  float gpu_ms = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, evs, eve));
  std::printf("[GPU] SAXPY %d repeticiones: %.2f ms\n", args.repeats, gpu_ms);

  // Validación ligera (muestra 3 elementos)
  std::vector<float> hy_out(args.N);
  CUDA_CHECK(cudaMemcpy(hy_out.data(), dy, bytes, cudaMemcpyDeviceToHost));
  for (int i=0;i<3;++i) {
    std::printf("check y[%d]: cpu=%.5f gpu=%.5f (diff=%.5e)\n",
                i, hy_ref[i], hy_out[i], std::abs(double(hy_ref[i]) - double(hy_out[i])));
  }

  // Speedup (tiempo CPU / tiempo GPU)
  double speedup = cpu_ms / gpu_ms;
  std::printf("Speedup aproximado CPU/GPU: %.2fx\n", speedup);

  // -------- Kernel ocupado para mostrar utilización (ajustable) --------
  cudaDeviceProp prop; CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  // clockRate en kHz -> clocks por ms = clockRate
  unsigned long long target_clocks = (unsigned long long)( (unsigned long long)args.busy_ms * (unsigned long long)prop.clockRate );
  // Lanzamos muchos bloques para ocupar la GPU (32x SM es un buen punto de partida)
  int busy_blocks = prop.multiProcessorCount * 32;
  int busy_threads = 256;
  std::printf("[GPU] Lanzando busy_kernel por ~%d ms (SMs=%d, blocks=%d, threads=%d)\n",
              args.busy_ms, prop.multiProcessorCount, busy_blocks, busy_threads);

  CUDA_CHECK(cudaEventRecord(evs));
  busy_kernel<<<busy_blocks, busy_threads>>>(target_clocks);
  CUDA_CHECK(cudaEventRecord(eve));
  CUDA_CHECK(cudaEventSynchronize(eve));
  float busy_elapsed = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&busy_elapsed, evs, eve));
  std::printf("[GPU] busy_kernel completado en %.2f ms\n", busy_elapsed);

  CUDA_CHECK(cudaEventDestroy(evs));
  CUDA_CHECK(cudaEventDestroy(eve));
  cudaFree(dx); cudaFree(dy);

  std::puts("\nConsejos:");
  std::puts(" - Usa `watch -n 0.5 nvidia-smi` en otra terminal para observar la utilización mientras corre busy_kernel.");
  std::puts(" - Ajusta --N, --repeats para controlar trabajo de SAXPY; ajusta --busy_ms para mantener la GPU ocupada.");
  return 0;
}
