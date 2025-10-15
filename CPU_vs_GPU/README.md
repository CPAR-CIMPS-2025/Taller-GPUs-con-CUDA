# README

## Compilación

```
nvcc -O3 -std=c++17 cpu_vs_gpu_saxpy_busy.cu -o compare
```

# Ejecución (por defecto: ~16M elementos, 20 repeticiones, ocupación ~3 s)

```
./compare
```

# Ejecución (más intenso: 33.5M elems, 50 repeticiones, 8 hilos CPU, ocupación 5 s)

```
./compare --N 33554432 --repeats 50 --threads 8 --busy_ms 5000
```

# Monitoreo de la utilización del GPU (en paralelo, es decir en otra terminal)

```
watch -n 0.5 nvidia-smi
```
