# 2D Heat Equation Solver - HPC Performance Analysis

> Parallel implementation and performance analysis of a two-dimensional heat equation solver using Serial, OpenMP, MPI, Hybrid MPI+OpenMP, and CUDA.

## Project Overview

This project solves the 2D heat equation using the explicit FTCS finite difference method. It compares several HPC programming models:

- Serial CPU baseline
- OpenMP shared-memory parallelism
- MPI distributed-memory parallelism
- Hybrid MPI + OpenMP parallelism
- CUDA GPU parallelism

The comparison measures execution time, speedup, parallel efficiency, analytical RMSE, RMSE against the serial baseline for directly-run solvers, maximum temperature, and throughput in million grid updates per second.

## Method

The 2D heat equation is:

```text
dT/dt = alpha * (d2T/dx2 + d2T/dy2)
```

The FTCS update is:

```text
T_new[i,j] = T_old[i,j]
           + rx * (T_old[i+1,j] - 2*T_old[i,j] + T_old[i-1,j])
           + ry * (T_old[i,j+1] - 2*T_old[i,j] + T_old[i,j-1])
```

where `rx = alpha * dt / dx^2`, `ry = alpha * dt / dy^2`, and the stability condition is `rx + ry <= 0.5`.

## Project Structure

```text
hpc-proj/
  Serial/heat2D_serial.cpp       Serial FTCS solver
  Parallel/heat2D_omp.cpp        OpenMP solver
  MPI/heat2D_mpi.cpp             Pure MPI solver
  Hybrid/heat2D_hybrid.cpp       MPI + OpenMP solver
  cuda/heat2D_cuda.cu            CUDA solver
  Compare/compare.cpp            Benchmark and graph generator
```

## Build And Run

Serial:

```bash
g++ -O2 -o Serial/heat2d_serial Serial/heat2D_serial.cpp
./Serial/heat2d_serial
```

OpenMP:

```bash
g++ -O2 -fopenmp -o Parallel/heat2d_openmp Parallel/heat2D_omp.cpp
./Parallel/heat2d_openmp 4
```

MPI:

```bash
mpicxx -O2 -o MPI/heat2d_mpi MPI/heat2D_mpi.cpp
mpirun -np 4 ./MPI/heat2d_mpi 500 500
```

Hybrid MPI + OpenMP:

```bash
mpicxx -O2 -fopenmp -o Hybrid/heat2d_hybrid Hybrid/heat2D_hybrid.cpp
mpirun -np 4 ./Hybrid/heat2d_hybrid 2 500 500
```

CUDA:

using the x64 Native tools command prompt
change directory to project folder /cuda

```bash
nvcc -O2 heat2D_cuda.cu -o heat2d_cuda
heat2d_cuda.exe
```

Comparison:

```bash
g++ -O2 -fopenmp -o Compare/compare Compare/compare.cpp
./Compare/compare
./Compare/compare 1000 1000
```

Run MPI, Hybrid, and CUDA first if you want their latest summary CSVs included in the comparison table. The comparison driver runs Serial and OpenMP directly, then loads:

- `MPI/summary_2d_mpi.csv`
- `Hybrid/summary_2d_hybrid.csv`
- `cuda/summary_2d_cuda.csv`

Summary files include grid dimensions, so mismatched summaries are skipped when the comparison grid is different.

## Outputs

- `Serial/results_2d_seq.csv`
- `Serial/summary_2d_seq.csv`
- `Parallel/results_2d_omp.csv`
- `MPI/results_2d_mpi.csv`
- `MPI/summary_2d_mpi.csv`
- `Hybrid/results_2d_hybrid.csv`
- `Hybrid/summary_2d_hybrid.csv`
- `cuda/results_2d_cuda.csv`
- `cuda/summary_2d_cuda.csv`
- `Compare/comparison_results.csv`
- `Compare/comparison_results.js`
- `Compare/index.html`
- `Compare/execution_time.png`
- `Compare/speedup.png`
- `Compare/efficiency.png`
- `Compare/rmse.png`
- `Compare/throughput.png`

Open `Compare/index.html` in a browser to view the comparison dashboard. Re-run
`Compare/compare` to refresh the dashboard data and chart images.

## Notes

The recommended CUDA executable name is `cuda/heat2d_cuda`.
