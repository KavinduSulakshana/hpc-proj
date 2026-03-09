# 2D Heat Equation Solver — Serial vs OpenMP Performance Analysis

> **Parallel Implementation and Performance Analysis of a Two-Dimensional Heat Equation Solver**  
> High Performance Computing (HPC) Project | OpenMP Shared Memory Parallelism
---

## Project Overview

This project implements and compares **Serial** and **OpenMP parallel** solvers for the 2D Heat Equation using the explicit Finite Difference Time Domain (FTCS) method on a 500×500 grid.

The system measures:
- **Execution time** across different thread counts
- **Speedup** vs serial baseline
- **Parallel efficiency** percentage
- **RMSE accuracy** vs analytical solution
- **Throughput** in Million Grid Points/second

---

## Mathematical Background

The 2D Heat Equation:

```
∂T/∂t = α (∂²T/∂x² + ∂²T/∂y²)
```

**Discretized using FTCS scheme:**
```
T[i,j]^(n+1) = T[i,j]^n
              + rx * (T[i+1,j] - 2*T[i,j] + T[i-1,j])
              + ry * (T[i,j+1] - 2*T[i,j] + T[i,j-1])
```

Where:
- `rx = α * dt / dx²`
- `ry = α * dt / dy²`
- **CFL Stability Condition:** `rx + ry ≤ 0.5`

**Analytical Solution:**
```
T(x,y,t) = 100 * exp(-α*π²*(1/Lx² + 1/Ly²)*t) * sin(π*x/Lx) * sin(π*y/Ly)
```

---

## Project Structure

```
hpc-heat-equation/
│
├── Serial/
│   ├── heat2D_serial.cpp      # Serial FTCS solver
│   └── results_2d_seq.csv         
│
├── Parallel/
│   ├── heat2D_omp.cpp       # OpenMP parallel solver
│   └── results_2d_omp.csv
│
├── Compare/
│   ├── compare.cpp            # Serial vs OpenMP benchmark
│   ├── comparison_results.csv # Benchmark results (generated)
│   ├── execution_time.png     # Time comparison graph (generated)
│   ├── speedup.png            # Speedup graph (generated)
│   ├── efficiency.png         # Efficiency graph (generated)
│   ├── rmse.png               # RMSE accuracy graph (generated)
│   └── throughput.png         # Throughput graph (generated)
│
└── README.md
```

---

## Simulation Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `LX`, `LY` | 1.0 m | Domain size |
| `NX`, `NY` | 500 | Grid points per axis |
| `ALPHA` | 0.01 m²/s | Thermal diffusivity |
| `T_FINAL` | 0.5 s | Simulation end time |
| `DX`, `DY` | 1/(NX-1) | Spatial step size |
| `DT` | Auto (CFL) | Time step (stability enforced) |
| `rx + ry` | ≤ 0.5 | CFL stability condition |

---

## Requirements

| Tool | Version | Purpose |
|------|---------|---------|
| GCC | 9+ | C++ compiler |
| OpenMP | 4.5+ | Parallel directives |
| Python 3 | 3.x | Graph generation |
| matplotlib | Latest | Plot generation |

### Install Dependencies
```bash
# Ubuntu/WSL
sudo apt update
sudo apt install g++ python3 python3-pip -y
pip3 install matplotlib
```

---

## Compilation & Running

### Serial Solver

```bash
# Compile
g++ -o Serial/heat2d_serial Serial/heat2D_serial.cpp

# Run
.Serial/heat2d_serial
```

**Expected output:**
```
========================================
  Sequential 2D Heat Equation Solver
========================================

Parameters:
  Plate size      = 1 x 1 m
  Grid points     = 500 x 500 (250000 total)
  Spatial step dx = X.XXXXXXXX m
  Spatial step dy = X.XXXXXXXX m
  Time step dt    = X.XXXXXe-XX s
  Diffusivity α   = X.XX m²/s
  Diffusion rx    = X.X
  Diffusion ry    = X.X
  rx + ry         = X.X (must be <= 0.5)
  Final time      = X.X s

Running XXXX time steps...

========================================
  RESULTS
========================================
  Final time:       X.XXXXXX s
  Execution time:   XXXXX.XXXXXX ms
  RMSE Error:       X.XXXXXXe-XX
  Max temperature:  XX.XXXXXX °C

Results saved to Serial/results_2d_seq.csv
```

---

### OpenMP Parallel Solver

```bash
# Compile
g++ -fopenmp -o Parallel/heat2d_openmp Parallel/heat2D_omp.cpp

# Run with default threads (max available)
.Parallel/heat2d_openmp

# Run with specific thread count
.Parallel/heat2d_openmp 4    # 4 threads
.Parallel/heat2d_openmp 8    # 8 threads
```

**Expected output:**
```
========================================
  OpenMP 2D Heat Equation Solver
========================================
Parameters:
  OpenMP threads  = 4

Running XXXX time steps...

========================================
  RESULTS
========================================
  Execution time:   XXXX ms
  L2 Error (RMS):   X.XXXXe-XX
  Max temperature:  X.XXXX °C
  Threads used:     4

Results saved to results_2d_omp.csv
```

---

### Comparison Benchmark

```bash
# Compile
g++ -fopenmp -o Compare/compare Compare/compare.cpp

# Run full benchmark
.Compare/compare
```

**This automatically runs:**
- Serial solver (baseline)
- OpenMP with 1 thread
- OpenMP with 2 threads
- OpenMP with 4 threads
- OpenMP with max available threads

**Expected output:**
```
============================================================================================================
  2D HEAT EQUATION SOLVER -- SERIAL vs OpenMP COMPARISON
  Grid: 500x500  | alpha=0.01  | dt=2.5000e-05  | T_final=0.5  | Steps=XXXX
  CFL check: rx+ry = 0.4000  (must be <= 0.5)
============================================================================================================
Solver          Threads   Time (ms)     Speedup     Efficiency%   RMSE             MaxTemp(C)    MPoints/s
------------------------------------------------------------------------------------------------------------
Serial          1         XXXX.XX       1.000       100.00        X.XXXXe-XX       X.XXXX        XXX.XX
OpenMP-1T       1         XXXX.XX       X.XXX       XX.XX         X.XXXXe-XX       X.XXXX        XXX.XX
OpenMP-2T       2         XXXX.XX       X.XXX       XX.XX         X.XXXXe-XX       X.XXXX        XXX.XX
OpenMP-4T       4         XXXX.XX       X.XXX       XX.XX         X.XXXXe-XX       X.XXXX        XXX.XX
============================================================================================================

Graphs saved:
  execution_time.png
  speedup.png
  efficiency.png
  rmse.png
  throughput.png
```

---

## Output Files

### CSV Results

| File | Contents |
|------|----------|
| `results_2d_seq.csv` | x, y, T_numerical, T_analytical (serial) |
| `results_2d_omp.csv` | x, y, T_numerical, T_analytical (OpenMP) |
| `Compare/comparison_results.csv` | Solver, threads, time, speedup, efficiency, RMSE |

### Generated Graphs

| Graph | Description |
|-------|-------------|
| `execution_time.png` | Bar chart — time per solver |
| `speedup.png` | Measured vs ideal speedup |
| `efficiency.png` | Parallel efficiency % vs threads |
| `rmse.png` | RMSE accuracy comparison |
| `throughput.png` | Million grid updates per second |

---

## OpenMP Parallelization Strategy

### Main Computation Loop
```cpp
#pragma omp parallel for collapse(2) schedule(static)
for (int i = 1; i < NX-1; i++)
    for (int j = 1; j < NY-1; j++) {
        // FTCS stencil update — no data dependency between iterations
        T_new[idx(i,j)] = T_old[idx(i,j)] + RX*d2x + RY*d2y;
    }
```

### RMSE Calculation
```cpp
#pragma omp parallel for collapse(2) reduction(+:err)
for (int i = 0; i < NX; i++)
    for (int j = 0; j < NY; j++) {
        err += diff * diff;  // reduction prevents race condition
    }
```

### Why `collapse(2)`?
- Merges two nested loops into one parallel region
- Increases total work per thread
- Better load balancing for large 2D grids

### Why `schedule(static)`?
- Grid updates are uniform cost per cell
- Static scheduling minimizes overhead
- No dynamic load imbalance in FTCS stencil

---

## Accuracy Verification

The RMSE between numerical and analytical solutions is computed at every run:

```
RMSE = sqrt( Σ(T_numerical - T_analytical)² / (NX × NY) )
```

All OpenMP versions should produce **identical RMSE** to serial — any difference is only floating-point rounding order, not a correctness issue.

| Solver | RMSE | Match Serial? |
|--------|------|---------------|
| Serial | X.XXe-XX | Baseline |
| OpenMP-1T | X.XXe-XX | YES |
| OpenMP-2T | X.XXe-XX | YES (fp rounding) |
| OpenMP-4T | X.XXe-XX | YES (fp rounding) |

---

## Expected Performance Results

| Solver | Expected Speedup | Expected Efficiency |
|--------|-----------------|---------------------|
| Serial | 1.0x | 100% |
| OpenMP-1T | ~1.0x | ~100% |
| OpenMP-2T | ~1.7-1.9x | ~85-95% |
| OpenMP-4T | ~3.0-3.5x | ~75-87% |
| OpenMP-8T | ~4.5-6.0x | ~56-75% |

> Note: Actual results depend on CPU architecture and available cores.

---

## Upcoming Implementations

- [ ] MPI — Distributed memory solver
- [ ] Hybrid MPI + OpenMP — Combined parallelism
- [ ] CUDA — GPU acceleration
- [ ] Performance comparison across all implementations

## License

This project is submitted for academic purposes only.
