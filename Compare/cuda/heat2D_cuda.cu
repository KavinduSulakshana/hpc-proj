/**
 * CUDA Parallel Heat Equation Solver
 * 2D Plate Temperature Evolution Simulation
 *
 * Physics: dT/dt = alpha * (d²T/dx² + d²T/dy²)  (2D Heat Equation)
 * Method:  FTCS (Forward-Time Central-Space) Finite Difference
 * Parallelization: 2D GPU kernel - one thread per grid point (i, j)
 *
 * Boundary: T = 0 on all edges
 * Initial:  T(x,y,0) = 100 * sin(pi*x) * sin(pi*y)
 * Analytical: T(x,y,t) = 100 * exp(-2*alpha*pi²*t) * sin(pi*x) * sin(pi*y)
 */

#include <iostream>
#include <cmath>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <cuda_runtime.h>

// ============== SIMULATION PARAMETERS ==============
#define LX      1.0         // Plate width  (meters)
#define LY      1.0         // Plate height (meters)
#define ALPHA   0.01        // Thermal diffusivity (m²/s)
#define NX      500         // Grid points in x
#define NY      500         // Grid points in y
#define T_FINAL 0.5         // Simulation time (seconds)

#define DX (LX / (NX - 1))
#define DY (LY / (NY - 1))

// 2D CFL: dt = 0.4 * 0.5 * (dx²*dy²) / (alpha*(dx²+dy²))
#define DT (0.4 * 0.5 * (DX*DX*DY*DY) / (ALPHA * (DX*DX + DY*DY)))

#define RX (ALPHA * DT / (DX * DX))   // Diffusion number in x
#define RY (ALPHA * DT / (DY * DY))   // Diffusion number in y

#define PI 3.14159265358979323846

// 2D thread block: 16x16 = 256 threads per block
#define BLOCK_X 16
#define BLOCK_Y 16

// Row-major flat index
#define IDX(i, j) ((i) * NY + (j))

// ============== CUDA ERROR CHECKING ==============
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while(0)

// ============== CUDA KERNELS ==============

/**
 * Initialization kernel
 * Each thread sets T(x,y) = 100 * sin(pi*x) * sin(pi*y)
 * Boundaries are explicitly set to 0
 */
__global__ void initialize_kernel(double* T, int nx, int ny, double dx, double dy) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // x-index
    int j = blockIdx.y * blockDim.y + threadIdx.y;  // y-index

    if (i >= nx || j >= ny) return;

    double x = i * dx;
    double y = j * dy;

    // Interior: sinusoidal initial condition
    T[IDX(i, j)] = 100.0 * sin(PI * x) * sin(PI * y);

    // Boundary conditions: all edges = 0
    if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1) {
        T[IDX(i, j)] = 0.0;
    }
}

/**
 * FTCS time-step kernel (2D 5-point stencil)
 * Each thread computes one interior point:
 *
 *          T[i][j+1]
 *              |
 * T[i-1][j] - T[i][j] - T[i+1][j]
 *              |
 *          T[i][j-1]
 */
__global__ void heat_step_kernel(const double* __restrict__ T_old,
                                        double* __restrict__ T_new,
                                        int nx, int ny,
                                        double rx, double ry) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= nx || j >= ny) return;

    // Boundary conditions: edges stay zero
    if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1) {
        T_new[IDX(i, j)] = 0.0;
        return;
    }

    // 2D FTCS stencil
    T_new[IDX(i, j)] = T_old[IDX(i, j)]
        + rx * (T_old[IDX(i+1, j)] - 2.0 * T_old[IDX(i, j)] + T_old[IDX(i-1, j)])  // x-direction
        + ry * (T_old[IDX(i, j+1)] - 2.0 * T_old[IDX(i, j)] + T_old[IDX(i, j-1)]); // y-direction
}

// ============== HOST FUNCTIONS ==============

double analytical_solution(double x, double y, double t) {
    return 100.0 * exp(-2.0 * ALPHA * PI * PI * t) * sin(PI * x) * sin(PI * y);
}

double calculate_error(const double* T, double t, int nx, int ny) {
    double error = 0.0;
    for (int i = 0; i < nx; i++) {
        double x = i * DX;
        for (int j = 0; j < ny; j++) {
            double y = j * DY;
            double diff = T[IDX(i, j)] - analytical_solution(x, y, t);
            error += diff * diff;
        }
    }
    return sqrt(error / (nx * ny));
}

void save_results(const double* T, double t, int nx, int ny, const char* filename) {
    std::ofstream file(filename);
    file << "# x, y, T_numerical, T_analytical\n";
    int step = std::max(1, nx / 100);  // Sample to keep file manageable
    for (int i = 0; i < nx; i += step) {
        double x = i * DX;
        for (int j = 0; j < ny; j += step) {
            double y = j * DY;
            file << x << ", " << y << ", "
                 << T[IDX(i, j)] << ", "
                 << analytical_solution(x, y, t) << "\n";
        }
    }
    file.close();
}

// ============== MAIN ==============
int main() {
    std::cout << "========================================\n";
    std::cout << "  CUDA 2D Heat Equation Solver\n";
    std::cout << "========================================\n\n";

    // Get GPU info
    int deviceId;
    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDevice(&deviceId));
    CUDA_CHECK(cudaGetDeviceProperties(&props, deviceId));
    std::cout << "GPU: " << props.name << "\n\n";

    // Compute values for printing
    double dt  = DT;
    double rx  = RX;
    double ry  = RY;

    std::cout << "Parameters:\n";
    std::cout << "  Plate size      = " << LX << " x " << LY << " m\n";
    std::cout << "  Grid points     = " << NX << " x " << NY << " (" << NX * NY << " total)\n";
    std::cout << "  Spatial step dx = " << DX << " m\n";
    std::cout << "  Spatial step dy = " << DY << " m\n";
    std::cout << "  Time step dt    = " << dt  << " s\n";
    std::cout << "  Diffusivity a   = " << ALPHA << " m^2/s\n";
    std::cout << "  Diffusion rx    = " << rx << "\n";
    std::cout << "  Diffusion ry    = " << ry << "\n";
    std::cout << "  rx + ry         = " << (rx + ry) << " (must be <= 0.5)\n";
    std::cout << "  Block size      = " << BLOCK_X << " x " << BLOCK_Y << "\n";
    std::cout << "  Final time      = " << T_FINAL << " s\n\n";

    // Stability check
    if ((rx + ry) > 0.5) {
        std::cerr << "ERROR: Unstable! rx + ry = " << (rx + ry) << " > 0.5\n";
        return 1;
    }

    // ---- Memory allocation ----
    size_t size = NX * NY * sizeof(double);

    double* h_T = new double[NX * NY];  // Host array

    double *d_T_old, *d_T_new;
    CUDA_CHECK(cudaMalloc(&d_T_old, size));
    CUDA_CHECK(cudaMalloc(&d_T_new, size));

    // ---- 2D grid/block dimensions ----
    dim3 blockDim(BLOCK_X, BLOCK_Y);                                          // 16x16 threads per block
    dim3 gridDim((NX + BLOCK_X - 1) / BLOCK_X, (NY + BLOCK_Y - 1) / BLOCK_Y); // enough blocks to cover grid

    // ---- Initialize on GPU ----
    initialize_kernel<<<gridDim, blockDim>>>(d_T_old, NX, NY, DX, DY);
    CUDA_CHECK(cudaDeviceSynchronize());

    int num_steps = static_cast<int>(T_FINAL / DT);
    std::cout << "Running " << num_steps << " time steps on GPU...\n\n";

    // ---- CUDA events for GPU timing ----
    cudaEvent_t start_ev, stop_ev;
    CUDA_CHECK(cudaEventCreate(&start_ev));
    CUDA_CHECK(cudaEventCreate(&stop_ev));

    CUDA_CHECK(cudaEventRecord(start_ev));

    // ---- Time-stepping loop ----
    double t = 0.0;
    for (int step = 0; step < num_steps; step++) {
        heat_step_kernel<<<gridDim, blockDim>>>(d_T_old, d_T_new, NX, NY, RX, RY);

        // Swap device pointers (no data copy, just pointer swap)
        double* tmp = d_T_old;
        d_T_old = d_T_new;
        d_T_new = tmp;

        t += DT;
    }

    CUDA_CHECK(cudaEventRecord(stop_ev));
    CUDA_CHECK(cudaEventSynchronize(stop_ev));

    float elapsed_ms;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start_ev, stop_ev));

    // ---- Copy results back to host ----
    CUDA_CHECK(cudaMemcpy(h_T, d_T_old, size, cudaMemcpyDeviceToHost));

    // ---- Post-processing on CPU ----
    double error = calculate_error(h_T, t, NX, NY);

    double max_temp = 0.0;
    for (int k = 0; k < NX * NY; k++) {
        if (h_T[k] > max_temp) max_temp = h_T[k];
    }

    // ---- Print results ----
    std::cout << "========================================\n";
    std::cout << "  RESULTS\n";
    std::cout << "========================================\n";
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "  Final time:       " << t          << " s\n";
    std::cout << "  GPU time:         " << elapsed_ms << " ms\n";
    std::cout << "  RMSE Error:       " << std::scientific << error << "\n";
    std::cout << "  Max temperature:  " << std::fixed << max_temp << " C\n";

    save_results(h_T, t, NX, NY, "results_2d_cuda.csv");
    std::cout << "\nResults saved to results_2d_cuda.csv\n";

    // ---- Cleanup ----
    delete[] h_T;
    CUDA_CHECK(cudaFree(d_T_old));
    CUDA_CHECK(cudaFree(d_T_new));
    CUDA_CHECK(cudaEventDestroy(start_ev));
    CUDA_CHECK(cudaEventDestroy(stop_ev));

    return 0;
}
