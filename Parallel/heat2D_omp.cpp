/**
 * OpenMP Parallel Heat Equation Solver
 */

#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <omp.h>

// ============== SIMULATION PARAMETERS ==============
const double LX    = 1.0;      // Plate width  (meters)
const double LY    = 1.0;      // Plate height (meters)
const double ALPHA = 0.01;     // Thermal diffusivity (m²/s)
const int    NX    = 500;      // Grid points in x
const int    NY    = 500;      // Grid points in y
const double T_FINAL = 0.5;   // Simulation time (seconds)

const double DX = LX / (NX - 1);
const double DY = LY / (NY - 1);

const double DT = 0.4 * 0.5 * (DX*DX * DY*DY) / (ALPHA * (DX*DX + DY*DY));

// Diffusion numbers
const double RX = ALPHA * DT / (DX * DX);
const double RY = ALPHA * DT / (DY * DY);

const double PI = 3.14159265358979323846;

// ============== FLAT INDEX HELPER ==============
// Row-major: T[i*NY + j]  where i = x-index, j = y-index
inline int idx(int i, int j) { return i * NY + j; }

// ============== FUNCTIONS ==============

void initialize(std::vector<double>& T) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            double x = i * DX;
            double y = j * DY;
            T[idx(i,j)] = 100.0 * sin(PI * x / LX) * sin(PI * y / LY);
        }
    }

    #pragma omp parallel for
    for (int j = 0; j < NY; j++) {
        T[idx(0,    j)] = 0.0;   // left edge  (x = 0)
        T[idx(NX-1, j)] = 0.0;   // right edge  (x = Lx)
    }
    #pragma omp parallel for
    for (int i = 0; i < NX; i++) {
        T[idx(i, 0   )] = 0.0;   // bottom edge (y = 0)
        T[idx(i, NY-1)] = 0.0;   // top edge (y = Ly)
    }
}

double analytical_solution(double x, double y, double t) {
    double decay = -ALPHA * PI * PI * (1.0/(LX*LX) + 1.0/(LY*LY));
    return 100.0 * exp(decay * t) * sin(PI * x / LX) * sin(PI * y / LY);
}

double calculate_error(const std::vector<double>& T, double t) {
    double error = 0.0;
    #pragma omp parallel for collapse(2) reduction(+:error)
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            double x    = i * DX;
            double y    = j * DY;
            double diff = T[idx(i,j)] - analytical_solution(x, y, t);
            error += diff * diff;
        }
    }
    return sqrt(error / (NX * NY));
}

void save_results(const std::vector<double>& T, double t,
                  const std::string& filename, int stride = 5) {
    std::ofstream file(filename);
    file << "# x, y, T_numerical, T_analytical\n";
    file << std::fixed << std::setprecision(6);
    for (int i = 0; i < NX; i += stride) {
        for (int j = 0; j < NY; j += stride) {
            double x = i * DX;
            double y = j * DY;
            file << x << ", " << y << ", "
                 << T[idx(i,j)] << ", "
                 << analytical_solution(x, y, t) << "\n";
        }
    }
    file.close();
}

// ============== MAIN SOLVER ==============
int main(int argc, char* argv[]) {
    int num_threads = omp_get_max_threads();
    if (argc > 1) num_threads = atoi(argv[1]);
    omp_set_num_threads(num_threads);

    std::cout << "========================================\n";
    std::cout << "  OpenMP 2D Heat Equation Solver\n";
    std::cout << "========================================\n\n";

    std::cout << "Parameters:\n";
    std::cout << "  Domain          = " << LX << " m  x  " << LY << " m\n";
    std::cout << "  Grid            = " << NX << " x " << NY
              << "  (" << (long long)NX*NY << " points)\n";
    std::cout << "  Spatial steps   = dx=" << DX << "  dy=" << DY << " m\n";
    std::cout << "  Time step dt    = " << DT << " s\n";
    std::cout << "  Diffusivity α   = " << ALPHA << " m²/s\n";
    std::cout << "  Diffusion num   = rx=" << RX << "  ry=" << RY
              << "  (rx+ry must be <= 0.5)\n";
    std::cout << "  Final time      = " << T_FINAL << " s\n";
    std::cout << "  OpenMP threads  = " << num_threads << "\n\n";

    if (RX + RY > 0.5) {
        std::cerr << "ERROR: Unstable! rx+ry = " << RX+RY << " > 0.5\n";
        return 1;
    }

    // Allocate flat 2D arrays
    std::vector<double> T_old(NX * NY, 0.0);
    std::vector<double> T_new(NX * NY, 0.0);

    initialize(T_old);

    int num_steps = static_cast<int>(T_FINAL / DT);
    std::cout << "Running " << num_steps << " time steps...\n\n";

    auto start = std::chrono::high_resolution_clock::now();

    double t = 0.0;
    for (int step = 0; step < num_steps; step++) {

        // Parallel 2D FTCS update (interior points only)
        #pragma omp parallel for collapse(2) schedule(static)
        for (int i = 1; i < NX - 1; i++) {
            for (int j = 1; j < NY - 1; j++) {
                double d2x = T_old[idx(i+1,j)] - 2.0*T_old[idx(i,j)] + T_old[idx(i-1,j)];
                double d2y = T_old[idx(i,j+1)] - 2.0*T_old[idx(i,j)] + T_old[idx(i,j-1)];
                T_new[idx(i,j)] = T_old[idx(i,j)] + RX*d2x + RY*d2y;
            }
        }

        // Dirichlet BCs (edges stay at 0)
        #pragma omp parallel for
        for (int j = 0; j < NY; j++) {
            T_new[idx(0,    j)] = 0.0;
            T_new[idx(NX-1, j)] = 0.0;
        }
        #pragma omp parallel for
        for (int i = 0; i < NX; i++) {
            T_new[idx(i, 0   )] = 0.0;
            T_new[idx(i, NY-1)] = 0.0;
        }

        std::swap(T_old, T_new);
        t += DT;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    double error = calculate_error(T_old, t);
    double max_T = *std::max_element(T_old.begin(), T_old.end());

    std::cout << "========================================\n";
    std::cout << "  RESULTS\n";
    std::cout << "========================================\n";
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "  Final time:       " << t << " s\n";
    std::cout << "  Execution time:   " << elapsed.count() << " ms\n";
    std::cout << "  L2 Error (RMS):   " << std::scientific << error << "\n";
    std::cout << "  Max temperature:  " << std::fixed << max_T << " °C\n";
    std::cout << "  Threads used:     " << num_threads << "\n";

    save_results(T_old, t, "hpc-proj/Parallel/results_2d_omp.csv");
    std::cout << "\nResults saved to results_2d_omp.csv\n";
    std::cout << "(Coarsened by stride=5 to keep file size manageable)\n";

    return 0;
}