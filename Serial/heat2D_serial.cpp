/**
 * Sequential Heat Equation Solver
 */

#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <algorithm>

// ============== SIMULATION PARAMETERS ==============
const double LX = 1.0;          // Plate width  (meters)
const double LY = 1.0;          // Plate height (meters)
const double ALPHA = 0.01;      // Thermal diffusivity (m²/s)
const int NX = 500;             // Number of spatial points in x
const int NY = 500;             // Number of spatial points in y
const double T_FINAL = 0.5;    // Simulation time (seconds)

const double DX = LX / (NX - 1);  // Spatial step in x
const double DY = LY / (NY - 1);  // Spatial step in y

// CFL stability condition for 2D: r <= 0.25
const double DT = 0.4 * 0.5 * (DX * DX * DY * DY) / (ALPHA * (DX * DX + DY * DY));

const double RX = ALPHA * DT / (DX * DX);  // Diffusion number in x
const double RY = ALPHA * DT / (DY * DY);  // Diffusion number in y

const double PI = 3.14159265358979323846;

// ============== HELPER: 2D Index ==============
// Maps (i, j) -> flat array index (row-major)
inline int IDX(int i, int j) { return i * NY + j; }

// ============== FUNCTIONS ==============

// Initial condition: T(x,y,0) = 100 * sin(pi*x) * sin(pi*y)
void initialize(std::vector<double>& T) {
    for (int i = 0; i < NX; i++) {
        double x = i * DX;
        for (int j = 0; j < NY; j++) {
            double y = j * DY;
            T[IDX(i, j)] = 100.0 * sin(PI * x) * sin(PI * y);
        }
    }
    // Enforce boundary conditions (already zero from sin, but explicit for clarity)
    for (int i = 0; i < NX; i++) {
        T[IDX(i, 0)]    = 0.0;  // Bottom edge
        T[IDX(i, NY-1)] = 0.0;  // Top edge
    }
    for (int j = 0; j < NY; j++) {
        T[IDX(0, j)]    = 0.0;  // Left edge
        T[IDX(NX-1, j)] = 0.0;  // Right edge
    }
}

// Analytical solution: T(x,y,t) = 100 * exp(-2*alpha*pi²*t) * sin(pi*x) * sin(pi*y)
double analytical_solution(double x, double y, double t) {
    double decay = -ALPHA * PI * PI * (1.0/(LX*LX) + 1.0/(LY*LY));
    return 100.0 * exp(decay * t) * sin(PI * x / LX) * sin(PI * y / LY);
}

// Calculate L2 (RMSE) error between numerical and analytical
double calculate_error(const std::vector<double>& T, double t) {
    double error = 0.0;
    for (int i = 0; i < NX; i++) {
        double x = i * DX;
        for (int j = 0; j < NY; j++) {
            double y = j * DY;
            double diff = T[IDX(i, j)] - analytical_solution(x, y, t);
            error += diff * diff;
        }
    }
    return sqrt(error / (NX * NY));
}

// Save temperature profile to CSV file (sampled for large grids)
void save_results(const std::vector<double>& T, double t, const std::string& filename, int stride = 5) {
    std::ofstream file(filename);
    file << "# x, y, T_numerical, T_analytical\n";

    for (int i = 0; i < NX; i += stride) {
        double x = i * DX;
        for (int j = 0; j < NY; j += stride) {
            double y = j * DY;
            file << x << ", " << y << ", "
                 << T[IDX(i, j)] << ", "
                 << analytical_solution(x, y, t) << "\n";
        }
    }
    file.close();
}

// ============== MAIN SOLVER ==============
int main() {
    std::cout << "========================================\n";
    std::cout << "  Sequential 2D Heat Equation Solver\n";
    std::cout << "========================================\n\n";

    // Print parameters
    std::cout << "Parameters:\n";
    std::cout << "  Plate size      = " << LX << " x " << LY << " m\n";
    std::cout << "  Grid points     = " << NX << " x " << NY << " (" << NX * NY << " total)\n";
    std::cout << "  Spatial step dx = " << DX << " m\n";
    std::cout << "  Spatial step dy = " << DY << " m\n";
    std::cout << "  Time step dt    = " << DT << " s\n";
    std::cout << "  Diffusivity α   = " << ALPHA << " m²/s\n";
    std::cout << "  Diffusion rx    = " << RX << "\n";
    std::cout << "  Diffusion ry    = " << RY << "\n";
    std::cout << "  rx + ry         = " << (RX + RY) << " (must be <= 0.5)\n";
    std::cout << "  Final time      = " << T_FINAL << " s\n\n";

    // Check stability (2D CFL: rx + ry <= 0.5)
    if ((RX + RY) > 0.5) {
        std::cerr << "ERROR: Unstable! rx + ry = " << (RX + RY) << " > 0.5\n";
        return 1;
    }

    // Allocate flat 2D arrays (row-major)
    std::vector<double> T_old(NX * NY, 0.0);
    std::vector<double> T_new(NX * NY, 0.0);

    // Initialize
    initialize(T_old);

    // Count time steps
    int num_steps = static_cast<int>(T_FINAL / DT);
    std::cout << "Running " << num_steps << " time steps...\n\n";

    // Start timing
    auto start = std::chrono::high_resolution_clock::now();

    // ============== TIME-STEPPING LOOP ==============
    double t = 0.0;
    for (int step = 0; step < num_steps; step++) {

        // Update interior points using FTCS scheme (2D stencil)
        for (int i = 1; i < NX - 1; i++) {
            for (int j = 1; j < NY - 1; j++) {
                T_new[IDX(i, j)] = T_old[IDX(i, j)]
                    + RX * (T_old[IDX(i+1, j)] - 2.0 * T_old[IDX(i, j)] + T_old[IDX(i-1, j)])
                    + RY * (T_old[IDX(i, j+1)] - 2.0 * T_old[IDX(i, j)] + T_old[IDX(i, j-1)]);
            }
        }

        // Apply boundary conditions (all edges remain zero)
        for (int i = 0; i < NX; i++) {
            T_new[IDX(i, 0)]    = 0.0;
            T_new[IDX(i, NY-1)] = 0.0;
        }
        for (int j = 0; j < NY; j++) {
            T_new[IDX(0, j)]    = 0.0;
            T_new[IDX(NX-1, j)] = 0.0;
        }

        // Swap arrays
        std::swap(T_old, T_new);

        t += DT;
    }

    // Stop timing
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    // Calculate final RMSE error
    double error = calculate_error(T_old, t);

    // Find max temperature
    double max_temp = *std::max_element(T_old.begin(), T_old.end());

    // Print results
    std::cout << "========================================\n";
    std::cout << "  RESULTS\n";
    std::cout << "========================================\n";
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "  Final time:       " << t << " s\n";
    std::cout << "  Execution time:   " << elapsed.count() << " ms\n";
    std::cout << "  RMSE Error:       " << std::scientific << error << "\n";
    std::cout << "  Max temperature:  " << std::fixed << max_temp << " °C\n";

    // Save results
    save_results(T_old, t, "Serial/results_2d_seq.csv");
    std::cout << "\nResults saved to Serial/results_2d_seq.csv\n";

    return 0;
}