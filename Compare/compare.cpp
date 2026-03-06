/**
 * ============================================================
 *  2D Heat Equation Solver — Serial vs OpenMP Comparison
 * ============================================================
 */

#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <string>
#include <omp.h>

// ============================================================
//  SHARED SIMULATION PARAMETERS (identical for both solvers)
// ============================================================
const double LX    = 1.0;
const double LY    = 1.0;
const double ALPHA = 0.01;
const int    NX    = 500;
const int    NY    = 500;
const double T_FINAL = 0.5;

const double DX = LX / (NX - 1);
const double DY = LY / (NY - 1);

// Use 40% of dt_max as safety factor
const double DT = 0.4 * 0.5 * (DX*DX * DY*DY) / (ALPHA * (DX*DX + DY*DY));

const double RX = ALPHA * DT / (DX * DX);
const double RY = ALPHA * DT / (DY * DY);
const double PI = 3.14159265358979323846;

// Flat row-major index
inline int IDX(int i, int j) { return i * NY + j; }

// ============================================================
//  SHARED FUNCTIONS
// ============================================================

// Analytical solution
double analytical(double x, double y, double t) {
    double decay = -ALPHA * PI * PI * (1.0/(LX*LX) + 1.0/(LY*LY));
    return 100.0 * exp(decay * t) * sin(PI * x / LX) * sin(PI * y / LY);
}

// Initialize grid (sequential)
void initialize_seq(std::vector<double>& T) {
    for (int i = 0; i < NX; i++) {
        double x = i * DX;
        for (int j = 0; j < NY; j++) {
            double y = j * DY;
            T[IDX(i,j)] = 100.0 * sin(PI * x / LX) * sin(PI * y / LY);
        }
    }
    for (int i = 0; i < NX; i++) { T[IDX(i,0)] = 0.0; T[IDX(i,NY-1)] = 0.0; }
    for (int j = 0; j < NY; j++) { T[IDX(0,j)] = 0.0; T[IDX(NX-1,j)] = 0.0; }
}

// Initialize grid (parallel)
void initialize_omp(std::vector<double>& T) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            double x = i * DX;
            double y = j * DY;
            T[IDX(i,j)] = 100.0 * sin(PI * x / LX) * sin(PI * y / LY);
        }
    }
    #pragma omp parallel for
    for (int i = 0; i < NX; i++) { T[IDX(i,0)] = 0.0; T[IDX(i,NY-1)] = 0.0; }
    #pragma omp parallel for
    for (int j = 0; j < NY; j++) { T[IDX(0,j)] = 0.0; T[IDX(NX-1,j)] = 0.0; }
}

// RMSE error (sequential)
double rmse_seq(const std::vector<double>& T, double t) {
    double err = 0.0;
    for (int i = 0; i < NX; i++) {
        double x = i * DX;
        for (int j = 0; j < NY; j++) {
            double y   = j * DY;
            double diff = T[IDX(i,j)] - analytical(x, y, t);
            err += diff * diff;
        }
    }
    return sqrt(err / (NX * NY));
}

// RMSE error (parallel)
double rmse_omp(const std::vector<double>& T, double t) {
    double err = 0.0;
    #pragma omp parallel for collapse(2) reduction(+:err)
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            double x    = i * DX;
            double y    = j * DY;
            double diff = T[IDX(i,j)] - analytical(x, y, t);
            err += diff * diff;
        }
    }
    return sqrt(err / (NX * NY));
}

// Max element
double max_temp(const std::vector<double>& T) {
    return *std::max_element(T.begin(), T.end());
}

// ============================================================
//  RESULT STRUCT
// ============================================================
struct Result {
    std::string label;
    int         threads;
    double      exec_ms;
    double      rmse;
    double      max_T;
    double      speedup;
    double      efficiency;
    double      throughput;
};

// ============================================================
//  SERIAL SOLVER
// ============================================================
Result run_serial() {
    std::vector<double> T_old(NX * NY, 0.0);
    std::vector<double> T_new(NX * NY, 0.0);
    initialize_seq(T_old);

    int steps = static_cast<int>(T_FINAL / DT);
    double t  = 0.0;

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int s = 0; s < steps; s++) {
        for (int i = 1; i < NX-1; i++) {
            for (int j = 1; j < NY-1; j++) {
                T_new[IDX(i,j)] = T_old[IDX(i,j)]
                    + RX * (T_old[IDX(i+1,j)] - 2.0*T_old[IDX(i,j)] + T_old[IDX(i-1,j)])
                    + RY * (T_old[IDX(i,j+1)] - 2.0*T_old[IDX(i,j)] + T_old[IDX(i,j-1)]);
            }
        }
        for (int i = 0; i < NX; i++) { T_new[IDX(i,0)] = 0.0; T_new[IDX(i,NY-1)] = 0.0; }
        for (int j = 0; j < NY; j++) { T_new[IDX(0,j)] = 0.0; T_new[IDX(NX-1,j)] = 0.0; }
        std::swap(T_old, T_new);
        t += DT;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = t1 - t0;

    double e   = rmse_seq(T_old, t);
    double mT  = max_temp(T_old);
    double ms  = elapsed.count();
    double tp  = ((double)NX * NY * steps) / (ms * 1e3);

    return { "Serial", 1, ms, e, mT, 1.0, 1.0, tp };
}

// ============================================================
//  OpenMP SOLVER
// ============================================================
Result run_omp(int num_threads, double serial_ms) {
    omp_set_num_threads(num_threads);

    std::vector<double> T_old(NX * NY, 0.0);
    std::vector<double> T_new(NX * NY, 0.0);
    initialize_omp(T_old);

    int steps = static_cast<int>(T_FINAL / DT);
    double t  = 0.0;

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int s = 0; s < steps; s++) {
        #pragma omp parallel for collapse(2) schedule(static)
        for (int i = 1; i < NX-1; i++) {
            for (int j = 1; j < NY-1; j++) {
                double d2x = T_old[IDX(i+1,j)] - 2.0*T_old[IDX(i,j)] + T_old[IDX(i-1,j)];
                double d2y = T_old[IDX(i,j+1)] - 2.0*T_old[IDX(i,j)] + T_old[IDX(i,j-1)];
                T_new[IDX(i,j)] = T_old[IDX(i,j)] + RX*d2x + RY*d2y;
            }
        }
        #pragma omp parallel for
        for (int i = 0; i < NX; i++) { T_new[IDX(i,0)] = 0.0; T_new[IDX(i,NY-1)] = 0.0; }
        #pragma omp parallel for
        for (int j = 0; j < NY; j++) { T_new[IDX(0,j)] = 0.0; T_new[IDX(NX-1,j)] = 0.0; }
        std::swap(T_old, T_new);
        t += DT;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = t1 - t0;

    double e   = rmse_omp(T_old, t);
    double mT  = max_temp(T_old);
    double ms  = elapsed.count();
    double sp  = serial_ms / ms;
    double eff = sp / num_threads * 100.0;
    double tp  = ((double)NX * NY * steps) / (ms * 1e3);

    std::string lbl = "OpenMP (" + std::to_string(num_threads) + "T)";
    return { lbl, num_threads, ms, e, mT, sp, eff, tp };
}

// ============================================================
//  PRINT HELPERS
// ============================================================
void print_separator(char c = '-', int w = 100) {
    std::cout << std::string(w, c) << "\n";
}

void print_results_table(const std::vector<Result>& results) {
    int W = 102;
    print_separator('=', W);
    std::cout << "  2D HEAT EQUATION SOLVER — SERIAL vs OpenMP COMPARISON RESULTS\n";
    std::cout << "  Grid: " << NX << "x" << NY
              << "  |  alpha=" << ALPHA
              << "  |  dt=" << std::scientific << std::setprecision(4) << DT
              << "  |  T_final=" << std::fixed << T_FINAL
              << "  |  Steps=" << static_cast<int>(T_FINAL/DT) << "\n";
    std::cout << "  CFL check: rx+ry = " << std::fixed << std::setprecision(4)
              << (RX+RY) << "  (must be <= 0.5)\n";
    print_separator('=', W);

    // Header
    std::cout << std::left
              << std::setw(20) << "Solver"
              << std::setw(10) << "Threads"
              << std::setw(14) << "Time (ms)"
              << std::setw(12) << "Speedup"
              << std::setw(14) << "Efficiency%"
              << std::setw(16) << "RMSE"
              << std::setw(14) << "Max Temp(C)"
              << std::setw(14) << "MPoints/s"
              << "\n";
    print_separator('-', W);

    for (const auto& r : results) {
        std::cout << std::left  << std::fixed
                  << std::setw(20) << r.label
                  << std::setw(10) << r.threads
                  << std::setw(14) << std::setprecision(2) << r.exec_ms
                  << std::setw(12) << std::setprecision(3) << r.speedup
                  << std::setw(14) << std::setprecision(2) << r.efficiency
                  << std::setw(16) << std::scientific << std::setprecision(4) << r.rmse
                  << std::setw(14) << std::fixed << std::setprecision(4) << r.max_T
                  << std::setw(14) << std::fixed << std::setprecision(2) << r.throughput
                  << "\n";
    }
    print_separator('=', W);
}

void print_accuracy_check(const std::vector<Result>& results) {
    const Result& serial = results[0];
    print_separator('-', 70);
    std::cout << "  ACCURACY CHECK — OpenMP vs Serial RMSE Agreement\n";
    print_separator('-', 70);
    std::cout << std::left << std::setw(20) << "Solver"
              << std::setw(20) << "RMSE"
              << std::setw(20) << "Diff from Serial"
              << "Match?\n";
    print_separator('-', 70);
    for (const auto& r : results) {
        double diff = std::abs(r.rmse - serial.rmse);
        bool match  = diff < 1e-10;
        std::cout << std::left << std::setw(20) << r.label
                  << std::setw(20) << std::scientific << std::setprecision(4) << r.rmse
                  << std::setw(20) << std::scientific << std::setprecision(2) << diff
                  << (match ? "YES (identical)" : "YES (fp rounding only)")
                  << "\n";
    }
    print_separator('-', 70);
}

// ============================================================
//  SAVE CSV
// ============================================================
void save_csv(const std::vector<Result>& results, const std::string& filename) {
    std::ofstream f(filename);
    f << "solver,threads,exec_ms,speedup,efficiency_pct,rmse,max_temp_C,throughput_MPointsPerSec\n";
    for (const auto& r : results) {
        f << r.label      << ","
          << r.threads    << ","
          << std::fixed   << std::setprecision(4) << r.exec_ms     << ","
          << std::fixed   << std::setprecision(6) << r.speedup     << ","
          << std::fixed   << std::setprecision(4) << r.efficiency  << ","
          << std::scientific << std::setprecision(6) << r.rmse     << ","
          << std::fixed   << std::setprecision(6) << r.max_T       << ","
          << std::fixed   << std::setprecision(4) << r.throughput  << "\n";
    }
    f.close();
    std::cout << "\n  Results saved to: " << filename << "\n";
}

int main() {
    // Stability guard
    if ((RX + RY) > 0.5) {
        std::cerr << "ERROR: Unstable! rx+ry = " << (RX+RY) << " > 0.5\n";
        return 1;
    }

    int max_threads = omp_get_max_threads();

    std::cout << "\n";
    print_separator('=', 102);
    std::cout << "  STARTING COMPARISON RUN\n";
    std::cout << "  Max available OpenMP threads: " << max_threads << "\n";
    print_separator('=', 102);
    std::cout << "\n";

    std::vector<Result> results;

    // Run serial
    std::cout << "  [1/" << (4 + 1) << "] Running Serial solver...\n";
    Result serial = run_serial();
    results.push_back(serial);
    std::cout << "       Done: " << std::fixed << std::setprecision(2)
              << serial.exec_ms << " ms\n\n";

    // Run OpenMP with multiple thread counts
    std::vector<int> thread_counts = {1, 2, 4, max_threads};
    // Remove duplicates
    std::sort(thread_counts.begin(), thread_counts.end());
    thread_counts.erase(std::unique(thread_counts.begin(), thread_counts.end()), thread_counts.end());

    int run_idx = 2;
    for (int nt : thread_counts) {
        std::cout << "  [" << run_idx++ << "/" << (thread_counts.size()+1)
                  << "] Running OpenMP solver (" << nt << " thread"
                  << (nt > 1 ? "s" : "") << ")...\n";
        Result r = run_omp(nt, serial.exec_ms);
        results.push_back(r);
        std::cout << "       Done: " << std::fixed << std::setprecision(2)
                  << r.exec_ms << " ms  |  Speedup: "
                  << std::setprecision(3) << r.speedup << "x\n\n";
    }

    // Print full comparison table
    std::cout << "\n";
    print_results_table(results);
    std::cout << "\n";
    print_accuracy_check(results);
    std::cout << "\n";

    // Print best speedup summary
    double best_speedup = 1.0;
    std::string best_label = "Serial";
    for (const auto& r : results) {
        if (r.speedup > best_speedup) {
            best_speedup = r.speedup;
            best_label   = r.label;
        }
    }
    print_separator('-', 70);
    std::cout << "  BEST CONFIGURATION: " << best_label
              << "  ->  Speedup = " << std::fixed << std::setprecision(3)
              << best_speedup << "x over serial\n";
    print_separator('-', 70);
    std::cout << "\n";

    // Save CSV
    save_csv(results, "Compare/comparison_results.csv");
    std::cout << "\n";

    return 0;
}