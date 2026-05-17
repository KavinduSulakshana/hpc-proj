/**
 * Hybrid MPI + OpenMP 2D Heat Equation Solver
 *
 * MPI distributes contiguous x-rows across processes. Each process stores
 * two ghost rows for halo exchange and uses OpenMP over its local rows.
 */

#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

// ============== SIMULATION PARAMETERS ==============
const double LX = 1.0;
const double LY = 1.0;
const double ALPHA = 0.01;
int NX = 500;
int NY = 500;
const double T_FINAL = 0.5;

double DX = 0.0;
double DY = 0.0;
double DT = 0.0;
double RX = 0.0;
double RY = 0.0;
const double PI = 3.14159265358979323846;

inline int lidx(int local_i, int j) { return local_i * NY + j; }
inline int gidx(int global_i, int j) { return global_i * NY + j; }

void configure_grid(int nx, int ny) {
    NX = std::max(3, nx);
    NY = std::max(3, ny);
    DX = LX / (NX - 1);
    DY = LY / (NY - 1);
    DT = 0.4 * 0.5 * (DX * DX * DY * DY) / (ALPHA * (DX * DX + DY * DY));
    RX = ALPHA * DT / (DX * DX);
    RY = ALPHA * DT / (DY * DY);
}

struct Decomposition {
    int start_row;
    int local_rows;
};

Decomposition decompose_rows(int rank, int size) {
    int base = NX / size;
    int extra = NX % size;
    int local_rows = base + (rank < extra ? 1 : 0);
    int start_row = rank * base + std::min(rank, extra);
    return {start_row, local_rows};
}

double analytical_solution(double x, double y, double t) {
    double decay = -ALPHA * PI * PI * (1.0 / (LX * LX) + 1.0 / (LY * LY));
    return 100.0 * std::exp(decay * t) * std::sin(PI * x / LX) * std::sin(PI * y / LY);
}

void initialize_local(std::vector<double>& T, const Decomposition& d) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (int li = 1; li <= d.local_rows; li++) {
        for (int j = 0; j < NY; j++) {
            int gi = d.start_row + li - 1;
            double x = gi * DX;
            double y = j * DY;
            T[lidx(li, j)] = 100.0 * std::sin(PI * x / LX) * std::sin(PI * y / LY);
        }
    }

    #pragma omp parallel for schedule(static)
    for (int li = 1; li <= d.local_rows; li++) {
        int gi = d.start_row + li - 1;
        T[lidx(li, 0)] = 0.0;
        T[lidx(li, NY - 1)] = 0.0;
        if (gi == 0 || gi == NX - 1) {
            for (int j = 0; j < NY; j++) {
                T[lidx(li, j)] = 0.0;
            }
        }
    }
}

void exchange_halos(std::vector<double>& T, const Decomposition& d, int rank, int size) {
    int up = rank - 1;
    int down = rank + 1;
    MPI_Status status;

    if (up >= 0) {
        MPI_Sendrecv(&T[lidx(1, 0)], NY, MPI_DOUBLE, up, 0,
                     &T[lidx(0, 0)], NY, MPI_DOUBLE, up, 1,
                     MPI_COMM_WORLD, &status);
    } else {
        std::fill(T.begin(), T.begin() + NY, 0.0);
    }

    if (down < size) {
        MPI_Sendrecv(&T[lidx(d.local_rows, 0)], NY, MPI_DOUBLE, down, 1,
                     &T[lidx(d.local_rows + 1, 0)], NY, MPI_DOUBLE, down, 0,
                     MPI_COMM_WORLD, &status);
    } else {
        std::fill(T.begin() + lidx(d.local_rows + 1, 0),
                  T.begin() + lidx(d.local_rows + 2, 0), 0.0);
    }
}

void update_local(const std::vector<double>& T_old, std::vector<double>& T_new,
                  const Decomposition& d) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (int li = 1; li <= d.local_rows; li++) {
        for (int j = 1; j < NY - 1; j++) {
            int gi = d.start_row + li - 1;
            if (gi == 0 || gi == NX - 1) {
                T_new[lidx(li, j)] = 0.0;
            } else {
                double d2x = T_old[lidx(li + 1, j)] - 2.0 * T_old[lidx(li, j)]
                           + T_old[lidx(li - 1, j)];
                double d2y = T_old[lidx(li, j + 1)] - 2.0 * T_old[lidx(li, j)]
                           + T_old[lidx(li, j - 1)];
                T_new[lidx(li, j)] = T_old[lidx(li, j)] + RX * d2x + RY * d2y;
            }
        }
    }

    #pragma omp parallel for schedule(static)
    for (int li = 1; li <= d.local_rows; li++) {
        int gi = d.start_row + li - 1;
        T_new[lidx(li, 0)] = 0.0;
        T_new[lidx(li, NY - 1)] = 0.0;
        if (gi == 0 || gi == NX - 1) {
            for (int j = 0; j < NY; j++) {
                T_new[lidx(li, j)] = 0.0;
            }
        }
    }
}

double local_squared_error(const std::vector<double>& T, const Decomposition& d, double t) {
    double err = 0.0;
    #pragma omp parallel for collapse(2) reduction(+:err) schedule(static)
    for (int li = 1; li <= d.local_rows; li++) {
        for (int j = 0; j < NY; j++) {
            int gi = d.start_row + li - 1;
            double diff = T[lidx(li, j)] - analytical_solution(gi * DX, j * DY, t);
            err += diff * diff;
        }
    }
    return err;
}

double local_max_temperature(const std::vector<double>& T, const Decomposition& d) {
    double local_max = 0.0;
    #pragma omp parallel for reduction(max:local_max) schedule(static)
    for (int li = 1; li <= d.local_rows; li++) {
        for (int j = 0; j < NY; j++) {
            local_max = std::max(local_max, T[lidx(li, j)]);
        }
    }
    return local_max;
}

void save_results(const std::vector<double>& T, double t, const std::string& filename) {
    std::ofstream file(filename);
    file << "# x, y, T_numerical, T_analytical\n";
    file << std::fixed << std::setprecision(6);

    int stride = std::max(1, NX / 100);
    for (int i = 0; i < NX; i += stride) {
        double x = i * DX;
        for (int j = 0; j < NY; j += stride) {
            double y = j * DY;
            file << x << ", " << y << ", "
                 << T[gidx(i, j)] << ", "
                 << analytical_solution(x, y, t) << "\n";
        }
    }
}

void save_summary(double final_time, double exec_ms, double rmse, double max_temp,
                  int ranks, int threads, const std::string& filename) {
    std::ofstream file(filename);
    file << "Final_time,Execution_time_ms,RMSE_Error,Max_temperature,MPI_ranks,OpenMP_threads,NX,NY\n";
    file << final_time << "," << exec_ms << "," << rmse << "," << max_temp
         << "," << ranks << "," << threads << "," << NX << "," << NY << "\n";
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int num_threads = omp_get_max_threads();
    if (argc > 1) {
        num_threads = std::max(1, std::atoi(argv[1]));
    }
    if (argc > 2) NX = std::max(3, std::atoi(argv[2]));
    if (argc > 3) NY = std::max(3, std::atoi(argv[3]));
    configure_grid(NX, NY);
    omp_set_num_threads(num_threads);

    if (size > NX) {
        if (rank == 0) {
            std::cerr << "ERROR: MPI ranks cannot exceed NX (" << NX << ").\n";
        }
        MPI_Finalize();
        return 1;
    }

    if (RX + RY > 0.5) {
        if (rank == 0) {
            std::cerr << "ERROR: Unstable! rx+ry = " << RX + RY << " > 0.5\n";
        }
        MPI_Finalize();
        return 1;
    }

    Decomposition d = decompose_rows(rank, size);
    std::vector<double> T_old((d.local_rows + 2) * NY, 0.0);
    std::vector<double> T_new((d.local_rows + 2) * NY, 0.0);
    initialize_local(T_old, d);

    int steps = static_cast<int>(T_FINAL / DT);
    double t = 0.0;

    if (rank == 0) {
        std::cout << "========================================\n";
        std::cout << "  Hybrid MPI + OpenMP 2D Heat Solver\n";
        std::cout << "========================================\n\n";
        std::cout << "Parameters:\n";
        std::cout << "  Domain          = " << LX << " m x " << LY << " m\n";
        std::cout << "  Grid            = " << NX << " x " << NY
                  << " (" << static_cast<long long>(NX) * NY << " points)\n";
        std::cout << "  Time step dt    = " << DT << " s\n";
        std::cout << "  Diffusion num   = rx=" << RX << " ry=" << RY
                  << " (rx+ry must be <= 0.5)\n";
        std::cout << "  Final time      = " << T_FINAL << " s\n";
        std::cout << "  MPI ranks       = " << size << "\n";
        std::cout << "  OpenMP threads  = " << num_threads << " per rank\n";
        std::cout << "  Running " << steps << " time steps...\n\n";
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    for (int step = 0; step < steps; step++) {
        exchange_halos(T_old, d, rank, size);
        update_local(T_old, T_new, d);
        std::swap(T_old, T_new);
        t += DT;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double local_elapsed_ms = (MPI_Wtime() - t0) * 1000.0;
    double elapsed_ms = 0.0;
    MPI_Reduce(&local_elapsed_ms, &elapsed_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    double local_err = local_squared_error(T_old, d, t);
    double global_err = 0.0;
    MPI_Reduce(&local_err, &global_err, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    double local_max = local_max_temperature(T_old, d);
    double global_max = 0.0;
    MPI_Reduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    std::vector<int> recv_counts;
    std::vector<int> displs;
    std::vector<double> global_T;
    if (rank == 0) {
        recv_counts.resize(size);
        displs.resize(size);
        global_T.resize(NX * NY);
        for (int r = 0; r < size; r++) {
            Decomposition rd = decompose_rows(r, size);
            recv_counts[r] = rd.local_rows * NY;
            displs[r] = rd.start_row * NY;
        }
    }

    MPI_Gatherv(&T_old[lidx(1, 0)], d.local_rows * NY, MPI_DOUBLE,
                rank == 0 ? global_T.data() : nullptr,
                rank == 0 ? recv_counts.data() : nullptr,
                rank == 0 ? displs.data() : nullptr,
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double rmse = std::sqrt(global_err / (NX * NY));
        double throughput = (static_cast<double>(NX) * NY * steps) / (elapsed_ms * 1e3);

        std::cout << "========================================\n";
        std::cout << "  RESULTS\n";
        std::cout << "========================================\n";
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "  Final time:       " << t << " s\n";
        std::cout << "  Execution time:   " << elapsed_ms << " ms\n";
        std::cout << "  RMSE Error:       " << std::scientific << rmse << "\n";
        std::cout << "  Max temperature:  " << std::fixed << global_max << " C\n";
        std::cout << "  Throughput:       " << throughput << " MPoints/s\n";
        std::cout << "  MPI ranks:        " << size << "\n";
        std::cout << "  Threads/rank:     " << num_threads << "\n";

        save_results(global_T, t, "Hybrid/results_2d_hybrid.csv");
        save_summary(t, elapsed_ms, rmse, global_max, size, num_threads,
                     "Hybrid/summary_2d_hybrid.csv");

        std::cout << "\nResults saved to:\n";
        std::cout << "  Hybrid/results_2d_hybrid.csv\n";
        std::cout << "  Hybrid/summary_2d_hybrid.csv\n";
    }

    MPI_Finalize();
    return 0;
}
