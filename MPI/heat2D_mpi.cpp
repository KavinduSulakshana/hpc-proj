/**
 * Pure MPI 2D Heat Equation Solver
 *
 * MPI distributes contiguous x-rows across processes. Each process stores
 * two ghost rows for halo exchange with neighboring ranks.
 */

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

const double LX = 1.0;
const double LY = 1.0;
const double ALPHA = 0.01;
int NX = 500;
int NY = 500;
const double T_FINAL = 0.5;
const double PI = 3.14159265358979323846;

struct SimParams {
    double dx;
    double dy;
    double dt;
    double rx;
    double ry;
    int steps;
};

struct Decomposition {
    int start_row;
    int local_rows;
};

inline int lidx(int local_i, int j) { return local_i * NY + j; }
inline int gidx(int global_i, int j) { return global_i * NY + j; }

SimParams make_params() {
    SimParams p;
    p.dx = LX / (NX - 1);
    p.dy = LY / (NY - 1);
    p.dt = 0.4 * 0.5 * (p.dx * p.dx * p.dy * p.dy) /
           (ALPHA * (p.dx * p.dx + p.dy * p.dy));
    p.rx = ALPHA * p.dt / (p.dx * p.dx);
    p.ry = ALPHA * p.dt / (p.dy * p.dy);
    p.steps = static_cast<int>(T_FINAL / p.dt);
    return p;
}

Decomposition decompose_rows(int rank, int size) {
    int base = NX / size;
    int extra = NX % size;
    int local_rows = base + (rank < extra ? 1 : 0);
    int start_row = rank * base + std::min(rank, extra);
    return {start_row, local_rows};
}

double analytical_solution(double x, double y, double t) {
    double decay = -ALPHA * PI * PI * (1.0 / (LX * LX) + 1.0 / (LY * LY));
    return 100.0 * std::exp(decay * t) *
           std::sin(PI * x / LX) * std::sin(PI * y / LY);
}

void initialize_local(std::vector<double>& T, const Decomposition& d,
                      const SimParams& p) {
    for (int li = 1; li <= d.local_rows; li++) {
        int gi = d.start_row + li - 1;
        double x = gi * p.dx;
        for (int j = 0; j < NY; j++) {
            double y = j * p.dy;
            T[lidx(li, j)] = 100.0 * std::sin(PI * x / LX) *
                             std::sin(PI * y / LY);
        }
    }

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

void exchange_halos(std::vector<double>& T, const Decomposition& d,
                    int rank, int size) {
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
                  const Decomposition& d, const SimParams& p) {
    for (int li = 1; li <= d.local_rows; li++) {
        int gi = d.start_row + li - 1;
        for (int j = 1; j < NY - 1; j++) {
            if (gi == 0 || gi == NX - 1) {
                T_new[lidx(li, j)] = 0.0;
            } else {
                double d2x = T_old[lidx(li + 1, j)] - 2.0 * T_old[lidx(li, j)] +
                             T_old[lidx(li - 1, j)];
                double d2y = T_old[lidx(li, j + 1)] - 2.0 * T_old[lidx(li, j)] +
                             T_old[lidx(li, j - 1)];
                T_new[lidx(li, j)] = T_old[lidx(li, j)] + p.rx * d2x + p.ry * d2y;
            }
        }
    }

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

double local_squared_error(const std::vector<double>& T, const Decomposition& d,
                           const SimParams& p, double t) {
    double err = 0.0;
    for (int li = 1; li <= d.local_rows; li++) {
        int gi = d.start_row + li - 1;
        for (int j = 0; j < NY; j++) {
            double diff = T[lidx(li, j)] -
                          analytical_solution(gi * p.dx, j * p.dy, t);
            err += diff * diff;
        }
    }
    return err;
}

double local_max_temperature(const std::vector<double>& T,
                             const Decomposition& d) {
    double local_max = 0.0;
    for (int li = 1; li <= d.local_rows; li++) {
        for (int j = 0; j < NY; j++) {
            local_max = std::max(local_max, T[lidx(li, j)]);
        }
    }
    return local_max;
}

void save_results(const std::vector<double>& T, double t, const SimParams& p,
                  const std::string& filename) {
    std::ofstream file(filename);
    file << "# x, y, T_numerical, T_analytical\n";
    file << std::fixed << std::setprecision(6);

    int stride = std::max(1, NX / 100);
    for (int i = 0; i < NX; i += stride) {
        double x = i * p.dx;
        for (int j = 0; j < NY; j += stride) {
            double y = j * p.dy;
            file << x << ", " << y << ", "
                 << T[gidx(i, j)] << ", "
                 << analytical_solution(x, y, t) << "\n";
        }
    }
}

void save_summary(double final_time, double exec_ms, double rmse, double max_temp,
                  int ranks, const std::string& filename) {
    std::ofstream file(filename);
    file << "Final_time,Execution_time_ms,RMSE_Error,Max_temperature,MPI_ranks,NX,NY\n";
    file << final_time << "," << exec_ms << "," << rmse << ","
         << max_temp << "," << ranks << "," << NX << "," << NY << "\n";
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc > 1) NX = std::max(3, std::atoi(argv[1]));
    if (argc > 2) NY = std::max(3, std::atoi(argv[2]));

    SimParams p = make_params();

    if (size > NX) {
        if (rank == 0) {
            std::cerr << "ERROR: MPI ranks cannot exceed NX (" << NX << ").\n";
        }
        MPI_Finalize();
        return 1;
    }

    if (p.rx + p.ry > 0.5) {
        if (rank == 0) {
            std::cerr << "ERROR: Unstable! rx+ry = " << p.rx + p.ry << " > 0.5\n";
        }
        MPI_Finalize();
        return 1;
    }

    Decomposition d = decompose_rows(rank, size);
    std::vector<double> T_old((d.local_rows + 2) * NY, 0.0);
    std::vector<double> T_new((d.local_rows + 2) * NY, 0.0);
    initialize_local(T_old, d, p);

    double t = 0.0;

    if (rank == 0) {
        std::cout << "========================================\n";
        std::cout << "  Pure MPI 2D Heat Equation Solver\n";
        std::cout << "========================================\n\n";
        std::cout << "Parameters:\n";
        std::cout << "  Grid            = " << NX << " x " << NY
                  << " (" << static_cast<long long>(NX) * NY << " points)\n";
        std::cout << "  Time step dt    = " << p.dt << " s\n";
        std::cout << "  Diffusion num   = rx=" << p.rx << " ry=" << p.ry
                  << " (rx+ry must be <= 0.5)\n";
        std::cout << "  Final time      = " << T_FINAL << " s\n";
        std::cout << "  MPI ranks       = " << size << "\n";
        std::cout << "  Running " << p.steps << " time steps...\n\n";
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    for (int step = 0; step < p.steps; step++) {
        exchange_halos(T_old, d, rank, size);
        update_local(T_old, T_new, d, p);
        std::swap(T_old, T_new);
        t += p.dt;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double local_elapsed_ms = (MPI_Wtime() - t0) * 1000.0;
    double elapsed_ms = 0.0;
    MPI_Reduce(&local_elapsed_ms, &elapsed_ms, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);

    double local_err = local_squared_error(T_old, d, p, t);
    double global_err = 0.0;
    MPI_Reduce(&local_err, &global_err, 1, MPI_DOUBLE, MPI_SUM, 0,
               MPI_COMM_WORLD);

    double local_max = local_max_temperature(T_old, d);
    double global_max = 0.0;
    MPI_Reduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);

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
        double throughput = (static_cast<double>(NX) * NY * p.steps) /
                            (elapsed_ms * 1e3);

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

        save_results(global_T, t, p, "MPI/results_2d_mpi.csv");
        save_summary(t, elapsed_ms, rmse, global_max, size,
                     "MPI/summary_2d_mpi.csv");

        std::cout << "\nResults saved to:\n";
        std::cout << "  MPI/results_2d_mpi.csv\n";
        std::cout << "  MPI/summary_2d_mpi.csv\n";
    }

    MPI_Finalize();
    return 0;
}
