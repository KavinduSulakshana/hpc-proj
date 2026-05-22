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
#include <sstream>
#include <cstdlib>
#include <limits>
#include <omp.h>

//  SIMULATION PARAMETERS
const double LX      = 1.0;
const double LY      = 1.0;
const double ALPHA   = 0.01;
int          NX      = 500;
int          NY      = 500;
const double T_FINAL = 0.5;

double DX = 0.0;
double DY = 0.0;
double DT = 0.0;
double RX = 0.0;
double RY = 0.0;
const double PI = 3.14159265358979323846;

inline int IDX(int i, int j) { return i * NY + j; }
std::vector<double> serial_baseline;

void configure_grid(int nx, int ny) {
    NX = std::max(3, nx);
    NY = std::max(3, ny);
    DX = LX / (NX - 1);
    DY = LY / (NY - 1);
    DT = 0.4 * 0.5 * (DX*DX * DY*DY) / (ALPHA * (DX*DX + DY*DY));
    RX = ALPHA * DT / (DX * DX);
    RY = ALPHA * DT / (DY * DY);
}

//  SHARED PHYSICS FUNCTIONS

double analytical(double x, double y, double t) {
    double decay = -ALPHA * PI * PI * (1.0/(LX*LX) + 1.0/(LY*LY));
    return 100.0 * exp(decay * t) * sin(PI * x / LX) * sin(PI * y / LY);
}

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

void initialize_omp(std::vector<double>& T) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < NX; i++)
        for (int j = 0; j < NY; j++) {
            T[IDX(i,j)] = 100.0 * sin(PI*(i*DX)/LX) * sin(PI*(j*DY)/LY);
        }
    #pragma omp parallel for
    for (int i = 0; i < NX; i++) { T[IDX(i,0)] = 0.0; T[IDX(i,NY-1)] = 0.0; }
    #pragma omp parallel for
    for (int j = 0; j < NY; j++) { T[IDX(0,j)] = 0.0; T[IDX(NX-1,j)] = 0.0; }
}

double rmse_seq(const std::vector<double>& T, double t) {
    double err = 0.0;
    for (int i = 0; i < NX; i++) {
        double x = i * DX;
        for (int j = 0; j < NY; j++) {
            double diff = T[IDX(i,j)] - analytical(x, j*DY, t);
            err += diff * diff;
        }
    }
    return sqrt(err / (NX * NY));
}

double rmse_omp(const std::vector<double>& T, double t) {
    double err = 0.0;
    #pragma omp parallel for collapse(2) reduction(+:err)
    for (int i = 0; i < NX; i++)
        for (int j = 0; j < NY; j++) {
            double diff = T[IDX(i,j)] - analytical(i*DX, j*DY, t);
            err += diff * diff;
        }
    return sqrt(err / (NX * NY));
}

double max_temp(const std::vector<double>& T) {
    return *std::max_element(T.begin(), T.end());
}

double rmse_vs_reference(const std::vector<double>& T,
                         const std::vector<double>& ref) {
    if (T.size() != ref.size() || T.empty()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    double err = 0.0;
    for (size_t k = 0; k < T.size(); k++) {
        double diff = T[k] - ref[k];
        err += diff * diff;
    }
    return std::sqrt(err / T.size());
}

double file_rmse_vs_serial(const std::string& fname) {
    if (serial_baseline.empty()) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    std::ifstream f(fname);
    if (!f) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    std::string line;
    double err = 0.0;
    size_t count = 0;

    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }

        std::replace(line.begin(), line.end(), ',', ' ');
        std::stringstream ss(line);

        double x = 0.0;
        double y = 0.0;
        double numerical = 0.0;
        double analytical_value = 0.0;
        if (!(ss >> x >> y >> numerical >> analytical_value)) {
            continue;
        }

        int i = static_cast<int>(std::llround(x / DX));
        int j = static_cast<int>(std::llround(y / DY));
        if (i < 0 || i >= NX || j < 0 || j >= NY) {
            continue;
        }

        double diff = numerical - serial_baseline[IDX(i, j)];
        err += diff * diff;
        count++;
    }

    if (count == 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return std::sqrt(err / count);
}

//  RESULT
struct Result {
    std::string label;       // e.g. "Serial", "OpenMP-4T", "Hybrid-4x2T"
    int         threads;     // total workers used for speedup/efficiency
    int         mpi_ranks;
    int         omp_threads;
    double      exec_ms;
    double      rmse;
    double      rmse_vs_serial;
    double      max_T;
    double      speedup;
    double      efficiency;  // percent
    double      throughput;  // MPoints/s
};

//  SERIAL SOLVER
Result run_serial() {
    std::vector<double> T_old(NX * NY, 0.0);
    std::vector<double> T_new(NX * NY, 0.0);
    initialize_seq(T_old);

    int    steps = static_cast<int>(T_FINAL / DT);
    double t     = 0.0;

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int s = 0; s < steps; s++) {
        for (int i = 1; i < NX-1; i++)
            for (int j = 1; j < NY-1; j++)
                T_new[IDX(i,j)] = T_old[IDX(i,j)]
                    + RX*(T_old[IDX(i+1,j)] - 2.0*T_old[IDX(i,j)] + T_old[IDX(i-1,j)])
                    + RY*(T_old[IDX(i,j+1)] - 2.0*T_old[IDX(i,j)] + T_old[IDX(i,j-1)]);
        for (int i = 0; i < NX; i++) { T_new[IDX(i,0)] = 0.0; T_new[IDX(i,NY-1)] = 0.0; }
        for (int j = 0; j < NY; j++) { T_new[IDX(0,j)] = 0.0; T_new[IDX(NX-1,j)] = 0.0; }
        std::swap(T_old, T_new);
        t += DT;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    double e  = rmse_seq(T_old, t);
    serial_baseline = T_old;
    double mT = max_temp(T_old);
    double tp = ((double)NX * NY * steps) / (ms * 1e3);

    return { "Serial", 1, 1, 1, ms, e, 0.0, mT, 1.0, 100.0, tp };
}

//  OpenMP SOLVER
Result run_omp(int num_threads, double serial_ms) {
    omp_set_num_threads(num_threads);

    std::vector<double> T_old(NX * NY, 0.0);
    std::vector<double> T_new(NX * NY, 0.0);
    initialize_omp(T_old);

    int    steps = static_cast<int>(T_FINAL / DT);
    double t     = 0.0;

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int s = 0; s < steps; s++) {
        #pragma omp parallel for collapse(2) schedule(static)
        for (int i = 1; i < NX-1; i++)
            for (int j = 1; j < NY-1; j++) {
                double d2x = T_old[IDX(i+1,j)] - 2.0*T_old[IDX(i,j)] + T_old[IDX(i-1,j)];
                double d2y = T_old[IDX(i,j+1)] - 2.0*T_old[IDX(i,j)] + T_old[IDX(i,j-1)];
                T_new[IDX(i,j)] = T_old[IDX(i,j)] + RX*d2x + RY*d2y;
            }
        #pragma omp parallel for
        for (int i = 0; i < NX; i++) { T_new[IDX(i,0)] = 0.0; T_new[IDX(i,NY-1)] = 0.0; }
        #pragma omp parallel for
        for (int j = 0; j < NY; j++) { T_new[IDX(0,j)] = 0.0; T_new[IDX(NX-1,j)] = 0.0; }
        std::swap(T_old, T_new);
        t += DT;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms  = std::chrono::duration<double, std::milli>(t1 - t0).count();

    double e   = rmse_omp(T_old, t);
    double es  = rmse_vs_reference(T_old, serial_baseline);
    double mT  = max_temp(T_old);
    double sp  = serial_ms / ms;
    double eff = (sp / num_threads) * 100.0;
    double tp  = ((double)NX * NY * steps) / (ms * 1e3);

    return { "OpenMP-" + std::to_string(num_threads) + "T",
             num_threads, 1, num_threads, ms, e, es, mT, sp, eff, tp };
}

bool read_hybrid_summary(Result& out, double serial_ms,
                         const std::string& fname = "Hybrid/summary_2d_hybrid.csv",
                         const std::string& result_fname = "Hybrid/results_2d_hybrid.csv") {
    std::ifstream f(fname);
    if (!f) return false;

    std::string header;
    std::string row;
    std::getline(f, header);
    if (!std::getline(f, row)) return false;

    std::replace(row.begin(), row.end(), ',', ' ');
    std::stringstream ss(row);

    double final_time = 0.0;
    double exec_ms = 0.0;
    double rmse = 0.0;
    double max_T = 0.0;
    int mpi_ranks = 1;
    int omp_threads = 1;
    int file_nx = NX;
    int file_ny = NY;
    ss >> final_time >> exec_ms >> rmse >> max_T >> mpi_ranks >> omp_threads;
    ss >> file_nx >> file_ny;

    if (exec_ms <= 0.0 || mpi_ranks <= 0 || omp_threads <= 0 ||
        file_nx != NX || file_ny != NY) {
        return false;
    }

    int steps = static_cast<int>(T_FINAL / DT);
    int workers = mpi_ranks * omp_threads;
    double speedup = serial_ms / exec_ms;
    double efficiency = (speedup / workers) * 100.0;
    double throughput = (static_cast<double>(NX) * NY * steps) / (exec_ms * 1e3);

    out = { "Hybrid-" + std::to_string(mpi_ranks) + "x" +
            std::to_string(omp_threads) + "T",
            workers, mpi_ranks, omp_threads, exec_ms, rmse,
            file_rmse_vs_serial(result_fname), max_T,
            speedup, efficiency, throughput };
    return true;
}

bool read_mpi_summary(Result& out, double serial_ms,
                      const std::string& fname = "MPI/summary_2d_mpi.csv",
                      const std::string& result_fname = "MPI/results_2d_mpi.csv") {
    std::ifstream f(fname);
    if (!f) return false;

    std::string header;
    std::string row;
    std::getline(f, header);
    if (!std::getline(f, row)) return false;

    std::replace(row.begin(), row.end(), ',', ' ');
    std::stringstream ss(row);

    double final_time = 0.0;
    double exec_ms = 0.0;
    double rmse = 0.0;
    double max_T = 0.0;
    int mpi_ranks = 1;
    int file_nx = NX;
    int file_ny = NY;
    ss >> final_time >> exec_ms >> rmse >> max_T >> mpi_ranks;
    ss >> file_nx >> file_ny;

    if (exec_ms <= 0.0 || mpi_ranks <= 0 || file_nx != NX || file_ny != NY) {
        return false;
    }

    int steps = static_cast<int>(T_FINAL / DT);
    double speedup = serial_ms / exec_ms;
    double efficiency = (speedup / mpi_ranks) * 100.0;
    double throughput = (static_cast<double>(NX) * NY * steps) / (exec_ms * 1e3);

    out = { "MPI-" + std::to_string(mpi_ranks) + "R",
            mpi_ranks, mpi_ranks, 0, exec_ms, rmse,
            file_rmse_vs_serial(result_fname), max_T,
            speedup, efficiency, throughput };
    return true;
}

bool read_cuda_summary(Result& out, double serial_ms,
                       const std::string& fname = "cuda/summary_2d_cuda.csv",
                       const std::string& result_fname = "cuda/results_2d_cuda.csv") {
    std::ifstream f(fname);
    if (!f) return false;

    std::string header;
    std::string row;
    std::getline(f, header);
    if (!std::getline(f, row)) return false;

    std::replace(row.begin(), row.end(), ',', ' ');
    std::stringstream ss(row);

    double final_time = 0.0;
    double exec_ms = 0.0;
    double rmse = 0.0;
    double max_T = 0.0;
    int file_nx = NX;
    int file_ny = NY;
    ss >> final_time >> exec_ms >> rmse >> max_T;
    ss >> file_nx >> file_ny;

    if (exec_ms <= 0.0 || file_nx != NX || file_ny != NY) {
        return false;
    }

    int steps = static_cast<int>(T_FINAL / DT);
    double speedup = serial_ms / exec_ms;
    double throughput = (static_cast<double>(NX) * NY * steps) / (exec_ms * 1e3);

    out = { "CUDA", 0, 0, 0, exec_ms, rmse,
            file_rmse_vs_serial(result_fname), max_T,
            speedup, 0.0, throughput };
    return true;
}

//  CONSOLE PRINT HELPERS
void sep(char c = '-', int w = 128) { std::cout << std::string(w, c) << "\n"; }

void print_table(const std::vector<Result>& R) {
    sep('=');
    std::cout << "  2D HEAT EQUATION SOLVER -- SERIAL vs OpenMP vs MPI vs HYBRID vs CUDA\n";
    std::cout << "  Grid: " << NX << "x" << NY
              << "  | alpha=" << ALPHA
              << "  | dt=" << std::scientific << std::setprecision(4) << DT
              << "  | T_final=" << std::fixed << T_FINAL
              << "  | Steps=" << (int)(T_FINAL/DT) << "\n";
    std::cout << "  CFL check: rx+ry = " << std::fixed << std::setprecision(4)
              << (RX+RY) << "  (must be <= 0.5)\n";
    sep('=');
    std::cout << std::left
              << std::setw(16) << "Solver"
              << std::setw(10) << "Workers"
              << std::setw(10) << "MPI"
              << std::setw(10) << "OMP"
              << std::setw(14) << "Time (ms)"
              << std::setw(12) << "Speedup"
              << std::setw(14) << "Efficiency%"
              << std::setw(16) << "RMSE"
              << std::setw(16) << "RMSEvsSerial"
              << std::setw(14) << "MaxTemp(C)"
              << std::setw(12) << "MPoints/s" << "\n";
    sep();
    for (const auto& r : R)
        std::cout << std::left << std::fixed
                  << std::setw(16) << r.label
                  << std::setw(10) << r.threads
                  << std::setw(10) << r.mpi_ranks
                  << std::setw(10) << r.omp_threads
                  << std::setw(14) << std::setprecision(2)  << r.exec_ms
                  << std::setw(12) << std::setprecision(3)  << r.speedup
                  << std::setw(14) << std::setprecision(2)  << r.efficiency
                  << std::setw(16) << std::scientific << std::setprecision(4) << r.rmse
                  << std::setw(16) << std::scientific << std::setprecision(4) << r.rmse_vs_serial
                  << std::setw(14) << std::fixed << std::setprecision(4) << r.max_T
                  << std::setw(12) << std::fixed << std::setprecision(2) << r.throughput
                  << "\n";
    sep('=');
}

void print_accuracy(const std::vector<Result>& R) {
    sep('-', 75);
    std::cout << "  ACCURACY CHECK -- RMSE vs Serial Baseline\n";
    sep('-', 75);
    std::cout << std::left
              << std::setw(16) << "Solver"
              << std::setw(22) << "Analytical RMSE"
              << std::setw(22) << "RMSE vs Serial"
              << "Match?\n";
    sep('-', 75);
    for (const auto& r : R) {
        bool has_serial_rmse = !std::isnan(r.rmse_vs_serial);
        std::cout << std::left
                  << std::setw(16) << r.label
                  << std::setw(22) << std::scientific << std::setprecision(4) << r.rmse
                  << std::setw(22) << std::scientific << std::setprecision(4)
                  << r.rmse_vs_serial
                  << (has_serial_rmse
                          ? (r.rmse_vs_serial < 1e-10 ? "YES" : "CHECK")
                          : "N/A (results CSV missing)")
                  << "\n";
    }
    sep('-', 75);
}

//  SAVE CSV
void save_csv(const std::vector<Result>& R, const std::string& fname) {
    std::ofstream f(fname);
    f << "solver,workers,mpi_ranks,openmp_threads,exec_ms,speedup,efficiency_pct,rmse,rmse_vs_serial,max_temp_C,throughput_MPointsPerSec\n";
    for (const auto& r : R)
        f << r.label     << ","
          << r.threads   << ","
          << r.mpi_ranks << ","
          << r.omp_threads << ","
          << std::fixed       << std::setprecision(4) << r.exec_ms    << ","
          << std::fixed       << std::setprecision(6) << r.speedup    << ","
          << std::fixed       << std::setprecision(4) << r.efficiency << ","
          << std::scientific  << std::setprecision(6) << r.rmse       << ","
          << std::scientific  << std::setprecision(6) << r.rmse_vs_serial << ","
          << std::fixed       << std::setprecision(6) << r.max_T      << ","
          << std::fixed       << std::setprecision(4) << r.throughput << "\n";
    f.close();
    std::cout << "  CSV  saved: " << fname << "\n";
}

// ============================================================
//  GENERATE SEPARATE USER-FRIENDLY GRAPHS
// ============================================================
void generate_graphs(const std::vector<Result>& R)
{
    std::ofstream py("_heat_plots.py");

    // Collect data
    py << "import matplotlib\n";
    py << "matplotlib.use('Agg')\n";
    py << "import matplotlib.pyplot as plt\n\n";

    py << "labels = [";
    for(size_t i=0;i<R.size();i++){
        py << "'" << R[i].label << "'";
        if(i<R.size()-1) py << ",";
    }
    py << "]\n";

    py << "threads = [";
    for(size_t i=0;i<R.size();i++){
        py << R[i].threads;
        if(i<R.size()-1) py << ",";
    }
    py << "]\n";

    py << "time_ms = [";
    for(size_t i=0;i<R.size();i++){
        py << R[i].exec_ms;
        if(i<R.size()-1) py << ",";
    }
    py << "]\n";

    py << "speedup = [";
    for(size_t i=0;i<R.size();i++){
        py << R[i].speedup;
        if(i<R.size()-1) py << ",";
    }
    py << "]\n";

    py << "eff = [";
    for(size_t i=0;i<R.size();i++){
        py << R[i].efficiency;
        if(i<R.size()-1) py << ",";
    }
    py << "]\n";

    py << "rmse = [";
    for(size_t i=0;i<R.size();i++){
        py << R[i].rmse;
        if(i<R.size()-1) py << ",";
    }
    py << "]\n";

    py << "throughput = [";
    for(size_t i=0;i<R.size();i++){
        py << R[i].throughput;
        if(i<R.size()-1) py << ",";
    }
    py << "]\n\n";

    // Execution Time Graph
    py << R"PY(
plt.figure(figsize=(7,5))
plt.bar(labels, time_ms)
plt.title("Execution Time Comparison")
plt.ylabel("Time (ms)")
plt.xlabel("Solver")
plt.xticks(rotation=20, ha='right')
plt.grid(axis='y')
plt.tight_layout()
plt.savefig("Compare/execution_time.png",dpi=200)
plt.close()
)PY";

    // Speedup Graph
    py << R"PY(
plt.figure(figsize=(8,5))
plt.bar(labels, speedup)
plt.title("Speedup Comparison")
plt.xlabel("Solver")
plt.ylabel("Speedup")
plt.xticks(rotation=20, ha='right')
plt.grid(axis='y')
plt.tight_layout()
plt.savefig("Compare/speedup.png",dpi=200)
plt.close()
)PY";

    // Efficiency Graph
    py << R"PY(
plt.figure(figsize=(8,5))
plt.bar(labels, eff)
plt.title("Parallel Efficiency by Solver")
plt.xlabel("Solver")
plt.ylabel("Efficiency (%) = Speedup / Workers x 100")
plt.xticks(rotation=20, ha='right')
plt.grid(axis='y')
for i, (label, value) in enumerate(zip(labels, eff)):
    text = "N/A" if label == "CUDA" else f"{value:.1f}%"
    plt.text(i, value, text, ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.savefig("Compare/efficiency.png",dpi=200)
plt.close()
)PY";

    // RMSE Graph
    py << R"PY(
plt.figure(figsize=(7,5))
plt.bar(labels, rmse)
plt.title("RMSE Accuracy Comparison")
plt.ylabel("RMSE")
plt.xlabel("Solver")
plt.xticks(rotation=20, ha='right')
plt.grid(axis='y')
plt.tight_layout()
plt.savefig("Compare/rmse.png",dpi=200)
plt.close()
)PY";

    // Throughput Graph
    py << R"PY(
plt.figure(figsize=(7,5))
plt.bar(labels, throughput)
plt.title("Throughput Comparison")
plt.ylabel("Million Grid Updates / sec")
plt.xlabel("Solver")
plt.xticks(rotation=20, ha='right')
plt.grid(axis='y')
plt.tight_layout()
plt.savefig("Compare/throughput.png",dpi=200)
plt.close()
)PY";

    py.close();

    int ret = std::system("python _heat_plots.py");
    if (ret != 0) {
        ret = std::system("python3 _heat_plots.py");
    }
    if(ret!=0)
        std::cerr<<"WARNING: Graph generation failed. Install Python + matplotlib.\n";

    std::remove("_heat_plots.py");

    std::cout << "Graphs saved:\n";
    std::cout << "  execution_time.png\n";
    std::cout << "  speedup.png\n";
    std::cout << "  efficiency.png\n";
    std::cout << "  rmse.png\n";
    std::cout << "  throughput.png\n";
}

int main(int argc, char* argv[]) {
    int requested_nx = 500;
    int requested_ny = 500;
    if (argc > 1) requested_nx = std::atoi(argv[1]);
    if (argc > 2) requested_ny = std::atoi(argv[2]);
    configure_grid(requested_nx, requested_ny);

    if ((RX + RY) > 0.5) {
        std::cerr << "ERROR: Unstable! rx+ry = " << (RX+RY) << " > 0.5\n";
        return 1;
    }

    int max_threads = omp_get_max_threads();

    std::cout << "\n";
    sep('=');
    std::cout << "  STARTING COMPARISON RUN\n";
    std::cout << "  Max available OpenMP threads: " << max_threads << "\n";
    sep('=');
    std::cout << "\n";

    std::vector<Result> results;

    // Serial
    std::cout << "  [1] Running Serial solver...\n";
    Result serial = run_serial();
    results.push_back(serial);
    std::cout << "      Done: " << std::fixed << std::setprecision(2)
              << serial.exec_ms << " ms\n\n";

    // OpenMP: 1, 2, 4, max_threads
    std::vector<int> tcounts = { 1, 2, 4, max_threads };
    std::sort(tcounts.begin(), tcounts.end());
    tcounts.erase(std::unique(tcounts.begin(), tcounts.end()), tcounts.end());

    int idx = 2;
    for (int nt : tcounts) {
        std::cout << "  [" << idx++ << "] Running OpenMP ("
                  << nt << " thread" << (nt > 1 ? "s" : "") << ")...\n";
        Result r = run_omp(nt, serial.exec_ms);
        results.push_back(r);
        std::cout << "      Done: " << std::fixed << std::setprecision(2)
                  << r.exec_ms << " ms  |  Speedup: "
                  << std::setprecision(3) << r.speedup << "x\n\n";
    }

    Result mpi;
    std::cout << "  [" << idx++ << "] Loading MPI result...\n";
    if (read_mpi_summary(mpi, serial.exec_ms)) {
        results.push_back(mpi);
        std::cout << "      Done: " << std::fixed << std::setprecision(2)
                  << mpi.exec_ms << " ms  |  Speedup: "
                  << std::setprecision(3) << mpi.speedup << "x"
                  << "  |  MPI ranks: " << mpi.mpi_ranks << "\n\n";
    } else {
        std::cout << "      Skipped: run MPI/heat2d_mpi first to create "
                  << "MPI/summary_2d_mpi.csv\n\n";
    }

    Result hybrid;
    std::cout << "  [" << idx++ << "] Loading Hybrid MPI+OpenMP result...\n";
    if (read_hybrid_summary(hybrid, serial.exec_ms)) {
        results.push_back(hybrid);
        std::cout << "      Done: " << std::fixed << std::setprecision(2)
                  << hybrid.exec_ms << " ms  |  Speedup: "
                  << std::setprecision(3) << hybrid.speedup << "x"
                  << "  |  MPI ranks: " << hybrid.mpi_ranks
                  << "  |  OMP threads/rank: " << hybrid.omp_threads << "\n\n";
    } else {
        std::cout << "      Skipped: run Hybrid/heat2d_hybrid first to create "
                  << "Hybrid/summary_2d_hybrid.csv\n\n";
    }

    Result cuda;
    std::cout << "  [" << idx++ << "] Loading CUDA result...\n";
    if (read_cuda_summary(cuda, serial.exec_ms)) {
        results.push_back(cuda);
        std::cout << "      Done: " << std::fixed << std::setprecision(2)
                  << cuda.exec_ms << " ms  |  Speedup: "
                  << std::setprecision(3) << cuda.speedup << "x\n\n";
    } else {
        std::cout << "      Skipped: run cuda/heat2d_cuda first to create "
                  << "cuda/summary_2d_cuda.csv\n\n";
    }

    // Print console table
    std::cout << "\n";
    print_table(results);
    std::cout << "\n";
    print_accuracy(results);
    std::cout << "\n";

    double best_sp = 1.0; std::string best_lbl = "Serial";
    for (const auto& r : results)
        if (r.speedup > best_sp) { best_sp = r.speedup; best_lbl = r.label; }
    sep('-', 75);
    std::cout << "  BEST: " << best_lbl << "  ->  Speedup = "
              << std::fixed << std::setprecision(3) << best_sp << "x\n";
    sep('-', 75);
    std::cout << "\n";

    // Save CSV
    save_csv(results, "Compare/comparison_results.csv");

    // Generate PNG from actual results
    std::cout << "  Generating graph from actual results...\n";
    generate_graphs(results);
    std::cout << "\n";

    return 0;
}
