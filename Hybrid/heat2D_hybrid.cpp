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