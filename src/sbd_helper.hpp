/*
# This code is part of Qiskit.
#
# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
*/

#ifndef SBD_HELPER_HPP_
#define SBD_HELPER_HPP_

#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#ifdef _MSC_VER
#include <windows.h>
#else
#include <unistd.h>
#endif

#define USE_MATH_DEFINES
#include <cmath>

#include "mpi.h"
#include "sbd/sbd.h"

struct SBD {
    int task_comm_size = 1;
    int adet_comm_size = 1;
    int bdet_comm_size = 1;
    int h_comm_size    = 1;

    int max_it   = 1;
    int max_nb   = 10;
    double eps   = 1.0e-12;
    double max_time = 600.0;
    int init     = 0;

    double threshold = 0.0;

    double energy_target   = -108.972943488072397;
    double energy_variance = 1.0;

    std::string adetfile    = "AlphaDets.bin";
    std::string fcidumpfile = "";
};

inline SBD generate_sbd_data(int argc, char *argv[])
{
    SBD sbd;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];

        if (arg == "--fcidump" && i + 1 < argc) {
            sbd.fcidumpfile = std::string(argv[++i]);
        } else if (arg == "--iteration" && i + 1 < argc) {
            sbd.max_it = std::atoi(argv[++i]);
        } else if (arg == "--block" && i + 1 < argc) {
            sbd.max_nb = std::atoi(argv[++i]);
        } else if (arg == "--tolerance" && i + 1 < argc) {
            sbd.eps = std::atof(argv[++i]);
        } else if (arg == "--max_time" && i + 1 < argc) {
            sbd.max_time = std::atof(argv[++i]);
        } else if (arg == "--adet_comm_size" && i + 1 < argc) {
            sbd.adet_comm_size = std::atoi(argv[++i]);
        } else if (arg == "--bdet_comm_size" && i + 1 < argc) {
            sbd.bdet_comm_size = std::atoi(argv[++i]);
        } else if (arg == "--task_comm_size" && i + 1 < argc) {
            sbd.task_comm_size = std::atoi(argv[++i]);
        } else if (arg == "--energy_target" && i + 1 < argc) {
            sbd.energy_target = std::atof(argv[++i]);
        } else if (arg == "--energy_variance" && i + 1 < argc) {
            sbd.energy_variance = std::atof(argv[++i]);
        }
    }
    return sbd;
}

// energy, occupancy
inline std::tuple<double, std::vector<double>>
sbd_main(const MPI_Comm &comm, const SBD &sbd_data)
{
    double E = 0.0;

    int mpi_rank = 0;
    int mpi_size = 1;
    MPI_Comm_rank(comm, &mpi_rank);
    MPI_Comm_size(comm, &mpi_size);

    const int task_comm_size = sbd_data.task_comm_size;
    const int adet_comm_size = sbd_data.adet_comm_size;
    const int bdet_comm_size = sbd_data.bdet_comm_size;

    int L = 0;
    int N = 0;

    const int max_it   = sbd_data.max_it;
    const int max_nb   = sbd_data.max_nb;
    const double eps   = sbd_data.eps;
    const double max_time = sbd_data.max_time;
    const int init     = sbd_data.init;

    const double energy_target   = sbd_data.energy_target;
    const double energy_variance = sbd_data.energy_variance;

    const size_t bit_length      = SBD_BIT_LENGTH;
    const std::string adetfile   = sbd_data.adetfile;
    const std::string fcidumpfile= sbd_data.fcidumpfile;

    int base_comm_size = adet_comm_size * bdet_comm_size * task_comm_size;
    int h_comm_size    = mpi_size / base_comm_size;

    if (mpi_size != base_comm_size * h_comm_size) {
        throw std::invalid_argument("communicator size is not appropriate");
    }

    // ---- Load FCIDUMP on rank 0 and broadcast ----
    sbd::FCIDump fcidump;
    if (mpi_rank == 0) {
        fcidump = sbd::LoadFCIDump(fcidumpfile);
    }
    sbd::MpiBcast(fcidump, 0, comm);

    double I0;
    sbd::oneInt<double> I1;
    sbd::twoInt<double> I2;
    sbd::SetupIntegrals(fcidump, L, N, I0, I1, I2);

    // ---- Prepare determinants ----
    std::vector<std::vector<size_t>> adet;
    std::vector<std::vector<size_t>> bdet;

    if (mpi_rank == 0) {
        adet = sbd::DecodeAlphaDets(adetfile, L);
        sbd::change_bitlength(1, adet, bit_length);
        sbd::sort_bitarray(adet);
    }

    sbd::MpiBcast(adet, 0, comm);
    bdet = adet;

    // ---- Setup helpers and communicators ----
    std::vector<sbd::TaskHelpers> helper;
    std::vector<std::vector<size_t>> sharedMemory;
    MPI_Comm h_comm;
    MPI_Comm b_comm;
    MPI_Comm t_comm;

    sbd::TaskCommunicator(
        comm, h_comm_size, adet_comm_size, bdet_comm_size, task_comm_size,
        h_comm, b_comm, t_comm
    );

    sbd::MakeHelpers(
        adet, bdet, bit_length, L, helper, sharedMemory,
        h_comm, b_comm, t_comm, adet_comm_size, bdet_comm_size
    );
    sbd::RemakeHelpers(
        adet, bdet, bit_length, L, helper, sharedMemory,
        h_comm, b_comm, t_comm, adet_comm_size, bdet_comm_size
    );

    int mpi_rank_h; MPI_Comm_rank(h_comm, &mpi_rank_h);
    int mpi_rank_b; MPI_Comm_rank(b_comm, &mpi_rank_b);
    int mpi_rank_t; MPI_Comm_rank(t_comm, &mpi_rank_t);
    int mpi_size_t; MPI_Comm_size(t_comm, &mpi_size_t);
    int mpi_size_h; MPI_Comm_size(h_comm, &mpi_size_h);
    (void)mpi_rank_b; (void)mpi_rank_t; (void)mpi_size_t; (void)mpi_size_h; // silence warnings

    // ---- Initialize/load wavefunction ----
    std::vector<double> W;
    sbd::BasisInitVector(
        W, adet, bdet, adet_comm_size, bdet_comm_size,
        h_comm, b_comm, t_comm, init
    );

    // ---- Diagonalization (Davidson) ----
    std::vector<double> hii;
    auto time_start_diag = std::chrono::high_resolution_clock::now();
    sbd::makeQChamDiagTerms(
        adet, bdet, bit_length, L, helper, I0, I1, I2,
        hii, h_comm, b_comm, t_comm
    );
    sbd::Davidson(
        hii, W, adet, bdet, bit_length, static_cast<size_t>(L),
        adet_comm_size, bdet_comm_size, helper, I0, I1, I2,
        h_comm, b_comm, t_comm, max_it, max_nb, eps, max_time
    );
    auto time_end_diag = std::chrono::high_resolution_clock::now();
    auto elapsed_diag_count =
        std::chrono::duration_cast<std::chrono::microseconds>(
            time_end_diag - time_start_diag
        ).count();
    double elapsed_diag = 1.0e-6 * static_cast<double>(elapsed_diag_count);
    if (mpi_rank == 0) {
        std::cout << " Elapsed time for diagonalization "
                  << elapsed_diag << " (sec)" << std::endl;
    }

    // ---- Expectation value <W|H|W> ----
    std::vector<double> C(W.size(), 0.0);
    sbd::mult(
        hii, W, C, adet, bdet, bit_length, static_cast<size_t>(L),
        adet_comm_size, bdet_comm_size, helper, I0, I1, I2,
        h_comm, b_comm, t_comm
    );
    sbd::InnerProduct(W, C, E, b_comm);

    // IMPORTANT: do NOT zero out E anymore.
    // Instead, just print diagnostics if a target was provided.
    if (mpi_rank == 0) {
        std::cout.precision(16);
        if (energy_target != 0.0) {
            double delta = std::abs(E - energy_target);
            std::cout << " Energy = " << E
                      << "  (target=" << energy_target
                      << ", |Δ|=" << delta
                      << ", variance=" << energy_variance << ")"
                      << std::endl;
        } else {
            std::cout << " Energy = " << E << std::endl;
        }
    }

    // ---- Single-particle occupation density ----
    int p_size = mpi_size_t * mpi_size_h;
    int p_rank = mpi_rank_h * mpi_size_t + mpi_rank_t;

    size_t o_start = 0, o_end = static_cast<size_t>(L);
    sbd::get_mpi_range(p_size, p_rank, o_start, o_end);
    size_t o_size = o_end - o_start;

    std::vector<int> oIdx(o_size);
    std::iota(oIdx.begin(), oIdx.end(), static_cast<int>(o_start));

    std::vector<double> res_density;
    sbd::OccupationDensity(
        oIdx, W, adet, bdet, bit_length,
        adet_comm_size, bdet_comm_size, b_comm, res_density
    );

    std::vector<double> density_rank(static_cast<size_t>(2 * L), 0.0);
    std::vector<double> density_group(static_cast<size_t>(2 * L), 0.0);
    std::vector<double> density(static_cast<size_t>(2 * L), 0.0);

    for (size_t io = o_start; io < o_end; ++io) {
        density_rank[2 * io]     = res_density[2 * (io - o_start)];
        density_rank[2 * io + 1] = res_density[2 * (io - o_start) + 1];
    }

    MPI_Allreduce(
        density_rank.data(), density_group.data(), 2 * L,
        MPI_DOUBLE, MPI_SUM, t_comm
    );
    MPI_Allreduce(
        density_group.data(), density.data(), 2 * L,
        MPI_DOUBLE, MPI_SUM, h_comm
    );

    FreeHelpers(helper);
    return {E, density};
}

#endif  // SBD_HELPER_HPP_
