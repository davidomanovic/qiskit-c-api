/*
# This code is part of Qiskit.
# Licensed under the Apache License, Version 2.0.
*/

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <optional>
#include <regex>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>
#include "boost/dynamic_bitset.hpp"

// Project / addon headers available in this repo
#include "qiskit/addon/sqd/configuration_recovery.hpp"
#include "qiskit/addon/sqd/subsampling.hpp"
#include "sbd_helper.hpp"   // generate_sbd_data, sbd_main, write_alphadets_file
#include "sqd_helper.hpp"   // generate_sqd_data, log

// -------- CLI helper --------
static std::string get_flag_arg(int argc, char** argv, const std::string& flag) {
    for (int i = 1; i + 1 < argc; ++i)
        if (flag == argv[i]) return std::string(argv[i + 1]);
    return std::string();
}

// -------- counts loader: "<bitstring> <count>" per line --------
static std::unordered_map<std::string, std::uint64_t>
read_counts_file_to_map(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Could not open counts file: " + path);
    std::unordered_map<std::string, std::uint64_t> out;
    std::string bits; std::uint64_t cnt = 0;
    while (in >> bits >> cnt) out[bits] += cnt;
    if (out.empty()) throw std::runtime_error("Counts file is empty: " + path);
    return out;
}

// -------- normalize counts -> probabilities --------
static std::unordered_map<std::string, double>
normalize_counts_dict(const std::unordered_map<std::string, std::uint64_t>& counts) {
    if (counts.empty()) return {};
    std::uint64_t total = 0;
    for (const auto& kv : counts) total += kv.second;
    std::unordered_map<std::string, double> probs;
    probs.reserve(counts.size());
    for (const auto& kv : counts)
        probs[kv.first] = static_cast<double>(kv.second) / static_cast<double>(total);
    return probs;
}

// -------- convert map -> (bitstrings[], probs[]) --------
static std::pair<std::vector<boost::dynamic_bitset<>>, std::vector<double>>
counts_to_arrays(const std::unordered_map<std::string, std::uint64_t>& counts) {
    std::vector<boost::dynamic_bitset<>> bitstrings;
    std::vector<double> probs;
    if (counts.empty()) return {bitstrings, probs};
    auto prob_map = normalize_counts_dict(counts);
    bitstrings.reserve(prob_map.size());
    probs.reserve(prob_map.size());
    for (const auto& kv : prob_map) {
        bitstrings.emplace_back(kv.first);   // ctor from string
        probs.emplace_back(kv.second);
    }
    return {bitstrings, probs};
}

// -------- parse NORB/NELEC/MS2 from FCIDUMP header --------
static std::tuple<uint64_t, uint64_t, int64_t>
parse_fcidump_header(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Could not open FCIDUMP: " + path);
    std::string header, line;
    while (std::getline(in, line)) {
        if (line.find("&FCI") != std::string::npos || line.find("&FC") != std::string::npos) {
            header = line;
            while (header.find("/") == std::string::npos && std::getline(in, line))
                header += " " + line;
            break;
        }
    }
    if (header.empty())
        throw std::runtime_error("FCIDUMP header (&FCI) not found: " + path);

    uint64_t norb = 0, nelec = 0;
    int64_t ms2 = 0;
    std::regex re(R"(NORB\s*=\s*([0-9]+)|NELEC\s*=\s*([0-9]+)|MS2\s*=\s*(-?[0-9]+))");
    for (auto it = std::sregex_iterator(header.begin(), header.end(), re);
         it != std::sregex_iterator(); ++it) {
        const auto& m = *it;
        if (m[1].matched) norb  = static_cast<uint64_t>(std::stoull(m[1].str()));
        if (m[2].matched) nelec = static_cast<uint64_t>(std::stoull(m[2].str()));
        if (m[3].matched) ms2   = static_cast<int64_t>(std::stoll(m[3].str()));
    }
    if (norb == 0) throw std::runtime_error("FCIDUMP missing NORB in header: " + path);
    return {norb, nelec, ms2};
}

// -------- local helpers to avoid missing symbols --------
static void write_occupancies_json(const SQD& sqd_data,
                                   const std::array<std::vector<double>,2>& occ,
                                   int iter) {
    nlohmann::json j;
    j["alpha"] = occ[0];
    j["beta"]  = occ[1];
    std::string fname =
        "occupancies_" + sqd_data.run_id + "_iter" + std::to_string(iter) + ".json";
    std::ofstream o(fname);
    o << j.dump(2) << std::endl;
}

static void bcast_occupancies(MPI_Comm comm,
                              std::array<std::vector<double>,2>& occ) {
    int rank = 0; MPI_Comm_rank(comm, &rank);
    std::uint64_t sizes[2] = {
        static_cast<std::uint64_t>(occ[0].size()),
        static_cast<std::uint64_t>(occ[1].size())
    };
    if (rank != 0) sizes[0] = sizes[1] = 0ULL;
    MPI_Bcast(sizes, 2, MPI_UNSIGNED_LONG_LONG, 0, comm);
    if (rank != 0) {
        occ[0].resize(static_cast<std::size_t>(sizes[0]));
        occ[1].resize(static_cast<std::size_t>(sizes[1]));
    }
    if (sizes[0]) MPI_Bcast(occ[0].data(), static_cast<int>(sizes[0]), MPI_DOUBLE, 0, comm);
    if (sizes[1]) MPI_Bcast(occ[1].data(), static_cast<int>(sizes[1]), MPI_DOUBLE, 0, comm);
}

int main(int argc, char** argv) {
    try {
        // --- MPI init ---
        int provided = 0;
        if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS) {
            std::cerr << "MPI_Init_thread failed\n";
            return 1;
        }

        // --- required flags ---
        const std::string counts_file = get_flag_arg(argc, argv, "--counts_file");
        const std::string fcidump_path = get_flag_arg(argc, argv, "--fcidump");
        if (counts_file.empty()) {
            std::cerr << "ERROR: --counts_file <path> is required.\n";
            MPI_Abort(MPI_COMM_WORLD, 2); return 2;
        }
        if (fcidump_path.empty()) {
            std::cerr << "ERROR: --fcidump <path> is required.\n";
            MPI_Abort(MPI_COMM_WORLD, 2); return 2;
        }

        // --- build helper configs from argv ---
        SBD diag_data = generate_sbd_data(argc, argv);
        SQD sqd_data = generate_sqd_data(argc, argv);
        sqd_data.comm = MPI_COMM_WORLD;
        MPI_Comm_rank(sqd_data.comm, &sqd_data.mpi_rank);
        MPI_Comm_size(sqd_data.comm, &sqd_data.mpi_size);

        // share run_id string
        int runid_len = (sqd_data.mpi_rank == 0) ? static_cast<int>(sqd_data.run_id.size()) : 0;
        MPI_Bcast(&runid_len, 1, MPI_INT, 0, sqd_data.comm);
        if (sqd_data.mpi_rank != 0) sqd_data.run_id.resize(runid_len);
        if (runid_len) MPI_Bcast(sqd_data.run_id.data(), runid_len, MPI_CHAR, 0, sqd_data.comm);

        // RNGs
        std::mt19937 rng(1234);
        std::mt19937 rc_rng(rng());

        // --- rank 0: read counts, expand to arrays ---
        std::vector<boost::dynamic_bitset<>> bitstring_matrix_full;
        std::vector<double> probs_arr_full;
        uint64_t norb_counts = 0;
        if (sqd_data.mpi_rank == 0) {
            auto counts = read_counts_file_to_map(counts_file);
            log(sqd_data, std::vector<std::string>{
                "Loaded counts from: ", counts_file,
                " (unique bitstrings=", std::to_string(counts.size()), ")"
            });
            auto arrays = counts_to_arrays(counts);
            bitstring_matrix_full = std::move(arrays.first);
            probs_arr_full        = std::move(arrays.second);

            if (bitstring_matrix_full.empty())
                throw std::runtime_error("Counts expanded to zero entries.");
            const std::size_t bitlen = bitstring_matrix_full[0].size();
            if ((bitlen % 2) != 0)
                throw std::runtime_error("Bitstring length must be even (alpha+beta).");
            norb_counts = static_cast<uint64_t>(bitlen / 2);
        }
        MPI_Bcast(&norb_counts, 1, MPI_UNSIGNED_LONG_LONG, 0, sqd_data.comm);

        // --- parse FCIDUMP header and compute (na, nb) ---
        uint64_t norb_fd = 0, nelec_fd = 0; int64_t ms2_fd = 0;
        if (sqd_data.mpi_rank == 0) {
            std::tie(norb_fd, nelec_fd, ms2_fd) = parse_fcidump_header(fcidump_path);
            if (norb_fd != norb_counts) {
                std::cerr << "WARNING: FCIDUMP NORB(" << norb_fd
                          << ") != counts-derived NORB(" << norb_counts
                          << "). Proceeding with counts-derived value.\n";
            }
        }
        MPI_Bcast(&nelec_fd, 1, MPI_UNSIGNED_LONG_LONG, 0, sqd_data.comm);
        MPI_Bcast(&ms2_fd,   1, MPI_LONG_LONG,        0, sqd_data.comm);

        const uint64_t norb = norb_counts;
        const uint64_t na = (nelec_fd || ms2_fd) ? static_cast<uint64_t>((static_cast<int64_t>(nelec_fd) + ms2_fd)/2) : 0ULL;
        const uint64_t nb = (nelec_fd || ms2_fd) ? static_cast<uint64_t>((static_cast<int64_t>(nelec_fd) - ms2_fd)/2) : 0ULL;
        if (na + nb == 0) {
            std::cerr << "WARNING: FCIDUMP missing NELEC/MS2; continuing with NELEC=0.\n";
        }

        // --- broadcast bitstrings and probabilities to all ranks ---
        std::uint64_t n_items = (sqd_data.mpi_rank == 0) ? bitstring_matrix_full.size() : 0ULL;
        MPI_Bcast(&n_items, 1, MPI_UNSIGNED_LONG_LONG, 0, sqd_data.comm);

        std::vector<std::string> bitstrings_as_text;
        if (sqd_data.mpi_rank == 0) {
            bitstrings_as_text.reserve(n_items);
            for (const auto& bs : bitstring_matrix_full) {
                std::string s; boost::to_string(bs, s);
                bitstrings_as_text.emplace_back(std::move(s));
            }
        }
        std::vector<boost::dynamic_bitset<>> bitstring_matrix_local(n_items);
        for (std::uint64_t i = 0; i < n_items; ++i) {
            std::uint64_t len = (sqd_data.mpi_rank == 0)
                ? static_cast<std::uint64_t>(bitstrings_as_text[i].size()) : 0ULL;
            MPI_Bcast(&len, 1, MPI_UNSIGNED_LONG_LONG, 0, sqd_data.comm);
            std::string tmp;
            if (sqd_data.mpi_rank == 0) tmp = bitstrings_as_text[i];
            tmp.resize(static_cast<std::size_t>(len));
            MPI_Bcast(tmp.data(), static_cast<int>(len), MPI_CHAR, 0, sqd_data.comm);
            if (sqd_data.mpi_rank != 0) bitstring_matrix_local[i] = boost::dynamic_bitset<>(tmp);
        }

        std::uint64_t n_probs = (sqd_data.mpi_rank == 0)
            ? static_cast<std::uint64_t>(probs_arr_full.size()) : 0ULL;
        MPI_Bcast(&n_probs, 1, MPI_UNSIGNED_LONG_LONG, 0, sqd_data.comm);
        std::vector<double> probs_arr_local(n_probs);
        if (sqd_data.mpi_rank == 0) {
            MPI_Bcast(probs_arr_full.data(), static_cast<int>(n_probs), MPI_DOUBLE, 0, sqd_data.comm);
        } else {
            MPI_Bcast(probs_arr_local.data(), static_cast<int>(n_probs), MPI_DOUBLE, 0, sqd_data.comm);
        }
        if (sqd_data.mpi_rank != 0) {
            bitstring_matrix_full = std::move(bitstring_matrix_local);
            probs_arr_full        = std::move(probs_arr_local);
        }

        // --- SQD recovery + SBD loop ---
        const uint64_t samples_per_batch = sqd_data.samples_per_batch;
        const int n_recovery = static_cast<int>(sqd_data.n_recovery);

        // Start with flat 0.5 prior of correct size (robust across systems)
        std::array<std::vector<double>,2> latest_occupancies, initial_occupancies;
        initial_occupancies[0].assign(norb, 0.5);
        initial_occupancies[1].assign(norb, 0.5);
        latest_occupancies = initial_occupancies;

        for (int irec = 0; irec < n_recovery; ++irec) {
            if (sqd_data.mpi_rank == 0) {
                log(sqd_data, std::vector<std::string>{
                    "Recovery iteration: ", std::to_string(irec)
                });
            }

            std::vector<boost::dynamic_bitset<>> recovered_bs;
            std::vector<double> recovered_probs;

            if (sqd_data.mpi_rank == 0) {
                auto recovered = Qiskit::addon::sqd::recover_configurations(
                    bitstring_matrix_full, probs_arr_full, latest_occupancies,
                    {na, nb}, rc_rng
                );
                recovered_bs    = std::move(recovered.first);
                recovered_probs = std::move(recovered.second);

                log(sqd_data, std::vector<std::string>{
                    "Recovered bitstrings: ", std::to_string(recovered_bs.size())
                });

                // Subsample one batch for SBD
                std::vector<boost::dynamic_bitset<>> batch;
                Qiskit::addon::sqd::subsample(
                    batch, recovered_bs, recovered_probs, samples_per_batch, rng
                );

                // Write alpha-determinants file for SBD
                diag_data.adetfile = write_alphadets_file(
                    sqd_data, norb, na, batch, sqd_data.samples_per_batch * 2, irec
                );
            }

            // Diagonalize (SBD). Returns energy and interleaved occupancies [a0,b0,a1,b1,...]
            auto [energy_sci, occs_batch] = sbd_main(sqd_data.comm, diag_data);
            double energy_root = energy_sci;
            MPI_Bcast(&energy_root, 1, MPI_DOUBLE, /*root=*/0, sqd_data.comm);
            
            if (sqd_data.mpi_rank == 0) {
                log(sqd_data, std::vector<std::string>{
                    "energy: ", std::to_string(energy_root)
                });
            }

            if (sqd_data.mpi_rank == 0) {
                const std::size_t norb_sz = latest_occupancies[0].size();
                if (occs_batch.size() != 2 * norb_sz) {
                    std::cerr << "Unexpected occupancies size from SBD\n";
                    MPI_Abort(sqd_data.comm, 1);
                    return 1;
                }
                for (std::size_t j = 0; j < norb_sz; ++j) {
                    latest_occupancies[0][j] = occs_batch[2*j];
                    latest_occupancies[1][j] = occs_batch[2*j+1];
                }
                write_occupancies_json(sqd_data, latest_occupancies, irec);
            }
            bcast_occupancies(sqd_data.comm, latest_occupancies);
        }

        MPI_Finalize();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Unhandled exception: " << e.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    } catch (...) {
        std::cerr << "Unknown exception\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}
