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

#ifndef SQD_HELPER_HPP_
#define SQD_HELPER_HPP_

#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <set>
#include <stdexcept>

#include <unistd.h>

#define USE_MATH_DEFINES
#include <cmath>

#include "boost/dynamic_bitset.hpp"

#include "mpi.h"
#include "sbd/sbd.h"

std::string get_time(bool compact = false)
{
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    if (compact)
        ss << std::put_time(
            std::localtime(&in_time_t),
            "%Y%m%d%H%M%S"
        ); // Format: YYYYMMDDHHMMSS
    else
        ss << std::put_time(
            std::localtime(&in_time_t),
            "%Y-%m-%d %H:%M:%S"
        ); // Format: YYYY-MM-DD HH:MM:SS
    return ss.str();
}

std::pair<std::vector<uint64_t>, std::vector<uint64_t>> bitstring_matrix_to_ci_strs(
    const std::vector<boost::dynamic_bitset<>> &bitstring_matrix,
    bool open_shell = false
)
{
    size_t num_configs = bitstring_matrix.size();
    size_t norb = bitstring_matrix[0].size() / 2;

    std::vector<uint64_t> ci_str_left(num_configs, 0);
    std::vector<uint64_t> ci_str_right(num_configs, 0);

    for (size_t config = 0; config < num_configs; ++config) {
        const auto &row = bitstring_matrix[config];

        for (size_t i = 0; i < norb; ++i) {
            ci_str_right[config] ^= static_cast<std::uint64_t>(row[i]) << i;
        }
        for (size_t i = 0; i < norb; ++i) {
            ci_str_left[config] ^= static_cast<std::uint64_t>(row[i + norb]) << i;
        }
    }

    std::set<uint64_t> unique_ci_str_left(ci_str_left.begin(), ci_str_left.end());
    std::set<uint64_t> unique_ci_str_right(ci_str_right.begin(), ci_str_right.end());

    if (!open_shell) {
        std::set<uint64_t> combined_set;
        combined_set.insert(unique_ci_str_left.begin(), unique_ci_str_left.end());
        combined_set.insert(unique_ci_str_right.begin(), unique_ci_str_right.end());
        unique_ci_str_left = unique_ci_str_right = combined_set;
    }

    std::vector<uint64_t> result_left(
        unique_ci_str_left.begin(), unique_ci_str_left.end()
    );
    std::vector<uint64_t> result_right(
        unique_ci_str_right.begin(), unique_ci_str_right.end()
    );

    return {result_right, result_left};
}

struct SQD {
    std::string date_str = get_time(true);
    std::string run_id = date_str;
    uint64_t n_recovery = 3;           // number of configuration recovery iterations
    uint64_t samples_per_batch = 1000; // number of samples per batch
    bool verbose = false;              // print messages to stdout
    bool with_hf = true;               // use Hartree-Fock as a reference state

    std::string backend_name = "";
    uint64_t num_shots = 10000;

    MPI_Comm comm;
    int mpi_rank;
    int mpi_size;

    std::string summary()
    {
        std::stringstream ss;
        ss << "# date: " << date_str << std::endl;
        ss << "# run_id:" << run_id << std::endl;
        ss << "# n_recovery: " << n_recovery << std::endl;
        ss << "# samples_per_batch: " << samples_per_batch << std::endl;
        ss << "# backend_name: " << backend_name << std::endl;
        ss << "# num_shots: " << num_shots << std::endl;
        return ss.str();
    }
};

void log(const SQD &sqd_data, const std::vector<std::string> &messages)
{
    if (sqd_data.verbose) {
        std::cout << get_time();
        std::cout << ": ";
        for (auto &msg : messages)
            std::cout << msg;
        std::cout << std::endl;
    }
}

void error(const SQD &sqd_data, const std::vector<std::string> &messages)
{
    if (sqd_data.verbose) {
        std::cerr << ": ";
        for (auto &msg : messages)
            std::cerr << msg;
        std::cerr << std::endl;
    }
}

SQD generate_sqd_data(int argc, char *argv[])
{
    SQD sqd;
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--recovery") {
            sqd.n_recovery = std::stoi(argv[i + 1]);
            i++;
        }
        if (std::string(argv[i]) == "--number_of_samples") {
            sqd.samples_per_batch = std::stoi(argv[i + 1]);
            i++;
        }
        if (std::string(argv[i]) == "--backend_name") {
            sqd.backend_name = std::string(argv[i + 1]);
            i++;
        }
        if (std::string(argv[i]) == "--num_shots") {
            sqd.num_shots = std::stoi(argv[i + 1]);
            i++;
        }
        if (std::string(argv[i]) == "-v") {
            sqd.verbose = true;
        }
    }
    return sqd;
}

std::vector<uint8_t>
integer_to_bytes(uint64_t n, int norb) // NOLINT(bugprone-easily-swappable-parameters)
{
    int num_bytes = (norb + 7) / 8;
    std::vector<uint8_t> result(num_bytes);

    for (int i = num_bytes - 1; i >= 0; --i) {
        result[i] = static_cast<uint8_t>(n & 0xFF);
        n >>= 8;
    }

    return result;
}

std::vector<std::vector<uint8_t>>
ci_strs_to_bytes(const std::vector<uint64_t> &ci_strs, int norb)
{
    std::vector<std::vector<uint8_t>> bytes;
    bytes.reserve(ci_strs.size());
    for (uint64_t ci_str : ci_strs) {
        bytes.push_back(integer_to_bytes(ci_str, norb));
    }
    return bytes;
}

std::vector<uint64_t> //
get_unique_ci_strs_with_HF(
    const SQD &sqd_data, const std::vector<uint64_t> &left_ci_strs,
    const std::vector<uint64_t> &right_ci_strs, const size_t num_elec
)
{
    std::set<uint64_t> unique_set;
    if (sqd_data.with_hf)
        unique_set.insert(((1ULL << num_elec) - 1));
    unique_set.insert(left_ci_strs.begin(), left_ci_strs.end());
    unique_set.insert(right_ci_strs.begin(), right_ci_strs.end());

    auto ret = std::vector<uint64_t>(unique_set.begin(), unique_set.end());
    std::sort(ret.begin(), ret.end());
    return std::move(ret);
}

void write_bytestrings_to_file(
    const std::vector<std::vector<uint8_t>> &byte_strings, const std::string &filename
)
{
    std::ofstream output_file(filename, std::ios::binary);

    if (!output_file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return; // Or throw an exception
    }

    for (const auto &byte_string : byte_strings) {
        output_file.write(
            reinterpret_cast<const char *>(byte_string.data()),
            static_cast<std::streamsize>(byte_string.size())
        );
    }

    output_file.close();
}

std::string write_alphadets_file(
    const SQD &sqd_data,
    const size_t norb,     // NOLINT(bugprone-easily-swappable-parameters)
    const size_t num_elec, // NOLINT(bugprone-easily-swappable-parameters)
    const std::vector<boost::dynamic_bitset<>> &batch,
    const size_t
        maximum_numbers_of_ctrs, // NOLINT(bugprone-easily-swappable-parameters)
    const size_t i_recovery
) // NOLINT(bugprone-easily-swappable-parameters)
{
    log(sqd_data, {"number of items in a batch: ", std::to_string(batch.size())});
    bool open_shell = false;

    auto ci_strs = bitstring_matrix_to_ci_strs(batch, open_shell);
    log(sqd_data,
        {"number of items in left ci_strs:", std::to_string(ci_strs.first.size())});
    log(sqd_data,
        {"number of items in right ci_strs:", std::to_string(ci_strs.second.size())});

    auto unique_ci_strs =
        get_unique_ci_strs_with_HF(sqd_data, ci_strs.first, ci_strs.second, num_elec);
    if (unique_ci_strs.size() < maximum_numbers_of_ctrs) {
        log(sqd_data,
            {"number of unique ci_strs:", std::to_string(unique_ci_strs.size())});
    } else {
        size_t truncated = unique_ci_strs.size() - maximum_numbers_of_ctrs;
        unique_ci_strs.resize(maximum_numbers_of_ctrs);
        log(sqd_data,
            {"number of unique ci_strs:", std::to_string(unique_ci_strs.size()),
             ", truncated:", std::to_string(truncated)});
    }
    auto bytestrings = ci_strs_to_bytes(unique_ci_strs, static_cast<int>(norb));
    std::string alphadets_bin_file =
        "AlphaDets_" + sqd_data.run_id + "_" + std::to_string(i_recovery) + "_cpp.bin";
    write_bytestrings_to_file(bytestrings, alphadets_bin_file);
    return alphadets_bin_file;
}

#endif
