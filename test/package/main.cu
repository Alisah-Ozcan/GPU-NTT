// Copyright 2023-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

// Compares the forward NTT on the GPU with the CPU NTT of an installed
// GPU-NTT. Needs a GPU to run; building and linking it already checks the
// installed package.

#include <gpuntt/ntt_merge/ntt.cuh>
#include <gpuntt/ntt_merge/ntt_cpu.cuh>
#include <cstdlib>
#include <iostream>
#include <vector>

using namespace gpuntt;

int main()
{
    constexpr int logn = 12;
    NTTParameters<Data64> parameters(logn, ReductionPolynomial::X_N_minus);
    NTTCPU<Data64> generator(parameters);

    std::vector<Data64> input(parameters.n);
    for (int i = 0; i < parameters.n; i++)
        input[i] = static_cast<Data64>(i) % parameters.modulus.value;
    std::vector<Data64> expected = generator.ntt(input);

    Data64* data;
    GPUNTT_CUDA_CHECK(cudaMalloc(&data, parameters.n * sizeof(Data64)));
    GPUNTT_CUDA_CHECK(cudaMemcpy(data, input.data(),
                                 parameters.n * sizeof(Data64),
                                 cudaMemcpyHostToDevice));

    std::vector<Data64> table = parameters.gpu_root_of_unity_table_generator(
        parameters.forward_root_of_unity_table);
    Data64* table_device;
    GPUNTT_CUDA_CHECK(
        cudaMalloc(&table_device, parameters.root_of_unity_size * sizeof(Data64)));
    GPUNTT_CUDA_CHECK(cudaMemcpy(table_device, table.data(),
                                 parameters.root_of_unity_size * sizeof(Data64),
                                 cudaMemcpyHostToDevice));

    ntt_configuration<Data64> cfg{};
    cfg.n_power = logn;
    cfg.ntt_type = FORWARD;
    cfg.ntt_layout = PerPolynomial;
    cfg.reduction_poly = ReductionPolynomial::X_N_minus;
    cfg.stream = 0;
    GPU_NTT_Inplace(data, table_device, parameters.modulus, cfg, 1);

    std::vector<Data64> result(parameters.n);
    GPUNTT_CUDA_CHECK(cudaMemcpy(result.data(), data,
                                 parameters.n * sizeof(Data64),
                                 cudaMemcpyDeviceToHost));
    GPUNTT_CUDA_CHECK(cudaFree(data));
    GPUNTT_CUDA_CHECK(cudaFree(table_device));

    const bool correct = (result == expected);
    std::cout << (correct ? "GPU NTT matches the CPU NTT."
                          : "GPU NTT does not match the CPU NTT.")
              << std::endl;
    return correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
