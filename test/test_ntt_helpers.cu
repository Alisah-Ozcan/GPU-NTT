// Copyright 2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

// Tests the helpers around the device API: Modulus built in a kernel must
// match the host one.

#include "gpuntt/common/modular_arith.cuh"

#include <cstdio>
#include <random>
#include <vector>

using namespace gpuntt;

namespace
{
    int failures = 0;
    int cases = 0;

    void check_cuda(cudaError_t error, const char* what)
    {
        if (error != cudaSuccess)
        {
            std::printf("CUDA error in %s: %s\n", what,
                        cudaGetErrorString(error));
            std::exit(1);
        }
    }

    void check(bool ok, const char* what)
    {
        cases++;
        if (!ok)
        {
            failures++;
            std::printf("FAILED: %s\n", what);
        }
    }

    template <typename T>
    __global__ void modulus_kernel(const T* values, Modulus<T>* moduli,
                                   int count)
    {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < count)
        {
            moduli[i] = Modulus<T>(values[i]);
        }
    }

    // Moduli of every bit length up to max_bit: 2^(b-1), 2^b - 1 and random
    // values in between.
    template <typename T> void test_device_modulus()
    {
        std::mt19937_64 rng(2026);
        std::vector<T> values;
        for (int b = 2; b <= Modulus<T>::max_bit; b++)
        {
            const T low = T(1) << (b - 1);
            const T high = low | (low - 1);
            values.push_back(low);
            values.push_back(high);
            for (int i = 0; i < 8; i++)
            {
                values.push_back(low | (T(rng()) & (low - 1)));
            }
        }

        const int count = static_cast<int>(values.size());
        T* device_values;
        Modulus<T>* device_moduli;
        check_cuda(cudaMalloc(&device_values, count * sizeof(T)), "cudaMalloc");
        check_cuda(cudaMalloc(&device_moduli, count * sizeof(Modulus<T>)),
                   "cudaMalloc");
        check_cuda(cudaMemcpy(device_values, values.data(), count * sizeof(T),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy");
        modulus_kernel<<<(count + 255) / 256, 256>>>(device_values,
                                                     device_moduli, count);
        check_cuda(cudaGetLastError(), "modulus_kernel");

        std::vector<Modulus<T>> moduli(count);
        check_cuda(cudaMemcpy(moduli.data(), device_moduli,
                              count * sizeof(Modulus<T>),
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy");
        cudaFree(device_values);
        cudaFree(device_moduli);

        int mismatches = 0;
        for (int i = 0; i < count; i++)
        {
            const Modulus<T> host(values[i]);
            if (moduli[i].value != host.value || moduli[i].bit != host.bit ||
                moduli[i].mu != host.mu)
            {
                mismatches++;
            }
        }
        check(mismatches == 0, sizeof(T) == 4
                                   ? "Modulus<Data32> built in a kernel"
                                   : "Modulus<Data64> built in a kernel");
    }

} // namespace

int main()
{
    test_device_modulus<Data32>();
    test_device_modulus<Data64>();

    std::printf("ntt_helpers: %d of %d cases passed\n", cases - failures,
                cases);
    return failures == 0 ? 0 : 1;
}
