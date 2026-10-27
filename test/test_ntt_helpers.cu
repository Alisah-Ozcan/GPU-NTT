// Copyright 2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

// Tests the helpers around the device API: Modulus built in a kernel must
// match the host one, and the root tables and N^-1 of a description must
// match the ones of NTTParameters.

#include "gpuntt/ntt_merge/descriptor.cuh"

#include <cstdio>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
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

    template <typename NTT, typename = void>
    struct has_root_table : std::false_type
    {
    };

    template <typename NTT>
    struct has_root_table<NTT, std::void_t<decltype(NTT::twiddle_table_size)>>
        : std::true_type
    {
    };

    static_assert(!has_root_table<decltype(Size<1024>() + Direction<FORWARD>() +
                                           Precision<Data64>())>::value);

    template <typename T, int LogN, type D, ReductionPolynomial R>
    void test_root_table()
    {
        using NTT = decltype(Size<(1 << LogN)>() + Direction<D>() +
                             Precision<T>() + ReductionPoly<R>());
        static_assert(NTT::twiddle_table_size ==
                      (R == X_N_plus ? (1 << LogN) : (1 << LogN) / 2));

        NTTParameters<T> parameters(LogN, R);
        const std::vector<Root<T>> expected =
            parameters.gpu_root_of_unity_table_generator(
                D == FORWARD ? parameters.forward_root_of_unity_table
                             : parameters.inverse_root_of_unity_table);
        const std::vector<Root<T>> table = NTT::make_twiddles_host(
            parameters.modulus, parameters.root_of_unity);

        const std::string name =
            std::string(sizeof(T) == 4 ? "Data32" : "Data64") + " LogN " +
            std::to_string(LogN) + (D == FORWARD ? " forward" : " inverse") +
            (R == X_N_plus ? " X^N+1" : " X^N-1");
        check(table == expected, (name + " root table").c_str());
        check(NTT::n_inv(parameters.modulus) == parameters.n_inv,
              (name + " n_inv").c_str());
    }

    template <typename T, int LogN> void test_root_tables()
    {
        test_root_table<T, LogN, FORWARD, X_N_plus>();
        test_root_table<T, LogN, FORWARD, X_N_minus>();
        test_root_table<T, LogN, INVERSE, X_N_plus>();
        test_root_table<T, LogN, INVERSE, X_N_minus>();
    }

    template <typename T, int... LogN>
    void test_root_tables(std::integer_sequence<int, LogN...>)
    {
        (test_root_tables<T, LogN + 1>(), ...);
    }

    // A root of the wrong order is rejected.
    template <typename T> void test_wrong_root()
    {
        using NTT = decltype(Size<1024>() + Direction<FORWARD>() +
                             Precision<T>() + ReductionPoly<X_N_plus>());
        NTTParameters<T> parameters(10, X_N_plus);
        // omega = psi^2 has order N instead of 2N.
        const T omega =
            OPERATOR<T>::mult(parameters.root_of_unity,
                              parameters.root_of_unity, parameters.modulus);
        bool thrown = false;
        try
        {
            NTT::make_twiddles_host(parameters.modulus, omega);
        }
        catch (const std::invalid_argument&)
        {
            thrown = true;
        }
        check(thrown, "root of the wrong order is rejected");
    }

} // namespace

int main()
{
    test_device_modulus<Data32>();
    test_device_modulus<Data64>();

    test_root_tables<Data32>(std::make_integer_sequence<int, 14>());
    test_root_tables<Data64>(std::make_integer_sequence<int, 14>());
    test_root_tables<Data32, 18>();
    test_root_tables<Data64, 18>();
    test_wrong_root<Data32>();
    test_wrong_root<Data64>();

    std::printf("ntt_helpers: %d of %d cases passed\n", cases - failures,
                cases);
    return failures == 0 ? 0 : 1;
}
