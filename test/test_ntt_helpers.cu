// Copyright 2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

// Tests the helpers around the device API: Modulus built in a kernel must
// match the host one, the root tables (built on the host and on the GPU) and
// N^-1 of a description must match the ones of NTTParameters, and every prime
// of fields.cuh must give the CPU NTT result through GPU_NTT and GPU_INTT.

#include "gpuntt/ntt_merge/descriptor.cuh"
#include "gpuntt/ntt_merge/ntt.cuh"
#include "gpuntt/ntt_merge/ntt_cpu.cuh"

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

    template <typename T> T* to_device(const std::vector<T>& values)
    {
        T* data;
        check_cuda(cudaMalloc(&data, values.size() * sizeof(T)), "cudaMalloc");
        check_cuda(cudaMemcpy(data, values.data(), values.size() * sizeof(T),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy");
        return data;
    }

    template <typename T>
    std::vector<T> from_device(const T* data, size_t count)
    {
        std::vector<T> values(count);
        check_cuda(cudaMemcpy(values.data(), data, count * sizeof(T),
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy");
        return values;
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
        const std::vector<T> expected =
            parameters.gpu_root_of_unity_table_generator(
                D == FORWARD ? parameters.forward_root_of_unity_table
                             : parameters.inverse_root_of_unity_table);
        const std::vector<T> table = NTT::make_twiddles_host(
            parameters.modulus, parameters.root_of_unity);

        const std::string name =
            std::string(sizeof(T) == 4 ? "Data32" : "Data64") + " LogN " +
            std::to_string(LogN) + (D == FORWARD ? " forward" : " inverse") +
            (R == X_N_plus ? " X^N+1" : " X^N-1");
        check(table == expected, (name + " root table").c_str());

        T* device_table;
        check_cuda(cudaMalloc(&device_table, table.size() * sizeof(T)),
                   "cudaMalloc");
        NTT::make_twiddles(device_table, parameters.modulus,
                           parameters.root_of_unity);
        check(from_device(device_table, table.size()) == expected,
              (name + " root table built on the GPU").c_str());
        cudaFree(device_table);
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

    // NTT with the roots of prime against the CPU NTT, then INTT back.
    template <typename T> void test_prime(const NTTPrime<T>& prime)
    {
        constexpr int log_n = 12;
        constexpr int n = 1 << log_n;
        constexpr int batch = 2;
        using Fwd = decltype(Size<n>() + Direction<FORWARD>() + Precision<T>() +
                             ReductionPoly<X_N_plus>());
        using Inv = decltype(Size<n>() + Direction<INVERSE>() + Precision<T>() +
                             ReductionPoly<X_N_plus>());

        const Modulus<T> modulus(prime.value);
        const T psi = root_of_unity(prime, log_n + 1);
        const T omega = OPERATOR<T>::mult(psi, psi, modulus);
        NTTParameters<T> parameters(log_n, NTTFactors<T>(modulus, omega, psi),
                                    X_N_plus);
        NTTCPU<T> cpu(parameters);

        std::mt19937_64 rng(prime.value);
        std::vector<T> input(size_t(batch) * n);
        for (T& x : input)
        {
            x = T(rng()) % prime.value;
        }
        std::vector<T> expected;
        for (int i = 0; i < batch; i++)
        {
            std::vector<T> poly(input.begin() + size_t(i) * n,
                                input.begin() + size_t(i + 1) * n);
            const std::vector<T> result = cpu.ntt(poly);
            expected.insert(expected.end(), result.begin(), result.end());
        }

        T* data = to_device(input);
        T* forward_roots = to_device(Fwd::make_twiddles_host(prime));
        T* inverse_roots;
        check_cuda(
            cudaMalloc(&inverse_roots, Inv::twiddle_table_size * sizeof(T)),
            "cudaMalloc");
        Inv::make_twiddles(inverse_roots, prime);

        ntt_configuration<T> cfg = {.n_power = log_n,
                                    .ntt_type = FORWARD,
                                    .ntt_layout = PerPolynomial,
                                    .reduction_poly = X_N_plus,
                                    .mod_inverse = Inv::n_inv(modulus),
                                    .stream = 0};
        GPU_NTT_Inplace(data, forward_roots, modulus, cfg, batch);
        const std::vector<T> output = from_device(data, input.size());

        cfg.ntt_type = INVERSE;
        GPU_INTT_Inplace(data, inverse_roots, modulus, cfg, batch);
        const std::vector<T> round_trip = from_device(data, input.size());

        cudaFree(data);
        cudaFree(forward_roots);
        cudaFree(inverse_roots);

        const std::string name = "prime " + std::to_string(prime.value);
        check(output == expected, (name + " NTT").c_str());
        check(round_trip == input, (name + " round trip").c_str());
    }

    // The root has the stated order and 2^log_max_order divides p - 1.
    template <typename T> void test_prime_root(const NTTPrime<T>& prime)
    {
        const Modulus<T> modulus(prime.value);
        const T p_minus_1 = prime.value - 1;
        const bool order =
            ((p_minus_1 >> prime.log_max_order) & 1) == 1 &&
            (p_minus_1 & ((T(1) << prime.log_max_order) - 1)) == 0 &&
            OPERATOR<T>::exp(prime.root, T(1) << (prime.log_max_order - 1),
                             modulus) == p_minus_1;
        check(order, ("prime " + std::to_string(prime.value) +
                      " root of order 2^log_max_order")
                         .c_str());

        bool thrown = false;
        try
        {
            root_of_unity(prime, prime.log_max_order + 1);
        }
        catch (const std::invalid_argument&)
        {
            thrown = true;
        }
        check(thrown, "root of too large order is rejected");
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

    for (const NTTPrime<Data32>& prime : ntt_primes32)
    {
        test_prime_root(prime);
        test_prime(prime);
    }
    for (const NTTPrime<Data64>& prime : ntt_primes64)
    {
        test_prime_root(prime);
        test_prime(prime);
    }

    std::printf("ntt_helpers: %d of %d cases passed\n", cases - failures,
                cases);
    return failures == 0 ? 0 : 1;
}
