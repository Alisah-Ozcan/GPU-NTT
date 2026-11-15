// Copyright 2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

// Tests the Barrett operations of OPERATOR_GPU against a CPU reference for
// odd moduli, as NTT moduli are, of every bit length the reduction supports:
// 2^(b-1) + 1, 2^b - 1 and random values in between, with random inputs and
// the edge values 0, 1 and modulus - 1.

#include "gpuntt/common/modular_arith.cuh"

#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

using namespace gpuntt;

namespace
{
    int failures = 0;
    int cases = 0;

    using u128 = unsigned __int128;

    // Outputs of the kernel for each input.
    enum Operation
    {
        mult_op,
        reduce_op,
        reduce_wide_op,
        reduce_forced_op,
        add_op,
        sub_op,
        operation_count
    };

    const char* operation_names[operation_count] = {
        "mult", "reduce", "reduce of two words", "reduce_forced", "add", "sub"};

    void check_cuda(cudaError_t error, const char* what)
    {
        if (error != cudaSuccess)
        {
            std::printf("CUDA error in %s: %s\n", what,
                        cudaGetErrorString(error));
            std::exit(1);
        }
    }

    // a and b are reduced, c is below 2^(2 * bit) and d is any value.
    template <typename T>
    __global__ void arithmetic_kernel(const T* a, const T* b, const T* c,
                                      const T* d, Modulus<T> modulus, T* output,
                                      int count)
    {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= count)
        {
            return;
        }
        T* out = output + i * operation_count;
        out[mult_op] = OPERATOR_GPU<T>::mult(a[i], b[i], modulus);
        out[reduce_op] = OPERATOR_GPU<T>::reduce(c[i], modulus);
        T wide[2];
        if constexpr (sizeof(T) == 4)
        {
            const Data64 product = Data64(a[i]) * b[i];
            wide[0] = static_cast<T>(product);
            wide[1] = static_cast<T>(product >> 32);
        }
        else
        {
            wide[0] = a[i] * b[i];
            wide[1] = __umul64hi(a[i], b[i]);
        }
        out[reduce_wide_op] = OPERATOR_GPU<T>::reduce(wide, modulus);
        out[reduce_forced_op] = OPERATOR_GPU<T>::reduce_forced(d[i], modulus);
        out[add_op] = OPERATOR_GPU<T>::add(a[i], b[i], modulus);
        out[sub_op] = OPERATOR_GPU<T>::sub(a[i], b[i], modulus);
    }

    template <typename T> void test_modulus(T value, std::mt19937_64& rng)
    {
        const Modulus<T> modulus(value);
        const int bits = static_cast<int>(modulus.bit);
        const u128 m = value;
        const int word = 8 * sizeof(T);
        const int c_bits = 2 * bits < word ? 2 * bits : word;
        const u128 c_limit = u128(1) << c_bits;

        std::vector<T> a, b, c, d;
        const T edges[3] = {0, 1, static_cast<T>(value - 1)};
        for (T x : edges)
        {
            for (T y : edges)
            {
                a.push_back(x);
                b.push_back(y);
            }
        }
        for (int i = 0; i < 4096; i++)
        {
            a.push_back(static_cast<T>(rng() % m));
            b.push_back(static_cast<T>(rng() % m));
        }
        for (size_t i = 0; i < a.size(); i++)
        {
            const T random = static_cast<T>(rng());
            c.push_back(i == 0 ? static_cast<T>(c_limit - 1)
                               : static_cast<T>(random % c_limit));
            d.push_back(i == 0 ? static_cast<T>(~T(0)) : random);
        }

        const int count = static_cast<int>(a.size());
        const size_t bytes = count * sizeof(T);
        T *device_a, *device_b, *device_c, *device_d, *device_output;
        check_cuda(cudaMalloc(&device_a, bytes), "cudaMalloc");
        check_cuda(cudaMalloc(&device_b, bytes), "cudaMalloc");
        check_cuda(cudaMalloc(&device_c, bytes), "cudaMalloc");
        check_cuda(cudaMalloc(&device_d, bytes), "cudaMalloc");
        check_cuda(cudaMalloc(&device_output, bytes * operation_count),
                   "cudaMalloc");
        check_cuda(
            cudaMemcpy(device_a, a.data(), bytes, cudaMemcpyHostToDevice),
            "cudaMemcpy");
        check_cuda(
            cudaMemcpy(device_b, b.data(), bytes, cudaMemcpyHostToDevice),
            "cudaMemcpy");
        check_cuda(
            cudaMemcpy(device_c, c.data(), bytes, cudaMemcpyHostToDevice),
            "cudaMemcpy");
        check_cuda(
            cudaMemcpy(device_d, d.data(), bytes, cudaMemcpyHostToDevice),
            "cudaMemcpy");
        arithmetic_kernel<<<(count + 255) / 256, 256>>>(
            device_a, device_b, device_c, device_d, modulus, device_output,
            count);
        check_cuda(cudaGetLastError(), "arithmetic_kernel");

        std::vector<T> output(count * operation_count);
        check_cuda(cudaMemcpy(output.data(), device_output,
                              bytes * operation_count, cudaMemcpyDeviceToHost),
                   "cudaMemcpy");
        cudaFree(device_a);
        cudaFree(device_b);
        cudaFree(device_c);
        cudaFree(device_d);
        cudaFree(device_output);

        int mismatches[operation_count] = {};
        for (int i = 0; i < count; i++)
        {
            const T expected[operation_count] = {
                static_cast<T>(u128(a[i]) * b[i] % m),
                static_cast<T>(c[i] % m),
                static_cast<T>(u128(a[i]) * b[i] % m),
                static_cast<T>(d[i] % m),
                static_cast<T>((u128(a[i]) + b[i]) % m),
                static_cast<T>((u128(a[i]) + m - b[i]) % m)};
            for (int k = 0; k < operation_count; k++)
            {
                mismatches[k] += output[i * operation_count + k] != expected[k];
            }
        }
        for (int k = 0; k < operation_count; k++)
        {
            cases++;
            if (mismatches[k] != 0)
            {
                failures++;
                std::printf("FAILED: %s of Data%d with the %d-bit modulus "
                            "%llu: %d of %d results\n",
                            operation_names[k], word, bits,
                            static_cast<unsigned long long>(value),
                            mismatches[k], count);
            }
        }
    }

    template <typename T> void test_moduli()
    {
        std::mt19937_64 rng(2026);
        for (int b = 2; b <= Modulus<T>::max_bit; b++)
        {
            const T low = T(1) << (b - 1);
            const T high = low | (low - 1);
            test_modulus<T>(low | 1, rng);
            test_modulus<T>(high, rng);
            for (int i = 0; i < 4; i++)
            {
                test_modulus<T>(low | (T(rng()) & (low - 1)) | 1, rng);
            }
        }
    }
} // namespace

int main()
{
    test_moduli<Data32>();
    test_moduli<Data64>();

    std::printf("modular_arith: %d of %d cases passed\n", cases - failures,
                cases);
    return failures == 0 ? 0 : 1;
}
