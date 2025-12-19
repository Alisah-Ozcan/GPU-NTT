// Copyright 2024 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef BENCH_UTIL_H
#define BENCH_UTIL_H

#include <cstdlib>
#include <random>
#include <nvbench/nvbench.cuh>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>
#include <cuda_runtime.h>
#include <type_traits>
#include <limits>
#include "gpuntt/ntt_4step/ntt_4step_cpu.cuh"
#include "gpuntt/ntt_merge/ntt.cuh"

template <typename T> struct random_functor
{
    std::uint32_t seed;

    __host__ __device__ random_functor(std::uint32_t _seed) : seed(_seed) {}

    __host__ __device__ T operator()(const int n) const
    {
        thrust::default_random_engine rng(seed);
        rng.discard(n);

        if constexpr (std::is_same<T, float>::value)
        {
            thrust::uniform_real_distribution<float> dist(0.0f, 1.0f);
            return dist(rng);
        }
        else if constexpr (std::is_same<T, double>::value)
        {
            thrust::uniform_real_distribution<double> dist(0.0, 1.0);
            return dist(rng);
        }
        else if constexpr (std::is_same<T, std::uint32_t>::value)
        {
            thrust::uniform_int_distribution<std::uint32_t> dist(0, UINT_MAX);
            return dist(rng);
        }
        else if constexpr (std::is_same<T, std::uint64_t>::value)
        {
            thrust::uniform_int_distribution<std::uint64_t> dist(0, ULLONG_MAX);
            return dist(rng);
        }
        else
        {
#ifndef __CUDA_ARCH__
            throw std::runtime_error("Unsupported type for random_functor");
#else
            return T();
#endif
        }
    }
};

// For 4-Step-NTT
std::vector<int> bench_matrix_dimention(int logn)
{
    gpuntt::customAssert(12 <= logn <= 24, "LOGN should be in range 12 to 24.");
    std::vector<int> shape;
    switch (logn)
    {
        case 12:
            shape = {32, 128};
            return shape;
        case 13:
            shape = {32, 256};
            return shape;
        case 14:
            shape = {32, 512};
            return shape;
        case 15:
            shape = {64, 512};
            return shape;
        case 16:
            shape = {128, 512};
            return shape;
        case 17:
            shape = {32, 4096};
            return shape;
        case 18:
            shape = {32, 8192};
            return shape;
        case 19:
            shape = {32, 16384};
            return shape;
        case 20:
            shape = {32, 32768};
            return shape;
        case 21:
            shape = {64, 32768};
            return shape;
        case 22:
            shape = {128, 32768};
            return shape;
        case 23:
            shape = {128, 65536};
            return shape;
        case 24:
            shape = {256, 65536};
            return shape;
        default:
            throw std::runtime_error("Invalid choice.\n");
    }
}

#endif // BENCH_UTIL_H
