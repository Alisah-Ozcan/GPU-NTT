// Copyright 2025-2026 Alişah Özcan
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

#endif // BENCH_UTIL_H
