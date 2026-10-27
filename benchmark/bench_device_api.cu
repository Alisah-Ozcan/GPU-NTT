// Copyright 2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

// Polynomial multiplication modulo X^N + 1 with one operand already in the
// NTT domain: a <- INTT(NTT(a) * b_hat) for about 2^22 coefficients.
// "fused" runs it in one kernel with the device API; "separate" runs GPU_NTT,
// a pointwise product kernel and GPU_INTT.

#include "bench_util.cuh"
#include "gpuntt/ntt_merge/descriptor.cuh"

#include <utility>

using namespace gpuntt;

namespace
{
    using DataTypes = nvbench::type_list<Data32, Data64>;

    // Ring sizes that run in a single kernel.
    constexpr int kMaxFusedLogN = 10;

    template <int LogN, typename T, type D>
    using Description =
        decltype(Size<(1 << LogN)>() + Direction<D>() + Precision<T>() +
                 ReductionPoly<X_N_plus>() + Block());

    template <int LogN, typename T>
    __global__ void
    fused_multiply_kernel(T* a, const T* b_hat, const T* forward_roots,
                          const T* inverse_roots, Modulus<T> modulus,
                          T n_inverse, int batch)
    {
        using Fwd = Description<LogN, T, FORWARD>;
        using Inv = Description<LogN, T, INVERSE>;
        extern __shared__ char shared_memory_raw[];
        T* shared_memory = reinterpret_cast<T*>(shared_memory_raw);

        const int first = blockIdx.x * Fwd::batches_per_block;
        const int num_polys = min(Fwd::batches_per_block, batch - first);
        const size_t offset = size_t(first) * Fwd::size;

        Fwd().load(shared_memory, a + offset, modulus, num_polys);
        Fwd().execute(shared_memory, forward_roots, modulus);
        for (int i = threadIdx.x + threadIdx.y * blockDim.x;
             i < num_polys * Fwd::size; i += blockDim.x * blockDim.y)
        {
            shared_memory[i] = OPERATOR_GPU<T>::mult(
                shared_memory[i], b_hat[offset + i], modulus);
        }
        __syncthreads();
        Inv().execute(shared_memory, inverse_roots, modulus);
        Inv().store(shared_memory, a + offset, modulus, n_inverse, num_polys);
    }

    template <typename T>
    __global__ void pointwise_multiply_kernel(T* a, const T* b,
                                              Modulus<T> modulus, size_t count)
    {
        const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i < count)
        {
            a[i] = OPERATOR_GPU<T>::mult(a[i], b[i], modulus);
        }
    }

    template <int LogN, typename T>
    void launch_fused(T* a, const T* b_hat, const T* forward_roots,
                      const T* inverse_roots, Modulus<T> modulus, T n_inverse,
                      int batch, cudaStream_t stream)
    {
        using Fwd = Description<LogN, T, FORWARD>;
        fused_multiply_kernel<LogN, T><<<Fwd::grid(batch), Fwd::block_dim,
                                         Fwd::shared_memory_size, stream>>>(
            a, b_hat, forward_roots, inverse_roots, modulus, n_inverse, batch);
    }

    template <typename T, int... LogN>
    void launch_fused(int logn, std::integer_sequence<int, LogN...>, T* a,
                      const T* b_hat, const T* forward_roots,
                      const T* inverse_roots, Modulus<T> modulus, T n_inverse,
                      int batch, cudaStream_t stream)
    {
        ((logn == LogN + 1
              ? launch_fused<LogN + 1>(a, b_hat, forward_roots, inverse_roots,
                                       modulus, n_inverse, batch, stream)
              : void()),
         ...);
    }

    template <typename T>
    void poly_mul(nvbench::state& state, nvbench::type_list<T>)
    {
        const int logn = static_cast<int>(state.get_int64("LogN"));
        const size_t n = size_t(1) << logn;
        const int batch = bulk_batch(logn, 1);
        const bool fused = state.get_string("Method") == "fused";
        const T prime = bench_prime<T>();
        const Modulus<T> modulus(prime);
        const T n_inverse = static_cast<T>(12345);

        auto a = random_vector<T>(n * batch, prime, 1234);
        auto b_hat = random_vector<T>(n * batch, prime, 5678);
        auto forward_roots = random_vector<T>(n, prime, 4321);
        auto inverse_roots = random_vector<T>(n, prime, 8765);
        state.add_element_count(n * batch, "Coefficients");

        BenchStream stream(state);
        ntt_configuration<T> cfg = {.n_power = logn,
                                    .ntt_type = FORWARD,
                                    .ntt_layout = PerPolynomial,
                                    .reduction_poly = X_N_plus,
                                    .mod_inverse = n_inverse,
                                    .stream = stream.get()};
        ntt_configuration<T> inverse_cfg = cfg;
        inverse_cfg.ntt_type = INVERSE;

        state.exec(
            [&](nvbench::launch&)
            {
                if (fused)
                {
                    launch_fused(
                        logn, std::make_integer_sequence<int, kMaxFusedLogN>(),
                        ptr(a), ptr(b_hat), ptr(forward_roots),
                        ptr(inverse_roots), modulus, n_inverse, batch,
                        stream.get());
                }
                else
                {
                    GPU_NTT_Inplace(ptr(a), ptr(forward_roots), modulus, cfg,
                                    batch);
                    pointwise_multiply_kernel<<<(n * batch + 255) / 256, 256, 0,
                                                stream.get()>>>(
                        ptr(a), ptr(b_hat), modulus, n * batch);
                    GPU_INTT_Inplace(ptr(a), ptr(inverse_roots), modulus,
                                     inverse_cfg, batch);
                }
            });
    }

} // namespace

NVBENCH_BENCH_TYPES(poly_mul, NVBENCH_TYPE_AXES(DataTypes))
    .set_type_axes_names({"Data"})
    .add_int64_axis("LogN", nvbench::range(1, kMaxFusedLogN))
    .add_string_axis("Method", {"fused", "separate"})
    .set_timeout(1);
