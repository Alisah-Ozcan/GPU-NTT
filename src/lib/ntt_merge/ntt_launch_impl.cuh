// Copyright 2023-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan
// Paper: https://eprint.iacr.org/2023/1410

#ifndef NTT_LAUNCH_IMPL_H
#define NTT_LAUNCH_IMPL_H

#include "gpuntt/ntt_merge/descriptor.cuh"
#include "gpuntt/ntt_merge/ntt_device.cuh"
#include "ntt_launch.cuh"
#include <type_traits>
#include <utility>

// Kernels and launchers of the host API's PerPolynomial transforms. The
// launchers declared in ntt_launch.cuh are instantiated in ntt_forward.cu,
// ntt_inverse.cu and ntt_ordered.cu, so that their kernels compile in
// parallel.

namespace gpuntt
{
    // The host API runs every transform with kernels that take the ring size
    // as a template parameter and read their launch parameters from the
    // kernel tables at compile time, like the device API. The modulus and
    // polynomial policies select the single modulus, RNS and ordered
    // variants.

    // Launch parameters of kernel I of a transform of ring size 2^LogN.
    template <typename TU, type D, int LogN, int I = 0>
    using StageConfig = NTTStage<TU, D, LogN, I, 1, false, X_N_minus>;

    // Low ring sizes: one kernel that transforms blockDim.y polynomials per
    // block.
    template <int LogN, typename T, typename ModulusPolicy>
    __global__ void
    ForwardLowRing(T* polynomial_in,
                   typename std::make_unsigned<T>::type* polynomial_out,
                   const typename std::make_unsigned<
                       T>::type* __restrict__ root_of_unity_table,
                   ModulusPolicy modulus_policy, bool reduction_poly_check,
                   int total_batch)
    {
        using TU = typename std::make_unsigned<T>::type;
        using Config = StageConfig<TU, FORWARD, LogN>;

        ForwardCoreLowRingBody(polynomial_in, polynomial_out,
                               root_of_unity_table, modulus_policy,
                               Config::config.shared_index, LogN,
                               reduction_poly_check, total_batch);
    }

    template <int LogN, typename T, typename ModulusPolicy>
    __global__ void
    InverseLowRing(typename std::make_unsigned<T>::type* polynomial_in,
                   T* polynomial_out,
                   const typename std::make_unsigned<
                       T>::type* __restrict__ inverse_root_of_unity_table,
                   ModulusPolicy modulus_policy, bool reduction_poly_check,
                   int total_batch)
    {
        using TU = typename std::make_unsigned<T>::type;
        using Config = StageConfig<TU, INVERSE, LogN>;

        InverseCoreLowRingBody(polynomial_in, polynomial_out,
                               inverse_root_of_unity_table, modulus_policy,
                               Config::config.shared_index, LogN,
                               reduction_poly_check, total_batch);
    }

    // Stage I of a transform that spans several kernels.
    template <int LogN, int I, typename T, typename ModulusPolicy,
              typename PolyPolicy>
    __global__ void
    ForwardStage(T* polynomial_in,
                 typename std::make_unsigned<T>::type* polynomial_out,
                 const typename std::make_unsigned<
                     T>::type* __restrict__ root_of_unity_table,
                 ModulusPolicy modulus_policy, PolyPolicy poly_policy,
                 bool reduction_poly_check)
    {
        using TU = typename std::make_unsigned<T>::type;
        using Config = StageConfig<TU, FORWARD, LogN, I>;
        constexpr KernelConfig k = Config::config;

        ForwardCoreBody<Config::swap_grid_axes>(
            polynomial_in, polynomial_out, root_of_unity_table, modulus_policy,
            poly_policy, k.shared_index, k.logm, k.outer_iteration_count, LogN,
            k.not_last_kernel, reduction_poly_check);
    }

    // The inverse kernel table stores the last_kernel flag in
    // not_last_kernel.
    template <int LogN, int I, typename T, typename ModulusPolicy,
              typename PolyPolicy>
    __global__ void
    InverseStage(typename std::make_unsigned<T>::type* polynomial_in,
                 T* polynomial_out,
                 const typename std::make_unsigned<
                     T>::type* __restrict__ inverse_root_of_unity_table,
                 ModulusPolicy modulus_policy, PolyPolicy poly_policy,
                 bool reduction_poly_check)
    {
        using TU = typename std::make_unsigned<T>::type;
        using Config = StageConfig<TU, INVERSE, LogN, I>;
        constexpr KernelConfig k = Config::config;

        InverseCoreBody<Config::swap_grid_axes>(
            polynomial_in, polynomial_out, inverse_root_of_unity_table,
            modulus_policy, poly_policy, k.shared_index, k.logm, k.k,
            k.outer_iteration_count, LogN, k.not_last_kernel,
            reduction_poly_check);
    }

    // Calls f(std::integral_constant<int, n_power>()) if n_power is in
    // [LogN, MaxLogN]; the host API checks the range before.
    template <int LogN, int MaxLogN, typename F>
    void dispatch_log_n(int n_power, F&& f)
    {
        if constexpr (LogN <= MaxLogN)
        {
            if (n_power == LogN)
            {
                f(std::integral_constant<int, LogN>());
            }
            else
            {
                dispatch_log_n<LogN + 1, MaxLogN>(n_power, f);
            }
        }
    }

    template <typename F, int... I>
    void for_each_stage(F&& f, std::integer_sequence<int, I...>)
    {
        (f(std::integral_constant<int, I>()), ...);
    }

    template <int LogN, typename T, typename ModulusPolicy>
    void forward_low_ring(
        T* device_in, typename std::make_unsigned<T>::type* device_out,
        typename std::make_unsigned<T>::type* root_of_unity_table,
        ModulusPolicy modulus_policy, ReductionPolynomial reduction_poly,
        int batch_size, cudaStream_t stream)
    {
        using TU = typename std::make_unsigned<T>::type;
        constexpr KernelConfig k = StageConfig<TU, FORWARD, LogN>::config;

        ForwardLowRing<LogN>
            <<<dim3((batch_size + (k.blockdim_y - 1)) / k.blockdim_y, 1, 1),
               dim3(k.blockdim_x, k.blockdim_y), k.shared_memory, stream>>>(
                device_in, device_out, root_of_unity_table, modulus_policy,
                (reduction_poly == ReductionPolynomial::X_N_minus), batch_size);
        GPUNTT_CUDA_CHECK(cudaGetLastError());
    }

    template <int LogN, typename T, typename ModulusPolicy>
    void inverse_low_ring(
        typename std::make_unsigned<T>::type* device_in, T* device_out,
        typename std::make_unsigned<T>::type* root_of_unity_table,
        ModulusPolicy modulus_policy, ReductionPolynomial reduction_poly,
        int batch_size, cudaStream_t stream)
    {
        using TU = typename std::make_unsigned<T>::type;
        constexpr KernelConfig k = StageConfig<TU, INVERSE, LogN>::config;

        InverseLowRing<LogN>
            <<<dim3((batch_size + (k.blockdim_y - 1)) / k.blockdim_y, 1, 1),
               dim3(k.blockdim_x, k.blockdim_y), k.shared_memory, stream>>>(
                device_in, device_out, root_of_unity_table, modulus_policy,
                (reduction_poly == ReductionPolynomial::X_N_minus), batch_size);
        GPUNTT_CUDA_CHECK(cudaGetLastError());
    }

    // Runs the stage kernels of one forward transform. Stage i reads the
    // output of stage i - 1 (the first one reads device_in) and writes
    // device_out.
    template <int LogN, typename T, typename ModulusPolicy, typename PolyPolicy>
    void forward_stages(
        T* device_in, typename std::make_unsigned<T>::type* device_out,
        typename std::make_unsigned<T>::type* root_of_unity_table,
        ModulusPolicy modulus_policy, PolyPolicy poly_policy,
        ReductionPolynomial reduction_poly, int batch_size, cudaStream_t stream)
    {
        using TU = typename std::make_unsigned<T>::type;
        const bool x_n_minus =
            (reduction_poly == ReductionPolynomial::X_N_minus);

        for_each_stage(
            [&](auto stage)
            {
                constexpr int I = decltype(stage)::value;
                constexpr KernelConfig k =
                    StageConfig<TU, FORWARD, LogN, I>::config;
                const dim3 grid(k.griddim_x, k.griddim_y, batch_size);
                const dim3 block(k.blockdim_x, k.blockdim_y);
                if constexpr (I == 0)
                {
                    ForwardStage<LogN, I>
                        <<<grid, block, k.shared_memory, stream>>>(
                            device_in, device_out, root_of_unity_table,
                            modulus_policy, poly_policy, x_n_minus);
                }
                else
                {
                    ForwardStage<LogN, I>
                        <<<grid, block, k.shared_memory, stream>>>(
                            device_out, device_out, root_of_unity_table,
                            modulus_policy, poly_policy, x_n_minus);
                }
                GPUNTT_CUDA_CHECK(cudaGetLastError());
            },
            std::make_integer_sequence<
                int, forward_kernel_table<TU>[LogN].size()>());
    }

    // Inverse counterpart of forward_stages. Only the last stage writes T
    // (signed output); the others keep unsigned values in device_out.
    template <int LogN, typename T, typename ModulusPolicy, typename PolyPolicy>
    void inverse_stages(
        typename std::make_unsigned<T>::type* device_in, T* device_out,
        typename std::make_unsigned<T>::type* root_of_unity_table,
        ModulusPolicy modulus_policy, PolyPolicy poly_policy,
        ReductionPolynomial reduction_poly, int batch_size, cudaStream_t stream)
    {
        using TU = typename std::make_unsigned<T>::type;
        constexpr int num_stages = inverse_kernel_table<TU>[LogN].size();
        TU* device_out_unsigned = reinterpret_cast<TU*>(device_out);
        const bool x_n_minus =
            (reduction_poly == ReductionPolynomial::X_N_minus);

        for_each_stage(
            [&](auto stage)
            {
                constexpr int I = decltype(stage)::value;
                constexpr KernelConfig k =
                    StageConfig<TU, INVERSE, LogN, I>::config;
                const dim3 grid(k.griddim_x, k.griddim_y, batch_size);
                const dim3 block(k.blockdim_x, k.blockdim_y);
                TU* in = (I == 0) ? device_in : device_out_unsigned;
                if constexpr (I + 1 == num_stages)
                {
                    InverseStage<LogN, I>
                        <<<grid, block, k.shared_memory, stream>>>(
                            in, device_out, root_of_unity_table, modulus_policy,
                            poly_policy, x_n_minus);
                }
                else
                {
                    InverseStage<LogN, I>
                        <<<grid, block, k.shared_memory, stream>>>(
                            in, device_out_unsigned, root_of_unity_table,
                            modulus_policy, poly_policy, x_n_minus);
                }
                GPUNTT_CUDA_CHECK(cudaGetLastError());
            },
            std::make_integer_sequence<int, num_stages>());
    }

    // PerPolynomial transforms. Ring sizes below kForwardMinStageLogN
    // (forward) and kInverseMinStageLogN (inverse) use the low ring kernel.
    constexpr int kForwardMinStageLogN = 10;
    constexpr int kInverseMinStageLogN = 11;

    template <typename T, typename ModulusPolicy>
    void forward_per_polynomial(
        T* device_in, typename std::make_unsigned<T>::type* device_out,
        typename std::make_unsigned<T>::type* root_of_unity_table,
        ModulusPolicy modulus_policy, int n_power,
        ReductionPolynomial reduction_poly, int batch_size, cudaStream_t stream)
    {
        dispatch_log_n<1, 28>(
            n_power,
            [&](auto log_n)
            {
                constexpr int LogN = decltype(log_n)::value;
                if constexpr (LogN < kForwardMinStageLogN)
                {
                    forward_low_ring<LogN>(device_in, device_out,
                                           root_of_unity_table, modulus_policy,
                                           reduction_poly, batch_size, stream);
                }
                else
                {
                    forward_stages<LogN>(device_in, device_out,
                                         root_of_unity_table, modulus_policy,
                                         SequentialPolys{}, reduction_poly,
                                         batch_size, stream);
                }
            });
    }

    template <typename T, typename ModulusPolicy>
    void inverse_per_polynomial(
        typename std::make_unsigned<T>::type* device_in, T* device_out,
        typename std::make_unsigned<T>::type* root_of_unity_table,
        ModulusPolicy modulus_policy, int n_power,
        ReductionPolynomial reduction_poly, int batch_size, cudaStream_t stream)
    {
        dispatch_log_n<1, 28>(
            n_power,
            [&](auto log_n)
            {
                constexpr int LogN = decltype(log_n)::value;
                if constexpr (LogN < kInverseMinStageLogN)
                {
                    inverse_low_ring<LogN>(device_in, device_out,
                                           root_of_unity_table, modulus_policy,
                                           reduction_poly, batch_size, stream);
                }
                else
                {
                    inverse_stages<LogN>(device_in, device_out,
                                         root_of_unity_table, modulus_policy,
                                         SequentialPolys{}, reduction_poly,
                                         batch_size, stream);
                }
            });
    }

    // Ordered RNS transforms (ring sizes 2^10 to 2^28, stage kernels only).
    template <typename T, typename ModulusPolicy, typename PolyPolicy>
    void ordered_stages(T* device_in, T* device_out, T* root_of_unity_table,
                        ModulusPolicy modulus_policy, PolyPolicy poly_policy,
                        type ntt_type, int n_power,
                        ReductionPolynomial reduction_poly, int batch_size,
                        cudaStream_t stream)
    {
        dispatch_log_n<10, 28>(
            n_power,
            [&](auto log_n)
            {
                constexpr int LogN = decltype(log_n)::value;
                if (ntt_type == FORWARD)
                {
                    forward_stages<LogN>(device_in, device_out,
                                         root_of_unity_table, modulus_policy,
                                         poly_policy, reduction_poly,
                                         batch_size, stream);
                }
                else
                {
                    inverse_stages<LogN>(device_in, device_out,
                                         root_of_unity_table, modulus_policy,
                                         poly_policy, reduction_poly,
                                         batch_size, stream);
                }
            });
    }

} // namespace gpuntt

#endif // NTT_LAUNCH_IMPL_H
