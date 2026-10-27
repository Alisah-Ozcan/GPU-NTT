// Copyright 2022 Ulvetanna Inc.
// Copyright 2023-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan
// Paper: https://eprint.iacr.org/2023/1410

#include "gpuntt/ntt_merge/ntt.cuh"
#include "gpuntt/ntt_merge/descriptor.cuh"
#include "gpuntt/ntt_merge/ntt_device.cuh"
#include <type_traits>
#include <utility>

namespace gpuntt
{
    // Kernels compute global memory offsets with 32-bit integers, so a call
    // can transform at most 2^31 coefficients in total.
    static void check_batch_size(int n_power, int batch_size)
    {
        if (batch_size <= 0)
        {
            throw std::invalid_argument("Invalid batch_size!");
        }
        if (n_power > 0 && n_power < 31 &&
            (static_cast<std::int64_t>(batch_size) << n_power) >
                (static_cast<std::int64_t>(1) << 31))
        {
            throw std::invalid_argument(
                "batch_size * ring size must be at most 2^31!");
        }
    }

    template <typename T>
    __global__ void ForwardCoreTranspose(
        T* polynomial_in, typename std::make_unsigned<T>::type* polynomial_out,
        const Root<typename std::make_unsigned<
            T>::type>* __restrict__ root_of_unity_table,
        const Modulus<typename std::make_unsigned<T>::type> modulus,
        int log_row, int log_column, bool reduction_poly_check)
    {
        using TU = typename std::make_unsigned<T>::type;

        extern __shared__ char shared_memory_typed[];
        TU* shared_memory = reinterpret_cast<TU*>(shared_memory_typed);

        const TransposeThread<TU> thread =
            TransposeThreadState<false, false, TU>(SingleModulus<TU>{modulus},
                                                   log_row, log_column);
        TransposeLoadRoots(shared_memory, root_of_unity_table,
                           reduction_poly_check);
        TransposeLoad(thread, shared_memory, polynomial_in);
        ForwardTransposeExecute(thread, shared_memory, log_row,
                                reduction_poly_check);
        ForwardTransposeStore(thread, shared_memory, polynomial_out);
    }

    template <typename T>
    __global__ void ForwardCoreTranspose(
        T* polynomial_in, typename std::make_unsigned<T>::type* polynomial_out,
        const Root<typename std::make_unsigned<
            T>::type>* __restrict__ root_of_unity_table,
        const Modulus<typename std::make_unsigned<T>::type>* modulus,
        int log_row, int log_column, bool reduction_poly_check, int mod_count)
    {
        using TU = typename std::make_unsigned<T>::type;

        extern __shared__ char shared_memory_typed[];
        TU* shared_memory = reinterpret_cast<TU*>(shared_memory_typed);

        const TransposeThread<TU> thread =
            TransposeThreadState<true, false, TU>(
                RnsModulus<TU>{modulus, mod_count}, log_row, log_column);
        TransposeLoadColumnRoots(thread, shared_memory, root_of_unity_table,
                                 log_row, reduction_poly_check);
        TransposeLoad(thread, shared_memory, polynomial_in);
        ForwardTransposeExecute(thread, shared_memory, log_row,
                                reduction_poly_check);
        ForwardTransposeStore(thread, shared_memory, polynomial_out);
    }

    template <typename T>
    __global__ void InverseCoreTranspose(
        typename std::make_unsigned<T>::type* polynomial_in, T* polynomial_out,
        const Root<typename std::make_unsigned<
            T>::type>* __restrict__ inverse_root_of_unity_table,
        Modulus<typename std::make_unsigned<T>::type> modulus,
        Ninverse<typename std::make_unsigned<T>::type> n_inverse, int log_row,
        int log_column, bool reduction_poly_check)
    {
        using TU = typename std::make_unsigned<T>::type;

        extern __shared__ char shared_memory_typed[];
        TU* shared_memory = reinterpret_cast<TU*>(shared_memory_typed);

        const TransposeThread<TU> thread =
            TransposeThreadState<false, true, TU>(
                SingleModulus<TU>{modulus, n_inverse}, log_row, log_column);
        TransposeLoadRoots(shared_memory, inverse_root_of_unity_table,
                           reduction_poly_check);
        TransposeLoad(thread, shared_memory, polynomial_in);
        InverseTransposeExecute(thread, shared_memory, log_row,
                                reduction_poly_check);
        InverseTransposeStore(thread, shared_memory, polynomial_out);
    }

    template <typename T>
    __global__ void InverseCoreTranspose(
        typename std::make_unsigned<T>::type* polynomial_in, T* polynomial_out,
        const Root<typename std::make_unsigned<
            T>::type>* __restrict__ inverse_root_of_unity_table,
        Modulus<typename std::make_unsigned<T>::type>* modulus,
        Ninverse<typename std::make_unsigned<T>::type>* n_inverse, int log_row,
        int log_column, bool reduction_poly_check, int mod_count)
    {
        using TU = typename std::make_unsigned<T>::type;

        extern __shared__ char shared_memory_typed[];
        TU* shared_memory = reinterpret_cast<TU*>(shared_memory_typed);

        const TransposeThread<TU> thread = TransposeThreadState<true, true, TU>(
            RnsModulus<TU>{modulus, mod_count, n_inverse}, log_row, log_column);
        TransposeLoadColumnRoots(thread, shared_memory,
                                 inverse_root_of_unity_table, log_row,
                                 reduction_poly_check);
        TransposeLoad(thread, shared_memory, polynomial_in);
        InverseTransposeExecute(thread, shared_memory, log_row,
                                reduction_poly_check);
        InverseTransposeStore(thread, shared_memory, polynomial_out);
    }

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
                   const Root<typename std::make_unsigned<
                       T>::type>* __restrict__ root_of_unity_table,
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
                   const Root<typename std::make_unsigned<
                       T>::type>* __restrict__ inverse_root_of_unity_table,
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
                 const Root<typename std::make_unsigned<
                     T>::type>* __restrict__ root_of_unity_table,
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
                 const Root<typename std::make_unsigned<
                     T>::type>* __restrict__ inverse_root_of_unity_table,
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
        Root<typename std::make_unsigned<T>::type>* root_of_unity_table,
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
        Root<typename std::make_unsigned<T>::type>* root_of_unity_table,
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
        Root<typename std::make_unsigned<T>::type>* root_of_unity_table,
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
        Root<typename std::make_unsigned<T>::type>* root_of_unity_table,
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
        Root<typename std::make_unsigned<T>::type>* root_of_unity_table,
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
        Root<typename std::make_unsigned<T>::type>* root_of_unity_table,
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
    void ordered_stages(T* device_in, T* device_out,
                        Root<T>* root_of_unity_table,
                        ModulusPolicy modulus_policy, PolyPolicy poly_policy,
                        ntt_rns_configuration<T> cfg, int batch_size)
    {
        dispatch_log_n<10, 28>(
            cfg.n_power,
            [&](auto log_n)
            {
                constexpr int LogN = decltype(log_n)::value;
                if (cfg.ntt_type == FORWARD)
                {
                    forward_stages<LogN>(device_in, device_out,
                                         root_of_unity_table, modulus_policy,
                                         poly_policy, cfg.reduction_poly,
                                         batch_size, cfg.stream);
                }
                else
                {
                    inverse_stages<LogN>(device_in, device_out,
                                         root_of_unity_table, modulus_policy,
                                         poly_policy, cfg.reduction_poly,
                                         batch_size, cfg.stream);
                }
            });
    }

    template <typename T>
    __host__ void
    GPU_NTT(T* device_in, typename std::make_unsigned<T>::type* device_out,
            Root<typename std::make_unsigned<T>::type>* root_of_unity_table,
            Modulus<typename std::make_unsigned<T>::type> modulus,
            ntt_configuration<typename std::make_unsigned<T>::type> cfg,
            int batch_size)
    {
        using TU = typename std::make_unsigned<T>::type;

        check_batch_size(cfg.n_power, batch_size);

        switch (cfg.ntt_layout)
        {
            case PerPolynomial:
            {
                if ((cfg.n_power <= 0 || cfg.n_power >= 29))
                {
                    throw std::invalid_argument("Invalid n_power range!");
                }

                forward_per_polynomial(
                    device_in, device_out, root_of_unity_table,
                    SingleModulus<TU>{modulus}, cfg.n_power, cfg.reduction_poly,
                    batch_size, cfg.stream);
            }
            break;
            case PerCoefficient:
            {
                if ((cfg.n_power <= 0 || cfg.n_power >= 10))
                {
                    throw std::invalid_argument("Invalid n_power range!");
                }

                if ((batch_size & (batch_size - 1)) != 0)
                {
                    throw std::invalid_argument(
                        "batch_size must be a power of two for "
                        "PerCoefficient layout!");
                }

                int log_batch_size = log2(batch_size);
                int total_size = 1 << (cfg.n_power + log_batch_size);
                int total_block_thread = 512;
                int total_block_count = total_size / (total_block_thread * 2);
                int blockdim_y = 1 << (cfg.n_power - 1);
                int blockdim_x = total_block_thread / blockdim_y;
                ForwardCoreTranspose<<<
                    dim3(total_block_count, 1, 1),
                    dim3(blockdim_x, blockdim_y, 1),
                    ((total_block_thread * 2) + (1 << cfg.n_power)) * sizeof(T),
                    cfg.stream>>>(
                    device_in, device_out, root_of_unity_table, modulus,
                    cfg.n_power, log_batch_size,
                    (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                GPUNTT_CUDA_CHECK(cudaGetLastError());
            }
            break;
            default:
                throw std::invalid_argument("Invalid ntt_layout!");
                break;
        }
    }

    template <typename T>
    __host__ void
    GPU_INTT(typename std::make_unsigned<T>::type* device_in, T* device_out,
             Root<typename std::make_unsigned<T>::type>* root_of_unity_table,
             Modulus<typename std::make_unsigned<T>::type> modulus,
             ntt_configuration<typename std::make_unsigned<T>::type> cfg,
             int batch_size)
    {
        using TU = typename std::make_unsigned<T>::type;

        check_batch_size(cfg.n_power, batch_size);

        switch (cfg.ntt_layout)
        {
            case PerPolynomial:
            {
                if ((cfg.n_power <= 0 || cfg.n_power >= 29))
                {
                    throw std::invalid_argument("Invalid n_power range!");
                }

                inverse_per_polynomial(
                    device_in, device_out, root_of_unity_table,
                    SingleModulus<TU>{modulus, cfg.mod_inverse}, cfg.n_power,
                    cfg.reduction_poly, batch_size, cfg.stream);
            }
            break;
            case PerCoefficient:
            {
                if ((cfg.n_power <= 0 || cfg.n_power >= 10))
                {
                    throw std::invalid_argument("Invalid n_power range!");
                }

                if ((batch_size & (batch_size - 1)) != 0)
                {
                    throw std::invalid_argument(
                        "batch_size must be a power of two for "
                        "PerCoefficient layout!");
                }

                int log_batch_size = log2(batch_size);
                int total_size = 1 << (cfg.n_power + log_batch_size);
                int total_block_thread = 512;
                int total_block_count = total_size / (total_block_thread * 2);
                int blockdim_y = 1 << (cfg.n_power - 1);
                int blockdim_x = total_block_thread / blockdim_y;
                InverseCoreTranspose<<<
                    dim3(total_block_count, 1, 1),
                    dim3(blockdim_x, blockdim_y, 1),
                    ((total_block_thread * 2) + (1 << cfg.n_power)) * sizeof(T),
                    cfg.stream>>>(
                    device_in, device_out, root_of_unity_table, modulus,
                    cfg.mod_inverse, cfg.n_power, log_batch_size,
                    (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                GPUNTT_CUDA_CHECK(cudaGetLastError());
            }
            break;
            default:
                throw std::invalid_argument("Invalid ntt_layout!");
                break;
        }
    }

    template <typename T>
    __host__ void
    GPU_NTT(T* device_in, typename std::make_unsigned<T>::type* device_out,
            Root<typename std::make_unsigned<T>::type>* root_of_unity_table,
            Modulus<typename std::make_unsigned<T>::type>* modulus,
            ntt_rns_configuration<typename std::make_unsigned<T>::type> cfg,
            int batch_size, int mod_count)
    {
        using TU = typename std::make_unsigned<T>::type;

        check_batch_size(cfg.n_power, batch_size);

        switch (cfg.ntt_layout)
        {
            case PerPolynomial:
            {
                if ((cfg.n_power <= 0 || cfg.n_power >= 29))
                {
                    throw std::invalid_argument("Invalid n_power range!");
                }

                forward_per_polynomial(
                    device_in, device_out, root_of_unity_table,
                    RnsModulus<TU>{modulus, mod_count}, cfg.n_power,
                    cfg.reduction_poly, batch_size, cfg.stream);
            }
            break;
            case PerCoefficient:
            {
                if ((cfg.n_power <= 0 || cfg.n_power >= 10))
                {
                    throw std::invalid_argument("Invalid n_power range!");
                }

                if ((batch_size & (batch_size - 1)) != 0)
                {
                    throw std::invalid_argument(
                        "batch_size must be a power of two for "
                        "PerCoefficient layout!");
                }

                int log_batch_size = log2(batch_size);
                int total_size = 1 << (cfg.n_power + log_batch_size);
                int total_block_thread = 512;
                int total_block_count = total_size / (total_block_thread * 2);
                int blockdim_y = 1 << (cfg.n_power - 1);
                int blockdim_x = total_block_thread / blockdim_y;
                ForwardCoreTranspose<<<
                    dim3(total_block_count, 1, 1),
                    dim3(blockdim_x, blockdim_y, 1),
                    // Data and one root table per column of the block.
                    (total_block_thread * 2 * 2) * sizeof(T), cfg.stream>>>(
                    device_in, device_out, root_of_unity_table, modulus,
                    cfg.n_power, log_batch_size,
                    (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                    mod_count);
                GPUNTT_CUDA_CHECK(cudaGetLastError());
            }
            break;
            default:
                throw std::invalid_argument("Invalid ntt_layout!");
                break;
        }
    }

    template <typename T>
    __host__ void
    GPU_INTT(typename std::make_unsigned<T>::type* device_in, T* device_out,
             Root<typename std::make_unsigned<T>::type>* root_of_unity_table,
             Modulus<typename std::make_unsigned<T>::type>* modulus,
             ntt_rns_configuration<typename std::make_unsigned<T>::type> cfg,
             int batch_size, int mod_count)
    {
        using TU = typename std::make_unsigned<T>::type;

        check_batch_size(cfg.n_power, batch_size);

        switch (cfg.ntt_layout)
        {
            case PerPolynomial:
            {
                if ((cfg.n_power <= 0 || cfg.n_power >= 29))
                {
                    throw std::invalid_argument("Invalid n_power range!");
                }

                inverse_per_polynomial(
                    device_in, device_out, root_of_unity_table,
                    RnsModulus<TU>{modulus, mod_count, cfg.mod_inverse},
                    cfg.n_power, cfg.reduction_poly, batch_size, cfg.stream);
            }
            break;
            case PerCoefficient:
            {
                if ((cfg.n_power <= 0 || cfg.n_power >= 10))
                {
                    throw std::invalid_argument("Invalid n_power range!");
                }

                if ((batch_size & (batch_size - 1)) != 0)
                {
                    throw std::invalid_argument(
                        "batch_size must be a power of two for "
                        "PerCoefficient layout!");
                }

                int log_batch_size = log2(batch_size);
                int total_size = 1 << (cfg.n_power + log_batch_size);
                int total_block_thread = 512;
                int total_block_count = total_size / (total_block_thread * 2);
                int blockdim_y = 1 << (cfg.n_power - 1);
                int blockdim_x = total_block_thread / blockdim_y;
                InverseCoreTranspose<<<
                    dim3(total_block_count, 1, 1),
                    dim3(blockdim_x, blockdim_y, 1),
                    // Data and one root table per column of the block.
                    (total_block_thread * 2 * 2) * sizeof(T), cfg.stream>>>(
                    device_in, device_out, root_of_unity_table, modulus,
                    cfg.mod_inverse, cfg.n_power, log_batch_size,
                    (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                    mod_count);
                GPUNTT_CUDA_CHECK(cudaGetLastError());
            }
            break;
            default:
                throw std::invalid_argument("Invalid ntt_layout!");
                break;
        }
    }

    template <typename T>
    __host__ void GPU_NTT_Inplace(T* device_inout, Root<T>* root_of_unity_table,
                                  Modulus<T> modulus, ntt_configuration<T> cfg,
                                  int batch_size)
    {
        GPU_NTT(device_inout, device_inout, root_of_unity_table, modulus, cfg,
                batch_size);
    }

    template <typename T>
    __host__ void GPU_NTT_Inplace(T* device_inout, Root<T>* root_of_unity_table,
                                  Modulus<T>* modulus,
                                  ntt_rns_configuration<T> cfg, int batch_size,
                                  int mod_count)
    {
        GPU_NTT(device_inout, device_inout, root_of_unity_table, modulus, cfg,
                batch_size, mod_count);
    }

    template <typename T>
    __host__ void GPU_INTT_Inplace(T* device_inout,
                                   Root<T>* root_of_unity_table,
                                   Modulus<T> modulus, ntt_configuration<T> cfg,
                                   int batch_size)
    {
        GPU_INTT(device_inout, device_inout, root_of_unity_table, modulus, cfg,
                 batch_size);
    }

    template <typename T>
    __host__ void
    GPU_INTT_Inplace(T* device_inout, Root<T>* root_of_unity_table,
                     Modulus<T>* modulus, ntt_rns_configuration<T> cfg,
                     int batch_size, int mod_count)
    {
        GPU_INTT(device_inout, device_inout, root_of_unity_table, modulus, cfg,
                 batch_size, mod_count);
    }

    ////////////////////////////////////
    // Modulus Ordered
    ////////////////////////////////////

    template <typename T>
    __host__ void
    GPU_NTT_Modulus_Ordered(T* device_in, T* device_out,
                            Root<T>* root_of_unity_table, Modulus<T>* modulus,
                            ntt_rns_configuration<T> cfg, int batch_size,
                            int mod_count, int* order)
    {
        check_batch_size(cfg.n_power, batch_size);

        if ((cfg.n_power <= 9 || cfg.n_power >= 29))
        {
            throw std::invalid_argument("Invalid n_power range!");
        }

        if (cfg.ntt_type != FORWARD && cfg.ntt_type != INVERSE)
        {
            throw std::invalid_argument("Invalid ntt_type!");
        }

        ordered_stages(
            device_in, device_out, root_of_unity_table,
            OrderedRnsModulus<T>{modulus, mod_count, order, cfg.mod_inverse},
            SequentialPolys{}, cfg, batch_size);
    }

    template <typename T>
    __host__ void GPU_NTT_Modulus_Ordered_Inplace(
        T* device_inout, Root<T>* root_of_unity_table, Modulus<T>* modulus,
        ntt_rns_configuration<T> cfg, int batch_size, int mod_count, int* order)
    {
        GPU_NTT_Modulus_Ordered(device_inout, device_inout, root_of_unity_table,
                                modulus, cfg, batch_size, mod_count, order);
    }

    ////////////////////////////////////
    // Poly Ordered
    ////////////////////////////////////

    template <typename T>
    __host__ void
    GPU_NTT_Poly_Ordered(T* device_in, T* device_out,
                         Root<T>* root_of_unity_table, Modulus<T>* modulus,
                         ntt_rns_configuration<T> cfg, int batch_size,
                         int mod_count, int* order)
    {
        check_batch_size(cfg.n_power, batch_size);

        if ((cfg.n_power <= 9 || cfg.n_power >= 29))
        {
            throw std::invalid_argument("Invalid n_power range!");
        }

        if (cfg.ntt_type != FORWARD && cfg.ntt_type != INVERSE)
        {
            throw std::invalid_argument("Invalid ntt_type!");
        }

        ordered_stages(device_in, device_out, root_of_unity_table,
                       RnsModulus<T>{modulus, mod_count, cfg.mod_inverse},
                       OrderedPolys{order}, cfg, batch_size);
    }

    template <typename T>
    __host__ void GPU_NTT_Poly_Ordered_Inplace(
        T* device_inout, Root<T>* root_of_unity_table, Modulus<T>* modulus,
        ntt_rns_configuration<T> cfg, int batch_size, int mod_count, int* order)
    {
        GPU_NTT_Poly_Ordered(device_inout, device_inout, root_of_unity_table,
                             modulus, cfg, batch_size, mod_count, order);
    }

    ////////////////////////////////////
    // Explicit Template Specializations
    ////////////////////////////////////

    template <> struct ntt_configuration<Data32>
    {
        int n_power;
        type ntt_type;
        NTTLayout ntt_layout;
        ReductionPolynomial reduction_poly;
        Ninverse<Data32> mod_inverse;
        cudaStream_t stream;
    };

    template <> struct ntt_configuration<Data64>
    {
        int n_power;
        type ntt_type;
        NTTLayout ntt_layout;
        ReductionPolynomial reduction_poly;
        Ninverse<Data64> mod_inverse;
        cudaStream_t stream;
    };

    template <> struct ntt_rns_configuration<Data32>
    {
        int n_power;
        type ntt_type;
        NTTLayout ntt_layout;
        ReductionPolynomial reduction_poly;
        Ninverse<Data32>* mod_inverse;
        cudaStream_t stream;
    };

    template <> struct ntt_rns_configuration<Data64>
    {
        int n_power;
        type ntt_type;
        NTTLayout ntt_layout;
        ReductionPolynomial reduction_poly;
        Ninverse<Data64>* mod_inverse;
        cudaStream_t stream;
    };

    // Explicit instantiations. The kernels are declared in the public
    // header, so every supported kernel and host function is
    // instantiated here. Signed input (NTT) and output (INTT) are
    // supported by the single modulus and RNS variants only.

    template <typename T>
    using UnsignedOf = typename std::make_unsigned<T>::type;

#define GPUNTT_INSTANTIATE_ALL_TYPES(T)                                        \
    template __global__ void ForwardCoreTranspose<T>(                          \
        T*, UnsignedOf<T>*, const Root<UnsignedOf<T>>* __restrict__,           \
        const Modulus<UnsignedOf<T>>, int, int, bool);                         \
    template __global__ void ForwardCoreTranspose<T>(                          \
        T*, UnsignedOf<T>*, const Root<UnsignedOf<T>>* __restrict__,           \
        const Modulus<UnsignedOf<T>>*, int, int, bool, int);                   \
    template __global__ void InverseCoreTranspose<T>(                          \
        UnsignedOf<T>*, T*, const Root<UnsignedOf<T>>* __restrict__,           \
        Modulus<UnsignedOf<T>>, Ninverse<UnsignedOf<T>>, int, int, bool);      \
    template __global__ void InverseCoreTranspose<T>(                          \
        UnsignedOf<T>*, T*, const Root<UnsignedOf<T>>* __restrict__,           \
        Modulus<UnsignedOf<T>>*, Ninverse<UnsignedOf<T>>*, int, int, bool,     \
        int);                                                                  \
    template __host__ void GPU_NTT<T>(                                         \
        T*, UnsignedOf<T>*, Root<UnsignedOf<T>>*, Modulus<UnsignedOf<T>>,      \
        ntt_configuration<UnsignedOf<T>>, int);                                \
    template __host__ void GPU_INTT<T>(                                        \
        UnsignedOf<T>*, T*, Root<UnsignedOf<T>>*, Modulus<UnsignedOf<T>>,      \
        ntt_configuration<UnsignedOf<T>>, int);                                \
    template __host__ void GPU_NTT<T>(                                         \
        T*, UnsignedOf<T>*, Root<UnsignedOf<T>>*, Modulus<UnsignedOf<T>>*,     \
        ntt_rns_configuration<UnsignedOf<T>>, int, int);                       \
    template __host__ void GPU_INTT<T>(                                        \
        UnsignedOf<T>*, T*, Root<UnsignedOf<T>>*, Modulus<UnsignedOf<T>>*,     \
        ntt_rns_configuration<UnsignedOf<T>>, int, int);

#define GPUNTT_INSTANTIATE_UNSIGNED_TYPES(T)                                   \
    template __host__ void GPU_NTT_Inplace<T>(T*, Root<T>*, Modulus<T>,        \
                                              ntt_configuration<T>, int);      \
    template __host__ void GPU_INTT_Inplace<T>(T*, Root<T>*, Modulus<T>,       \
                                               ntt_configuration<T>, int);     \
    template __host__ void GPU_NTT_Inplace<T>(                                 \
        T*, Root<T>*, Modulus<T>*, ntt_rns_configuration<T>, int, int);        \
    template __host__ void GPU_INTT_Inplace<T>(                                \
        T*, Root<T>*, Modulus<T>*, ntt_rns_configuration<T>, int, int);        \
    template __host__ void GPU_NTT_Modulus_Ordered<T>(                         \
        T*, T*, Root<T>*, Modulus<T>*, ntt_rns_configuration<T>, int, int,     \
        int*);                                                                 \
    template __host__ void GPU_NTT_Modulus_Ordered_Inplace<T>(                 \
        T*, Root<T>*, Modulus<T>*, ntt_rns_configuration<T>, int, int, int*);  \
    template __host__ void GPU_NTT_Poly_Ordered<T>(                            \
        T*, T*, Root<T>*, Modulus<T>*, ntt_rns_configuration<T>, int, int,     \
        int*);                                                                 \
    template __host__ void GPU_NTT_Poly_Ordered_Inplace<T>(                    \
        T*, Root<T>*, Modulus<T>*, ntt_rns_configuration<T>, int, int, int*);

    GPUNTT_INSTANTIATE_ALL_TYPES(Data32)
    GPUNTT_INSTANTIATE_ALL_TYPES(Data64)
    GPUNTT_INSTANTIATE_ALL_TYPES(Data32s)
    GPUNTT_INSTANTIATE_ALL_TYPES(Data64s)
    GPUNTT_INSTANTIATE_UNSIGNED_TYPES(Data32)
    GPUNTT_INSTANTIATE_UNSIGNED_TYPES(Data64)

#undef GPUNTT_INSTANTIATE_ALL_TYPES
#undef GPUNTT_INSTANTIATE_UNSIGNED_TYPES

} // namespace gpuntt
