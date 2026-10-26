// Copyright 2022 Ulvetanna Inc.
// Copyright 2023-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan
// Paper: https://eprint.iacr.org/2023/1410

#include "gpuntt/ntt_merge/ntt.cuh"
#include "gpuntt/ntt_merge/descriptor.cuh"
#include "gpuntt/ntt_merge/ntt_device.cuh"

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
    __global__ void ForwardCoreLowRing(
        T* polynomial_in, typename std::make_unsigned<T>::type* polynomial_out,
        const Root<typename std::make_unsigned<
            T>::type>* __restrict__ root_of_unity_table,
        Modulus<typename std::make_unsigned<T>::type> modulus, int shared_index,
        int N_power, bool reduction_poly_check, int total_batch)
    {
        using TU = typename std::make_unsigned<T>::type;

        ForwardCoreLowRingBody(polynomial_in, polynomial_out,
                               root_of_unity_table, SingleModulus<TU>{modulus},
                               shared_index, N_power, reduction_poly_check,
                               total_batch);
    }

    template <typename T>
    __global__ void
    ForwardCoreLowRing(T* polynomial_in,
                       typename std::make_unsigned<T>::type* polynomial_out,
                       const Root<typename std::make_unsigned<
                           T>::type>* __restrict__ root_of_unity_table,
                       Modulus<typename std::make_unsigned<T>::type>* modulus,
                       int shared_index, int N_power, bool reduction_poly_check,
                       int total_batch, int mod_count)
    {
        using TU = typename std::make_unsigned<T>::type;

        ForwardCoreLowRingBody(polynomial_in, polynomial_out,
                               root_of_unity_table,
                               RnsModulus<TU>{modulus, mod_count}, shared_index,
                               N_power, reduction_poly_check, total_batch);
    }

    template <typename T>
    __global__ void InverseCoreLowRing(
        typename std::make_unsigned<T>::type* polynomial_in, T* polynomial_out,
        const Root<typename std::make_unsigned<
            T>::type>* __restrict__ inverse_root_of_unity_table,
        Modulus<typename std::make_unsigned<T>::type> modulus, int shared_index,
        int N_power, Ninverse<typename std::make_unsigned<T>::type> n_inverse,
        bool reduction_poly_check, int total_batch)
    {
        using TU = typename std::make_unsigned<T>::type;

        InverseCoreLowRingBody(
            polynomial_in, polynomial_out, inverse_root_of_unity_table,
            SingleModulus<TU>{modulus, n_inverse}, shared_index, N_power,
            reduction_poly_check, total_batch);
    }

    template <typename T>
    __global__ void InverseCoreLowRing(
        typename std::make_unsigned<T>::type* polynomial_in, T* polynomial_out,
        const Root<typename std::make_unsigned<
            T>::type>* __restrict__ inverse_root_of_unity_table,
        Modulus<typename std::make_unsigned<T>::type>* modulus,
        int shared_index, int N_power,
        Ninverse<typename std::make_unsigned<T>::type>* n_inverse,
        bool reduction_poly_check, int total_batch, int mod_count)
    {
        using TU = typename std::make_unsigned<T>::type;

        InverseCoreLowRingBody(
            polynomial_in, polynomial_out, inverse_root_of_unity_table,
            RnsModulus<TU>{modulus, mod_count, n_inverse}, shared_index,
            N_power, reduction_poly_check, total_batch);
    }

    template <typename T>
    __global__ void
    ForwardCore(T* polynomial_in,
                typename std::make_unsigned<T>::type* polynomial_out,
                const Root<typename std::make_unsigned<
                    T>::type>* __restrict__ root_of_unity_table,
                Modulus<typename std::make_unsigned<T>::type> modulus,
                int shared_index, int logm, int outer_iteration_count,
                int N_power, bool not_last_kernel, bool reduction_poly_check)
    {
        using TU = typename std::make_unsigned<T>::type;

        ForwardCoreBody<false>(polynomial_in, polynomial_out,
                               root_of_unity_table, SingleModulus<TU>{modulus},
                               SequentialPolys{}, shared_index, logm,
                               outer_iteration_count, N_power, not_last_kernel,
                               reduction_poly_check);
    }

    template <typename T>
    __global__ void ForwardCore(
        T* polynomial_in, typename std::make_unsigned<T>::type* polynomial_out,
        const Root<typename std::make_unsigned<
            T>::type>* __restrict__ root_of_unity_table,
        Modulus<typename std::make_unsigned<T>::type>* modulus,
        int shared_index, int logm, int outer_iteration_count, int N_power,
        bool not_last_kernel, bool reduction_poly_check, int mod_count)
    {
        using TU = typename std::make_unsigned<T>::type;

        ForwardCoreBody<false>(
            polynomial_in, polynomial_out, root_of_unity_table,
            RnsModulus<TU>{modulus, mod_count}, SequentialPolys{}, shared_index,
            logm, outer_iteration_count, N_power, not_last_kernel,
            reduction_poly_check);
    }

    template <typename T>
    __global__ void
    ForwardCore_(T* polynomial_in,
                 typename std::make_unsigned<T>::type* polynomial_out,
                 const Root<typename std::make_unsigned<
                     T>::type>* __restrict__ root_of_unity_table,
                 Modulus<typename std::make_unsigned<T>::type> modulus,
                 int shared_index, int logm, int outer_iteration_count,
                 int N_power, bool not_last_kernel, bool reduction_poly_check)
    {
        using TU = typename std::make_unsigned<T>::type;

        ForwardCoreBody<true>(polynomial_in, polynomial_out,
                              root_of_unity_table, SingleModulus<TU>{modulus},
                              SequentialPolys{}, shared_index, logm,
                              outer_iteration_count, N_power, not_last_kernel,
                              reduction_poly_check);
    }

    template <typename T>
    __global__ void ForwardCore_(
        T* polynomial_in, typename std::make_unsigned<T>::type* polynomial_out,
        const Root<typename std::make_unsigned<
            T>::type>* __restrict__ root_of_unity_table,
        Modulus<typename std::make_unsigned<T>::type>* modulus,
        int shared_index, int logm, int outer_iteration_count, int N_power,
        bool not_last_kernel, bool reduction_poly_check, int mod_count)
    {
        using TU = typename std::make_unsigned<T>::type;

        ForwardCoreBody<true>(
            polynomial_in, polynomial_out, root_of_unity_table,
            RnsModulus<TU>{modulus, mod_count}, SequentialPolys{}, shared_index,
            logm, outer_iteration_count, N_power, not_last_kernel,
            reduction_poly_check);
    }

    template <typename T>
    __global__ void InverseCore(
        typename std::make_unsigned<T>::type* polynomial_in, T* polynomial_out,
        const Root<typename std::make_unsigned<
            T>::type>* __restrict__ inverse_root_of_unity_table,
        Modulus<typename std::make_unsigned<T>::type> modulus, int shared_index,
        int logm, int k, int outer_iteration_count, int N_power,
        Ninverse<typename std::make_unsigned<T>::type> n_inverse,
        bool last_kernel, bool reduction_poly_check)
    {
        using TU = typename std::make_unsigned<T>::type;

        InverseCoreBody<false>(
            polynomial_in, polynomial_out, inverse_root_of_unity_table,
            SingleModulus<TU>{modulus, n_inverse}, SequentialPolys{},
            shared_index, logm, k, outer_iteration_count, N_power, last_kernel,
            reduction_poly_check);
    }

    template <typename T>
    __global__ void InverseCore(
        typename std::make_unsigned<T>::type* polynomial_in, T* polynomial_out,
        const Root<typename std::make_unsigned<
            T>::type>* __restrict__ inverse_root_of_unity_table,
        Modulus<typename std::make_unsigned<T>::type>* modulus,
        int shared_index, int logm, int k, int outer_iteration_count,
        int N_power, Ninverse<typename std::make_unsigned<T>::type>* n_inverse,
        bool last_kernel, bool reduction_poly_check, int mod_count)
    {
        using TU = typename std::make_unsigned<T>::type;

        InverseCoreBody<false>(
            polynomial_in, polynomial_out, inverse_root_of_unity_table,
            RnsModulus<TU>{modulus, mod_count, n_inverse}, SequentialPolys{},
            shared_index, logm, k, outer_iteration_count, N_power, last_kernel,
            reduction_poly_check);
    }

    template <typename T>
    __global__ void InverseCore_(
        typename std::make_unsigned<T>::type* polynomial_in, T* polynomial_out,
        const Root<typename std::make_unsigned<
            T>::type>* __restrict__ inverse_root_of_unity_table,
        Modulus<typename std::make_unsigned<T>::type> modulus, int shared_index,
        int logm, int k, int outer_iteration_count, int N_power,
        Ninverse<typename std::make_unsigned<T>::type> n_inverse,
        bool last_kernel, bool reduction_poly_check)
    {
        using TU = typename std::make_unsigned<T>::type;

        InverseCoreBody<true>(
            polynomial_in, polynomial_out, inverse_root_of_unity_table,
            SingleModulus<TU>{modulus, n_inverse}, SequentialPolys{},
            shared_index, logm, k, outer_iteration_count, N_power, last_kernel,
            reduction_poly_check);
    }

    template <typename T>
    __global__ void InverseCore_(
        typename std::make_unsigned<T>::type* polynomial_in, T* polynomial_out,
        const Root<typename std::make_unsigned<
            T>::type>* __restrict__ inverse_root_of_unity_table,
        Modulus<typename std::make_unsigned<T>::type>* modulus,
        int shared_index, int logm, int k, int outer_iteration_count,
        int N_power, Ninverse<typename std::make_unsigned<T>::type>* n_inverse,
        bool last_kernel, bool reduction_poly_check, int mod_count)
    {
        using TU = typename std::make_unsigned<T>::type;

        InverseCoreBody<true>(
            polynomial_in, polynomial_out, inverse_root_of_unity_table,
            RnsModulus<TU>{modulus, mod_count, n_inverse}, SequentialPolys{},
            shared_index, logm, k, outer_iteration_count, N_power, last_kernel,
            reduction_poly_check);
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

    // n_power must be in [1, 28]; the host API checks it before the lookup.
    template <typename T>
    const KernelStages& forward_kernel_configs(int n_power)
    {
        return forward_kernel_table<T>[n_power];
    }

    template <typename T>
    const KernelStages& inverse_kernel_configs(int n_power)
    {
        return inverse_kernel_table<T>[n_power];
    }

    // Launchers for the multi-stage kernel families. swapped selects the _
    // variant, which swaps the roles of blockIdx.x and blockIdx.y.
    struct MergeKernels
    {
        template <typename TIn, typename TOut, typename... Args>
        static void forward(bool swapped, dim3 grid, dim3 block,
                            size_t shared_memory, cudaStream_t stream, TIn* in,
                            TOut* out, Args... args)
        {
            if (swapped)
            {
                ForwardCore_<<<grid, block, shared_memory, stream>>>(in, out,
                                                                     args...);
            }
            else
            {
                ForwardCore<<<grid, block, shared_memory, stream>>>(in, out,
                                                                    args...);
            }
        }

        template <typename TIn, typename TOut, typename... Args>
        static void inverse(bool swapped, dim3 grid, dim3 block,
                            size_t shared_memory, cudaStream_t stream, TIn* in,
                            TOut* out, Args... args)
        {
            if (swapped)
            {
                InverseCore_<<<grid, block, shared_memory, stream>>>(in, out,
                                                                     args...);
            }
            else
            {
                InverseCore<<<grid, block, shared_memory, stream>>>(in, out,
                                                                    args...);
            }
        }
    };

    struct ModulusOrderedKernels
    {
        template <typename TIn, typename TOut, typename... Args>
        static void forward(bool swapped, dim3 grid, dim3 block,
                            size_t shared_memory, cudaStream_t stream, TIn* in,
                            TOut* out, Args... args)
        {
            if (swapped)
            {
                ForwardCoreModulusOrdered_<<<grid, block, shared_memory,
                                             stream>>>(in, out, args...);
            }
            else
            {
                ForwardCoreModulusOrdered<<<grid, block, shared_memory,
                                            stream>>>(in, out, args...);
            }
        }

        template <typename TIn, typename TOut, typename... Args>
        static void inverse(bool swapped, dim3 grid, dim3 block,
                            size_t shared_memory, cudaStream_t stream, TIn* in,
                            TOut* out, Args... args)
        {
            if (swapped)
            {
                InverseCoreModulusOrdered_<<<grid, block, shared_memory,
                                             stream>>>(in, out, args...);
            }
            else
            {
                InverseCoreModulusOrdered<<<grid, block, shared_memory,
                                            stream>>>(in, out, args...);
            }
        }
    };

    struct PolyOrderedKernels
    {
        template <typename TIn, typename TOut, typename... Args>
        static void forward(bool swapped, dim3 grid, dim3 block,
                            size_t shared_memory, cudaStream_t stream, TIn* in,
                            TOut* out, Args... args)
        {
            if (swapped)
            {
                ForwardCorePolyOrdered_<<<grid, block, shared_memory, stream>>>(
                    in, out, args...);
            }
            else
            {
                ForwardCorePolyOrdered<<<grid, block, shared_memory, stream>>>(
                    in, out, args...);
            }
        }

        template <typename TIn, typename TOut, typename... Args>
        static void inverse(bool swapped, dim3 grid, dim3 block,
                            size_t shared_memory, cudaStream_t stream, TIn* in,
                            TOut* out, Args... args)
        {
            if (swapped)
            {
                InverseCorePolyOrdered_<<<grid, block, shared_memory, stream>>>(
                    in, out, args...);
            }
            else
            {
                InverseCorePolyOrdered<<<grid, block, shared_memory, stream>>>(
                    in, out, args...);
            }
        }
    };

    // Runs the multi-stage forward NTT kernels of one transform. Kernel i
    // reads the output of kernel i - 1 (the first one reads device_in) and
    // writes device_out. For log2(N) >= 25 the last kernel is the _ variant.
    // extra holds the trailing kernel arguments of the family (mod_count,
    // order).
    template <typename Kernels, typename T, typename ModulusArg,
              typename... Extra>
    void forward_stages(
        T* device_in, typename std::make_unsigned<T>::type* device_out,
        Root<typename std::make_unsigned<T>::type>* root_of_unity_table,
        ModulusArg modulus, int n_power, ReductionPolynomial reduction_poly,
        int batch_size, cudaStream_t stream, Extra... extra)
    {
        using TU = typename std::make_unsigned<T>::type;

        const KernelStages& kernels = forward_kernel_configs<TU>(n_power);
        const bool swap_last = (n_power >= 25);
        const bool x_n_minus =
            (reduction_poly == ReductionPolynomial::X_N_minus);

        for (int i = 0; i < kernels.size(); i++)
        {
            const KernelConfig& k = kernels[i];
            const dim3 grid(k.griddim_x, k.griddim_y, batch_size);
            const dim3 block(k.blockdim_x, k.blockdim_y);
            const bool swapped = swap_last && (i + 1 == kernels.size());
            if (i == 0)
            {
                Kernels::forward(swapped, grid, block, k.shared_memory, stream,
                                 device_in, device_out, root_of_unity_table,
                                 modulus, k.shared_index, k.logm,
                                 k.outer_iteration_count, n_power,
                                 k.not_last_kernel, x_n_minus, extra...);
            }
            else
            {
                Kernels::forward(swapped, grid, block, k.shared_memory, stream,
                                 device_out, device_out, root_of_unity_table,
                                 modulus, k.shared_index, k.logm,
                                 k.outer_iteration_count, n_power,
                                 k.not_last_kernel, x_n_minus, extra...);
            }
            GPUNTT_CUDA_CHECK(cudaGetLastError());
        }
    }

    // Inverse counterpart of forward_stages. Only the last kernel writes T
    // (signed output); the others keep unsigned values in device_out. For
    // log2(N) >= 25 the first kernel is the _ variant. The inverse kernel
    // table stores the last_kernel flag in not_last_kernel.
    template <typename Kernels, typename T, typename ModulusArg,
              typename NinverseArg, typename... Extra>
    void inverse_stages(
        typename std::make_unsigned<T>::type* device_in, T* device_out,
        Root<typename std::make_unsigned<T>::type>* root_of_unity_table,
        ModulusArg modulus, NinverseArg n_inverse, int n_power,
        ReductionPolynomial reduction_poly, int batch_size, cudaStream_t stream,
        Extra... extra)
    {
        using TU = typename std::make_unsigned<T>::type;

        TU* device_out_unsigned = reinterpret_cast<TU*>(device_out);
        const KernelStages& kernels = inverse_kernel_configs<TU>(n_power);
        const bool swap_first = (n_power >= 25);
        const bool x_n_minus =
            (reduction_poly == ReductionPolynomial::X_N_minus);

        for (int i = 0; i < kernels.size(); i++)
        {
            const KernelConfig& k = kernels[i];
            const dim3 grid(k.griddim_x, k.griddim_y, batch_size);
            const dim3 block(k.blockdim_x, k.blockdim_y);
            const bool swapped = swap_first && (i == 0);
            TU* in = (i == 0) ? device_in : device_out_unsigned;
            if (i + 1 == kernels.size())
            {
                Kernels::inverse(swapped, grid, block, k.shared_memory, stream,
                                 in, device_out, root_of_unity_table, modulus,
                                 k.shared_index, k.logm, k.k,
                                 k.outer_iteration_count, n_power, n_inverse,
                                 k.not_last_kernel, x_n_minus, extra...);
            }
            else
            {
                Kernels::inverse(swapped, grid, block, k.shared_memory, stream,
                                 in, device_out_unsigned, root_of_unity_table,
                                 modulus, k.shared_index, k.logm, k.k,
                                 k.outer_iteration_count, n_power, n_inverse,
                                 k.not_last_kernel, x_n_minus, extra...);
            }
            GPUNTT_CUDA_CHECK(cudaGetLastError());
        }
    }

    // Runs the low ring size forward kernel, which transforms several
    // polynomials per block.
    template <typename T, typename ModulusArg, typename... Extra>
    void forward_low_ring(
        T* device_in, typename std::make_unsigned<T>::type* device_out,
        Root<typename std::make_unsigned<T>::type>* root_of_unity_table,
        ModulusArg modulus, int n_power, ReductionPolynomial reduction_poly,
        int batch_size, cudaStream_t stream, Extra... extra)
    {
        using TU = typename std::make_unsigned<T>::type;

        const KernelConfig& k = forward_kernel_configs<TU>(n_power)[0];
        ForwardCoreLowRing<<<
            dim3((batch_size + (k.blockdim_y - 1)) / k.blockdim_y, 1, 1),
            dim3(k.blockdim_x, k.blockdim_y), k.shared_memory, stream>>>(
            device_in, device_out, root_of_unity_table, modulus, k.shared_index,
            n_power, (reduction_poly == ReductionPolynomial::X_N_minus),
            batch_size, extra...);
        GPUNTT_CUDA_CHECK(cudaGetLastError());
    }

    template <typename T, typename ModulusArg, typename NinverseArg,
              typename... Extra>
    void inverse_low_ring(
        typename std::make_unsigned<T>::type* device_in, T* device_out,
        Root<typename std::make_unsigned<T>::type>* root_of_unity_table,
        ModulusArg modulus, NinverseArg n_inverse, int n_power,
        ReductionPolynomial reduction_poly, int batch_size, cudaStream_t stream,
        Extra... extra)
    {
        using TU = typename std::make_unsigned<T>::type;

        const KernelConfig& k = inverse_kernel_configs<TU>(n_power)[0];
        InverseCoreLowRing<<<
            dim3((batch_size + (k.blockdim_y - 1)) / k.blockdim_y, 1, 1),
            dim3(k.blockdim_x, k.blockdim_y), k.shared_memory, stream>>>(
            device_in, device_out, root_of_unity_table, modulus, k.shared_index,
            n_power, n_inverse,
            (reduction_poly == ReductionPolynomial::X_N_minus), batch_size,
            extra...);
        GPUNTT_CUDA_CHECK(cudaGetLastError());
    }

    template <typename T>
    __host__ void
    GPU_NTT(T* device_in, typename std::make_unsigned<T>::type* device_out,
            Root<typename std::make_unsigned<T>::type>* root_of_unity_table,
            Modulus<typename std::make_unsigned<T>::type> modulus,
            ntt_configuration<typename std::make_unsigned<T>::type> cfg,
            int batch_size)
    {
        check_batch_size(cfg.n_power, batch_size);

        switch (cfg.ntt_layout)
        {
            case PerPolynomial:
            {
                if ((cfg.n_power <= 0 || cfg.n_power >= 29))
                {
                    throw std::invalid_argument("Invalid n_power range!");
                }

                if (cfg.n_power < 10)
                {
                    forward_low_ring(device_in, device_out, root_of_unity_table,
                                     modulus, cfg.n_power, cfg.reduction_poly,
                                     batch_size, cfg.stream);
                }
                else
                {
                    forward_stages<MergeKernels>(
                        device_in, device_out, root_of_unity_table, modulus,
                        cfg.n_power, cfg.reduction_poly, batch_size,
                        cfg.stream);
                }
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
        check_batch_size(cfg.n_power, batch_size);

        switch (cfg.ntt_layout)
        {
            case PerPolynomial:
            {
                if ((cfg.n_power <= 0 || cfg.n_power >= 29))
                {
                    throw std::invalid_argument("Invalid n_power range!");
                }

                if (cfg.n_power < 11)
                {
                    inverse_low_ring(device_in, device_out, root_of_unity_table,
                                     modulus, cfg.mod_inverse, cfg.n_power,
                                     cfg.reduction_poly, batch_size,
                                     cfg.stream);
                }
                else
                {
                    inverse_stages<MergeKernels>(
                        device_in, device_out, root_of_unity_table, modulus,
                        cfg.mod_inverse, cfg.n_power, cfg.reduction_poly,
                        batch_size, cfg.stream);
                }
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
        check_batch_size(cfg.n_power, batch_size);

        switch (cfg.ntt_layout)
        {
            case PerPolynomial:
            {
                if ((cfg.n_power <= 0 || cfg.n_power >= 29))
                {
                    throw std::invalid_argument("Invalid n_power range!");
                }

                if (cfg.n_power < 10)
                {
                    forward_low_ring(device_in, device_out, root_of_unity_table,
                                     modulus, cfg.n_power, cfg.reduction_poly,
                                     batch_size, cfg.stream, mod_count);
                }
                else
                {
                    forward_stages<MergeKernels>(
                        device_in, device_out, root_of_unity_table, modulus,
                        cfg.n_power, cfg.reduction_poly, batch_size, cfg.stream,
                        mod_count);
                }
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
        check_batch_size(cfg.n_power, batch_size);

        switch (cfg.ntt_layout)
        {
            case PerPolynomial:
            {
                if ((cfg.n_power <= 0 || cfg.n_power >= 29))
                {
                    throw std::invalid_argument("Invalid n_power range!");
                }

                if (cfg.n_power < 11)
                {
                    inverse_low_ring(device_in, device_out, root_of_unity_table,
                                     modulus, cfg.mod_inverse, cfg.n_power,
                                     cfg.reduction_poly, batch_size, cfg.stream,
                                     mod_count);
                }
                else
                {
                    inverse_stages<MergeKernels>(
                        device_in, device_out, root_of_unity_table, modulus,
                        cfg.mod_inverse, cfg.n_power, cfg.reduction_poly,
                        batch_size, cfg.stream, mod_count);
                }
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
    __global__ void ForwardCoreModulusOrdered(
        T* polynomial_in, T* polynomial_out, Root<T>* root_of_unity_table,
        Modulus<T>* modulus, int shared_index, int logm,
        int outer_iteration_count, int N_power, bool not_last_kernel,
        bool reduction_poly_check, int mod_count, int* order)
    {
        ForwardCoreBody<false>(
            polynomial_in, polynomial_out, root_of_unity_table,
            OrderedRnsModulus<T>{modulus, mod_count, order}, SequentialPolys{},
            shared_index, logm, outer_iteration_count, N_power, not_last_kernel,
            reduction_poly_check);
    }

    template <typename T>
    __global__ void ForwardCoreModulusOrdered_(
        T* polynomial_in, T* polynomial_out, Root<T>* root_of_unity_table,
        Modulus<T>* modulus, int shared_index, int logm,
        int outer_iteration_count, int N_power, bool not_last_kernel,
        bool reduction_poly_check, int mod_count, int* order)
    {
        ForwardCoreBody<true>(
            polynomial_in, polynomial_out, root_of_unity_table,
            OrderedRnsModulus<T>{modulus, mod_count, order}, SequentialPolys{},
            shared_index, logm, outer_iteration_count, N_power, not_last_kernel,
            reduction_poly_check);
    }

    template <typename T>
    __global__ void InverseCoreModulusOrdered(
        T* polynomial_in, T* polynomial_out,
        Root<T>* inverse_root_of_unity_table, Modulus<T>* modulus,
        int shared_index, int logm, int k, int outer_iteration_count,
        int N_power, Ninverse<T>* n_inverse, bool last_kernel,
        bool reduction_poly_check, int mod_count, int* order)
    {
        InverseCoreBody<false>(
            polynomial_in, polynomial_out, inverse_root_of_unity_table,
            OrderedRnsModulus<T>{modulus, mod_count, order, n_inverse},
            SequentialPolys{}, shared_index, logm, k, outer_iteration_count,
            N_power, last_kernel, reduction_poly_check);
    }

    template <typename T>
    __global__ void InverseCoreModulusOrdered_(
        T* polynomial_in, T* polynomial_out,
        Root<T>* inverse_root_of_unity_table, Modulus<T>* modulus,
        int shared_index, int logm, int k, int outer_iteration_count,
        int N_power, Ninverse<T>* n_inverse, bool last_kernel,
        bool reduction_poly_check, int mod_count, int* order)
    {
        InverseCoreBody<true>(
            polynomial_in, polynomial_out, inverse_root_of_unity_table,
            OrderedRnsModulus<T>{modulus, mod_count, order, n_inverse},
            SequentialPolys{}, shared_index, logm, k, outer_iteration_count,
            N_power, last_kernel, reduction_poly_check);
    }

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

        switch (cfg.ntt_type)
        {
            case FORWARD:
                forward_stages<ModulusOrderedKernels>(
                    device_in, device_out, root_of_unity_table, modulus,
                    cfg.n_power, cfg.reduction_poly, batch_size, cfg.stream,
                    mod_count, order);
                break;
            case INVERSE:
                inverse_stages<ModulusOrderedKernels>(
                    device_in, device_out, root_of_unity_table, modulus,
                    cfg.mod_inverse, cfg.n_power, cfg.reduction_poly,
                    batch_size, cfg.stream, mod_count, order);
                break;
            default:
                throw std::invalid_argument("Invalid ntt_type!");
                break;
        }
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
    __global__ void ForwardCorePolyOrdered(
        T* polynomial_in, T* polynomial_out, Root<T>* root_of_unity_table,
        Modulus<T>* modulus, int shared_index, int logm,
        int outer_iteration_count, int N_power, bool not_last_kernel,
        bool reduction_poly_check, int mod_count, int* order)
    {
        ForwardCoreBody<false>(
            polynomial_in, polynomial_out, root_of_unity_table,
            RnsModulus<T>{modulus, mod_count}, OrderedPolys{order},
            shared_index, logm, outer_iteration_count, N_power, not_last_kernel,
            reduction_poly_check);
    }

    template <typename T>
    __global__ void ForwardCorePolyOrdered_(
        T* polynomial_in, T* polynomial_out, Root<T>* root_of_unity_table,
        Modulus<T>* modulus, int shared_index, int logm,
        int outer_iteration_count, int N_power, bool not_last_kernel,
        bool reduction_poly_check, int mod_count, int* order)
    {
        ForwardCoreBody<true>(
            polynomial_in, polynomial_out, root_of_unity_table,
            RnsModulus<T>{modulus, mod_count}, OrderedPolys{order},
            shared_index, logm, outer_iteration_count, N_power, not_last_kernel,
            reduction_poly_check);
    }

    template <typename T>
    __global__ void
    InverseCorePolyOrdered(T* polynomial_in, T* polynomial_out,
                           Root<T>* inverse_root_of_unity_table,
                           Modulus<T>* modulus, int shared_index, int logm,
                           int k, int outer_iteration_count, int N_power,
                           Ninverse<T>* n_inverse, bool last_kernel,
                           bool reduction_poly_check, int mod_count, int* order)
    {
        InverseCoreBody<false>(
            polynomial_in, polynomial_out, inverse_root_of_unity_table,
            RnsModulus<T>{modulus, mod_count, n_inverse}, OrderedPolys{order},
            shared_index, logm, k, outer_iteration_count, N_power, last_kernel,
            reduction_poly_check);
    }

    template <typename T>
    __global__ void InverseCorePolyOrdered_(
        T* polynomial_in, T* polynomial_out,
        Root<T>* inverse_root_of_unity_table, Modulus<T>* modulus,
        int shared_index, int logm, int k, int outer_iteration_count,
        int N_power, Ninverse<T>* n_inverse, bool last_kernel,
        bool reduction_poly_check, int mod_count, int* order)
    {
        InverseCoreBody<true>(
            polynomial_in, polynomial_out, inverse_root_of_unity_table,
            RnsModulus<T>{modulus, mod_count, n_inverse}, OrderedPolys{order},
            shared_index, logm, k, outer_iteration_count, N_power, last_kernel,
            reduction_poly_check);
    }

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

        switch (cfg.ntt_type)
        {
            case FORWARD:
                forward_stages<PolyOrderedKernels>(
                    device_in, device_out, root_of_unity_table, modulus,
                    cfg.n_power, cfg.reduction_poly, batch_size, cfg.stream,
                    mod_count, order);
                break;
            case INVERSE:
                inverse_stages<PolyOrderedKernels>(
                    device_in, device_out, root_of_unity_table, modulus,
                    cfg.mod_inverse, cfg.n_power, cfg.reduction_poly,
                    batch_size, cfg.stream, mod_count, order);
                break;
            default:
                throw std::invalid_argument("Invalid ntt_type!");
                break;
        }
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
    template __global__ void ForwardCoreLowRing<T>(                            \
        T*, UnsignedOf<T>*, const Root<UnsignedOf<T>>* __restrict__,           \
        Modulus<UnsignedOf<T>>, int, int, bool, int);                          \
    template __global__ void ForwardCoreLowRing<T>(                            \
        T*, UnsignedOf<T>*, const Root<UnsignedOf<T>>* __restrict__,           \
        Modulus<UnsignedOf<T>>*, int, int, bool, int, int);                    \
    template __global__ void ForwardCore<T>(                                   \
        T*, UnsignedOf<T>*, const Root<UnsignedOf<T>>* __restrict__,           \
        Modulus<UnsignedOf<T>>, int, int, int, int, bool, bool);               \
    template __global__ void ForwardCore<T>(                                   \
        T*, UnsignedOf<T>*, const Root<UnsignedOf<T>>* __restrict__,           \
        Modulus<UnsignedOf<T>>*, int, int, int, int, bool, bool, int);         \
    template __global__ void ForwardCore_<T>(                                  \
        T*, UnsignedOf<T>*, const Root<UnsignedOf<T>>* __restrict__,           \
        Modulus<UnsignedOf<T>>, int, int, int, int, bool, bool);               \
    template __global__ void ForwardCore_<T>(                                  \
        T*, UnsignedOf<T>*, const Root<UnsignedOf<T>>* __restrict__,           \
        Modulus<UnsignedOf<T>>*, int, int, int, int, bool, bool, int);         \
    template __global__ void InverseCoreLowRing<T>(                            \
        UnsignedOf<T>*, T*, const Root<UnsignedOf<T>>* __restrict__,           \
        Modulus<UnsignedOf<T>>, int, int, Ninverse<UnsignedOf<T>>, bool, int); \
    template __global__ void InverseCoreLowRing<T>(                            \
        UnsignedOf<T>*, T*, const Root<UnsignedOf<T>>* __restrict__,           \
        Modulus<UnsignedOf<T>>*, int, int, Ninverse<UnsignedOf<T>>*, bool,     \
        int, int);                                                             \
    template __global__ void InverseCore<T>(                                   \
        UnsignedOf<T>*, T*, const Root<UnsignedOf<T>>* __restrict__,           \
        Modulus<UnsignedOf<T>>, int, int, int, int, int,                       \
        Ninverse<UnsignedOf<T>>, bool, bool);                                  \
    template __global__ void InverseCore<T>(                                   \
        UnsignedOf<T>*, T*, const Root<UnsignedOf<T>>* __restrict__,           \
        Modulus<UnsignedOf<T>>*, int, int, int, int, int,                      \
        Ninverse<UnsignedOf<T>>*, bool, bool, int);                            \
    template __global__ void InverseCore_<T>(                                  \
        UnsignedOf<T>*, T*, const Root<UnsignedOf<T>>* __restrict__,           \
        Modulus<UnsignedOf<T>>, int, int, int, int, int,                       \
        Ninverse<UnsignedOf<T>>, bool, bool);                                  \
    template __global__ void InverseCore_<T>(                                  \
        UnsignedOf<T>*, T*, const Root<UnsignedOf<T>>* __restrict__,           \
        Modulus<UnsignedOf<T>>*, int, int, int, int, int,                      \
        Ninverse<UnsignedOf<T>>*, bool, bool, int);                            \
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
    template __global__ void ForwardCoreModulusOrdered<T>(                     \
        T*, T*, Root<T>*, Modulus<T>*, int, int, int, int, bool, bool, int,    \
        int*);                                                                 \
    template __global__ void ForwardCoreModulusOrdered_<T>(                    \
        T*, T*, Root<T>*, Modulus<T>*, int, int, int, int, bool, bool, int,    \
        int*);                                                                 \
    template __global__ void InverseCoreModulusOrdered<T>(                     \
        T*, T*, Root<T>*, Modulus<T>*, int, int, int, int, int, Ninverse<T>*,  \
        bool, bool, int, int*);                                                \
    template __global__ void InverseCoreModulusOrdered_<T>(                    \
        T*, T*, Root<T>*, Modulus<T>*, int, int, int, int, int, Ninverse<T>*,  \
        bool, bool, int, int*);                                                \
    template __host__ void GPU_NTT_Modulus_Ordered<T>(                         \
        T*, T*, Root<T>*, Modulus<T>*, ntt_rns_configuration<T>, int, int,     \
        int*);                                                                 \
    template __host__ void GPU_NTT_Modulus_Ordered_Inplace<T>(                 \
        T*, Root<T>*, Modulus<T>*, ntt_rns_configuration<T>, int, int, int*);  \
    template __global__ void ForwardCorePolyOrdered<T>(                        \
        T*, T*, Root<T>*, Modulus<T>*, int, int, int, int, bool, bool, int,    \
        int*);                                                                 \
    template __global__ void ForwardCorePolyOrdered_<T>(                       \
        T*, T*, Root<T>*, Modulus<T>*, int, int, int, int, bool, bool, int,    \
        int*);                                                                 \
    template __global__ void InverseCorePolyOrdered<T>(                        \
        T*, T*, Root<T>*, Modulus<T>*, int, int, int, int, int, Ninverse<T>*,  \
        bool, bool, int, int*);                                                \
    template __global__ void InverseCorePolyOrdered_<T>(                       \
        T*, T*, Root<T>*, Modulus<T>*, int, int, int, int, int, Ninverse<T>*,  \
        bool, bool, int, int*);                                                \
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
