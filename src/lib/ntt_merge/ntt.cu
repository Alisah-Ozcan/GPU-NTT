// Copyright 2022 Ulvetanna Inc.
// Copyright 2023-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan
// Paper: https://eprint.iacr.org/2023/1410

#include "gpuntt/ntt_merge/ntt.cuh"
#include "gpuntt/ntt_merge/ntt_device.cuh"
#include "ntt_launch.cuh"

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
        const typename std::make_unsigned<
            T>::type* __restrict__ root_of_unity_table,
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
        const typename std::make_unsigned<
            T>::type* __restrict__ root_of_unity_table,
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
    __global__ void
    InverseCoreTranspose(typename std::make_unsigned<T>::type* polynomial_in,
                         T* polynomial_out,
                         const typename std::make_unsigned<
                             T>::type* __restrict__ inverse_root_of_unity_table,
                         Modulus<typename std::make_unsigned<T>::type> modulus,
                         typename std::make_unsigned<T>::type n_inverse,
                         int log_row, int log_column, bool reduction_poly_check)
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
        const typename std::make_unsigned<
            T>::type* __restrict__ inverse_root_of_unity_table,
        Modulus<typename std::make_unsigned<T>::type>* modulus,
        typename std::make_unsigned<T>::type* n_inverse, int log_row,
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

    template <typename T>
    __host__ void
    GPU_NTT(T* device_in, typename std::make_unsigned<T>::type* device_out,
            typename std::make_unsigned<T>::type* root_of_unity_table,
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
             typename std::make_unsigned<T>::type* root_of_unity_table,
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
            typename std::make_unsigned<T>::type* root_of_unity_table,
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
             typename std::make_unsigned<T>::type* root_of_unity_table,
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
    __host__ void GPU_NTT_Inplace(T* device_inout, T* root_of_unity_table,
                                  Modulus<T> modulus, ntt_configuration<T> cfg,
                                  int batch_size)
    {
        GPU_NTT(device_inout, device_inout, root_of_unity_table, modulus, cfg,
                batch_size);
    }

    template <typename T>
    __host__ void GPU_NTT_Inplace(T* device_inout, T* root_of_unity_table,
                                  Modulus<T>* modulus,
                                  ntt_rns_configuration<T> cfg, int batch_size,
                                  int mod_count)
    {
        GPU_NTT(device_inout, device_inout, root_of_unity_table, modulus, cfg,
                batch_size, mod_count);
    }

    template <typename T>
    __host__ void GPU_INTT_Inplace(T* device_inout, T* root_of_unity_table,
                                   Modulus<T> modulus, ntt_configuration<T> cfg,
                                   int batch_size)
    {
        GPU_INTT(device_inout, device_inout, root_of_unity_table, modulus, cfg,
                 batch_size);
    }

    template <typename T>
    __host__ void GPU_INTT_Inplace(T* device_inout, T* root_of_unity_table,
                                   Modulus<T>* modulus,
                                   ntt_rns_configuration<T> cfg, int batch_size,
                                   int mod_count)
    {
        GPU_INTT(device_inout, device_inout, root_of_unity_table, modulus, cfg,
                 batch_size, mod_count);
    }

    ////////////////////////////////////
    // Modulus Ordered
    ////////////////////////////////////

    template <typename T>
    __host__ void
    GPU_NTT_Modulus_Ordered(T* device_in, T* device_out, T* root_of_unity_table,
                            Modulus<T>* modulus, ntt_rns_configuration<T> cfg,
                            int batch_size, int mod_count, int* order)
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
            SequentialPolys{}, cfg.ntt_type, cfg.n_power, cfg.reduction_poly,
            batch_size, cfg.stream);
    }

    template <typename T>
    __host__ void GPU_NTT_Modulus_Ordered_Inplace(
        T* device_inout, T* root_of_unity_table, Modulus<T>* modulus,
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
    GPU_NTT_Poly_Ordered(T* device_in, T* device_out, T* root_of_unity_table,
                         Modulus<T>* modulus, ntt_rns_configuration<T> cfg,
                         int batch_size, int mod_count, int* order)
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
                       OrderedPolys{order}, cfg.ntt_type, cfg.n_power,
                       cfg.reduction_poly, batch_size, cfg.stream);
    }

    template <typename T>
    __host__ void GPU_NTT_Poly_Ordered_Inplace(
        T* device_inout, T* root_of_unity_table, Modulus<T>* modulus,
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
        Data32 mod_inverse;
        cudaStream_t stream;
    };

    template <> struct ntt_configuration<Data64>
    {
        int n_power;
        type ntt_type;
        NTTLayout ntt_layout;
        ReductionPolynomial reduction_poly;
        Data64 mod_inverse;
        cudaStream_t stream;
    };

    template <> struct ntt_rns_configuration<Data32>
    {
        int n_power;
        type ntt_type;
        NTTLayout ntt_layout;
        ReductionPolynomial reduction_poly;
        Data32* mod_inverse;
        cudaStream_t stream;
    };

    template <> struct ntt_rns_configuration<Data64>
    {
        int n_power;
        type ntt_type;
        NTTLayout ntt_layout;
        ReductionPolynomial reduction_poly;
        Data64* mod_inverse;
        cudaStream_t stream;
    };

    // Explicit instantiations. The kernels are declared in the public
    // header, so every supported kernel and host function is
    // instantiated here. Signed input (NTT) and output (INTT) are
    // supported by the single modulus and RNS variants only.

#define GPUNTT_INSTANTIATE_ALL_TYPES(T)                                        \
    template __global__ void ForwardCoreTranspose<T>(                          \
        T*, UnsignedOf<T>*, const UnsignedOf<T>* __restrict__,                 \
        const Modulus<UnsignedOf<T>>, int, int, bool);                         \
    template __global__ void ForwardCoreTranspose<T>(                          \
        T*, UnsignedOf<T>*, const UnsignedOf<T>* __restrict__,                 \
        const Modulus<UnsignedOf<T>>*, int, int, bool, int);                   \
    template __global__ void InverseCoreTranspose<T>(                          \
        UnsignedOf<T>*, T*, const UnsignedOf<T>* __restrict__,                 \
        Modulus<UnsignedOf<T>>, UnsignedOf<T>, int, int, bool);                \
    template __global__ void InverseCoreTranspose<T>(                          \
        UnsignedOf<T>*, T*, const UnsignedOf<T>* __restrict__,                 \
        Modulus<UnsignedOf<T>>*, UnsignedOf<T>*, int, int, bool, int);         \
    template __host__ void GPU_NTT<T>(T*, UnsignedOf<T>*, UnsignedOf<T>*,      \
                                      Modulus<UnsignedOf<T>>,                  \
                                      ntt_configuration<UnsignedOf<T>>, int);  \
    template __host__ void GPU_INTT<T>(UnsignedOf<T>*, T*, UnsignedOf<T>*,     \
                                       Modulus<UnsignedOf<T>>,                 \
                                       ntt_configuration<UnsignedOf<T>>, int); \
    template __host__ void GPU_NTT<T>(                                         \
        T*, UnsignedOf<T>*, UnsignedOf<T>*, Modulus<UnsignedOf<T>>*,           \
        ntt_rns_configuration<UnsignedOf<T>>, int, int);                       \
    template __host__ void GPU_INTT<T>(                                        \
        UnsignedOf<T>*, T*, UnsignedOf<T>*, Modulus<UnsignedOf<T>>*,           \
        ntt_rns_configuration<UnsignedOf<T>>, int, int);

#define GPUNTT_INSTANTIATE_UNSIGNED_TYPES(T)                                   \
    template __host__ void GPU_NTT_Inplace<T>(T*, T*, Modulus<T>,              \
                                              ntt_configuration<T>, int);      \
    template __host__ void GPU_INTT_Inplace<T>(T*, T*, Modulus<T>,             \
                                               ntt_configuration<T>, int);     \
    template __host__ void GPU_NTT_Inplace<T>(                                 \
        T*, T*, Modulus<T>*, ntt_rns_configuration<T>, int, int);              \
    template __host__ void GPU_INTT_Inplace<T>(                                \
        T*, T*, Modulus<T>*, ntt_rns_configuration<T>, int, int);              \
    template __host__ void GPU_NTT_Modulus_Ordered<T>(                         \
        T*, T*, T*, Modulus<T>*, ntt_rns_configuration<T>, int, int, int*);    \
    template __host__ void GPU_NTT_Modulus_Ordered_Inplace<T>(                 \
        T*, T*, Modulus<T>*, ntt_rns_configuration<T>, int, int, int*);        \
    template __host__ void GPU_NTT_Poly_Ordered<T>(                            \
        T*, T*, T*, Modulus<T>*, ntt_rns_configuration<T>, int, int, int*);    \
    template __host__ void GPU_NTT_Poly_Ordered_Inplace<T>(                    \
        T*, T*, Modulus<T>*, ntt_rns_configuration<T>, int, int, int*);

    GPUNTT_INSTANTIATE_ALL_TYPES(Data32)
    GPUNTT_INSTANTIATE_ALL_TYPES(Data64)
    GPUNTT_INSTANTIATE_ALL_TYPES(Data32s)
    GPUNTT_INSTANTIATE_ALL_TYPES(Data64s)
    GPUNTT_INSTANTIATE_UNSIGNED_TYPES(Data32)
    GPUNTT_INSTANTIATE_UNSIGNED_TYPES(Data64)

#undef GPUNTT_INSTANTIATE_ALL_TYPES
#undef GPUNTT_INSTANTIATE_UNSIGNED_TYPES

} // namespace gpuntt
