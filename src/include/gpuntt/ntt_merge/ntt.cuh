// Copyright 2022 Ulvetanna Inc.
// Copyright 2023-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan
// Paper: https://eprint.iacr.org/2023/1410

#ifndef NTT_CORE_H
#define NTT_CORE_H

#include "cuda_runtime.h"
#include "gpuntt/common/nttparameters.cuh"
#include <functional>
#include <unordered_map>

// --------------------- //
// Authors: Alisah Ozcan
// --------------------- //

namespace gpuntt
{
    typedef std::uint32_t location_t;
    /*
    #if MAX_LOG2_RINGSIZE <= 32
    typedef std::uint32_t location_t;
    #else
    typedef std::uint64_t location_t;
    #endif
    */

    template <typename T> struct ntt_configuration
    {
        int n_power;
        type ntt_type;
        NTTLayout ntt_layout;
        ReductionPolynomial reduction_poly;
        Ninverse<T> mod_inverse;
        cudaStream_t stream;
    };

    template <typename T> struct ntt_rns_configuration
    {
        int n_power;
        type ntt_type;
        NTTLayout ntt_layout;
        ReductionPolynomial reduction_poly;
        Ninverse<T>* mod_inverse;
        cudaStream_t stream;
    };

    template <typename T>
    __device__ __forceinline__ void
    CooleyTukeyUnit(T& U, T& V, const Root<T>& root, const Modulus<T>& modulus)
    {
        T u_ = U;
        T v_ = OPERATOR_GPU<T>::mult(V, root, modulus);

        U = OPERATOR_GPU<T>::add(u_, v_, modulus);
        V = OPERATOR_GPU<T>::sub(u_, v_, modulus);
    }

    template <typename T>
    __device__ __forceinline__ void
    GentlemanSandeUnit(T& U, T& V, const Root<T>& root,
                       const Modulus<T>& modulus)
    {
        T u_ = U;
        T v_ = V;

        U = OPERATOR_GPU<T>::add(u_, v_, modulus);

        v_ = OPERATOR_GPU<T>::sub(u_, v_, modulus);
        V = OPERATOR_GPU<T>::mult(v_, root, modulus);
    }

    // It provides multiple NTT operation with using single prime.
    template <typename T>
    __global__ void ForwardCoreLowRing(
        T* polynomial_in, typename std::make_unsigned<T>::type* polynomial_out,
        const Root<typename std::make_unsigned<
            T>::type>* __restrict__ root_of_unity_table,
        Modulus<typename std::make_unsigned<T>::type> modulus, int shared_index,
        int N_power, bool reduction_poly_check, int total_batch);

    // It provides multiple NTT operation with using single prime.
    template <typename T>
    __global__ void
    ForwardCoreLowRing(T* polynomial_in,
                       typename std::make_unsigned<T>::type* polynomial_out,
                       const Root<typename std::make_unsigned<
                           T>::type>* __restrict__ root_of_unity_table,
                       Modulus<typename std::make_unsigned<T>::type>* modulus,
                       int shared_index, int N_power, bool reduction_poly_check,
                       int total_batch, int mod_count);

    // It provides multiple NTT operation with using single prime.
    template <typename T>
    __global__ void
    ForwardCore(T* polynomial_in,
                typename std::make_unsigned<T>::type* polynomial_out,
                const Root<typename std::make_unsigned<
                    T>::type>* __restrict__ root_of_unity_table,
                Modulus<typename std::make_unsigned<T>::type> modulus,
                int shared_index, int logm, int outer_iteration_count,
                int N_power, bool not_last_kernel, bool reduction_poly_check);

    // It provides multiple NTT operation with using multiple prime for
    // RNS(Residue Number System).
    template <typename T>
    __global__ void ForwardCore(
        T* polynomial_in, typename std::make_unsigned<T>::type* polynomial_out,
        const Root<typename std::make_unsigned<
            T>::type>* __restrict__ root_of_unity_table,
        Modulus<typename std::make_unsigned<T>::type>* modulus,
        int shared_index, int logm, int outer_iteration_count, int N_power,
        bool not_last_kernel, bool reduction_poly_check, int mod_count);

    // It provides multiple NTT operation with using single prime.
    template <typename T>
    __global__ void
    ForwardCore_(T* polynomial_in,
                 typename std::make_unsigned<T>::type* polynomial_out,
                 const Root<typename std::make_unsigned<
                     T>::type>* __restrict__ root_of_unity_table,
                 Modulus<typename std::make_unsigned<T>::type> modulus,
                 int shared_index, int logm, int outer_iteration_count,
                 int N_power, bool not_last_kernel, bool reduction_poly_check);

    // It provides multiple NTT operation with using multiple prime for
    // RNS(Residue Number System).
    template <typename T>
    __global__ void ForwardCore_(
        T* polynomial_in, typename std::make_unsigned<T>::type* polynomial_out,
        const Root<typename std::make_unsigned<
            T>::type>* __restrict__ root_of_unity_table,
        Modulus<typename std::make_unsigned<T>::type>* modulus,
        int shared_index, int logm, int outer_iteration_count, int N_power,
        bool not_last_kernel, bool reduction_poly_check, int mod_count);

    // It provides multiple NTT operation with using single prime.
    template <typename T>
    __global__ void InverseCoreLowRing(
        typename std::make_unsigned<T>::type* polynomial_in, T* polynomial_out,
        const Root<typename std::make_unsigned<
            T>::type>* __restrict__ inverse_root_of_unity_table,
        Modulus<typename std::make_unsigned<T>::type> modulus, int shared_index,
        int N_power, Ninverse<typename std::make_unsigned<T>::type> n_inverse,
        bool reduction_poly_check, int total_batch);

    // It provides multiple NTT operation with using single prime.
    template <typename T>
    __global__ void InverseCoreLowRing(
        typename std::make_unsigned<T>::type* polynomial_in, T* polynomial_out,
        const Root<typename std::make_unsigned<
            T>::type>* __restrict__ inverse_root_of_unity_table,
        Modulus<typename std::make_unsigned<T>::type>* modulus,
        int shared_index, int N_power,
        Ninverse<typename std::make_unsigned<T>::type>* n_inverse,
        bool reduction_poly_check, int total_batch, int mod_count);

    // It provides multiple NTT operation with using single prime.
    template <typename T>
    __global__ void InverseCore(
        typename std::make_unsigned<T>::type* polynomial_in, T* polynomial_out,
        const Root<typename std::make_unsigned<
            T>::type>* __restrict__ inverse_root_of_unity_table,
        Modulus<typename std::make_unsigned<T>::type> modulus, int shared_index,
        int logm, int k, int outer_iteration_count, int N_power,
        Ninverse<typename std::make_unsigned<T>::type> n_inverse,
        bool last_kernel, bool reduction_poly_check);

    // It provides multiple NTT operation with using multiple prime for
    // RNS(Residue Number System).
    template <typename T>
    __global__ void InverseCore(
        typename std::make_unsigned<T>::type* polynomial_in, T* polynomial_out,
        const Root<typename std::make_unsigned<
            T>::type>* __restrict__ inverse_root_of_unity_table,
        Modulus<typename std::make_unsigned<T>::type>* modulus,
        int shared_index, int logm, int k, int outer_iteration_count,
        int N_power, Ninverse<typename std::make_unsigned<T>::type>* n_inverse,
        bool last_kernel, bool reduction_poly_check, int mod_count);

    // It provides multiple NTT operation with using single prime.
    template <typename T>
    __global__ void InverseCore_(
        typename std::make_unsigned<T>::type* polynomial_in, T* polynomial_out,
        const Root<typename std::make_unsigned<
            T>::type>* __restrict__ inverse_root_of_unity_table,
        Modulus<typename std::make_unsigned<T>::type> modulus, int shared_index,
        int logm, int k, int outer_iteration_count, int N_power,
        Ninverse<typename std::make_unsigned<T>::type> n_inverse,
        bool last_kernel, bool reduction_poly_check);

    // It provides multiple NTT operation with using multiple prime for
    // RNS(Residue Number System).
    template <typename T>
    __global__ void InverseCore_(
        typename std::make_unsigned<T>::type* polynomial_in, T* polynomial_out,
        const Root<typename std::make_unsigned<
            T>::type>* __restrict__ inverse_root_of_unity_table,
        Modulus<typename std::make_unsigned<T>::type>* modulus,
        int shared_index, int logm, int k, int outer_iteration_count,
        int N_power, Ninverse<typename std::make_unsigned<T>::type>* n_inverse,
        bool last_kernel, bool reduction_poly_check, int mod_count);

    template <typename T>
    __global__ void ForwardCoreTranspose(
        T* polynomial_in, typename std::make_unsigned<T>::type* polynomial_out,
        const Root<typename std::make_unsigned<
            T>::type>* __restrict__ root_of_unity_table,
        const Modulus<typename std::make_unsigned<T>::type> modulus,
        int log_row, int log_column, bool reduction_poly_check);

    template <typename T>
    __global__ void ForwardCoreTranspose(
        T* polynomial_in, typename std::make_unsigned<T>::type* polynomial_out,
        const Root<typename std::make_unsigned<
            T>::type>* __restrict__ root_of_unity_table,
        const Modulus<typename std::make_unsigned<T>::type>* modulus,
        int log_row, int log_column, bool reduction_poly_check, int mod_count);

    template <typename T>
    __global__ void InverseCoreTranspose(
        typename std::make_unsigned<T>::type* polynomial_in, T* polynomial_out,
        const Root<typename std::make_unsigned<
            T>::type>* __restrict__ inverse_root_of_unity_table,
        Modulus<typename std::make_unsigned<T>::type> modulus,
        Ninverse<typename std::make_unsigned<T>::type> n_inverse, int log_row,
        int log_column, bool reduction_poly_check);

    template <typename T>
    __global__ void InverseCoreTranspose(
        typename std::make_unsigned<T>::type* polynomial_in, T* polynomial_out,
        const Root<typename std::make_unsigned<
            T>::type>* __restrict__ inverse_root_of_unity_table,
        Modulus<typename std::make_unsigned<T>::type>* modulus,
        Ninverse<typename std::make_unsigned<T>::type>* n_inverse, int log_row,
        int log_column, bool reduction_poly_check, int mod_count);

    /**
     * @brief Performs Number Theoretic Transform (NTT) over batched polynomials
     * on GPU.
     *
     * This function interprets the input data as a 2D matrix of size
     * (batch_size × poly_size), where each row represents a single polynomial
     * of degree (poly_size - 1). It applies the NTT independently on each row —
     * i.e., a *row-wise NTT*.
     *
     * Optionally, a different kernel can be used to perform **column-wise NTT**
     * on the same memory layout without requiring transposition.
     *
     * # Data Layout in Memory
     * Let the input pointer represent `batch_size` polynomials, each of size
     * `poly_size`. The layout in memory is assumed to be *row-major*, like
     * this:
     *
     * ┌──────────────────────────── poly_size ───────────────────────────┐
     * │   a₀₀    a₀₁   ...    a₀ₙ₋₁  ← Polynomial 0 (Row 0) (mod q)      │
     * │   a₁₀    a₁₁   ...    a₁ₙ₋₁  ← Polynomial 1 (Row 1) (mod q)      │
     * │   a₂₀    a₂₁   ...    a₂ₙ₋₁  ← Polynomial 2 (Row 2) (mod q)      │
     * │    ⋮     ⋮     ⋱       ⋮                                      │
     * │   aₘ₋₁₀  aₘ₋₁₁ ...  aₘ₋₁ₙ₋₁  ← Polynomial (m-1) (Row m-1) (mod q)│
     * └──────────────────────────────────────────────────────────────────┘
     *   ↑
     *   └─ Each row contains one polynomial with `poly_size` coefficients.
     *      These are transformed *independently* in a row-wise manner.
     *
     * # Row-wise NTT (Standard Mode)
     * Applies NTT to each polynomial (i.e., each row), as if calling:
     *
     *
     * for (int i = 0; i < batch_size; ++i)
     *     NTT(input[i * poly_size : (i + 1) * poly_size]);
     *
     *
     * # Column-wise NTT (Transpose-free Mode)
     * An alternative mode where the same memory is interpreted **column-wise**:
     *
     * ┌────────────────── Columns (same index across batches)──────────────┐
     * │   a₀₀    a₁₀  ...    aₘ₋₁₀  ← Coefficient 0 of all polys(mod q)    │
     * │   a₀₁    a₁₁  ...    aₘ₋₁₁  ← Coefficient 1 of all polys(mod q)    │
     * │   a₀₂    a₁₂  ...    aₘ₋₁₂  ← Coefficient 2 of all polys(mod q)    │
     * │    ⋮      ⋮     ⋱    ⋮                                          │
     * │   a₀ₙ₋₁  a₁ₙ₋₁ ...  aₘ₋₁ₙ₋₁ ← Coefficient (n-1) of all polys(mod q)│
     * └────────────────────────────────────────────────────────────────────┘
     *   ↑
     *   └─ Each column is a sequence: same-index coefficients from all
     * polynomials. NTT is applied to each column independently.
     *
     * No transposition is performed — both modes read from the same memory
     * layout but interpret it differently depending on the NTT strategy.
     */
    template <typename T>
    __host__ void
    GPU_NTT(T* device_in, typename std::make_unsigned<T>::type* device_out,
            Root<typename std::make_unsigned<T>::type>* root_of_unity_table,
            Modulus<typename std::make_unsigned<T>::type> modulus,
            ntt_configuration<typename std::make_unsigned<T>::type> cfg,
            int batch_size);

    template <typename T>
    __host__ void
    GPU_INTT(typename std::make_unsigned<T>::type* device_in, T* device_out,
             Root<typename std::make_unsigned<T>::type>* root_of_unity_table,
             Modulus<typename std::make_unsigned<T>::type> modulus,
             ntt_configuration<typename std::make_unsigned<T>::type> cfg,
             int batch_size);

    template <typename T>
    __host__ void GPU_NTT_Inplace(T* device_inout, Root<T>* root_of_unity_table,
                                  Modulus<T> modulus, ntt_configuration<T> cfg,
                                  int batch_size);

    template <typename T>
    __host__ void GPU_INTT_Inplace(T* device_inout,
                                   Root<T>* root_of_unity_table,
                                   Modulus<T> modulus, ntt_configuration<T> cfg,
                                   int batch_size);

    /**
     * @brief Performs Number Theoretic Transform (NTT) over batched polynomials
     * on GPU.
     *
     * This function interprets the input data as a 2D matrix of size
     * (batch_size × poly_size), where each row represents a single polynomial
     * of degree (poly_size - 1). It applies the NTT independently on each row —
     * i.e., a *row-wise NTT*.
     *
     * Optionally, a different kernel can be used to perform **column-wise NTT**
     * on the same memory layout without requiring transposition.
     *
     * # Data Layout in Memory
     * Let the input pointer represent `batch_size` polynomials, each of size
     * `poly_size`. The layout in memory is assumed to be *row-major*, like
     * this:
     *
     * ┌──────────────────────────── poly_size ────────────────────────────┐
     * │   a₀₀    a₀₁   ...    a₀ₙ₋₁  ← Polynomial 0 (Row 0) (mod q0)      │
     * │   a₁₀    a₁₁   ...    a₁ₙ₋₁  ← Polynomial 1 (Row 1) (mod q1)      │
     * │   a₂₀    a₂₁   ...    a₂ₙ₋₁  ← Polynomial 2 (Row 2) (mod q2)      │
     * │    ⋮     ⋮     ⋱       ⋮                                       │
     * │   aₘ₋₁₀  aₘ₋₁₁ ...  aₘ₋₁ₙ₋₁  ← Polynomial (m-1) (Row m-1) (mod qx)│
     * └───────────────────────────────────────────────────────────────────┘
     *   ↑
     *   └─ Each row contains one polynomial with `poly_size` coefficients.
     *      These are transformed *independently* in a row-wise manner.
     *
     * # Row-wise NTT (Standard Mode)
     * Applies NTT to each polynomial (i.e., each row), as if calling:
     *
     *
     * for (int i = 0; i < batch_size; ++i)
     *     NTT(input[i * poly_size : (i + 1) * poly_size]);
     *
     *
     * # Column-wise NTT (Transpose-free Mode)
     * An alternative mode where the same memory is interpreted **column-wise**:
     *
     * ┌────────────────── Columns (same index across batches)───────────────┐
     * │   a₀₀    a₁₀  ...    aₘ₋₁₀  ← Coefficient 0 of all polys(mod q0)    │
     * │   a₀₁    a₁₁  ...    aₘ₋₁₁  ← Coefficient 1 of all polys(mod q1)    │
     * │   a₀₂    a₁₂  ...    aₘ₋₁₂  ← Coefficient 2 of all polys(mod q2)    │
     * │    ⋮      ⋮     ⋱    ⋮                                           │
     * │   a₀ₙ₋₁  a₁ₙ₋₁ ...  aₘ₋₁ₙ₋₁ ← Coefficient (n-1) of all polys(mod qx)│
     * └─────────────────────────────────────────────────────────────────────┘
     *   ↑
     *   └─ Each column is a sequence: same-index coefficients from all
     * polynomials. NTT is applied to each column independently.
     *
     * No transposition is performed — both modes read from the same memory
     * layout but interpret it differently depending on the NTT strategy.
     */
    template <typename T>
    __host__ void
    GPU_NTT(T* device_in, typename std::make_unsigned<T>::type* device_out,
            Root<typename std::make_unsigned<T>::type>* root_of_unity_table,
            Modulus<typename std::make_unsigned<T>::type>* modulus,
            ntt_rns_configuration<typename std::make_unsigned<T>::type> cfg,
            int batch_size, int mod_count);

    template <typename T>
    __host__ void
    GPU_INTT(typename std::make_unsigned<T>::type* device_in, T* device_out,
             Root<typename std::make_unsigned<T>::type>* root_of_unity_table,
             Modulus<typename std::make_unsigned<T>::type>* modulus,
             ntt_rns_configuration<typename std::make_unsigned<T>::type> cfg,
             int batch_size, int mod_count);

    template <typename T>
    __host__ void GPU_NTT_Inplace(T* device_inout, Root<T>* root_of_unity_table,
                                  Modulus<T>* modulus,
                                  ntt_rns_configuration<T> cfg, int batch_size,
                                  int mod_count);

    template <typename T>
    __host__ void
    GPU_INTT_Inplace(T* device_inout, Root<T>* root_of_unity_table,
                     Modulus<T>* modulus, ntt_rns_configuration<T> cfg,
                     int batch_size, int mod_count);

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Modulus Ordered

    // It provides multiple NTT operation with using multiple prime for RNS with
    // cetain modulus order.
    template <typename T>
    __global__ void ForwardCoreModulusOrdered(
        T* polynomial_in, T* polynomial_out, Root<T>* root_of_unity_table,
        Modulus<T>* modulus, int shared_index, int logm,
        int outer_iteration_count, int N_power, bool not_last_kernel,
        bool reduction_poly_check, int mod_count, int* order);

    // It provides multiple NTT operation with using multiple prime for RNS with
    // cetain modulus order.
    template <typename T>
    __global__ void ForwardCoreModulusOrdered_(
        T* polynomial_in, T* polynomial_out, Root<T>* root_of_unity_table,
        Modulus<T>* modulus, int shared_index, int logm,
        int outer_iteration_count, int N_power, bool not_last_kernel,
        bool reduction_poly_check, int mod_count, int* order);

    // It provides multiple NTT operation with using multiple prime for RNS with
    // cetain modulus order.
    template <typename T>
    __global__ void InverseCoreModulusOrdered(
        T* polynomial_in, T* polynomial_out,
        Root<T>* inverse_root_of_unity_table, Modulus<T>* modulus,
        int shared_index, int logm, int k, int outer_iteration_count,
        int N_power, Ninverse<T>* n_inverse, bool last_kernel,
        bool reduction_poly_check, int mod_count, int* order);

    // It provides multiple NTT operation with using multiple prime for RNS with
    // cetain modulus order.
    template <typename T>
    __global__ void InverseCoreModulusOrdered_(
        T* polynomial_in, T* polynomial_out,
        Root<T>* inverse_root_of_unity_table, Modulus<T>* modulus,
        int shared_index, int logm, int k, int outer_iteration_count,
        int N_power, Ninverse<T>* n_inverse, bool last_kernel,
        bool reduction_poly_check, int mod_count, int* order);

    /*
     * | GPU_NTT_Ordered2 & GPU_NTT_Ordered_Inplace2 |
     *
     * [batch_size]: polynomial count
     * [mod_count]:  modulus count
     * [order]:      pre-computed indexs(modulus orders)
     *
     * example1: batch_size = 8, mod_count = 2
     *   - order        : [  1  ,   7  ]
     *   - modulus order: [ q1  ,   q2 ,   q3 ,   q4 ,   q5 ,   q6 ,   q7 ,   q8
     * ]
     *
     *   - poly order   : [poly1, poly2, poly3, poly4, poly5, poly6, poly7,
     * poly8]
     *   - NTT modulus  : [ q1  ,   q7 ,   q1 ,   q7 ,   q1 ,   q7 ,   q1 ,   q7
     * ]
     *
     * example2: batch_size = 8, mod_count = 4
     *   - order        : [  1  ,   3 ,   4 ,   6  ]
     *   - modulus order: [ q1  ,   q2 ,   q3 ,   q4 ,   q5 ,   q6 ,   q7 ,   q8
     * ]
     *
     *   - poly order   : [poly1, poly2, poly3, poly4, poly5, poly6, poly7,
     * poly8]
     *   - NTT modulus  : [ q1  ,   q3 ,   q4 ,   q6 ,   q1 ,   q3 ,   q4 ,   q6
     * ]
     */
    template <typename T>
    __host__ void
    GPU_NTT_Modulus_Ordered(T* device_in, T* device_out,
                            Root<T>* root_of_unity_table, Modulus<T>* modulus,
                            ntt_rns_configuration<T> cfg, int batch_size,
                            int mod_count, int* order);
    template <typename T>
    __host__ void GPU_NTT_Modulus_Ordered_Inplace(T* device_inout,
                                                  Root<T>* root_of_unity_table,
                                                  Modulus<T>* modulus,
                                                  ntt_rns_configuration<T> cfg,
                                                  int batch_size, int mod_count,
                                                  int* order);

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Polynomial Ordered

    // It provides multiple NTT operation with using multiple prime for RNS with
    // cetain modulus order.
    template <typename T>
    __global__ void ForwardCorePolyOrdered(
        T* polynomial_in, T* polynomial_out, Root<T>* root_of_unity_table,
        Modulus<T>* modulus, int shared_index, int logm,
        int outer_iteration_count, int N_power, bool not_last_kernel,
        bool reduction_poly_check, int mod_count, int* order);

    // It provides multiple NTT operation with using multiple prime for RNS with
    // cetain modulus order.
    template <typename T>
    __global__ void ForwardCorePolyOrdered_(
        T* polynomial_in, T* polynomial_out, Root<T>* root_of_unity_table,
        Modulus<T>* modulus, int shared_index, int logm,
        int outer_iteration_count, int N_power, bool not_last_kernel,
        bool reduction_poly_check, int mod_count, int* order);

    // It provides multiple NTT operation with using multiple prime for RNS with
    // cetain modulus order.
    template <typename T>
    __global__ void InverseCorePolyOrdered(
        T* polynomial_in, T* polynomial_out,
        Root<T>* inverse_root_of_unity_table, Modulus<T>* modulus,
        int shared_index, int logm, int k, int outer_iteration_count,
        int N_power, Ninverse<T>* n_inverse, bool last_kernel,
        bool reduction_poly_check, int mod_count, int* order);

    // It provides multiple NTT operation with using multiple prime for RNS with
    // cetain modulus order.
    template <typename T>
    __global__ void InverseCorePolyOrdered_(
        T* polynomial_in, T* polynomial_out,
        Root<T>* inverse_root_of_unity_table, Modulus<T>* modulus,
        int shared_index, int logm, int k, int outer_iteration_count,
        int N_power, Ninverse<T>* n_inverse, bool last_kernel,
        bool reduction_poly_check, int mod_count, int* order);

    /*
     * | GPU_NTT_Ordered2_2 & GPU_NTT_Ordered_Inplace2_2 |
     *
     * [batch_size]: polynomial count
     * [mod_count]:  modulus count
     * [order]:      pre-computed indexs(poly orders)
     *
     * example1: batch_size = 8, mod_count = 2
     *   - order        : [  1  ,   2  ,   3  ,   4  ,   5  ,   6  ,   7  ,   8
     * ]
     *
     *   - poly order   : [poly1, poly2, poly3, poly4, poly5, poly6, poly7,
     * poly8]
     *   - NTT modulus  : [ q1  ,   q2 ,   q1 ,   q2 ,   q1 ,   q2 ,   q1 ,   q2
     * ]
     *
     * example2: batch_size = 8, mod_count = 4
     *   - order        : [  1  ,   4  ,   8  ,   3  ,   5  ,   6  ,   7  ,   2
     * ]
     *   - poly order   : [poly1, poly2, poly3, poly4, poly5, poly6, poly7,
     * poly8]
     *
     *   - NTT poly     : [poly1, poly4, poly8, poly3, poly5, poly6, poly7,
     * poly2]
     *   - NTT modulus  : [ q1  ,   q3 ,   q4 ,   q6 ,   q1 ,   q3 ,   q4 ,   q6
     * ]
     *
     * example3: batch_size = 8, mod_count = 4
     *   - order        : [  9  ,   4  ,   8  ,   3  ,   10  ,   6  ,   1  ,   2
     * ]
     *   - poly order   : [poly1, poly2, poly3, poly4, poly5, poly6, poly7,
     * poly8, poly9, poly10]
     *
     *   - NTT poly     : [poly9, poly4, poly8, poly3, poly10, poly6, poly1,
     * poly2]
     *   - NTT modulus  : [ q1  ,   q3 ,   q4 ,   q6 ,   q1 ,   q3 ,   q4 ,   q6
     * ]
     */
    template <typename T>
    __host__ void
    GPU_NTT_Poly_Ordered(T* device_in, T* device_out,
                         Root<T>* root_of_unity_table, Modulus<T>* modulus,
                         ntt_rns_configuration<T> cfg, int batch_size,
                         int mod_count, int* order);
    template <typename T>
    __host__ void
    GPU_NTT_Poly_Ordered_Inplace(T* device_inout, Root<T>* root_of_unity_table,
                                 Modulus<T>* modulus,
                                 ntt_rns_configuration<T> cfg, int batch_size,
                                 int mod_count, int* order);

} // namespace gpuntt
#endif // NTT_CORE_H
