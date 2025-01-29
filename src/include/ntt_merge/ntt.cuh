// (C) Ulvetanna Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan
// Paper: https://eprint.iacr.org/2023/1410

#ifndef NTT_CORE_H
#define NTT_CORE_H

#define CC_89 // for RTX 4090

#include "cuda_runtime.h"
#include "ntt_cpu.cuh"
#include <functional>
#include <unordered_map>

// --------------------- //
// Authors: Alisah Ozcan
// --------------------- //
typedef unsigned location_t;
/*
#if MAX_LOG2_RINGSIZE <= 32
typedef unsigned location_t;
#else
typedef unsigned long long location_t;
#endif
*/

template <typename T> struct ntt_configuration
{
    int n_power;
    type ntt_type;
    ReductionPolynomial reduction_poly;
    bool zero_padding;
    Ninverse<T> mod_inverse;
    cudaStream_t stream;
};

template <typename T> struct ntt_rns_configuration
{
    int n_power;
    type ntt_type;
    ReductionPolynomial reduction_poly;
    bool zero_padding;
    Ninverse<T>* mod_inverse;
    cudaStream_t stream;
};

struct KernelConfig
{
    int griddim_x;
    int griddim_y;
    int blockdim_x;
    int blockdim_y;
    size_t shared_memory;

    int shared_index;
    int logm;
    int k;
    int outer_iteration_count;

    bool not_last_kernel;
};

template <typename T>
__device__ void CooleyTukeyUnit(T& U, T& V, const Root<T>& root,
                                const Modulus<T>& modulus);

template <typename T>
__device__ void GentlemanSandeUnit(T& U, T& V, const Root<T>& root,
                                   const Modulus<T>& modulus);

// It provides multiple NTT operation with using single prime.
template <typename T>
__global__ void
ForwardCore(T* polynomial_in, T* polynomial_out,
            const Root<T>* __restrict__ root_of_unity_table, Modulus<T> modulus,
            int shared_index, int logm, int outer_iteration_count, int N_power,
            bool zero_padding, bool not_last_kernel, bool reduction_poly_check);

// It provides multiple NTT operation with using multiple prime for RNS(Residue
// Number System).
template <typename T>
__global__ void ForwardCore(T* polynomial_in, T* polynomial_out,
                            const Root<T>* __restrict__ root_of_unity_table,
                            Modulus<T>* modulus, int shared_index, int logm,
                            int outer_iteration_count, int N_power,
                            bool zero_padding, bool not_last_kernel,
                            bool reduction_poly_check, int mod_count);

// It provides multiple NTT operation with using single prime.
template <typename T>
__global__ void ForwardCore_(T* polynomial_in, T* polynomial_out,
                             const Root<T>* __restrict__ root_of_unity_table,
                             Modulus<T> modulus, int shared_index, int logm,
                             int outer_iteration_count, int N_power,
                             bool zero_padding, bool not_last_kernel,
                             bool reduction_poly_check);

// It provides multiple NTT operation with using multiple prime for RNS(Residue
// Number System).
template <typename T>
__global__ void ForwardCore_(T* polynomial_in, T* polynomial_out,
                             const Root<T>* __restrict__ root_of_unity_table,
                             Modulus<T>* modulus, int shared_index, int logm,
                             int outer_iteration_count, int N_power,
                             bool zero_padding, bool not_last_kernel,
                             bool reduction_poly_check, int mod_count);

// It provides multiple NTT operation with using single prime.
template <typename T>
__global__ void
InverseCore(T* polynomial_in, T* polynomial_out,
            const Root<T>* __restrict__ inverse_root_of_unity_table,
            Modulus<T> modulus, int shared_index, int logm, int k,
            int outer_iteration_count, int N_power, Ninverse<T> n_inverse,
            bool last_kernel, bool reduction_poly_check);

// It provides multiple NTT operation with using multiple prime for RNS(Residue
// Number System).
template <typename T>
__global__ void
InverseCore(T* polynomial_in, T* polynomial_out,
            const Root<T>* __restrict__ inverse_root_of_unity_table,
            Modulus<T>* modulus, int shared_index, int logm, int k,
            int outer_iteration_count, int N_power, Ninverse<T>* n_inverse,
            bool last_kernel, bool reduction_poly_check, int mod_count);

// It provides multiple NTT operation with using single prime.
template <typename T>
__global__ void
InverseCore_(T* polynomial_in, T* polynomial_out,
             const Root<T>* __restrict__ inverse_root_of_unity_table,
             Modulus<T> modulus, int shared_index, int logm, int k,
             int outer_iteration_count, int N_power, Ninverse<T> n_inverse,
             bool last_kernel, bool reduction_poly_check);

// It provides multiple NTT operation with using multiple prime for RNS(Residue
// Number System).
template <typename T>
__global__ void
InverseCore_(T* polynomial_in, T* polynomial_out,
             const Root<T>* __restrict__ inverse_root_of_unity_table,
             Modulus<T>* modulus, int shared_index, int logm, int k,
             int outer_iteration_count, int N_power, Ninverse<T>* n_inverse,
             bool last_kernel, bool reduction_poly_check, int mod_count);

/*
 * | GPU_NTT & GPU_NTT_Inplace |
 *
 * [batch_size]: polynomial count
 *
 * example1: batch_size = 8
 *   - poly order   : [poly1, poly2, poly3, poly4, poly5, poly6, poly7, poly8]
 *   - modulus order: [ q1  ,   q1 ,   q1 ,   q1 ,   q1 ,   q1 ,   q1 ,   q1 ]
 */
template <typename T>
__host__ void GPU_NTT(T* device_in, T* device_out, Root<T>* root_of_unity_table,
                      Modulus<T> modulus, ntt_configuration<T> cfg,
                      int batch_size);

template <typename T>
__host__ void GPU_NTT_Inplace(T* device_inout, Root<T>* root_of_unity_table,
                              Modulus<T> modulus, ntt_configuration<T> cfg,
                              int batch_size);

/*
 * | GPU_NTT & GPU_NTT_Inplace |
 *
 * [batch_size]: polynomial count
 * [mod_count]:  modulus count
 *
 * example1: batch_size = 8, mod_count = 1
 *   - poly order   : [poly1, poly2, poly3, poly4, poly5, poly6, poly7, poly8]
 *   - modulus order: [ q1  ,   q1 ,   q1 ,   q1 ,   q1 ,   q1 ,   q1 ,   q1 ]
 *
 * example2: batch_size = 8, mod_count = 4
 *   - poly order   : [poly1, poly2, poly3, poly4, poly5, poly6, poly7, poly8]
 *   - modulus order: [ q1  ,   q2 ,   q3 ,   q4 ,   q1 ,   q2 ,   q3 ,   q4 ]
 */
template <typename T>
__host__ void GPU_NTT(T* device_in, T* device_out, Root<T>* root_of_unity_table,
                      Modulus<T>* modulus, ntt_rns_configuration<T> cfg,
                      int batch_size, int mod_count);

template <typename T>
__host__ void GPU_NTT_Inplace(T* device_inout, Root<T>* root_of_unity_table,
                              Modulus<T>* modulus, ntt_rns_configuration<T> cfg,
                              int batch_size, int mod_count);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Modulus Ordered

// It provides multiple NTT operation with using multiple prime for RNS with
// cetain modulus order.
template <typename T>
__global__ void
ForwardCoreModulusOrdered(T* polynomial_in, T* polynomial_out,
                          Root<T>* root_of_unity_table, Modulus<T>* modulus,
                          int shared_index, int logm, int outer_iteration_count,
                          int N_power, bool zero_padding, bool not_last_kernel,
                          bool reduction_poly_check, int mod_count, int* order);

// It provides multiple NTT operation with using multiple prime for RNS with
// cetain modulus order.
template <typename T>
__global__ void ForwardCoreModulusOrdered_(
    T* polynomial_in, T* polynomial_out, Root<T>* root_of_unity_table,
    Modulus<T>* modulus, int shared_index, int logm, int outer_iteration_count,
    int N_power, bool zero_padding, bool not_last_kernel,
    bool reduction_poly_check, int mod_count, int* order);

// It provides multiple NTT operation with using multiple prime for RNS with
// cetain modulus order.
template <typename T>
__global__ void InverseCoreModulusOrdered(
    T* polynomial_in, T* polynomial_out, Root<T>* inverse_root_of_unity_table,
    Modulus<T>* modulus, int shared_index, int logm, int k,
    int outer_iteration_count, int N_power, Ninverse<T>* n_inverse,
    bool last_kernel, bool reduction_poly_check, int mod_count, int* order);

// It provides multiple NTT operation with using multiple prime for RNS with
// cetain modulus order.
template <typename T>
__global__ void InverseCoreModulusOrdered_(
    T* polynomial_in, T* polynomial_out, Root<T>* inverse_root_of_unity_table,
    Modulus<T>* modulus, int shared_index, int logm, int k,
    int outer_iteration_count, int N_power, Ninverse<T>* n_inverse,
    bool last_kernel, bool reduction_poly_check, int mod_count, int* order);

/*
 * | GPU_NTT_Ordered2 & GPU_NTT_Ordered_Inplace2 |
 *
 * [batch_size]: polynomial count
 * [mod_count]:  modulus count
 * [order]:      pre-computed indexs(modulus orders)
 *
 * example1: batch_size = 8, mod_count = 2
 *   - order        : [  1  ,   7  ]
 *   - modulus order: [ q1  ,   q2 ,   q3 ,   q4 ,   q5 ,   q6 ,   q7 ,   q8 ]
 *
 *   - poly order   : [poly1, poly2, poly3, poly4, poly5, poly6, poly7, poly8]
 *   - NTT modulus  : [ q1  ,   q7 ,   q1 ,   q7 ,   q1 ,   q7 ,   q1 ,   q7 ]
 *
 * example2: batch_size = 8, mod_count = 4
 *   - order        : [  1  ,   3 ,   4 ,   6  ]
 *   - modulus order: [ q1  ,   q2 ,   q3 ,   q4 ,   q5 ,   q6 ,   q7 ,   q8 ]
 *
 *   - poly order   : [poly1, poly2, poly3, poly4, poly5, poly6, poly7, poly8]
 *   - NTT modulus  : [ q1  ,   q3 ,   q4 ,   q6 ,   q1 ,   q3 ,   q4 ,   q6 ]
 */
template <typename T>
__host__ void
GPU_NTT_Modulus_Ordered(T* device_in, T* device_out,
                        Root<T>* root_of_unity_table, Modulus<T>* modulus,
                        ntt_rns_configuration<T> cfg, int batch_size,
                        int mod_count, int* order);
template <typename T>
__host__ void GPU_NTT_Modulus_Ordered_Inplace(
    T* device_inout, Root<T>* root_of_unity_table, Modulus<T>* modulus,
    ntt_rns_configuration<T> cfg, int batch_size, int mod_count, int* order);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Polynomial Ordered

// It provides multiple NTT operation with using multiple prime for RNS with
// cetain modulus order.
template <typename T>
__global__ void
ForwardCorePolyOrdered(T* polynomial_in, T* polynomial_out,
                       Root<T>* root_of_unity_table, Modulus<T>* modulus,
                       int shared_index, int logm, int outer_iteration_count,
                       int N_power, bool zero_padding, bool not_last_kernel,
                       bool reduction_poly_check, int mod_count, int* order);

// It provides multiple NTT operation with using multiple prime for RNS with
// cetain modulus order.
template <typename T>
__global__ void
ForwardCorePolyOrdered_(T* polynomial_in, T* polynomial_out,
                        Root<T>* root_of_unity_table, Modulus<T>* modulus,
                        int shared_index, int logm, int outer_iteration_count,
                        int N_power, bool zero_padding, bool not_last_kernel,
                        bool reduction_poly_check, int mod_count, int* order);

// It provides multiple NTT operation with using multiple prime for RNS with
// cetain modulus order.
template <typename T>
__global__ void InverseCorePolyOrdered(
    T* polynomial_in, T* polynomial_out, Root<T>* inverse_root_of_unity_table,
    Modulus<T>* modulus, int shared_index, int logm, int k,
    int outer_iteration_count, int N_power, Ninverse<T>* n_inverse,
    bool last_kernel, bool reduction_poly_check, int mod_count, int* order);

// It provides multiple NTT operation with using multiple prime for RNS with
// cetain modulus order.
template <typename T>
__global__ void InverseCorePolyOrdered_(
    T* polynomial_in, T* polynomial_out, Root<T>* inverse_root_of_unity_table,
    Modulus<T>* modulus, int shared_index, int logm, int k,
    int outer_iteration_count, int N_power, Ninverse<T>* n_inverse,
    bool last_kernel, bool reduction_poly_check, int mod_count, int* order);

/*
 * | GPU_NTT_Ordered2_2 & GPU_NTT_Ordered_Inplace2_2 |
 *
 * [batch_size]: polynomial count
 * [mod_count]:  modulus count
 * [order]:      pre-computed indexs(poly orders)
 *
 * example1: batch_size = 8, mod_count = 2
 *   - order        : [  1  ,   2  ,   3  ,   4  ,   5  ,   6  ,   7  ,   8  ]
 *
 *   - poly order   : [poly1, poly2, poly3, poly4, poly5, poly6, poly7, poly8]
 *   - NTT modulus  : [ q1  ,   q2 ,   q1 ,   q2 ,   q1 ,   q2 ,   q1 ,   q2 ]
 *
 * example2: batch_size = 8, mod_count = 4
 *   - order        : [  1  ,   4  ,   8  ,   3  ,   5  ,   6  ,   7  ,   2  ]
 *   - poly order   : [poly1, poly2, poly3, poly4, poly5, poly6, poly7, poly8]
 *
 *   - NTT poly     : [poly1, poly4, poly8, poly3, poly5, poly6, poly7, poly2]
 *   - NTT modulus  : [ q1  ,   q3 ,   q4 ,   q6 ,   q1 ,   q3 ,   q4 ,   q6 ]
 *
 * example3: batch_size = 8, mod_count = 4
 *   - order        : [  9  ,   4  ,   8  ,   3  ,   10  ,   6  ,   1  ,   2  ]
 *   - poly order   : [poly1, poly2, poly3, poly4, poly5, poly6, poly7, poly8,
 * poly9, poly10]
 *
 *   - NTT poly     : [poly9, poly4, poly8, poly3, poly10, poly6, poly1, poly2]
 *   - NTT modulus  : [ q1  ,   q3 ,   q4 ,   q6 ,   q1 ,   q3 ,   q4 ,   q6 ]
 */
template <typename T>
__host__ void
GPU_NTT_Poly_Ordered(T* device_in, T* device_out, Root<T>* root_of_unity_table,
                     Modulus<T>* modulus, ntt_rns_configuration<T> cfg,
                     int batch_size, int mod_count, int* order);
template <typename T>
__host__ void
GPU_NTT_Poly_Ordered_Inplace(T* device_inout, Root<T>* root_of_unity_table,
                             Modulus<T>* modulus, ntt_rns_configuration<T> cfg,
                             int batch_size, int mod_count, int* order);

// Kernel Parameters
template <typename T> auto CreateForwardNTTKernel()
{
    return std::unordered_map<int, std::vector<KernelConfig>>{
        {12,
         {{8, 1, 64, 4, 512 * sizeof(T), 8, 0, 0, 3, true},
          {1, 8, 256, 1, 512 * sizeof(T), 8, 3, 0, 9, false}}},
        {13,
         {{16, 1, 32, 8, 512 * sizeof(T), 8, 0, 0, 4, true},
          {1, 16, 256, 1, 512 * sizeof(T), 8, 4, 0, 9, false}}},
        {14,
         {{32, 1, 16, 16, 512 * sizeof(T), 8, 0, 0, 5, true},
          {1, 32, 256, 1, 512 * sizeof(T), 8, 5, 0, 9, false}}},
        {15,
         {{64, 1, 8, 32, 512 * sizeof(T), 8, 0, 0, 6, true},
          {1, 64, 256, 1, 512 * sizeof(T), 8, 6, 0, 9, false}}},
        {16,
         {{128, 1, 4, 64, 512 * sizeof(T), 8, 0, 0, 7, true},
          {1, 128, 256, 1, 512 * sizeof(T), 8, 7, 0, 9, false}}},
        {17,
         {{256, 1, 32, 8, 512 * sizeof(T), 8, 0, 0, 4, true},
          {16, 16, 32, 8, 512 * sizeof(T), 8, 4, 0, 4, true},
          {1, 256, 256, 1, 512 * sizeof(T), 8, 8, 0, 9, false}}},
        {18,
         {{512, 1, 32, 8, 512 * sizeof(T), 8, 0, 0, 4, true},
          {32, 16, 16, 16, 512 * sizeof(T), 8, 4, 0, 5, true},
          {1, 512, 256, 1, 512 * sizeof(T), 8, 9, 0, 9, false}}},
        {19,
         {{1024, 1, 16, 16, 512 * sizeof(T), 8, 0, 0, 5, true},
          {32, 32, 16, 16, 512 * sizeof(T), 8, 5, 0, 5, true},
          {1, 1024, 256, 1, 512 * sizeof(T), 8, 10, 0, 9, false}}},
        {20,
         {{2048, 1, 16, 16, 512 * sizeof(T), 8, 0, 0, 5, true},
          {64, 32, 8, 32, 512 * sizeof(T), 8, 5, 0, 6, true},
          {1, 2048, 256, 1, 512 * sizeof(T), 8, 11, 0, 9, false}}},
        {21,
         {{4096, 1, 8, 32, 512 * sizeof(T), 8, 0, 0, 6, true},
          {64, 64, 8, 32, 512 * sizeof(T), 8, 6, 0, 6, true},
          {1, 4096, 256, 1, 512 * sizeof(T), 8, 12, 0, 9, false}}},
        {22,
         {{8192, 1, 8, 32, 512 * sizeof(T), 8, 0, 0, 6, true},
          {128, 64, 4, 64, 512 * sizeof(T), 8, 6, 0, 7, true},
          {1, 8192, 256, 1, 512 * sizeof(T), 8, 13, 0, 9, false}}},
        {23,
         {{16384, 1, 4, 64, 512 * sizeof(T), 8, 0, 0, 7, true},
          {128, 128, 4, 64, 512 * sizeof(T), 8, 7, 0, 7, true},
          {1, 16384, 256, 1, 512 * sizeof(T), 8, 14, 0, 9, false}}},
        {24,
         {{16384, 1, 8, 64, 1024 * sizeof(T), 9, 0, 0, 7, true},
          {128, 128, 8, 64, 1024 * sizeof(T), 9, 7, 0, 7, true},
          {1, 16384, 512, 1, 1024 * sizeof(T), 9, 14, 0, 10, false}}},
        {25,
         {
             {32768, 1, 8, 64, 1024 * sizeof(T), 9, 0, 0, 7, true},
             {256, 128, 4, 128, 1024 * sizeof(T), 9, 7, 0, 8, true},
             {32768, 1, 512, 1, 1024 * sizeof(T), 9, 15, 0, 10, false}
             //{1, 32768, 512, 1, 1024 * sizeof(T), 9, 15, 0, 10, false}
         }},
        {26,
         {{65536, 1, 4, 128, 1024 * sizeof(T), 9, 0, 0, 8, true},
          {256, 256, 4, 128, 1024 * sizeof(T), 9, 8, 0, 8, true},
          {65536, 1, 512, 1, 1024 * sizeof(T), 9, 16, 0, 10, false}}},
#ifndef CC_89
        {27,
         {{262144, 1, 16, 16, 512 * sizeof(T), 8, 0, 0, 5, true},
          {8192, 32, 8, 32, 512 * sizeof(T), 8, 5, 0, 6, true},
          {128, 2048, 4, 64, 512 * sizeof(T), 8, 11, 0, 7, true},
          {262144, 1, 256, 1, 512 * sizeof(T), 8, 18, 0, 9, false}}},

        {28,
         {{524288, 1, 8, 32, 512 * sizeof(T), 8, 0, 0, 6, true},
          {8192, 64, 8, 32, 512 * sizeof(T), 8, 6, 0, 6, true},
          {128, 4096, 4, 64, 512 * sizeof(T), 8, 12, 0, 7,
           true} {524288, 1, 256, 1, 512 * sizeof(T), 8, 19, 0, 9, false}}}
#else
        {27,
         {{131072, 1, 4, 128, 1024 * sizeof(T), 9, 0, 0, 8, true},
          {512, 256, 2, 256, 1024 * sizeof(T), 9, 8, 0, 9, true},
          {131072, 1, 512, 1, 1024 * sizeof(T), 9, 17, 0, 10, false}}},
        {28,
         {{262144, 1, 2, 256, 1024 * sizeof(T), 9, 0, 0, 9, true},
          {512, 512, 2, 256, 1024 * sizeof(T), 9, 9, 0, 9, true},
          {262144, 1, 512, 1, 1024 * sizeof(T), 9, 18, 0, 10, false}}}
#endif
    };
}

template <typename T> auto CreateInverseNTTKernel()
{
    return std::unordered_map<int, std::vector<KernelConfig>>{
        {12,
         {{1, 8, 256, 1, 512 * sizeof(T), 8, 11, 3, 9, false},
          {8, 1, 64, 4, 512 * sizeof(T), 8, 2, 0, 3, true}}},
        {13,
         {{1, 16, 256, 1, 512 * sizeof(T), 8, 12, 4, 9, false},
          {16, 1, 32, 8, 512 * sizeof(T), 8, 3, 0, 4, true}}},
        {14,
         {{1, 32, 256, 1, 512 * sizeof(T), 8, 13, 5, 9, false},
          {32, 1, 16, 16, 512 * sizeof(T), 8, 4, 0, 5, true}}},
        {15,
         {{1, 64, 256, 1, 512 * sizeof(T), 8, 14, 6, 9, false},
          {64, 1, 8, 32, 512 * sizeof(T), 8, 5, 0, 6, true}}},
        {16,
         {{1, 128, 256, 1, 512 * sizeof(T), 8, 15, 7, 9, false},
          {128, 1, 4, 64, 512 * sizeof(T), 8, 6, 0, 7, true}}},
        {17,
         {{1, 256, 256, 1, 512 * sizeof(T), 8, 16, 8, 9, false},
          {16, 16, 32, 8, 512 * sizeof(T), 8, 7, 4, 4, false},
          {256, 1, 32, 8, 512 * sizeof(T), 8, 3, 0, 4, true}}},
        {18,
         {{1, 512, 256, 1, 512 * sizeof(T), 8, 17, 9, 9, false},
          {32, 16, 16, 16, 512 * sizeof(T), 8, 8, 4, 5, false},
          {512, 1, 32, 8, 512 * sizeof(T), 8, 3, 0, 4, true}}},
        {19,
         {{1, 1024, 256, 1, 512 * sizeof(T), 8, 18, 10, 9, false},
          {32, 32, 16, 16, 512 * sizeof(T), 8, 9, 5, 5, false},
          {1024, 1, 16, 16, 512 * sizeof(T), 8, 4, 0, 5, true}}},
        {20,
         {{1, 2048, 256, 1, 512 * sizeof(T), 8, 19, 11, 9, false},
          {64, 32, 8, 32, 512 * sizeof(T), 8, 10, 5, 6, false},
          {2048, 1, 16, 16, 512 * sizeof(T), 8, 4, 0, 5, true}}},
        {21,
         {{1, 4096, 256, 1, 512 * sizeof(T), 8, 20, 12, 9, false},
          {64, 64, 8, 32, 512 * sizeof(T), 8, 11, 6, 6, false},
          {4096, 1, 8, 32, 512 * sizeof(T), 8, 5, 0, 6, true}}},
        {22,
         {{1, 8192, 256, 1, 512 * sizeof(T), 8, 21, 13, 9, false},
          {128, 64, 4, 64, 512 * sizeof(T), 8, 12, 6, 7, false},
          {8192, 1, 8, 32, 512 * sizeof(T), 8, 5, 0, 6, true}}},
        {23,
         {{1, 16384, 256, 1, 512 * sizeof(T), 8, 22, 14, 9, false},
          {128, 128, 4, 64, 512 * sizeof(T), 8, 13, 7, 7, false},
          {16384, 1, 4, 64, 512 * sizeof(T), 8, 6, 0, 7, true}}},
        {24,
         {{1, 16384, 512, 1, 1024 * sizeof(T), 9, 23, 14, 10, false},
          {128, 128, 8, 64, 1024 * sizeof(T), 9, 13, 7, 7, false},
          {16384, 1, 8, 64, 1024 * sizeof(T), 9, 6, 0, 7, true}}},
        {25,
         {{32768, 1, 512, 1, 1024 * sizeof(T), 9, 24, 15, 10, false},
          {256, 128, 4, 128, 1024 * sizeof(T), 9, 14, 7, 8, false},
          {32768, 1, 8, 64, 1024 * sizeof(T), 9, 6, 0, 7, true}}},
        {26,
         {{65536, 1, 512, 1, 1024 * sizeof(T), 9, 25, 16, 10, false},
          {256, 256, 4, 128, 1024 * sizeof(T), 9, 15, 8, 8, false},
          {65536, 1, 4, 128, 1024 * sizeof(T), 9, 7, 0, 8, true}}},
#ifndef CC_89
        {27,
         {{262144, 1, 256, 1, 512 * sizeof(T), 8, 26, 18, 9, false},
          {128, 2048, 4, 64, 512 * sizeof(T), 8, 17, 11, 7, false},
          {8192, 32, 8, 32, 512 * sizeof(T), 8, 10, 5, 6, false},
          {262144, 1, 16, 16, 512 * sizeof(T), 8, 4, 0, 5, true}}},
        {28,
         {
             {524288, 1, 256, 1, 512 * sizeof(T), 8, 27, 19, 9, false},
             {128, 4096, 4, 64, 512 * sizeof(T), 8, 18, 12, 7, false},
             {8192, 64, 8, 32, 512 * sizeof(T), 8, 11, 6, 6, false},
             {524288, 1, 8, 32, 512 * sizeof(T), 8, 5, 0, 6, true},
         }}
#else
        {27,
         {{131072, 1, 512, 1, 1024 * sizeof(T), 9, 26, 17, 10, false},
          {512, 256, 2, 256, 1024 * sizeof(T), 9, 16, 8, 9, false},
          {131072, 1, 4, 128, 1024 * sizeof(T), 9, 7, 0, 8, true}}},
        {28,
         {{262144, 1, 512, 1, 1024 * sizeof(T), 9, 27, 18, 10, false},
          {512, 512, 2, 256, 1024 * sizeof(T), 9, 17, 9, 9, false},
          {262144, 1, 2, 256, 1024 * sizeof(T), 9, 8, 0, 9, true}}}
#endif
    };
}

#endif // NTT_CORE_H
