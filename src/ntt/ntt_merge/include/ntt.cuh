// (C) Ulvetanna Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan
// Paper: https://eprint.iacr.org/2023/1410

#ifndef NTT_CORE_H
#define NTT_CORE_H

#include <curand_kernel.h>

#include "cuda_runtime.h"
#include "ntt_cpu.cuh"

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


struct ntt_configuration
{
    int n_power;
    type ntt_type;
    ReductionPolynomial reduction_poly;
    bool zero_padding;
    Ninverse mod_inverse;
    cudaStream_t stream;
};

struct ntt_rns_configuration
{
    int n_power;
    type ntt_type;
    ReductionPolynomial reduction_poly;
    bool zero_padding;
    Ninverse* mod_inverse;
    cudaStream_t stream;
};

__device__ void CooleyTukeyUnit(Data& U, Data& V, Root& root, Modulus& modulus);

__device__ void GentlemanSandeUnit(Data& U, Data& V, Root& root, Modulus& modulus);

// It provides multiple NTT operation with using single prime.
__global__ void ForwardCore(Data* polynomial_in, Data* polynomial_out, Root* root_of_unity_table,
                            Modulus modulus, int shared_index, int logm, int outer_iteration_count,
                            int N_power, bool zero_padding, bool not_last_kernel,
                            bool reduction_poly_check);

// It provides multiple NTT operation with using multiple prime for RNS(Residue Number System).
__global__ void ForwardCore(Data* polynomial_in, Data* polynomial_out, Root* root_of_unity_table,
                            Modulus* modulus, int shared_index, int logm, int outer_iteration_count,
                            int N_power, bool zero_padding, bool not_last_kernel,
                            bool reduction_poly_check, int mod_count);

// It provides multiple NTT operation with using single prime.
__global__ void ForwardCore_(Data* polynomial_in, Data* polynomial_out, Root* root_of_unity_table,
                             Modulus modulus, int shared_index, int logm, int outer_iteration_count,
                             int N_power, bool zero_padding, bool not_last_kernel,
                             bool reduction_poly_check);

// It provides multiple NTT operation with using multiple prime for RNS(Residue Number System).
__global__ void ForwardCore_(Data* polynomial_in, Data* polynomial_out, Root* root_of_unity_table,
                             Modulus* modulus, int shared_index, int logm,
                             int outer_iteration_count, int N_power, bool zero_padding,
                             bool not_last_kernel, bool reduction_poly_check, int mod_count);

// It provides multiple NTT operation with using single prime.
__global__ void InverseCore(Data* polynomial_in, Data* polynomial_out,
                            Root* inverse_root_of_unity_table, Modulus modulus, int shared_index,
                            int logm, int k, int outer_iteration_count, int N_power,
                            Ninverse n_inverse, bool last_kernel, bool reduction_poly_check);

// It provides multiple NTT operation with using multiple prime for RNS(Residue Number System).
__global__ void InverseCore(Data* polynomial_in, Data* polynomial_out,
                            Root* inverse_root_of_unity_table, Modulus* modulus, int shared_index,
                            int logm, int k, int outer_iteration_count, int N_power,
                            Ninverse* n_inverse, bool last_kernel, bool reduction_poly_check,
                            int mod_count);

// It provides multiple NTT operation with using single prime.
__global__ void InverseCore_(Data* polynomial_in, Data* polynomial_out,
                             Root* inverse_root_of_unity_table, Modulus modulus, int shared_index,
                             int logm, int k, int outer_iteration_count, int N_power,
                             Ninverse n_inverse, bool last_kernel, bool reduction_poly_check);

// It provides multiple NTT operation with using multiple prime for RNS(Residue Number System).
__global__ void InverseCore_(Data* polynomial_in, Data* polynomial_out,
                             Root* inverse_root_of_unity_table, Modulus* modulus, int shared_index,
                             int logm, int k, int outer_iteration_count, int N_power,
                             Ninverse* n_inverse, bool last_kernel, bool reduction_poly_check,
                             int mod_count);

/*
 * | GPU_NTT & GPU_NTT_Inplace |
 *
 * [batch_size]: polynomial count
 *
 * example1: batch_size = 8
 *   - poly order   : [poly1, poly2, poly3, poly4, poly5, poly6, poly7, poly8]
 *   - modulus order: [ q1  ,   q1 ,   q1 ,   q1 ,   q1 ,   q1 ,   q1 ,   q1 ]
 */
__host__ void GPU_NTT(Data* device_in, Data* device_out, Root* root_of_unity_table, Modulus modulus,
                      ntt_configuration cfg, int batch_size);

__host__ void GPU_NTT_Inplace(Data* device_inout, Root* root_of_unity_table, Modulus modulus,
                              ntt_configuration cfg, int batch_size);

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
__host__ void GPU_NTT(Data* device_in, Data* device_out, Root* root_of_unity_table,
                      Modulus* modulus, ntt_rns_configuration cfg, int batch_size, int mod_count);

__host__ void GPU_NTT_Inplace(Data* device_inout, Root* root_of_unity_table, Modulus* modulus,
                              ntt_rns_configuration cfg, int batch_size, int mod_count);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Modulus Ordered

// It provides multiple NTT operation with using multiple prime for RNS with cetain modulus order.
__global__ void ForwardCoreModulusOrdered(Data* polynomial_in, Data* polynomial_out,
                                          Root* root_of_unity_table, Modulus* modulus,
                                          int shared_index, int logm, int outer_iteration_count,
                                          int N_power, bool zero_padding, bool not_last_kernel,
                                          bool reduction_poly_check, int mod_count, int* order);

// It provides multiple NTT operation with using multiple prime for RNS with cetain modulus order.
__global__ void ForwardCoreModulusOrdered_(Data* polynomial_in, Data* polynomial_out,
                                           Root* root_of_unity_table, Modulus* modulus,
                                           int shared_index, int logm, int outer_iteration_count,
                                           int N_power, bool zero_padding, bool not_last_kernel,
                                           bool reduction_poly_check, int mod_count, int* order);

// It provides multiple NTT operation with using multiple prime for RNS with cetain modulus order.
__global__ void InverseCoreModulusOrdered(Data* polynomial_in, Data* polynomial_out,
                                          Root* inverse_root_of_unity_table, Modulus* modulus,
                                          int shared_index, int logm, int k,
                                          int outer_iteration_count, int N_power,
                                          Ninverse* n_inverse, bool last_kernel,
                                          bool reduction_poly_check, int mod_count, int* order);

// It provides multiple NTT operation with using multiple prime for RNS with cetain modulus order.
__global__ void InverseCoreModulusOrdered_(Data* polynomial_in, Data* polynomial_out,
                                           Root* inverse_root_of_unity_table, Modulus* modulus,
                                           int shared_index, int logm, int k,
                                           int outer_iteration_count, int N_power,
                                           Ninverse* n_inverse, bool last_kernel,
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

__host__ void GPU_NTT_Modulus_Ordered(Data* device_in, Data* device_out, Root* root_of_unity_table,
                                      Modulus* modulus, ntt_rns_configuration cfg, int batch_size,
                                      int mod_count, int* order);

__host__ void GPU_NTT_Modulus_Ordered_Inplace(Data* device_inout, Root* root_of_unity_table,
                                              Modulus* modulus, ntt_rns_configuration cfg,
                                              int batch_size, int mod_count, int* order);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Polynomial Ordered

// It provides multiple NTT operation with using multiple prime for RNS with cetain modulus order.
__global__ void ForwardCorePolyOrdered(Data* polynomial_in, Data* polynomial_out,
                                       Root* root_of_unity_table, Modulus* modulus,
                                       int shared_index, int logm, int outer_iteration_count,
                                       int N_power, bool zero_padding, bool not_last_kernel,
                                       bool reduction_poly_check, int mod_count, int* order);

// It provides multiple NTT operation with using multiple prime for RNS with cetain modulus order.
__global__ void ForwardCorePolyOrdered_(Data* polynomial_in, Data* polynomial_out,
                                        Root* root_of_unity_table, Modulus* modulus,
                                        int shared_index, int logm, int outer_iteration_count,
                                        int N_power, bool zero_padding, bool not_last_kernel,
                                        bool reduction_poly_check, int mod_count, int* order);

// It provides multiple NTT operation with using multiple prime for RNS with cetain modulus order.
__global__ void InverseCorePolyOrdered(Data* polynomial_in, Data* polynomial_out,
                                       Root* inverse_root_of_unity_table, Modulus* modulus,
                                       int shared_index, int logm, int k, int outer_iteration_count,
                                       int N_power, Ninverse* n_inverse, bool last_kernel,
                                       bool reduction_poly_check, int mod_count, int* order);

// It provides multiple NTT operation with using multiple prime for RNS with cetain modulus order.
__global__ void InverseCorePolyOrdered_(Data* polynomial_in, Data* polynomial_out,
                                        Root* inverse_root_of_unity_table, Modulus* modulus,
                                        int shared_index, int logm, int k,
                                        int outer_iteration_count, int N_power, Ninverse* n_inverse,
                                        bool last_kernel, bool reduction_poly_check, int mod_count,
                                        int* order);

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
 *   - poly order   : [poly1, poly2, poly3, poly4, poly5, poly6, poly7, poly8, poly9, poly10]
 *
 *   - NTT poly     : [poly9, poly4, poly8, poly3, poly10, poly6, poly1, poly2]
 *   - NTT modulus  : [ q1  ,   q3 ,   q4 ,   q6 ,   q1 ,   q3 ,   q4 ,   q6 ]
 */

__host__ void GPU_NTT_Poly_Ordered(Data* device_in, Data* device_out, Root* root_of_unity_table,
                                   Modulus* modulus, ntt_rns_configuration cfg, int batch_size,
                                   int mod_count, int* order);

__host__ void GPU_NTT_Poly_Ordered_Inplace(Data* device_inout, Root* root_of_unity_table,
                                           Modulus* modulus, ntt_rns_configuration cfg,
                                           int batch_size, int mod_count, int* order);

#endif  // NTT_CORE_H
