// (C) Ulvetanna Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan
// Paper: https://eprint.iacr.org/2023/1410

#ifndef NTT_4STEP_CORE_H
#define NTT_4STEP_CORE_H

#include <curand_kernel.h>

#include "cuda_runtime.h"
#include "ntt_4step_cpu.cuh"


// --------------------- //
// Authors: Alisah Ozcan
// --------------------- //

struct ntt4step_configuration
{
    int n_power;
    type ntt_type;
    Ninverse mod_inverse;
    cudaStream_t stream;
};

struct ntt4step_rns_configuration
{
    int n_power;
    type ntt_type;
    Ninverse* mod_inverse;
    cudaStream_t stream;
};

__device__ void CooleyTukeyUnit_(Data& U, Data& V, Root& root, Modulus& modulus);

__device__ void GentlemanSandeUnit_(Data& U, Data& V, Root& root, Modulus& modulus);

__global__ void Transpose_Batch(Data* polynomial_in, Data* polynomial_out, const int row,
                                const int col, int n_power);

__host__ void GPU_Transpose(Data* polynomial_in, Data* polynomial_out, const int row, const int col,
                            const int n_power, const int batch_size);

// Ring Size   -   Matrix Size
//   4096     ---> 32 x 128
//   8192     ---> 32 x 256
//   16384    ---> 32 x 512
//   32768    ---> 64 x 512
//   65536    ---> 128 x 512
//   131072   ---> 32 x 4096
//   262144   ---> 32 x 8192
//   524288   ---> 32 x 16384
//   1048576  ---> 32 x 32768
//   2097152  ---> 64 x 32768
//   4194304  ---> 128 x 32768
//   8388608  ---> 128 x 65536
//   16777216 ---> 256 x 65536

//////////////////////////////////////////////////////////////////////////////////////

// 4 STEP NTT:(without first and last transpose)
// [Transpose]  [-]
//    [CT]      [+]
// [Transpose]  [+]
//   [W Mult]   [+]
//    [CT]      [+]
// [Transpose]  [-]

__global__ void FourStepForwardCoreT1(Data* polynomial_in, Data* polynomial_out,
                                      Root* n1_root_of_unity_table, Modulus* modulus, int index1,
                                      int index2, int n_power, int mod_count);
__global__ void FourStepForwardCoreT1(Data* polynomial_in, Data* polynomial_out,
                                      Root* n1_root_of_unity_table, Modulus modulus, int index1,
                                      int index2, int n_power);
__global__ void FourStepForwardCoreT2(Data* polynomial_in, Data* polynomial_out,
                                      Root* n1_root_of_unity_table, Modulus* modulus, int index1,
                                      int index2, int index3, int n_power, int mod_count);
__global__ void FourStepForwardCoreT2(Data* polynomial_in, Data* polynomial_out,
                                      Root* n1_root_of_unity_table, Modulus modulus, int index1,
                                      int index2, int index3, int n_power);
__global__ void FourStepForwardCoreT3(Data* polynomial_in, Data* polynomial_out,
                                      Root* n1_root_of_unity_table, Modulus* modulus, int index1,
                                      int index2, int index3, int n_power, int mod_count);
__global__ void FourStepForwardCoreT3(Data* polynomial_in, Data* polynomial_out,
                                      Root* n1_root_of_unity_table, Modulus modulus, int index1,
                                      int index2, int index3, int n_power);
__global__ void FourStepForwardCoreT4(Data* polynomial_in, Data* polynomial_out,
                                      Root* n1_root_of_unity_table, Modulus* modulus, int index1,
                                      int index2, int index3, int n_power, int mod_count);
__global__ void FourStepForwardCoreT4(Data* polynomial_in, Data* polynomial_out,
                                      Root* n1_root_of_unity_table, Modulus modulus, int index1,
                                      int index2, int index3, int n_power);
__global__ void FourStepPartialForwardCore1(Data* polynomial_in, Root* n2_root_of_unity_table,
                                            Root* w_root_of_unity_table, Modulus* modulus,
                                            int small_npower, int loc1, int loc2, int loop,
                                            int n_power, int mod_count);
__global__ void FourStepPartialForwardCore1(Data* polynomial_in, Root* n2_root_of_unity_table,
                                            Root* w_root_of_unity_table, Modulus modulus,
                                            int small_npower, int loc1, int loc2, int loop,
                                            int n_power);
__global__ void FourStepPartialForwardCore2(Data* polynomial_in, Root* n2_root_of_unity_table,
                                            Modulus* modulus, int small_npower, int n_power,
                                            int mod_count);
__global__ void FourStepPartialForwardCore2(Data* polynomial_in, Root* n2_root_of_unity_table,
                                            Modulus modulus, int small_npower, int n_power);
__global__ void FourStepPartialForwardCore(Data* polynomial_in, Root* n2_root_of_unity_table,
                                           Root* w_root_of_unity_table, Modulus* modulus,
                                           int small_npower, int T, int LOOP, int n_power,
                                           int mod_count);
__global__ void FourStepPartialForwardCore(Data* polynomial_in, Root* n2_root_of_unity_table,
                                           Root* w_root_of_unity_table, Modulus modulus,
                                           int small_npower, int T, int LOOP, int n_power);

//////////////////////////////////////////////////////////////////////////////////////

// 4 STEP INTT:(without first and last transpose)
// [INTT Transpose]  [-]
//    [GS]           [+]
// [Transpose]       [+]
//   [W Mult]        [+]
//    [GS]           [+]
// [Transpose]       [-]

__global__ void FourStepInverseCoreT1(Data* polynomial_in, Data* polynomial_out,
                                      Root* n1_root_of_unity_table, Modulus* modulus, int index1,
                                      int index2, int n_power, int mod_count);
__global__ void FourStepInverseCoreT1(Data* polynomial_in, Data* polynomial_out,
                                      Root* n1_root_of_unity_table, Modulus modulus, int index1,
                                      int index2, int n_power);
__global__ void FourStepInverseCoreT2(Data* polynomial_in, Data* polynomial_out,
                                      Root* n1_root_of_unity_table, Modulus* modulus, int index1,
                                      int index2, int index3, int n_power, int mod_count);
__global__ void FourStepInverseCoreT2(Data* polynomial_in, Data* polynomial_out,
                                      Root* n1_root_of_unity_table, Modulus modulus, int index1,
                                      int index2, int index3, int n_power);
__global__ void FourStepInverseCoreT3(Data* polynomial_in, Data* polynomial_out,
                                      Root* n1_root_of_unity_table, Modulus* modulus, int index1,
                                      int index2, int index3, int n_power, int mod_count);
__global__ void FourStepInverseCoreT3(Data* polynomial_in, Data* polynomial_out,
                                      Root* n1_root_of_unity_table, Modulus modulus, int index1,
                                      int index2, int index3, int n_power);
__global__ void FourStepInverseCoreT4(Data* polynomial_in, Data* polynomial_out,
                                      Root* n1_root_of_unity_table, Modulus* modulus, int index1,
                                      int index2, int index3, int n_power, int mod_count);
__global__ void FourStepInverseCoreT4(Data* polynomial_in, Data* polynomial_out,
                                      Root* n1_root_of_unity_table, Modulus modulus, int index1,
                                      int index2, int index3, int n_power);
__global__ void FourStepPartialInverseCore(Data* polynomial_in, Root* n2_root_of_unity_table,
                                           Root* w_root_of_unity_table, Modulus* modulus,
                                           int small_npower, int LOOP, Ninverse* inverse,
                                           int poly_n_power, int mod_count);
__global__ void FourStepPartialInverseCore(Data* polynomial_in, Root* n2_root_of_unity_table,
                                           Root* w_root_of_unity_table, Modulus modulus,
                                           int small_npower, int LOOP, Ninverse inverse,
                                           int poly_n_power);
__global__ void FourStepPartialInverseCore1(Data* polynomial_in, Root* n2_root_of_unity_table,
                                            Root* w_root_of_unity_table, Modulus* modulus,
                                            int small_npower, int poly_n_power, int mod_count);
__global__ void FourStepPartialInverseCore1(Data* polynomial_in, Root* n2_root_of_unity_table,
                                            Root* w_root_of_unity_table, Modulus modulus,
                                            int small_npower, int poly_n_power);
__global__ void FourStepPartialInverseCore2(Data* polynomial_in, Root* n2_root_of_unity_table,
                                            Modulus* modulus, int small_npower, int T, int loc1,
                                            int loc2, int loc3, int loop, Ninverse* inverse,
                                            int poly_n_power, int mod_count);
__global__ void FourStepPartialInverseCore2(Data* polynomial_in, Root* n2_root_of_unity_table,
                                            Modulus modulus, int small_npower, int T, int loc1,
                                            int loc2, int loc3, int loop, Ninverse inverse,
                                            int poly_n_power);

// HOST

/*
 * | GPU_4STEP_NTT |
 *
 * [batch_size]: polynomial count
 *
 * example1: batch_size = 8
 *   - poly order   : [poly1, poly2, poly3, poly4, poly5, poly6, poly7, poly8]
 *   - modulus order: [ q1  ,   q1 ,   q1 ,   q1 ,   q1 ,   q1 ,   q1 ,   q1 ]
 */

__host__ void GPU_4STEP_NTT(Data* device_in, Data* device_out, Root* n1_root_of_unity_table,
                            Root* n2_root_of_unity_table, Root* W_root_of_unity_table,
                            Modulus modulus, ntt4step_configuration cfg, int batch_size);
/*
 * | GPU_4STEP_NTT |
 * [batch_size]: polynomial count
 * [mod_count]:  modulus count
 *
 * example1: batch_size = 8, mod_count = 1
 *   - poly order   : [poly1, poly2, poly3, poly4, poly5, poly6, poly7, poly8]
 *   - modulus order: [ q1  ,   q1 ,   q1 ,   q1 ,   q1 ,   q1 ,   q1 ,   q1 ]
 *
 * example1: batch_size = 8, mod_count = 4
 *   - poly order   : [poly1, poly2, poly3, poly4, poly5, poly6, poly7, poly8]
 *   - modulus order: [ q1  ,   q2 ,   q3 ,   q4 ,   q1 ,   q2 ,   q3 ,   q4 ]
 */

__host__ void GPU_4STEP_NTT(Data* device_in, Data* device_out, Root* n1_root_of_unity_table,
                            Root* n2_root_of_unity_table, Root* W_root_of_unity_table,
                            Modulus* modulus, ntt4step_rns_configuration cfg, int batch_size,
                            int mod_count);

#endif  // NTT_4STEP_CORE_H
