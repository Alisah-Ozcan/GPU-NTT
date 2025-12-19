// (C) Ulvetanna Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan
// Paper: https://eprint.iacr.org/2023/1410

#ifndef NTT_4STEP_CORE_H
#define NTT_4STEP_CORE_H

#include "cuda_runtime.h"
#include "gpuntt/ntt_4step/ntt_4step_cpu.cuh"

// --------------------- //
// Authors: Alisah Ozcan
// --------------------- //

namespace gpuntt
{
    template <typename T> struct ntt4step_configuration
    {
        int n_power;
        type ntt_type;
        Ninverse<T> mod_inverse;
        cudaStream_t stream;
    };

    template <typename T> struct ntt4step_rns_configuration
    {
        int n_power;
        type ntt_type;
        Ninverse<T>* mod_inverse;
        cudaStream_t stream;
    };

    template <typename T>
    __device__ void CooleyTukeyUnit_(T& U, T& V, Root<T>& root,
                                     Modulus<T>& modulus);

    template <typename T>
    __device__ void GentlemanSandeUnit_(T& U, T& V, Root<T>& root,
                                        Modulus<T>& modulus);
    template <typename T>
    __global__ void Transpose_Batch(T* polynomial_in, T* polynomial_out,
                                    const int row, const int col, int n_power);

    template <typename T>
    __host__ void GPU_Transpose(T* polynomial_in, T* polynomial_out,
                                const int row, const int col, const int n_power,
                                const int batch_size);

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

    template <typename T>
    __global__ void
    FourStepForwardCoreT1(T* polynomial_in, T* polynomial_out,
                          Root<T>* n1_root_of_unity_table, Modulus<T>* modulus,
                          int index1, int index2, int n_power, int mod_count);

    template <typename T>
    __global__ void FourStepForwardCoreT1(T* polynomial_in, T* polynomial_out,
                                          Root<T>* n1_root_of_unity_table,
                                          Modulus<T> modulus, int index1,
                                          int index2, int n_power);
    template <typename T>
    __global__ void FourStepForwardCoreT2(T* polynomial_in, T* polynomial_out,
                                          Root<T>* n1_root_of_unity_table,
                                          Modulus<T>* modulus, int index1,
                                          int index2, int index3, int n_power,
                                          int mod_count);

    template <typename T>
    __global__ void FourStepForwardCoreT2(T* polynomial_in, T* polynomial_out,
                                          Root<T>* n1_root_of_unity_table,
                                          Modulus<T> modulus, int index1,
                                          int index2, int index3, int n_power);

    template <typename T>
    __global__ void FourStepForwardCoreT3(T* polynomial_in, T* polynomial_out,
                                          Root<T>* n1_root_of_unity_table,
                                          Modulus<T>* modulus, int index1,
                                          int index2, int index3, int n_power,
                                          int mod_count);

    template <typename T>
    __global__ void FourStepForwardCoreT3(T* polynomial_in, T* polynomial_out,
                                          Root<T>* n1_root_of_unity_table,
                                          Modulus<T> modulus, int index1,
                                          int index2, int index3, int n_power);

    template <typename T>
    __global__ void FourStepForwardCoreT4(T* polynomial_in, T* polynomial_out,
                                          Root<T>* n1_root_of_unity_table,
                                          Modulus<T>* modulus, int index1,
                                          int index2, int index3, int n_power,
                                          int mod_count);

    template <typename T>
    __global__ void FourStepForwardCoreT4(T* polynomial_in, T* polynomial_out,
                                          Root<T>* n1_root_of_unity_table,
                                          Modulus<T> modulus, int index1,
                                          int index2, int index3, int n_power);

    template <typename T>
    __global__ void FourStepPartialForwardCore1(
        T* polynomial_in, Root<T>* n2_root_of_unity_table,
        Root<T>* w_root_of_unity_table, Modulus<T>* modulus, int small_npower,
        int loc1, int loc2, int loop, int n_power, int mod_count);

    template <typename T>
    __global__ void FourStepPartialForwardCore1(
        T* polynomial_in, Root<T>* n2_root_of_unity_table,
        Root<T>* w_root_of_unity_table, Modulus<T> modulus, int small_npower,
        int loc1, int loc2, int loop, int n_power);

    template <typename T>
    __global__ void FourStepPartialForwardCore2(T* polynomial_in,
                                                Root<T>* n2_root_of_unity_table,
                                                Modulus<T>* modulus,
                                                int small_npower, int n_power,
                                                int mod_count);

    template <typename T>
    __global__ void FourStepPartialForwardCore2(T* polynomial_in,
                                                Root<T>* n2_root_of_unity_table,
                                                Modulus<T> modulus,
                                                int small_npower, int n_power);

    template <typename T>
    __global__ void FourStepPartialForwardCore(
        T* polynomial_in, Root<T>* n2_root_of_unity_table,
        Root<T>* w_root_of_unity_table, Modulus<T>* modulus, int small_npower,
        int t1, int LOOP, int n_power, int mod_count);

    template <typename T>
    __global__ void FourStepPartialForwardCore(T* polynomial_in,
                                               Root<T>* n2_root_of_unity_table,
                                               Root<T>* w_root_of_unity_table,
                                               Modulus<T> modulus,
                                               int small_npower, int t1,
                                               int LOOP, int n_power);

    //////////////////////////////////////////////////////////////////////////////////////

    // 4 STEP INTT:(without first and last transpose)
    // [INTT Transpose]  [-]
    //    [GS]           [+]
    // [Transpose]       [+]
    //   [W Mult]        [+]
    //    [GS]           [+]
    // [Transpose]       [-]

    template <typename T>
    __global__ void
    FourStepInverseCoreT1(T* polynomial_in, T* polynomial_out,
                          Root<T>* n1_root_of_unity_table, Modulus<T>* modulus,
                          int index1, int index2, int n_power, int mod_count);

    template <typename T>
    __global__ void FourStepInverseCoreT1(T* polynomial_in, T* polynomial_out,
                                          Root<T>* n1_root_of_unity_table,
                                          Modulus<T> modulus, int index1,
                                          int index2, int n_power);

    template <typename T>
    __global__ void FourStepInverseCoreT2(T* polynomial_in, T* polynomial_out,
                                          Root<T>* n1_root_of_unity_table,
                                          Modulus<T>* modulus, int index1,
                                          int index2, int index3, int n_power,
                                          int mod_count);

    template <typename T>
    __global__ void FourStepInverseCoreT2(T* polynomial_in, T* polynomial_out,
                                          Root<T>* n1_root_of_unity_table,
                                          Modulus<T> modulus, int index1,
                                          int index2, int index3, int n_power);

    template <typename T>
    __global__ void FourStepInverseCoreT3(T* polynomial_in, T* polynomial_out,
                                          Root<T>* n1_root_of_unity_table,
                                          Modulus<T>* modulus, int index1,
                                          int index2, int index3, int n_power,
                                          int mod_count);

    template <typename T>
    __global__ void FourStepInverseCoreT3(T* polynomial_in, T* polynomial_out,
                                          Root<T>* n1_root_of_unity_table,
                                          Modulus<T> modulus, int index1,
                                          int index2, int index3, int n_power);

    template <typename T>
    __global__ void FourStepInverseCoreT4(T* polynomial_in, T* polynomial_out,
                                          Root<T>* n1_root_of_unity_table,
                                          Modulus<T>* modulus, int index1,
                                          int index2, int index3, int n_power,
                                          int mod_count);

    template <typename T>
    __global__ void FourStepInverseCoreT4(T* polynomial_in, T* polynomial_out,
                                          Root<T>* n1_root_of_unity_table,
                                          Modulus<T> modulus, int index1,
                                          int index2, int index3, int n_power);

    template <typename T>
    __global__ void FourStepPartialInverseCore(
        T* polynomial_in, Root<T>* n2_root_of_unity_table,
        Root<T>* w_root_of_unity_table, Modulus<T>* modulus, int small_npower,
        int LOOP, Ninverse<T>* inverse, int poly_n_power, int mod_count);

    template <typename T>
    __global__ void FourStepPartialInverseCore(
        T* polynomial_in, Root<T>* n2_root_of_unity_table,
        Root<T>* w_root_of_unity_table, Modulus<T> modulus, int small_npower,
        int LOOP, Ninverse<T> inverse, int poly_n_power);

    template <typename T>
    __global__ void FourStepPartialInverseCore1(
        T* polynomial_in, Root<T>* n2_root_of_unity_table,
        Root<T>* w_root_of_unity_table, Modulus<T>* modulus, int small_npower,
        int poly_n_power, int mod_count);

    template <typename T>
    __global__ void FourStepPartialInverseCore1(T* polynomial_in,
                                                Root<T>* n2_root_of_unity_table,
                                                Root<T>* w_root_of_unity_table,
                                                Modulus<T> modulus,
                                                int small_npower,
                                                int poly_n_power);

    template <typename T>
    __global__ void FourStepPartialInverseCore2(
        T* polynomial_in, Root<T>* n2_root_of_unity_table, Modulus<T>* modulus,
        int small_npower, int t1, int loc1, int loc2, int loc3, int loop,
        Ninverse<T>* inverse, int poly_n_power, int mod_count);

    template <typename T>
    __global__ void FourStepPartialInverseCore2(
        T* polynomial_in, Root<T>* n2_root_of_unity_table, Modulus<T> modulus,
        int small_npower, int t1, int loc1, int loc2, int loc3, int loop,
        Ninverse<T> inverse, int poly_n_power);

    // HOST

    /*
     * | GPU_4STEP_NTT |
     *
     * [batch_size]: polynomial count
     *
     * example1: batch_size = 8
     *   - poly order   : [poly1, poly2, poly3, poly4, poly5, poly6, poly7,
     * poly8]
     *   - modulus order: [ q1  ,   q1 ,   q1 ,   q1 ,   q1 ,   q1 ,   q1 ,   q1
     * ]
     */

    template <typename T>
    __host__ void
    GPU_4STEP_NTT(T* device_in, T* device_out, Root<T>* n1_root_of_unity_table,
                  Root<T>* n2_root_of_unity_table,
                  Root<T>* W_root_of_unity_table, Modulus<T> modulus,
                  ntt4step_configuration<T> cfg, int batch_size);
    /*
     * | GPU_4STEP_NTT |
     * [batch_size]: polynomial count
     * [mod_count]:  modulus count
     *
     * example1: batch_size = 8, mod_count = 1
     *   - poly order   : [poly1, poly2, poly3, poly4, poly5, poly6, poly7,
     * poly8]
     *   - modulus order: [ q1  ,   q1 ,   q1 ,   q1 ,   q1 ,   q1 ,   q1 ,   q1
     * ]
     *
     * example1: batch_size = 8, mod_count = 4
     *   - poly order   : [poly1, poly2, poly3, poly4, poly5, poly6, poly7,
     * poly8]
     *   - modulus order: [ q1  ,   q2 ,   q3 ,   q4 ,   q1 ,   q2 ,   q3 ,   q4
     * ]
     */

    template <typename T>
    __host__ void
    GPU_4STEP_NTT(T* device_in, T* device_out, Root<T>* n1_root_of_unity_table,
                  Root<T>* n2_root_of_unity_table,
                  Root<T>* W_root_of_unity_table, Modulus<T>* modulus,
                  ntt4step_rns_configuration<T> cfg, int batch_size,
                  int mod_count);

} // namespace gpuntt
#endif // NTT_4STEP_CORE_H
