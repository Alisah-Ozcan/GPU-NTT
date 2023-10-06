// (C) Ulvetanna Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan
// Paper: https://eprint.iacr.org/2023/1410

#include <curand_kernel.h>
#include <stdio.h>

#include <chrono>
#include <fstream>
#include <iostream>

#include "common.cuh"
#include "cuda_runtime.h"
#include "ntt_cpu.cuh"

// --------------------- //
// Authors: Alisah Ozcan
// --------------------- //

#ifndef NTT_FFT_CORE_H
#define NTT_FFT_CORE_H

typedef unsigned location_t;
/*
#if MAX_LOG2_RINGSIZE <= 32
typedef unsigned location_t;
#else
typedef unsigned long long location_t;
#endif
*/
enum type
{
    FORWARD,
    INVERSE
};

struct ntt_configuration
{
    int n_power;
    type ntt_type;
    ReductionPolynomial reduction_poly;
    bool zero_padding;
    Ninverse mod_inverse;
    cudaStream_t stream;
};

__device__ void CooleyTukeyUnit(Data& U, Data& V, Root& root, Modulus& modulus);

__device__ void GentlemanSandeUnit(Data& U, Data& V, Root& root,
                                   Modulus& modulus);

__global__ void ForwardCore(Data* polynomial, Root* root_of_unity_table,
                            Modulus modulus, int shared_index, int logm,
                            int outer_iteration_count, int N_power,
                            bool zero_padding, bool not_last_kernel,
                            bool reduction_poly_check);

__global__ void ForwardCore_(Data* polynomial, Root* root_of_unity_table,
                            Modulus modulus, int shared_index, int logm,
                            int outer_iteration_count, int N_power,
                            bool zero_padding, bool not_last_kernel,
                            bool reduction_poly_check);

__global__ void InverseCore(Data* polynomial, Root* inverse_root_of_unity_table,
                            Modulus modulus, int shared_index, int logm, int k,
                            int outer_iteration_count, int N_power,
                            Ninverse n_inverse, bool last_kernel,
                            bool reduction_poly_check);

__global__ void InverseCore_(Data* polynomial, Root* inverse_root_of_unity_table,
                            Modulus modulus, int shared_index, int logm, int k,
                            int outer_iteration_count, int N_power,
                            Ninverse n_inverse, bool last_kernel,
                            bool reduction_poly_check);

__host__ void GPU_NTT(Data* device_inout, Root* root_of_unity_table,
                      Modulus modulus, ntt_configuration cfg, int batch_size);

__global__ void GPU_ACTIVITY(unsigned long long* output,
                             unsigned long long fix_num);
__host__ void GPU_ACTIVITY_HOST(unsigned long long* output,
                                unsigned long long fix_num);


__global__ void GPU_ACTIVITY2(unsigned long long* input1, unsigned long long* input2);
__host__ void GPU_ACTIVITY2_HOST(unsigned long long* input1, unsigned long long* input2, unsigned size);


#endif  // NTT_FFT_CORE_H
