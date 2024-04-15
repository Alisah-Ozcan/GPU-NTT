// (C) Ulvetanna Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan
// Paper: https://eprint.iacr.org/2023/1410

#ifndef BUTTERFLY_UNIT_H
#define BUTTERFLY_UNIT_H

#include <curand_kernel.h>
#include <stdio.h>

#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>

#include "cuda_runtime.h"
#include "nttparameters.cuh"

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

#endif  // BUTTERFLY_UNIT_H
