// Copyright 2023-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan
// Paper: https://eprint.iacr.org/2023/1410

#include "ntt_launch_impl.cuh"

namespace gpuntt
{
    // Modulus ordered and poly ordered RNS transforms.

#define GPUNTT_INSTANTIATE_ORDERED(T)                                          \
    template void ordered_stages<T>(T*, T*, T*, OrderedRnsModulus<T>,          \
                                    SequentialPolys, type, int,                \
                                    ReductionPolynomial, int, cudaStream_t);   \
    template void ordered_stages<T>(T*, T*, T*, RnsModulus<T>, OrderedPolys,   \
                                    type, int, ReductionPolynomial, int,       \
                                    cudaStream_t);

    GPUNTT_INSTANTIATE_ORDERED(Data32)
    GPUNTT_INSTANTIATE_ORDERED(Data64)

#undef GPUNTT_INSTANTIATE_ORDERED

} // namespace gpuntt
