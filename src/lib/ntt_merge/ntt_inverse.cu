// Copyright 2023-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan
// Paper: https://eprint.iacr.org/2023/1410

#include "ntt_launch_impl.cuh"

namespace gpuntt
{
    // Inverse PerPolynomial transforms, single modulus and RNS.

#define GPUNTT_INSTANTIATE_INVERSE(T)                                          \
    template void inverse_per_polynomial<T>(                                   \
        UnsignedOf<T>*, T*, UnsignedOf<T>*, SingleModulus<UnsignedOf<T>>, int, \
        ReductionPolynomial, int, cudaStream_t);                               \
    template void inverse_per_polynomial<T>(                                   \
        UnsignedOf<T>*, T*, UnsignedOf<T>*, RnsModulus<UnsignedOf<T>>, int,    \
        ReductionPolynomial, int, cudaStream_t);

    GPUNTT_INSTANTIATE_INVERSE(Data32)
    GPUNTT_INSTANTIATE_INVERSE(Data64)
    GPUNTT_INSTANTIATE_INVERSE(Data32s)
    GPUNTT_INSTANTIATE_INVERSE(Data64s)

#undef GPUNTT_INSTANTIATE_INVERSE

} // namespace gpuntt
