// Copyright 2023-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan
// Paper: https://eprint.iacr.org/2023/1410

#include "ntt_launch_impl.cuh"

namespace gpuntt
{
    // Forward PerPolynomial transforms, single modulus and RNS.

#define GPUNTT_INSTANTIATE_FORWARD(T)                                          \
    template void forward_per_polynomial<T>(                                   \
        T*, UnsignedOf<T>*, Root<UnsignedOf<T>>*,                              \
        SingleModulus<UnsignedOf<T>>, int, ReductionPolynomial, int,           \
        cudaStream_t);                                                         \
    template void forward_per_polynomial<T>(                                   \
        T*, UnsignedOf<T>*, Root<UnsignedOf<T>>*, RnsModulus<UnsignedOf<T>>,   \
        int, ReductionPolynomial, int, cudaStream_t);

    GPUNTT_INSTANTIATE_FORWARD(Data32)
    GPUNTT_INSTANTIATE_FORWARD(Data64)
    GPUNTT_INSTANTIATE_FORWARD(Data32s)
    GPUNTT_INSTANTIATE_FORWARD(Data64s)

#undef GPUNTT_INSTANTIATE_FORWARD

} // namespace gpuntt
