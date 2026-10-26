// Copyright 2023-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

// Compile-time tests of the NTT description operators. The checks are
// static_asserts; the test passes if this file compiles.

#include "gpuntt/ntt_merge/descriptor.cuh"

#include <cstdio>

using namespace gpuntt;

using FwdNTT =
    decltype(Size<4096>() + Direction<FORWARD>() + Precision<Data64>() +
             ReductionPoly<X_N_minus>() + Block());

static_assert(std::is_same<
              FwdNTT,
              NTTDescription<Size<4096>, Direction<FORWARD>, Precision<Data64>,
                             ReductionPoly<X_N_minus>, Block>>::value);

// Descriptions add up with operators on either side and with each other.
using Left = decltype(Size<1024>() + Direction<INVERSE>());
using Right = decltype(Precision<Data32>() + Block());
static_assert(std::is_same<decltype(Left() + Right()),
                           NTTDescription<Size<1024>, Direction<INVERSE>,
                                          Precision<Data32>, Block>>::value);
static_assert(std::is_same<
              decltype(SM<890>() + Left()),
              NTTDescription<SM<890>, Size<1024>, Direction<INVERSE>>>::value);

static_assert(Size<4096>::value == 4096 && Size<4096>::log_value == 12);
static_assert(Size<2>::log_value == 1 && Size<(1 << 28)>::log_value == 28);
static_assert(BlockDim<256, 2>::x == 256 && BlockDim<256, 2>::y == 2 &&
              BlockDim<256, 2>::z == 1);

static_assert(is_ntt_operator<Size<8>>::value);
static_assert(is_ntt_operator<Block>::value);
static_assert(!is_ntt_operator<int>::value);
static_assert(!is_ntt_operator<FwdNTT>::value);
static_assert(is_ntt_description<FwdNTT>::value);
static_assert(!is_ntt_description<Size<8>>::value);

static_assert(has_operator<NTTOperator::Size, FwdNTT>::value);
static_assert(has_operator<NTTOperator::Block, FwdNTT>::value);
static_assert(!has_operator<NTTOperator::BlockDim, FwdNTT>::value);

static_assert(std::is_same<get_or_default_t<NTTOperator::Precision, FwdNTT,
                                            Precision<Data32>>,
                           Precision<Data64>>::value);
static_assert(
    std::is_same<get_or_default_t<NTTOperator::BlockDim, FwdNTT, BlockDim<512>>,
                 BlockDim<512>>::value);

// Operator values are usable in device code.
__global__ void use_in_kernel(int* out)
{
    out[0] = Size<4096>::log_value + BlockDim<256>::x;
}

int main()
{
    std::printf("descriptor: all compile-time checks passed\n");
    return 0;
}
