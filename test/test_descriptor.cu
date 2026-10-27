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

constexpr bool same_dim(dim3 d, unsigned x, unsigned y, unsigned z)
{
    return d.x == x && d.y == y && d.z == z;
}

template <typename T, typename = void> struct has_block_dim : std::false_type
{
};

template <typename T>
struct has_block_dim<T, std::void_t<decltype(T::block_dim)>> : std::true_type
{
};

// A partial description has no members.
static_assert(!has_block_dim<decltype(Size<256>() + Block())>::value);

// 2^12 forward, Data64: two stage kernels (kernel table entry 12).
static_assert(FwdNTT::size == 4096 && FwdNTT::log_size == 12);
static_assert(FwdNTT::direction == FORWARD);
static_assert(std::is_same<FwdNTT::value_type, Data64>::value);
static_assert(FwdNTT::num_stages == 2 && FwdNTT::batches_per_block == 1);
static_assert(same_dim(FwdNTT::stage<0>::block_dim, 64, 4, 1));
static_assert(same_dim(FwdNTT::stage<0>::grid(3), 8, 1, 3));
static_assert(FwdNTT::stage<0>::shared_memory_size == 512 * sizeof(Data64));
static_assert(same_dim(FwdNTT::stage<1>::block_dim, 256, 1, 1));
static_assert(same_dim(FwdNTT::stage<1>::grid(3), 1, 8, 3));
static_assert(!has_block_dim<FwdNTT>::value);

// The inverse runs the stages in the opposite order.
using InvNTT = decltype(Size<4096>() + Direction<INVERSE>() +
                        Precision<Data64>() + Block());
static_assert(same_dim(InvNTT::stage<0>::block_dim, 256, 1, 1));
static_assert(same_dim(InvNTT::stage<1>::block_dim, 64, 4, 1));

// 2^8: low ring kernel, 2 polynomials per block by default.
using Small = decltype(Size<256>() + Direction<FORWARD>() +
                       Precision<Data32>() + Block());
static_assert(Small::num_stages == 1 && Small::batches_per_block == 2);
static_assert(same_dim(Small::block_dim, 128, 2, 1));
static_assert(Small::shared_memory_size == 512 * sizeof(Data32));
static_assert(same_dim(Small::grid(5), 3, 1, 1));

using SmallBatches = decltype(Small() + BatchesPerBlock<4>());
static_assert(SmallBatches::batches_per_block == 4);
static_assert(same_dim(SmallBatches::block_dim, 128, 4, 1));
static_assert(SmallBatches::shared_memory_size == 1024 * sizeof(Data32));
static_assert(same_dim(SmallBatches::grid(5), 2, 1, 1));

using SmallBlockDim = decltype(Small() + BlockDim<128, 8>());
static_assert(SmallBlockDim::batches_per_block == 8);
static_assert(same_dim(SmallBlockDim::block_dim, 128, 8, 1));

// 2^10: a single stage kernel, one polynomial per block. The single kernel
// API indexes blocks by blockIdx.x; the stage API, like the host API, by
// blockIdx.z.
using Mid = decltype(Size<1024>() + Direction<FORWARD>() + Precision<Data64>() +
                     Block());
static_assert(Mid::num_stages == 1 && Mid::batches_per_block == 1);
static_assert(same_dim(Mid::block_dim, 512, 1, 1));
static_assert(same_dim(Mid::grid(7), 7, 1, 1));
static_assert(same_dim(Mid::stage<0>::grid(7), 1, 1, 7));
static_assert(Mid::shared_memory_size == 1024 * sizeof(Data64));

// The last forward stage and the first inverse stage of every multi-kernel
// ring size have the same launch parameters and thread mapping, so a kernel
// can run both without storing in between (see NTTStage::for_each).
template <typename T> constexpr bool last_forward_matches_first_inverse()
{
    for (int log_n = 11; log_n <= 28; log_n++)
    {
        const KernelStages forward = forward_kernel_table<T>[log_n];
        const KernelConfig a = forward[forward.size() - 1];
        const KernelConfig b = inverse_kernel_table<T>[log_n][0];
        if (a.griddim_x != b.griddim_x || a.griddim_y != b.griddim_y ||
            a.blockdim_x != b.blockdim_x || a.blockdim_y != b.blockdim_y ||
            a.shared_memory != b.shared_memory || a.logm != b.k ||
            a.outer_iteration_count != b.outer_iteration_count)
        {
            return false;
        }
    }
    return true;
}
static_assert(last_forward_matches_first_inverse<Data32>());
static_assert(last_forward_matches_first_inverse<Data64>());

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
