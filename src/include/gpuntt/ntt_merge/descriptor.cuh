// Copyright 2023-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan
// Paper: https://eprint.iacr.org/2023/1410

#ifndef NTT_DESCRIPTOR_H
#define NTT_DESCRIPTOR_H

#include "gpuntt/config.cuh"
#include "gpuntt/common/fields.cuh"
#include "gpuntt/common/nttparameters.cuh"
#include "gpuntt/ntt_merge/ntt_device.cuh"
#include <cstddef>
#include <type_traits>

namespace gpuntt
{
    struct KernelConfig
    {
        int griddim_x;
        int griddim_y;
        int blockdim_x;
        int blockdim_y;
        size_t shared_memory;

        int shared_index;
        int logm;
        int k;
        int outer_iteration_count;

        bool not_last_kernel;
    };

    // The kernels of one transform, in launch order; unused entries have
    // blockdim_x == 0.
    struct KernelStages
    {
        int n_power;
        KernelConfig stage[4];

        constexpr int size() const
        {
            int count = 0;
            while (count < 4 && stage[count].blockdim_x != 0)
            {
                count++;
            }
            return count;
        }

        constexpr const KernelConfig& operator[](int i) const
        {
            return stage[i];
        }
    };

    constexpr KernelStages Stages(int n_power, KernelConfig stage0,
                                  KernelConfig stage1 = {},
                                  KernelConfig stage2 = {},
                                  KernelConfig stage3 = {})
    {
        return KernelStages{n_power, {stage0, stage1, stage2, stage3}};
    }

    // Kernel parameters per log2(ring size), indexed by log2(ring size).
    // GPUNTT_CC_89_KERNEL_CONFIG (CMake option, ON by default, recorded in
    // gpuntt/config.cuh) selects the configuration tuned for compute
    // capability 8.9 for log2(ring size) 27 and 28.
    template <typename T>
    inline constexpr KernelStages forward_kernel_table[] = {
        KernelStages{},
        Stages(1, {1, 1, 1, 256, 512 * sizeof(T), 0, 0, 0, 1, false}),
        Stages(2, {1, 1, 2, 128, 512 * sizeof(T), 1, 0, 0, 2, false}),
        Stages(3, {1, 1, 4, 64, 512 * sizeof(T), 2, 0, 0, 3, false}),
        Stages(4, {1, 1, 8, 32, 512 * sizeof(T), 3, 0, 0, 4, false}),
        Stages(5, {1, 1, 16, 16, 512 * sizeof(T), 4, 0, 0, 5, false}),
        Stages(6, {1, 1, 32, 8, 512 * sizeof(T), 5, 0, 0, 6, false}),
        Stages(7, {1, 1, 64, 4, 512 * sizeof(T), 6, 0, 0, 7, false}),
        Stages(8, {1, 1, 128, 2, 512 * sizeof(T), 7, 0, 0, 8, false}),
        Stages(9, {1, 1, 256, 1, 512 * sizeof(T), 8, 0, 0, 9, false}),
        Stages(10, {1, 1, 512, 1, 1024 * sizeof(T), 9, 0, 0, 10, false}),
        Stages(11, {4, 1, 128, 2, 512 * sizeof(T), 8, 0, 0, 2, true},
               {1, 4, 256, 1, 512 * sizeof(T), 8, 2, 0, 9, false}),
        Stages(12, {8, 1, 64, 4, 512 * sizeof(T), 8, 0, 0, 3, true},
               {1, 8, 256, 1, 512 * sizeof(T), 8, 3, 0, 9, false}),
        Stages(13, {16, 1, 32, 8, 512 * sizeof(T), 8, 0, 0, 4, true},
               {1, 16, 256, 1, 512 * sizeof(T), 8, 4, 0, 9, false}),
        Stages(14, {32, 1, 16, 16, 512 * sizeof(T), 8, 0, 0, 5, true},
               {1, 32, 256, 1, 512 * sizeof(T), 8, 5, 0, 9, false}),
        Stages(15, {64, 1, 8, 32, 512 * sizeof(T), 8, 0, 0, 6, true},
               {1, 64, 256, 1, 512 * sizeof(T), 8, 6, 0, 9, false}),
        Stages(16, {128, 1, 4, 64, 512 * sizeof(T), 8, 0, 0, 7, true},
               {1, 128, 256, 1, 512 * sizeof(T), 8, 7, 0, 9, false}),
        Stages(17, {256, 1, 32, 8, 512 * sizeof(T), 8, 0, 0, 4, true},
               {16, 16, 32, 8, 512 * sizeof(T), 8, 4, 0, 4, true},
               {1, 256, 256, 1, 512 * sizeof(T), 8, 8, 0, 9, false}),
        Stages(18, {512, 1, 32, 8, 512 * sizeof(T), 8, 0, 0, 4, true},
               {32, 16, 16, 16, 512 * sizeof(T), 8, 4, 0, 5, true},
               {1, 512, 256, 1, 512 * sizeof(T), 8, 9, 0, 9, false}),
        Stages(19, {1024, 1, 16, 16, 512 * sizeof(T), 8, 0, 0, 5, true},
               {32, 32, 16, 16, 512 * sizeof(T), 8, 5, 0, 5, true},
               {1, 1024, 256, 1, 512 * sizeof(T), 8, 10, 0, 9, false}),
        Stages(20, {2048, 1, 16, 16, 512 * sizeof(T), 8, 0, 0, 5, true},
               {64, 32, 8, 32, 512 * sizeof(T), 8, 5, 0, 6, true},
               {1, 2048, 256, 1, 512 * sizeof(T), 8, 11, 0, 9, false}),
        Stages(21, {4096, 1, 8, 32, 512 * sizeof(T), 8, 0, 0, 6, true},
               {64, 64, 8, 32, 512 * sizeof(T), 8, 6, 0, 6, true},
               {1, 4096, 256, 1, 512 * sizeof(T), 8, 12, 0, 9, false}),
        Stages(22, {8192, 1, 8, 32, 512 * sizeof(T), 8, 0, 0, 6, true},
               {128, 64, 4, 64, 512 * sizeof(T), 8, 6, 0, 7, true},
               {1, 8192, 256, 1, 512 * sizeof(T), 8, 13, 0, 9, false}),
        Stages(23, {16384, 1, 4, 64, 512 * sizeof(T), 8, 0, 0, 7, true},
               {128, 128, 4, 64, 512 * sizeof(T), 8, 7, 0, 7, true},
               {1, 16384, 256, 1, 512 * sizeof(T), 8, 14, 0, 9, false}),
        Stages(24, {16384, 1, 8, 64, 1024 * sizeof(T), 9, 0, 0, 7, true},
               {128, 128, 8, 64, 1024 * sizeof(T), 9, 7, 0, 7, true},
               {1, 16384, 512, 1, 1024 * sizeof(T), 9, 14, 0, 10, false}),
        Stages(25, {32768, 1, 8, 64, 1024 * sizeof(T), 9, 0, 0, 7, true},
               {256, 128, 4, 128, 1024 * sizeof(T), 9, 7, 0, 8, true},
               {32768, 1, 512, 1, 1024 * sizeof(T), 9, 15, 0, 10, false}),
        Stages(26, {65536, 1, 4, 128, 1024 * sizeof(T), 9, 0, 0, 8, true},
               {256, 256, 4, 128, 1024 * sizeof(T), 9, 8, 0, 8, true},
               {65536, 1, 512, 1, 1024 * sizeof(T), 9, 16, 0, 10, false}),
#ifndef GPUNTT_CC_89_KERNEL_CONFIG
        Stages(27, {262144, 1, 16, 16, 512 * sizeof(T), 8, 0, 0, 5, true},
               {8192, 32, 8, 32, 512 * sizeof(T), 8, 5, 0, 6, true},
               {128, 2048, 4, 64, 512 * sizeof(T), 8, 11, 0, 7, true},
               {262144, 1, 256, 1, 512 * sizeof(T), 8, 18, 0, 9, false}),
        Stages(28, {524288, 1, 8, 32, 512 * sizeof(T), 8, 0, 0, 6, true},
               {8192, 64, 8, 32, 512 * sizeof(T), 8, 6, 0, 6, true},
               {128, 4096, 4, 64, 512 * sizeof(T), 8, 12, 0, 7, true},
               {524288, 1, 256, 1, 512 * sizeof(T), 8, 19, 0, 9, false}),
#else
        Stages(27, {131072, 1, 4, 128, 1024 * sizeof(T), 9, 0, 0, 8, true},
               {512, 256, 2, 256, 1024 * sizeof(T), 9, 8, 0, 9, true},
               {131072, 1, 512, 1, 1024 * sizeof(T), 9, 17, 0, 10, false}),
        Stages(28, {262144, 1, 2, 256, 1024 * sizeof(T), 9, 0, 0, 9, true},
               {512, 512, 2, 256, 1024 * sizeof(T), 9, 9, 0, 9, true},
               {262144, 1, 512, 1, 1024 * sizeof(T), 9, 18, 0, 10, false}),
#endif
    };

    template <typename T>
    inline constexpr KernelStages inverse_kernel_table[] = {
        KernelStages{},
        Stages(1, {1, 1, 1, 256, 512 * sizeof(T), 0, 0, 0, 1, true}),
        Stages(2, {1, 1, 2, 128, 512 * sizeof(T), 1, 1, 0, 2, true}),
        Stages(3, {1, 1, 4, 64, 512 * sizeof(T), 2, 2, 0, 3, true}),
        Stages(4, {1, 1, 8, 32, 512 * sizeof(T), 3, 3, 0, 4, true}),
        Stages(5, {1, 1, 16, 16, 512 * sizeof(T), 4, 4, 0, 5, true}),
        Stages(6, {1, 1, 32, 8, 512 * sizeof(T), 5, 5, 0, 6, true}),
        Stages(7, {1, 1, 64, 4, 512 * sizeof(T), 6, 6, 0, 7, true}),
        Stages(8, {1, 1, 128, 2, 512 * sizeof(T), 7, 7, 0, 8, true}),
        Stages(9, {1, 1, 256, 1, 512 * sizeof(T), 8, 8, 0, 9, true}),
        Stages(10, {1, 1, 512, 1, 1024 * sizeof(T), 9, 9, 0, 10, true}),
        Stages(11, {1, 4, 256, 1, 512 * sizeof(T), 8, 10, 2, 9, false},
               {4, 1, 128, 2, 512 * sizeof(T), 8, 1, 0, 2, true}),
        Stages(12, {1, 8, 256, 1, 512 * sizeof(T), 8, 11, 3, 9, false},
               {8, 1, 64, 4, 512 * sizeof(T), 8, 2, 0, 3, true}),
        Stages(13, {1, 16, 256, 1, 512 * sizeof(T), 8, 12, 4, 9, false},
               {16, 1, 32, 8, 512 * sizeof(T), 8, 3, 0, 4, true}),
        Stages(14, {1, 32, 256, 1, 512 * sizeof(T), 8, 13, 5, 9, false},
               {32, 1, 16, 16, 512 * sizeof(T), 8, 4, 0, 5, true}),
        Stages(15, {1, 64, 256, 1, 512 * sizeof(T), 8, 14, 6, 9, false},
               {64, 1, 8, 32, 512 * sizeof(T), 8, 5, 0, 6, true}),
        Stages(16, {1, 128, 256, 1, 512 * sizeof(T), 8, 15, 7, 9, false},
               {128, 1, 4, 64, 512 * sizeof(T), 8, 6, 0, 7, true}),
        Stages(17, {1, 256, 256, 1, 512 * sizeof(T), 8, 16, 8, 9, false},
               {16, 16, 32, 8, 512 * sizeof(T), 8, 7, 4, 4, false},
               {256, 1, 32, 8, 512 * sizeof(T), 8, 3, 0, 4, true}),
        Stages(18, {1, 512, 256, 1, 512 * sizeof(T), 8, 17, 9, 9, false},
               {32, 16, 16, 16, 512 * sizeof(T), 8, 8, 4, 5, false},
               {512, 1, 32, 8, 512 * sizeof(T), 8, 3, 0, 4, true}),
        Stages(19, {1, 1024, 256, 1, 512 * sizeof(T), 8, 18, 10, 9, false},
               {32, 32, 16, 16, 512 * sizeof(T), 8, 9, 5, 5, false},
               {1024, 1, 16, 16, 512 * sizeof(T), 8, 4, 0, 5, true}),
        Stages(20, {1, 2048, 256, 1, 512 * sizeof(T), 8, 19, 11, 9, false},
               {64, 32, 8, 32, 512 * sizeof(T), 8, 10, 5, 6, false},
               {2048, 1, 16, 16, 512 * sizeof(T), 8, 4, 0, 5, true}),
        Stages(21, {1, 4096, 256, 1, 512 * sizeof(T), 8, 20, 12, 9, false},
               {64, 64, 8, 32, 512 * sizeof(T), 8, 11, 6, 6, false},
               {4096, 1, 8, 32, 512 * sizeof(T), 8, 5, 0, 6, true}),
        Stages(22, {1, 8192, 256, 1, 512 * sizeof(T), 8, 21, 13, 9, false},
               {128, 64, 4, 64, 512 * sizeof(T), 8, 12, 6, 7, false},
               {8192, 1, 8, 32, 512 * sizeof(T), 8, 5, 0, 6, true}),
        Stages(23, {1, 16384, 256, 1, 512 * sizeof(T), 8, 22, 14, 9, false},
               {128, 128, 4, 64, 512 * sizeof(T), 8, 13, 7, 7, false},
               {16384, 1, 4, 64, 512 * sizeof(T), 8, 6, 0, 7, true}),
        Stages(24, {1, 16384, 512, 1, 1024 * sizeof(T), 9, 23, 14, 10, false},
               {128, 128, 8, 64, 1024 * sizeof(T), 9, 13, 7, 7, false},
               {16384, 1, 8, 64, 1024 * sizeof(T), 9, 6, 0, 7, true}),
        Stages(25, {32768, 1, 512, 1, 1024 * sizeof(T), 9, 24, 15, 10, false},
               {256, 128, 4, 128, 1024 * sizeof(T), 9, 14, 7, 8, false},
               {32768, 1, 8, 64, 1024 * sizeof(T), 9, 6, 0, 7, true}),
        Stages(26, {65536, 1, 512, 1, 1024 * sizeof(T), 9, 25, 16, 10, false},
               {256, 256, 4, 128, 1024 * sizeof(T), 9, 15, 8, 8, false},
               {65536, 1, 4, 128, 1024 * sizeof(T), 9, 7, 0, 8, true}),
#ifndef GPUNTT_CC_89_KERNEL_CONFIG
        Stages(27, {262144, 1, 256, 1, 512 * sizeof(T), 8, 26, 18, 9, false},
               {128, 2048, 4, 64, 512 * sizeof(T), 8, 17, 11, 7, false},
               {8192, 32, 8, 32, 512 * sizeof(T), 8, 10, 5, 6, false},
               {262144, 1, 16, 16, 512 * sizeof(T), 8, 4, 0, 5, true}),
        Stages(28, {524288, 1, 256, 1, 512 * sizeof(T), 8, 27, 19, 9, false},
               {128, 4096, 4, 64, 512 * sizeof(T), 8, 18, 12, 7, false},
               {8192, 64, 8, 32, 512 * sizeof(T), 8, 11, 6, 6, false},
               {524288, 1, 8, 32, 512 * sizeof(T), 8, 5, 0, 6, true}),
#else
        Stages(27, {131072, 1, 512, 1, 1024 * sizeof(T), 9, 26, 17, 10, false},
               {512, 256, 2, 256, 1024 * sizeof(T), 9, 16, 8, 9, false},
               {131072, 1, 4, 128, 1024 * sizeof(T), 9, 7, 0, 8, true}),
        Stages(28, {262144, 1, 512, 1, 1024 * sizeof(T), 9, 27, 18, 10, false},
               {512, 512, 2, 256, 1024 * sizeof(T), 9, 17, 9, 9, false},
               {262144, 1, 2, 256, 1024 * sizeof(T), 9, 8, 0, 9, true}),
#endif
    };

    // Checks that entry n of a table describes ring size 2^n, that its
    // stages are contiguous, and that only the last stage has
    // not_last_kernel == last_flag (forward: false, inverse: the last_kernel
    // flag, true).
    template <size_t Count>
    constexpr bool IsValidKernelTable(const KernelStages (&table)[Count],
                                      bool last_flag)
    {
        if (Count != 29 || table[0].size() != 0)
        {
            return false;
        }
        for (int n = 1; n < 29; n++)
        {
            const KernelStages& stages = table[n];
            const int count = stages.size();
            if (stages.n_power != n || count == 0)
            {
                return false;
            }
            for (int i = 0; i < 4; i++)
            {
                if (i >= count && stages[i].blockdim_x != 0)
                {
                    return false;
                }
                if (i < count && stages[i].not_last_kernel !=
                                     (i + 1 == count ? last_flag : !last_flag))
                {
                    return false;
                }
            }
        }
        return true;
    }

    static_assert(IsValidKernelTable(forward_kernel_table<Data32>, false));
    static_assert(IsValidKernelTable(forward_kernel_table<Data64>, false));
    static_assert(IsValidKernelTable(inverse_kernel_table<Data32>, true));
    static_assert(IsValidKernelTable(inverse_kernel_table<Data64>, true));

    // Compile-time description of an NTT, for use inside user kernels. A
    // description is the sum of operators, each fixing one property:
    //
    //   using FwdNTT = decltype(Size<4096>() + Direction<FORWARD>() +
    //                           Precision<Data64>() +
    //                           ReductionPoly<X_N_minus>() + Block());
    //
    // Every operator may appear at most once.

    enum class NTTOperator
    {
        Size,
        Direction,
        Precision,
        ReductionPoly,
        Layout,
        NumModuli,
        Block,
        BlockDim,
        BatchesPerBlock,
        SM
    };

    template <NTTOperator Op> struct NTTOperatorBase
    {
        static constexpr NTTOperator op = Op;
    };

    constexpr bool is_power_of_two(int n)
    {
        return n > 0 && (n & (n - 1)) == 0;
    }

    constexpr int log2_of_power_of_two(int n)
    {
        int log = 0;
        while ((1 << log) < n)
        {
            log++;
        }
        return log;
    }

    // Ring size N (number of coefficients of one polynomial).
    template <int N> struct Size : NTTOperatorBase<NTTOperator::Size>
    {
        static_assert(is_power_of_two(N) && N >= 2 && N <= (1 << 28),
                      "Size must be a power of two in [2, 2^28]");
        static constexpr int value = N;
        static constexpr int log_value = log2_of_power_of_two(N);
    };

    template <type D> struct Direction : NTTOperatorBase<NTTOperator::Direction>
    {
        static_assert(D == FORWARD || D == INVERSE,
                      "Direction must be FORWARD or INVERSE");
        static constexpr type value = D;
    };

    // Coefficient type: Data32 or Data64.
    template <typename T>
    struct Precision : NTTOperatorBase<NTTOperator::Precision>
    {
        static_assert(std::is_same<T, Data32>::value ||
                          std::is_same<T, Data64>::value,
                      "Precision must be Data32 or Data64");
        using value_type = T;
    };

    template <ReductionPolynomial R>
    struct ReductionPoly : NTTOperatorBase<NTTOperator::ReductionPoly>
    {
        static_assert(R == X_N_minus || R == X_N_plus,
                      "ReductionPoly must be X_N_minus or X_N_plus");
        static constexpr ReductionPolynomial value = R;
    };

    template <NTTLayout L> struct Layout : NTTOperatorBase<NTTOperator::Layout>
    {
        static_assert(L == PerPolynomial || L == PerCoefficient,
                      "Layout must be PerPolynomial or PerCoefficient");
        static constexpr NTTLayout value = L;
    };

    // Number of RNS moduli the polynomials of a batch cycle through.
    template <int K> struct NumModuli : NTTOperatorBase<NTTOperator::NumModuli>
    {
        static_assert(K >= 1, "NumModuli must be at least 1");
        static constexpr int value = K;
    };

    // The NTT is computed by a thread block.
    struct Block : NTTOperatorBase<NTTOperator::Block>
    {
    };

    template <int X, int Y = 1, int Z = 1>
    struct BlockDim : NTTOperatorBase<NTTOperator::BlockDim>
    {
        static_assert(X >= 1 && Y >= 1 && Z >= 1 && X * Y * Z <= 1024,
                      "BlockDim must have between 1 and 1024 threads");
        static constexpr int x = X;
        static constexpr int y = Y;
        static constexpr int z = Z;
    };

    template <int B>
    struct BatchesPerBlock : NTTOperatorBase<NTTOperator::BatchesPerBlock>
    {
        static_assert(B >= 1, "BatchesPerBlock must be at least 1");
        static constexpr int value = B;
    };

    // Target architecture as 100 * major + 10 * minor, e.g. SM<890>. Not
    // used yet: all architectures use the same kernel tables (see
    // GPUNTT_CC_89_KERNEL_CONFIG).
    template <int Arch> struct SM : NTTOperatorBase<NTTOperator::SM>
    {
        static_assert(Arch >= 700 && Arch % 10 == 0,
                      "SM must be 100 * major + 10 * minor, at least 700");
        static constexpr int value = Arch;
    };

    template <typename... Operators> struct NTTDescription;

    template <typename T, typename = void>
    struct is_ntt_operator : std::false_type
    {
    };

    template <typename T>
    struct is_ntt_operator<T, std::void_t<decltype(T::op)>>
        : std::is_same<std::remove_cv_t<decltype(T::op)>, NTTOperator>
    {
    };

    template <typename T> struct is_ntt_description : std::false_type
    {
    };

    template <typename... Operators>
    struct is_ntt_description<NTTDescription<Operators...>> : std::true_type
    {
    };

    template <NTTOperator Op, typename Description> struct has_operator;

    template <NTTOperator Op, typename... Operators>
    struct has_operator<Op, NTTDescription<Operators...>>
        : std::bool_constant<((Operators::op == Op) || ...)>
    {
    };

    // The first operator of kind Op in Operators, or Default if there is
    // none.
    template <NTTOperator Op, typename Default, typename... Operators>
    struct find_operator
    {
        using type = Default;
    };

    template <NTTOperator Op, typename Default, typename First,
              typename... Rest>
    struct find_operator<Op, Default, First, Rest...>
    {
        using type = std::conditional_t<
            First::op == Op, First,
            typename find_operator<Op, Default, Rest...>::type>;
    };

    template <NTTOperator Op, typename Description, typename Default>
    struct get_or_default;

    template <NTTOperator Op, typename... Operators, typename Default>
    struct get_or_default<Op, NTTDescription<Operators...>, Default>
    {
        using type = typename find_operator<Op, Default, Operators...>::type;
    };

    template <NTTOperator Op, typename Description, typename Default>
    using get_or_default_t =
        typename get_or_default<Op, Description, Default>::type;

    constexpr int kLowRingMaxLogN = 9;

    // Launch parameters and device functions of kernel I of an NTT of ring
    // size 2^LogN. Ring sizes up to 2^kLowRingMaxLogN use the low ring
    // kernel, which transforms B polynomials per block (use the single kernel
    // API for them); larger ones use one or more stage kernels. A stage
    // kernel transforms part of polynomial blockIdx.z of data: launch kernel
    // I with grid(batch), block_dim and shared_memory_size, in order I = 0,
    // 1, ..., num_stages - 1. Kernel 0 reads the input; every kernel writes
    // the output array, which the later kernels read and write in place.
    // Inside kernel I:
    //
    //   NTT::stage<I>().load(shared_memory, data, modulus);
    //   NTT::stage<I>().execute(shared_memory, roots, modulus);
    //   NTT::stage<I>().store(shared_memory, out);
    //
    // The last inverse kernel stores with store(shared_memory, out, modulus,
    // n_inverse). For RNS, pick the modulus of polynomial blockIdx.z and
    // pass roots + (prime_index << LogN) as the root table.
    //
    // for_each gives each entry with its position in data, for element-wise
    // work inside a stage kernel. The last forward stage and the first
    // inverse stage of a ring size have the same launch parameters and keep
    // the same entries in shared memory, so one kernel can run the last
    // forward execute, element-wise work and the first inverse execute
    // without storing and loading in between.
    template <typename T, type D, int LogN, int I, int B, bool HasReductionPoly,
              ReductionPolynomial R>
    struct NTTStage
    {
        static constexpr KernelStages stages =
            (D == FORWARD ? forward_kernel_table<T>
                          : inverse_kernel_table<T>) [LogN];

        static_assert(I >= 0 && I < stages.size(), "Stage index out of range");

        static constexpr KernelConfig config = stages[I];

        static constexpr bool low_ring = (LogN <= kLowRingMaxLogN);

        static constexpr dim3 block_dim =
            low_ring ? dim3(1 << (LogN - 1), B)
                     : dim3(config.blockdim_x, config.blockdim_y);

        static constexpr size_t shared_memory_size =
            low_ring ? (size_t(B) << LogN) * sizeof(T) : config.shared_memory;

        static constexpr dim3 grid(int batch)
        {
            return low_ring ? dim3((batch + B - 1) / B)
                            : dim3(config.griddim_x, config.griddim_y, batch);
        }

        static constexpr bool last = (I + 1 == stages.size());

        // For log2(ring size) >= 25 the last forward and the first inverse
        // kernel swap the roles of blockIdx.x and blockIdx.y; grid() already
        // accounts for it.
        static constexpr bool swap_grid_axes =
            LogN >= 25 && (D == FORWARD ? last : I == 0);

        template <typename TIn>
        __device__ __forceinline__ void load(T* shared_memory, TIn* data,
                                             Modulus<T> modulus) const
        {
            static_assert(!low_ring, "Use the single kernel API for Size up "
                                     "to 2^9");
            static_assert(
                std::is_same<std::make_unsigned_t<TIn>, T>::value,
                "The input must be the description's Precision or its "
                "signed type");
            StageLoad(thread(modulus, 0), shared_memory, data);
        }

        __device__ __forceinline__ void
        execute(T* shared_memory, const T* roots, Modulus<T> modulus) const
        {
            static_assert(!low_ring, "Use the single kernel API for Size up "
                                     "to 2^9");
            static_assert(HasReductionPoly,
                          "execute needs the ReductionPoly operator");
            constexpr bool x_n_minus = (R == X_N_minus);
            if constexpr (D == FORWARD)
            {
                ForwardStageExecute(thread(modulus, 0), shared_memory, roots,
                                    config.shared_index, config.logm,
                                    config.outer_iteration_count, LogN,
                                    config.not_last_kernel, x_n_minus);
            }
            else
            {
                InverseStageExecute(thread(modulus, 0), shared_memory, roots,
                                    config.shared_index, config.logm,
                                    config.outer_iteration_count, LogN,
                                    x_n_minus);
            }
        }

        // Calls f(value, index) for the two shared memory entries of this
        // thread; index is the entry's position in data, the array load
        // reads and store writes. Between load and store an entry keeps its
        // position, so after the last forward execute value is element index
        // of the GPU_NTT output. Ends with a block barrier.
        template <typename F>
        __device__ __forceinline__ void for_each(T* shared_memory, F f) const
        {
            static_assert(!low_ring, "Use the single kernel API for Size up "
                                     "to 2^9");
            const StageThread<T> state = thread(Modulus<T>(), 0);
            f(shared_memory[state.shared_address],
              size_t(state.global_address));
            f(shared_memory[state.shared_address + blockDim.x * blockDim.y],
              size_t(state.global_address + state.offset));
            __syncthreads();
        }

        __device__ __forceinline__ void store(T* shared_memory, T* data) const
        {
            static_assert(!low_ring, "Use the single kernel API for Size up "
                                     "to 2^9");
            static_assert(D == FORWARD || !last,
                          "The last inverse store needs modulus and n_inverse");
            if constexpr (D == FORWARD)
            {
                ForwardStageStore(thread(Modulus<T>(), 0), shared_memory, data);
            }
            else
            {
                InverseStageStore(thread(Modulus<T>(), 0), shared_memory, data,
                                  false);
            }
        }

        template <typename TOut>
        __device__ __forceinline__ void store(T* shared_memory, TOut* data,
                                              Modulus<T> modulus,
                                              T n_inverse) const
        {
            static_assert(!low_ring, "Use the single kernel API for Size up "
                                     "to 2^9");
            static_assert(D == INVERSE && last,
                          "Only the last inverse store takes modulus and "
                          "n_inverse");
            static_assert(
                std::is_same<std::make_unsigned_t<TOut>, T>::value,
                "The output must be the description's Precision or its "
                "signed type");
            InverseStageStore(thread(modulus, n_inverse), shared_memory, data,
                              true);
        }

      private:
        __device__ __forceinline__ static StageThread<T>
        thread(Modulus<T> modulus, T n_inverse)
        {
            return StageThreadState<swap_grid_axes, D == INVERSE, T>(
                SingleModulus<T>{modulus, n_inverse}, SequentialPolys{},
                D == FORWARD ? config.logm : config.k,
                config.outer_iteration_count, LogN);
        }
    };

    // Launch parameters and device functions of an NTT that runs in a single
    // kernel (ring sizes up to 2^10). Block b transforms the B polynomials
    // that start at data + b * B * N; launch it with grid(batch), block_dim
    // and shared_memory_size. Inside the kernel:
    //
    //   NTT().load(shared_memory, data, modulus);
    //   NTT().execute(shared_memory, roots, modulus);
    //   ... optional work on shared_memory ...
    //   NTT().store(shared_memory, data);                       // forward
    //   NTT().store(shared_memory, data, modulus, n_inverse);   // inverse
    //
    // num_polys (at most B) is the number of polynomials of the block, for a
    // last block that is not full. execute ends with a block barrier; code
    // that writes shared memory between execute and store must end with
    // __syncthreads(). The inverse execute leaves N * INTT(x) in shared
    // memory; store multiplies by n_inverse = N^-1.
    template <typename T, type D, int LogN, int B, bool HasReductionPoly,
              ReductionPolynomial R>
    struct NTTSingleKernel
    {
        static constexpr KernelConfig config =
            (D == FORWARD ? forward_kernel_table<T>
                          : inverse_kernel_table<T>) [LogN][0];

        static constexpr bool low_ring = (LogN <= kLowRingMaxLogN);

        static constexpr dim3 block_dim =
            low_ring ? dim3(1 << (LogN - 1), B)
                     : dim3(config.blockdim_x, config.blockdim_y);

        static constexpr size_t shared_memory_size =
            low_ring ? (size_t(B) << LogN) * sizeof(T) : config.shared_memory;

        static constexpr dim3 grid(int batch)
        {
            return dim3((batch + B - 1) / B);
        }

        template <typename TIn>
        __device__ __forceinline__ void load(T* shared_memory, TIn* data,
                                             Modulus<T> modulus,
                                             int num_polys = B) const
        {
            static_assert(
                std::is_same<std::make_unsigned_t<TIn>, T>::value,
                "The input must be the description's Precision or its "
                "signed type");
            if constexpr (low_ring)
            {
                LowRingLoad(low_ring_thread(modulus, 0, num_polys),
                            shared_memory, data);
            }
            else
            {
                StageLoad(stage_thread(modulus, 0), shared_memory, data);
            }
        }

        __device__ __forceinline__ void
        execute(T* shared_memory, const T* roots, Modulus<T> modulus) const
        {
            static_assert(HasReductionPoly,
                          "execute needs the ReductionPoly operator");
            constexpr bool x_n_minus = (R == X_N_minus);
            if constexpr (low_ring)
            {
                const LowRingThread<T> thread = low_ring_thread(modulus, 0, B);
                if constexpr (D == FORWARD)
                {
                    ForwardLowRingExecute(thread, shared_memory, roots,
                                          config.shared_index, LogN, x_n_minus);
                }
                else
                {
                    InverseLowRingExecute(thread, shared_memory, roots, LogN,
                                          x_n_minus);
                }
            }
            else
            {
                const StageThread<T> thread = stage_thread(modulus, 0);
                if constexpr (D == FORWARD)
                {
                    ForwardStageExecute(thread, shared_memory, roots,
                                        config.shared_index, config.logm,
                                        config.outer_iteration_count, LogN,
                                        config.not_last_kernel, x_n_minus);
                }
                else
                {
                    InverseStageExecute(thread, shared_memory, roots,
                                        config.shared_index, config.logm,
                                        config.outer_iteration_count, LogN,
                                        x_n_minus);
                }
            }
        }

        __device__ __forceinline__ void store(T* shared_memory, T* data,
                                              int num_polys = B) const
        {
            static_assert(D == FORWARD,
                          "The inverse store needs modulus and n_inverse");
            if constexpr (low_ring)
            {
                ForwardLowRingStore(low_ring_thread(Modulus<T>(), 0, num_polys),
                                    shared_memory, data);
            }
            else
            {
                ForwardStageStore(stage_thread(Modulus<T>(), 0), shared_memory,
                                  data);
            }
        }

        template <typename TOut>
        __device__ __forceinline__ void store(T* shared_memory, TOut* data,
                                              Modulus<T> modulus, T n_inverse,
                                              int num_polys = B) const
        {
            static_assert(D == INVERSE,
                          "Only the inverse store takes modulus and n_inverse");
            static_assert(
                std::is_same<std::make_unsigned_t<TOut>, T>::value,
                "The output must be the description's Precision or its "
                "signed type");
            if constexpr (low_ring)
            {
                InverseLowRingStore(
                    low_ring_thread(modulus, n_inverse, num_polys),
                    shared_memory, data);
            }
            else
            {
                InverseStageStore(stage_thread(modulus, n_inverse),
                                  shared_memory, data, config.not_last_kernel);
            }
        }

      private:
        // Thread (x, y) owns coefficients x and x + N/2 of polynomial y of
        // the block. Rows y >= num_polys read polynomial 0 and store nothing.
        __device__ __forceinline__ static LowRingThread<T>
        low_ring_thread(Modulus<T> modulus, T n_inverse, int num_polys)
        {
            const int idx_x = threadIdx.x;
            const int idx_y = threadIdx.y;
            const bool active = (idx_y < num_polys);
            const int half_n = 1 << (LogN - 1);
            const int row_base = idx_y << LogN;
            return LowRingThread<T>{
                active,
                0,
                modulus,
                n_inverse,
                location_t(idx_x + ((active ? idx_y : 0) << LogN)),
                half_n,
                row_base,
                row_base + idx_x,
                row_base + idx_x + half_n};
        }

        // The single stage kernel transforms one polynomial per block, as in
        // StageThreadState with block (0, 0).
        __device__ __forceinline__ static StageThread<T>
        stage_thread(Modulus<T> modulus, T n_inverse)
        {
            const int idx_x = threadIdx.x;
            const int idx_y = threadIdx.y;
            const int level = (D == FORWARD) ? config.logm : config.k;
            const location_t offset = 1 << (LogN - level - 1);
            const location_t address =
                idx_x +
                (location_t) (idx_y *
                              (offset /
                               (1 << (config.outer_iteration_count - 1))));
            return StageThread<T>{0,
                                  modulus,
                                  n_inverse,
                                  offset,
                                  address,
                                  address,
                                  location_t(idx_x + (idx_y * blockDim.x))};
        }
    };

    // step^(2^b) for b < log_size; a root table has at most 2^28 entries.
    template <typename T> struct TwiddleSquares
    {
        T value[28];
    };

    // table[i] = step^bitreverse(i) for i < 2^log_size: bit b of i selects
    // step^(2^(log_size - 1 - b)).
    template <typename T>
    __global__ void TwiddleTableKernel(T* table, Modulus<T> modulus,
                                       TwiddleSquares<T> squares, int log_size)
    {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= (1 << log_size))
        {
            return;
        }
        T result = 1;
        for (int b = 0; b < log_size; b++)
        {
            if ((i >> b) & 1)
            {
                result = OPERATOR_GPU<T>::mult(
                    result, squares.value[log_size - 1 - b], modulus);
            }
        }
        table[i] = result;
    }

    // Root table of an NTT of ring size N, built on the host or by a kernel.
    // root is a primitive 2N-th root of unity for X^N + 1 or a primitive
    // N-th root for X^N - 1 (psi and omega of NTTParameters); the inverse
    // table is built from root^-1. Descriptions without ReductionPoly have no
    // root table.
    template <typename T, int N, type D, bool HasReductionPoly,
              ReductionPolynomial R>
    struct NTTRootTable
    {
    };

    template <typename T, int N, type D, ReductionPolynomial R>
    struct NTTRootTable<T, N, D, true, R>
    {
        // Entries the kernels read. RNS root tables for the host API are N
        // entries apart for both reduction polynomials.
        static constexpr int twiddle_table_size = (R == X_N_plus) ? N : N / 2;

        // The powers of root (forward) or root^-1 (inverse) in bit-reversed
        // order, the table that execute and GPU_NTT / GPU_INTT take.
        static std::vector<T> make_twiddles_host(const Modulus<T>& modulus,
                                                 T root)
        {
            const T step = table_step(modulus, root);
            std::vector<T> powers(twiddle_table_size);
            powers[0] = 1;
            for (int i = 1; i < twiddle_table_size; i++)
            {
                powers[i] = OPERATOR<T>::mult(powers[i - 1], step, modulus);
            }

            std::vector<T> table(twiddle_table_size);
            for (int i = 0; i < twiddle_table_size; i++)
            {
                int reversed = 0;
                for (int b = 0; b < log_table_size; b++)
                {
                    reversed |= ((i >> b) & 1) << (log_table_size - 1 - b);
                }
                table[i] = powers[reversed];
            }
            return table;
        }

        // The same table for a prime of fields.cuh, with its root of unity
        // of order 2 * twiddle_table_size.
        static std::vector<T> make_twiddles_host(const NTTPrime<T>& prime)
        {
            return make_twiddles_host(Modulus<T>(prime.value),
                                      prime_root(prime));
        }

        // The table of make_twiddles_host, written by a kernel to table,
        // which holds twiddle_table_size entries in device memory. Faster
        // than building it on the host for large N; the kernel runs on
        // stream and table is ready when it finishes.
        static void make_twiddles(T* table, const Modulus<T>& modulus, T root,
                                  cudaStream_t stream = 0)
        {
            TwiddleSquares<T> squares = {};
            T square = table_step(modulus, root);
            for (int b = 0; b < log_table_size; b++)
            {
                squares.value[b] = square;
                square = OPERATOR<T>::mult(square, square, modulus);
            }
            constexpr int threads = 256;
            const int blocks = (twiddle_table_size + threads - 1) / threads;
            TwiddleTableKernel<<<blocks, threads, 0, stream>>>(
                table, modulus, squares, log_table_size);
            GPUNTT_CUDA_CHECK(cudaGetLastError());
        }

        static void make_twiddles(T* table, const NTTPrime<T>& prime,
                                  cudaStream_t stream = 0)
        {
            make_twiddles(table, Modulus<T>(prime.value), prime_root(prime),
                          stream);
        }

      private:
        static constexpr int log_table_size =
            log2_of_power_of_two(twiddle_table_size);

        // root (forward) or root^-1 (inverse), after checking that root has
        // order 2 * twiddle_table_size: root^twiddle_table_size = -1 holds
        // exactly for those roots.
        static T table_step(const Modulus<T>& modulus, T root)
        {
            if (OPERATOR<T>::exp(root, twiddle_table_size, modulus) !=
                modulus.value - 1)
            {
                throw std::invalid_argument(
                    "Root " + std::to_string(root) + " is not a primitive " +
                    std::to_string(2 * twiddle_table_size) +
                    "-th root of unity modulo " +
                    std::to_string(modulus.value) + ".");
            }
            return (D == FORWARD) ? root : OPERATOR<T>::modinv(root, modulus);
        }

        static T prime_root(const NTTPrime<T>& prime)
        {
            return root_of_unity(prime,
                                 log2_of_power_of_two(2 * twiddle_table_size));
        }
    };

    // Values of a description that has Size, Direction and Precision.
    template <typename... Operators> struct NTTDescriptionValues
    {
      private:
        template <NTTOperator Op>
        using get = typename find_operator<Op, void, Operators...>::type;

        using batches_op = get<NTTOperator::BatchesPerBlock>;
        using block_dim_op = get<NTTOperator::BlockDim>;
        using layout_op = get<NTTOperator::Layout>;
        using reduction_op = get<NTTOperator::ReductionPoly>;

        static constexpr bool has_batches = !std::is_void<batches_op>::value;
        static constexpr bool has_block_dim =
            !std::is_void<block_dim_op>::value;

        // True if Op is absent or is BlockDim<x, y> with x == half_size.
        template <typename Op>
        static constexpr bool block_dim_is_rows(int half_size)
        {
            if constexpr (std::is_void<Op>::value)
            {
                return true;
            }
            else
            {
                return Op::x == half_size && Op::z == 1;
            }
        }

        template <typename Op> static constexpr int block_dim_y()
        {
            if constexpr (std::is_void<Op>::value)
            {
                return 0;
            }
            else
            {
                return Op::y;
            }
        }

        template <typename Op> static constexpr int batches_value()
        {
            if constexpr (std::is_void<Op>::value)
            {
                return 0;
            }
            else
            {
                return Op::value;
            }
        }

        template <typename Op>
        static constexpr ReductionPolynomial reduction_value()
        {
            if constexpr (std::is_void<Op>::value)
            {
                return X_N_minus;
            }
            else
            {
                return Op::value;
            }
        }

        template <typename Op> static constexpr bool is_per_polynomial()
        {
            if constexpr (std::is_void<Op>::value)
            {
                return true;
            }
            else
            {
                return Op::value == PerPolynomial;
            }
        }

      public:
        using value_type = typename get<NTTOperator::Precision>::value_type;

        static constexpr int size = get<NTTOperator::Size>::value;
        static constexpr int log_size = get<NTTOperator::Size>::log_value;
        static constexpr type direction = get<NTTOperator::Direction>::value;

        // N^-1 modulo a prime modulus, the n_inverse of the inverse store.
        static value_type n_inv(const Modulus<value_type>& modulus)
        {
            return OPERATOR<value_type>::modinv(
                static_cast<value_type>(size) % modulus.value, modulus);
        }

      private:
        static constexpr KernelStages stages =
            (direction == FORWARD
                 ? forward_kernel_table<value_type>
                 : inverse_kernel_table<value_type>) [log_size];

      public:
        static constexpr int num_stages = stages.size();

        // Polynomials per block. Only the low ring kernel takes more than one;
        // its default comes from the kernel table.
        static constexpr int batches_per_block =
            has_batches                   ? batches_value<batches_op>()
            : has_block_dim               ? block_dim_y<block_dim_op>()
            : log_size <= kLowRingMaxLogN ? stages[0].blockdim_y
                                          : 1;

        static_assert(is_per_polynomial<layout_op>(),
                      "Only Layout<PerPolynomial> is supported");
        static_assert(log_size <= kLowRingMaxLogN ||
                          (!has_batches && !has_block_dim),
                      "BlockDim and BatchesPerBlock can only be set for Size "
                      "up to 2^9");
        static_assert(block_dim_is_rows<block_dim_op>(size / 2),
                      "BlockDim must be (Size / 2, BatchesPerBlock)");
        static_assert(!has_block_dim || !has_batches ||
                          block_dim_y<block_dim_op>() ==
                              batches_value<batches_op>(),
                      "BlockDim must be (Size / 2, BatchesPerBlock)");
        static_assert(
            log_size > kLowRingMaxLogN ||
                (size / 2) * batches_per_block <= 1024,
            "Size / 2 * BatchesPerBlock must be at most 1024 threads");

        template <int I>
        using stage =
            NTTStage<value_type, direction, log_size, I, batches_per_block,
                     !std::is_void<reduction_op>::value,
                     reduction_value<reduction_op>()>;

        using single_kernel =
            NTTSingleKernel<value_type, direction, log_size, batches_per_block,
                            !std::is_void<reduction_op>::value,
                            reduction_value<reduction_op>()>;

        using root_table = NTTRootTable<value_type, size, direction,
                                        !std::is_void<reduction_op>::value,
                                        reduction_value<reduction_op>()>;
    };

    struct NTTNoMembers
    {
    };

    template <typename... Operators>
    constexpr bool has_required_operators =
        ((Operators::op == NTTOperator::Size) || ...) &&
        ((Operators::op == NTTOperator::Direction) || ...) &&
        ((Operators::op == NTTOperator::Precision) || ...);

    template <bool Complete, typename... Operators>
    struct NTTDescriptionMembers : NTTNoMembers
    {
    };

    // A description with a single kernel also has that kernel's launch
    // parameters and device functions (see NTTSingleKernel); one with
    // ReductionPoly also has the root table helpers (see NTTRootTable).
    template <typename... Operators>
    struct NTTDescriptionMembers<true, Operators...>
        : NTTDescriptionValues<Operators...>,
          NTTDescriptionValues<Operators...>::root_table,
          std::conditional_t<
              NTTDescriptionValues<Operators...>::num_stages == 1,
              typename NTTDescriptionValues<Operators...>::single_kernel,
              NTTNoMembers>
    {
    };

    // Descriptions without Size, Direction and Precision (for example the
    // partial sums while a description is built) have no members.
    template <typename... Operators>
    struct NTTDescription
        : NTTDescriptionMembers<has_required_operators<Operators...>,
                                Operators...>
    {
    };

    template <typename... Operators>
    constexpr bool has_unique_operators(NTTDescription<Operators...>)
    {
        constexpr NTTOperator ops[] = {Operators::op..., NTTOperator::Size};
        for (int i = 0; i < int(sizeof...(Operators)); i++)
        {
            for (int j = i + 1; j < int(sizeof...(Operators)); j++)
            {
                if (ops[i] == ops[j])
                {
                    return false;
                }
            }
        }
        return true;
    }

    template <typename T> constexpr auto as_ntt_description(T)
    {
        if constexpr (is_ntt_description<T>::value)
        {
            return T{};
        }
        else
        {
            return NTTDescription<T>{};
        }
    }

    template <typename... Left, typename... Right>
    constexpr auto concat_ntt_descriptions(NTTDescription<Left...>,
                                           NTTDescription<Right...>)
    {
        return NTTDescription<Left..., Right...>{};
    }

    // Operators and descriptions add up to a description. The return type is
    // deduced, so the check below also runs inside decltype.
    template <
        typename L, typename R,
        std::enable_if_t<
            (is_ntt_operator<L>::value || is_ntt_description<L>::value) &&
                (is_ntt_operator<R>::value || is_ntt_description<R>::value),
            int> = 0>
    constexpr auto operator+(L, R)
    {
        constexpr auto result = concat_ntt_descriptions(
            as_ntt_description(L{}), as_ntt_description(R{}));
        static_assert(has_unique_operators(result),
                      "Each NTT operator may appear at most once");
        return result;
    }

} // namespace gpuntt

#endif // NTT_DESCRIPTOR_H
