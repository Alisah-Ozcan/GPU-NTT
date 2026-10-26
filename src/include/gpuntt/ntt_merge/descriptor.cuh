// Copyright 2023-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan
// Paper: https://eprint.iacr.org/2023/1410

#ifndef NTT_DESCRIPTOR_H
#define NTT_DESCRIPTOR_H

#include "gpuntt/common/nttparameters.cuh"
#include <type_traits>

namespace gpuntt
{
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

    // Target architecture as 100 * major + 10 * minor, e.g. SM<890>.
    template <int Arch> struct SM : NTTOperatorBase<NTTOperator::SM>
    {
        static_assert(Arch >= 700 && Arch % 10 == 0,
                      "SM must be 100 * major + 10 * minor, at least 700");
        static constexpr int value = Arch;
    };

    template <typename... Operators> struct NTTDescription
    {
    };

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
