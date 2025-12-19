// Copyright 2024 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef NTT_CPU_H
#define NTT_CPU_H

#include "gpuntt/common/nttparameters.cuh"

namespace gpuntt
{
    template <typename T>
    std::vector<T>
    schoolbook_poly_multiplication(std::vector<T> a, std::vector<T> b,
                                   Modulus<T> modulus,
                                   ReductionPolynomial reduction_poly);

    template <typename T> class NTTCPU
    {
      public:
        NTTParameters<T> parameters;

        NTTCPU(NTTParameters<T> parameters_);

      public:
        std::vector<T> mult(std::vector<T>& input1, std::vector<T>& input2);

        std::vector<T> ntt(std::vector<T>& input);

        std::vector<T> intt(std::vector<T>& input);
    };

} // namespace gpuntt
#endif // NTT_CPU_H
