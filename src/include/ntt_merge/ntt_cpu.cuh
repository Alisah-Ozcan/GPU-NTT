// Copyright 2024 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef NTT_CPU_H
#define NTT_CPU_H

#include "nttparameters.cuh"

std::vector<Data>
schoolbook_poly_multiplication(std::vector<Data> a, std::vector<Data> b,
                               Modulus modulus,
                               ReductionPolynomial reduction_poly);

class NTT_CPU
{
  public:
    NTTParameters parameters;

    NTT_CPU(NTTParameters parameters_);

  public:
    std::vector<Data> mult(std::vector<Data>& input1,
                           std::vector<Data>& input2);

    std::vector<Data> ntt(std::vector<Data>& input);

    std::vector<Data> intt(std::vector<Data>& input);
};

#endif // NTT_CPU_H
