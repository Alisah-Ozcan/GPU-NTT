// Copyright 2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef NTT_FIELDS_H
#define NTT_FIELDS_H

#include "gpuntt/common/modular_arith.cuh"

namespace gpuntt
{
    // A prime p and a root of unity modulo p of order 2^log_max_order, the
    // largest power of two that divides p - 1. An NTT of ring size N needs
    // log_max_order >= log2(N) + 1 for X^N + 1 and >= log2(N) for X^N - 1.
    template <typename T> struct NTTPrime
    {
        T value;
        T root;
        int log_max_order;
    };

    // The first prime is the one NTTParameters uses; the others are the
    // largest 30-bit primes with log_max_order >= 23.
    inline constexpr NTTPrime<Data32> ntt_primes32[] = {
        {469762049, 2187, 26},      {998244353, 15311432, 23},
        {897581057, 872686320, 23}, {880803841, 98722167, 23},
        {754974721, 739831874, 24}, {645922817, 224270701, 23},
        {595591169, 361399025, 23},
    };

    // The first prime is the one NTTParameters uses; the others are the
    // largest 61-bit primes with log_max_order >= 29, enough for every ring
    // size up to 2^28.
    inline constexpr NTTPrime<Data64> ntt_primes64[] = {
        {576460756061519873ULL, 478458933433709261ULL, 29},
        {2305843003308113921ULL, 1792440666383980664ULL, 29},
        {2305842995791921153ULL, 2062396343075984903ULL, 29},
        {2305842981296406529ULL, 791222564632478678ULL, 31},
        {2305842979148922881ULL, 1861299213724964929ULL, 32},
        {2305842974317084673ULL, 587132349836065757ULL, 29},
        {2305842973243342849ULL, 2300355246160886231ULL, 29},
        {2305842949084151809ULL, 1677524048036628098ULL, 33},
        {2305842937809862657ULL, 259088720336081454ULL, 29},
    };

    // A root of unity of order 2^log_order modulo prime.value.
    template <typename T>
    T root_of_unity(const NTTPrime<T>& prime, int log_order)
    {
        if (log_order < 0 || log_order > prime.log_max_order)
        {
            throw std::invalid_argument("Prime " + std::to_string(prime.value) +
                                        " has no root of unity of order 2^" +
                                        std::to_string(log_order) + ".");
        }
        return OPERATOR<T>::exp(prime.root,
                                T(1) << (prime.log_max_order - log_order),
                                Modulus<T>(prime.value));
    }

} // namespace gpuntt

#endif // NTT_FIELDS_H
