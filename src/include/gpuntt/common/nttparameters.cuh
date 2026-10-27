// Copyright 2023-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef NTT_PARAMETERS_H
#define NTT_PARAMETERS_H

#include <vector>
#include "gpuntt/common/common.cuh"
#include "gpuntt/common/modular_arith.cuh"
#include <type_traits>

namespace gpuntt
{

    int bitreverse(int index, int n_power);

    enum type
    {
        FORWARD,
        INVERSE
    };

    enum NTTLayout
    {
        PerPolynomial, // NTT per row (i.e., per polynomial)
        PerCoefficient // NTT per column (i.e., per coefficient index across
                       // batch)
    };

    enum ReductionPolynomial
    {
        X_N_plus,
        X_N_minus
    }; // X_N_minus: X^n - 1, X_N_plus: X^n + 1

    template <typename T> struct NTTFactors
    {
        Modulus<T> modulus;
        T omega;
        T psi;

        // Constructor to initialize the NTTFactors
        __host__ NTTFactors(Modulus<T> q_, T omega_, T psi_)
        {
            modulus = q_;
            omega = omega_;
            psi = psi_;
        }
        __host__ NTTFactors() {}

        // TODO: add check mechanism here
    };

    template <typename T> class NTTParameters
    {
      public:
        int logn;
        T n;

        ReductionPolynomial poly_reduction;

        Modulus<T> modulus;

        T omega;
        T psi;

        T n_inv;

        T root_of_unity;
        T inverse_root_of_unity;

        T root_of_unity_size;

        std::vector<T> forward_root_of_unity_table;
        std::vector<T> inverse_root_of_unity_table;

        // For testing all cases with barretti goldilock and plantard reduction
        NTTParameters(int LOGN, ReductionPolynomial poly_reduce_type);

        // For any prime(64-bit)
        NTTParameters(int LOGN, NTTFactors<T> ntt_factors,
                      ReductionPolynomial poly_reduce_type);

        NTTParameters(); // = delete;

      private:
        Modulus<T> modulus_pool();

        T omega_pool();

        T psi_pool();

        void forward_root_of_unity_table_generator();

        void inverse_root_of_unity_table_generator();

        void n_inverse_generator();

      public:
        std::vector<Root<T>>
        gpu_root_of_unity_table_generator(std::vector<T> table);
    };

} // namespace gpuntt
#endif // NTT_PARAMETERS_H
