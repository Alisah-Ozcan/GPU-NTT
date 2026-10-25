// Copyright 2023-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include "gpuntt/common/nttparameters.cuh"

namespace gpuntt
{

    int bitreverse(int index, int n_power)
    {
        int res_1 = 0;
        for (int i = 0; i < n_power; i++)
        {
            res_1 <<= 1;
            res_1 = (index & 1) | res_1;
            index >>= 1;
        }
        return res_1;
    }

    template <typename T>
    NTTParameters<T>::NTTParameters(int LOGN,
                                    ReductionPolynomial poly_reduce_type)
    {
        logn = LOGN;
        n = 1 << logn;

        poly_reduction = poly_reduce_type;

        modulus = modulus_pool();

        omega = omega_pool();
        psi = psi_pool();

        root_of_unity =
            (poly_reduce_type == ReductionPolynomial::X_N_minus) ? omega : psi;
        inverse_root_of_unity = OPERATOR<T>::modinv(root_of_unity, modulus);

        root_of_unity_size =
            (poly_reduce_type == ReductionPolynomial::X_N_minus)
                ? (1 << (logn - 1))
                : (1 << logn);

        forward_root_of_unity_table_generator();
        inverse_root_of_unity_table_generator();

        n_inverse_generator();
    }

    template <typename T>
    NTTParameters<T>::NTTParameters(int LOGN, NTTFactors<T> ntt_factors,
                                    ReductionPolynomial poly_reduce_type)
    {
        logn = LOGN;
        n = 1 << logn;

        poly_reduction = poly_reduce_type;

        modulus = ntt_factors.modulus;

        omega = ntt_factors.omega;
        psi = ntt_factors.psi;

        root_of_unity =
            (poly_reduce_type == ReductionPolynomial::X_N_minus) ? omega : psi;
        inverse_root_of_unity = OPERATOR<T>::modinv(root_of_unity, modulus);

        root_of_unity_size =
            (poly_reduce_type == ReductionPolynomial::X_N_minus)
                ? (1 << (logn - 1))
                : (1 << logn);

        forward_root_of_unity_table_generator();
        inverse_root_of_unity_table_generator();

        n_inverse_generator();
    }

    template <typename T> NTTParameters<T>::NTTParameters(){};

    template <typename T> Modulus<T> NTTParameters<T>::modulus_pool()
    {
        if constexpr (std::is_same<T, Data32>::value)
        {
            customAssert(0 < logn && logn <= 25,
                         "LOGN should be in range 2^0 to 2^25.");
            return Modulus32(469762049);
        }
        else
        {
            customAssert(0 < logn && logn <= 28,
                         "LOGN should be in range 2^0 to 2^28.");
            return Modulus64(576460756061519873ULL);
        }
    }

    template <typename T> T NTTParameters<T>::omega_pool()
    {
        if constexpr (std::is_same<T, Data32>::value)
        {
            customAssert(0 < logn && logn <= 25,
                         "LOGN should be in range 2^0 to 2^25.");
            Data32 w = 900;
            Modulus32 mod = modulus_pool();
            return OPERATOR32::exp(w, static_cast<Data32>(1 << (25 - logn)),
                                   mod);
        }
        else
        {
            customAssert(0 < logn && logn <= 28,
                         "LOGN should be in range 2^0 to 2^28.");
            Data64 w = 229929041166717729ULL;
            Modulus64 mod = modulus_pool();
            return OPERATOR64::exp(w, static_cast<Data64>(1 << (28 - logn)),
                                   mod);
        }
    }

    template <typename T> T NTTParameters<T>::psi_pool()
    {
        if constexpr (std::is_same<T, Data32>::value)
        {
            customAssert(0 < logn && logn <= 25,
                         "LOGN should be in range 2^0 to 2^25.");
            Data32 w = 30;
            Modulus32 mod = modulus_pool();
            return OPERATOR32::exp(w, static_cast<Data32>(1 << (25 - logn)),
                                   mod);
        }
        else
        {
            customAssert(0 < logn && logn <= 28,
                         "LOGN should be in range 2^0 to 2^28.");
            Data64 psi = 4517306222ULL;
            Modulus64 mod = modulus_pool();
            return OPERATOR64::exp(psi, static_cast<Data64>(1 << (28 - logn)),
                                   mod);
        }
    }

    template <typename T>
    void NTTParameters<T>::forward_root_of_unity_table_generator()
    {
        forward_root_of_unity_table.push_back(1);

        for (int i = 1; i < root_of_unity_size; i++)
        {
            T exp = OPERATOR<T>::mult(forward_root_of_unity_table[i - 1],
                                      root_of_unity, modulus);
            forward_root_of_unity_table.push_back(exp);
        }
    }

    template <typename T>
    void NTTParameters<T>::inverse_root_of_unity_table_generator()
    {
        inverse_root_of_unity_table.push_back(1);

        for (int i = 1; i < root_of_unity_size; i++)
        {
            T exp = OPERATOR<T>::mult(inverse_root_of_unity_table[i - 1],
                                      inverse_root_of_unity, modulus);
            inverse_root_of_unity_table.push_back(exp);
        }
    }

    template <typename T> void NTTParameters<T>::n_inverse_generator()
    {
        n_inv = OPERATOR<T>::modinv(n, modulus);
    }

    template <typename T>
    std::vector<Root<T>>
    NTTParameters<T>::gpu_root_of_unity_table_generator(std::vector<T> table)
    {
        // Taking Bitreverse order of root of unity table

        std::vector<T> new_table;
        int lg = log2(root_of_unity_size);
        for (int i = 0; i < root_of_unity_size; i++)
        {
            new_table.push_back(table[bitreverse(i, lg)]);
        }

        return new_table;
    }

    template class NTTParameters<Data32>;
    template class NTTParameters<Data64>;

} // namespace gpuntt
