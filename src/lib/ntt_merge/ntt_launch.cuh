// Copyright 2023-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan
// Paper: https://eprint.iacr.org/2023/1410

#ifndef NTT_LAUNCH_H
#define NTT_LAUNCH_H

#include "gpuntt/ntt_merge/ntt.cuh"
#include "gpuntt/ntt_merge/ntt_device.cuh"

// Launchers of the host API's PerPolynomial transforms (see
// ntt_launch_impl.cuh). n_power must be in the range of the transform; the
// host API checks it before.

namespace gpuntt
{
    template <typename T, typename ModulusPolicy>
    void forward_per_polynomial(
        T* device_in, typename std::make_unsigned<T>::type* device_out,
        typename std::make_unsigned<T>::type* root_of_unity_table,
        ModulusPolicy modulus_policy, int n_power,
        ReductionPolynomial reduction_poly, int batch_size,
        cudaStream_t stream);

    template <typename T, typename ModulusPolicy>
    void inverse_per_polynomial(
        typename std::make_unsigned<T>::type* device_in, T* device_out,
        typename std::make_unsigned<T>::type* root_of_unity_table,
        ModulusPolicy modulus_policy, int n_power,
        ReductionPolynomial reduction_poly, int batch_size,
        cudaStream_t stream);

    template <typename T, typename ModulusPolicy, typename PolyPolicy>
    void ordered_stages(T* device_in, T* device_out, T* root_of_unity_table,
                        ModulusPolicy modulus_policy, PolyPolicy poly_policy,
                        type ntt_type, int n_power,
                        ReductionPolynomial reduction_poly, int batch_size,
                        cudaStream_t stream);

    template <typename T>
    using UnsignedOf = typename std::make_unsigned<T>::type;

} // namespace gpuntt

#endif // NTT_LAUNCH_H
