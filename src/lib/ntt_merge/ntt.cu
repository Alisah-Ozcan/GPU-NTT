// Copyright 2022 Ulvetanna Inc.
// Copyright 2023-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan
// Paper: https://eprint.iacr.org/2023/1410

#include "gpuntt/ntt_merge/ntt.cuh"

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

    // Kernel parameters per log2(ring size). GPUNTT_CC_89_KERNEL_CONFIG
    // (CMake option, ON by default) selects the configuration tuned for
    // compute capability 8.9 for log2(ring size) 27 and 28.
    template <typename T> auto CreateForwardNTTKernel()
    {
        return std::unordered_map<int, std::vector<KernelConfig>>{
            {1, {{1, 1, 1, 256, 512 * sizeof(T), 0, 0, 0, 1, false}}},
            {2, {{1, 1, 2, 128, 512 * sizeof(T), 1, 0, 0, 2, false}}},
            {3, {{1, 1, 4, 64, 512 * sizeof(T), 2, 0, 0, 3, false}}},
            {4, {{1, 1, 8, 32, 512 * sizeof(T), 3, 0, 0, 4, false}}},
            {5, {{1, 1, 16, 16, 512 * sizeof(T), 4, 0, 0, 5, false}}},
            {6, {{1, 1, 32, 8, 512 * sizeof(T), 5, 0, 0, 6, false}}},
            {7, {{1, 1, 64, 4, 512 * sizeof(T), 6, 0, 0, 7, false}}},
            {8, {{1, 1, 128, 2, 512 * sizeof(T), 7, 0, 0, 8, false}}},
            {9, {{1, 1, 256, 1, 512 * sizeof(T), 8, 0, 0, 9, false}}},
            {10, {{1, 1, 512, 1, 1024 * sizeof(T), 9, 0, 0, 10, false}}},
            {11,
             {{4, 1, 128, 2, 512 * sizeof(T), 8, 0, 0, 2, true},
              {1, 4, 256, 1, 512 * sizeof(T), 8, 2, 0, 9, false}}},
            {12,
             {{8, 1, 64, 4, 512 * sizeof(T), 8, 0, 0, 3, true},
              {1, 8, 256, 1, 512 * sizeof(T), 8, 3, 0, 9, false}}},
            {13,
             {{16, 1, 32, 8, 512 * sizeof(T), 8, 0, 0, 4, true},
              {1, 16, 256, 1, 512 * sizeof(T), 8, 4, 0, 9, false}}},
            {14,
             {{32, 1, 16, 16, 512 * sizeof(T), 8, 0, 0, 5, true},
              {1, 32, 256, 1, 512 * sizeof(T), 8, 5, 0, 9, false}}},
            {15,
             {{64, 1, 8, 32, 512 * sizeof(T), 8, 0, 0, 6, true},
              {1, 64, 256, 1, 512 * sizeof(T), 8, 6, 0, 9, false}}},
            {16,
             {{128, 1, 4, 64, 512 * sizeof(T), 8, 0, 0, 7, true},
              {1, 128, 256, 1, 512 * sizeof(T), 8, 7, 0, 9, false}}},
            {17,
             {{256, 1, 32, 8, 512 * sizeof(T), 8, 0, 0, 4, true},
              {16, 16, 32, 8, 512 * sizeof(T), 8, 4, 0, 4, true},
              {1, 256, 256, 1, 512 * sizeof(T), 8, 8, 0, 9, false}}},
            {18,
             {{512, 1, 32, 8, 512 * sizeof(T), 8, 0, 0, 4, true},
              {32, 16, 16, 16, 512 * sizeof(T), 8, 4, 0, 5, true},
              {1, 512, 256, 1, 512 * sizeof(T), 8, 9, 0, 9, false}}},
            {19,
             {{1024, 1, 16, 16, 512 * sizeof(T), 8, 0, 0, 5, true},
              {32, 32, 16, 16, 512 * sizeof(T), 8, 5, 0, 5, true},
              {1, 1024, 256, 1, 512 * sizeof(T), 8, 10, 0, 9, false}}},
            {20,
             {{2048, 1, 16, 16, 512 * sizeof(T), 8, 0, 0, 5, true},
              {64, 32, 8, 32, 512 * sizeof(T), 8, 5, 0, 6, true},
              {1, 2048, 256, 1, 512 * sizeof(T), 8, 11, 0, 9, false}}},
            {21,
             {{4096, 1, 8, 32, 512 * sizeof(T), 8, 0, 0, 6, true},
              {64, 64, 8, 32, 512 * sizeof(T), 8, 6, 0, 6, true},
              {1, 4096, 256, 1, 512 * sizeof(T), 8, 12, 0, 9, false}}},
            {22,
             {{8192, 1, 8, 32, 512 * sizeof(T), 8, 0, 0, 6, true},
              {128, 64, 4, 64, 512 * sizeof(T), 8, 6, 0, 7, true},
              {1, 8192, 256, 1, 512 * sizeof(T), 8, 13, 0, 9, false}}},
            {23,
             {{16384, 1, 4, 64, 512 * sizeof(T), 8, 0, 0, 7, true},
              {128, 128, 4, 64, 512 * sizeof(T), 8, 7, 0, 7, true},
              {1, 16384, 256, 1, 512 * sizeof(T), 8, 14, 0, 9, false}}},
            {24,
             {{16384, 1, 8, 64, 1024 * sizeof(T), 9, 0, 0, 7, true},
              {128, 128, 8, 64, 1024 * sizeof(T), 9, 7, 0, 7, true},
              {1, 16384, 512, 1, 1024 * sizeof(T), 9, 14, 0, 10, false}}},
            {25,
             {{32768, 1, 8, 64, 1024 * sizeof(T), 9, 0, 0, 7, true},
              {256, 128, 4, 128, 1024 * sizeof(T), 9, 7, 0, 8, true},
              {32768, 1, 512, 1, 1024 * sizeof(T), 9, 15, 0, 10, false}}},
            {26,
             {{65536, 1, 4, 128, 1024 * sizeof(T), 9, 0, 0, 8, true},
              {256, 256, 4, 128, 1024 * sizeof(T), 9, 8, 0, 8, true},
              {65536, 1, 512, 1, 1024 * sizeof(T), 9, 16, 0, 10, false}}},
#ifndef GPUNTT_CC_89_KERNEL_CONFIG
            {27,
             {{262144, 1, 16, 16, 512 * sizeof(T), 8, 0, 0, 5, true},
              {8192, 32, 8, 32, 512 * sizeof(T), 8, 5, 0, 6, true},
              {128, 2048, 4, 64, 512 * sizeof(T), 8, 11, 0, 7, true},
              {262144, 1, 256, 1, 512 * sizeof(T), 8, 18, 0, 9, false}}},

            {28,
             {{524288, 1, 8, 32, 512 * sizeof(T), 8, 0, 0, 6, true},
              {8192, 64, 8, 32, 512 * sizeof(T), 8, 6, 0, 6, true},
              {128, 4096, 4, 64, 512 * sizeof(T), 8, 12, 0, 7,
               true}, {524288, 1, 256, 1, 512 * sizeof(T), 8, 19, 0, 9, false}}}
#else
            {27,
             {{131072, 1, 4, 128, 1024 * sizeof(T), 9, 0, 0, 8, true},
              {512, 256, 2, 256, 1024 * sizeof(T), 9, 8, 0, 9, true},
              {131072, 1, 512, 1, 1024 * sizeof(T), 9, 17, 0, 10, false}}},
            {28,
             {{262144, 1, 2, 256, 1024 * sizeof(T), 9, 0, 0, 9, true},
              {512, 512, 2, 256, 1024 * sizeof(T), 9, 9, 0, 9, true},
              {262144, 1, 512, 1, 1024 * sizeof(T), 9, 18, 0, 10, false}}}
#endif
        };
    }

    template <typename T> auto CreateInverseNTTKernel()
    {
        return std::unordered_map<int, std::vector<KernelConfig>>{
            {1, {{1, 1, 1, 256, 512 * sizeof(T), 0, 0, 0, 1, true}}},
            {2, {{1, 1, 2, 128, 512 * sizeof(T), 1, 1, 0, 2, true}}},
            {3, {{1, 1, 4, 64, 512 * sizeof(T), 2, 2, 0, 3, true}}},
            {4, {{1, 1, 8, 32, 512 * sizeof(T), 3, 3, 0, 4, true}}},
            {5, {{1, 1, 16, 16, 512 * sizeof(T), 4, 4, 0, 5, true}}},
            {6, {{1, 1, 32, 8, 512 * sizeof(T), 5, 5, 0, 6, true}}},
            {7, {{1, 1, 64, 4, 512 * sizeof(T), 6, 6, 0, 7, true}}},
            {8, {{1, 1, 128, 2, 512 * sizeof(T), 7, 7, 0, 8, true}}},
            {9, {{1, 1, 256, 1, 512 * sizeof(T), 8, 8, 0, 9, true}}},
            {10, {{1, 1, 512, 1, 1024 * sizeof(T), 9, 9, 0, 10, true}}},
            {11,
             {{1, 4, 256, 1, 512 * sizeof(T), 8, 10, 2, 9, false},
              {4, 1, 128, 2, 512 * sizeof(T), 8, 1, 0, 2, true}}},
            {12,
             {{1, 8, 256, 1, 512 * sizeof(T), 8, 11, 3, 9, false},
              {8, 1, 64, 4, 512 * sizeof(T), 8, 2, 0, 3, true}}},
            {13,
             {{1, 16, 256, 1, 512 * sizeof(T), 8, 12, 4, 9, false},
              {16, 1, 32, 8, 512 * sizeof(T), 8, 3, 0, 4, true}}},
            {14,
             {{1, 32, 256, 1, 512 * sizeof(T), 8, 13, 5, 9, false},
              {32, 1, 16, 16, 512 * sizeof(T), 8, 4, 0, 5, true}}},
            {15,
             {{1, 64, 256, 1, 512 * sizeof(T), 8, 14, 6, 9, false},
              {64, 1, 8, 32, 512 * sizeof(T), 8, 5, 0, 6, true}}},
            {16,
             {{1, 128, 256, 1, 512 * sizeof(T), 8, 15, 7, 9, false},
              {128, 1, 4, 64, 512 * sizeof(T), 8, 6, 0, 7, true}}},
            {17,
             {{1, 256, 256, 1, 512 * sizeof(T), 8, 16, 8, 9, false},
              {16, 16, 32, 8, 512 * sizeof(T), 8, 7, 4, 4, false},
              {256, 1, 32, 8, 512 * sizeof(T), 8, 3, 0, 4, true}}},
            {18,
             {{1, 512, 256, 1, 512 * sizeof(T), 8, 17, 9, 9, false},
              {32, 16, 16, 16, 512 * sizeof(T), 8, 8, 4, 5, false},
              {512, 1, 32, 8, 512 * sizeof(T), 8, 3, 0, 4, true}}},
            {19,
             {{1, 1024, 256, 1, 512 * sizeof(T), 8, 18, 10, 9, false},
              {32, 32, 16, 16, 512 * sizeof(T), 8, 9, 5, 5, false},
              {1024, 1, 16, 16, 512 * sizeof(T), 8, 4, 0, 5, true}}},
            {20,
             {{1, 2048, 256, 1, 512 * sizeof(T), 8, 19, 11, 9, false},
              {64, 32, 8, 32, 512 * sizeof(T), 8, 10, 5, 6, false},
              {2048, 1, 16, 16, 512 * sizeof(T), 8, 4, 0, 5, true}}},
            {21,
             {{1, 4096, 256, 1, 512 * sizeof(T), 8, 20, 12, 9, false},
              {64, 64, 8, 32, 512 * sizeof(T), 8, 11, 6, 6, false},
              {4096, 1, 8, 32, 512 * sizeof(T), 8, 5, 0, 6, true}}},
            {22,
             {{1, 8192, 256, 1, 512 * sizeof(T), 8, 21, 13, 9, false},
              {128, 64, 4, 64, 512 * sizeof(T), 8, 12, 6, 7, false},
              {8192, 1, 8, 32, 512 * sizeof(T), 8, 5, 0, 6, true}}},
            {23,
             {{1, 16384, 256, 1, 512 * sizeof(T), 8, 22, 14, 9, false},
              {128, 128, 4, 64, 512 * sizeof(T), 8, 13, 7, 7, false},
              {16384, 1, 4, 64, 512 * sizeof(T), 8, 6, 0, 7, true}}},
            {24,
             {{1, 16384, 512, 1, 1024 * sizeof(T), 9, 23, 14, 10, false},
              {128, 128, 8, 64, 1024 * sizeof(T), 9, 13, 7, 7, false},
              {16384, 1, 8, 64, 1024 * sizeof(T), 9, 6, 0, 7, true}}},
            {25,
             {{32768, 1, 512, 1, 1024 * sizeof(T), 9, 24, 15, 10, false},
              {256, 128, 4, 128, 1024 * sizeof(T), 9, 14, 7, 8, false},
              {32768, 1, 8, 64, 1024 * sizeof(T), 9, 6, 0, 7, true}}},
            {26,
             {{65536, 1, 512, 1, 1024 * sizeof(T), 9, 25, 16, 10, false},
              {256, 256, 4, 128, 1024 * sizeof(T), 9, 15, 8, 8, false},
              {65536, 1, 4, 128, 1024 * sizeof(T), 9, 7, 0, 8, true}}},
#ifndef GPUNTT_CC_89_KERNEL_CONFIG
            {27,
             {{262144, 1, 256, 1, 512 * sizeof(T), 8, 26, 18, 9, false},
              {128, 2048, 4, 64, 512 * sizeof(T), 8, 17, 11, 7, false},
              {8192, 32, 8, 32, 512 * sizeof(T), 8, 10, 5, 6, false},
              {262144, 1, 16, 16, 512 * sizeof(T), 8, 4, 0, 5, true}}},
            {28,
             {
                 {524288, 1, 256, 1, 512 * sizeof(T), 8, 27, 19, 9, false},
                 {128, 4096, 4, 64, 512 * sizeof(T), 8, 18, 12, 7, false},
                 {8192, 64, 8, 32, 512 * sizeof(T), 8, 11, 6, 6, false},
                 {524288, 1, 8, 32, 512 * sizeof(T), 8, 5, 0, 6, true},
             }}
#else
            {27,
             {{131072, 1, 512, 1, 1024 * sizeof(T), 9, 26, 17, 10, false},
              {512, 256, 2, 256, 1024 * sizeof(T), 9, 16, 8, 9, false},
              {131072, 1, 4, 128, 1024 * sizeof(T), 9, 7, 0, 8, true}}},
            {28,
             {{262144, 1, 512, 1, 1024 * sizeof(T), 9, 27, 18, 10, false},
              {512, 512, 2, 256, 1024 * sizeof(T), 9, 17, 9, 9, false},
              {262144, 1, 2, 256, 1024 * sizeof(T), 9, 8, 0, 9, true}}}
#endif
        };
    }

    // Kernels compute global memory offsets with 32-bit integers, so a call
    // can transform at most 2^31 coefficients in total.
    static void check_batch_size(int n_power, int batch_size)
    {
        if (batch_size <= 0)
        {
            throw std::invalid_argument("Invalid batch_size!");
        }
        if (n_power > 0 && n_power < 31 &&
            (static_cast<std::int64_t>(batch_size) << n_power) >
                (static_cast<std::int64_t>(1) << 31))
        {
            throw std::invalid_argument(
                "batch_size * ring size must be at most 2^31!");
        }
    }

    // Policies selecting, at compile time, which modulus a polynomial uses
    // and which polynomial a block transforms. They let the single modulus,
    // RNS and ordered variants of a kernel share one body.

    // Every polynomial uses the same modulus.
    template <typename TU> struct SingleModulus
    {
        Modulus<TU> modulus;
        Ninverse<TU> n_inverse;

        __device__ __forceinline__ int prime_index(int) const { return 0; }

        __device__ __forceinline__ Modulus<TU> get_modulus(int) const
        {
            return modulus;
        }

        __device__ __forceinline__ Ninverse<TU> get_n_inverse(int) const
        {
            return n_inverse;
        }
    };

    // Polynomial z uses modulus z % mod_count.
    template <typename TU> struct RnsModulus
    {
        const Modulus<TU>* modulus;
        int mod_count;
        const Ninverse<TU>* n_inverse;

        __device__ __forceinline__ int prime_index(int poly) const
        {
            return poly % mod_count;
        }

        __device__ __forceinline__ Modulus<TU> get_modulus(int prime) const
        {
            return modulus[prime];
        }

        __device__ __forceinline__ Ninverse<TU> get_n_inverse(int prime) const
        {
            return n_inverse[prime];
        }
    };

    // Polynomial z uses modulus order[z % mod_count].
    template <typename TU> struct OrderedRnsModulus
    {
        const Modulus<TU>* modulus;
        int mod_count;
        const int* order;
        const Ninverse<TU>* n_inverse;

        __device__ __forceinline__ int prime_index(int poly) const
        {
            return order[poly % mod_count];
        }

        __device__ __forceinline__ Modulus<TU> get_modulus(int prime) const
        {
            return modulus[prime];
        }

        __device__ __forceinline__ Ninverse<TU> get_n_inverse(int prime) const
        {
            return n_inverse[prime];
        }
    };

    // Block z transforms polynomial z.
    struct SequentialPolys
    {
        __device__ __forceinline__ int poly_index(int block_z) const
        {
            return block_z;
        }
    };

    // Block z transforms polynomial order[z].
    struct OrderedPolys
    {
        const int* order;

        __device__ __forceinline__ int poly_index(int block_z) const
        {
            return order[block_z];
        }
    };

    // The low ring size kernels transform several small polynomials per
    // block; the single modulus and RNS variants share one body.

    template <typename T, typename ModulusPolicy>
    __device__ __forceinline__ void ForwardCoreLowRingBody(
        T* polynomial_in, typename std::make_unsigned<T>::type* polynomial_out,
        const Root<typename std::make_unsigned<T>::type>* root_of_unity_table,
        ModulusPolicy modulus_policy, int shared_index, int N_power,
        bool reduction_poly_check, int total_batch)
    {
        using TU = typename std::make_unsigned<T>::type;

        const int idx_x = threadIdx.x;
        const int idx_y = threadIdx.y;
        const int block_x = blockIdx.x;
        const int batch_index = (block_x * blockDim.y) + idx_y;
        const bool active_batch = (batch_index < total_batch);
        const int batch_index_safe = active_batch ? batch_index : 0;

        const int prime_index = modulus_policy.prime_index(batch_index_safe);

        extern __shared__ char shared_memory_typed[];
        TU* shared_memory = reinterpret_cast<TU*>(shared_memory_typed);

        const Modulus<TU> modulus_reg = modulus_policy.get_modulus(prime_index);

        int t_2 = N_power - 1;
        int offset = idx_y << N_power;
        int t_ = shared_index;
        int m = 1;

        const int half_n = 1 << (N_power - 1);
        const int row_base = idx_y << N_power;
        const int shared_address0 = row_base + idx_x;
        const int shared_address1 = shared_address0 + half_n;
        location_t global_addresss =
            idx_x + (location_t) (batch_index_safe << N_power);

        location_t omega_addresss = idx_x;

        // Load T from global & store to shared
        if constexpr (std::is_signed<T>::value)
        {
            T input1_reg = polynomial_in[global_addresss];
            T input2_reg = polynomial_in[global_addresss + half_n];
            shared_memory[shared_address0] =
                OPERATOR_GPU<TU>::reduce(input1_reg, modulus_reg);
            shared_memory[shared_address1] =
                OPERATOR_GPU<TU>::reduce(input2_reg, modulus_reg);
        }
        else
        {
            shared_memory[shared_address0] = polynomial_in[global_addresss];
            shared_memory[shared_address1] =
                polynomial_in[global_addresss + half_n];
        }

        int shared_addresss = idx_x;

        int t = 1 << t_;
        int in_shared_address =
            ((shared_addresss >> t_) << t_) + shared_addresss;
        location_t current_root_index;
        __syncthreads();

#pragma unroll
        for (int lp = 0; lp < (shared_index + 1); lp++)
        {
            int group_in_shared_address = in_shared_address + offset;
            if (reduction_poly_check)
            { // X_N_minus
                current_root_index = (omega_addresss >> t_2) +
                                     (location_t) (prime_index << N_power);
            }
            else
            { // X_N_plus
                current_root_index = m + (omega_addresss >> t_2) +
                                     (location_t) (prime_index << N_power);
            }

            CooleyTukeyUnit(shared_memory[group_in_shared_address],
                            shared_memory[group_in_shared_address + t],
                            root_of_unity_table[current_root_index],
                            modulus_reg);

            t = t >> 1;
            t_2 -= 1;
            t_ -= 1;
            m <<= 1;
            if (lp + 1 < (shared_index + 1))
            {
                in_shared_address =
                    ((shared_addresss >> t_) << t_) + shared_addresss;
            }
            __syncthreads();
        }

        __syncthreads();

        if (active_batch)
        {
            polynomial_out[global_addresss] = shared_memory[shared_address0];
            polynomial_out[global_addresss + half_n] =
                shared_memory[shared_address1];
        }
    }

    template <typename T, typename ModulusPolicy>
    __device__ __forceinline__ void InverseCoreLowRingBody(
        typename std::make_unsigned<T>::type* polynomial_in, T* polynomial_out,
        const Root<typename std::make_unsigned<T>::type>*
            inverse_root_of_unity_table,
        ModulusPolicy modulus_policy, int shared_index, int N_power,
        bool reduction_poly_check, int total_batch)
    {
        using TU = typename std::make_unsigned<T>::type;

        const int idx_x = threadIdx.x;
        const int idx_y = threadIdx.y;
        const int block_x = blockIdx.x;
        const int batch_index = (block_x * blockDim.y) + idx_y;
        const bool active_batch = (batch_index < total_batch);
        const int batch_index_safe = active_batch ? batch_index : 0;

        const int prime_index = modulus_policy.prime_index(batch_index_safe);

        extern __shared__ char shared_memory_typed[];
        TU* shared_memory = reinterpret_cast<TU*>(shared_memory_typed);

        const Modulus<TU> modulus_reg = modulus_policy.get_modulus(prime_index);
        const Ninverse<TU> n_inverse_reg = modulus_policy.get_n_inverse(prime_index);

        int t_2 = 0;
        int t_ = 0;
        int offset = idx_y << N_power;
        int loops = N_power;
        int m = (int) 1 << (N_power - 1);

        const int half_n = 1 << (N_power - 1);
        const int row_base = idx_y << N_power;
        const int shared_address0 = row_base + idx_x;
        const int shared_address1 = shared_address0 + half_n;
        location_t global_addresss =
            idx_x + (location_t) (batch_index_safe << N_power);

        location_t omega_addresss = idx_x;

        // Load T from global & store to shared
        shared_memory[shared_address0] = polynomial_in[global_addresss];
        shared_memory[shared_address1] = polynomial_in[global_addresss + half_n];

        int shared_addresss = idx_x;

        int t = 1 << t_;
        int in_shared_address =
            ((shared_addresss >> t_) << t_) + shared_addresss;
        location_t current_root_index;
        __syncthreads();

#pragma unroll
        for (int lp = 0; lp < loops; lp++)
        {
            int group_in_shared_address = in_shared_address + offset;
            if (reduction_poly_check)
            { // X_N_minus
                current_root_index = (omega_addresss >> t_2) +
                                     (location_t) (prime_index << N_power);
            }
            else
            { // X_N_plus
                current_root_index = m + (omega_addresss >> t_2) +
                                     (location_t) (prime_index << N_power);
            }

            GentlemanSandeUnit(shared_memory[group_in_shared_address],
                               shared_memory[group_in_shared_address + t],
                               inverse_root_of_unity_table[current_root_index],
                               modulus_reg);

            t = t << 1;
            t_2 += 1;
            t_ += 1;
            m >>= 1;
            if (lp + 1 < loops)
            {
                in_shared_address =
                    ((shared_addresss >> t_) << t_) + shared_addresss;
            }
            __syncthreads();
        }
        __syncthreads();

        TU output1_reg = OPERATOR_GPU<TU>::mult(shared_memory[shared_address0],
                                                n_inverse_reg, modulus_reg);
        TU output2_reg =
            OPERATOR_GPU<TU>::mult(shared_memory[shared_address1],
                                   n_inverse_reg, modulus_reg);

        if (active_batch)
        {
            if constexpr (std::is_signed<T>::value)
            {
                polynomial_out[global_addresss] =
                    OPERATOR_GPU<TU>::centered_reduction(output1_reg, modulus_reg);
                polynomial_out[global_addresss + half_n] =
                    OPERATOR_GPU<TU>::centered_reduction(output2_reg, modulus_reg);
            }
            else
            {
                polynomial_out[global_addresss] = output1_reg;
                polynomial_out[global_addresss + half_n] = output2_reg;
            }
        }
    }

    template <typename T>
    __global__ void
    ForwardCoreLowRing(T* polynomial_in,
                       typename std::make_unsigned<T>::type* polynomial_out,
                       const Root<typename std::make_unsigned<
                           T>::type>* __restrict__ root_of_unity_table,
                       Modulus<typename std::make_unsigned<T>::type> modulus,
                       int shared_index, int N_power,
                       bool reduction_poly_check, int total_batch)
    {
        using TU = typename std::make_unsigned<T>::type;

        ForwardCoreLowRingBody(
            polynomial_in, polynomial_out, root_of_unity_table,
            SingleModulus<TU>{modulus}, shared_index, N_power,
            reduction_poly_check, total_batch);
    }

    template <typename T>
    __global__ void ForwardCoreLowRing(
        T* polynomial_in, typename std::make_unsigned<T>::type* polynomial_out,
        const Root<typename std::make_unsigned<
            T>::type>* __restrict__ root_of_unity_table,
        Modulus<typename std::make_unsigned<T>::type>* modulus,
        int shared_index, int N_power,
        bool reduction_poly_check, int total_batch, int mod_count)
    {
        using TU = typename std::make_unsigned<T>::type;

        ForwardCoreLowRingBody(
            polynomial_in, polynomial_out, root_of_unity_table,
            RnsModulus<TU>{modulus, mod_count}, shared_index, N_power,
            reduction_poly_check, total_batch);
    }

    template <typename T>
    __global__ void InverseCoreLowRing(
        typename std::make_unsigned<T>::type* polynomial_in, T* polynomial_out,
        const Root<typename std::make_unsigned<
            T>::type>* __restrict__ inverse_root_of_unity_table,
        Modulus<typename std::make_unsigned<T>::type> modulus, int shared_index,
        int N_power, Ninverse<typename std::make_unsigned<T>::type> n_inverse,
        bool reduction_poly_check, int total_batch)
    {
        using TU = typename std::make_unsigned<T>::type;

        InverseCoreLowRingBody(
            polynomial_in, polynomial_out, inverse_root_of_unity_table,
            SingleModulus<TU>{modulus, n_inverse}, shared_index, N_power,
            reduction_poly_check, total_batch);
    }

    template <typename T>
    __global__ void InverseCoreLowRing(
        typename std::make_unsigned<T>::type* polynomial_in, T* polynomial_out,
        const Root<typename std::make_unsigned<
            T>::type>* __restrict__ inverse_root_of_unity_table,
        Modulus<typename std::make_unsigned<T>::type>* modulus,
        int shared_index, int N_power,
        Ninverse<typename std::make_unsigned<T>::type>* n_inverse,
        bool reduction_poly_check, int total_batch, int mod_count)
    {
        using TU = typename std::make_unsigned<T>::type;

        InverseCoreLowRingBody(
            polynomial_in, polynomial_out, inverse_root_of_unity_table,
            RnsModulus<TU>{modulus, mod_count, n_inverse},
            shared_index, N_power,
            reduction_poly_check, total_batch);
    }

    // The multi-stage NTT kernels (ForwardCore, InverseCore, their _ variants
    // and the modulus ordered and poly ordered variants) share one body. They
    // differ only in the modulus and polynomial policies and in
    // SwapGridAxes: the _ variants swap the roles of blockIdx.x and
    // blockIdx.y for large grids.

    template <bool SwapGridAxes, typename T, typename ModulusPolicy,
              typename PolyPolicy>
    __device__ __forceinline__ void ForwardCoreBody(
        T* polynomial_in, typename std::make_unsigned<T>::type* polynomial_out,
        const Root<typename std::make_unsigned<T>::type>* root_of_unity_table,
        ModulusPolicy modulus_policy, PolyPolicy poly_policy, int shared_index,
        int logm, int outer_iteration_count, int N_power,
        bool not_last_kernel, bool reduction_poly_check)
    {
        using TU = typename std::make_unsigned<T>::type;

        const int idx_x = threadIdx.x;
        const int idx_y = threadIdx.y;
        const int block_x = SwapGridAxes ? blockIdx.y : blockIdx.x;
        const int block_y = SwapGridAxes ? blockIdx.x : blockIdx.y;
        const int block_z = blockIdx.z;

        const int prime_index = modulus_policy.prime_index(block_z);
        const int poly_index = poly_policy.poly_index(block_z);

        extern __shared__ char shared_memory_typed[];
        TU* shared_memory = reinterpret_cast<TU*>(shared_memory_typed);

        const Modulus<TU> modulus_reg = modulus_policy.get_modulus(prime_index);

        int t_2 = N_power - logm - 1;
        location_t offset = 1 << (N_power - logm - 1);
        int t_ = shared_index;
        location_t m = (location_t) 1 << logm;

        location_t global_addresss =
            idx_x +
            (location_t) (idx_y *
                          (offset / (1 << (outer_iteration_count - 1)))) +
            (location_t) (blockDim.x * block_x) +
            (location_t) (2 * block_y * offset) +
            (location_t) (poly_index << N_power);

        location_t omega_addresss =
            idx_x +
            (location_t) (idx_y *
                          (offset / (1 << (outer_iteration_count - 1)))) +
            (location_t) (blockDim.x * block_x) +
            (location_t) (block_y * offset);

        location_t shared_addresss = (idx_x + (idx_y * blockDim.x));

        // Load T from global & store to shared
        if constexpr (std::is_signed<T>::value)
        {
            T input1_reg = polynomial_in[global_addresss];
            T input2_reg = polynomial_in[global_addresss + offset];
            shared_memory[shared_addresss] =
                OPERATOR_GPU<TU>::reduce(input1_reg, modulus_reg);
            shared_memory[shared_addresss + (blockDim.x * blockDim.y)] =
                OPERATOR_GPU<TU>::reduce(input2_reg, modulus_reg);
        }
        else
        {
            shared_memory[shared_addresss] = polynomial_in[global_addresss];
            shared_memory[shared_addresss + (blockDim.x * blockDim.y)] =
                polynomial_in[global_addresss + offset];
        }

        int t = 1 << t_;
        int in_shared_address =
            ((shared_addresss >> t_) << t_) + shared_addresss;
        location_t current_root_index;
        if (not_last_kernel)
        {
#pragma unroll
            for (int lp = 0; lp < outer_iteration_count; lp++)
            {
                __syncthreads();
                if (reduction_poly_check)
                { // X_N_minus
                    current_root_index = (omega_addresss >> t_2) +
                                         (location_t) (prime_index << N_power);
                }
                else
                { // X_N_plus
                    current_root_index = m + (omega_addresss >> t_2) +
                                         (location_t) (prime_index << N_power);
                }
                CooleyTukeyUnit(shared_memory[in_shared_address],
                                shared_memory[in_shared_address + t],
                                root_of_unity_table[current_root_index],
                                modulus_reg);

                t = t >> 1;
                t_2 -= 1;
                t_ -= 1;
                m <<= 1;

                in_shared_address =
                    ((shared_addresss >> t_) << t_) + shared_addresss;
            }
            __syncthreads();
        }
        else
        {
#pragma unroll
            for (int lp = 0; lp < (shared_index - 5); lp++) // 4 for 512 thread
            {
                __syncthreads();
                if (reduction_poly_check)
                { // X_N_minus
                    current_root_index = (omega_addresss >> t_2) +
                                         (location_t) (prime_index << N_power);
                }
                else
                { // X_N_plus
                    current_root_index = m + (omega_addresss >> t_2) +
                                         (location_t) (prime_index << N_power);
                }

                CooleyTukeyUnit(shared_memory[in_shared_address],
                                shared_memory[in_shared_address + t],
                                root_of_unity_table[current_root_index],
                                modulus_reg);

                t = t >> 1;
                t_2 -= 1;
                t_ -= 1;
                m <<= 1;

                in_shared_address =
                    ((shared_addresss >> t_) << t_) + shared_addresss;
            }
            __syncthreads();

#pragma unroll
            for (int lp = 0; lp < 6; lp++)
            {
                if (reduction_poly_check)
                { // X_N_minus
                    current_root_index = (omega_addresss >> t_2) +
                                         (location_t) (prime_index << N_power);
                }
                else
                { // X_N_plus
                    current_root_index = m + (omega_addresss >> t_2) +
                                         (location_t) (prime_index << N_power);
                }
                CooleyTukeyUnit(shared_memory[in_shared_address],
                                shared_memory[in_shared_address + t],
                                root_of_unity_table[current_root_index],
                                modulus_reg);

                t = t >> 1;
                t_2 -= 1;
                t_ -= 1;
                m <<= 1;

                in_shared_address =
                    ((shared_addresss >> t_) << t_) + shared_addresss;
            }
            __syncthreads();
        }

        polynomial_out[global_addresss] = shared_memory[shared_addresss];
        polynomial_out[global_addresss + offset] =
            shared_memory[shared_addresss + (blockDim.x * blockDim.y)];
    }

    template <bool SwapGridAxes, typename T, typename ModulusPolicy,
              typename PolyPolicy>
    __device__ __forceinline__ void InverseCoreBody(
        typename std::make_unsigned<T>::type* polynomial_in, T* polynomial_out,
        const Root<typename std::make_unsigned<T>::type>*
            inverse_root_of_unity_table,
        ModulusPolicy modulus_policy, PolyPolicy poly_policy, int shared_index,
        int logm, int k, int outer_iteration_count, int N_power,
        bool last_kernel, bool reduction_poly_check)
    {
        using TU = typename std::make_unsigned<T>::type;

        const int idx_x = threadIdx.x;
        const int idx_y = threadIdx.y;
        const int block_x = SwapGridAxes ? blockIdx.y : blockIdx.x;
        const int block_y = SwapGridAxes ? blockIdx.x : blockIdx.y;
        const int block_z = blockIdx.z;

        const int prime_index = modulus_policy.prime_index(block_z);
        const int poly_index = poly_policy.poly_index(block_z);

        extern __shared__ char shared_memory_typed[];
        TU* shared_memory = reinterpret_cast<TU*>(shared_memory_typed);

        const Modulus<TU> modulus_reg = modulus_policy.get_modulus(prime_index);
        const Ninverse<TU> n_inverse_reg =
            modulus_policy.get_n_inverse(prime_index);

        int t_2 = N_power - logm - 1;
        location_t offset = 1 << (N_power - k - 1);
        int t_ = (shared_index + 1) - outer_iteration_count;
        int loops = outer_iteration_count;
        location_t m = (location_t) 1 << logm;

        location_t global_addresss =
            idx_x +
            (location_t) (idx_y *
                          (offset / (1 << (outer_iteration_count - 1)))) +
            (location_t) (blockDim.x * block_x) +
            (location_t) (2 * block_y * offset) +
            (location_t) (poly_index << N_power);

        location_t omega_addresss =
            idx_x +
            (location_t) (idx_y *
                          (offset / (1 << (outer_iteration_count - 1)))) +
            (location_t) (blockDim.x * block_x) +
            (location_t) (block_y * offset);
        location_t shared_addresss = (idx_x + (idx_y * blockDim.x));

        shared_memory[shared_addresss] = polynomial_in[global_addresss];
        shared_memory[shared_addresss + (blockDim.x * blockDim.y)] =
            polynomial_in[global_addresss + offset];

        int t = 1 << t_;
        int in_shared_address =
            ((shared_addresss >> t_) << t_) + shared_addresss;
        location_t current_root_index;
#pragma unroll
        for (int lp = 0; lp < loops; lp++)
        {
            __syncthreads();
            if (reduction_poly_check)
            { // X_N_minus
                current_root_index = (omega_addresss >> t_2) +
                                     (location_t) (prime_index << N_power);
            }
            else
            { // X_N_plus
                current_root_index = m + (omega_addresss >> t_2) +
                                     (location_t) (prime_index << N_power);
            }

            GentlemanSandeUnit(shared_memory[in_shared_address],
                               shared_memory[in_shared_address + t],
                               inverse_root_of_unity_table[current_root_index],
                               modulus_reg);

            t = t << 1;
            t_2 += 1;
            t_ += 1;
            m >>= 1;

            in_shared_address =
                ((shared_addresss >> t_) << t_) + shared_addresss;
        }
        __syncthreads();

        if (last_kernel)
        {
            TU output1_reg = OPERATOR_GPU<TU>::mult(
                shared_memory[shared_addresss], n_inverse_reg, modulus_reg);
            TU output2_reg = OPERATOR_GPU<TU>::mult(
                shared_memory[shared_addresss + (blockDim.x * blockDim.y)],
                n_inverse_reg, modulus_reg);

            if constexpr (std::is_signed<T>::value)
            {
                polynomial_out[global_addresss] =
                    OPERATOR_GPU<TU>::centered_reduction(output1_reg,
                                                         modulus_reg);
                polynomial_out[global_addresss + offset] =
                    OPERATOR_GPU<TU>::centered_reduction(output2_reg,
                                                         modulus_reg);
            }
            else
            {
                polynomial_out[global_addresss] = output1_reg;
                polynomial_out[global_addresss + offset] = output2_reg;
            }
        }
        else
        {
            polynomial_out[global_addresss] = shared_memory[shared_addresss];
            polynomial_out[global_addresss + offset] =
                shared_memory[shared_addresss + (blockDim.x * blockDim.y)];
        }
    }

    template <typename T>
    __global__ void ForwardCore(
        T* polynomial_in, typename std::make_unsigned<T>::type* polynomial_out,
        const Root<typename std::make_unsigned<
            T>::type>* __restrict__ root_of_unity_table,
        Modulus<typename std::make_unsigned<T>::type> modulus, int shared_index,
        int logm, int outer_iteration_count, int N_power,
        bool not_last_kernel, bool reduction_poly_check)
    {
        using TU = typename std::make_unsigned<T>::type;

        ForwardCoreBody<false>(
            polynomial_in, polynomial_out, root_of_unity_table,
            SingleModulus<TU>{modulus}, SequentialPolys{},
            shared_index, logm, outer_iteration_count, N_power,
            not_last_kernel, reduction_poly_check);
    }

    template <typename T>
    __global__ void
    ForwardCore(T* polynomial_in,
                typename std::make_unsigned<T>::type* polynomial_out,
                const Root<typename std::make_unsigned<
                    T>::type>* __restrict__ root_of_unity_table,
                Modulus<typename std::make_unsigned<T>::type>* modulus,
                int shared_index, int logm, int outer_iteration_count,
                int N_power, bool not_last_kernel,
                bool reduction_poly_check, int mod_count)
    {
        using TU = typename std::make_unsigned<T>::type;

        ForwardCoreBody<false>(
            polynomial_in, polynomial_out, root_of_unity_table,
            RnsModulus<TU>{modulus, mod_count}, SequentialPolys{},
            shared_index, logm, outer_iteration_count, N_power,
            not_last_kernel, reduction_poly_check);
    }

    template <typename T>
    __global__ void ForwardCore_(
        T* polynomial_in, typename std::make_unsigned<T>::type* polynomial_out,
        const Root<typename std::make_unsigned<
            T>::type>* __restrict__ root_of_unity_table,
        Modulus<typename std::make_unsigned<T>::type> modulus, int shared_index,
        int logm, int outer_iteration_count, int N_power,
        bool not_last_kernel, bool reduction_poly_check)
    {
        using TU = typename std::make_unsigned<T>::type;

        ForwardCoreBody<true>(
            polynomial_in, polynomial_out, root_of_unity_table,
            SingleModulus<TU>{modulus}, SequentialPolys{},
            shared_index, logm, outer_iteration_count, N_power,
            not_last_kernel, reduction_poly_check);
    }

    template <typename T>
    __global__ void
    ForwardCore_(T* polynomial_in,
                 typename std::make_unsigned<T>::type* polynomial_out,
                 const Root<typename std::make_unsigned<
                     T>::type>* __restrict__ root_of_unity_table,
                 Modulus<typename std::make_unsigned<T>::type>* modulus,
                 int shared_index, int logm, int outer_iteration_count,
                 int N_power, bool not_last_kernel,
                 bool reduction_poly_check, int mod_count)
    {
        using TU = typename std::make_unsigned<T>::type;

        ForwardCoreBody<true>(
            polynomial_in, polynomial_out, root_of_unity_table,
            RnsModulus<TU>{modulus, mod_count}, SequentialPolys{},
            shared_index, logm, outer_iteration_count, N_power,
            not_last_kernel, reduction_poly_check);
    }

    template <typename T>
    __global__ void InverseCore(
        typename std::make_unsigned<T>::type* polynomial_in, T* polynomial_out,
        const Root<typename std::make_unsigned<
            T>::type>* __restrict__ inverse_root_of_unity_table,
        Modulus<typename std::make_unsigned<T>::type> modulus, int shared_index,
        int logm, int k, int outer_iteration_count, int N_power,
        Ninverse<typename std::make_unsigned<T>::type> n_inverse,
        bool last_kernel, bool reduction_poly_check)
    {
        using TU = typename std::make_unsigned<T>::type;

        InverseCoreBody<false>(
            polynomial_in, polynomial_out, inverse_root_of_unity_table,
            SingleModulus<TU>{modulus, n_inverse}, SequentialPolys{},
            shared_index, logm, k, outer_iteration_count, N_power,
            last_kernel, reduction_poly_check);
    }

    template <typename T>
    __global__ void InverseCore(
        typename std::make_unsigned<T>::type* polynomial_in, T* polynomial_out,
        const Root<typename std::make_unsigned<
            T>::type>* __restrict__ inverse_root_of_unity_table,
        Modulus<typename std::make_unsigned<T>::type>* modulus,
        int shared_index, int logm, int k, int outer_iteration_count,
        int N_power, Ninverse<typename std::make_unsigned<T>::type>* n_inverse,
        bool last_kernel, bool reduction_poly_check, int mod_count)
    {
        using TU = typename std::make_unsigned<T>::type;

        InverseCoreBody<false>(
            polynomial_in, polynomial_out, inverse_root_of_unity_table,
            RnsModulus<TU>{modulus, mod_count, n_inverse}, SequentialPolys{},
            shared_index, logm, k, outer_iteration_count, N_power,
            last_kernel, reduction_poly_check);
    }

    template <typename T>
    __global__ void InverseCore_(
        typename std::make_unsigned<T>::type* polynomial_in, T* polynomial_out,
        const Root<typename std::make_unsigned<
            T>::type>* __restrict__ inverse_root_of_unity_table,
        Modulus<typename std::make_unsigned<T>::type> modulus, int shared_index,
        int logm, int k, int outer_iteration_count, int N_power,
        Ninverse<typename std::make_unsigned<T>::type> n_inverse,
        bool last_kernel, bool reduction_poly_check)
    {
        using TU = typename std::make_unsigned<T>::type;

        InverseCoreBody<true>(
            polynomial_in, polynomial_out, inverse_root_of_unity_table,
            SingleModulus<TU>{modulus, n_inverse}, SequentialPolys{},
            shared_index, logm, k, outer_iteration_count, N_power,
            last_kernel, reduction_poly_check);
    }

    template <typename T>
    __global__ void InverseCore_(
        typename std::make_unsigned<T>::type* polynomial_in, T* polynomial_out,
        const Root<typename std::make_unsigned<
            T>::type>* __restrict__ inverse_root_of_unity_table,
        Modulus<typename std::make_unsigned<T>::type>* modulus,
        int shared_index, int logm, int k, int outer_iteration_count,
        int N_power, Ninverse<typename std::make_unsigned<T>::type>* n_inverse,
        bool last_kernel, bool reduction_poly_check, int mod_count)
    {
        using TU = typename std::make_unsigned<T>::type;

        InverseCoreBody<true>(
            polynomial_in, polynomial_out, inverse_root_of_unity_table,
            RnsModulus<TU>{modulus, mod_count, n_inverse}, SequentialPolys{},
            shared_index, logm, k, outer_iteration_count, N_power,
            last_kernel, reduction_poly_check);
    }

    template <typename T>
    __global__ void ForwardCoreTranspose(
        T* polynomial_in, typename std::make_unsigned<T>::type* polynomial_out,
        const Root<typename std::make_unsigned<
            T>::type>* __restrict__ root_of_unity_table,
        const Modulus<typename std::make_unsigned<T>::type> modulus,
        int log_row, int log_column, bool reduction_poly_check)
    {
        using TU = typename std::make_unsigned<T>::type;

        const int idx_x = threadIdx.x;
        const int idx_y = threadIdx.y;
        const int block_x = blockIdx.x;

        extern __shared__ char shared_memory_typed[];
        TU* shared_memory = reinterpret_cast<TU*>(shared_memory_typed);
        TU* root_shared_memory = shared_memory + (1024);

        const Modulus<TU> modulus_thread = modulus;

        if (idx_x == 0)
        {
            if (reduction_poly_check)
            { // X_N_minus
                root_shared_memory[idx_y] = root_of_unity_table[idx_y];
            }
            else
            { // X_N_plus
                root_shared_memory[idx_y] = root_of_unity_table[idx_y];
                root_shared_memory[idx_y + blockDim.y] =
                    root_of_unity_table[idx_y + blockDim.y];
            }
        }

        location_t global_addresss =
            (idx_y << log_column) + (blockDim.x * block_x) + idx_x;
        location_t global_offset = (blockDim.y << log_column);

        int t_ = log_row - 1;
        int m = 1;

        // Load T from global & store to shared
        const int transpose_block = idx_y + (idx_x * (2 * blockDim.y));
        if constexpr (std::is_signed<T>::value)
        {
            T input1_reg = polynomial_in[global_addresss];
            T input2_reg = polynomial_in[global_addresss + global_offset];
            shared_memory[transpose_block] =
                OPERATOR_GPU<TU>::reduce(input1_reg, modulus_thread);
            shared_memory[transpose_block + blockDim.y] =
                OPERATOR_GPU<TU>::reduce(input2_reg, modulus_thread);
        }
        else
        {
            shared_memory[transpose_block] = polynomial_in[global_addresss];
            shared_memory[transpose_block + blockDim.y] =
                polynomial_in[global_addresss + global_offset];
        }

        const int block_thread_index = idx_x + (idx_y * blockDim.x);
        const int ntt_thread_index = block_thread_index & ((1 << t_) - 1);
        const int ntt_block_index = block_thread_index >> t_;
        int offset = ntt_block_index << log_row;

        int shared_addresss = ntt_thread_index;
        int omega_addresss = ntt_thread_index;
        int t = 1 << t_;
        int in_shared_address =
            ((shared_addresss >> t_) << t_) + shared_addresss;
        location_t current_root_index;
        __syncthreads();

        int loop = ((log_row - 6) <= 0) ? log_row : 6;

#pragma unroll
        for (int lp = 0; lp < (log_row - 6); lp++)
        {
            int group_in_shared_address = in_shared_address + offset;

            if (reduction_poly_check)
            { // X_N_minus
                current_root_index = (omega_addresss >> t_);
            }
            else
            { // X_N_plus
                current_root_index = m + (omega_addresss >> t_);
            }

            Root<TU> root = root_shared_memory[current_root_index];

            CooleyTukeyUnit(shared_memory[group_in_shared_address],
                            shared_memory[group_in_shared_address + t], root,
                            modulus_thread);

            t = t >> 1;
            t_ -= 1;
            m <<= 1;

            in_shared_address =
                ((shared_addresss >> t_) << t_) + shared_addresss;
            __syncthreads();
        }

#pragma unroll
        for (int lp = 0; lp < loop; lp++)
        {
            int group_in_shared_address = in_shared_address + offset;

            if (reduction_poly_check)
            { // X_N_minus
                current_root_index = (omega_addresss >> t_);
            }
            else
            { // X_N_plus
                current_root_index = m + (omega_addresss >> t_);
            }

            Root<TU> root = root_shared_memory[current_root_index];

            CooleyTukeyUnit(shared_memory[group_in_shared_address],
                            shared_memory[group_in_shared_address + t], root,
                            modulus_thread);

            t = t >> 1;
            t_ -= 1;
            m <<= 1;

            in_shared_address =
                ((shared_addresss >> t_) << t_) + shared_addresss;
        }

        __syncthreads();

        polynomial_out[global_addresss] = shared_memory[transpose_block];
        polynomial_out[global_addresss + global_offset] =
            shared_memory[transpose_block + blockDim.y];
    }

    template <typename T>
    __global__ void ForwardCoreTranspose(
        T* polynomial_in, typename std::make_unsigned<T>::type* polynomial_out,
        const Root<typename std::make_unsigned<
            T>::type>* __restrict__ root_of_unity_table,
        const Modulus<typename std::make_unsigned<T>::type>* modulus,
        int log_row, int log_column, bool reduction_poly_check, int mod_count)
    {
        using TU = typename std::make_unsigned<T>::type;

        const int idx_x = threadIdx.x;
        const int idx_y = threadIdx.y;
        const int block_x = blockIdx.x;

        extern __shared__ char shared_memory_typed[];
        TU* shared_memory = reinterpret_cast<TU*>(shared_memory_typed);
        TU* root_shared_memory = shared_memory + (1024);

        // Each column of the block (one NTT) keeps the root table of its own
        // modulus at root_shared_memory[column << log_row].
        const int column_base = blockDim.x * block_x;
        const int load_mod_index = (column_base + idx_x) % mod_count;
        const Modulus<TU> load_modulus = modulus[load_mod_index];
        {
            const location_t root_src = (location_t) load_mod_index << log_row;
            const int root_dst = idx_x << log_row;
            if (reduction_poly_check)
            { // X_N_minus
                root_shared_memory[root_dst + idx_y] =
                    root_of_unity_table[root_src + idx_y];
            }
            else
            { // X_N_plus
                root_shared_memory[root_dst + idx_y] =
                    root_of_unity_table[root_src + idx_y];
                root_shared_memory[root_dst + idx_y + blockDim.y] =
                    root_of_unity_table[root_src + idx_y + blockDim.y];
            }
        }

        location_t global_addresss =
            (idx_y << log_column) + (blockDim.x * block_x) + idx_x;
        location_t global_offset = (blockDim.y << log_column);

        int t_ = log_row - 1;
        int m = 1;

        const int block_thread_index = idx_x + (idx_y * blockDim.x);
        const int ntt_thread_index = block_thread_index & ((1 << t_) - 1);
        const int ntt_block_index = block_thread_index >> t_;
        int offset = ntt_block_index << log_row;

        const int mod_index = (column_base + ntt_block_index) % mod_count;
        const int root_offset = ntt_block_index << log_row;

        const Modulus<TU> modulus_thread = modulus[mod_index];

        // Load T from global & store to shared
        const int transpose_block = idx_y + (idx_x * (2 * blockDim.y));
        if constexpr (std::is_signed<T>::value)
        {
            T input1_reg = polynomial_in[global_addresss];
            T input2_reg = polynomial_in[global_addresss + global_offset];
            shared_memory[transpose_block] =
                OPERATOR_GPU<TU>::reduce(input1_reg, load_modulus);
            shared_memory[transpose_block + blockDim.y] =
                OPERATOR_GPU<TU>::reduce(input2_reg, load_modulus);
        }
        else
        {
            shared_memory[transpose_block] = polynomial_in[global_addresss];
            shared_memory[transpose_block + blockDim.y] =
                polynomial_in[global_addresss + global_offset];
        }

        int shared_addresss = ntt_thread_index;
        int omega_addresss = ntt_thread_index;
        int t = 1 << t_;
        int in_shared_address =
            ((shared_addresss >> t_) << t_) + shared_addresss;
        location_t current_root_index;
        __syncthreads();

        int loop = ((log_row - 6) <= 0) ? log_row : 6;

#pragma unroll
        for (int lp = 0; lp < (log_row - 6); lp++)
        {
            int group_in_shared_address = in_shared_address + offset;

            if (reduction_poly_check)
            { // X_N_minus
                current_root_index = (omega_addresss >> t_) + root_offset;
            }
            else
            { // X_N_plus
                current_root_index = m + (omega_addresss >> t_) + root_offset;
            }

            Root<TU> root = root_shared_memory[current_root_index];

            CooleyTukeyUnit(shared_memory[group_in_shared_address],
                            shared_memory[group_in_shared_address + t], root,
                            modulus_thread);

            t = t >> 1;
            t_ -= 1;
            m <<= 1;

            in_shared_address =
                ((shared_addresss >> t_) << t_) + shared_addresss;
            __syncthreads();
        }

#pragma unroll
        for (int lp = 0; lp < loop; lp++)
        {
            int group_in_shared_address = in_shared_address + offset;

            if (reduction_poly_check)
            { // X_N_minus
                current_root_index = (omega_addresss >> t_) + root_offset;
            }
            else
            { // X_N_plus
                current_root_index = m + (omega_addresss >> t_) + root_offset;
            }

            Root<TU> root = root_shared_memory[current_root_index];

            CooleyTukeyUnit(shared_memory[group_in_shared_address],
                            shared_memory[group_in_shared_address + t], root,
                            modulus_thread);

            t = t >> 1;
            t_ -= 1;
            m <<= 1;

            in_shared_address =
                ((shared_addresss >> t_) << t_) + shared_addresss;
        }

        __syncthreads();

        polynomial_out[global_addresss] = shared_memory[transpose_block];
        polynomial_out[global_addresss + global_offset] =
            shared_memory[transpose_block + blockDim.y];
    }

    template <typename T>
    __global__ void InverseCoreTranspose(
        typename std::make_unsigned<T>::type* polynomial_in, T* polynomial_out,
        const Root<typename std::make_unsigned<
            T>::type>* __restrict__ inverse_root_of_unity_table,
        Modulus<typename std::make_unsigned<T>::type> modulus,
        Ninverse<typename std::make_unsigned<T>::type> n_inverse, int log_row,
        int log_column, bool reduction_poly_check)
    {
        using TU = typename std::make_unsigned<T>::type;

        const int idx_x = threadIdx.x;
        const int idx_y = threadIdx.y;
        const int block_x = blockIdx.x;

        extern __shared__ char shared_memory_typed[];
        TU* shared_memory = reinterpret_cast<TU*>(shared_memory_typed);
        TU* root_shared_memory = shared_memory + (1024);

        if (idx_x == 0)
        {
            if (reduction_poly_check)
            { // X_N_minus
                root_shared_memory[idx_y] = inverse_root_of_unity_table[idx_y];
            }
            else
            { // X_N_plus
                root_shared_memory[idx_y] = inverse_root_of_unity_table[idx_y];
                root_shared_memory[idx_y + blockDim.y] =
                    inverse_root_of_unity_table[idx_y + blockDim.y];
            }
        }

        const Modulus<TU> modulus_thread = modulus;
        const Ninverse<TU> n_inverse_thread = n_inverse;

        location_t global_addresss =
            (idx_y << log_column) + (blockDim.x * block_x) + idx_x;
        location_t global_offset = (blockDim.y << log_column);

        int t_ = 0;
        int loops = log_row;
        int m = (int) 1 << (log_row - 1);

        // Load T from global & store to shared
        const int transpose_block = idx_y + (idx_x * (2 * blockDim.y));
        shared_memory[transpose_block] = polynomial_in[global_addresss];
        shared_memory[transpose_block + blockDim.y] =
            polynomial_in[global_addresss + global_offset];

        int log_row_r = log_row - 1;
        const int block_thread_index = idx_x + (idx_y * blockDim.x);
        const int ntt_thread_index =
            block_thread_index & ((1 << log_row_r) - 1);
        const int ntt_block_index = block_thread_index >> log_row_r;
        int offset = ntt_block_index << log_row;

        int omega_addresss = ntt_thread_index;
        int shared_addresss = ntt_thread_index;
        int t = 1 << t_;
        int in_shared_address =
            ((shared_addresss >> t_) << t_) + shared_addresss;
        location_t current_root_index;
        __syncthreads();

#pragma unroll
        for (int lp = 0; lp < loops; lp++)
        {
            int group_in_shared_address = in_shared_address + offset;

            if (reduction_poly_check)
            { // X_N_minus
                current_root_index = (omega_addresss >> t_);
            }
            else
            { // X_N_plus
                current_root_index = m + (omega_addresss >> t_);
            }

            Root<TU> root = root_shared_memory[current_root_index];

            GentlemanSandeUnit(shared_memory[group_in_shared_address],
                               shared_memory[group_in_shared_address + t], root,
                               modulus_thread);

            t = t << 1;
            t_ += 1;
            m >>= 1;

            in_shared_address =
                ((shared_addresss >> t_) << t_) + shared_addresss;
            __syncthreads();
        }

        TU output1_reg = OPERATOR_GPU<TU>::mult(
            shared_memory[transpose_block], n_inverse_thread, modulus_thread);
        TU output2_reg =
            OPERATOR_GPU<TU>::mult(shared_memory[transpose_block + blockDim.y],
                                   n_inverse_thread, modulus_thread);

        if constexpr (std::is_signed<T>::value)
        {
            polynomial_out[global_addresss] =
                OPERATOR_GPU<TU>::centered_reduction(output1_reg,
                                                     modulus_thread);
            polynomial_out[global_addresss + global_offset] =
                OPERATOR_GPU<TU>::centered_reduction(output2_reg,
                                                     modulus_thread);
        }
        else
        {
            polynomial_out[global_addresss] = output1_reg;
            polynomial_out[global_addresss + global_offset] = output2_reg;
        }
    }

    template <typename T>
    __global__ void InverseCoreTranspose(
        typename std::make_unsigned<T>::type* polynomial_in, T* polynomial_out,
        const Root<typename std::make_unsigned<
            T>::type>* __restrict__ inverse_root_of_unity_table,
        Modulus<typename std::make_unsigned<T>::type>* modulus,
        Ninverse<typename std::make_unsigned<T>::type>* n_inverse, int log_row,
        int log_column, bool reduction_poly_check, int mod_count)
    {
        using TU = typename std::make_unsigned<T>::type;

        const int idx_x = threadIdx.x;
        const int idx_y = threadIdx.y;
        const int block_x = blockIdx.x;

        extern __shared__ char shared_memory_typed[];
        TU* shared_memory = reinterpret_cast<TU*>(shared_memory_typed);
        TU* root_shared_memory = shared_memory + (1024);

        // Each column of the block (one NTT) keeps the root table of its own
        // modulus at root_shared_memory[column << log_row].
        const int column_base = blockDim.x * block_x;
        const int store_mod_index = (column_base + idx_x) % mod_count;
        {
            const location_t root_src = (location_t) store_mod_index << log_row;
            const int root_dst = idx_x << log_row;
            if (reduction_poly_check)
            { // X_N_minus
                root_shared_memory[root_dst + idx_y] =
                    inverse_root_of_unity_table[root_src + idx_y];
            }
            else
            { // X_N_plus
                root_shared_memory[root_dst + idx_y] =
                    inverse_root_of_unity_table[root_src + idx_y];
                root_shared_memory[root_dst + idx_y + blockDim.y] =
                    inverse_root_of_unity_table[root_src + idx_y + blockDim.y];
            }
        }

        location_t global_addresss =
            (idx_y << log_column) + (blockDim.x * block_x) + idx_x;
        location_t global_offset = (blockDim.y << log_column);

        int t_ = 0;
        int loops = log_row;
        int m = (int) 1 << (log_row - 1);

        // Load T from global & store to shared
        const int transpose_block = idx_y + (idx_x * (2 * blockDim.y));
        shared_memory[transpose_block] = polynomial_in[global_addresss];
        shared_memory[transpose_block + blockDim.y] =
            polynomial_in[global_addresss + global_offset];

        int log_row_r = log_row - 1;
        const int block_thread_index = idx_x + (idx_y * blockDim.x);
        const int ntt_thread_index =
            block_thread_index & ((1 << log_row_r) - 1);
        const int ntt_block_index = block_thread_index >> log_row_r;
        int offset = ntt_block_index << log_row;

        const int mod_index = (column_base + ntt_block_index) % mod_count;
        const int root_offset = ntt_block_index << log_row;

        const Modulus<TU> modulus_thread = modulus[mod_index];
        const Modulus<TU> store_modulus = modulus[store_mod_index];
        const Ninverse<TU> store_n_inverse = n_inverse[store_mod_index];

        int omega_addresss = ntt_thread_index;
        int shared_addresss = ntt_thread_index;
        int t = 1 << t_;
        int in_shared_address =
            ((shared_addresss >> t_) << t_) + shared_addresss;
        location_t current_root_index;
        __syncthreads();

#pragma unroll
        for (int lp = 0; lp < loops; lp++)
        {
            int group_in_shared_address = in_shared_address + offset;

            if (reduction_poly_check)
            { // X_N_minus
                current_root_index = (omega_addresss >> t_) + root_offset;
            }
            else
            { // X_N_plus
                current_root_index = m + (omega_addresss >> t_) + root_offset;
            }

            Root<TU> root = root_shared_memory[current_root_index];

            GentlemanSandeUnit(shared_memory[group_in_shared_address],
                               shared_memory[group_in_shared_address + t], root,
                               modulus_thread);

            t = t << 1;
            t_ += 1;
            m >>= 1;

            in_shared_address =
                ((shared_addresss >> t_) << t_) + shared_addresss;
            __syncthreads();
        }

        // The thread stores its own column, so it uses that column's modulus.
        TU output1_reg = OPERATOR_GPU<TU>::mult(
            shared_memory[transpose_block], store_n_inverse, store_modulus);
        TU output2_reg =
            OPERATOR_GPU<TU>::mult(shared_memory[transpose_block + blockDim.y],
                                   store_n_inverse, store_modulus);

        if constexpr (std::is_signed<T>::value)
        {
            polynomial_out[global_addresss] =
                OPERATOR_GPU<TU>::centered_reduction(output1_reg,
                                                     store_modulus);
            polynomial_out[global_addresss + global_offset] =
                OPERATOR_GPU<TU>::centered_reduction(output2_reg,
                                                     store_modulus);
        }
        else
        {
            polynomial_out[global_addresss] = output1_reg;
            polynomial_out[global_addresss + global_offset] = output2_reg;
        }
    }

    template <typename T>
    __host__ void
    GPU_NTT(T* device_in, typename std::make_unsigned<T>::type* device_out,
            Root<typename std::make_unsigned<T>::type>* root_of_unity_table,
            Modulus<typename std::make_unsigned<T>::type> modulus,
            ntt_configuration<typename std::make_unsigned<T>::type> cfg,
            int batch_size)
    {
        check_batch_size(cfg.n_power, batch_size);

        switch (cfg.ntt_layout)
        {
            case PerPolynomial:
            {
                if ((cfg.n_power <= 0 || cfg.n_power >= 29))
                {
                    throw std::invalid_argument("Invalid n_power range!");
                }

                auto kernel_parameters = CreateForwardNTTKernel<
                    typename std::make_unsigned<T>::type>();
                bool low_ring_size = (cfg.n_power < 10) ? true : false;
                bool standart_kernel = (cfg.n_power < 25) ? true : false;

                if (low_ring_size)
                {
                    auto& current_kernel_params =
                        kernel_parameters[cfg.n_power][0];
                    ForwardCoreLowRing<<<
                        dim3((batch_size +
                              (current_kernel_params.blockdim_y - 1)) /
                                 current_kernel_params.blockdim_y,
                             1, 1),
                        dim3(current_kernel_params.blockdim_x,
                             current_kernel_params.blockdim_y),
                        current_kernel_params.shared_memory, cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus,
                        current_kernel_params.shared_index, cfg.n_power,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                        batch_size);
                    GPUNTT_CUDA_CHECK(cudaGetLastError());
                }
                else
                {
                    if (standart_kernel)
                    {
                        auto& current_kernel_params =
                            kernel_parameters[cfg.n_power][0];
                        ForwardCore<<<
                            dim3(current_kernel_params.griddim_x,
                                 current_kernel_params.griddim_y, batch_size),
                            dim3(current_kernel_params.blockdim_x,
                                 current_kernel_params.blockdim_y),
                            current_kernel_params.shared_memory, cfg.stream>>>(
                            device_in, device_out, root_of_unity_table, modulus,
                            current_kernel_params.shared_index,
                            current_kernel_params.logm,
                            current_kernel_params.outer_iteration_count,
                            cfg.n_power,
                            current_kernel_params.not_last_kernel,
                            (cfg.reduction_poly ==
                             ReductionPolynomial::X_N_minus));
                        GPUNTT_CUDA_CHECK(cudaGetLastError());

                        for (int i = 1;
                             i < kernel_parameters[cfg.n_power].size(); i++)
                        {
                            auto& current_kernel_params =
                                kernel_parameters[cfg.n_power][i];
                            ForwardCore<<<dim3(current_kernel_params.griddim_x,
                                               current_kernel_params.griddim_y,
                                               batch_size),
                                          dim3(
                                              current_kernel_params.blockdim_x,
                                              current_kernel_params.blockdim_y),
                                          current_kernel_params.shared_memory,
                                          cfg.stream>>>(
                                device_out, device_out, root_of_unity_table,
                                modulus, current_kernel_params.shared_index,
                                current_kernel_params.logm,
                                current_kernel_params.outer_iteration_count,
                                cfg.n_power,
                                current_kernel_params.not_last_kernel,
                                (cfg.reduction_poly ==
                                 ReductionPolynomial::X_N_minus));
                            GPUNTT_CUDA_CHECK(cudaGetLastError());
                        }
                    }
                    else
                    {
                        auto& current_kernel_params =
                            kernel_parameters[cfg.n_power][0];
                        ForwardCore<<<
                            dim3(current_kernel_params.griddim_x,
                                 current_kernel_params.griddim_y, batch_size),
                            dim3(current_kernel_params.blockdim_x,
                                 current_kernel_params.blockdim_y),
                            current_kernel_params.shared_memory, cfg.stream>>>(
                            device_in, device_out, root_of_unity_table, modulus,
                            current_kernel_params.shared_index,
                            current_kernel_params.logm,
                            current_kernel_params.outer_iteration_count,
                            cfg.n_power,
                            current_kernel_params.not_last_kernel,
                            (cfg.reduction_poly ==
                             ReductionPolynomial::X_N_minus));
                        GPUNTT_CUDA_CHECK(cudaGetLastError());

                        for (int i = 1;
                             i < kernel_parameters[cfg.n_power].size() - 1; i++)
                        {
                            auto& current_kernel_params =
                                kernel_parameters[cfg.n_power][i];
                            ForwardCore<<<dim3(current_kernel_params.griddim_x,
                                               current_kernel_params.griddim_y,
                                               batch_size),
                                          dim3(
                                              current_kernel_params.blockdim_x,
                                              current_kernel_params.blockdim_y),
                                          current_kernel_params.shared_memory,
                                          cfg.stream>>>(
                                device_out, device_out, root_of_unity_table,
                                modulus, current_kernel_params.shared_index,
                                current_kernel_params.logm,
                                current_kernel_params.outer_iteration_count,
                                cfg.n_power,
                                current_kernel_params.not_last_kernel,
                                (cfg.reduction_poly ==
                                 ReductionPolynomial::X_N_minus));
                            GPUNTT_CUDA_CHECK(cudaGetLastError());
                        }
                        current_kernel_params = kernel_parameters
                            [cfg.n_power]
                            [kernel_parameters[cfg.n_power].size() - 1];
                        ForwardCore_<<<
                            dim3(current_kernel_params.griddim_x,
                                 current_kernel_params.griddim_y, batch_size),
                            dim3(current_kernel_params.blockdim_x,
                                 current_kernel_params.blockdim_y),
                            current_kernel_params.shared_memory, cfg.stream>>>(
                            device_out, device_out, root_of_unity_table,
                            modulus, current_kernel_params.shared_index,
                            current_kernel_params.logm,
                            current_kernel_params.outer_iteration_count,
                            cfg.n_power,
                            current_kernel_params.not_last_kernel,
                            (cfg.reduction_poly ==
                             ReductionPolynomial::X_N_minus));
                        GPUNTT_CUDA_CHECK(cudaGetLastError());
                    }
                }
            }
            break;
            case PerCoefficient:
            {
                if ((cfg.n_power <= 0 || cfg.n_power >= 10))
                {
                    throw std::invalid_argument("Invalid n_power range!");
                }

                if ((batch_size & (batch_size - 1)) != 0)
                {
                    throw std::invalid_argument(
                        "batch_size must be a power of two for "
                        "PerCoefficient layout!");
                }

                int log_batch_size = log2(batch_size);
                int total_size = 1 << (cfg.n_power + log_batch_size);
                int total_block_thread = 512;
                int total_block_count = total_size / (total_block_thread * 2);
                int blockdim_y = 1 << (cfg.n_power - 1);
                int blockdim_x = total_block_thread / blockdim_y;
                ForwardCoreTranspose<<<
                    dim3(total_block_count, 1, 1),
                    dim3(blockdim_x, blockdim_y, 1),
                    ((total_block_thread * 2) + (1 << cfg.n_power)) * sizeof(T),
                    cfg.stream>>>(
                    device_in, device_out, root_of_unity_table, modulus,
                    cfg.n_power, log_batch_size,
                    (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                GPUNTT_CUDA_CHECK(cudaGetLastError());
            }
            break;
            default:
                throw std::invalid_argument("Invalid ntt_layout!");
                break;
        }
    }

    template <typename T>
    __host__ void
    GPU_INTT(typename std::make_unsigned<T>::type* device_in, T* device_out,
             Root<typename std::make_unsigned<T>::type>* root_of_unity_table,
             Modulus<typename std::make_unsigned<T>::type> modulus,
             ntt_configuration<typename std::make_unsigned<T>::type> cfg,
             int batch_size)
    {
        check_batch_size(cfg.n_power, batch_size);

        switch (cfg.ntt_layout)
        {
            case PerPolynomial:
            {
                if ((cfg.n_power <= 0 || cfg.n_power >= 29))
                {
                    throw std::invalid_argument("Invalid n_power range!");
                }

                auto kernel_parameters = CreateInverseNTTKernel<
                    typename std::make_unsigned<T>::type>();
                bool low_ring_size = (cfg.n_power < 11) ? true : false;
                bool standart_kernel = (cfg.n_power < 25) ? true : false;

                if (low_ring_size)
                {
                    auto& current_kernel_params =
                        kernel_parameters[cfg.n_power][0];
                    InverseCoreLowRing<<<
                        dim3((batch_size +
                              (current_kernel_params.blockdim_y - 1)) /
                                 current_kernel_params.blockdim_y,
                             1, 1),
                        dim3(current_kernel_params.blockdim_x,
                             current_kernel_params.blockdim_y),
                        current_kernel_params.shared_memory, cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus,
                        current_kernel_params.shared_index, cfg.n_power,
                        cfg.mod_inverse,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                        batch_size);
                    GPUNTT_CUDA_CHECK(cudaGetLastError());
                }
                else
                {
                    if (standart_kernel)
                    {
                        if constexpr (std::is_signed<T>::value)
                        {
                            typename std::make_unsigned<T>::type* device_in_ =
                                device_in;
                            for (int i = 0;
                                 i < kernel_parameters[cfg.n_power].size() - 1;
                                 i++)
                            {
                                auto& current_kernel_params =
                                    kernel_parameters[cfg.n_power][i];
                                InverseCore<<<
                                    dim3(current_kernel_params.griddim_x,
                                         current_kernel_params.griddim_y,
                                         batch_size),
                                    dim3(current_kernel_params.blockdim_x,
                                         current_kernel_params.blockdim_y),
                                    current_kernel_params.shared_memory,
                                    cfg.stream>>>(
                                    device_in_,
                                    reinterpret_cast<
                                        typename std::make_unsigned<T>::type*>(
                                        device_out),
                                    root_of_unity_table, modulus,
                                    current_kernel_params.shared_index,
                                    current_kernel_params.logm,
                                    current_kernel_params.k,
                                    current_kernel_params.outer_iteration_count,
                                    cfg.n_power, cfg.mod_inverse,
                                    current_kernel_params.not_last_kernel,
                                    (cfg.reduction_poly ==
                                     ReductionPolynomial::X_N_minus));
                                GPUNTT_CUDA_CHECK(cudaGetLastError());
                                device_in_ = reinterpret_cast<
                                    typename std::make_unsigned<T>::type*>(
                                    device_out);
                            }

                            auto& current_kernel_params = kernel_parameters
                                [cfg.n_power]
                                [kernel_parameters[cfg.n_power].size() - 1];
                            InverseCore<<<dim3(current_kernel_params.griddim_x,
                                               current_kernel_params.griddim_y,
                                               batch_size),
                                          dim3(
                                              current_kernel_params.blockdim_x,
                                              current_kernel_params.blockdim_y),
                                          current_kernel_params.shared_memory,
                                          cfg.stream>>>(
                                reinterpret_cast<
                                    typename std::make_unsigned<T>::type*>(
                                    device_out),
                                device_out, root_of_unity_table, modulus,
                                current_kernel_params.shared_index,
                                current_kernel_params.logm,
                                current_kernel_params.k,
                                current_kernel_params.outer_iteration_count,
                                cfg.n_power, cfg.mod_inverse,
                                current_kernel_params.not_last_kernel,
                                (cfg.reduction_poly ==
                                 ReductionPolynomial::X_N_minus));
                            GPUNTT_CUDA_CHECK(cudaGetLastError());
                        }
                        else
                        {
                            T* device_in_ = device_in;
                            for (int i = 0;
                                 i < kernel_parameters[cfg.n_power].size(); i++)
                            {
                                auto& current_kernel_params =
                                    kernel_parameters[cfg.n_power][i];
                                InverseCore<<<
                                    dim3(current_kernel_params.griddim_x,
                                         current_kernel_params.griddim_y,
                                         batch_size),
                                    dim3(current_kernel_params.blockdim_x,
                                         current_kernel_params.blockdim_y),
                                    current_kernel_params.shared_memory,
                                    cfg.stream>>>(
                                    device_in_, device_out, root_of_unity_table,
                                    modulus, current_kernel_params.shared_index,
                                    current_kernel_params.logm,
                                    current_kernel_params.k,
                                    current_kernel_params.outer_iteration_count,
                                    cfg.n_power, cfg.mod_inverse,
                                    current_kernel_params.not_last_kernel,
                                    (cfg.reduction_poly ==
                                     ReductionPolynomial::X_N_minus));
                                GPUNTT_CUDA_CHECK(cudaGetLastError());
                                device_in_ = device_out;
                            }
                        }
                    }
                    else
                    {
                        if constexpr (std::is_signed<T>::value)
                        {
                            auto& current_kernel_params =
                                kernel_parameters[cfg.n_power][0];
                            InverseCore_<<<
                                dim3(current_kernel_params.griddim_x,
                                     current_kernel_params.griddim_y,
                                     batch_size),
                                dim3(current_kernel_params.blockdim_x,
                                     current_kernel_params.blockdim_y),
                                current_kernel_params.shared_memory,
                                cfg.stream>>>(
                                device_in,
                                reinterpret_cast<
                                    typename std::make_unsigned<T>::type*>(
                                    device_out),
                                root_of_unity_table, modulus,
                                current_kernel_params.shared_index,
                                current_kernel_params.logm,
                                current_kernel_params.k,
                                current_kernel_params.outer_iteration_count,
                                cfg.n_power, cfg.mod_inverse,
                                current_kernel_params.not_last_kernel,
                                (cfg.reduction_poly ==
                                 ReductionPolynomial::X_N_minus));
                            GPUNTT_CUDA_CHECK(cudaGetLastError());
                            for (int i = 1;
                                 i < kernel_parameters[cfg.n_power].size() - 1;
                                 i++)
                            {
                                auto& current_kernel_params =
                                    kernel_parameters[cfg.n_power][i];
                                InverseCore<<<
                                    dim3(current_kernel_params.griddim_x,
                                         current_kernel_params.griddim_y,
                                         batch_size),
                                    dim3(current_kernel_params.blockdim_x,
                                         current_kernel_params.blockdim_y),
                                    current_kernel_params.shared_memory,
                                    cfg.stream>>>(
                                    reinterpret_cast<
                                        typename std::make_unsigned<T>::type*>(
                                        device_out),
                                    reinterpret_cast<
                                        typename std::make_unsigned<T>::type*>(
                                        device_out),
                                    root_of_unity_table, modulus,
                                    current_kernel_params.shared_index,
                                    current_kernel_params.logm,
                                    current_kernel_params.k,
                                    current_kernel_params.outer_iteration_count,
                                    cfg.n_power, cfg.mod_inverse,
                                    current_kernel_params.not_last_kernel,
                                    (cfg.reduction_poly ==
                                     ReductionPolynomial::X_N_minus));
                                GPUNTT_CUDA_CHECK(cudaGetLastError());
                            }

                            current_kernel_params = kernel_parameters
                                [cfg.n_power]
                                [kernel_parameters[cfg.n_power].size() - 1];
                            InverseCore<<<dim3(current_kernel_params.griddim_x,
                                               current_kernel_params.griddim_y,
                                               batch_size),
                                          dim3(
                                              current_kernel_params.blockdim_x,
                                              current_kernel_params.blockdim_y),
                                          current_kernel_params.shared_memory,
                                          cfg.stream>>>(
                                reinterpret_cast<
                                    typename std::make_unsigned<T>::type*>(
                                    device_out),
                                device_out, root_of_unity_table, modulus,
                                current_kernel_params.shared_index,
                                current_kernel_params.logm,
                                current_kernel_params.k,
                                current_kernel_params.outer_iteration_count,
                                cfg.n_power, cfg.mod_inverse,
                                current_kernel_params.not_last_kernel,
                                (cfg.reduction_poly ==
                                 ReductionPolynomial::X_N_minus));
                            GPUNTT_CUDA_CHECK(cudaGetLastError());
                        }
                        else
                        {
                            auto& current_kernel_params =
                                kernel_parameters[cfg.n_power][0];
                            InverseCore_<<<
                                dim3(current_kernel_params.griddim_x,
                                     current_kernel_params.griddim_y,
                                     batch_size),
                                dim3(current_kernel_params.blockdim_x,
                                     current_kernel_params.blockdim_y),
                                current_kernel_params.shared_memory,
                                cfg.stream>>>(
                                device_in, device_out, root_of_unity_table,
                                modulus, current_kernel_params.shared_index,
                                current_kernel_params.logm,
                                current_kernel_params.k,
                                current_kernel_params.outer_iteration_count,
                                cfg.n_power, cfg.mod_inverse,
                                current_kernel_params.not_last_kernel,
                                (cfg.reduction_poly ==
                                 ReductionPolynomial::X_N_minus));
                            GPUNTT_CUDA_CHECK(cudaGetLastError());

                            for (int i = 1;
                                 i < kernel_parameters[cfg.n_power].size(); i++)
                            {
                                auto& current_kernel_params =
                                    kernel_parameters[cfg.n_power][i];
                                InverseCore<<<
                                    dim3(current_kernel_params.griddim_x,
                                         current_kernel_params.griddim_y,
                                         batch_size),
                                    dim3(current_kernel_params.blockdim_x,
                                         current_kernel_params.blockdim_y),
                                    current_kernel_params.shared_memory,
                                    cfg.stream>>>(
                                    device_out, device_out, root_of_unity_table,
                                    modulus, current_kernel_params.shared_index,
                                    current_kernel_params.logm,
                                    current_kernel_params.k,
                                    current_kernel_params.outer_iteration_count,
                                    cfg.n_power, cfg.mod_inverse,
                                    current_kernel_params.not_last_kernel,
                                    (cfg.reduction_poly ==
                                     ReductionPolynomial::X_N_minus));
                                GPUNTT_CUDA_CHECK(cudaGetLastError());
                            }
                        }
                    }
                }
            }
            break;
            case PerCoefficient:
            {
                if ((cfg.n_power <= 0 || cfg.n_power >= 10))
                {
                    throw std::invalid_argument("Invalid n_power range!");
                }

                if ((batch_size & (batch_size - 1)) != 0)
                {
                    throw std::invalid_argument(
                        "batch_size must be a power of two for "
                        "PerCoefficient layout!");
                }

                int log_batch_size = log2(batch_size);
                int total_size = 1 << (cfg.n_power + log_batch_size);
                int total_block_thread = 512;
                int total_block_count = total_size / (total_block_thread * 2);
                int blockdim_y = 1 << (cfg.n_power - 1);
                int blockdim_x = total_block_thread / blockdim_y;
                InverseCoreTranspose<<<
                    dim3(total_block_count, 1, 1),
                    dim3(blockdim_x, blockdim_y, 1),
                    ((total_block_thread * 2) + (1 << cfg.n_power)) * sizeof(T),
                    cfg.stream>>>(
                    device_in, device_out, root_of_unity_table, modulus,
                    cfg.mod_inverse, cfg.n_power, log_batch_size,
                    (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                GPUNTT_CUDA_CHECK(cudaGetLastError());
            }
            break;
            default:
                throw std::invalid_argument("Invalid ntt_layout!");
                break;
        }
    }

    template <typename T>
    __host__ void
    GPU_NTT(T* device_in, typename std::make_unsigned<T>::type* device_out,
            Root<typename std::make_unsigned<T>::type>* root_of_unity_table,
            Modulus<typename std::make_unsigned<T>::type>* modulus,
            ntt_rns_configuration<typename std::make_unsigned<T>::type> cfg,
            int batch_size, int mod_count)
    {
        check_batch_size(cfg.n_power, batch_size);

        switch (cfg.ntt_layout)
        {
            case PerPolynomial:
            {
                if ((cfg.n_power <= 0 || cfg.n_power >= 29))
                {
                    throw std::invalid_argument("Invalid n_power range!");
                }

                auto kernel_parameters = CreateForwardNTTKernel<
                    typename std::make_unsigned<T>::type>();
                bool low_ring_size = (cfg.n_power < 10) ? true : false;
                bool standart_kernel = (cfg.n_power < 25) ? true : false;

                if (low_ring_size)
                {
                    auto& current_kernel_params =
                        kernel_parameters[cfg.n_power][0];
                    ForwardCoreLowRing<<<
                        dim3((batch_size +
                              (current_kernel_params.blockdim_y - 1)) /
                                 current_kernel_params.blockdim_y,
                             1, 1),
                        dim3(current_kernel_params.blockdim_x,
                             current_kernel_params.blockdim_y),
                        current_kernel_params.shared_memory, cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus,
                        current_kernel_params.shared_index, cfg.n_power,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                        batch_size, mod_count);
                    GPUNTT_CUDA_CHECK(cudaGetLastError());
                }
                else
                {
                    if (standart_kernel)
                    {
                        auto& current_kernel_params =
                            kernel_parameters[cfg.n_power][0];
                        ForwardCore<<<
                            dim3(current_kernel_params.griddim_x,
                                 current_kernel_params.griddim_y, batch_size),
                            dim3(current_kernel_params.blockdim_x,
                                 current_kernel_params.blockdim_y),
                            current_kernel_params.shared_memory, cfg.stream>>>(
                            device_in, device_out, root_of_unity_table, modulus,
                            current_kernel_params.shared_index,
                            current_kernel_params.logm,
                            current_kernel_params.outer_iteration_count,
                            cfg.n_power,
                            current_kernel_params.not_last_kernel,
                            (cfg.reduction_poly ==
                             ReductionPolynomial::X_N_minus),
                            mod_count);
                        GPUNTT_CUDA_CHECK(cudaGetLastError());

                        for (int i = 1;
                             i < kernel_parameters[cfg.n_power].size(); i++)
                        {
                            auto& current_kernel_params =
                                kernel_parameters[cfg.n_power][i];
                            ForwardCore<<<dim3(current_kernel_params.griddim_x,
                                               current_kernel_params.griddim_y,
                                               batch_size),
                                          dim3(
                                              current_kernel_params.blockdim_x,
                                              current_kernel_params.blockdim_y),
                                          current_kernel_params.shared_memory,
                                          cfg.stream>>>(
                                device_out, device_out, root_of_unity_table,
                                modulus, current_kernel_params.shared_index,
                                current_kernel_params.logm,
                                current_kernel_params.outer_iteration_count,
                                cfg.n_power,
                                current_kernel_params.not_last_kernel,
                                (cfg.reduction_poly ==
                                 ReductionPolynomial::X_N_minus),
                                mod_count);
                            GPUNTT_CUDA_CHECK(cudaGetLastError());
                        }
                    }
                    else
                    {
                        auto& current_kernel_params =
                            kernel_parameters[cfg.n_power][0];
                        ForwardCore<<<
                            dim3(current_kernel_params.griddim_x,
                                 current_kernel_params.griddim_y, batch_size),
                            dim3(current_kernel_params.blockdim_x,
                                 current_kernel_params.blockdim_y),
                            current_kernel_params.shared_memory, cfg.stream>>>(
                            device_in, device_out, root_of_unity_table, modulus,
                            current_kernel_params.shared_index,
                            current_kernel_params.logm,
                            current_kernel_params.outer_iteration_count,
                            cfg.n_power,
                            current_kernel_params.not_last_kernel,
                            (cfg.reduction_poly ==
                             ReductionPolynomial::X_N_minus),
                            mod_count);
                        GPUNTT_CUDA_CHECK(cudaGetLastError());

                        for (int i = 1;
                             i < kernel_parameters[cfg.n_power].size() - 1; i++)
                        {
                            auto& current_kernel_params =
                                kernel_parameters[cfg.n_power][i];
                            ForwardCore<<<dim3(current_kernel_params.griddim_x,
                                               current_kernel_params.griddim_y,
                                               batch_size),
                                          dim3(
                                              current_kernel_params.blockdim_x,
                                              current_kernel_params.blockdim_y),
                                          current_kernel_params.shared_memory,
                                          cfg.stream>>>(
                                device_out, device_out, root_of_unity_table,
                                modulus, current_kernel_params.shared_index,
                                current_kernel_params.logm,
                                current_kernel_params.outer_iteration_count,
                                cfg.n_power,
                                current_kernel_params.not_last_kernel,
                                (cfg.reduction_poly ==
                                 ReductionPolynomial::X_N_minus),
                                mod_count);
                            GPUNTT_CUDA_CHECK(cudaGetLastError());
                        }
                        current_kernel_params = kernel_parameters
                            [cfg.n_power]
                            [kernel_parameters[cfg.n_power].size() - 1];
                        ForwardCore_<<<
                            dim3(current_kernel_params.griddim_x,
                                 current_kernel_params.griddim_y, batch_size),
                            dim3(current_kernel_params.blockdim_x,
                                 current_kernel_params.blockdim_y),
                            current_kernel_params.shared_memory, cfg.stream>>>(
                            device_out, device_out, root_of_unity_table,
                            modulus, current_kernel_params.shared_index,
                            current_kernel_params.logm,
                            current_kernel_params.outer_iteration_count,
                            cfg.n_power,
                            current_kernel_params.not_last_kernel,
                            (cfg.reduction_poly ==
                             ReductionPolynomial::X_N_minus),
                            mod_count);
                        GPUNTT_CUDA_CHECK(cudaGetLastError());
                    }
                }
            }
            break;
            case PerCoefficient:
            {
                if ((cfg.n_power <= 0 || cfg.n_power >= 10))
                {
                    throw std::invalid_argument("Invalid n_power range!");
                }

                if ((batch_size & (batch_size - 1)) != 0)
                {
                    throw std::invalid_argument(
                        "batch_size must be a power of two for "
                        "PerCoefficient layout!");
                }

                int log_batch_size = log2(batch_size);
                int total_size = 1 << (cfg.n_power + log_batch_size);
                int total_block_thread = 512;
                int total_block_count = total_size / (total_block_thread * 2);
                int blockdim_y = 1 << (cfg.n_power - 1);
                int blockdim_x = total_block_thread / blockdim_y;
                ForwardCoreTranspose<<<
                    dim3(total_block_count, 1, 1),
                    dim3(blockdim_x, blockdim_y, 1),
                    // Data and one root table per column of the block.
                    (total_block_thread * 2 * 2) * sizeof(T),
                    cfg.stream>>>(
                    device_in, device_out, root_of_unity_table, modulus,
                    cfg.n_power, log_batch_size,
                    (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                    mod_count);
                GPUNTT_CUDA_CHECK(cudaGetLastError());
            }
            break;
            default:
                throw std::invalid_argument("Invalid ntt_layout!");
                break;
        }
    }

    template <typename T>
    __host__ void
    GPU_INTT(typename std::make_unsigned<T>::type* device_in, T* device_out,
             Root<typename std::make_unsigned<T>::type>* root_of_unity_table,
             Modulus<typename std::make_unsigned<T>::type>* modulus,
             ntt_rns_configuration<typename std::make_unsigned<T>::type> cfg,
             int batch_size, int mod_count)
    {
        check_batch_size(cfg.n_power, batch_size);

        switch (cfg.ntt_layout)
        {
            case PerPolynomial:
            {
                if ((cfg.n_power <= 0 || cfg.n_power >= 29))
                {
                    throw std::invalid_argument("Invalid n_power range!");
                }

                auto kernel_parameters = CreateInverseNTTKernel<
                    typename std::make_unsigned<T>::type>();
                bool low_ring_size = (cfg.n_power < 11) ? true : false;
                bool standart_kernel = (cfg.n_power < 25) ? true : false;

                if (low_ring_size)
                {
                    auto& current_kernel_params =
                        kernel_parameters[cfg.n_power][0];
                    InverseCoreLowRing<<<
                        dim3((batch_size +
                              (current_kernel_params.blockdim_y - 1)) /
                                 current_kernel_params.blockdim_y,
                             1, 1),
                        dim3(current_kernel_params.blockdim_x,
                             current_kernel_params.blockdim_y),
                        current_kernel_params.shared_memory, cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus,
                        current_kernel_params.shared_index, cfg.n_power,
                        cfg.mod_inverse,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                        batch_size, mod_count);
                    GPUNTT_CUDA_CHECK(cudaGetLastError());
                }
                else
                {
                    if (standart_kernel)
                    {
                        if constexpr (std::is_signed<T>::value)
                        {
                            typename std::make_unsigned<T>::type* device_in_ =
                                device_in;
                            for (int i = 0;
                                 i < kernel_parameters[cfg.n_power].size() - 1;
                                 i++)
                            {
                                auto& current_kernel_params =
                                    kernel_parameters[cfg.n_power][i];
                                InverseCore<<<
                                    dim3(current_kernel_params.griddim_x,
                                         current_kernel_params.griddim_y,
                                         batch_size),
                                    dim3(current_kernel_params.blockdim_x,
                                         current_kernel_params.blockdim_y),
                                    current_kernel_params.shared_memory,
                                    cfg.stream>>>(
                                    device_in_,
                                    reinterpret_cast<
                                        typename std::make_unsigned<T>::type*>(
                                        device_out),
                                    root_of_unity_table, modulus,
                                    current_kernel_params.shared_index,
                                    current_kernel_params.logm,
                                    current_kernel_params.k,
                                    current_kernel_params.outer_iteration_count,
                                    cfg.n_power, cfg.mod_inverse,
                                    current_kernel_params.not_last_kernel,
                                    (cfg.reduction_poly ==
                                     ReductionPolynomial::X_N_minus),
                                    mod_count);
                                GPUNTT_CUDA_CHECK(cudaGetLastError());
                                device_in_ = reinterpret_cast<
                                    typename std::make_unsigned<T>::type*>(
                                    device_out);
                            }

                            auto& current_kernel_params = kernel_parameters
                                [cfg.n_power]
                                [kernel_parameters[cfg.n_power].size() - 1];
                            InverseCore<<<dim3(current_kernel_params.griddim_x,
                                               current_kernel_params.griddim_y,
                                               batch_size),
                                          dim3(
                                              current_kernel_params.blockdim_x,
                                              current_kernel_params.blockdim_y),
                                          current_kernel_params.shared_memory,
                                          cfg.stream>>>(
                                reinterpret_cast<
                                    typename std::make_unsigned<T>::type*>(
                                    device_out),
                                device_out, root_of_unity_table, modulus,
                                current_kernel_params.shared_index,
                                current_kernel_params.logm,
                                current_kernel_params.k,
                                current_kernel_params.outer_iteration_count,
                                cfg.n_power, cfg.mod_inverse,
                                current_kernel_params.not_last_kernel,
                                (cfg.reduction_poly ==
                                 ReductionPolynomial::X_N_minus),
                                mod_count);
                            GPUNTT_CUDA_CHECK(cudaGetLastError());
                        }
                        else
                        {
                            T* device_in_ = device_in;
                            for (int i = 0;
                                 i < kernel_parameters[cfg.n_power].size(); i++)
                            {
                                auto& current_kernel_params =
                                    kernel_parameters[cfg.n_power][i];
                                InverseCore<<<
                                    dim3(current_kernel_params.griddim_x,
                                         current_kernel_params.griddim_y,
                                         batch_size),
                                    dim3(current_kernel_params.blockdim_x,
                                         current_kernel_params.blockdim_y),
                                    current_kernel_params.shared_memory,
                                    cfg.stream>>>(
                                    device_in_, device_out, root_of_unity_table,
                                    modulus, current_kernel_params.shared_index,
                                    current_kernel_params.logm,
                                    current_kernel_params.k,
                                    current_kernel_params.outer_iteration_count,
                                    cfg.n_power, cfg.mod_inverse,
                                    current_kernel_params.not_last_kernel,
                                    (cfg.reduction_poly ==
                                     ReductionPolynomial::X_N_minus),
                                    mod_count);
                                GPUNTT_CUDA_CHECK(cudaGetLastError());
                                device_in_ = device_out;
                            }
                        }
                    }
                    else
                    {
                        if constexpr (std::is_signed<T>::value)
                        {
                            auto& current_kernel_params =
                                kernel_parameters[cfg.n_power][0];
                            InverseCore_<<<
                                dim3(current_kernel_params.griddim_x,
                                     current_kernel_params.griddim_y,
                                     batch_size),
                                dim3(current_kernel_params.blockdim_x,
                                     current_kernel_params.blockdim_y),
                                current_kernel_params.shared_memory,
                                cfg.stream>>>(
                                device_in,
                                reinterpret_cast<
                                    typename std::make_unsigned<T>::type*>(
                                    device_out),
                                root_of_unity_table, modulus,
                                current_kernel_params.shared_index,
                                current_kernel_params.logm,
                                current_kernel_params.k,
                                current_kernel_params.outer_iteration_count,
                                cfg.n_power, cfg.mod_inverse,
                                current_kernel_params.not_last_kernel,
                                (cfg.reduction_poly ==
                                 ReductionPolynomial::X_N_minus),
                                mod_count);
                            GPUNTT_CUDA_CHECK(cudaGetLastError());
                            for (int i = 1;
                                 i < kernel_parameters[cfg.n_power].size() - 1;
                                 i++)
                            {
                                auto& current_kernel_params =
                                    kernel_parameters[cfg.n_power][i];
                                InverseCore<<<
                                    dim3(current_kernel_params.griddim_x,
                                         current_kernel_params.griddim_y,
                                         batch_size),
                                    dim3(current_kernel_params.blockdim_x,
                                         current_kernel_params.blockdim_y),
                                    current_kernel_params.shared_memory,
                                    cfg.stream>>>(
                                    reinterpret_cast<
                                        typename std::make_unsigned<T>::type*>(
                                        device_out),
                                    reinterpret_cast<
                                        typename std::make_unsigned<T>::type*>(
                                        device_out),
                                    root_of_unity_table, modulus,
                                    current_kernel_params.shared_index,
                                    current_kernel_params.logm,
                                    current_kernel_params.k,
                                    current_kernel_params.outer_iteration_count,
                                    cfg.n_power, cfg.mod_inverse,
                                    current_kernel_params.not_last_kernel,
                                    (cfg.reduction_poly ==
                                     ReductionPolynomial::X_N_minus),
                                    mod_count);
                                GPUNTT_CUDA_CHECK(cudaGetLastError());
                            }

                            current_kernel_params = kernel_parameters
                                [cfg.n_power]
                                [kernel_parameters[cfg.n_power].size() - 1];
                            InverseCore<<<dim3(current_kernel_params.griddim_x,
                                               current_kernel_params.griddim_y,
                                               batch_size),
                                          dim3(
                                              current_kernel_params.blockdim_x,
                                              current_kernel_params.blockdim_y),
                                          current_kernel_params.shared_memory,
                                          cfg.stream>>>(
                                reinterpret_cast<
                                    typename std::make_unsigned<T>::type*>(
                                    device_out),
                                device_out, root_of_unity_table, modulus,
                                current_kernel_params.shared_index,
                                current_kernel_params.logm,
                                current_kernel_params.k,
                                current_kernel_params.outer_iteration_count,
                                cfg.n_power, cfg.mod_inverse,
                                current_kernel_params.not_last_kernel,
                                (cfg.reduction_poly ==
                                 ReductionPolynomial::X_N_minus),
                                mod_count);
                            GPUNTT_CUDA_CHECK(cudaGetLastError());
                        }
                        else
                        {
                            auto& current_kernel_params =
                                kernel_parameters[cfg.n_power][0];
                            InverseCore_<<<
                                dim3(current_kernel_params.griddim_x,
                                     current_kernel_params.griddim_y,
                                     batch_size),
                                dim3(current_kernel_params.blockdim_x,
                                     current_kernel_params.blockdim_y),
                                current_kernel_params.shared_memory,
                                cfg.stream>>>(
                                device_in, device_out, root_of_unity_table,
                                modulus, current_kernel_params.shared_index,
                                current_kernel_params.logm,
                                current_kernel_params.k,
                                current_kernel_params.outer_iteration_count,
                                cfg.n_power, cfg.mod_inverse,
                                current_kernel_params.not_last_kernel,
                                (cfg.reduction_poly ==
                                 ReductionPolynomial::X_N_minus),
                                mod_count);
                            GPUNTT_CUDA_CHECK(cudaGetLastError());
                            for (int i = 1;
                                 i < kernel_parameters[cfg.n_power].size(); i++)
                            {
                                auto& current_kernel_params =
                                    kernel_parameters[cfg.n_power][i];
                                InverseCore<<<
                                    dim3(current_kernel_params.griddim_x,
                                         current_kernel_params.griddim_y,
                                         batch_size),
                                    dim3(current_kernel_params.blockdim_x,
                                         current_kernel_params.blockdim_y),
                                    current_kernel_params.shared_memory,
                                    cfg.stream>>>(
                                    device_out, device_out, root_of_unity_table,
                                    modulus, current_kernel_params.shared_index,
                                    current_kernel_params.logm,
                                    current_kernel_params.k,
                                    current_kernel_params.outer_iteration_count,
                                    cfg.n_power, cfg.mod_inverse,
                                    current_kernel_params.not_last_kernel,
                                    (cfg.reduction_poly ==
                                     ReductionPolynomial::X_N_minus),
                                    mod_count);
                                GPUNTT_CUDA_CHECK(cudaGetLastError());
                            }
                        }
                    }
                }
            }
            break;
            case PerCoefficient:
            {
                if ((cfg.n_power <= 0 || cfg.n_power >= 10))
                {
                    throw std::invalid_argument("Invalid n_power range!");
                }

                if ((batch_size & (batch_size - 1)) != 0)
                {
                    throw std::invalid_argument(
                        "batch_size must be a power of two for "
                        "PerCoefficient layout!");
                }

                int log_batch_size = log2(batch_size);
                int total_size = 1 << (cfg.n_power + log_batch_size);
                int total_block_thread = 512;
                int total_block_count = total_size / (total_block_thread * 2);
                int blockdim_y = 1 << (cfg.n_power - 1);
                int blockdim_x = total_block_thread / blockdim_y;
                InverseCoreTranspose<<<
                    dim3(total_block_count, 1, 1),
                    dim3(blockdim_x, blockdim_y, 1),
                    // Data and one root table per column of the block.
                    (total_block_thread * 2 * 2) * sizeof(T),
                    cfg.stream>>>(
                    device_in, device_out, root_of_unity_table, modulus,
                    cfg.mod_inverse, cfg.n_power, log_batch_size,
                    (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                    mod_count);
                GPUNTT_CUDA_CHECK(cudaGetLastError());
            }
            break;
            default:
                throw std::invalid_argument("Invalid ntt_layout!");
                break;
        }
    }

    template <typename T>
    __host__ void GPU_NTT_Inplace(T* device_inout, Root<T>* root_of_unity_table,
                                  Modulus<T> modulus, ntt_configuration<T> cfg,
                                  int batch_size)
    {
        GPU_NTT(device_inout, device_inout, root_of_unity_table, modulus, cfg,
                batch_size);
    }

    template <typename T>
    __host__ void GPU_NTT_Inplace(T* device_inout, Root<T>* root_of_unity_table,
                                  Modulus<T>* modulus,
                                  ntt_rns_configuration<T> cfg, int batch_size,
                                  int mod_count)
    {
        GPU_NTT(device_inout, device_inout, root_of_unity_table, modulus, cfg,
                batch_size, mod_count);
    }

    template <typename T>
    __host__ void GPU_INTT_Inplace(T* device_inout,
                                   Root<T>* root_of_unity_table,
                                   Modulus<T> modulus, ntt_configuration<T> cfg,
                                   int batch_size)
    {
        GPU_INTT(device_inout, device_inout, root_of_unity_table, modulus, cfg,
                 batch_size);
    }

    template <typename T>
    __host__ void
    GPU_INTT_Inplace(T* device_inout, Root<T>* root_of_unity_table,
                     Modulus<T>* modulus, ntt_rns_configuration<T> cfg,
                     int batch_size, int mod_count)
    {
        GPU_INTT(device_inout, device_inout, root_of_unity_table, modulus, cfg,
                 batch_size, mod_count);
    }

    ////////////////////////////////////
    // Modulus Ordered
    ////////////////////////////////////

    template <typename T>
    __global__ void ForwardCoreModulusOrdered(
        T* polynomial_in, T* polynomial_out, Root<T>* root_of_unity_table,
        Modulus<T>* modulus, int shared_index, int logm,
        int outer_iteration_count, int N_power,
        bool not_last_kernel, bool reduction_poly_check, int mod_count,
        int* order)
    {
        ForwardCoreBody<false>(
            polynomial_in, polynomial_out, root_of_unity_table,
            OrderedRnsModulus<T>{modulus, mod_count, order}, SequentialPolys{},
            shared_index, logm, outer_iteration_count, N_power,
            not_last_kernel, reduction_poly_check);
    }

    template <typename T>
    __global__ void ForwardCoreModulusOrdered_(
        T* polynomial_in, T* polynomial_out, Root<T>* root_of_unity_table,
        Modulus<T>* modulus, int shared_index, int logm,
        int outer_iteration_count, int N_power,
        bool not_last_kernel, bool reduction_poly_check, int mod_count,
        int* order)
    {
        ForwardCoreBody<true>(
            polynomial_in, polynomial_out, root_of_unity_table,
            OrderedRnsModulus<T>{modulus, mod_count, order}, SequentialPolys{},
            shared_index, logm, outer_iteration_count, N_power,
            not_last_kernel, reduction_poly_check);
    }

    template <typename T>
    __global__ void InverseCoreModulusOrdered(
        T* polynomial_in, T* polynomial_out,
        Root<T>* inverse_root_of_unity_table, Modulus<T>* modulus,
        int shared_index, int logm, int k, int outer_iteration_count,
        int N_power, Ninverse<T>* n_inverse, bool last_kernel,
        bool reduction_poly_check, int mod_count, int* order)
    {
        InverseCoreBody<false>(
            polynomial_in, polynomial_out, inverse_root_of_unity_table,
            OrderedRnsModulus<T>{modulus, mod_count, order, n_inverse},
            SequentialPolys{},
            shared_index, logm, k, outer_iteration_count, N_power,
            last_kernel, reduction_poly_check);
    }

    template <typename T>
    __global__ void InverseCoreModulusOrdered_(
        T* polynomial_in, T* polynomial_out,
        Root<T>* inverse_root_of_unity_table, Modulus<T>* modulus,
        int shared_index, int logm, int k, int outer_iteration_count,
        int N_power, Ninverse<T>* n_inverse, bool last_kernel,
        bool reduction_poly_check, int mod_count, int* order)
    {
        InverseCoreBody<true>(
            polynomial_in, polynomial_out, inverse_root_of_unity_table,
            OrderedRnsModulus<T>{modulus, mod_count, order, n_inverse},
            SequentialPolys{},
            shared_index, logm, k, outer_iteration_count, N_power,
            last_kernel, reduction_poly_check);
    }

    template <typename T>
    __host__ void
    GPU_NTT_Modulus_Ordered(T* device_in, T* device_out,
                            Root<T>* root_of_unity_table, Modulus<T>* modulus,
                            ntt_rns_configuration<T> cfg, int batch_size,
                            int mod_count, int* order)
    {
        check_batch_size(cfg.n_power, batch_size);

        if ((cfg.n_power <= 9 || cfg.n_power >= 29))
        {
            throw std::invalid_argument("Invalid n_power range!");
        }

        auto kernel_parameters = (cfg.ntt_type == FORWARD)
                                     ? CreateForwardNTTKernel<T>()
                                     : CreateInverseNTTKernel<T>();
        bool standart_kernel = (cfg.n_power < 25) ? true : false;
        T* device_in_ = device_in;

        switch (cfg.ntt_type)
        {
            case FORWARD:
                if (standart_kernel)
                {
                    for (int i = 0; i < kernel_parameters[cfg.n_power].size();
                         i++)
                    {
                        auto& current_kernel_params =
                            kernel_parameters[cfg.n_power][i];
                        ForwardCoreModulusOrdered<<<
                            dim3(current_kernel_params.griddim_x,
                                 current_kernel_params.griddim_y, batch_size),
                            dim3(current_kernel_params.blockdim_x,
                                 current_kernel_params.blockdim_y),
                            current_kernel_params.shared_memory, cfg.stream>>>(
                            device_in_, device_out, root_of_unity_table,
                            modulus, current_kernel_params.shared_index,
                            current_kernel_params.logm,
                            current_kernel_params.outer_iteration_count,
                            cfg.n_power,
                            current_kernel_params.not_last_kernel,
                            (cfg.reduction_poly ==
                             ReductionPolynomial::X_N_minus),
                            mod_count, order);
                        GPUNTT_CUDA_CHECK(cudaGetLastError());
                        device_in_ = device_out;
                    }
                }
                else
                {
                    for (int i = 0;
                         i < kernel_parameters[cfg.n_power].size() - 1; i++)
                    {
                        auto& current_kernel_params =
                            kernel_parameters[cfg.n_power][i];
                        ForwardCoreModulusOrdered<<<
                            dim3(current_kernel_params.griddim_x,
                                 current_kernel_params.griddim_y, batch_size),
                            dim3(current_kernel_params.blockdim_x,
                                 current_kernel_params.blockdim_y),
                            current_kernel_params.shared_memory, cfg.stream>>>(
                            device_in_, device_out, root_of_unity_table,
                            modulus, current_kernel_params.shared_index,
                            current_kernel_params.logm,
                            current_kernel_params.outer_iteration_count,
                            cfg.n_power,
                            current_kernel_params.not_last_kernel,
                            (cfg.reduction_poly ==
                             ReductionPolynomial::X_N_minus),
                            mod_count, order);
                        GPUNTT_CUDA_CHECK(cudaGetLastError());
                        device_in_ = device_out;
                    }
                    auto& current_kernel_params = kernel_parameters
                        [cfg.n_power]
                        [kernel_parameters[cfg.n_power].size() - 1];
                    ForwardCoreModulusOrdered_<<<
                        dim3(current_kernel_params.griddim_x,
                             current_kernel_params.griddim_y, batch_size),
                        dim3(current_kernel_params.blockdim_x,
                             current_kernel_params.blockdim_y),
                        current_kernel_params.shared_memory, cfg.stream>>>(
                        device_in_, device_out, root_of_unity_table, modulus,
                        current_kernel_params.shared_index,
                        current_kernel_params.logm,
                        current_kernel_params.outer_iteration_count,
                        cfg.n_power,
                        current_kernel_params.not_last_kernel,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                        mod_count, order);
                    GPUNTT_CUDA_CHECK(cudaGetLastError());
                }
                break;
            case INVERSE:
                if (standart_kernel)
                {
                    for (int i = 0; i < kernel_parameters[cfg.n_power].size();
                         i++)
                    {
                        auto& current_kernel_params =
                            kernel_parameters[cfg.n_power][i];
                        InverseCoreModulusOrdered<<<
                            dim3(current_kernel_params.griddim_x,
                                 current_kernel_params.griddim_y, batch_size),
                            dim3(current_kernel_params.blockdim_x,
                                 current_kernel_params.blockdim_y),
                            current_kernel_params.shared_memory, cfg.stream>>>(
                            device_in_, device_out, root_of_unity_table,
                            modulus, current_kernel_params.shared_index,
                            current_kernel_params.logm, current_kernel_params.k,
                            current_kernel_params.outer_iteration_count,
                            cfg.n_power, cfg.mod_inverse,
                            current_kernel_params.not_last_kernel,
                            (cfg.reduction_poly ==
                             ReductionPolynomial::X_N_minus),
                            mod_count, order);
                        GPUNTT_CUDA_CHECK(cudaGetLastError());
                        device_in_ = device_out;
                    }
                }
                else
                {
                    auto& current_kernel_params =
                        kernel_parameters[cfg.n_power][0];
                    InverseCoreModulusOrdered_<<<
                        dim3(current_kernel_params.griddim_x,
                             current_kernel_params.griddim_y, batch_size),
                        dim3(current_kernel_params.blockdim_x,
                             current_kernel_params.blockdim_y),
                        current_kernel_params.shared_memory, cfg.stream>>>(
                        device_in_, device_out, root_of_unity_table, modulus,
                        current_kernel_params.shared_index,
                        current_kernel_params.logm, current_kernel_params.k,
                        current_kernel_params.outer_iteration_count,
                        cfg.n_power, cfg.mod_inverse,
                        current_kernel_params.not_last_kernel,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                        mod_count, order);
                    GPUNTT_CUDA_CHECK(cudaGetLastError());
                    device_in_ = device_out;
                    for (int i = 1; i < kernel_parameters[cfg.n_power].size();
                         i++)
                    {
                        auto& current_kernel_params =
                            kernel_parameters[cfg.n_power][i];
                        InverseCoreModulusOrdered<<<
                            dim3(current_kernel_params.griddim_x,
                                 current_kernel_params.griddim_y, batch_size),
                            dim3(current_kernel_params.blockdim_x,
                                 current_kernel_params.blockdim_y),
                            current_kernel_params.shared_memory, cfg.stream>>>(
                            device_in_, device_out, root_of_unity_table,
                            modulus, current_kernel_params.shared_index,
                            current_kernel_params.logm, current_kernel_params.k,
                            current_kernel_params.outer_iteration_count,
                            cfg.n_power, cfg.mod_inverse,
                            current_kernel_params.not_last_kernel,
                            (cfg.reduction_poly ==
                             ReductionPolynomial::X_N_minus),
                            mod_count, order);
                        GPUNTT_CUDA_CHECK(cudaGetLastError());
                    }
                }
                break;
            default:
                throw std::invalid_argument("Invalid ntt_type!");
                break;
        }
    }

    template <typename T>
    __host__ void GPU_NTT_Modulus_Ordered_Inplace(
        T* device_inout, Root<T>* root_of_unity_table, Modulus<T>* modulus,
        ntt_rns_configuration<T> cfg, int batch_size, int mod_count, int* order)
    {
        GPU_NTT_Modulus_Ordered(device_inout, device_inout, root_of_unity_table,
                                modulus, cfg, batch_size, mod_count, order);
    }

    ////////////////////////////////////
    // Poly Ordered
    ////////////////////////////////////

    template <typename T>
    __global__ void
    ForwardCorePolyOrdered(T* polynomial_in, T* polynomial_out,
                           Root<T>* root_of_unity_table, Modulus<T>* modulus,
                           int shared_index, int logm,
                           int outer_iteration_count, int N_power,
                           bool not_last_kernel,
                           bool reduction_poly_check, int mod_count, int* order)
    {
        ForwardCoreBody<false>(
            polynomial_in, polynomial_out, root_of_unity_table,
            RnsModulus<T>{modulus, mod_count}, OrderedPolys{order},
            shared_index, logm, outer_iteration_count, N_power,
            not_last_kernel, reduction_poly_check);
    }

    template <typename T>
    __global__ void ForwardCorePolyOrdered_(
        T* polynomial_in, T* polynomial_out, Root<T>* root_of_unity_table,
        Modulus<T>* modulus, int shared_index, int logm,
        int outer_iteration_count, int N_power,
        bool not_last_kernel, bool reduction_poly_check, int mod_count,
        int* order)
    {
        ForwardCoreBody<true>(
            polynomial_in, polynomial_out, root_of_unity_table,
            RnsModulus<T>{modulus, mod_count}, OrderedPolys{order},
            shared_index, logm, outer_iteration_count, N_power,
            not_last_kernel, reduction_poly_check);
    }

    template <typename T>
    __global__ void
    InverseCorePolyOrdered(T* polynomial_in, T* polynomial_out,
                           Root<T>* inverse_root_of_unity_table,
                           Modulus<T>* modulus, int shared_index, int logm,
                           int k, int outer_iteration_count, int N_power,
                           Ninverse<T>* n_inverse, bool last_kernel,
                           bool reduction_poly_check, int mod_count, int* order)
    {
        InverseCoreBody<false>(
            polynomial_in, polynomial_out, inverse_root_of_unity_table,
            RnsModulus<T>{modulus, mod_count, n_inverse}, OrderedPolys{order},
            shared_index, logm, k, outer_iteration_count, N_power,
            last_kernel, reduction_poly_check);
    }

    template <typename T>
    __global__ void InverseCorePolyOrdered_(
        T* polynomial_in, T* polynomial_out,
        Root<T>* inverse_root_of_unity_table, Modulus<T>* modulus,
        int shared_index, int logm, int k, int outer_iteration_count,
        int N_power, Ninverse<T>* n_inverse, bool last_kernel,
        bool reduction_poly_check, int mod_count, int* order)
    {
        InverseCoreBody<true>(
            polynomial_in, polynomial_out, inverse_root_of_unity_table,
            RnsModulus<T>{modulus, mod_count, n_inverse}, OrderedPolys{order},
            shared_index, logm, k, outer_iteration_count, N_power,
            last_kernel, reduction_poly_check);
    }

    template <typename T>
    __host__ void
    GPU_NTT_Poly_Ordered(T* device_in, T* device_out,
                         Root<T>* root_of_unity_table, Modulus<T>* modulus,
                         ntt_rns_configuration<T> cfg, int batch_size,
                         int mod_count, int* order)
    {
        check_batch_size(cfg.n_power, batch_size);

        if ((cfg.n_power <= 9 || cfg.n_power >= 29))
        {
            throw std::invalid_argument("Invalid n_power range!");
        }

        auto kernel_parameters = (cfg.ntt_type == FORWARD)
                                     ? CreateForwardNTTKernel<T>()
                                     : CreateInverseNTTKernel<T>();
        bool standart_kernel = (cfg.n_power < 25) ? true : false;
        T* device_in_ = device_in;

        switch (cfg.ntt_type)
        {
            case FORWARD:
                if (standart_kernel)
                {
                    for (int i = 0; i < kernel_parameters[cfg.n_power].size();
                         i++)
                    {
                        auto& current_kernel_params =
                            kernel_parameters[cfg.n_power][i];
                        ForwardCorePolyOrdered<<<
                            dim3(current_kernel_params.griddim_x,
                                 current_kernel_params.griddim_y, batch_size),
                            dim3(current_kernel_params.blockdim_x,
                                 current_kernel_params.blockdim_y),
                            current_kernel_params.shared_memory, cfg.stream>>>(
                            device_in_, device_out, root_of_unity_table,
                            modulus, current_kernel_params.shared_index,
                            current_kernel_params.logm,
                            current_kernel_params.outer_iteration_count,
                            cfg.n_power,
                            current_kernel_params.not_last_kernel,
                            (cfg.reduction_poly ==
                             ReductionPolynomial::X_N_minus),
                            mod_count, order);
                        GPUNTT_CUDA_CHECK(cudaGetLastError());
                        device_in_ = device_out;
                    }
                }
                else
                {
                    for (int i = 0;
                         i < kernel_parameters[cfg.n_power].size() - 1; i++)
                    {
                        auto& current_kernel_params =
                            kernel_parameters[cfg.n_power][i];
                        ForwardCorePolyOrdered<<<
                            dim3(current_kernel_params.griddim_x,
                                 current_kernel_params.griddim_y, batch_size),
                            dim3(current_kernel_params.blockdim_x,
                                 current_kernel_params.blockdim_y),
                            current_kernel_params.shared_memory, cfg.stream>>>(
                            device_in_, device_out, root_of_unity_table,
                            modulus, current_kernel_params.shared_index,
                            current_kernel_params.logm,
                            current_kernel_params.outer_iteration_count,
                            cfg.n_power,
                            current_kernel_params.not_last_kernel,
                            (cfg.reduction_poly ==
                             ReductionPolynomial::X_N_minus),
                            mod_count, order);
                        GPUNTT_CUDA_CHECK(cudaGetLastError());
                        device_in_ = device_out;
                    }
                    auto& current_kernel_params = kernel_parameters
                        [cfg.n_power]
                        [kernel_parameters[cfg.n_power].size() - 1];
                    ForwardCorePolyOrdered_<<<
                        dim3(current_kernel_params.griddim_x,
                             current_kernel_params.griddim_y, batch_size),
                        dim3(current_kernel_params.blockdim_x,
                             current_kernel_params.blockdim_y),
                        current_kernel_params.shared_memory, cfg.stream>>>(
                        device_in_, device_out, root_of_unity_table, modulus,
                        current_kernel_params.shared_index,
                        current_kernel_params.logm,
                        current_kernel_params.outer_iteration_count,
                        cfg.n_power,
                        current_kernel_params.not_last_kernel,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                        mod_count, order);
                    GPUNTT_CUDA_CHECK(cudaGetLastError());
                }
                break;
            case INVERSE:
                if (standart_kernel)
                {
                    for (int i = 0; i < kernel_parameters[cfg.n_power].size();
                         i++)
                    {
                        auto& current_kernel_params =
                            kernel_parameters[cfg.n_power][i];
                        InverseCorePolyOrdered<<<
                            dim3(current_kernel_params.griddim_x,
                                 current_kernel_params.griddim_y, batch_size),
                            dim3(current_kernel_params.blockdim_x,
                                 current_kernel_params.blockdim_y),
                            current_kernel_params.shared_memory, cfg.stream>>>(
                            device_in_, device_out, root_of_unity_table,
                            modulus, current_kernel_params.shared_index,
                            current_kernel_params.logm, current_kernel_params.k,
                            current_kernel_params.outer_iteration_count,
                            cfg.n_power, cfg.mod_inverse,
                            current_kernel_params.not_last_kernel,
                            (cfg.reduction_poly ==
                             ReductionPolynomial::X_N_minus),
                            mod_count, order);
                        GPUNTT_CUDA_CHECK(cudaGetLastError());
                        device_in_ = device_out;
                    }
                }
                else
                {
                    auto& current_kernel_params =
                        kernel_parameters[cfg.n_power][0];
                    InverseCorePolyOrdered_<<<
                        dim3(current_kernel_params.griddim_x,
                             current_kernel_params.griddim_y, batch_size),
                        dim3(current_kernel_params.blockdim_x,
                             current_kernel_params.blockdim_y),
                        current_kernel_params.shared_memory, cfg.stream>>>(
                        device_in_, device_out, root_of_unity_table, modulus,
                        current_kernel_params.shared_index,
                        current_kernel_params.logm, current_kernel_params.k,
                        current_kernel_params.outer_iteration_count,
                        cfg.n_power, cfg.mod_inverse,
                        current_kernel_params.not_last_kernel,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                        mod_count, order);
                    GPUNTT_CUDA_CHECK(cudaGetLastError());
                    device_in_ = device_out;
                    for (int i = 1; i < kernel_parameters[cfg.n_power].size();
                         i++)
                    {
                        auto& current_kernel_params =
                            kernel_parameters[cfg.n_power][i];
                        InverseCorePolyOrdered<<<
                            dim3(current_kernel_params.griddim_x,
                                 current_kernel_params.griddim_y, batch_size),
                            dim3(current_kernel_params.blockdim_x,
                                 current_kernel_params.blockdim_y),
                            current_kernel_params.shared_memory, cfg.stream>>>(
                            device_in_, device_out, root_of_unity_table,
                            modulus, current_kernel_params.shared_index,
                            current_kernel_params.logm, current_kernel_params.k,
                            current_kernel_params.outer_iteration_count,
                            cfg.n_power, cfg.mod_inverse,
                            current_kernel_params.not_last_kernel,
                            (cfg.reduction_poly ==
                             ReductionPolynomial::X_N_minus),
                            mod_count, order);
                        GPUNTT_CUDA_CHECK(cudaGetLastError());
                    }
                }

                break;
            default:
                throw std::invalid_argument("Invalid ntt_type!");
                break;
        }
    }

    template <typename T>
    __host__ void GPU_NTT_Poly_Ordered_Inplace(
        T* device_inout, Root<T>* root_of_unity_table, Modulus<T>* modulus,
        ntt_rns_configuration<T> cfg, int batch_size, int mod_count, int* order)
    {
        GPU_NTT_Poly_Ordered(device_inout, device_inout, root_of_unity_table,
                             modulus, cfg, batch_size, mod_count, order);
    }

    ////////////////////////////////////
    // Explicit Template Specializations
    ////////////////////////////////////

    template <> struct ntt_configuration<Data32>
    {
        int n_power;
        type ntt_type;
        NTTLayout ntt_layout;
        ReductionPolynomial reduction_poly;
        Ninverse<Data32> mod_inverse;
        cudaStream_t stream;
    };

    template <> struct ntt_configuration<Data64>
    {
        int n_power;
        type ntt_type;
        NTTLayout ntt_layout;
        ReductionPolynomial reduction_poly;
        Ninverse<Data64> mod_inverse;
        cudaStream_t stream;
    };

    template <> struct ntt_rns_configuration<Data32>
    {
        int n_power;
        type ntt_type;
        NTTLayout ntt_layout;
        ReductionPolynomial reduction_poly;
        Ninverse<Data32>* mod_inverse;
        cudaStream_t stream;
    };

    template <> struct ntt_rns_configuration<Data64>
    {
        int n_power;
        type ntt_type;
        NTTLayout ntt_layout;
        ReductionPolynomial reduction_poly;
        Ninverse<Data64>* mod_inverse;
        cudaStream_t stream;
    };

    template __global__ void ForwardCoreLowRing<Data32>(
        Data32* polynomial_in, Data32* polynomial_out,
        const Root<Data32>* __restrict__ root_of_unity_table,
        Modulus<Data32> modulus, int shared_index, int N_power,
        bool reduction_poly_check, int total_batch);

    template __global__ void ForwardCoreLowRing<Data64>(
        Data64* polynomial_in, Data64* polynomial_out,
        const Root<Data64>* __restrict__ root_of_unity_table,
        Modulus<Data64> modulus, int shared_index, int N_power,
        bool reduction_poly_check, int total_batch);

    template __global__ void ForwardCoreLowRing<Data32s>(
        Data32s* polynomial_in, Data32* polynomial_out,
        const Root<Data32>* __restrict__ root_of_unity_table,
        Modulus<Data32> modulus, int shared_index, int N_power,
        bool reduction_poly_check, int total_batch);

    template __global__ void ForwardCoreLowRing<Data64s>(
        Data64s* polynomial_in, Data64* polynomial_out,
        const Root<Data64>* __restrict__ root_of_unity_table,
        Modulus<Data64> modulus, int shared_index, int N_power,
        bool reduction_poly_check, int total_batch);

    template __global__ void ForwardCoreLowRing<Data32>(
        Data32* polynomial_in, Data32* polynomial_out,
        const Root<Data32>* __restrict__ root_of_unity_table,
        Modulus<Data32>* modulus, int shared_index, int N_power,
        bool reduction_poly_check, int total_batch,
        int mod_count);

    template __global__ void ForwardCoreLowRing<Data64>(
        Data64* polynomial_in, Data64* polynomial_out,
        const Root<Data64>* __restrict__ root_of_unity_table,
        Modulus<Data64>* modulus, int shared_index, int N_power,
        bool reduction_poly_check, int total_batch,
        int mod_count);

    template __global__ void ForwardCoreLowRing<Data32s>(
        Data32s* polynomial_in, Data32* polynomial_out,
        const Root<Data32>* __restrict__ root_of_unity_table,
        Modulus<Data32>* modulus, int shared_index, int N_power,
        bool reduction_poly_check, int total_batch,
        int mod_count);

    template __global__ void ForwardCoreLowRing<Data64s>(
        Data64s* polynomial_in, Data64* polynomial_out,
        const Root<Data64>* __restrict__ root_of_unity_table,
        Modulus<Data64>* modulus, int shared_index, int N_power,
        bool reduction_poly_check, int total_batch,
        int mod_count);

    template __global__ void
    ForwardCore<Data32>(Data32* polynomial_in, Data32* polynomial_out,
                        const Root<Data32>* __restrict__ root_of_unity_table,
                        Modulus<Data32> modulus, int shared_index, int logm,
                        int outer_iteration_count, int N_power,
                        bool not_last_kernel,
                        bool reduction_poly_check);

    template __global__ void
    ForwardCore<Data64>(Data64* polynomial_in, Data64* polynomial_out,
                        const Root<Data64>* __restrict__ root_of_unity_table,
                        Modulus<Data64> modulus, int shared_index, int logm,
                        int outer_iteration_count, int N_power,
                        bool not_last_kernel,
                        bool reduction_poly_check);

    template __global__ void
    ForwardCore<Data32s>(Data32s* polynomial_in, Data32* polynomial_out,
                         const Root<Data32>* __restrict__ root_of_unity_table,
                         Modulus<Data32> modulus, int shared_index, int logm,
                         int outer_iteration_count, int N_power,
                         bool not_last_kernel,
                         bool reduction_poly_check);

    template __global__ void
    ForwardCore<Data64s>(Data64s* polynomial_in, Data64* polynomial_out,
                         const Root<Data64>* __restrict__ root_of_unity_table,
                         Modulus<Data64> modulus, int shared_index, int logm,
                         int outer_iteration_count, int N_power,
                         bool not_last_kernel,
                         bool reduction_poly_check);

    template __global__ void
    ForwardCore<Data32>(Data32* polynomial_in, Data32* polynomial_out,
                        const Root<Data32>* __restrict__ root_of_unity_table,
                        Modulus<Data32>* modulus, int shared_index, int logm,
                        int outer_iteration_count, int N_power,
                        bool not_last_kernel,
                        bool reduction_poly_check, int mod_count);

    template __global__ void
    ForwardCore<Data64>(Data64* polynomial_in, Data64* polynomial_out,
                        const Root<Data64>* __restrict__ root_of_unity_table,
                        Modulus<Data64>* modulus, int shared_index, int logm,
                        int outer_iteration_count, int N_power,
                        bool not_last_kernel,
                        bool reduction_poly_check, int mod_count);

    template __global__ void
    ForwardCore<Data32s>(Data32s* polynomial_in, Data32* polynomial_out,
                         const Root<Data32>* __restrict__ root_of_unity_table,
                         Modulus<Data32>* modulus, int shared_index, int logm,
                         int outer_iteration_count, int N_power,
                         bool not_last_kernel,
                         bool reduction_poly_check, int mod_count);

    template __global__ void
    ForwardCore<Data64s>(Data64s* polynomial_in, Data64* polynomial_out,
                         const Root<Data64>* __restrict__ root_of_unity_table,
                         Modulus<Data64>* modulus, int shared_index, int logm,
                         int outer_iteration_count, int N_power,
                         bool not_last_kernel,
                         bool reduction_poly_check, int mod_count);

    template __global__ void
    ForwardCore_<Data32>(Data32* polynomial_in, Data32* polynomial_out,
                         const Root<Data32>* __restrict__ root_of_unity_table,
                         Modulus<Data32> modulus, int shared_index, int logm,
                         int outer_iteration_count, int N_power,
                         bool not_last_kernel,
                         bool reduction_poly_check);

    template __global__ void
    ForwardCore_<Data64>(Data64* polynomial_in, Data64* polynomial_out,
                         const Root<Data64>* __restrict__ root_of_unity_table,
                         Modulus<Data64> modulus, int shared_index, int logm,
                         int outer_iteration_count, int N_power,
                         bool not_last_kernel,
                         bool reduction_poly_check);

    template __global__ void
    ForwardCore_<Data32s>(Data32s* polynomial_in, Data32* polynomial_out,
                          const Root<Data32>* __restrict__ root_of_unity_table,
                          Modulus<Data32> modulus, int shared_index, int logm,
                          int outer_iteration_count, int N_power,
                          bool not_last_kernel,
                          bool reduction_poly_check);

    template __global__ void
    ForwardCore_<Data64s>(Data64s* polynomial_in, Data64* polynomial_out,
                          const Root<Data64>* __restrict__ root_of_unity_table,
                          Modulus<Data64> modulus, int shared_index, int logm,
                          int outer_iteration_count, int N_power,
                          bool not_last_kernel,
                          bool reduction_poly_check);

    template __global__ void
    ForwardCore_<Data32>(Data32* polynomial_in, Data32* polynomial_out,
                         const Root<Data32>* __restrict__ root_of_unity_table,
                         Modulus<Data32>* modulus, int shared_index, int logm,
                         int outer_iteration_count, int N_power,
                         bool not_last_kernel,
                         bool reduction_poly_check, int mod_count);

    template __global__ void
    ForwardCore_<Data64>(Data64* polynomial_in, Data64* polynomial_out,
                         const Root<Data64>* __restrict__ root_of_unity_table,
                         Modulus<Data64>* modulus, int shared_index, int logm,
                         int outer_iteration_count, int N_power,
                         bool not_last_kernel,
                         bool reduction_poly_check, int mod_count);

    template __global__ void
    ForwardCore_<Data32s>(Data32s* polynomial_in, Data32* polynomial_out,
                          const Root<Data32>* __restrict__ root_of_unity_table,
                          Modulus<Data32>* modulus, int shared_index, int logm,
                          int outer_iteration_count, int N_power,
                          bool not_last_kernel,
                          bool reduction_poly_check, int mod_count);

    template __global__ void
    ForwardCore_<Data64s>(Data64s* polynomial_in, Data64* polynomial_out,
                          const Root<Data64>* __restrict__ root_of_unity_table,
                          Modulus<Data64>* modulus, int shared_index, int logm,
                          int outer_iteration_count, int N_power,
                          bool not_last_kernel,
                          bool reduction_poly_check, int mod_count);

    template __global__ void InverseCoreLowRing<Data32>(
        Data32* polynomial_in, Data32* polynomial_out,
        const Root<Data32>* __restrict__ inverse_root_of_unity_table,
        Modulus<Data32> modulus, int shared_index, int N_power,
        Ninverse<Data32> n_inverse, bool reduction_poly_check, int total_batch);

    template __global__ void InverseCoreLowRing<Data64>(
        Data64* polynomial_in, Data64* polynomial_out,
        const Root<Data64>* __restrict__ inverse_root_of_unity_table,
        Modulus<Data64> modulus, int shared_index, int N_power,
        Ninverse<Data64> n_inverse, bool reduction_poly_check, int total_batch);

    template __global__ void InverseCoreLowRing<Data32s>(
        Data32* polynomial_in, Data32s* polynomial_out,
        const Root<Data32>* __restrict__ inverse_root_of_unity_table,
        Modulus<Data32> modulus, int shared_index, int N_power,
        Ninverse<Data32> n_inverse, bool reduction_poly_check, int total_batch);

    template __global__ void InverseCoreLowRing<Data64s>(
        Data64* polynomial_in, Data64s* polynomial_out,
        const Root<Data64>* __restrict__ inverse_root_of_unity_table,
        Modulus<Data64> modulus, int shared_index, int N_power,
        Ninverse<Data64> n_inverse, bool reduction_poly_check, int total_batch);

    template __global__ void InverseCoreLowRing<Data32>(
        Data32* polynomial_in, Data32* polynomial_out,
        const Root<Data32>* __restrict__ inverse_root_of_unity_table,
        Modulus<Data32>* modulus, int shared_index, int N_power,
        Ninverse<Data32>* n_inverse, bool reduction_poly_check, int total_batch,
        int mod_count);

    template __global__ void InverseCoreLowRing<Data64>(
        Data64* polynomial_in, Data64* polynomial_out,
        const Root<Data64>* __restrict__ inverse_root_of_unity_table,
        Modulus<Data64>* modulus, int shared_index, int N_power,
        Ninverse<Data64>* n_inverse, bool reduction_poly_check, int total_batch,
        int mod_count);

    template __global__ void InverseCoreLowRing<Data32s>(
        Data32* polynomial_in, Data32s* polynomial_out,
        const Root<Data32>* __restrict__ inverse_root_of_unity_table,
        Modulus<Data32>* modulus, int shared_index, int N_power,
        Ninverse<Data32>* n_inverse, bool reduction_poly_check, int total_batch,
        int mod_count);

    template __global__ void InverseCoreLowRing<Data64s>(
        Data64* polynomial_in, Data64s* polynomial_out,
        const Root<Data64>* __restrict__ inverse_root_of_unity_table,
        Modulus<Data64>* modulus, int shared_index, int N_power,
        Ninverse<Data64>* n_inverse, bool reduction_poly_check, int total_batch,
        int mod_count);

    template __global__ void InverseCore<Data32>(
        Data32* polynomial_in, Data32* polynomial_out,
        const Root<Data32>* __restrict__ inverse_root_of_unity_table,
        Modulus<Data32> modulus, int shared_index, int logm, int k,
        int outer_iteration_count, int N_power, Ninverse<Data32> n_inverse,
        bool last_kernel, bool reduction_poly_check);

    template __global__ void InverseCore<Data64>(
        Data64* polynomial_in, Data64* polynomial_out,
        const Root<Data64>* __restrict__ inverse_root_of_unity_table,
        Modulus<Data64> modulus, int shared_index, int logm, int k,
        int outer_iteration_count, int N_power, Ninverse<Data64> n_inverse,
        bool last_kernel, bool reduction_poly_check);

    template __global__ void InverseCore<Data32s>(
        Data32* polynomial_in, Data32s* polynomial_out,
        const Root<Data32>* __restrict__ inverse_root_of_unity_table,
        Modulus<Data32> modulus, int shared_index, int logm, int k,
        int outer_iteration_count, int N_power, Ninverse<Data32> n_inverse,
        bool last_kernel, bool reduction_poly_check);

    template __global__ void InverseCore<Data64s>(
        Data64* polynomial_in, Data64s* polynomial_out,
        const Root<Data64>* __restrict__ inverse_root_of_unity_table,
        Modulus<Data64> modulus, int shared_index, int logm, int k,
        int outer_iteration_count, int N_power, Ninverse<Data64> n_inverse,
        bool last_kernel, bool reduction_poly_check);

    template __global__ void InverseCore<Data32>(
        Data32* polynomial_in, Data32* polynomial_out,
        const Root<Data32>* __restrict__ inverse_root_of_unity_table,
        Modulus<Data32>* modulus, int shared_index, int logm, int k,
        int outer_iteration_count, int N_power, Ninverse<Data32>* n_inverse,
        bool last_kernel, bool reduction_poly_check, int mod_count);

    template __global__ void InverseCore<Data64>(
        Data64* polynomial_in, Data64* polynomial_out,
        const Root<Data64>* __restrict__ inverse_root_of_unity_table,
        Modulus<Data64>* modulus, int shared_index, int logm, int k,
        int outer_iteration_count, int N_power, Ninverse<Data64>* n_inverse,
        bool last_kernel, bool reduction_poly_check, int mod_count);

    template __global__ void InverseCore<Data32s>(
        Data32* polynomial_in, Data32s* polynomial_out,
        const Root<Data32>* __restrict__ inverse_root_of_unity_table,
        Modulus<Data32>* modulus, int shared_index, int logm, int k,
        int outer_iteration_count, int N_power, Ninverse<Data32>* n_inverse,
        bool last_kernel, bool reduction_poly_check, int mod_count);

    template __global__ void InverseCore<Data64s>(
        Data64* polynomial_in, Data64s* polynomial_out,
        const Root<Data64>* __restrict__ inverse_root_of_unity_table,
        Modulus<Data64>* modulus, int shared_index, int logm, int k,
        int outer_iteration_count, int N_power, Ninverse<Data64>* n_inverse,
        bool last_kernel, bool reduction_poly_check, int mod_count);

    template __global__ void InverseCore_<Data32>(
        Data32* polynomial_in, Data32* polynomial_out,
        const Root<Data32>* __restrict__ inverse_root_of_unity_table,
        Modulus<Data32> modulus, int shared_index, int logm, int k,
        int outer_iteration_count, int N_power, Ninverse<Data32> n_inverse,
        bool last_kernel, bool reduction_poly_check);

    template __global__ void InverseCore_<Data64>(
        Data64* polynomial_in, Data64* polynomial_out,
        const Root<Data64>* __restrict__ inverse_root_of_unity_table,
        Modulus<Data64> modulus, int shared_index, int logm, int k,
        int outer_iteration_count, int N_power, Ninverse<Data64> n_inverse,
        bool last_kernel, bool reduction_poly_check);

    template __global__ void InverseCore_<Data32s>(
        Data32* polynomial_in, Data32s* polynomial_out,
        const Root<Data32>* __restrict__ inverse_root_of_unity_table,
        Modulus<Data32> modulus, int shared_index, int logm, int k,
        int outer_iteration_count, int N_power, Ninverse<Data32> n_inverse,
        bool last_kernel, bool reduction_poly_check);

    template __global__ void InverseCore_<Data64s>(
        Data64* polynomial_in, Data64s* polynomial_out,
        const Root<Data64>* __restrict__ inverse_root_of_unity_table,
        Modulus<Data64> modulus, int shared_index, int logm, int k,
        int outer_iteration_count, int N_power, Ninverse<Data64> n_inverse,
        bool last_kernel, bool reduction_poly_check);

    template __global__ void InverseCore_<Data32>(
        Data32* polynomial_in, Data32* polynomial_out,
        const Root<Data32>* __restrict__ inverse_root_of_unity_table,
        Modulus<Data32>* modulus, int shared_index, int logm, int k,
        int outer_iteration_count, int N_power, Ninverse<Data32>* n_inverse,
        bool last_kernel, bool reduction_poly_check, int mod_count);

    template __global__ void InverseCore_<Data64>(
        Data64* polynomial_in, Data64* polynomial_out,
        const Root<Data64>* __restrict__ inverse_root_of_unity_table,
        Modulus<Data64>* modulus, int shared_index, int logm, int k,
        int outer_iteration_count, int N_power, Ninverse<Data64>* n_inverse,
        bool last_kernel, bool reduction_poly_check, int mod_count);

    template __global__ void InverseCore_<Data32s>(
        Data32* polynomial_in, Data32s* polynomial_out,
        const Root<Data32>* __restrict__ inverse_root_of_unity_table,
        Modulus<Data32>* modulus, int shared_index, int logm, int k,
        int outer_iteration_count, int N_power, Ninverse<Data32>* n_inverse,
        bool last_kernel, bool reduction_poly_check, int mod_count);

    template __global__ void InverseCore_<Data64s>(
        Data64* polynomial_in, Data64s* polynomial_out,
        const Root<Data64>* __restrict__ inverse_root_of_unity_table,
        Modulus<Data64>* modulus, int shared_index, int logm, int k,
        int outer_iteration_count, int N_power, Ninverse<Data64>* n_inverse,
        bool last_kernel, bool reduction_poly_check, int mod_count);

    template __global__ void ForwardCoreTranspose<Data32>(
        Data32* polynomial_in, Data32* polynomial_out,
        const Root<Data32>* __restrict__ root_of_unity_table,
        const Modulus<Data32> modulus, int log_row, int log_column,
        bool reduction_poly_check);

    template __global__ void ForwardCoreTranspose<Data64>(
        Data64* polynomial_in, Data64* polynomial_out,
        const Root<Data64>* __restrict__ root_of_unity_table,
        const Modulus<Data64> modulus, int log_row, int log_column,
        bool reduction_poly_check);

    template __global__ void ForwardCoreTranspose<Data32s>(
        Data32s* polynomial_in, Data32* polynomial_out,
        const Root<Data32>* __restrict__ root_of_unity_table,
        const Modulus<Data32> modulus, int log_row, int log_column,
        bool reduction_poly_check);

    template __global__ void ForwardCoreTranspose<Data64s>(
        Data64s* polynomial_in, Data64* polynomial_out,
        const Root<Data64>* __restrict__ root_of_unity_table,
        const Modulus<Data64> modulus, int log_row, int log_column,
        bool reduction_poly_check);

    template __global__ void ForwardCoreTranspose<Data32>(
        Data32* polynomial_in, Data32* polynomial_out,
        const Root<Data32>* __restrict__ root_of_unity_table,
        const Modulus<Data32>* modulus, int log_row, int log_column,
        bool reduction_poly_check, int mod_count);

    template __global__ void ForwardCoreTranspose<Data64>(
        Data64* polynomial_in, Data64* polynomial_out,
        const Root<Data64>* __restrict__ root_of_unity_table,
        const Modulus<Data64>* modulus, int log_row, int log_column,
        bool reduction_poly_check, int mod_count);

    template __global__ void ForwardCoreTranspose<Data32s>(
        Data32s* polynomial_in, Data32* polynomial_out,
        const Root<Data32>* __restrict__ root_of_unity_table,
        const Modulus<Data32>* modulus, int log_row, int log_column,
        bool reduction_poly_check, int mod_count);

    template __global__ void ForwardCoreTranspose<Data64s>(
        Data64s* polynomial_in, Data64* polynomial_out,
        const Root<Data64>* __restrict__ root_of_unity_table,
        const Modulus<Data64>* modulus, int log_row, int log_column,
        bool reduction_poly_check, int mod_count);

    template __global__ void InverseCoreTranspose<Data32>(
        Data32* polynomial_in, Data32* polynomial_out,
        const Root<Data32>* __restrict__ inverse_root_of_unity_table,
        Modulus<Data32> modulus, Ninverse<Data32> n_inverse, int log_row,
        int log_column, bool reduction_poly_check);

    template __global__ void InverseCoreTranspose<Data64>(
        Data64* polynomial_in, Data64* polynomial_out,
        const Root<Data64>* __restrict__ inverse_root_of_unity_table,
        Modulus<Data64> modulus, Ninverse<Data64> n_inverse, int log_row,
        int log_column, bool reduction_poly_check);

    template __global__ void InverseCoreTranspose<Data32s>(
        Data32* polynomial_in, Data32s* polynomial_out,
        const Root<Data32>* __restrict__ inverse_root_of_unity_table,
        Modulus<Data32> modulus, Ninverse<Data32> n_inverse, int log_row,
        int log_column, bool reduction_poly_check);

    template __global__ void InverseCoreTranspose<Data64s>(
        Data64* polynomial_in, Data64s* polynomial_out,
        const Root<Data64>* __restrict__ inverse_root_of_unity_table,
        Modulus<Data64> modulus, Ninverse<Data64> n_inverse, int log_row,
        int log_column, bool reduction_poly_check);

    template __global__ void InverseCoreTranspose<Data32>(
        Data32* polynomial_in, Data32* polynomial_out,
        const Root<Data32>* __restrict__ inverse_root_of_unity_table,
        Modulus<Data32>* modulus, Ninverse<Data32>* n_inverse, int log_row,
        int log_column, bool reduction_poly_check, int mod_count);

    template __global__ void InverseCoreTranspose<Data64>(
        Data64* polynomial_in, Data64* polynomial_out,
        const Root<Data64>* __restrict__ inverse_root_of_unity_table,
        Modulus<Data64>* modulus, Ninverse<Data64>* n_inverse, int log_row,
        int log_column, bool reduction_poly_check, int mod_count);

    template __global__ void InverseCoreTranspose<Data32s>(
        Data32* polynomial_in, Data32s* polynomial_out,
        const Root<Data32>* __restrict__ inverse_root_of_unity_table,
        Modulus<Data32>* modulus, Ninverse<Data32>* n_inverse, int log_row,
        int log_column, bool reduction_poly_check, int mod_count);

    template __global__ void InverseCoreTranspose<Data64s>(
        Data64* polynomial_in, Data64s* polynomial_out,
        const Root<Data64>* __restrict__ inverse_root_of_unity_table,
        Modulus<Data64>* modulus, Ninverse<Data64>* n_inverse, int log_row,
        int log_column, bool reduction_poly_check, int mod_count);

    template __host__ void
    GPU_NTT<Data32>(Data32* device_in, Data32* device_out,
                    Root<Data32>* root_of_unity_table, Modulus<Data32> modulus,
                    ntt_configuration<Data32> cfg, int batch_size);

    template __host__ void
    GPU_NTT<Data64>(Data64* device_in, Data64* device_out,
                    Root<Data64>* root_of_unity_table, Modulus<Data64> modulus,
                    ntt_configuration<Data64> cfg, int batch_size);

    template __host__ void
    GPU_NTT<Data32s>(Data32s* device_in, Data32* device_out,
                     Root<Data32>* root_of_unity_table, Modulus<Data32> modulus,
                     ntt_configuration<Data32> cfg, int batch_size);

    template __host__ void
    GPU_NTT<Data64s>(Data64s* device_in, Data64* device_out,
                     Root<Data64>* root_of_unity_table, Modulus<Data64> modulus,
                     ntt_configuration<Data64> cfg, int batch_size);

    template __host__ void
    GPU_INTT<Data32>(Data32* device_in, Data32* device_out,
                     Root<Data32>* root_of_unity_table, Modulus<Data32> modulus,
                     ntt_configuration<Data32> cfg, int batch_size);

    template __host__ void
    GPU_INTT<Data64>(Data64* device_in, Data64* device_out,
                     Root<Data64>* root_of_unity_table, Modulus<Data64> modulus,
                     ntt_configuration<Data64> cfg, int batch_size);

    template __host__ void GPU_INTT<Data32s>(Data32* device_in,
                                             Data32s* device_out,
                                             Root<Data32>* root_of_unity_table,
                                             Modulus<Data32> modulus,
                                             ntt_configuration<Data32> cfg,
                                             int batch_size);

    template __host__ void GPU_INTT<Data64s>(Data64* device_in,
                                             Data64s* device_out,
                                             Root<Data64>* root_of_unity_table,
                                             Modulus<Data64> modulus,
                                             ntt_configuration<Data64> cfg,
                                             int batch_size);

    template __host__ void GPU_NTT_Inplace<Data32>(
        Data32* device_inout, Root<Data32>* root_of_unity_table,
        Modulus<Data32> modulus, ntt_configuration<Data32> cfg, int batch_size);

    template __host__ void GPU_NTT_Inplace<Data64>(
        Data64* device_inout, Root<Data64>* root_of_unity_table,
        Modulus<Data64> modulus, ntt_configuration<Data64> cfg, int batch_size);

    template __host__ void GPU_INTT_Inplace<Data32>(
        Data32* device_inout, Root<Data32>* root_of_unity_table,
        Modulus<Data32> modulus, ntt_configuration<Data32> cfg, int batch_size);

    template __host__ void GPU_INTT_Inplace<Data64>(
        Data64* device_inout, Root<Data64>* root_of_unity_table,
        Modulus<Data64> modulus, ntt_configuration<Data64> cfg, int batch_size);

    template __host__ void GPU_NTT<Data32>(Data32* device_in,
                                           Data32* device_out,
                                           Root<Data32>* root_of_unity_table,
                                           Modulus<Data32>* modulus,
                                           ntt_rns_configuration<Data32> cfg,
                                           int batch_size, int mod_count);

    template __host__ void GPU_NTT<Data64>(Data64* device_in,
                                           Data64* device_out,
                                           Root<Data64>* root_of_unity_table,
                                           Modulus<Data64>* modulus,
                                           ntt_rns_configuration<Data64> cfg,
                                           int batch_size, int mod_count);

    template __host__ void GPU_NTT<Data32s>(Data32s* device_in,
                                            Data32* device_out,
                                            Root<Data32>* root_of_unity_table,
                                            Modulus<Data32>* modulus,
                                            ntt_rns_configuration<Data32> cfg,
                                            int batch_size, int mod_count);

    template __host__ void GPU_NTT<Data64s>(Data64s* device_in,
                                            Data64* device_out,
                                            Root<Data64>* root_of_unity_table,
                                            Modulus<Data64>* modulus,
                                            ntt_rns_configuration<Data64> cfg,
                                            int batch_size, int mod_count);

    template __host__ void GPU_INTT<Data32>(Data32* device_in,
                                            Data32* device_out,
                                            Root<Data32>* root_of_unity_table,
                                            Modulus<Data32>* modulus,
                                            ntt_rns_configuration<Data32> cfg,
                                            int batch_size, int mod_count);

    template __host__ void GPU_INTT<Data64>(Data64* device_in,
                                            Data64* device_out,
                                            Root<Data64>* root_of_unity_table,
                                            Modulus<Data64>* modulus,
                                            ntt_rns_configuration<Data64> cfg,
                                            int batch_size, int mod_count);

    template __host__ void GPU_INTT<Data32s>(Data32* device_in,
                                             Data32s* device_out,
                                             Root<Data32>* root_of_unity_table,
                                             Modulus<Data32>* modulus,
                                             ntt_rns_configuration<Data32> cfg,
                                             int batch_size, int mod_count);

    template __host__ void GPU_INTT<Data64s>(Data64* device_in,
                                             Data64s* device_out,
                                             Root<Data64>* root_of_unity_table,
                                             Modulus<Data64>* modulus,
                                             ntt_rns_configuration<Data64> cfg,
                                             int batch_size, int mod_count);

    template __host__ void GPU_NTT_Inplace<Data32>(
        Data32* device_inout, Root<Data32>* root_of_unity_table,
        Modulus<Data32>* modulus, ntt_rns_configuration<Data32> cfg,
        int batch_size, int mod_count);

    template __host__ void GPU_NTT_Inplace<Data64>(
        Data64* device_inout, Root<Data64>* root_of_unity_table,
        Modulus<Data64>* modulus, ntt_rns_configuration<Data64> cfg,
        int batch_size, int mod_count);

    template __host__ void GPU_INTT_Inplace<Data32>(
        Data32* device_inout, Root<Data32>* root_of_unity_table,
        Modulus<Data32>* modulus, ntt_rns_configuration<Data32> cfg,
        int batch_size, int mod_count);

    template __host__ void GPU_INTT_Inplace<Data64>(
        Data64* device_inout, Root<Data64>* root_of_unity_table,
        Modulus<Data64>* modulus, ntt_rns_configuration<Data64> cfg,
        int batch_size, int mod_count);

    template __global__ void ForwardCoreModulusOrdered<Data32>(
        Data32* polynomial_in, Data32* polynomial_out,
        Root<Data32>* root_of_unity_table, Modulus<Data32>* modulus,
        int shared_index, int logm, int outer_iteration_count, int N_power,
        bool not_last_kernel, bool reduction_poly_check,
        int mod_count, int* order);

    template __global__ void ForwardCoreModulusOrdered_<Data32>(
        Data32* polynomial_in, Data32* polynomial_out,
        Root<Data32>* root_of_unity_table, Modulus<Data32>* modulus,
        int shared_index, int logm, int outer_iteration_count, int N_power,
        bool not_last_kernel, bool reduction_poly_check,
        int mod_count, int* order);

    template __global__ void ForwardCoreModulusOrdered<Data64>(
        Data64* polynomial_in, Data64* polynomial_out,
        Root<Data64>* root_of_unity_table, Modulus<Data64>* modulus,
        int shared_index, int logm, int outer_iteration_count, int N_power,
        bool not_last_kernel, bool reduction_poly_check,
        int mod_count, int* order);

    template __global__ void ForwardCoreModulusOrdered_<Data64>(
        Data64* polynomial_in, Data64* polynomial_out,
        Root<Data64>* root_of_unity_table, Modulus<Data64>* modulus,
        int shared_index, int logm, int outer_iteration_count, int N_power,
        bool not_last_kernel, bool reduction_poly_check,
        int mod_count, int* order);

    template __global__ void InverseCoreModulusOrdered<Data32>(
        Data32* polynomial_in, Data32* polynomial_out,
        Root<Data32>* inverse_root_of_unity_table, Modulus<Data32>* modulus,
        int shared_index, int logm, int k, int outer_iteration_count,
        int N_power, Ninverse<Data32>* n_inverse, bool last_kernel,
        bool reduction_poly_check, int mod_count, int* order);

    template __global__ void InverseCoreModulusOrdered_<Data32>(
        Data32* polynomial_in, Data32* polynomial_out,
        Root<Data32>* inverse_root_of_unity_table, Modulus<Data32>* modulus,
        int shared_index, int logm, int k, int outer_iteration_count,
        int N_power, Ninverse<Data32>* n_inverse, bool last_kernel,
        bool reduction_poly_check, int mod_count, int* order);

    template __global__ void InverseCoreModulusOrdered<Data64>(
        Data64* polynomial_in, Data64* polynomial_out,
        Root<Data64>* inverse_root_of_unity_table, Modulus<Data64>* modulus,
        int shared_index, int logm, int k, int outer_iteration_count,
        int N_power, Ninverse<Data64>* n_inverse, bool last_kernel,
        bool reduction_poly_check, int mod_count, int* order);

    template __global__ void InverseCoreModulusOrdered_<Data64>(
        Data64* polynomial_in, Data64* polynomial_out,
        Root<Data64>* inverse_root_of_unity_table, Modulus<Data64>* modulus,
        int shared_index, int logm, int k, int outer_iteration_count,
        int N_power, Ninverse<Data64>* n_inverse, bool last_kernel,
        bool reduction_poly_check, int mod_count, int* order);

    template __host__ void
    GPU_NTT_Modulus_Ordered<Data32>(Data32* device_in, Data32* device_out,
                                    Root<Data32>* root_of_unity_table,
                                    Modulus<Data32>* modulus,
                                    ntt_rns_configuration<Data32> cfg,
                                    int batch_size, int mod_count, int* order);

    template __host__ void GPU_NTT_Modulus_Ordered_Inplace<Data32>(
        Data32* device_inout, Root<Data32>* root_of_unity_table,
        Modulus<Data32>* modulus, ntt_rns_configuration<Data32> cfg,
        int batch_size, int mod_count, int* order);

    template __host__ void
    GPU_NTT_Modulus_Ordered<Data64>(Data64* device_in, Data64* device_out,
                                    Root<Data64>* root_of_unity_table,
                                    Modulus<Data64>* modulus,
                                    ntt_rns_configuration<Data64> cfg,
                                    int batch_size, int mod_count, int* order);

    template __host__ void GPU_NTT_Modulus_Ordered_Inplace<Data64>(
        Data64* device_inout, Root<Data64>* root_of_unity_table,
        Modulus<Data64>* modulus, ntt_rns_configuration<Data64> cfg,
        int batch_size, int mod_count, int* order);

    template __global__ void ForwardCorePolyOrdered<Data32>(
        Data32* polynomial_in, Data32* polynomial_out,
        Root<Data32>* root_of_unity_table, Modulus<Data32>* modulus,
        int shared_index, int logm, int outer_iteration_count, int N_power,
        bool not_last_kernel, bool reduction_poly_check,
        int mod_count, int* order);

    template __global__ void ForwardCorePolyOrdered_<Data32>(
        Data32* polynomial_in, Data32* polynomial_out,
        Root<Data32>* root_of_unity_table, Modulus<Data32>* modulus,
        int shared_index, int logm, int outer_iteration_count, int N_power,
        bool not_last_kernel, bool reduction_poly_check,
        int mod_count, int* order);

    template __global__ void ForwardCorePolyOrdered<Data64>(
        Data64* polynomial_in, Data64* polynomial_out,
        Root<Data64>* root_of_unity_table, Modulus<Data64>* modulus,
        int shared_index, int logm, int outer_iteration_count, int N_power,
        bool not_last_kernel, bool reduction_poly_check,
        int mod_count, int* order);

    template __global__ void ForwardCorePolyOrdered_<Data64>(
        Data64* polynomial_in, Data64* polynomial_out,
        Root<Data64>* root_of_unity_table, Modulus<Data64>* modulus,
        int shared_index, int logm, int outer_iteration_count, int N_power,
        bool not_last_kernel, bool reduction_poly_check,
        int mod_count, int* order);

    template __global__ void InverseCorePolyOrdered<Data32>(
        Data32* polynomial_in, Data32* polynomial_out,
        Root<Data32>* inverse_root_of_unity_table, Modulus<Data32>* modulus,
        int shared_index, int logm, int k, int outer_iteration_count,
        int N_power, Ninverse<Data32>* n_inverse, bool last_kernel,
        bool reduction_poly_check, int mod_count, int* order);

    template __global__ void InverseCorePolyOrdered_<Data32>(
        Data32* polynomial_in, Data32* polynomial_out,
        Root<Data32>* inverse_root_of_unity_table, Modulus<Data32>* modulus,
        int shared_index, int logm, int k, int outer_iteration_count,
        int N_power, Ninverse<Data32>* n_inverse, bool last_kernel,
        bool reduction_poly_check, int mod_count, int* order);

    template __global__ void InverseCorePolyOrdered<Data64>(
        Data64* polynomial_in, Data64* polynomial_out,
        Root<Data64>* inverse_root_of_unity_table, Modulus<Data64>* modulus,
        int shared_index, int logm, int k, int outer_iteration_count,
        int N_power, Ninverse<Data64>* n_inverse, bool last_kernel,
        bool reduction_poly_check, int mod_count, int* order);

    template __global__ void InverseCorePolyOrdered_<Data64>(
        Data64* polynomial_in, Data64* polynomial_out,
        Root<Data64>* inverse_root_of_unity_table, Modulus<Data64>* modulus,
        int shared_index, int logm, int k, int outer_iteration_count,
        int N_power, Ninverse<Data64>* n_inverse, bool last_kernel,
        bool reduction_poly_check, int mod_count, int* order);

    template __host__ void
    GPU_NTT_Poly_Ordered<Data32>(Data32* device_in, Data32* device_out,
                                 Root<Data32>* root_of_unity_table,
                                 Modulus<Data32>* modulus,
                                 ntt_rns_configuration<Data32> cfg,
                                 int batch_size, int mod_count, int* order);

    template __host__ void GPU_NTT_Poly_Ordered_Inplace<Data32>(
        Data32* device_inout, Root<Data32>* root_of_unity_table,
        Modulus<Data32>* modulus, ntt_rns_configuration<Data32> cfg,
        int batch_size, int mod_count, int* order);

    template __host__ void
    GPU_NTT_Poly_Ordered<Data64>(Data64* device_in, Data64* device_out,
                                 Root<Data64>* root_of_unity_table,
                                 Modulus<Data64>* modulus,
                                 ntt_rns_configuration<Data64> cfg,
                                 int batch_size, int mod_count, int* order);

    template __host__ void GPU_NTT_Poly_Ordered_Inplace<Data64>(
        Data64* device_inout, Root<Data64>* root_of_unity_table,
        Modulus<Data64>* modulus, ntt_rns_configuration<Data64> cfg,
        int batch_size, int mod_count, int* order);

} // namespace gpuntt
