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
              {128, 4096, 4, 64, 512 * sizeof(T), 8, 12, 0, 7, true},
              {524288, 1, 256, 1, 512 * sizeof(T), 8, 19, 0, 9, false}}}
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

    // The low ring size kernels transform blockDim.y polynomials per block;
    // thread (x, y) owns coefficients x and x + N/2 of polynomial
    // blockIdx.x * blockDim.y + y. The transform is split into load, execute
    // and store steps that share the thread's state, so that the modulus is
    // read from global memory only once. The single modulus and RNS variants
    // differ only in the modulus policy.
    template <typename TU> struct LowRingThread
    {
        bool active_batch;
        int prime_index;
        Modulus<TU> modulus;
        Ninverse<TU> n_inverse;
        location_t global_address;
        int half_n;
        int row_base;
        int shared_address0;
        int shared_address1;
    };

    // n_inverse is only read when WithNinverse is set (inverse transform).
    template <bool WithNinverse, typename TU, typename ModulusPolicy>
    __device__ __forceinline__ LowRingThread<TU>
    LowRingThreadState(ModulusPolicy modulus_policy, int N_power,
                       int total_batch)
    {
        const int idx_x = threadIdx.x;
        const int idx_y = threadIdx.y;
        const int block_x = blockIdx.x;
        const int batch_index = (block_x * blockDim.y) + idx_y;

        const bool active_batch = (batch_index < total_batch);
        const int batch_index_safe = active_batch ? batch_index : 0;

        const int prime_index = modulus_policy.prime_index(batch_index_safe);
        const Modulus<TU> modulus = modulus_policy.get_modulus(prime_index);
        Ninverse<TU> n_inverse = 0;
        if constexpr (WithNinverse)
        {
            n_inverse = modulus_policy.get_n_inverse(prime_index);
        }

        const int half_n = 1 << (N_power - 1);
        const int row_base = idx_y << N_power;
        const int shared_address0 = row_base + idx_x;
        const location_t global_address =
            idx_x + (location_t) (batch_index_safe << N_power);

        // Modulus has no device default constructor, so the state is built
        // in one aggregate initialization.
        return LowRingThread<TU>{
            active_batch, prime_index,     modulus,
            n_inverse,    global_address,  half_n,
            row_base,     shared_address0, shared_address0 + half_n};
    }

    // Copies the thread's two coefficients to shared memory; signed input is
    // reduced to [0, q).
    template <typename T>
    __device__ __forceinline__ void LowRingLoad(
        const LowRingThread<typename std::make_unsigned<T>::type>& thread,
        typename std::make_unsigned<T>::type* shared_memory, T* polynomial_in)
    {
        using TU = typename std::make_unsigned<T>::type;

        if constexpr (std::is_signed<T>::value)
        {
            T input1_reg = polynomial_in[thread.global_address];
            T input2_reg = polynomial_in[thread.global_address + thread.half_n];
            shared_memory[thread.shared_address0] =
                OPERATOR_GPU<TU>::reduce(input1_reg, thread.modulus);
            shared_memory[thread.shared_address1] =
                OPERATOR_GPU<TU>::reduce(input2_reg, thread.modulus);
        }
        else
        {
            shared_memory[thread.shared_address0] =
                polynomial_in[thread.global_address];
            shared_memory[thread.shared_address1] =
                polynomial_in[thread.global_address + thread.half_n];
        }
    }

    // Forward NTT of the block's polynomials in shared memory.
    template <typename TU>
    __device__ __forceinline__ void
    ForwardLowRingExecute(const LowRingThread<TU>& thread, TU* shared_memory,
                          const Root<TU>* root_of_unity_table, int shared_index,
                          int N_power, bool reduction_poly_check)
    {
        const int idx_x = threadIdx.x;

        int t_2 = N_power - 1;
        int offset = thread.row_base;
        int t_ = shared_index;
        int m = 1;

        location_t omega_address = idx_x;
        int shared_address = idx_x;

        int t = 1 << t_;
        int in_shared_address = ((shared_address >> t_) << t_) + shared_address;
        location_t current_root_index;
        __syncthreads();

#pragma unroll
        for (int lp = 0; lp < (shared_index + 1); lp++)
        {
            int group_in_shared_address = in_shared_address + offset;
            if (reduction_poly_check)
            { // X_N_minus
                current_root_index =
                    (omega_address >> t_2) +
                    (location_t) (thread.prime_index << N_power);
            }
            else
            { // X_N_plus
                current_root_index =
                    m + (omega_address >> t_2) +
                    (location_t) (thread.prime_index << N_power);
            }

            CooleyTukeyUnit(shared_memory[group_in_shared_address],
                            shared_memory[group_in_shared_address + t],
                            root_of_unity_table[current_root_index],
                            thread.modulus);

            t = t >> 1;
            t_2 -= 1;
            t_ -= 1;
            m <<= 1;
            if (lp + 1 < (shared_index + 1))
            {
                in_shared_address =
                    ((shared_address >> t_) << t_) + shared_address;
            }
            __syncthreads();
        }
    }

    // Inverse NTT of the block's polynomials in shared memory, without the
    // scaling by N^-1 (done by InverseLowRingStore).
    template <typename TU>
    __device__ __forceinline__ void
    InverseLowRingExecute(const LowRingThread<TU>& thread, TU* shared_memory,
                          const Root<TU>* inverse_root_of_unity_table,
                          int N_power, bool reduction_poly_check)
    {
        const int idx_x = threadIdx.x;

        int t_2 = 0;
        int t_ = 0;
        int offset = thread.row_base;
        int loops = N_power;
        int m = (int) 1 << (N_power - 1);

        location_t omega_address = idx_x;
        int shared_address = idx_x;

        int t = 1 << t_;
        int in_shared_address = ((shared_address >> t_) << t_) + shared_address;
        location_t current_root_index;
        __syncthreads();

#pragma unroll
        for (int lp = 0; lp < loops; lp++)
        {
            int group_in_shared_address = in_shared_address + offset;
            if (reduction_poly_check)
            { // X_N_minus
                current_root_index =
                    (omega_address >> t_2) +
                    (location_t) (thread.prime_index << N_power);
            }
            else
            { // X_N_plus
                current_root_index =
                    m + (omega_address >> t_2) +
                    (location_t) (thread.prime_index << N_power);
            }

            GentlemanSandeUnit(shared_memory[group_in_shared_address],
                               shared_memory[group_in_shared_address + t],
                               inverse_root_of_unity_table[current_root_index],
                               thread.modulus);

            t = t << 1;
            t_2 += 1;
            t_ += 1;
            m >>= 1;
            if (lp + 1 < loops)
            {
                in_shared_address =
                    ((shared_address >> t_) << t_) + shared_address;
            }
            __syncthreads();
        }
    }

    // Copies the thread's two coefficients back to global memory.
    template <typename TU>
    __device__ __forceinline__ void
    ForwardLowRingStore(const LowRingThread<TU>& thread, TU* shared_memory,
                        TU* polynomial_out)
    {
        __syncthreads();

        if (thread.active_batch)
        {
            polynomial_out[thread.global_address] =
                shared_memory[thread.shared_address0];
            polynomial_out[thread.global_address + thread.half_n] =
                shared_memory[thread.shared_address1];
        }
    }

    // Scales the thread's two coefficients by N^-1 and writes them to global
    // memory; signed output is mapped to the centered range.
    template <typename T>
    __device__ __forceinline__ void InverseLowRingStore(
        const LowRingThread<typename std::make_unsigned<T>::type>& thread,
        typename std::make_unsigned<T>::type* shared_memory, T* polynomial_out)
    {
        using TU = typename std::make_unsigned<T>::type;

        __syncthreads();

        TU output1_reg =
            OPERATOR_GPU<TU>::mult(shared_memory[thread.shared_address0],
                                   thread.n_inverse, thread.modulus);
        TU output2_reg =
            OPERATOR_GPU<TU>::mult(shared_memory[thread.shared_address1],
                                   thread.n_inverse, thread.modulus);

        if (thread.active_batch)
        {
            if constexpr (std::is_signed<T>::value)
            {
                polynomial_out[thread.global_address] =
                    OPERATOR_GPU<TU>::centered_reduction(output1_reg,
                                                         thread.modulus);
                polynomial_out[thread.global_address + thread.half_n] =
                    OPERATOR_GPU<TU>::centered_reduction(output2_reg,
                                                         thread.modulus);
            }
            else
            {
                polynomial_out[thread.global_address] = output1_reg;
                polynomial_out[thread.global_address + thread.half_n] =
                    output2_reg;
            }
        }
    }

    template <typename T, typename ModulusPolicy>
    __device__ __forceinline__ void ForwardCoreLowRingBody(
        T* polynomial_in, typename std::make_unsigned<T>::type* polynomial_out,
        const Root<typename std::make_unsigned<T>::type>* root_of_unity_table,
        ModulusPolicy modulus_policy, int shared_index, int N_power,
        bool reduction_poly_check, int total_batch)
    {
        using TU = typename std::make_unsigned<T>::type;

        extern __shared__ char shared_memory_typed[];
        TU* shared_memory = reinterpret_cast<TU*>(shared_memory_typed);

        const LowRingThread<TU> thread =
            LowRingThreadState<false, TU>(modulus_policy, N_power, total_batch);
        LowRingLoad(thread, shared_memory, polynomial_in);
        ForwardLowRingExecute(thread, shared_memory, root_of_unity_table,
                              shared_index, N_power, reduction_poly_check);
        ForwardLowRingStore(thread, shared_memory, polynomial_out);
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

        extern __shared__ char shared_memory_typed[];
        TU* shared_memory = reinterpret_cast<TU*>(shared_memory_typed);

        const LowRingThread<TU> thread =
            LowRingThreadState<true, TU>(modulus_policy, N_power, total_batch);
        LowRingLoad(thread, shared_memory, polynomial_in);
        InverseLowRingExecute(thread, shared_memory,
                              inverse_root_of_unity_table, N_power,
                              reduction_poly_check);
        InverseLowRingStore(thread, shared_memory, polynomial_out);
    }

    template <typename T>
    __global__ void ForwardCoreLowRing(
        T* polynomial_in, typename std::make_unsigned<T>::type* polynomial_out,
        const Root<typename std::make_unsigned<
            T>::type>* __restrict__ root_of_unity_table,
        Modulus<typename std::make_unsigned<T>::type> modulus, int shared_index,
        int N_power, bool reduction_poly_check, int total_batch)
    {
        using TU = typename std::make_unsigned<T>::type;

        ForwardCoreLowRingBody(polynomial_in, polynomial_out,
                               root_of_unity_table, SingleModulus<TU>{modulus},
                               shared_index, N_power, reduction_poly_check,
                               total_batch);
    }

    template <typename T>
    __global__ void
    ForwardCoreLowRing(T* polynomial_in,
                       typename std::make_unsigned<T>::type* polynomial_out,
                       const Root<typename std::make_unsigned<
                           T>::type>* __restrict__ root_of_unity_table,
                       Modulus<typename std::make_unsigned<T>::type>* modulus,
                       int shared_index, int N_power, bool reduction_poly_check,
                       int total_batch, int mod_count)
    {
        using TU = typename std::make_unsigned<T>::type;

        ForwardCoreLowRingBody(polynomial_in, polynomial_out,
                               root_of_unity_table,
                               RnsModulus<TU>{modulus, mod_count}, shared_index,
                               N_power, reduction_poly_check, total_batch);
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
            RnsModulus<TU>{modulus, mod_count, n_inverse}, shared_index,
            N_power, reduction_poly_check, total_batch);
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
        int logm, int outer_iteration_count, int N_power, bool not_last_kernel,
        bool reduction_poly_check)
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

        location_t global_address =
            idx_x +
            (location_t) (idx_y *
                          (offset / (1 << (outer_iteration_count - 1)))) +
            (location_t) (blockDim.x * block_x) +
            (location_t) (2 * block_y * offset) +
            (location_t) (poly_index << N_power);

        location_t omega_address =
            idx_x +
            (location_t) (idx_y *
                          (offset / (1 << (outer_iteration_count - 1)))) +
            (location_t) (blockDim.x * block_x) +
            (location_t) (block_y * offset);

        location_t shared_address = (idx_x + (idx_y * blockDim.x));

        // Load T from global & store to shared
        if constexpr (std::is_signed<T>::value)
        {
            T input1_reg = polynomial_in[global_address];
            T input2_reg = polynomial_in[global_address + offset];
            shared_memory[shared_address] =
                OPERATOR_GPU<TU>::reduce(input1_reg, modulus_reg);
            shared_memory[shared_address + (blockDim.x * blockDim.y)] =
                OPERATOR_GPU<TU>::reduce(input2_reg, modulus_reg);
        }
        else
        {
            shared_memory[shared_address] = polynomial_in[global_address];
            shared_memory[shared_address + (blockDim.x * blockDim.y)] =
                polynomial_in[global_address + offset];
        }

        int t = 1 << t_;
        int in_shared_address = ((shared_address >> t_) << t_) + shared_address;
        location_t current_root_index;
        if (not_last_kernel)
        {
#pragma unroll
            for (int lp = 0; lp < outer_iteration_count; lp++)
            {
                __syncthreads();
                if (reduction_poly_check)
                { // X_N_minus
                    current_root_index = (omega_address >> t_2) +
                                         (location_t) (prime_index << N_power);
                }
                else
                { // X_N_plus
                    current_root_index = m + (omega_address >> t_2) +
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
                    ((shared_address >> t_) << t_) + shared_address;
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
                    current_root_index = (omega_address >> t_2) +
                                         (location_t) (prime_index << N_power);
                }
                else
                { // X_N_plus
                    current_root_index = m + (omega_address >> t_2) +
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
                    ((shared_address >> t_) << t_) + shared_address;
            }
            __syncthreads();

#pragma unroll
            for (int lp = 0; lp < 6; lp++)
            {
                if (reduction_poly_check)
                { // X_N_minus
                    current_root_index = (omega_address >> t_2) +
                                         (location_t) (prime_index << N_power);
                }
                else
                { // X_N_plus
                    current_root_index = m + (omega_address >> t_2) +
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
                    ((shared_address >> t_) << t_) + shared_address;
            }
            __syncthreads();
        }

        polynomial_out[global_address] = shared_memory[shared_address];
        polynomial_out[global_address + offset] =
            shared_memory[shared_address + (blockDim.x * blockDim.y)];
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

        location_t global_address =
            idx_x +
            (location_t) (idx_y *
                          (offset / (1 << (outer_iteration_count - 1)))) +
            (location_t) (blockDim.x * block_x) +
            (location_t) (2 * block_y * offset) +
            (location_t) (poly_index << N_power);

        location_t omega_address =
            idx_x +
            (location_t) (idx_y *
                          (offset / (1 << (outer_iteration_count - 1)))) +
            (location_t) (blockDim.x * block_x) +
            (location_t) (block_y * offset);
        location_t shared_address = (idx_x + (idx_y * blockDim.x));

        shared_memory[shared_address] = polynomial_in[global_address];
        shared_memory[shared_address + (blockDim.x * blockDim.y)] =
            polynomial_in[global_address + offset];

        int t = 1 << t_;
        int in_shared_address = ((shared_address >> t_) << t_) + shared_address;
        location_t current_root_index;
#pragma unroll
        for (int lp = 0; lp < loops; lp++)
        {
            __syncthreads();
            if (reduction_poly_check)
            { // X_N_minus
                current_root_index = (omega_address >> t_2) +
                                     (location_t) (prime_index << N_power);
            }
            else
            { // X_N_plus
                current_root_index = m + (omega_address >> t_2) +
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

            in_shared_address = ((shared_address >> t_) << t_) + shared_address;
        }
        __syncthreads();

        if (last_kernel)
        {
            TU output1_reg = OPERATOR_GPU<TU>::mult(
                shared_memory[shared_address], n_inverse_reg, modulus_reg);
            TU output2_reg = OPERATOR_GPU<TU>::mult(
                shared_memory[shared_address + (blockDim.x * blockDim.y)],
                n_inverse_reg, modulus_reg);

            if constexpr (std::is_signed<T>::value)
            {
                polynomial_out[global_address] =
                    OPERATOR_GPU<TU>::centered_reduction(output1_reg,
                                                         modulus_reg);
                polynomial_out[global_address + offset] =
                    OPERATOR_GPU<TU>::centered_reduction(output2_reg,
                                                         modulus_reg);
            }
            else
            {
                polynomial_out[global_address] = output1_reg;
                polynomial_out[global_address + offset] = output2_reg;
            }
        }
        else
        {
            polynomial_out[global_address] = shared_memory[shared_address];
            polynomial_out[global_address + offset] =
                shared_memory[shared_address + (blockDim.x * blockDim.y)];
        }
    }

    template <typename T>
    __global__ void
    ForwardCore(T* polynomial_in,
                typename std::make_unsigned<T>::type* polynomial_out,
                const Root<typename std::make_unsigned<
                    T>::type>* __restrict__ root_of_unity_table,
                Modulus<typename std::make_unsigned<T>::type> modulus,
                int shared_index, int logm, int outer_iteration_count,
                int N_power, bool not_last_kernel, bool reduction_poly_check)
    {
        using TU = typename std::make_unsigned<T>::type;

        ForwardCoreBody<false>(polynomial_in, polynomial_out,
                               root_of_unity_table, SingleModulus<TU>{modulus},
                               SequentialPolys{}, shared_index, logm,
                               outer_iteration_count, N_power, not_last_kernel,
                               reduction_poly_check);
    }

    template <typename T>
    __global__ void ForwardCore(
        T* polynomial_in, typename std::make_unsigned<T>::type* polynomial_out,
        const Root<typename std::make_unsigned<
            T>::type>* __restrict__ root_of_unity_table,
        Modulus<typename std::make_unsigned<T>::type>* modulus,
        int shared_index, int logm, int outer_iteration_count, int N_power,
        bool not_last_kernel, bool reduction_poly_check, int mod_count)
    {
        using TU = typename std::make_unsigned<T>::type;

        ForwardCoreBody<false>(
            polynomial_in, polynomial_out, root_of_unity_table,
            RnsModulus<TU>{modulus, mod_count}, SequentialPolys{}, shared_index,
            logm, outer_iteration_count, N_power, not_last_kernel,
            reduction_poly_check);
    }

    template <typename T>
    __global__ void
    ForwardCore_(T* polynomial_in,
                 typename std::make_unsigned<T>::type* polynomial_out,
                 const Root<typename std::make_unsigned<
                     T>::type>* __restrict__ root_of_unity_table,
                 Modulus<typename std::make_unsigned<T>::type> modulus,
                 int shared_index, int logm, int outer_iteration_count,
                 int N_power, bool not_last_kernel, bool reduction_poly_check)
    {
        using TU = typename std::make_unsigned<T>::type;

        ForwardCoreBody<true>(polynomial_in, polynomial_out,
                              root_of_unity_table, SingleModulus<TU>{modulus},
                              SequentialPolys{}, shared_index, logm,
                              outer_iteration_count, N_power, not_last_kernel,
                              reduction_poly_check);
    }

    template <typename T>
    __global__ void ForwardCore_(
        T* polynomial_in, typename std::make_unsigned<T>::type* polynomial_out,
        const Root<typename std::make_unsigned<
            T>::type>* __restrict__ root_of_unity_table,
        Modulus<typename std::make_unsigned<T>::type>* modulus,
        int shared_index, int logm, int outer_iteration_count, int N_power,
        bool not_last_kernel, bool reduction_poly_check, int mod_count)
    {
        using TU = typename std::make_unsigned<T>::type;

        ForwardCoreBody<true>(
            polynomial_in, polynomial_out, root_of_unity_table,
            RnsModulus<TU>{modulus, mod_count}, SequentialPolys{}, shared_index,
            logm, outer_iteration_count, N_power, not_last_kernel,
            reduction_poly_check);
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
            shared_index, logm, k, outer_iteration_count, N_power, last_kernel,
            reduction_poly_check);
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
            shared_index, logm, k, outer_iteration_count, N_power, last_kernel,
            reduction_poly_check);
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
            shared_index, logm, k, outer_iteration_count, N_power, last_kernel,
            reduction_poly_check);
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
            shared_index, logm, k, outer_iteration_count, N_power, last_kernel,
            reduction_poly_check);
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

        location_t global_address =
            (idx_y << log_column) + (blockDim.x * block_x) + idx_x;
        location_t global_offset = (blockDim.y << log_column);

        int t_ = log_row - 1;
        int m = 1;

        // Load T from global & store to shared
        const int transpose_block = idx_y + (idx_x * (2 * blockDim.y));
        if constexpr (std::is_signed<T>::value)
        {
            T input1_reg = polynomial_in[global_address];
            T input2_reg = polynomial_in[global_address + global_offset];
            shared_memory[transpose_block] =
                OPERATOR_GPU<TU>::reduce(input1_reg, modulus_thread);
            shared_memory[transpose_block + blockDim.y] =
                OPERATOR_GPU<TU>::reduce(input2_reg, modulus_thread);
        }
        else
        {
            shared_memory[transpose_block] = polynomial_in[global_address];
            shared_memory[transpose_block + blockDim.y] =
                polynomial_in[global_address + global_offset];
        }

        const int block_thread_index = idx_x + (idx_y * blockDim.x);
        const int ntt_thread_index = block_thread_index & ((1 << t_) - 1);
        const int ntt_block_index = block_thread_index >> t_;
        int offset = ntt_block_index << log_row;

        int shared_address = ntt_thread_index;
        int omega_address = ntt_thread_index;
        int t = 1 << t_;
        int in_shared_address = ((shared_address >> t_) << t_) + shared_address;
        location_t current_root_index;
        __syncthreads();

        int loop = ((log_row - 6) <= 0) ? log_row : 6;

#pragma unroll
        for (int lp = 0; lp < (log_row - 6); lp++)
        {
            int group_in_shared_address = in_shared_address + offset;

            if (reduction_poly_check)
            { // X_N_minus
                current_root_index = (omega_address >> t_);
            }
            else
            { // X_N_plus
                current_root_index = m + (omega_address >> t_);
            }

            Root<TU> root = root_shared_memory[current_root_index];

            CooleyTukeyUnit(shared_memory[group_in_shared_address],
                            shared_memory[group_in_shared_address + t], root,
                            modulus_thread);

            t = t >> 1;
            t_ -= 1;
            m <<= 1;

            in_shared_address = ((shared_address >> t_) << t_) + shared_address;
            __syncthreads();
        }

#pragma unroll
        for (int lp = 0; lp < loop; lp++)
        {
            int group_in_shared_address = in_shared_address + offset;

            if (reduction_poly_check)
            { // X_N_minus
                current_root_index = (omega_address >> t_);
            }
            else
            { // X_N_plus
                current_root_index = m + (omega_address >> t_);
            }

            Root<TU> root = root_shared_memory[current_root_index];

            CooleyTukeyUnit(shared_memory[group_in_shared_address],
                            shared_memory[group_in_shared_address + t], root,
                            modulus_thread);

            t = t >> 1;
            t_ -= 1;
            m <<= 1;

            in_shared_address = ((shared_address >> t_) << t_) + shared_address;
        }

        __syncthreads();

        polynomial_out[global_address] = shared_memory[transpose_block];
        polynomial_out[global_address + global_offset] =
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

        location_t global_address =
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
            T input1_reg = polynomial_in[global_address];
            T input2_reg = polynomial_in[global_address + global_offset];
            shared_memory[transpose_block] =
                OPERATOR_GPU<TU>::reduce(input1_reg, load_modulus);
            shared_memory[transpose_block + blockDim.y] =
                OPERATOR_GPU<TU>::reduce(input2_reg, load_modulus);
        }
        else
        {
            shared_memory[transpose_block] = polynomial_in[global_address];
            shared_memory[transpose_block + blockDim.y] =
                polynomial_in[global_address + global_offset];
        }

        int shared_address = ntt_thread_index;
        int omega_address = ntt_thread_index;
        int t = 1 << t_;
        int in_shared_address = ((shared_address >> t_) << t_) + shared_address;
        location_t current_root_index;
        __syncthreads();

        int loop = ((log_row - 6) <= 0) ? log_row : 6;

#pragma unroll
        for (int lp = 0; lp < (log_row - 6); lp++)
        {
            int group_in_shared_address = in_shared_address + offset;

            if (reduction_poly_check)
            { // X_N_minus
                current_root_index = (omega_address >> t_) + root_offset;
            }
            else
            { // X_N_plus
                current_root_index = m + (omega_address >> t_) + root_offset;
            }

            Root<TU> root = root_shared_memory[current_root_index];

            CooleyTukeyUnit(shared_memory[group_in_shared_address],
                            shared_memory[group_in_shared_address + t], root,
                            modulus_thread);

            t = t >> 1;
            t_ -= 1;
            m <<= 1;

            in_shared_address = ((shared_address >> t_) << t_) + shared_address;
            __syncthreads();
        }

#pragma unroll
        for (int lp = 0; lp < loop; lp++)
        {
            int group_in_shared_address = in_shared_address + offset;

            if (reduction_poly_check)
            { // X_N_minus
                current_root_index = (omega_address >> t_) + root_offset;
            }
            else
            { // X_N_plus
                current_root_index = m + (omega_address >> t_) + root_offset;
            }

            Root<TU> root = root_shared_memory[current_root_index];

            CooleyTukeyUnit(shared_memory[group_in_shared_address],
                            shared_memory[group_in_shared_address + t], root,
                            modulus_thread);

            t = t >> 1;
            t_ -= 1;
            m <<= 1;

            in_shared_address = ((shared_address >> t_) << t_) + shared_address;
        }

        __syncthreads();

        polynomial_out[global_address] = shared_memory[transpose_block];
        polynomial_out[global_address + global_offset] =
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

        location_t global_address =
            (idx_y << log_column) + (blockDim.x * block_x) + idx_x;
        location_t global_offset = (blockDim.y << log_column);

        int t_ = 0;
        int loops = log_row;
        int m = (int) 1 << (log_row - 1);

        // Load T from global & store to shared
        const int transpose_block = idx_y + (idx_x * (2 * blockDim.y));
        shared_memory[transpose_block] = polynomial_in[global_address];
        shared_memory[transpose_block + blockDim.y] =
            polynomial_in[global_address + global_offset];

        int log_row_r = log_row - 1;
        const int block_thread_index = idx_x + (idx_y * blockDim.x);
        const int ntt_thread_index =
            block_thread_index & ((1 << log_row_r) - 1);
        const int ntt_block_index = block_thread_index >> log_row_r;
        int offset = ntt_block_index << log_row;

        int omega_address = ntt_thread_index;
        int shared_address = ntt_thread_index;
        int t = 1 << t_;
        int in_shared_address = ((shared_address >> t_) << t_) + shared_address;
        location_t current_root_index;
        __syncthreads();

#pragma unroll
        for (int lp = 0; lp < loops; lp++)
        {
            int group_in_shared_address = in_shared_address + offset;

            if (reduction_poly_check)
            { // X_N_minus
                current_root_index = (omega_address >> t_);
            }
            else
            { // X_N_plus
                current_root_index = m + (omega_address >> t_);
            }

            Root<TU> root = root_shared_memory[current_root_index];

            GentlemanSandeUnit(shared_memory[group_in_shared_address],
                               shared_memory[group_in_shared_address + t], root,
                               modulus_thread);

            t = t << 1;
            t_ += 1;
            m >>= 1;

            in_shared_address = ((shared_address >> t_) << t_) + shared_address;
            __syncthreads();
        }

        TU output1_reg = OPERATOR_GPU<TU>::mult(
            shared_memory[transpose_block], n_inverse_thread, modulus_thread);
        TU output2_reg =
            OPERATOR_GPU<TU>::mult(shared_memory[transpose_block + blockDim.y],
                                   n_inverse_thread, modulus_thread);

        if constexpr (std::is_signed<T>::value)
        {
            polynomial_out[global_address] =
                OPERATOR_GPU<TU>::centered_reduction(output1_reg,
                                                     modulus_thread);
            polynomial_out[global_address + global_offset] =
                OPERATOR_GPU<TU>::centered_reduction(output2_reg,
                                                     modulus_thread);
        }
        else
        {
            polynomial_out[global_address] = output1_reg;
            polynomial_out[global_address + global_offset] = output2_reg;
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

        location_t global_address =
            (idx_y << log_column) + (blockDim.x * block_x) + idx_x;
        location_t global_offset = (blockDim.y << log_column);

        int t_ = 0;
        int loops = log_row;
        int m = (int) 1 << (log_row - 1);

        // Load T from global & store to shared
        const int transpose_block = idx_y + (idx_x * (2 * blockDim.y));
        shared_memory[transpose_block] = polynomial_in[global_address];
        shared_memory[transpose_block + blockDim.y] =
            polynomial_in[global_address + global_offset];

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

        int omega_address = ntt_thread_index;
        int shared_address = ntt_thread_index;
        int t = 1 << t_;
        int in_shared_address = ((shared_address >> t_) << t_) + shared_address;
        location_t current_root_index;
        __syncthreads();

#pragma unroll
        for (int lp = 0; lp < loops; lp++)
        {
            int group_in_shared_address = in_shared_address + offset;

            if (reduction_poly_check)
            { // X_N_minus
                current_root_index = (omega_address >> t_) + root_offset;
            }
            else
            { // X_N_plus
                current_root_index = m + (omega_address >> t_) + root_offset;
            }

            Root<TU> root = root_shared_memory[current_root_index];

            GentlemanSandeUnit(shared_memory[group_in_shared_address],
                               shared_memory[group_in_shared_address + t], root,
                               modulus_thread);

            t = t << 1;
            t_ += 1;
            m >>= 1;

            in_shared_address = ((shared_address >> t_) << t_) + shared_address;
            __syncthreads();
        }

        // The thread stores its own column, so it uses that column's modulus.
        TU output1_reg = OPERATOR_GPU<TU>::mult(shared_memory[transpose_block],
                                                store_n_inverse, store_modulus);
        TU output2_reg =
            OPERATOR_GPU<TU>::mult(shared_memory[transpose_block + blockDim.y],
                                   store_n_inverse, store_modulus);

        if constexpr (std::is_signed<T>::value)
        {
            polynomial_out[global_address] =
                OPERATOR_GPU<TU>::centered_reduction(output1_reg,
                                                     store_modulus);
            polynomial_out[global_address + global_offset] =
                OPERATOR_GPU<TU>::centered_reduction(output2_reg,
                                                     store_modulus);
        }
        else
        {
            polynomial_out[global_address] = output1_reg;
            polynomial_out[global_address + global_offset] = output2_reg;
        }
    }

    // The tables are built once, on first use, and only read afterwards.
    template <typename T>
    const std::vector<KernelConfig>& forward_kernel_configs(int n_power)
    {
        static const auto table = CreateForwardNTTKernel<T>();
        return table.at(n_power);
    }

    template <typename T>
    const std::vector<KernelConfig>& inverse_kernel_configs(int n_power)
    {
        static const auto table = CreateInverseNTTKernel<T>();
        return table.at(n_power);
    }

    // Launchers for the multi-stage kernel families. swapped selects the _
    // variant, which swaps the roles of blockIdx.x and blockIdx.y.
    struct MergeKernels
    {
        template <typename TIn, typename TOut, typename... Args>
        static void forward(bool swapped, dim3 grid, dim3 block,
                            size_t shared_memory, cudaStream_t stream, TIn* in,
                            TOut* out, Args... args)
        {
            if (swapped)
            {
                ForwardCore_<<<grid, block, shared_memory, stream>>>(in, out,
                                                                     args...);
            }
            else
            {
                ForwardCore<<<grid, block, shared_memory, stream>>>(in, out,
                                                                    args...);
            }
        }

        template <typename TIn, typename TOut, typename... Args>
        static void inverse(bool swapped, dim3 grid, dim3 block,
                            size_t shared_memory, cudaStream_t stream, TIn* in,
                            TOut* out, Args... args)
        {
            if (swapped)
            {
                InverseCore_<<<grid, block, shared_memory, stream>>>(in, out,
                                                                     args...);
            }
            else
            {
                InverseCore<<<grid, block, shared_memory, stream>>>(in, out,
                                                                    args...);
            }
        }
    };

    struct ModulusOrderedKernels
    {
        template <typename TIn, typename TOut, typename... Args>
        static void forward(bool swapped, dim3 grid, dim3 block,
                            size_t shared_memory, cudaStream_t stream, TIn* in,
                            TOut* out, Args... args)
        {
            if (swapped)
            {
                ForwardCoreModulusOrdered_<<<grid, block, shared_memory,
                                             stream>>>(in, out, args...);
            }
            else
            {
                ForwardCoreModulusOrdered<<<grid, block, shared_memory,
                                            stream>>>(in, out, args...);
            }
        }

        template <typename TIn, typename TOut, typename... Args>
        static void inverse(bool swapped, dim3 grid, dim3 block,
                            size_t shared_memory, cudaStream_t stream, TIn* in,
                            TOut* out, Args... args)
        {
            if (swapped)
            {
                InverseCoreModulusOrdered_<<<grid, block, shared_memory,
                                             stream>>>(in, out, args...);
            }
            else
            {
                InverseCoreModulusOrdered<<<grid, block, shared_memory,
                                            stream>>>(in, out, args...);
            }
        }
    };

    struct PolyOrderedKernels
    {
        template <typename TIn, typename TOut, typename... Args>
        static void forward(bool swapped, dim3 grid, dim3 block,
                            size_t shared_memory, cudaStream_t stream, TIn* in,
                            TOut* out, Args... args)
        {
            if (swapped)
            {
                ForwardCorePolyOrdered_<<<grid, block, shared_memory, stream>>>(
                    in, out, args...);
            }
            else
            {
                ForwardCorePolyOrdered<<<grid, block, shared_memory, stream>>>(
                    in, out, args...);
            }
        }

        template <typename TIn, typename TOut, typename... Args>
        static void inverse(bool swapped, dim3 grid, dim3 block,
                            size_t shared_memory, cudaStream_t stream, TIn* in,
                            TOut* out, Args... args)
        {
            if (swapped)
            {
                InverseCorePolyOrdered_<<<grid, block, shared_memory, stream>>>(
                    in, out, args...);
            }
            else
            {
                InverseCorePolyOrdered<<<grid, block, shared_memory, stream>>>(
                    in, out, args...);
            }
        }
    };

    // Runs the multi-stage forward NTT kernels of one transform. Kernel i
    // reads the output of kernel i - 1 (the first one reads device_in) and
    // writes device_out. For log2(N) >= 25 the last kernel is the _ variant.
    // extra holds the trailing kernel arguments of the family (mod_count,
    // order).
    template <typename Kernels, typename T, typename ModulusArg,
              typename... Extra>
    void forward_stages(
        T* device_in, typename std::make_unsigned<T>::type* device_out,
        Root<typename std::make_unsigned<T>::type>* root_of_unity_table,
        ModulusArg modulus, int n_power, ReductionPolynomial reduction_poly,
        int batch_size, cudaStream_t stream, Extra... extra)
    {
        using TU = typename std::make_unsigned<T>::type;

        const std::vector<KernelConfig>& kernels =
            forward_kernel_configs<TU>(n_power);
        const bool swap_last = (n_power >= 25);
        const bool x_n_minus =
            (reduction_poly == ReductionPolynomial::X_N_minus);

        for (size_t i = 0; i < kernels.size(); i++)
        {
            const KernelConfig& k = kernels[i];
            const dim3 grid(k.griddim_x, k.griddim_y, batch_size);
            const dim3 block(k.blockdim_x, k.blockdim_y);
            const bool swapped = swap_last && (i + 1 == kernels.size());
            if (i == 0)
            {
                Kernels::forward(swapped, grid, block, k.shared_memory, stream,
                                 device_in, device_out, root_of_unity_table,
                                 modulus, k.shared_index, k.logm,
                                 k.outer_iteration_count, n_power,
                                 k.not_last_kernel, x_n_minus, extra...);
            }
            else
            {
                Kernels::forward(swapped, grid, block, k.shared_memory, stream,
                                 device_out, device_out, root_of_unity_table,
                                 modulus, k.shared_index, k.logm,
                                 k.outer_iteration_count, n_power,
                                 k.not_last_kernel, x_n_minus, extra...);
            }
            GPUNTT_CUDA_CHECK(cudaGetLastError());
        }
    }

    // Inverse counterpart of forward_stages. Only the last kernel writes T
    // (signed output); the others keep unsigned values in device_out. For
    // log2(N) >= 25 the first kernel is the _ variant. The inverse kernel
    // table stores the last_kernel flag in not_last_kernel.
    template <typename Kernels, typename T, typename ModulusArg,
              typename NinverseArg, typename... Extra>
    void inverse_stages(
        typename std::make_unsigned<T>::type* device_in, T* device_out,
        Root<typename std::make_unsigned<T>::type>* root_of_unity_table,
        ModulusArg modulus, NinverseArg n_inverse, int n_power,
        ReductionPolynomial reduction_poly, int batch_size, cudaStream_t stream,
        Extra... extra)
    {
        using TU = typename std::make_unsigned<T>::type;

        TU* device_out_unsigned = reinterpret_cast<TU*>(device_out);
        const std::vector<KernelConfig>& kernels =
            inverse_kernel_configs<TU>(n_power);
        const bool swap_first = (n_power >= 25);
        const bool x_n_minus =
            (reduction_poly == ReductionPolynomial::X_N_minus);

        for (size_t i = 0; i < kernels.size(); i++)
        {
            const KernelConfig& k = kernels[i];
            const dim3 grid(k.griddim_x, k.griddim_y, batch_size);
            const dim3 block(k.blockdim_x, k.blockdim_y);
            const bool swapped = swap_first && (i == 0);
            TU* in = (i == 0) ? device_in : device_out_unsigned;
            if (i + 1 == kernels.size())
            {
                Kernels::inverse(swapped, grid, block, k.shared_memory, stream,
                                 in, device_out, root_of_unity_table, modulus,
                                 k.shared_index, k.logm, k.k,
                                 k.outer_iteration_count, n_power, n_inverse,
                                 k.not_last_kernel, x_n_minus, extra...);
            }
            else
            {
                Kernels::inverse(swapped, grid, block, k.shared_memory, stream,
                                 in, device_out_unsigned, root_of_unity_table,
                                 modulus, k.shared_index, k.logm, k.k,
                                 k.outer_iteration_count, n_power, n_inverse,
                                 k.not_last_kernel, x_n_minus, extra...);
            }
            GPUNTT_CUDA_CHECK(cudaGetLastError());
        }
    }

    // Runs the low ring size forward kernel, which transforms several
    // polynomials per block.
    template <typename T, typename ModulusArg, typename... Extra>
    void forward_low_ring(
        T* device_in, typename std::make_unsigned<T>::type* device_out,
        Root<typename std::make_unsigned<T>::type>* root_of_unity_table,
        ModulusArg modulus, int n_power, ReductionPolynomial reduction_poly,
        int batch_size, cudaStream_t stream, Extra... extra)
    {
        using TU = typename std::make_unsigned<T>::type;

        const KernelConfig& k = forward_kernel_configs<TU>(n_power)[0];
        ForwardCoreLowRing<<<
            dim3((batch_size + (k.blockdim_y - 1)) / k.blockdim_y, 1, 1),
            dim3(k.blockdim_x, k.blockdim_y), k.shared_memory, stream>>>(
            device_in, device_out, root_of_unity_table, modulus, k.shared_index,
            n_power, (reduction_poly == ReductionPolynomial::X_N_minus),
            batch_size, extra...);
        GPUNTT_CUDA_CHECK(cudaGetLastError());
    }

    template <typename T, typename ModulusArg, typename NinverseArg,
              typename... Extra>
    void inverse_low_ring(
        typename std::make_unsigned<T>::type* device_in, T* device_out,
        Root<typename std::make_unsigned<T>::type>* root_of_unity_table,
        ModulusArg modulus, NinverseArg n_inverse, int n_power,
        ReductionPolynomial reduction_poly, int batch_size, cudaStream_t stream,
        Extra... extra)
    {
        using TU = typename std::make_unsigned<T>::type;

        const KernelConfig& k = inverse_kernel_configs<TU>(n_power)[0];
        InverseCoreLowRing<<<
            dim3((batch_size + (k.blockdim_y - 1)) / k.blockdim_y, 1, 1),
            dim3(k.blockdim_x, k.blockdim_y), k.shared_memory, stream>>>(
            device_in, device_out, root_of_unity_table, modulus, k.shared_index,
            n_power, n_inverse,
            (reduction_poly == ReductionPolynomial::X_N_minus), batch_size,
            extra...);
        GPUNTT_CUDA_CHECK(cudaGetLastError());
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

                if (cfg.n_power < 10)
                {
                    forward_low_ring(device_in, device_out, root_of_unity_table,
                                     modulus, cfg.n_power, cfg.reduction_poly,
                                     batch_size, cfg.stream);
                }
                else
                {
                    forward_stages<MergeKernels>(
                        device_in, device_out, root_of_unity_table, modulus,
                        cfg.n_power, cfg.reduction_poly, batch_size,
                        cfg.stream);
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

                if (cfg.n_power < 11)
                {
                    inverse_low_ring(device_in, device_out, root_of_unity_table,
                                     modulus, cfg.mod_inverse, cfg.n_power,
                                     cfg.reduction_poly, batch_size,
                                     cfg.stream);
                }
                else
                {
                    inverse_stages<MergeKernels>(
                        device_in, device_out, root_of_unity_table, modulus,
                        cfg.mod_inverse, cfg.n_power, cfg.reduction_poly,
                        batch_size, cfg.stream);
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

                if (cfg.n_power < 10)
                {
                    forward_low_ring(device_in, device_out, root_of_unity_table,
                                     modulus, cfg.n_power, cfg.reduction_poly,
                                     batch_size, cfg.stream, mod_count);
                }
                else
                {
                    forward_stages<MergeKernels>(
                        device_in, device_out, root_of_unity_table, modulus,
                        cfg.n_power, cfg.reduction_poly, batch_size, cfg.stream,
                        mod_count);
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
                    (total_block_thread * 2 * 2) * sizeof(T), cfg.stream>>>(
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

                if (cfg.n_power < 11)
                {
                    inverse_low_ring(device_in, device_out, root_of_unity_table,
                                     modulus, cfg.mod_inverse, cfg.n_power,
                                     cfg.reduction_poly, batch_size, cfg.stream,
                                     mod_count);
                }
                else
                {
                    inverse_stages<MergeKernels>(
                        device_in, device_out, root_of_unity_table, modulus,
                        cfg.mod_inverse, cfg.n_power, cfg.reduction_poly,
                        batch_size, cfg.stream, mod_count);
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
                    (total_block_thread * 2 * 2) * sizeof(T), cfg.stream>>>(
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
        int outer_iteration_count, int N_power, bool not_last_kernel,
        bool reduction_poly_check, int mod_count, int* order)
    {
        ForwardCoreBody<false>(
            polynomial_in, polynomial_out, root_of_unity_table,
            OrderedRnsModulus<T>{modulus, mod_count, order}, SequentialPolys{},
            shared_index, logm, outer_iteration_count, N_power, not_last_kernel,
            reduction_poly_check);
    }

    template <typename T>
    __global__ void ForwardCoreModulusOrdered_(
        T* polynomial_in, T* polynomial_out, Root<T>* root_of_unity_table,
        Modulus<T>* modulus, int shared_index, int logm,
        int outer_iteration_count, int N_power, bool not_last_kernel,
        bool reduction_poly_check, int mod_count, int* order)
    {
        ForwardCoreBody<true>(
            polynomial_in, polynomial_out, root_of_unity_table,
            OrderedRnsModulus<T>{modulus, mod_count, order}, SequentialPolys{},
            shared_index, logm, outer_iteration_count, N_power, not_last_kernel,
            reduction_poly_check);
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
            SequentialPolys{}, shared_index, logm, k, outer_iteration_count,
            N_power, last_kernel, reduction_poly_check);
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
            SequentialPolys{}, shared_index, logm, k, outer_iteration_count,
            N_power, last_kernel, reduction_poly_check);
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

        switch (cfg.ntt_type)
        {
            case FORWARD:
                forward_stages<ModulusOrderedKernels>(
                    device_in, device_out, root_of_unity_table, modulus,
                    cfg.n_power, cfg.reduction_poly, batch_size, cfg.stream,
                    mod_count, order);
                break;
            case INVERSE:
                inverse_stages<ModulusOrderedKernels>(
                    device_in, device_out, root_of_unity_table, modulus,
                    cfg.mod_inverse, cfg.n_power, cfg.reduction_poly,
                    batch_size, cfg.stream, mod_count, order);
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
    __global__ void ForwardCorePolyOrdered(
        T* polynomial_in, T* polynomial_out, Root<T>* root_of_unity_table,
        Modulus<T>* modulus, int shared_index, int logm,
        int outer_iteration_count, int N_power, bool not_last_kernel,
        bool reduction_poly_check, int mod_count, int* order)
    {
        ForwardCoreBody<false>(
            polynomial_in, polynomial_out, root_of_unity_table,
            RnsModulus<T>{modulus, mod_count}, OrderedPolys{order},
            shared_index, logm, outer_iteration_count, N_power, not_last_kernel,
            reduction_poly_check);
    }

    template <typename T>
    __global__ void ForwardCorePolyOrdered_(
        T* polynomial_in, T* polynomial_out, Root<T>* root_of_unity_table,
        Modulus<T>* modulus, int shared_index, int logm,
        int outer_iteration_count, int N_power, bool not_last_kernel,
        bool reduction_poly_check, int mod_count, int* order)
    {
        ForwardCoreBody<true>(
            polynomial_in, polynomial_out, root_of_unity_table,
            RnsModulus<T>{modulus, mod_count}, OrderedPolys{order},
            shared_index, logm, outer_iteration_count, N_power, not_last_kernel,
            reduction_poly_check);
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
            shared_index, logm, k, outer_iteration_count, N_power, last_kernel,
            reduction_poly_check);
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
            shared_index, logm, k, outer_iteration_count, N_power, last_kernel,
            reduction_poly_check);
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

        switch (cfg.ntt_type)
        {
            case FORWARD:
                forward_stages<PolyOrderedKernels>(
                    device_in, device_out, root_of_unity_table, modulus,
                    cfg.n_power, cfg.reduction_poly, batch_size, cfg.stream,
                    mod_count, order);
                break;
            case INVERSE:
                inverse_stages<PolyOrderedKernels>(
                    device_in, device_out, root_of_unity_table, modulus,
                    cfg.mod_inverse, cfg.n_power, cfg.reduction_poly,
                    batch_size, cfg.stream, mod_count, order);
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

    // Explicit instantiations. The kernels are declared in the public
    // header, so every supported kernel and host function is
    // instantiated here. Signed input (NTT) and output (INTT) are
    // supported by the single modulus and RNS variants only.

    template <typename T>
    using UnsignedOf = typename std::make_unsigned<T>::type;

#define GPUNTT_INSTANTIATE_ALL_TYPES(T)                                        \
    template __global__ void ForwardCoreLowRing<T>(                            \
        T*, UnsignedOf<T>*, const Root<UnsignedOf<T>>* __restrict__,           \
        Modulus<UnsignedOf<T>>, int, int, bool, int);                          \
    template __global__ void ForwardCoreLowRing<T>(                            \
        T*, UnsignedOf<T>*, const Root<UnsignedOf<T>>* __restrict__,           \
        Modulus<UnsignedOf<T>>*, int, int, bool, int, int);                    \
    template __global__ void ForwardCore<T>(                                   \
        T*, UnsignedOf<T>*, const Root<UnsignedOf<T>>* __restrict__,           \
        Modulus<UnsignedOf<T>>, int, int, int, int, bool, bool);               \
    template __global__ void ForwardCore<T>(                                   \
        T*, UnsignedOf<T>*, const Root<UnsignedOf<T>>* __restrict__,           \
        Modulus<UnsignedOf<T>>*, int, int, int, int, bool, bool, int);         \
    template __global__ void ForwardCore_<T>(                                  \
        T*, UnsignedOf<T>*, const Root<UnsignedOf<T>>* __restrict__,           \
        Modulus<UnsignedOf<T>>, int, int, int, int, bool, bool);               \
    template __global__ void ForwardCore_<T>(                                  \
        T*, UnsignedOf<T>*, const Root<UnsignedOf<T>>* __restrict__,           \
        Modulus<UnsignedOf<T>>*, int, int, int, int, bool, bool, int);         \
    template __global__ void InverseCoreLowRing<T>(                            \
        UnsignedOf<T>*, T*, const Root<UnsignedOf<T>>* __restrict__,           \
        Modulus<UnsignedOf<T>>, int, int, Ninverse<UnsignedOf<T>>, bool, int); \
    template __global__ void InverseCoreLowRing<T>(                            \
        UnsignedOf<T>*, T*, const Root<UnsignedOf<T>>* __restrict__,           \
        Modulus<UnsignedOf<T>>*, int, int, Ninverse<UnsignedOf<T>>*, bool,     \
        int, int);                                                             \
    template __global__ void InverseCore<T>(                                   \
        UnsignedOf<T>*, T*, const Root<UnsignedOf<T>>* __restrict__,           \
        Modulus<UnsignedOf<T>>, int, int, int, int, int,                       \
        Ninverse<UnsignedOf<T>>, bool, bool);                                  \
    template __global__ void InverseCore<T>(                                   \
        UnsignedOf<T>*, T*, const Root<UnsignedOf<T>>* __restrict__,           \
        Modulus<UnsignedOf<T>>*, int, int, int, int, int,                      \
        Ninverse<UnsignedOf<T>>*, bool, bool, int);                            \
    template __global__ void InverseCore_<T>(                                  \
        UnsignedOf<T>*, T*, const Root<UnsignedOf<T>>* __restrict__,           \
        Modulus<UnsignedOf<T>>, int, int, int, int, int,                       \
        Ninverse<UnsignedOf<T>>, bool, bool);                                  \
    template __global__ void InverseCore_<T>(                                  \
        UnsignedOf<T>*, T*, const Root<UnsignedOf<T>>* __restrict__,           \
        Modulus<UnsignedOf<T>>*, int, int, int, int, int,                      \
        Ninverse<UnsignedOf<T>>*, bool, bool, int);                            \
    template __global__ void ForwardCoreTranspose<T>(                          \
        T*, UnsignedOf<T>*, const Root<UnsignedOf<T>>* __restrict__,           \
        const Modulus<UnsignedOf<T>>, int, int, bool);                         \
    template __global__ void ForwardCoreTranspose<T>(                          \
        T*, UnsignedOf<T>*, const Root<UnsignedOf<T>>* __restrict__,           \
        const Modulus<UnsignedOf<T>>*, int, int, bool, int);                   \
    template __global__ void InverseCoreTranspose<T>(                          \
        UnsignedOf<T>*, T*, const Root<UnsignedOf<T>>* __restrict__,           \
        Modulus<UnsignedOf<T>>, Ninverse<UnsignedOf<T>>, int, int, bool);      \
    template __global__ void InverseCoreTranspose<T>(                          \
        UnsignedOf<T>*, T*, const Root<UnsignedOf<T>>* __restrict__,           \
        Modulus<UnsignedOf<T>>*, Ninverse<UnsignedOf<T>>*, int, int, bool,     \
        int);                                                                  \
    template __host__ void GPU_NTT<T>(                                         \
        T*, UnsignedOf<T>*, Root<UnsignedOf<T>>*, Modulus<UnsignedOf<T>>,      \
        ntt_configuration<UnsignedOf<T>>, int);                                \
    template __host__ void GPU_INTT<T>(                                        \
        UnsignedOf<T>*, T*, Root<UnsignedOf<T>>*, Modulus<UnsignedOf<T>>,      \
        ntt_configuration<UnsignedOf<T>>, int);                                \
    template __host__ void GPU_NTT<T>(                                         \
        T*, UnsignedOf<T>*, Root<UnsignedOf<T>>*, Modulus<UnsignedOf<T>>*,     \
        ntt_rns_configuration<UnsignedOf<T>>, int, int);                       \
    template __host__ void GPU_INTT<T>(                                        \
        UnsignedOf<T>*, T*, Root<UnsignedOf<T>>*, Modulus<UnsignedOf<T>>*,     \
        ntt_rns_configuration<UnsignedOf<T>>, int, int);

#define GPUNTT_INSTANTIATE_UNSIGNED_TYPES(T)                                   \
    template __host__ void GPU_NTT_Inplace<T>(T*, Root<T>*, Modulus<T>,        \
                                              ntt_configuration<T>, int);      \
    template __host__ void GPU_INTT_Inplace<T>(T*, Root<T>*, Modulus<T>,       \
                                               ntt_configuration<T>, int);     \
    template __host__ void GPU_NTT_Inplace<T>(                                 \
        T*, Root<T>*, Modulus<T>*, ntt_rns_configuration<T>, int, int);        \
    template __host__ void GPU_INTT_Inplace<T>(                                \
        T*, Root<T>*, Modulus<T>*, ntt_rns_configuration<T>, int, int);        \
    template __global__ void ForwardCoreModulusOrdered<T>(                     \
        T*, T*, Root<T>*, Modulus<T>*, int, int, int, int, bool, bool, int,    \
        int*);                                                                 \
    template __global__ void ForwardCoreModulusOrdered_<T>(                    \
        T*, T*, Root<T>*, Modulus<T>*, int, int, int, int, bool, bool, int,    \
        int*);                                                                 \
    template __global__ void InverseCoreModulusOrdered<T>(                     \
        T*, T*, Root<T>*, Modulus<T>*, int, int, int, int, int, Ninverse<T>*,  \
        bool, bool, int, int*);                                                \
    template __global__ void InverseCoreModulusOrdered_<T>(                    \
        T*, T*, Root<T>*, Modulus<T>*, int, int, int, int, int, Ninverse<T>*,  \
        bool, bool, int, int*);                                                \
    template __host__ void GPU_NTT_Modulus_Ordered<T>(                         \
        T*, T*, Root<T>*, Modulus<T>*, ntt_rns_configuration<T>, int, int,     \
        int*);                                                                 \
    template __host__ void GPU_NTT_Modulus_Ordered_Inplace<T>(                 \
        T*, Root<T>*, Modulus<T>*, ntt_rns_configuration<T>, int, int, int*);  \
    template __global__ void ForwardCorePolyOrdered<T>(                        \
        T*, T*, Root<T>*, Modulus<T>*, int, int, int, int, bool, bool, int,    \
        int*);                                                                 \
    template __global__ void ForwardCorePolyOrdered_<T>(                       \
        T*, T*, Root<T>*, Modulus<T>*, int, int, int, int, bool, bool, int,    \
        int*);                                                                 \
    template __global__ void InverseCorePolyOrdered<T>(                        \
        T*, T*, Root<T>*, Modulus<T>*, int, int, int, int, int, Ninverse<T>*,  \
        bool, bool, int, int*);                                                \
    template __global__ void InverseCorePolyOrdered_<T>(                       \
        T*, T*, Root<T>*, Modulus<T>*, int, int, int, int, int, Ninverse<T>*,  \
        bool, bool, int, int*);                                                \
    template __host__ void GPU_NTT_Poly_Ordered<T>(                            \
        T*, T*, Root<T>*, Modulus<T>*, ntt_rns_configuration<T>, int, int,     \
        int*);                                                                 \
    template __host__ void GPU_NTT_Poly_Ordered_Inplace<T>(                    \
        T*, Root<T>*, Modulus<T>*, ntt_rns_configuration<T>, int, int, int*);

    GPUNTT_INSTANTIATE_ALL_TYPES(Data32)
    GPUNTT_INSTANTIATE_ALL_TYPES(Data64)
    GPUNTT_INSTANTIATE_ALL_TYPES(Data32s)
    GPUNTT_INSTANTIATE_ALL_TYPES(Data64s)
    GPUNTT_INSTANTIATE_UNSIGNED_TYPES(Data32)
    GPUNTT_INSTANTIATE_UNSIGNED_TYPES(Data64)

#undef GPUNTT_INSTANTIATE_ALL_TYPES
#undef GPUNTT_INSTANTIATE_UNSIGNED_TYPES

} // namespace gpuntt
