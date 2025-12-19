// (C) Ulvetanna Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan
// Paper: https://eprint.iacr.org/2023/1410

#include "ntt_merge/ntt.cuh"

namespace gpuntt
{
    template <typename T>
    __global__ void
    ForwardCoreLowRing(T* polynomial_in,
                       typename std::make_unsigned<T>::type* polynomial_out,
                       const Root<typename std::make_unsigned<
                           T>::type>* __restrict__ root_of_unity_table,
                       Modulus<typename std::make_unsigned<T>::type> modulus,
                       int shared_index, int N_power, bool zero_padding,
                       bool reduction_poly_check, int total_batch)
    {
        using TU = typename std::make_unsigned<T>::type;

        const int idx_x = threadIdx.x;
        const int idx_y = threadIdx.y;
        const int block_x = blockIdx.x;
        const int block_thread = idx_x + (idx_y * blockDim.x);
        const int batch_index = (block_x * blockDim.y) + idx_y;

        if (batch_index >= total_batch)
            return;

        int batch_offset = ((block_x + 1) * blockDim.y);
        int batch_offset_size =
            (batch_offset > total_batch)
                ? (blockDim.y - (batch_offset - total_batch))
                : blockDim.y;
        int block_size = blockDim.x * batch_offset_size;

        // extern __shared__ T shared_memory[];
        extern __shared__ char shared_memory_typed[];
        TU* shared_memory = reinterpret_cast<TU*>(shared_memory_typed);

        const Modulus<TU> modulus_reg = modulus;

        int t_2 = N_power - 1;
        int offset = idx_y << N_power;
        int t_ = shared_index;
        int m = 1;

        location_t global_addresss =
            block_thread + (location_t) ((blockDim.y * block_x) << N_power);

        location_t omega_addresss = idx_x;

        // Load T from global & store to shared
        if constexpr (std::is_signed<T>::value)
        {
            T input1_reg = polynomial_in[global_addresss];
            T input2_reg = polynomial_in[global_addresss + block_size];
            shared_memory[block_thread] =
                OPERATOR_GPU<TU>::reduce(input1_reg, modulus_reg);
            shared_memory[block_thread + block_size] =
                OPERATOR_GPU<TU>::reduce(input2_reg, modulus_reg);
        }
        else
        {
            shared_memory[block_thread] = polynomial_in[global_addresss];
            shared_memory[block_thread + block_size] =
                polynomial_in[global_addresss + block_size];
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
                current_root_index = (omega_addresss >> t_2);
            }
            else
            { // X_N_plus
                current_root_index = m + (omega_addresss >> t_2);
            }

            CooleyTukeyUnit(shared_memory[group_in_shared_address],
                            shared_memory[group_in_shared_address + t],
                            root_of_unity_table[current_root_index],
                            modulus_reg);

            t = t >> 1;
            t_2 -= 1;
            t_ -= 1;
            m <<= 1;

            in_shared_address =
                ((shared_addresss >> t_) << t_) + shared_addresss;
            __syncthreads();
        }

        __syncthreads();

        polynomial_out[global_addresss] = shared_memory[block_thread];
        polynomial_out[global_addresss + block_size] =
            shared_memory[block_thread + block_size];
    }

    template <typename T>
    __global__ void ForwardCoreLowRing(
        T* polynomial_in, typename std::make_unsigned<T>::type* polynomial_out,
        const Root<typename std::make_unsigned<
            T>::type>* __restrict__ root_of_unity_table,
        Modulus<typename std::make_unsigned<T>::type>* modulus,
        int shared_index, int N_power, bool zero_padding,
        bool reduction_poly_check, int total_batch, int mod_count)
    {
        using TU = typename std::make_unsigned<T>::type;

        const int idx_x = threadIdx.x;
        const int idx_y = threadIdx.y;
        const int block_x = blockIdx.x;
        const int block_thread = idx_x + (idx_y * blockDim.x);
        const int batch_index = (block_x * blockDim.y) + idx_y;

        if (batch_index >= total_batch)
            return;

        int mod_index = batch_index % mod_count;
        int batch_offset = ((block_x + 1) * blockDim.y);
        int batch_offset_size =
            (batch_offset > total_batch)
                ? (blockDim.y - (batch_offset - total_batch))
                : blockDim.y;
        int block_size = blockDim.x * batch_offset_size;

        // extern __shared__ T shared_memory[];
        extern __shared__ char shared_memory_typed[];
        TU* shared_memory = reinterpret_cast<TU*>(shared_memory_typed);

        const Modulus<TU> modulus_reg = modulus[mod_index];

        int t_2 = N_power - 1;
        int offset = idx_y << N_power;
        int t_ = shared_index;
        int m = 1;

        location_t global_addresss =
            block_thread + (location_t) ((blockDim.y * block_x) << N_power);

        location_t omega_addresss = idx_x;

        // Load T from global & store to shared
        if constexpr (std::is_signed<T>::value)
        {
            T input1_reg = polynomial_in[global_addresss];
            T input2_reg = polynomial_in[global_addresss + block_size];
            shared_memory[block_thread] =
                OPERATOR_GPU<TU>::reduce(input1_reg, modulus_reg);
            shared_memory[block_thread + block_size] =
                OPERATOR_GPU<TU>::reduce(input2_reg, modulus_reg);
        }
        else
        {
            shared_memory[block_thread] = polynomial_in[global_addresss];
            shared_memory[block_thread + block_size] =
                polynomial_in[global_addresss + block_size];
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
                                     (location_t) (mod_index << N_power);
            }
            else
            { // X_N_plus
                current_root_index = m + (omega_addresss >> t_2) +
                                     (location_t) (mod_index << N_power);
            }

            CooleyTukeyUnit(shared_memory[group_in_shared_address],
                            shared_memory[group_in_shared_address + t],
                            root_of_unity_table[current_root_index],
                            modulus_reg);

            t = t >> 1;
            t_2 -= 1;
            t_ -= 1;
            m <<= 1;

            in_shared_address =
                ((shared_addresss >> t_) << t_) + shared_addresss;
            __syncthreads();
        }

        __syncthreads();

        polynomial_out[global_addresss] = shared_memory[block_thread];
        polynomial_out[global_addresss + block_size] =
            shared_memory[block_thread + block_size];
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

        const int idx_x = threadIdx.x;
        const int idx_y = threadIdx.y;
        const int block_x = blockIdx.x;
        const int block_thread = idx_x + (idx_y * blockDim.x);
        const int batch_index = (block_x * blockDim.y) + idx_y;

        if (batch_index >= total_batch)
            return;

        int batch_offset = ((block_x + 1) * blockDim.y);
        int batch_offset_size =
            (batch_offset > total_batch)
                ? (blockDim.y - (batch_offset - total_batch))
                : blockDim.y;
        int block_size = blockDim.x * batch_offset_size;

        // extern __shared__ T shared_memory[];
        extern __shared__ char shared_memory_typed[];
        TU* shared_memory = reinterpret_cast<TU*>(shared_memory_typed);

        const Modulus<TU> modulus_reg = modulus;

        int t_2 = 0;
        int t_ = 0;
        int offset = idx_y << N_power;
        int loops = N_power;
        int m = (int) 1 << (N_power - 1);

        location_t global_addresss =
            block_thread + (location_t) ((blockDim.y * block_x) << N_power);

        location_t omega_addresss = idx_x;

        // Load T from global & store to shared
        shared_memory[block_thread] = polynomial_in[global_addresss];
        shared_memory[block_thread + block_size] =
            polynomial_in[global_addresss + block_size];

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
                current_root_index = (omega_addresss >> t_2);
            }
            else
            { // X_N_plus
                current_root_index = m + (omega_addresss >> t_2);
            }

            GentlemanSandeUnit(shared_memory[group_in_shared_address],
                               shared_memory[group_in_shared_address + t],
                               inverse_root_of_unity_table[current_root_index],
                               modulus_reg);

            t = t << 1;
            t_2 += 1;
            t_ += 1;
            m >>= 1;

            in_shared_address =
                ((shared_addresss >> t_) << t_) + shared_addresss;
            __syncthreads();
        }
        __syncthreads();

        TU output1_reg = OPERATOR_GPU<TU>::mult(shared_memory[block_thread],
                                                n_inverse, modulus_reg);
        TU output2_reg = OPERATOR_GPU<TU>::mult(
            shared_memory[block_thread + block_size], n_inverse, modulus_reg);

        if constexpr (std::is_signed<T>::value)
        {
            polynomial_out[global_addresss] =
                OPERATOR_GPU<TU>::centered_reduction(output1_reg, modulus_reg);
            polynomial_out[global_addresss + block_size] =
                OPERATOR_GPU<TU>::centered_reduction(output2_reg, modulus_reg);
        }
        else
        {
            polynomial_out[global_addresss] = output1_reg;
            polynomial_out[global_addresss + block_size] = output2_reg;
        }
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

        const int idx_x = threadIdx.x;
        const int idx_y = threadIdx.y;
        const int block_x = blockIdx.x;
        const int block_thread = idx_x + (idx_y * blockDim.x);
        const int batch_index = (block_x * blockDim.y) + idx_y;

        if (batch_index >= total_batch)
            return;

        int mod_index = batch_index % mod_count;
        int batch_offset = ((block_x + 1) * blockDim.y);
        int batch_offset_size =
            (batch_offset > total_batch)
                ? (blockDim.y - (batch_offset - total_batch))
                : blockDim.y;
        int block_size = blockDim.x * batch_offset_size;

        // extern __shared__ T shared_memory[];
        extern __shared__ char shared_memory_typed[];
        TU* shared_memory = reinterpret_cast<TU*>(shared_memory_typed);

        const Modulus<TU> modulus_reg = modulus[mod_index];
        const Ninverse<TU> n_inverse_reg = n_inverse[mod_index];

        int t_2 = 0;
        int t_ = 0;
        int offset = idx_y << N_power;
        int loops = N_power;
        int m = (int) 1 << (N_power - 1);

        location_t global_addresss =
            block_thread + (location_t) ((blockDim.y * block_x) << N_power);

        location_t omega_addresss = idx_x;

        // Load T from global & store to shared
        shared_memory[block_thread] = polynomial_in[global_addresss];
        shared_memory[block_thread + block_size] =
            polynomial_in[global_addresss + block_size];

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
                current_root_index = (omega_addresss >> t_2);
            }
            else
            { // X_N_plus
                current_root_index = m + (omega_addresss >> t_2);
            }

            GentlemanSandeUnit(shared_memory[group_in_shared_address],
                               shared_memory[group_in_shared_address + t],
                               inverse_root_of_unity_table[current_root_index],
                               modulus_reg);

            t = t << 1;
            t_2 += 1;
            t_ += 1;
            m >>= 1;

            in_shared_address =
                ((shared_addresss >> t_) << t_) + shared_addresss;
            __syncthreads();
        }
        __syncthreads();

        TU output1_reg = OPERATOR_GPU<TU>::mult(shared_memory[block_thread],
                                                n_inverse_reg, modulus_reg);
        TU output2_reg =
            OPERATOR_GPU<TU>::mult(shared_memory[block_thread + block_size],
                                   n_inverse_reg, modulus_reg);

        if constexpr (std::is_signed<T>::value)
        {
            polynomial_out[global_addresss] =
                OPERATOR_GPU<TU>::centered_reduction(output1_reg, modulus_reg);
            polynomial_out[global_addresss + block_size] =
                OPERATOR_GPU<TU>::centered_reduction(output2_reg, modulus_reg);
        }
        else
        {
            polynomial_out[global_addresss] = output1_reg;
            polynomial_out[global_addresss + block_size] = output2_reg;
        }
    }

    template <typename T>
    __global__ void ForwardCore(
        T* polynomial_in, typename std::make_unsigned<T>::type* polynomial_out,
        const Root<typename std::make_unsigned<
            T>::type>* __restrict__ root_of_unity_table,
        Modulus<typename std::make_unsigned<T>::type> modulus, int shared_index,
        int logm, int outer_iteration_count, int N_power, bool zero_padding,
        bool not_last_kernel, bool reduction_poly_check)
    {
        using TU = typename std::make_unsigned<T>::type;

        const int idx_x = threadIdx.x;
        const int idx_y = threadIdx.y;
        const int block_x = blockIdx.x;
        const int block_y = blockIdx.y;
        const int block_z = blockIdx.z;

        // extern __shared__ T shared_memory[];
        extern __shared__ char shared_memory_typed[];
        TU* shared_memory = reinterpret_cast<TU*>(shared_memory_typed);

        const Modulus<TU> modulus_reg = modulus;

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
            (location_t) (block_z << N_power);

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
                    current_root_index = (omega_addresss >> t_2);
                }
                else
                { // X_N_plus
                    current_root_index = m + (omega_addresss >> t_2);
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
                //__syncthreads();
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
                    current_root_index = (omega_addresss >> t_2);
                }
                else
                { // X_N_plus
                    current_root_index = m + (omega_addresss >> t_2);
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
                //__syncthreads();
            }
            __syncthreads();

#pragma unroll
            for (int lp = 0; lp < 6; lp++)
            {
                if (reduction_poly_check)
                { // X_N_minus
                    current_root_index = (omega_addresss >> t_2);
                }
                else
                { // X_N_plus
                    current_root_index = m + (omega_addresss >> t_2);
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

    template <typename T>
    __global__ void
    ForwardCore(T* polynomial_in,
                typename std::make_unsigned<T>::type* polynomial_out,
                const Root<typename std::make_unsigned<
                    T>::type>* __restrict__ root_of_unity_table,
                Modulus<typename std::make_unsigned<T>::type>* modulus,
                int shared_index, int logm, int outer_iteration_count,
                int N_power, bool zero_padding, bool not_last_kernel,
                bool reduction_poly_check, int mod_count)
    {
        using TU = typename std::make_unsigned<T>::type;

        const int idx_x = threadIdx.x;
        const int idx_y = threadIdx.y;
        const int block_x = blockIdx.x;
        const int block_y = blockIdx.y;
        const int block_z = blockIdx.z;

        const int mod_index = block_z % mod_count;

        // extern __shared__ T shared_memory[];
        extern __shared__ char shared_memory_typed[];
        TU* shared_memory = reinterpret_cast<TU*>(shared_memory_typed);

        const Modulus<TU> modulus_reg = modulus[mod_index];

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
            (location_t) (block_z << N_power);

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
                                         (location_t) (mod_index << N_power);
                }
                else
                { // X_N_plus
                    current_root_index = m + (omega_addresss >> t_2) +
                                         (location_t) (mod_index << N_power);
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
                //__syncthreads();
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
                                         (location_t) (mod_index << N_power);
                }
                else
                { // X_N_plus
                    current_root_index = m + (omega_addresss >> t_2) +
                                         (location_t) (mod_index << N_power);
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
                //__syncthreads();
            }
            __syncthreads();

#pragma unroll
            for (int lp = 0; lp < 6; lp++)
            {
                if (reduction_poly_check)
                { // X_N_minus
                    current_root_index = (omega_addresss >> t_2) +
                                         (location_t) (mod_index << N_power);
                }
                else
                { // X_N_plus
                    current_root_index = m + (omega_addresss >> t_2) +
                                         (location_t) (mod_index << N_power);
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

    template <typename T>
    __global__ void ForwardCore_(
        T* polynomial_in, typename std::make_unsigned<T>::type* polynomial_out,
        const Root<typename std::make_unsigned<
            T>::type>* __restrict__ root_of_unity_table,
        Modulus<typename std::make_unsigned<T>::type> modulus, int shared_index,
        int logm, int outer_iteration_count, int N_power, bool zero_padding,
        bool not_last_kernel, bool reduction_poly_check)
    {
        using TU = typename std::make_unsigned<T>::type;

        const int idx_x = threadIdx.x;
        const int idx_y = threadIdx.y;
        const int block_x = blockIdx.x;
        const int block_y = blockIdx.y;
        const int block_z = blockIdx.z;

        // extern __shared__ T shared_memory[];
        extern __shared__ char shared_memory_typed[];
        TU* shared_memory = reinterpret_cast<TU*>(shared_memory_typed);

        const Modulus<TU> modulus_reg = modulus;

        int t_2 = N_power - logm - 1;
        location_t offset = 1 << (N_power - logm - 1);
        int t_ = shared_index;
        location_t m = (location_t) 1 << logm;

        location_t global_addresss =
            idx_x +
            (location_t) (idx_y *
                          (offset / (1 << (outer_iteration_count - 1)))) +
            (location_t) (blockDim.x * block_y) +
            (location_t) (2 * block_x * offset) +
            (location_t) (block_z << N_power);
        location_t omega_addresss =
            idx_x +
            (location_t) (idx_y *
                          (offset / (1 << (outer_iteration_count - 1)))) +
            (location_t) (blockDim.x * block_y) +
            (location_t) (block_x * offset);
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
                    current_root_index = (omega_addresss >> t_2);
                }
                else
                { // X_N_plus
                    current_root_index = m + (omega_addresss >> t_2);
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
                //__syncthreads();
            }
            __syncthreads();
        }
        else
        {
#pragma unroll
            for (int lp = 0; lp < (shared_index - 5); lp++)
            {
                __syncthreads();
                if (reduction_poly_check)
                { // X_N_minus
                    current_root_index = (omega_addresss >> t_2);
                }
                else
                { // X_N_plus
                    current_root_index = m + (omega_addresss >> t_2);
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
                //__syncthreads();
            }
            __syncthreads();

#pragma unroll
            for (int lp = 0; lp < 6; lp++)
            {
                if (reduction_poly_check)
                { // X_N_minus
                    current_root_index = (omega_addresss >> t_2);
                }
                else
                { // X_N_plus
                    current_root_index = m + (omega_addresss >> t_2);
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

    template <typename T>
    __global__ void
    ForwardCore_(T* polynomial_in,
                 typename std::make_unsigned<T>::type* polynomial_out,
                 const Root<typename std::make_unsigned<
                     T>::type>* __restrict__ root_of_unity_table,
                 Modulus<typename std::make_unsigned<T>::type>* modulus,
                 int shared_index, int logm, int outer_iteration_count,
                 int N_power, bool zero_padding, bool not_last_kernel,
                 bool reduction_poly_check, int mod_count)
    {
        using TU = typename std::make_unsigned<T>::type;

        const int idx_x = threadIdx.x;
        const int idx_y = threadIdx.y;
        const int block_x = blockIdx.x;
        const int block_y = blockIdx.y;
        const int block_z = blockIdx.z;

        const int mod_index = block_z % mod_count;

        // extern __shared__ T shared_memory[];
        extern __shared__ char shared_memory_typed[];
        TU* shared_memory = reinterpret_cast<TU*>(shared_memory_typed);

        const Modulus<TU> modulus_reg = modulus[mod_index];

        int t_2 = N_power - logm - 1;
        location_t offset = 1 << (N_power - logm - 1);
        int t_ = shared_index;
        location_t m = (location_t) 1 << logm;

        location_t global_addresss =
            idx_x +
            (location_t) (idx_y *
                          (offset / (1 << (outer_iteration_count - 1)))) +
            (location_t) (blockDim.x * block_y) +
            (location_t) (2 * block_x * offset) +
            (location_t) (block_z << N_power);
        location_t omega_addresss =
            idx_x +
            (location_t) (idx_y *
                          (offset / (1 << (outer_iteration_count - 1)))) +
            (location_t) (blockDim.x * block_y) +
            (location_t) (block_x * offset);
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
                                         (location_t) (mod_index << N_power);
                }
                else
                { // X_N_plus
                    current_root_index = m + (omega_addresss >> t_2) +
                                         (location_t) (mod_index << N_power);
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
                //__syncthreads();
            }
            __syncthreads();
        }
        else
        {
#pragma unroll
            for (int lp = 0; lp < (shared_index - 5); lp++)
            {
                __syncthreads();
                if (reduction_poly_check)
                { // X_N_minus
                    current_root_index = (omega_addresss >> t_2) +
                                         (location_t) (mod_index << N_power);
                }
                else
                { // X_N_plus
                    current_root_index = m + (omega_addresss >> t_2) +
                                         (location_t) (mod_index << N_power);
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
                //__syncthreads();
            }
            __syncthreads();

#pragma unroll
            for (int lp = 0; lp < 6; lp++)
            {
                if (reduction_poly_check)
                { // X_N_minus
                    current_root_index = (omega_addresss >> t_2) +
                                         (location_t) (mod_index << N_power);
                }
                else
                { // X_N_plus
                    current_root_index = m + (omega_addresss >> t_2) +
                                         (location_t) (mod_index << N_power);
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

        const int idx_x = threadIdx.x;
        const int idx_y = threadIdx.y;
        const int block_x = blockIdx.x;
        const int block_y = blockIdx.y;
        const int block_z = blockIdx.z;

        // extern __shared__ T shared_memory[];
        extern __shared__ char shared_memory_typed[];
        TU* shared_memory = reinterpret_cast<TU*>(shared_memory_typed);

        const Modulus<TU> modulus_reg = modulus;
        const Ninverse<TU> n_inverse_reg = n_inverse;

        int t_2 = N_power - logm - 1;
        location_t offset = 1 << (N_power - k - 1);
        // int t_ = 9 - outer_iteration_count;
        int t_ = (shared_index + 1) - outer_iteration_count;
        int loops = outer_iteration_count;
        location_t m = (location_t) 1 << logm;

        location_t global_addresss =
            idx_x +
            (location_t) (idx_y *
                          (offset / (1 << (outer_iteration_count - 1)))) +
            (location_t) (blockDim.x * block_x) +
            (location_t) (2 * block_y * offset) +
            (location_t) (block_z << N_power);

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
                current_root_index = (omega_addresss >> t_2);
            }
            else
            { // X_N_plus
                current_root_index = m + (omega_addresss >> t_2);
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

        const int idx_x = threadIdx.x;
        const int idx_y = threadIdx.y;
        const int block_x = blockIdx.x;
        const int block_y = blockIdx.y;
        const int block_z = blockIdx.z;

        const int mod_index = block_z % mod_count;

        // extern __shared__ T shared_memory[];
        extern __shared__ char shared_memory_typed[];
        TU* shared_memory = reinterpret_cast<TU*>(shared_memory_typed);

        const Modulus<TU> modulus_reg = modulus[mod_index];
        const Ninverse<TU> n_inverse_reg = n_inverse[mod_index];

        int t_2 = N_power - logm - 1;
        location_t offset = 1 << (N_power - k - 1);
        // int t_ = 9 - outer_iteration_count;
        int t_ = (shared_index + 1) - outer_iteration_count;
        int loops = outer_iteration_count;
        location_t m = (location_t) 1 << logm;

        location_t global_addresss =
            idx_x +
            (location_t) (idx_y *
                          (offset / (1 << (outer_iteration_count - 1)))) +
            (location_t) (blockDim.x * block_x) +
            (location_t) (2 * block_y * offset) +
            (location_t) (block_z << N_power);

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
                                     (location_t) (mod_index << N_power);
            }
            else
            { // X_N_plus
                current_root_index = m + (omega_addresss >> t_2) +
                                     (location_t) (mod_index << N_power);
            }

            GentlemanSandeUnit(shared_memory[in_shared_address],
                               shared_memory[in_shared_address + t],
                               inverse_root_of_unity_table[current_root_index],
                               modulus[mod_index]);

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

        const int idx_x = threadIdx.x;
        const int idx_y = threadIdx.y;
        const int block_x = blockIdx.x;
        const int block_y = blockIdx.y;
        const int block_z = blockIdx.z;

        // extern __shared__ T shared_memory[];
        extern __shared__ char shared_memory_typed[];
        TU* shared_memory = reinterpret_cast<TU*>(shared_memory_typed);

        const Modulus<TU> modulus_reg = modulus;
        const Ninverse<TU> n_inverse_reg = n_inverse;

        int t_2 = N_power - logm - 1;
        location_t offset = 1 << (N_power - k - 1);
        // int t_ = 9 - outer_iteration_count;
        int t_ = (shared_index + 1) - outer_iteration_count;
        int loops = outer_iteration_count;
        location_t m = (location_t) 1 << logm;

        location_t global_addresss =
            idx_x +
            (location_t) (idx_y *
                          (offset / (1 << (outer_iteration_count - 1)))) +
            (location_t) (blockDim.x * block_y) +
            (location_t) (2 * block_x * offset) +
            (location_t) (block_z << N_power);

        location_t omega_addresss =
            idx_x +
            (location_t) (idx_y *
                          (offset / (1 << (outer_iteration_count - 1)))) +
            (location_t) (blockDim.x * block_y) +
            (location_t) (block_x * offset);
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
                current_root_index = (omega_addresss >> t_2);
            }
            else
            { // X_N_plus
                current_root_index = m + (omega_addresss >> t_2);
            }

            GentlemanSandeUnit(shared_memory[in_shared_address],
                               shared_memory[in_shared_address + t],
                               inverse_root_of_unity_table[current_root_index],
                               modulus);

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

        const int idx_x = threadIdx.x;
        const int idx_y = threadIdx.y;
        const int block_x = blockIdx.x;
        const int block_y = blockIdx.y;
        const int block_z = blockIdx.z;

        const int mod_index = block_z % mod_count;

        // extern __shared__ T shared_memory[];
        extern __shared__ char shared_memory_typed[];
        TU* shared_memory = reinterpret_cast<TU*>(shared_memory_typed);

        const Modulus<TU> modulus_reg = modulus[mod_index];
        const Ninverse<TU> n_inverse_reg = n_inverse[mod_index];

        int t_2 = N_power - logm - 1;
        location_t offset = 1 << (N_power - k - 1);
        // int t_ = 9 - outer_iteration_count;
        int t_ = (shared_index + 1) - outer_iteration_count;
        int loops = outer_iteration_count;
        location_t m = (location_t) 1 << logm;

        location_t global_addresss =
            idx_x +
            (location_t) (idx_y *
                          (offset / (1 << (outer_iteration_count - 1)))) +
            (location_t) (blockDim.x * block_y) +
            (location_t) (2 * block_x * offset) +
            (location_t) (block_z << N_power);

        location_t omega_addresss =
            idx_x +
            (location_t) (idx_y *
                          (offset / (1 << (outer_iteration_count - 1)))) +
            (location_t) (blockDim.x * block_y) +
            (location_t) (block_x * offset);
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
                                     (location_t) (mod_index << N_power);
            }
            else
            { // X_N_plus
                current_root_index = m + (omega_addresss >> t_2) +
                                     (location_t) (mod_index << N_power);
            }

            GentlemanSandeUnit(shared_memory[in_shared_address],
                               shared_memory[in_shared_address + t],
                               inverse_root_of_unity_table[current_root_index],
                               modulus[mod_index]);

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
        const int total_block_thread = blockDim.x * blockDim.y;

        extern __shared__ char shared_memory_typed[];
        TU* shared_memory = reinterpret_cast<TU*>(shared_memory_typed);
        TU* root_shared_memory = shared_memory + (1024);

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

        const int block_thread_index = idx_x + (idx_y * blockDim.x);
        const int ntt_thread_index = block_thread_index & ((1 << t_) - 1);
        const int ntt_block_index = block_thread_index >> t_;
        int offset = ntt_block_index << log_row;

        const int batch_index =
            (block_x * (total_block_thread >> t_)) + ntt_block_index;
        int mod_index = batch_index % mod_count;

        const Modulus<TU> modulus_thread = modulus[mod_index];

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
                current_root_index =
                    (omega_addresss >> t_) + (mod_index << log_row);
            }
            else
            { // X_N_plus
                current_root_index =
                    m + (omega_addresss >> t_) + (mod_index << log_row);
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
        const int total_block_thread = blockDim.x * blockDim.y;

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

        const int batch_index =
            (block_x * (total_block_thread >> t_)) + ntt_block_index;
        int mod_index = batch_index % mod_count;

        const Modulus<TU> modulus_thread = modulus[mod_index];
        const Ninverse<TU> n_inverse_thread = n_inverse[mod_index];

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
                current_root_index =
                    (omega_addresss >> t_) + (mod_index << log_row);
            }
            else
            { // X_N_plus
                current_root_index =
                    m + (omega_addresss >> t_) + (mod_index << log_row);
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
    __host__ void
    GPU_NTT(T* device_in, typename std::make_unsigned<T>::type* device_out,
            Root<typename std::make_unsigned<T>::type>* root_of_unity_table,
            Modulus<typename std::make_unsigned<T>::type> modulus,
            ntt_configuration<typename std::make_unsigned<T>::type> cfg,
            int batch_size)
    {
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
                        cfg.zero_padding,
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
                            cfg.n_power, cfg.zero_padding,
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
                                cfg.n_power, cfg.zero_padding,
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
                            cfg.n_power, cfg.zero_padding,
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
                                cfg.n_power, cfg.zero_padding,
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
                            cfg.n_power, cfg.zero_padding,
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
                        cfg.zero_padding,
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
                            cfg.n_power, cfg.zero_padding,
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
                                cfg.n_power, cfg.zero_padding,
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
                            cfg.n_power, cfg.zero_padding,
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
                                cfg.n_power, cfg.zero_padding,
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
                            cfg.n_power, cfg.zero_padding,
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
        int outer_iteration_count, int N_power, bool zero_padding,
        bool not_last_kernel, bool reduction_poly_check, int mod_count,
        int* order)
    {
        const int idx_x = threadIdx.x;
        const int idx_y = threadIdx.y;
        const int block_x = blockIdx.x;
        const int block_y = blockIdx.y;
        const int block_z = blockIdx.z;

        const int mod_index = block_z % mod_count;
        const int prime_index = order[mod_index];

        // extern __shared__ T shared_memory[];
        extern __shared__ char shared_memory_typed[];
        T* shared_memory = reinterpret_cast<T*>(shared_memory_typed);

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
            (location_t) (block_z << N_power);

        location_t omega_addresss =
            idx_x +
            (location_t) (idx_y *
                          (offset / (1 << (outer_iteration_count - 1)))) +
            (location_t) (blockDim.x * block_x) +
            (location_t) (block_y * offset);

        location_t shared_addresss = (idx_x + (idx_y * blockDim.x));

        // Load UInt64 from global & store to shared
        shared_memory[shared_addresss] = polynomial_in[global_addresss];
        shared_memory[shared_addresss + (blockDim.x * blockDim.y)] =
            polynomial_in[global_addresss + offset];

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
                                modulus[prime_index]);

                t = t >> 1;
                t_2 -= 1;
                t_ -= 1;
                m <<= 1;

                in_shared_address =
                    ((shared_addresss >> t_) << t_) + shared_addresss;
                //__syncthreads();
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
                                modulus[prime_index]);

                t = t >> 1;
                t_2 -= 1;
                t_ -= 1;
                m <<= 1;

                in_shared_address =
                    ((shared_addresss >> t_) << t_) + shared_addresss;
                //__syncthreads();
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
                                modulus[prime_index]);

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

    template <typename T>
    __global__ void ForwardCoreModulusOrdered_(
        T* polynomial_in, T* polynomial_out, Root<T>* root_of_unity_table,
        Modulus<T>* modulus, int shared_index, int logm,
        int outer_iteration_count, int N_power, bool zero_padding,
        bool not_last_kernel, bool reduction_poly_check, int mod_count,
        int* order)
    {
        const int idx_x = threadIdx.x;
        const int idx_y = threadIdx.y;
        const int block_x = blockIdx.x;
        const int block_y = blockIdx.y;
        const int block_z = blockIdx.z;

        const int mod_index = block_z % mod_count;
        const int prime_index = order[mod_index];

        // extern __shared__ T shared_memory[];
        extern __shared__ char shared_memory_typed[];
        T* shared_memory = reinterpret_cast<T*>(shared_memory_typed);

        int t_2 = N_power - logm - 1;
        location_t offset = 1 << (N_power - logm - 1);
        int t_ = shared_index;
        location_t m = (location_t) 1 << logm;

        location_t global_addresss =
            idx_x +
            (location_t) (idx_y *
                          (offset / (1 << (outer_iteration_count - 1)))) +
            (location_t) (blockDim.x * block_y) +
            (location_t) (2 * block_x * offset) +
            (location_t) (block_z << N_power);
        location_t omega_addresss =
            idx_x +
            (location_t) (idx_y *
                          (offset / (1 << (outer_iteration_count - 1)))) +
            (location_t) (blockDim.x * block_y) +
            (location_t) (block_x * offset);
        location_t shared_addresss = (idx_x + (idx_y * blockDim.x));

        // Load UInt64 from global & store to shared
        shared_memory[shared_addresss] = polynomial_in[global_addresss];
        shared_memory[shared_addresss + (blockDim.x * blockDim.y)] =
            polynomial_in[global_addresss + offset];

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
                                modulus[prime_index]);

                t = t >> 1;
                t_2 -= 1;
                t_ -= 1;
                m <<= 1;

                in_shared_address =
                    ((shared_addresss >> t_) << t_) + shared_addresss;
                //__syncthreads();
            }
            __syncthreads();
        }
        else
        {
#pragma unroll
            for (int lp = 0; lp < (shared_index - 5); lp++)
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
                                modulus[prime_index]);

                t = t >> 1;
                t_2 -= 1;
                t_ -= 1;
                m <<= 1;

                in_shared_address =
                    ((shared_addresss >> t_) << t_) + shared_addresss;
                //__syncthreads();
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
                                modulus[prime_index]);

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

    template <typename T>
    __global__ void InverseCoreModulusOrdered(
        T* polynomial_in, T* polynomial_out,
        Root<T>* inverse_root_of_unity_table, Modulus<T>* modulus,
        int shared_index, int logm, int k, int outer_iteration_count,
        int N_power, Ninverse<T>* n_inverse, bool last_kernel,
        bool reduction_poly_check, int mod_count, int* order)
    {
        const int idx_x = threadIdx.x;
        const int idx_y = threadIdx.y;
        const int block_x = blockIdx.x;
        const int block_y = blockIdx.y;
        const int block_z = blockIdx.z;

        const int mod_index = block_z % mod_count;
        const int prime_index = order[mod_index];

        // extern __shared__ T shared_memory[];
        extern __shared__ char shared_memory_typed[];
        T* shared_memory = reinterpret_cast<T*>(shared_memory_typed);

        int t_2 = N_power - logm - 1;
        location_t offset = 1 << (N_power - k - 1);
        // int t_ = 9 - outer_iteration_count;
        int t_ = (shared_index + 1) - outer_iteration_count;
        int loops = outer_iteration_count;
        location_t m = (location_t) 1 << logm;

        location_t global_addresss =
            idx_x +
            (location_t) (idx_y *
                          (offset / (1 << (outer_iteration_count - 1)))) +
            (location_t) (blockDim.x * block_x) +
            (location_t) (2 * block_y * offset) +
            (location_t) (block_z << N_power);

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
                               modulus[prime_index]);

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
            polynomial_out[global_addresss] = OPERATOR_GPU<T>::mult(
                shared_memory[shared_addresss], n_inverse[prime_index],
                modulus[prime_index]);
            polynomial_out[global_addresss + offset] = OPERATOR_GPU<T>::mult(
                shared_memory[shared_addresss + (blockDim.x * blockDim.y)],
                n_inverse[prime_index], modulus[prime_index]);
        }
        else
        {
            polynomial_out[global_addresss] = shared_memory[shared_addresss];
            polynomial_out[global_addresss + offset] =
                shared_memory[shared_addresss + (blockDim.x * blockDim.y)];
        }
    }

    template <typename T>
    __global__ void InverseCoreModulusOrdered_(
        T* polynomial_in, T* polynomial_out,
        Root<T>* inverse_root_of_unity_table, Modulus<T>* modulus,
        int shared_index, int logm, int k, int outer_iteration_count,
        int N_power, Ninverse<T>* n_inverse, bool last_kernel,
        bool reduction_poly_check, int mod_count, int* order)
    {
        const int idx_x = threadIdx.x;
        const int idx_y = threadIdx.y;
        const int block_x = blockIdx.x;
        const int block_y = blockIdx.y;
        const int block_z = blockIdx.z;

        const int mod_index = block_z % mod_count;
        const int prime_index = order[mod_index];

        // extern __shared__ T shared_memory[];
        extern __shared__ char shared_memory_typed[];
        T* shared_memory = reinterpret_cast<T*>(shared_memory_typed);

        int t_2 = N_power - logm - 1;
        location_t offset = 1 << (N_power - k - 1);
        // int t_ = 9 - outer_iteration_count;
        int t_ = (shared_index + 1) - outer_iteration_count;
        int loops = outer_iteration_count;
        location_t m = (location_t) 1 << logm;

        location_t global_addresss =
            idx_x +
            (location_t) (idx_y *
                          (offset / (1 << (outer_iteration_count - 1)))) +
            (location_t) (blockDim.x * block_y) +
            (location_t) (2 * block_x * offset) +
            (location_t) (block_z << N_power);

        location_t omega_addresss =
            idx_x +
            (location_t) (idx_y *
                          (offset / (1 << (outer_iteration_count - 1)))) +
            (location_t) (blockDim.x * block_y) +
            (location_t) (block_x * offset);
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
                               modulus[prime_index]);

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
            polynomial_out[global_addresss] = OPERATOR_GPU<T>::mult(
                shared_memory[shared_addresss], n_inverse[prime_index],
                modulus[prime_index]);
            polynomial_out[global_addresss + offset] = OPERATOR_GPU<T>::mult(
                shared_memory[shared_addresss + (blockDim.x * blockDim.y)],
                n_inverse[prime_index], modulus[prime_index]);
        }
        else
        {
            polynomial_out[global_addresss] = shared_memory[shared_addresss];
            polynomial_out[global_addresss + offset] =
                shared_memory[shared_addresss + (blockDim.x * blockDim.y)];
        }
    }

    template <typename T>
    __host__ void
    GPU_NTT_Modulus_Ordered(T* device_in, T* device_out,
                            Root<T>* root_of_unity_table, Modulus<T>* modulus,
                            ntt_rns_configuration<T> cfg, int batch_size,
                            int mod_count, int* order)
    {
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
                            cfg.n_power, cfg.zero_padding,
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
                            cfg.n_power, cfg.zero_padding,
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
                        cfg.n_power, cfg.zero_padding,
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
                           bool zero_padding, bool not_last_kernel,
                           bool reduction_poly_check, int mod_count, int* order)
    {
        const int idx_x = threadIdx.x;
        const int idx_y = threadIdx.y;
        const int block_x = blockIdx.x;
        const int block_y = blockIdx.y;
        const int block_z = blockIdx.z;

        const int mod_index = block_z % mod_count;
        const int input_index = order[block_z];

        // extern __shared__ T shared_memory[];
        extern __shared__ char shared_memory_typed[];
        T* shared_memory = reinterpret_cast<T*>(shared_memory_typed);

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
            (location_t) (input_index << N_power);

        location_t omega_addresss =
            idx_x +
            (location_t) (idx_y *
                          (offset / (1 << (outer_iteration_count - 1)))) +
            (location_t) (blockDim.x * block_x) +
            (location_t) (block_y * offset);

        location_t shared_addresss = (idx_x + (idx_y * blockDim.x));

        // Load UInt64 from global & store to shared
        shared_memory[shared_addresss] = polynomial_in[global_addresss];
        shared_memory[shared_addresss + (blockDim.x * blockDim.y)] =
            polynomial_in[global_addresss + offset];

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
                                         (location_t) (mod_index << N_power);
                }
                else
                { // X_N_plus
                    current_root_index = m + (omega_addresss >> t_2) +
                                         (location_t) (mod_index << N_power);
                }
                CooleyTukeyUnit(shared_memory[in_shared_address],
                                shared_memory[in_shared_address + t],
                                root_of_unity_table[current_root_index],
                                modulus[mod_index]);

                t = t >> 1;
                t_2 -= 1;
                t_ -= 1;
                m <<= 1;

                in_shared_address =
                    ((shared_addresss >> t_) << t_) + shared_addresss;
                //__syncthreads();
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
                                         (location_t) (mod_index << N_power);
                }
                else
                { // X_N_plus
                    current_root_index = m + (omega_addresss >> t_2) +
                                         (location_t) (mod_index << N_power);
                }

                CooleyTukeyUnit(shared_memory[in_shared_address],
                                shared_memory[in_shared_address + t],
                                root_of_unity_table[current_root_index],
                                modulus[mod_index]);

                t = t >> 1;
                t_2 -= 1;
                t_ -= 1;
                m <<= 1;

                in_shared_address =
                    ((shared_addresss >> t_) << t_) + shared_addresss;
                //__syncthreads();
            }
            __syncthreads();

#pragma unroll
            for (int lp = 0; lp < 6; lp++)
            {
                if (reduction_poly_check)
                { // X_N_minus
                    current_root_index = (omega_addresss >> t_2) +
                                         (location_t) (mod_index << N_power);
                }
                else
                { // X_N_plus
                    current_root_index = m + (omega_addresss >> t_2) +
                                         (location_t) (mod_index << N_power);
                }
                CooleyTukeyUnit(shared_memory[in_shared_address],
                                shared_memory[in_shared_address + t],
                                root_of_unity_table[current_root_index],
                                modulus[mod_index]);

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

    template <typename T>
    __global__ void ForwardCorePolyOrdered_(
        T* polynomial_in, T* polynomial_out, Root<T>* root_of_unity_table,
        Modulus<T>* modulus, int shared_index, int logm,
        int outer_iteration_count, int N_power, bool zero_padding,
        bool not_last_kernel, bool reduction_poly_check, int mod_count,
        int* order)
    {
        const int idx_x = threadIdx.x;
        const int idx_y = threadIdx.y;
        const int block_x = blockIdx.x;
        const int block_y = blockIdx.y;
        const int block_z = blockIdx.z;

        const int mod_index = block_z % mod_count;
        const int input_index = order[block_z];

        // extern __shared__ T shared_memory[];
        extern __shared__ char shared_memory_typed[];
        T* shared_memory = reinterpret_cast<T*>(shared_memory_typed);

        int t_2 = N_power - logm - 1;
        location_t offset = 1 << (N_power - logm - 1);
        int t_ = shared_index;
        location_t m = (location_t) 1 << logm;

        location_t global_addresss =
            idx_x +
            (location_t) (idx_y *
                          (offset / (1 << (outer_iteration_count - 1)))) +
            (location_t) (blockDim.x * block_y) +
            (location_t) (2 * block_x * offset) +
            (location_t) (input_index << N_power);
        location_t omega_addresss =
            idx_x +
            (location_t) (idx_y *
                          (offset / (1 << (outer_iteration_count - 1)))) +
            (location_t) (blockDim.x * block_y) +
            (location_t) (block_x * offset);
        location_t shared_addresss = (idx_x + (idx_y * blockDim.x));

        // Load UInt64 from global & store to shared
        shared_memory[shared_addresss] = polynomial_in[global_addresss];
        shared_memory[shared_addresss + (blockDim.x * blockDim.y)] =
            polynomial_in[global_addresss + offset];

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
                                         (location_t) (mod_index << N_power);
                }
                else
                { // X_N_plus
                    current_root_index = m + (omega_addresss >> t_2) +
                                         (location_t) (mod_index << N_power);
                }
                CooleyTukeyUnit(shared_memory[in_shared_address],
                                shared_memory[in_shared_address + t],
                                root_of_unity_table[current_root_index],
                                modulus[mod_index]);

                t = t >> 1;
                t_2 -= 1;
                t_ -= 1;
                m <<= 1;

                in_shared_address =
                    ((shared_addresss >> t_) << t_) + shared_addresss;
                //__syncthreads();
            }
            __syncthreads();
        }
        else
        {
#pragma unroll
            for (int lp = 0; lp < (shared_index - 5); lp++)
            {
                __syncthreads();
                if (reduction_poly_check)
                { // X_N_minus
                    current_root_index = (omega_addresss >> t_2) +
                                         (location_t) (mod_index << N_power);
                }
                else
                { // X_N_plus
                    current_root_index = m + (omega_addresss >> t_2) +
                                         (location_t) (mod_index << N_power);
                }
                CooleyTukeyUnit(shared_memory[in_shared_address],
                                shared_memory[in_shared_address + t],
                                root_of_unity_table[current_root_index],
                                modulus[mod_index]);

                t = t >> 1;
                t_2 -= 1;
                t_ -= 1;
                m <<= 1;

                in_shared_address =
                    ((shared_addresss >> t_) << t_) + shared_addresss;
                //__syncthreads();
            }
            __syncthreads();

#pragma unroll
            for (int lp = 0; lp < 6; lp++)
            {
                if (reduction_poly_check)
                { // X_N_minus
                    current_root_index = (omega_addresss >> t_2) +
                                         (location_t) (mod_index << N_power);
                }
                else
                { // X_N_plus
                    current_root_index = m + (omega_addresss >> t_2) +
                                         (location_t) (mod_index << N_power);
                }
                CooleyTukeyUnit(shared_memory[in_shared_address],
                                shared_memory[in_shared_address + t],
                                root_of_unity_table[current_root_index],
                                modulus[mod_index]);

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

    template <typename T>
    __global__ void
    InverseCorePolyOrdered(T* polynomial_in, T* polynomial_out,
                           Root<T>* inverse_root_of_unity_table,
                           Modulus<T>* modulus, int shared_index, int logm,
                           int k, int outer_iteration_count, int N_power,
                           Ninverse<T>* n_inverse, bool last_kernel,
                           bool reduction_poly_check, int mod_count, int* order)
    {
        const int idx_x = threadIdx.x;
        const int idx_y = threadIdx.y;
        const int block_x = blockIdx.x;
        const int block_y = blockIdx.y;
        const int block_z = blockIdx.z;

        const int mod_index = block_z % mod_count;
        const int input_index = order[block_z];

        // extern __shared__ T shared_memory[];
        extern __shared__ char shared_memory_typed[];
        T* shared_memory = reinterpret_cast<T*>(shared_memory_typed);

        int t_2 = N_power - logm - 1;
        location_t offset = 1 << (N_power - k - 1);
        // int t_ = 9 - outer_iteration_count;
        int t_ = (shared_index + 1) - outer_iteration_count;
        int loops = outer_iteration_count;
        location_t m = (location_t) 1 << logm;

        location_t global_addresss =
            idx_x +
            (location_t) (idx_y *
                          (offset / (1 << (outer_iteration_count - 1)))) +
            (location_t) (blockDim.x * block_x) +
            (location_t) (2 * block_y * offset) +
            (location_t) (input_index << N_power);

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
                                     (location_t) (mod_index << N_power);
            }
            else
            { // X_N_plus
                current_root_index = m + (omega_addresss >> t_2) +
                                     (location_t) (mod_index << N_power);
            }

            GentlemanSandeUnit(shared_memory[in_shared_address],
                               shared_memory[in_shared_address + t],
                               inverse_root_of_unity_table[current_root_index],
                               modulus[mod_index]);

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
            polynomial_out[global_addresss] =
                OPERATOR_GPU<T>::mult(shared_memory[shared_addresss],
                                      n_inverse[mod_index], modulus[mod_index]);
            polynomial_out[global_addresss + offset] = OPERATOR_GPU<T>::mult(
                shared_memory[shared_addresss + (blockDim.x * blockDim.y)],
                n_inverse[mod_index], modulus[mod_index]);
        }
        else
        {
            polynomial_out[global_addresss] = shared_memory[shared_addresss];
            polynomial_out[global_addresss + offset] =
                shared_memory[shared_addresss + (blockDim.x * blockDim.y)];
        }
    }

    template <typename T>
    __global__ void InverseCorePolyOrdered_(
        T* polynomial_in, T* polynomial_out,
        Root<T>* inverse_root_of_unity_table, Modulus<T>* modulus,
        int shared_index, int logm, int k, int outer_iteration_count,
        int N_power, Ninverse<T>* n_inverse, bool last_kernel,
        bool reduction_poly_check, int mod_count, int* order)
    {
        const int idx_x = threadIdx.x;
        const int idx_y = threadIdx.y;
        const int block_x = blockIdx.x;
        const int block_y = blockIdx.y;
        const int block_z = blockIdx.z;

        const int mod_index = block_z % mod_count;
        const int input_index = order[block_z];

        // extern __shared__ T shared_memory[];
        extern __shared__ char shared_memory_typed[];
        T* shared_memory = reinterpret_cast<T*>(shared_memory_typed);

        int t_2 = N_power - logm - 1;
        location_t offset = 1 << (N_power - k - 1);
        // int t_ = 9 - outer_iteration_count;
        int t_ = (shared_index + 1) - outer_iteration_count;
        int loops = outer_iteration_count;
        location_t m = (location_t) 1 << logm;

        location_t global_addresss =
            idx_x +
            (location_t) (idx_y *
                          (offset / (1 << (outer_iteration_count - 1)))) +
            (location_t) (blockDim.x * block_y) +
            (location_t) (2 * block_x * offset) +
            (location_t) (input_index << N_power);

        location_t omega_addresss =
            idx_x +
            (location_t) (idx_y *
                          (offset / (1 << (outer_iteration_count - 1)))) +
            (location_t) (blockDim.x * block_y) +
            (location_t) (block_x * offset);
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
                                     (location_t) (mod_index << N_power);
            }
            else
            { // X_N_plus
                current_root_index = m + (omega_addresss >> t_2) +
                                     (location_t) (mod_index << N_power);
            }

            GentlemanSandeUnit(shared_memory[in_shared_address],
                               shared_memory[in_shared_address + t],
                               inverse_root_of_unity_table[current_root_index],
                               modulus[mod_index]);

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
            polynomial_out[global_addresss] =
                OPERATOR_GPU<T>::mult(shared_memory[shared_addresss],
                                      n_inverse[mod_index], modulus[mod_index]);
            polynomial_out[global_addresss + offset] = OPERATOR_GPU<T>::mult(
                shared_memory[shared_addresss + (blockDim.x * blockDim.y)],
                n_inverse[mod_index], modulus[mod_index]);
        }
        else
        {
            polynomial_out[global_addresss] = shared_memory[shared_addresss];
            polynomial_out[global_addresss + offset] =
                shared_memory[shared_addresss + (blockDim.x * blockDim.y)];
        }
    }

    template <typename T>
    __host__ void
    GPU_NTT_Poly_Ordered(T* device_in, T* device_out,
                         Root<T>* root_of_unity_table, Modulus<T>* modulus,
                         ntt_rns_configuration<T> cfg, int batch_size,
                         int mod_count, int* order)
    {
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
                            cfg.n_power, cfg.zero_padding,
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
                            cfg.n_power, cfg.zero_padding,
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
                        cfg.n_power, cfg.zero_padding,
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
                            device_in, device_out, root_of_unity_table, modulus,
                            current_kernel_params.shared_index,
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
        bool zero_padding;
        Ninverse<Data32> mod_inverse;
        cudaStream_t stream;
    };

    template <> struct ntt_configuration<Data64>
    {
        int n_power;
        type ntt_type;
        NTTLayout ntt_layout;
        ReductionPolynomial reduction_poly;
        bool zero_padding;
        Ninverse<Data64> mod_inverse;
        cudaStream_t stream;
    };

    template <> struct ntt_rns_configuration<Data32>
    {
        int n_power;
        type ntt_type;
        NTTLayout ntt_layout;
        ReductionPolynomial reduction_poly;
        bool zero_padding;
        Ninverse<Data32>* mod_inverse;
        cudaStream_t stream;
    };

    template <> struct ntt_rns_configuration<Data64>
    {
        int n_power;
        type ntt_type;
        NTTLayout ntt_layout;
        ReductionPolynomial reduction_poly;
        bool zero_padding;
        Ninverse<Data64>* mod_inverse;
        cudaStream_t stream;
    };

    template __global__ void ForwardCoreLowRing<Data32>(
        Data32* polynomial_in, Data32* polynomial_out,
        const Root<Data32>* __restrict__ root_of_unity_table,
        Modulus<Data32> modulus, int shared_index, int N_power,
        bool zero_padding, bool reduction_poly_check, int total_batch);

    template __global__ void ForwardCoreLowRing<Data64>(
        Data64* polynomial_in, Data64* polynomial_out,
        const Root<Data64>* __restrict__ root_of_unity_table,
        Modulus<Data64> modulus, int shared_index, int N_power,
        bool zero_padding, bool reduction_poly_check, int total_batch);

    template __global__ void ForwardCoreLowRing<Data32s>(
        Data32s* polynomial_in, Data32* polynomial_out,
        const Root<Data32>* __restrict__ root_of_unity_table,
        Modulus<Data32> modulus, int shared_index, int N_power,
        bool zero_padding, bool reduction_poly_check, int total_batch);

    template __global__ void ForwardCoreLowRing<Data64s>(
        Data64s* polynomial_in, Data64* polynomial_out,
        const Root<Data64>* __restrict__ root_of_unity_table,
        Modulus<Data64> modulus, int shared_index, int N_power,
        bool zero_padding, bool reduction_poly_check, int total_batch);

    template __global__ void ForwardCoreLowRing<Data32>(
        Data32* polynomial_in, Data32* polynomial_out,
        const Root<Data32>* __restrict__ root_of_unity_table,
        Modulus<Data32>* modulus, int shared_index, int N_power,
        bool zero_padding, bool reduction_poly_check, int total_batch,
        int mod_count);

    template __global__ void ForwardCoreLowRing<Data64>(
        Data64* polynomial_in, Data64* polynomial_out,
        const Root<Data64>* __restrict__ root_of_unity_table,
        Modulus<Data64>* modulus, int shared_index, int N_power,
        bool zero_padding, bool reduction_poly_check, int total_batch,
        int mod_count);

    template __global__ void ForwardCoreLowRing<Data32s>(
        Data32s* polynomial_in, Data32* polynomial_out,
        const Root<Data32>* __restrict__ root_of_unity_table,
        Modulus<Data32>* modulus, int shared_index, int N_power,
        bool zero_padding, bool reduction_poly_check, int total_batch,
        int mod_count);

    template __global__ void ForwardCoreLowRing<Data64s>(
        Data64s* polynomial_in, Data64* polynomial_out,
        const Root<Data64>* __restrict__ root_of_unity_table,
        Modulus<Data64>* modulus, int shared_index, int N_power,
        bool zero_padding, bool reduction_poly_check, int total_batch,
        int mod_count);

    template __global__ void
    ForwardCore<Data32>(Data32* polynomial_in, Data32* polynomial_out,
                        const Root<Data32>* __restrict__ root_of_unity_table,
                        Modulus<Data32> modulus, int shared_index, int logm,
                        int outer_iteration_count, int N_power,
                        bool zero_padding, bool not_last_kernel,
                        bool reduction_poly_check);

    template __global__ void
    ForwardCore<Data64>(Data64* polynomial_in, Data64* polynomial_out,
                        const Root<Data64>* __restrict__ root_of_unity_table,
                        Modulus<Data64> modulus, int shared_index, int logm,
                        int outer_iteration_count, int N_power,
                        bool zero_padding, bool not_last_kernel,
                        bool reduction_poly_check);

    template __global__ void
    ForwardCore<Data32s>(Data32s* polynomial_in, Data32* polynomial_out,
                         const Root<Data32>* __restrict__ root_of_unity_table,
                         Modulus<Data32> modulus, int shared_index, int logm,
                         int outer_iteration_count, int N_power,
                         bool zero_padding, bool not_last_kernel,
                         bool reduction_poly_check);

    template __global__ void
    ForwardCore<Data64s>(Data64s* polynomial_in, Data64* polynomial_out,
                         const Root<Data64>* __restrict__ root_of_unity_table,
                         Modulus<Data64> modulus, int shared_index, int logm,
                         int outer_iteration_count, int N_power,
                         bool zero_padding, bool not_last_kernel,
                         bool reduction_poly_check);

    template __global__ void
    ForwardCore<Data32>(Data32* polynomial_in, Data32* polynomial_out,
                        const Root<Data32>* __restrict__ root_of_unity_table,
                        Modulus<Data32>* modulus, int shared_index, int logm,
                        int outer_iteration_count, int N_power,
                        bool zero_padding, bool not_last_kernel,
                        bool reduction_poly_check, int mod_count);

    template __global__ void
    ForwardCore<Data64>(Data64* polynomial_in, Data64* polynomial_out,
                        const Root<Data64>* __restrict__ root_of_unity_table,
                        Modulus<Data64>* modulus, int shared_index, int logm,
                        int outer_iteration_count, int N_power,
                        bool zero_padding, bool not_last_kernel,
                        bool reduction_poly_check, int mod_count);

    template __global__ void
    ForwardCore<Data32s>(Data32s* polynomial_in, Data32* polynomial_out,
                         const Root<Data32>* __restrict__ root_of_unity_table,
                         Modulus<Data32>* modulus, int shared_index, int logm,
                         int outer_iteration_count, int N_power,
                         bool zero_padding, bool not_last_kernel,
                         bool reduction_poly_check, int mod_count);

    template __global__ void
    ForwardCore<Data64s>(Data64s* polynomial_in, Data64* polynomial_out,
                         const Root<Data64>* __restrict__ root_of_unity_table,
                         Modulus<Data64>* modulus, int shared_index, int logm,
                         int outer_iteration_count, int N_power,
                         bool zero_padding, bool not_last_kernel,
                         bool reduction_poly_check, int mod_count);

    template __global__ void
    ForwardCore_<Data32>(Data32* polynomial_in, Data32* polynomial_out,
                         const Root<Data32>* __restrict__ root_of_unity_table,
                         Modulus<Data32> modulus, int shared_index, int logm,
                         int outer_iteration_count, int N_power,
                         bool zero_padding, bool not_last_kernel,
                         bool reduction_poly_check);

    template __global__ void
    ForwardCore_<Data64>(Data64* polynomial_in, Data64* polynomial_out,
                         const Root<Data64>* __restrict__ root_of_unity_table,
                         Modulus<Data64> modulus, int shared_index, int logm,
                         int outer_iteration_count, int N_power,
                         bool zero_padding, bool not_last_kernel,
                         bool reduction_poly_check);

    template __global__ void
    ForwardCore_<Data32s>(Data32s* polynomial_in, Data32* polynomial_out,
                          const Root<Data32>* __restrict__ root_of_unity_table,
                          Modulus<Data32> modulus, int shared_index, int logm,
                          int outer_iteration_count, int N_power,
                          bool zero_padding, bool not_last_kernel,
                          bool reduction_poly_check);

    template __global__ void
    ForwardCore_<Data64s>(Data64s* polynomial_in, Data64* polynomial_out,
                          const Root<Data64>* __restrict__ root_of_unity_table,
                          Modulus<Data64> modulus, int shared_index, int logm,
                          int outer_iteration_count, int N_power,
                          bool zero_padding, bool not_last_kernel,
                          bool reduction_poly_check);

    template __global__ void
    ForwardCore_<Data32>(Data32* polynomial_in, Data32* polynomial_out,
                         const Root<Data32>* __restrict__ root_of_unity_table,
                         Modulus<Data32>* modulus, int shared_index, int logm,
                         int outer_iteration_count, int N_power,
                         bool zero_padding, bool not_last_kernel,
                         bool reduction_poly_check, int mod_count);

    template __global__ void
    ForwardCore_<Data64>(Data64* polynomial_in, Data64* polynomial_out,
                         const Root<Data64>* __restrict__ root_of_unity_table,
                         Modulus<Data64>* modulus, int shared_index, int logm,
                         int outer_iteration_count, int N_power,
                         bool zero_padding, bool not_last_kernel,
                         bool reduction_poly_check, int mod_count);

    template __global__ void
    ForwardCore_<Data32s>(Data32s* polynomial_in, Data32* polynomial_out,
                          const Root<Data32>* __restrict__ root_of_unity_table,
                          Modulus<Data32>* modulus, int shared_index, int logm,
                          int outer_iteration_count, int N_power,
                          bool zero_padding, bool not_last_kernel,
                          bool reduction_poly_check, int mod_count);

    template __global__ void
    ForwardCore_<Data64s>(Data64s* polynomial_in, Data64* polynomial_out,
                          const Root<Data64>* __restrict__ root_of_unity_table,
                          Modulus<Data64>* modulus, int shared_index, int logm,
                          int outer_iteration_count, int N_power,
                          bool zero_padding, bool not_last_kernel,
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
        bool zero_padding, bool not_last_kernel, bool reduction_poly_check,
        int mod_count, int* order);

    template __global__ void ForwardCoreModulusOrdered_<Data32>(
        Data32* polynomial_in, Data32* polynomial_out,
        Root<Data32>* root_of_unity_table, Modulus<Data32>* modulus,
        int shared_index, int logm, int outer_iteration_count, int N_power,
        bool zero_padding, bool not_last_kernel, bool reduction_poly_check,
        int mod_count, int* order);

    template __global__ void ForwardCoreModulusOrdered<Data64>(
        Data64* polynomial_in, Data64* polynomial_out,
        Root<Data64>* root_of_unity_table, Modulus<Data64>* modulus,
        int shared_index, int logm, int outer_iteration_count, int N_power,
        bool zero_padding, bool not_last_kernel, bool reduction_poly_check,
        int mod_count, int* order);

    template __global__ void ForwardCoreModulusOrdered_<Data64>(
        Data64* polynomial_in, Data64* polynomial_out,
        Root<Data64>* root_of_unity_table, Modulus<Data64>* modulus,
        int shared_index, int logm, int outer_iteration_count, int N_power,
        bool zero_padding, bool not_last_kernel, bool reduction_poly_check,
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
        bool zero_padding, bool not_last_kernel, bool reduction_poly_check,
        int mod_count, int* order);

    template __global__ void ForwardCorePolyOrdered_<Data32>(
        Data32* polynomial_in, Data32* polynomial_out,
        Root<Data32>* root_of_unity_table, Modulus<Data32>* modulus,
        int shared_index, int logm, int outer_iteration_count, int N_power,
        bool zero_padding, bool not_last_kernel, bool reduction_poly_check,
        int mod_count, int* order);

    template __global__ void ForwardCorePolyOrdered<Data64>(
        Data64* polynomial_in, Data64* polynomial_out,
        Root<Data64>* root_of_unity_table, Modulus<Data64>* modulus,
        int shared_index, int logm, int outer_iteration_count, int N_power,
        bool zero_padding, bool not_last_kernel, bool reduction_poly_check,
        int mod_count, int* order);

    template __global__ void ForwardCorePolyOrdered_<Data64>(
        Data64* polynomial_in, Data64* polynomial_out,
        Root<Data64>* root_of_unity_table, Modulus<Data64>* modulus,
        int shared_index, int logm, int outer_iteration_count, int N_power,
        bool zero_padding, bool not_last_kernel, bool reduction_poly_check,
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
