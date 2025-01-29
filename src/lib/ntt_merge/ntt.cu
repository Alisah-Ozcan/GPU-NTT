// (C) Ulvetanna Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan
// Paper: https://eprint.iacr.org/2023/1410

#include "ntt.cuh"

template <typename T>
__device__ void CooleyTukeyUnit(T& U, T& V, const Root<T>& root,
                                const Modulus<T>& modulus)
{
    T u_ = U;
    T v_ = OPERATOR_GPU<T>::mult(V, root, modulus);

    U = OPERATOR_GPU<T>::add(u_, v_, modulus);
    V = OPERATOR_GPU<T>::sub(u_, v_, modulus);
}

template <typename T>
__device__ void GentlemanSandeUnit(T& U, T& V, const Root<T>& root,
                                   const Modulus<T>& modulus)
{
    T u_ = U;
    T v_ = V;

    U = OPERATOR_GPU<T>::add(u_, v_, modulus);

    v_ = OPERATOR_GPU<T>::sub(u_, v_, modulus);
    V = OPERATOR_GPU<T>::mult(v_, root, modulus);
}

template <typename T>
__global__ void
ForwardCore(T* polynomial_in, T* polynomial_out,
            const Root<T>* __restrict__ root_of_unity_table, Modulus<T> modulus,
            int shared_index, int logm, int outer_iteration_count, int N_power,
            bool zero_padding, bool not_last_kernel, bool reduction_poly_check)
{
    const int idx_x = threadIdx.x;
    const int idx_y = threadIdx.y;
    const int block_x = blockIdx.x;
    const int block_y = blockIdx.y;
    const int block_z = blockIdx.z;

    // extern __shared__ T shared_memory[];
    extern __shared__ char shared_memory_typed[];
    T* shared_memory = reinterpret_cast<T*>(shared_memory_typed);

    int t_2 = N_power - logm - 1;
    location_t offset = 1 << (N_power - logm - 1);
    int t_ = shared_index;
    location_t m = (location_t) 1 << logm;

    location_t global_addresss =
        idx_x +
        (location_t) (idx_y * (offset / (1 << (outer_iteration_count - 1)))) +
        (location_t) (blockDim.x * block_x) +
        (location_t) (2 * block_y * offset) + (location_t) (block_z << N_power);

    location_t omega_addresss =
        idx_x +
        (location_t) (idx_y * (offset / (1 << (outer_iteration_count - 1)))) +
        (location_t) (blockDim.x * block_x) + (location_t) (block_y * offset);

    location_t shared_addresss = (idx_x + (idx_y * blockDim.x));

    // Load T from global & store to shared
    shared_memory[shared_addresss] = polynomial_in[global_addresss];
    shared_memory[shared_addresss + (blockDim.x * blockDim.y)] =
        polynomial_in[global_addresss + offset];

    int t = 1 << t_;
    int in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
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
                            root_of_unity_table[current_root_index], modulus);

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
                            root_of_unity_table[current_root_index], modulus);

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
                            root_of_unity_table[current_root_index], modulus);

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
__global__ void ForwardCore(T* polynomial_in, T* polynomial_out,
                            const Root<T>* __restrict__ root_of_unity_table,
                            Modulus<T>* modulus, int shared_index, int logm,
                            int outer_iteration_count, int N_power,
                            bool zero_padding, bool not_last_kernel,
                            bool reduction_poly_check, int mod_count)
{
    const int idx_x = threadIdx.x;
    const int idx_y = threadIdx.y;
    const int block_x = blockIdx.x;
    const int block_y = blockIdx.y;
    const int block_z = blockIdx.z;

    const int mod_index = block_z % mod_count;

    // extern __shared__ T shared_memory[];
    extern __shared__ char shared_memory_typed[];
    T* shared_memory = reinterpret_cast<T*>(shared_memory_typed);

    int t_2 = N_power - logm - 1;
    location_t offset = 1 << (N_power - logm - 1);
    int t_ = shared_index;
    location_t m = (location_t) 1 << logm;

    location_t global_addresss =
        idx_x +
        (location_t) (idx_y * (offset / (1 << (outer_iteration_count - 1)))) +
        (location_t) (blockDim.x * block_x) +
        (location_t) (2 * block_y * offset) + (location_t) (block_z << N_power);

    location_t omega_addresss =
        idx_x +
        (location_t) (idx_y * (offset / (1 << (outer_iteration_count - 1)))) +
        (location_t) (blockDim.x * block_x) + (location_t) (block_y * offset);

    location_t shared_addresss = (idx_x + (idx_y * blockDim.x));

    // Load T from global & store to shared
    shared_memory[shared_addresss] = polynomial_in[global_addresss];
    shared_memory[shared_addresss + (blockDim.x * blockDim.y)] =
        polynomial_in[global_addresss + offset];

    int t = 1 << t_;
    int in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
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
__global__ void ForwardCore_(T* polynomial_in, T* polynomial_out,
                             const Root<T>* __restrict__ root_of_unity_table,
                             Modulus<T> modulus, int shared_index, int logm,
                             int outer_iteration_count, int N_power,
                             bool zero_padding, bool not_last_kernel,
                             bool reduction_poly_check)
{
    const int idx_x = threadIdx.x;
    const int idx_y = threadIdx.y;
    const int block_x = blockIdx.x;
    const int block_y = blockIdx.y;
    const int block_z = blockIdx.z;

    // extern __shared__ T shared_memory[];
    extern __shared__ char shared_memory_typed[];
    T* shared_memory = reinterpret_cast<T*>(shared_memory_typed);

    int t_2 = N_power - logm - 1;
    location_t offset = 1 << (N_power - logm - 1);
    int t_ = shared_index;
    location_t m = (location_t) 1 << logm;

    location_t global_addresss =
        idx_x +
        (location_t) (idx_y * (offset / (1 << (outer_iteration_count - 1)))) +
        (location_t) (blockDim.x * block_y) +
        (location_t) (2 * block_x * offset) + (location_t) (block_z << N_power);
    location_t omega_addresss =
        idx_x +
        (location_t) (idx_y * (offset / (1 << (outer_iteration_count - 1)))) +
        (location_t) (blockDim.x * block_y) + (location_t) (block_x * offset);
    location_t shared_addresss = (idx_x + (idx_y * blockDim.x));

    // Load T from global & store to shared
    shared_memory[shared_addresss] = polynomial_in[global_addresss];
    shared_memory[shared_addresss + (blockDim.x * blockDim.y)] =
        polynomial_in[global_addresss + offset];

    int t = 1 << t_;
    int in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
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
                            root_of_unity_table[current_root_index], modulus);

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
                            root_of_unity_table[current_root_index], modulus);

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
                            root_of_unity_table[current_root_index], modulus);

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
__global__ void ForwardCore_(T* polynomial_in, T* polynomial_out,
                             const Root<T>* __restrict__ root_of_unity_table,
                             Modulus<T>* modulus, int shared_index, int logm,
                             int outer_iteration_count, int N_power,
                             bool zero_padding, bool not_last_kernel,
                             bool reduction_poly_check, int mod_count)
{
    const int idx_x = threadIdx.x;
    const int idx_y = threadIdx.y;
    const int block_x = blockIdx.x;
    const int block_y = blockIdx.y;
    const int block_z = blockIdx.z;

    const int mod_index = block_z % mod_count;

    // extern __shared__ T shared_memory[];
    extern __shared__ char shared_memory_typed[];
    T* shared_memory = reinterpret_cast<T*>(shared_memory_typed);

    int t_2 = N_power - logm - 1;
    location_t offset = 1 << (N_power - logm - 1);
    int t_ = shared_index;
    location_t m = (location_t) 1 << logm;

    location_t global_addresss =
        idx_x +
        (location_t) (idx_y * (offset / (1 << (outer_iteration_count - 1)))) +
        (location_t) (blockDim.x * block_y) +
        (location_t) (2 * block_x * offset) + (location_t) (block_z << N_power);
    location_t omega_addresss =
        idx_x +
        (location_t) (idx_y * (offset / (1 << (outer_iteration_count - 1)))) +
        (location_t) (blockDim.x * block_y) + (location_t) (block_x * offset);
    location_t shared_addresss = (idx_x + (idx_y * blockDim.x));

    // Load T from global & store to shared
    shared_memory[shared_addresss] = polynomial_in[global_addresss];
    shared_memory[shared_addresss + (blockDim.x * blockDim.y)] =
        polynomial_in[global_addresss + offset];

    int t = 1 << t_;
    int in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
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
InverseCore(T* polynomial_in, T* polynomial_out,
            const Root<T>* __restrict__ inverse_root_of_unity_table,
            Modulus<T> modulus, int shared_index, int logm, int k,
            int outer_iteration_count, int N_power, Ninverse<T> n_inverse,
            bool last_kernel, bool reduction_poly_check)
{
    const int idx_x = threadIdx.x;
    const int idx_y = threadIdx.y;
    const int block_x = blockIdx.x;
    const int block_y = blockIdx.y;
    const int block_z = blockIdx.z;

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
        (location_t) (idx_y * (offset / (1 << (outer_iteration_count - 1)))) +
        (location_t) (blockDim.x * block_x) +
        (location_t) (2 * block_y * offset) + (location_t) (block_z << N_power);

    location_t omega_addresss =
        idx_x +
        (location_t) (idx_y * (offset / (1 << (outer_iteration_count - 1)))) +
        (location_t) (blockDim.x * block_x) + (location_t) (block_y * offset);
    location_t shared_addresss = (idx_x + (idx_y * blockDim.x));

    shared_memory[shared_addresss] = polynomial_in[global_addresss];
    shared_memory[shared_addresss + (blockDim.x * blockDim.y)] =
        polynomial_in[global_addresss + offset];

    int t = 1 << t_;
    int in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
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

        in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
    }
    __syncthreads();

    if (last_kernel)
    {
        polynomial_out[global_addresss] = OPERATOR_GPU<T>::mult(
            shared_memory[shared_addresss], n_inverse, modulus);
        polynomial_out[global_addresss + offset] = OPERATOR_GPU<T>::mult(
            shared_memory[shared_addresss + (blockDim.x * blockDim.y)],
            n_inverse, modulus);
    }
    else
    {
        polynomial_out[global_addresss] = shared_memory[shared_addresss];
        polynomial_out[global_addresss + offset] =
            shared_memory[shared_addresss + (blockDim.x * blockDim.y)];
    }
}

template <typename T>
__global__ void
InverseCore(T* polynomial_in, T* polynomial_out,
            const Root<T>* __restrict__ inverse_root_of_unity_table,
            Modulus<T>* modulus, int shared_index, int logm, int k,
            int outer_iteration_count, int N_power, Ninverse<T>* n_inverse,
            bool last_kernel, bool reduction_poly_check, int mod_count)
{
    const int idx_x = threadIdx.x;
    const int idx_y = threadIdx.y;
    const int block_x = blockIdx.x;
    const int block_y = blockIdx.y;
    const int block_z = blockIdx.z;

    const int mod_index = block_z % mod_count;

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
        (location_t) (idx_y * (offset / (1 << (outer_iteration_count - 1)))) +
        (location_t) (blockDim.x * block_x) +
        (location_t) (2 * block_y * offset) + (location_t) (block_z << N_power);

    location_t omega_addresss =
        idx_x +
        (location_t) (idx_y * (offset / (1 << (outer_iteration_count - 1)))) +
        (location_t) (blockDim.x * block_x) + (location_t) (block_y * offset);
    location_t shared_addresss = (idx_x + (idx_y * blockDim.x));

    shared_memory[shared_addresss] = polynomial_in[global_addresss];
    shared_memory[shared_addresss + (blockDim.x * blockDim.y)] =
        polynomial_in[global_addresss + offset];

    int t = 1 << t_;
    int in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
    location_t current_root_index;
#pragma unroll
    for (int lp = 0; lp < loops; lp++)
    {
        __syncthreads();
        if (reduction_poly_check)
        { // X_N_minus
            current_root_index =
                (omega_addresss >> t_2) + (location_t) (mod_index << N_power);
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

        in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
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
__global__ void
InverseCore_(T* polynomial_in, T* polynomial_out,
             const Root<T>* __restrict__ inverse_root_of_unity_table,
             Modulus<T> modulus, int shared_index, int logm, int k,
             int outer_iteration_count, int N_power, Ninverse<T> n_inverse,
             bool last_kernel, bool reduction_poly_check)
{
    const int idx_x = threadIdx.x;
    const int idx_y = threadIdx.y;
    const int block_x = blockIdx.x;
    const int block_y = blockIdx.y;
    const int block_z = blockIdx.z;

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
        (location_t) (idx_y * (offset / (1 << (outer_iteration_count - 1)))) +
        (location_t) (blockDim.x * block_y) +
        (location_t) (2 * block_x * offset) + (location_t) (block_z << N_power);

    location_t omega_addresss =
        idx_x +
        (location_t) (idx_y * (offset / (1 << (outer_iteration_count - 1)))) +
        (location_t) (blockDim.x * block_y) + (location_t) (block_x * offset);
    location_t shared_addresss = (idx_x + (idx_y * blockDim.x));

    shared_memory[shared_addresss] = polynomial_in[global_addresss];
    shared_memory[shared_addresss + (blockDim.x * blockDim.y)] =
        polynomial_in[global_addresss + offset];

    int t = 1 << t_;
    int in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
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

        in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
    }
    __syncthreads();

    if (last_kernel)
    {
        polynomial_out[global_addresss] = OPERATOR_GPU<T>::mult(
            shared_memory[shared_addresss], n_inverse, modulus);
        polynomial_out[global_addresss + offset] = OPERATOR_GPU<T>::mult(
            shared_memory[shared_addresss + (blockDim.x * blockDim.y)],
            n_inverse, modulus);
    }
    else
    {
        polynomial_out[global_addresss] = shared_memory[shared_addresss];
        polynomial_out[global_addresss + offset] =
            shared_memory[shared_addresss + (blockDim.x * blockDim.y)];
    }
}

template <typename T>
__global__ void
InverseCore_(T* polynomial_in, T* polynomial_out,
             const Root<T>* __restrict__ inverse_root_of_unity_table,
             Modulus<T>* modulus, int shared_index, int logm, int k,
             int outer_iteration_count, int N_power, Ninverse<T>* n_inverse,
             bool last_kernel, bool reduction_poly_check, int mod_count)
{
    const int idx_x = threadIdx.x;
    const int idx_y = threadIdx.y;
    const int block_x = blockIdx.x;
    const int block_y = blockIdx.y;
    const int block_z = blockIdx.z;

    const int mod_index = block_z % mod_count;

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
        (location_t) (idx_y * (offset / (1 << (outer_iteration_count - 1)))) +
        (location_t) (blockDim.x * block_y) +
        (location_t) (2 * block_x * offset) + (location_t) (block_z << N_power);

    location_t omega_addresss =
        idx_x +
        (location_t) (idx_y * (offset / (1 << (outer_iteration_count - 1)))) +
        (location_t) (blockDim.x * block_y) + (location_t) (block_x * offset);
    location_t shared_addresss = (idx_x + (idx_y * blockDim.x));

    shared_memory[shared_addresss] = polynomial_in[global_addresss];
    shared_memory[shared_addresss + (blockDim.x * blockDim.y)] =
        polynomial_in[global_addresss + offset];

    int t = 1 << t_;
    int in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
    location_t current_root_index;
#pragma unroll
    for (int lp = 0; lp < loops; lp++)
    {
        __syncthreads();
        if (reduction_poly_check)
        { // X_N_minus
            current_root_index =
                (omega_addresss >> t_2) + (location_t) (mod_index << N_power);
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

        in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
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
__host__ void GPU_NTT(T* device_in, T* device_out, Root<T>* root_of_unity_table,
                      Modulus<T> modulus, ntt_configuration<T> cfg,
                      int batch_size)
{
    if ((cfg.n_power <= 11 || cfg.n_power >= 29))
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
                for (int i = 0; i < kernel_parameters[cfg.n_power].size(); i++)
                {
                    auto& current_kernel_params =
                        kernel_parameters[cfg.n_power][i];
                    ForwardCore<<<
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
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    device_in_ = device_out;
                }
            }
            else
            {
                for (int i = 0; i < kernel_parameters[cfg.n_power].size() - 1;
                     i++)
                {
                    auto& current_kernel_params =
                        kernel_parameters[cfg.n_power][i];
                    ForwardCore<<<
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
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    device_in_ = device_out;
                }
                auto& current_kernel_params =
                    kernel_parameters[cfg.n_power]
                                     [kernel_parameters[cfg.n_power].size() -
                                      1];
                ForwardCore_<<<
                    dim3(current_kernel_params.griddim_x,
                         current_kernel_params.griddim_y, batch_size),
                    dim3(current_kernel_params.blockdim_x,
                         current_kernel_params.blockdim_y),
                    current_kernel_params.shared_memory, cfg.stream>>>(
                    device_in_, device_out, root_of_unity_table, modulus,
                    current_kernel_params.shared_index,
                    current_kernel_params.logm,
                    current_kernel_params.outer_iteration_count, cfg.n_power,
                    cfg.zero_padding, current_kernel_params.not_last_kernel,
                    (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
            }
            break;
        case INVERSE:
            if (standart_kernel)
            {
                for (int i = 0; i < kernel_parameters[cfg.n_power].size(); i++)
                {
                    auto& current_kernel_params =
                        kernel_parameters[cfg.n_power][i];
                    InverseCore<<<
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
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    device_in_ = device_out;
                }
            }
            else
            {
                auto& current_kernel_params = kernel_parameters[cfg.n_power][0];
                InverseCore_<<<
                    dim3(current_kernel_params.griddim_x,
                         current_kernel_params.griddim_y, batch_size),
                    dim3(current_kernel_params.blockdim_x,
                         current_kernel_params.blockdim_y),
                    current_kernel_params.shared_memory, cfg.stream>>>(
                    device_in_, device_out, root_of_unity_table, modulus,
                    current_kernel_params.shared_index,
                    current_kernel_params.logm, current_kernel_params.k,
                    current_kernel_params.outer_iteration_count, cfg.n_power,
                    cfg.mod_inverse, current_kernel_params.not_last_kernel,
                    (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                device_in_ = device_out;
                for (int i = 1; i < kernel_parameters[cfg.n_power].size(); i++)
                {
                    auto& current_kernel_params =
                        kernel_parameters[cfg.n_power][i];
                    InverseCore<<<
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
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                }
            }

            break;
        default:
            throw std::invalid_argument("Invalid ntt_type!");
            break;
    }
}

template <typename T>
__host__ void GPU_NTT(T* device_in, T* device_out, Root<T>* root_of_unity_table,
                      Modulus<T>* modulus, ntt_rns_configuration<T> cfg,
                      int batch_size, int mod_count)
{
    if ((cfg.n_power <= 11 || cfg.n_power >= 29))
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
                for (int i = 0; i < kernel_parameters[cfg.n_power].size(); i++)
                {
                    auto& current_kernel_params =
                        kernel_parameters[cfg.n_power][i];
                    ForwardCore<<<
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
                        mod_count);
                    device_in_ = device_out;
                }
            }
            else
            {
                for (int i = 0; i < kernel_parameters[cfg.n_power].size() - 1;
                     i++)
                {
                    auto& current_kernel_params =
                        kernel_parameters[cfg.n_power][i];
                    ForwardCore<<<
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
                        mod_count);
                    device_in_ = device_out;
                }
                auto& current_kernel_params =
                    kernel_parameters[cfg.n_power]
                                     [kernel_parameters[cfg.n_power].size() -
                                      1];
                ForwardCore_<<<
                    dim3(current_kernel_params.griddim_x,
                         current_kernel_params.griddim_y, batch_size),
                    dim3(current_kernel_params.blockdim_x,
                         current_kernel_params.blockdim_y),
                    current_kernel_params.shared_memory, cfg.stream>>>(
                    device_in_, device_out, root_of_unity_table, modulus,
                    current_kernel_params.shared_index,
                    current_kernel_params.logm,
                    current_kernel_params.outer_iteration_count, cfg.n_power,
                    cfg.zero_padding, current_kernel_params.not_last_kernel,
                    (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                    mod_count);
            }
            break;
        case INVERSE:
            if (standart_kernel)
            {
                for (int i = 0; i < kernel_parameters[cfg.n_power].size(); i++)
                {
                    auto& current_kernel_params =
                        kernel_parameters[cfg.n_power][i];
                    InverseCore<<<
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
                        mod_count);
                    device_in_ = device_out;
                }
            }
            else
            {
                auto& current_kernel_params = kernel_parameters[cfg.n_power][0];
                InverseCore_<<<
                    dim3(current_kernel_params.griddim_x,
                         current_kernel_params.griddim_y, batch_size),
                    dim3(current_kernel_params.blockdim_x,
                         current_kernel_params.blockdim_y),
                    current_kernel_params.shared_memory, cfg.stream>>>(
                    device_in_, device_out, root_of_unity_table, modulus,
                    current_kernel_params.shared_index,
                    current_kernel_params.logm, current_kernel_params.k,
                    current_kernel_params.outer_iteration_count, cfg.n_power,
                    cfg.mod_inverse, current_kernel_params.not_last_kernel,
                    (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                    mod_count);
                device_in_ = device_out;
                for (int i = 1; i < kernel_parameters[cfg.n_power].size(); i++)
                {
                    auto& current_kernel_params =
                        kernel_parameters[cfg.n_power][i];
                    InverseCore<<<
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
                        mod_count);
                }
            }
            break;
        default:
            throw std::invalid_argument("Invalid ntt_type!");
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
                              Modulus<T>* modulus, ntt_rns_configuration<T> cfg,
                              int batch_size, int mod_count)
{
    GPU_NTT(device_inout, device_inout, root_of_unity_table, modulus, cfg,
            batch_size, mod_count);
}

////////////////////////////////////
// Modulus Ordered
////////////////////////////////////

template <typename T>
__global__ void
ForwardCoreModulusOrdered(T* polynomial_in, T* polynomial_out,
                          Root<T>* root_of_unity_table, Modulus<T>* modulus,
                          int shared_index, int logm, int outer_iteration_count,
                          int N_power, bool zero_padding, bool not_last_kernel,
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
    location_t offset = 1 << (N_power - logm - 1);
    int t_ = shared_index;
    location_t m = (location_t) 1 << logm;

    location_t global_addresss =
        idx_x +
        (location_t) (idx_y * (offset / (1 << (outer_iteration_count - 1)))) +
        (location_t) (blockDim.x * block_x) +
        (location_t) (2 * block_y * offset) + (location_t) (block_z << N_power);

    location_t omega_addresss =
        idx_x +
        (location_t) (idx_y * (offset / (1 << (outer_iteration_count - 1)))) +
        (location_t) (blockDim.x * block_x) + (location_t) (block_y * offset);

    location_t shared_addresss = (idx_x + (idx_y * blockDim.x));

    // Load Data64 from global & store to shared
    shared_memory[shared_addresss] = polynomial_in[global_addresss];
    shared_memory[shared_addresss + (blockDim.x * blockDim.y)] =
        polynomial_in[global_addresss + offset];

    int t = 1 << t_;
    int in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
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
    Modulus<T>* modulus, int shared_index, int logm, int outer_iteration_count,
    int N_power, bool zero_padding, bool not_last_kernel,
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
    location_t offset = 1 << (N_power - logm - 1);
    int t_ = shared_index;
    location_t m = (location_t) 1 << logm;

    location_t global_addresss =
        idx_x +
        (location_t) (idx_y * (offset / (1 << (outer_iteration_count - 1)))) +
        (location_t) (blockDim.x * block_y) +
        (location_t) (2 * block_x * offset) + (location_t) (block_z << N_power);
    location_t omega_addresss =
        idx_x +
        (location_t) (idx_y * (offset / (1 << (outer_iteration_count - 1)))) +
        (location_t) (blockDim.x * block_y) + (location_t) (block_x * offset);
    location_t shared_addresss = (idx_x + (idx_y * blockDim.x));

    // Load Data64 from global & store to shared
    shared_memory[shared_addresss] = polynomial_in[global_addresss];
    shared_memory[shared_addresss + (blockDim.x * blockDim.y)] =
        polynomial_in[global_addresss + offset];

    int t = 1 << t_;
    int in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
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
    T* polynomial_in, T* polynomial_out, Root<T>* inverse_root_of_unity_table,
    Modulus<T>* modulus, int shared_index, int logm, int k,
    int outer_iteration_count, int N_power, Ninverse<T>* n_inverse,
    bool last_kernel, bool reduction_poly_check, int mod_count, int* order)
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
        (location_t) (idx_y * (offset / (1 << (outer_iteration_count - 1)))) +
        (location_t) (blockDim.x * block_x) +
        (location_t) (2 * block_y * offset) + (location_t) (block_z << N_power);

    location_t omega_addresss =
        idx_x +
        (location_t) (idx_y * (offset / (1 << (outer_iteration_count - 1)))) +
        (location_t) (blockDim.x * block_x) + (location_t) (block_y * offset);
    location_t shared_addresss = (idx_x + (idx_y * blockDim.x));

    shared_memory[shared_addresss] = polynomial_in[global_addresss];
    shared_memory[shared_addresss + (blockDim.x * blockDim.y)] =
        polynomial_in[global_addresss + offset];

    int t = 1 << t_;
    int in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
    location_t current_root_index;
#pragma unroll
    for (int lp = 0; lp < loops; lp++)
    {
        __syncthreads();
        if (reduction_poly_check)
        { // X_N_minus
            current_root_index =
                (omega_addresss >> t_2) + (location_t) (prime_index << N_power);
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

        in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
    }
    __syncthreads();

    if (last_kernel)
    {
        polynomial_out[global_addresss] =
            OPERATOR_GPU<T>::mult(shared_memory[shared_addresss],
                                  n_inverse[prime_index], modulus[prime_index]);
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
    T* polynomial_in, T* polynomial_out, Root<T>* inverse_root_of_unity_table,
    Modulus<T>* modulus, int shared_index, int logm, int k,
    int outer_iteration_count, int N_power, Ninverse<T>* n_inverse,
    bool last_kernel, bool reduction_poly_check, int mod_count, int* order)
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
        (location_t) (idx_y * (offset / (1 << (outer_iteration_count - 1)))) +
        (location_t) (blockDim.x * block_y) +
        (location_t) (2 * block_x * offset) + (location_t) (block_z << N_power);

    location_t omega_addresss =
        idx_x +
        (location_t) (idx_y * (offset / (1 << (outer_iteration_count - 1)))) +
        (location_t) (blockDim.x * block_y) + (location_t) (block_x * offset);
    location_t shared_addresss = (idx_x + (idx_y * blockDim.x));

    shared_memory[shared_addresss] = polynomial_in[global_addresss];
    shared_memory[shared_addresss + (blockDim.x * blockDim.y)] =
        polynomial_in[global_addresss + offset];

    int t = 1 << t_;
    int in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
    location_t current_root_index;
#pragma unroll
    for (int lp = 0; lp < loops; lp++)
    {
        __syncthreads();
        if (reduction_poly_check)
        { // X_N_minus
            current_root_index =
                (omega_addresss >> t_2) + (location_t) (prime_index << N_power);
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

        in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
    }
    __syncthreads();

    if (last_kernel)
    {
        polynomial_out[global_addresss] =
            OPERATOR_GPU<T>::mult(shared_memory[shared_addresss],
                                  n_inverse[prime_index], modulus[prime_index]);
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
__host__ void GPU_NTT_Modulus_Ordered(T* device_in, T* device_out,
                                      Root<T>* root_of_unity_table,
                                      Modulus<T>* modulus,
                                      ntt_rns_configuration<T> cfg,
                                      int batch_size, int mod_count, int* order)
{
    if ((cfg.n_power <= 11 || cfg.n_power >= 29))
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
                for (int i = 0; i < kernel_parameters[cfg.n_power].size(); i++)
                {
                    auto& current_kernel_params =
                        kernel_parameters[cfg.n_power][i];
                    ForwardCoreModulusOrdered<<<
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
                    device_in_ = device_out;
                }
            }
            else
            {
                for (int i = 0; i < kernel_parameters[cfg.n_power].size() - 1;
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
                        device_in_, device_out, root_of_unity_table, modulus,
                        current_kernel_params.shared_index,
                        current_kernel_params.logm,
                        current_kernel_params.outer_iteration_count,
                        cfg.n_power, cfg.zero_padding,
                        current_kernel_params.not_last_kernel,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                        mod_count, order);
                    device_in_ = device_out;
                }
                auto& current_kernel_params =
                    kernel_parameters[cfg.n_power]
                                     [kernel_parameters[cfg.n_power].size() -
                                      1];
                ForwardCoreModulusOrdered_<<<
                    dim3(current_kernel_params.griddim_x,
                         current_kernel_params.griddim_y, batch_size),
                    dim3(current_kernel_params.blockdim_x,
                         current_kernel_params.blockdim_y),
                    current_kernel_params.shared_memory, cfg.stream>>>(
                    device_in_, device_out, root_of_unity_table, modulus,
                    current_kernel_params.shared_index,
                    current_kernel_params.logm,
                    current_kernel_params.outer_iteration_count, cfg.n_power,
                    cfg.zero_padding, current_kernel_params.not_last_kernel,
                    (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                    mod_count, order);
            }
            break;
        case INVERSE:
            if (standart_kernel)
            {
                for (int i = 0; i < kernel_parameters[cfg.n_power].size(); i++)
                {
                    auto& current_kernel_params =
                        kernel_parameters[cfg.n_power][i];
                    InverseCoreModulusOrdered<<<
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
                    device_in_ = device_out;
                }
            }
            else
            {
                auto& current_kernel_params = kernel_parameters[cfg.n_power][0];
                InverseCoreModulusOrdered_<<<
                    dim3(current_kernel_params.griddim_x,
                         current_kernel_params.griddim_y, batch_size),
                    dim3(current_kernel_params.blockdim_x,
                         current_kernel_params.blockdim_y),
                    current_kernel_params.shared_memory, cfg.stream>>>(
                    device_in_, device_out, root_of_unity_table, modulus,
                    current_kernel_params.shared_index,
                    current_kernel_params.logm, current_kernel_params.k,
                    current_kernel_params.outer_iteration_count, cfg.n_power,
                    cfg.mod_inverse, current_kernel_params.not_last_kernel,
                    (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                    mod_count, order);
                device_in_ = device_out;
                for (int i = 1; i < kernel_parameters[cfg.n_power].size(); i++)
                {
                    auto& current_kernel_params =
                        kernel_parameters[cfg.n_power][i];
                    InverseCoreModulusOrdered<<<
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
                       int shared_index, int logm, int outer_iteration_count,
                       int N_power, bool zero_padding, bool not_last_kernel,
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
        (location_t) (idx_y * (offset / (1 << (outer_iteration_count - 1)))) +
        (location_t) (blockDim.x * block_x) +
        (location_t) (2 * block_y * offset) +
        (location_t) (input_index << N_power);

    location_t omega_addresss =
        idx_x +
        (location_t) (idx_y * (offset / (1 << (outer_iteration_count - 1)))) +
        (location_t) (blockDim.x * block_x) + (location_t) (block_y * offset);

    location_t shared_addresss = (idx_x + (idx_y * blockDim.x));

    // Load Data64 from global & store to shared
    shared_memory[shared_addresss] = polynomial_in[global_addresss];
    shared_memory[shared_addresss + (blockDim.x * blockDim.y)] =
        polynomial_in[global_addresss + offset];

    int t = 1 << t_;
    int in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
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
__global__ void
ForwardCorePolyOrdered_(T* polynomial_in, T* polynomial_out,
                        Root<T>* root_of_unity_table, Modulus<T>* modulus,
                        int shared_index, int logm, int outer_iteration_count,
                        int N_power, bool zero_padding, bool not_last_kernel,
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
        (location_t) (idx_y * (offset / (1 << (outer_iteration_count - 1)))) +
        (location_t) (blockDim.x * block_y) +
        (location_t) (2 * block_x * offset) +
        (location_t) (input_index << N_power);
    location_t omega_addresss =
        idx_x +
        (location_t) (idx_y * (offset / (1 << (outer_iteration_count - 1)))) +
        (location_t) (blockDim.x * block_y) + (location_t) (block_x * offset);
    location_t shared_addresss = (idx_x + (idx_y * blockDim.x));

    // Load Data64 from global & store to shared
    shared_memory[shared_addresss] = polynomial_in[global_addresss];
    shared_memory[shared_addresss + (blockDim.x * blockDim.y)] =
        polynomial_in[global_addresss + offset];

    int t = 1 << t_;
    int in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
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
__global__ void InverseCorePolyOrdered(
    T* polynomial_in, T* polynomial_out, Root<T>* inverse_root_of_unity_table,
    Modulus<T>* modulus, int shared_index, int logm, int k,
    int outer_iteration_count, int N_power, Ninverse<T>* n_inverse,
    bool last_kernel, bool reduction_poly_check, int mod_count, int* order)
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
        (location_t) (idx_y * (offset / (1 << (outer_iteration_count - 1)))) +
        (location_t) (blockDim.x * block_x) +
        (location_t) (2 * block_y * offset) +
        (location_t) (input_index << N_power);

    location_t omega_addresss =
        idx_x +
        (location_t) (idx_y * (offset / (1 << (outer_iteration_count - 1)))) +
        (location_t) (blockDim.x * block_x) + (location_t) (block_y * offset);
    location_t shared_addresss = (idx_x + (idx_y * blockDim.x));

    shared_memory[shared_addresss] = polynomial_in[global_addresss];
    shared_memory[shared_addresss + (blockDim.x * blockDim.y)] =
        polynomial_in[global_addresss + offset];

    int t = 1 << t_;
    int in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
    location_t current_root_index;
#pragma unroll
    for (int lp = 0; lp < loops; lp++)
    {
        __syncthreads();
        if (reduction_poly_check)
        { // X_N_minus
            current_root_index =
                (omega_addresss >> t_2) + (location_t) (mod_index << N_power);
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

        in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
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
    T* polynomial_in, T* polynomial_out, Root<T>* inverse_root_of_unity_table,
    Modulus<T>* modulus, int shared_index, int logm, int k,
    int outer_iteration_count, int N_power, Ninverse<T>* n_inverse,
    bool last_kernel, bool reduction_poly_check, int mod_count, int* order)
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
        (location_t) (idx_y * (offset / (1 << (outer_iteration_count - 1)))) +
        (location_t) (blockDim.x * block_y) +
        (location_t) (2 * block_x * offset) +
        (location_t) (input_index << N_power);

    location_t omega_addresss =
        idx_x +
        (location_t) (idx_y * (offset / (1 << (outer_iteration_count - 1)))) +
        (location_t) (blockDim.x * block_y) + (location_t) (block_x * offset);
    location_t shared_addresss = (idx_x + (idx_y * blockDim.x));

    shared_memory[shared_addresss] = polynomial_in[global_addresss];
    shared_memory[shared_addresss + (blockDim.x * blockDim.y)] =
        polynomial_in[global_addresss + offset];

    int t = 1 << t_;
    int in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
    location_t current_root_index;
#pragma unroll
    for (int lp = 0; lp < loops; lp++)
    {
        __syncthreads();
        if (reduction_poly_check)
        { // X_N_minus
            current_root_index =
                (omega_addresss >> t_2) + (location_t) (mod_index << N_power);
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

        in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
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
GPU_NTT_Poly_Ordered(T* device_in, T* device_out, Root<T>* root_of_unity_table,
                     Modulus<T>* modulus, ntt_rns_configuration<T> cfg,
                     int batch_size, int mod_count, int* order)
{
    if ((cfg.n_power <= 11 || cfg.n_power >= 29))
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
                for (int i = 0; i < kernel_parameters[cfg.n_power].size(); i++)
                {
                    auto& current_kernel_params =
                        kernel_parameters[cfg.n_power][i];
                    ForwardCorePolyOrdered<<<
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
                    device_in_ = device_out;
                }
            }
            else
            {
                for (int i = 0; i < kernel_parameters[cfg.n_power].size() - 1;
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
                        device_in_, device_out, root_of_unity_table, modulus,
                        current_kernel_params.shared_index,
                        current_kernel_params.logm,
                        current_kernel_params.outer_iteration_count,
                        cfg.n_power, cfg.zero_padding,
                        current_kernel_params.not_last_kernel,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                        mod_count, order);
                    device_in_ = device_out;
                }
                auto& current_kernel_params =
                    kernel_parameters[cfg.n_power]
                                     [kernel_parameters[cfg.n_power].size() -
                                      1];
                ForwardCorePolyOrdered_<<<
                    dim3(current_kernel_params.griddim_x,
                         current_kernel_params.griddim_y, batch_size),
                    dim3(current_kernel_params.blockdim_x,
                         current_kernel_params.blockdim_y),
                    current_kernel_params.shared_memory, cfg.stream>>>(
                    device_in_, device_out, root_of_unity_table, modulus,
                    current_kernel_params.shared_index,
                    current_kernel_params.logm,
                    current_kernel_params.outer_iteration_count, cfg.n_power,
                    cfg.zero_padding, current_kernel_params.not_last_kernel,
                    (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                    mod_count, order);
            }
            break;
        case INVERSE:
            if (standart_kernel)
            {
                for (int i = 0; i < kernel_parameters[cfg.n_power].size(); i++)
                {
                    auto& current_kernel_params =
                        kernel_parameters[cfg.n_power][i];
                    InverseCorePolyOrdered<<<
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
                    device_in_ = device_out;
                }
            }
            else
            {
                auto& current_kernel_params = kernel_parameters[cfg.n_power][0];
                InverseCorePolyOrdered_<<<
                    dim3(current_kernel_params.griddim_x,
                         current_kernel_params.griddim_y, batch_size),
                    dim3(current_kernel_params.blockdim_x,
                         current_kernel_params.blockdim_y),
                    current_kernel_params.shared_memory, cfg.stream>>>(
                    device_in_, device_out, root_of_unity_table, modulus,
                    current_kernel_params.shared_index,
                    current_kernel_params.logm, current_kernel_params.k,
                    current_kernel_params.outer_iteration_count, cfg.n_power,
                    cfg.mod_inverse, current_kernel_params.not_last_kernel,
                    (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                    mod_count, order);
                device_in_ = device_out;
                for (int i = 1; i < kernel_parameters[cfg.n_power].size(); i++)
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
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                        mod_count, order);
                }
            }

            break;
        default:
            throw std::invalid_argument("Invalid ntt_type!");
            break;
    }
}

template <typename T>
__host__ void
GPU_NTT_Poly_Ordered_Inplace(T* device_inout, Root<T>* root_of_unity_table,
                             Modulus<T>* modulus, ntt_rns_configuration<T> cfg,
                             int batch_size, int mod_count, int* order)
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
    ReductionPolynomial reduction_poly;
    bool zero_padding;
    Ninverse<Data32> mod_inverse;
    cudaStream_t stream;
};

template <> struct ntt_configuration<Data64>
{
    int n_power;
    type ntt_type;
    ReductionPolynomial reduction_poly;
    bool zero_padding;
    Ninverse<Data64> mod_inverse;
    cudaStream_t stream;
};

template <> struct ntt_rns_configuration<Data32>
{
    int n_power;
    type ntt_type;
    ReductionPolynomial reduction_poly;
    bool zero_padding;
    Ninverse<Data32>* mod_inverse;
    cudaStream_t stream;
};

template <> struct ntt_rns_configuration<Data64>
{
    int n_power;
    type ntt_type;
    ReductionPolynomial reduction_poly;
    bool zero_padding;
    Ninverse<Data64>* mod_inverse;
    cudaStream_t stream;
};

template __device__ void
CooleyTukeyUnit<Data32>(Data32& U, Data32& V, const Root<Data32>& root,
                        const Modulus<Data32>& modulus);
template __device__ void
CooleyTukeyUnit<Data64>(Data64& U, Data64& V, const Root<Data64>& root,
                        const Modulus<Data64>& modulus);
template __device__ void
GentlemanSandeUnit<Data32>(Data32& U, Data32& V, const Root<Data32>& root,
                           const Modulus<Data32>& modulus);
template __device__ void
GentlemanSandeUnit<Data64>(Data64& U, Data64& V, const Root<Data64>& root,
                           const Modulus<Data64>& modulus);

template __global__ void
ForwardCore<Data32>(Data32* polynomial_in, Data32* polynomial_out,
                    const Root<Data32>* __restrict__ root_of_unity_table,
                    Modulus<Data32> modulus, int shared_index, int logm,
                    int outer_iteration_count, int N_power, bool zero_padding,
                    bool not_last_kernel, bool reduction_poly_check);

template __global__ void
ForwardCore<Data64>(Data64* polynomial_in, Data64* polynomial_out,
                    const Root<Data64>* __restrict__ root_of_unity_table,
                    Modulus<Data64> modulus, int shared_index, int logm,
                    int outer_iteration_count, int N_power, bool zero_padding,
                    bool not_last_kernel, bool reduction_poly_check);

template __global__ void
ForwardCore<Data32>(Data32* polynomial_in, Data32* polynomial_out,
                    const Root<Data32>* __restrict__ root_of_unity_table,
                    Modulus<Data32>* modulus, int shared_index, int logm,
                    int outer_iteration_count, int N_power, bool zero_padding,
                    bool not_last_kernel, bool reduction_poly_check,
                    int mod_count);

template __global__ void
ForwardCore<Data64>(Data64* polynomial_in, Data64* polynomial_out,
                    const Root<Data64>* __restrict__ root_of_unity_table,
                    Modulus<Data64>* modulus, int shared_index, int logm,
                    int outer_iteration_count, int N_power, bool zero_padding,
                    bool not_last_kernel, bool reduction_poly_check,
                    int mod_count);

template __global__ void
ForwardCore_<Data32>(Data32* polynomial_in, Data32* polynomial_out,
                     const Root<Data32>* __restrict__ root_of_unity_table,
                     Modulus<Data32> modulus, int shared_index, int logm,
                     int outer_iteration_count, int N_power, bool zero_padding,
                     bool not_last_kernel, bool reduction_poly_check);

template __global__ void
ForwardCore_<Data64>(Data64* polynomial_in, Data64* polynomial_out,
                     const Root<Data64>* __restrict__ root_of_unity_table,
                     Modulus<Data64> modulus, int shared_index, int logm,
                     int outer_iteration_count, int N_power, bool zero_padding,
                     bool not_last_kernel, bool reduction_poly_check);

template __global__ void
ForwardCore_<Data32>(Data32* polynomial_in, Data32* polynomial_out,
                     const Root<Data32>* __restrict__ root_of_unity_table,
                     Modulus<Data32>* modulus, int shared_index, int logm,
                     int outer_iteration_count, int N_power, bool zero_padding,
                     bool not_last_kernel, bool reduction_poly_check,
                     int mod_count);

template __global__ void
ForwardCore_<Data64>(Data64* polynomial_in, Data64* polynomial_out,
                     const Root<Data64>* __restrict__ root_of_unity_table,
                     Modulus<Data64>* modulus, int shared_index, int logm,
                     int outer_iteration_count, int N_power, bool zero_padding,
                     bool not_last_kernel, bool reduction_poly_check,
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

template __host__ void GPU_NTT<Data32>(Data32* device_in, Data32* device_out,
                                       Root<Data32>* root_of_unity_table,
                                       Modulus<Data32> modulus,
                                       ntt_configuration<Data32> cfg,
                                       int batch_size);

template __host__ void
GPU_NTT_Inplace<Data32>(Data32* device_inout, Root<Data32>* root_of_unity_table,
                        Modulus<Data32> modulus, ntt_configuration<Data32> cfg,
                        int batch_size);

template __host__ void GPU_NTT<Data64>(Data64* device_in, Data64* device_out,
                                       Root<Data64>* root_of_unity_table,
                                       Modulus<Data64> modulus,
                                       ntt_configuration<Data64> cfg,
                                       int batch_size);

template __host__ void
GPU_NTT_Inplace<Data64>(Data64* device_inout, Root<Data64>* root_of_unity_table,
                        Modulus<Data64> modulus, ntt_configuration<Data64> cfg,
                        int batch_size);

template __host__ void GPU_NTT<Data32>(Data32* device_in, Data32* device_out,
                                       Root<Data32>* root_of_unity_table,
                                       Modulus<Data32>* modulus,
                                       ntt_rns_configuration<Data32> cfg,
                                       int batch_size, int mod_count);

template __host__ void
GPU_NTT_Inplace<Data32>(Data32* device_inout, Root<Data32>* root_of_unity_table,
                        Modulus<Data32>* modulus,
                        ntt_rns_configuration<Data32> cfg, int batch_size,
                        int mod_count);

template __host__ void GPU_NTT<Data64>(Data64* device_in, Data64* device_out,
                                       Root<Data64>* root_of_unity_table,
                                       Modulus<Data64>* modulus,
                                       ntt_rns_configuration<Data64> cfg,
                                       int batch_size, int mod_count);

template __host__ void
GPU_NTT_Inplace<Data64>(Data64* device_inout, Root<Data64>* root_of_unity_table,
                        Modulus<Data64>* modulus,
                        ntt_rns_configuration<Data64> cfg, int batch_size,
                        int mod_count);

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
    int shared_index, int logm, int k, int outer_iteration_count, int N_power,
    Ninverse<Data32>* n_inverse, bool last_kernel, bool reduction_poly_check,
    int mod_count, int* order);

template __global__ void InverseCoreModulusOrdered_<Data32>(
    Data32* polynomial_in, Data32* polynomial_out,
    Root<Data32>* inverse_root_of_unity_table, Modulus<Data32>* modulus,
    int shared_index, int logm, int k, int outer_iteration_count, int N_power,
    Ninverse<Data32>* n_inverse, bool last_kernel, bool reduction_poly_check,
    int mod_count, int* order);

template __global__ void InverseCoreModulusOrdered<Data64>(
    Data64* polynomial_in, Data64* polynomial_out,
    Root<Data64>* inverse_root_of_unity_table, Modulus<Data64>* modulus,
    int shared_index, int logm, int k, int outer_iteration_count, int N_power,
    Ninverse<Data64>* n_inverse, bool last_kernel, bool reduction_poly_check,
    int mod_count, int* order);

template __global__ void InverseCoreModulusOrdered_<Data64>(
    Data64* polynomial_in, Data64* polynomial_out,
    Root<Data64>* inverse_root_of_unity_table, Modulus<Data64>* modulus,
    int shared_index, int logm, int k, int outer_iteration_count, int N_power,
    Ninverse<Data64>* n_inverse, bool last_kernel, bool reduction_poly_check,
    int mod_count, int* order);

template __host__ void GPU_NTT_Modulus_Ordered<Data32>(
    Data32* device_in, Data32* device_out, Root<Data32>* root_of_unity_table,
    Modulus<Data32>* modulus, ntt_rns_configuration<Data32> cfg, int batch_size,
    int mod_count, int* order);

template __host__ void GPU_NTT_Modulus_Ordered_Inplace<Data32>(
    Data32* device_inout, Root<Data32>* root_of_unity_table,
    Modulus<Data32>* modulus, ntt_rns_configuration<Data32> cfg, int batch_size,
    int mod_count, int* order);

template __host__ void GPU_NTT_Modulus_Ordered<Data64>(
    Data64* device_in, Data64* device_out, Root<Data64>* root_of_unity_table,
    Modulus<Data64>* modulus, ntt_rns_configuration<Data64> cfg, int batch_size,
    int mod_count, int* order);

template __host__ void GPU_NTT_Modulus_Ordered_Inplace<Data64>(
    Data64* device_inout, Root<Data64>* root_of_unity_table,
    Modulus<Data64>* modulus, ntt_rns_configuration<Data64> cfg, int batch_size,
    int mod_count, int* order);

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
    int shared_index, int logm, int k, int outer_iteration_count, int N_power,
    Ninverse<Data32>* n_inverse, bool last_kernel, bool reduction_poly_check,
    int mod_count, int* order);

template __global__ void InverseCorePolyOrdered_<Data32>(
    Data32* polynomial_in, Data32* polynomial_out,
    Root<Data32>* inverse_root_of_unity_table, Modulus<Data32>* modulus,
    int shared_index, int logm, int k, int outer_iteration_count, int N_power,
    Ninverse<Data32>* n_inverse, bool last_kernel, bool reduction_poly_check,
    int mod_count, int* order);

template __global__ void InverseCorePolyOrdered<Data64>(
    Data64* polynomial_in, Data64* polynomial_out,
    Root<Data64>* inverse_root_of_unity_table, Modulus<Data64>* modulus,
    int shared_index, int logm, int k, int outer_iteration_count, int N_power,
    Ninverse<Data64>* n_inverse, bool last_kernel, bool reduction_poly_check,
    int mod_count, int* order);

template __global__ void InverseCorePolyOrdered_<Data64>(
    Data64* polynomial_in, Data64* polynomial_out,
    Root<Data64>* inverse_root_of_unity_table, Modulus<Data64>* modulus,
    int shared_index, int logm, int k, int outer_iteration_count, int N_power,
    Ninverse<Data64>* n_inverse, bool last_kernel, bool reduction_poly_check,
    int mod_count, int* order);

template __host__ void GPU_NTT_Poly_Ordered<Data32>(
    Data32* device_in, Data32* device_out, Root<Data32>* root_of_unity_table,
    Modulus<Data32>* modulus, ntt_rns_configuration<Data32> cfg, int batch_size,
    int mod_count, int* order);

template __host__ void GPU_NTT_Poly_Ordered_Inplace<Data32>(
    Data32* device_inout, Root<Data32>* root_of_unity_table,
    Modulus<Data32>* modulus, ntt_rns_configuration<Data32> cfg, int batch_size,
    int mod_count, int* order);

template __host__ void GPU_NTT_Poly_Ordered<Data64>(
    Data64* device_in, Data64* device_out, Root<Data64>* root_of_unity_table,
    Modulus<Data64>* modulus, ntt_rns_configuration<Data64> cfg, int batch_size,
    int mod_count, int* order);

template __host__ void GPU_NTT_Poly_Ordered_Inplace<Data64>(
    Data64* device_inout, Root<Data64>* root_of_unity_table,
    Modulus<Data64>* modulus, ntt_rns_configuration<Data64> cfg, int batch_size,
    int mod_count, int* order);