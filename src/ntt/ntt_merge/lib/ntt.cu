// (C) Ulvetanna Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan
// Paper: https://eprint.iacr.org/2023/1410

#include "ntt.cuh"

#define CC_89  // for RTX 4090

__device__ void CooleyTukeyUnit(Data& U, Data& V, Root& root, Modulus& modulus)
{
    Data u_ = U;
    Data v_ = VALUE_GPU::mult(V, root, modulus);

    U = VALUE_GPU::add(u_, v_, modulus);
    V = VALUE_GPU::sub(u_, v_, modulus);
}

__device__ void GentlemanSandeUnit(Data& U, Data& V, Root& root, Modulus& modulus)
{
    Data u_ = U;
    Data v_ = V;

    U = VALUE_GPU::add(u_, v_, modulus);

    v_ = VALUE_GPU::sub(u_, v_, modulus);
    V = VALUE_GPU::mult(v_, root, modulus);
}

__global__ void ForwardCore(Data* polynomial_in, Data* polynomial_out, Root* root_of_unity_table,
                            Modulus modulus, int shared_index, int logm, int outer_iteration_count,
                            int N_power, bool zero_padding, bool not_last_kernel,
                            bool reduction_poly_check)
{
    const int idx_x = threadIdx.x;
    const int idx_y = threadIdx.y;
    const int block_x = blockIdx.x;
    const int block_y = blockIdx.y;
    const int block_z = blockIdx.z;

    extern __shared__ Data shared_memory[];

    int t_2 = N_power - logm - 1;
    location_t offset = 1 << (N_power - logm - 1);
    int t_ = shared_index;
    location_t m = (location_t)1 << logm;

    location_t global_addresss =
        idx_x + (location_t)(idx_y * (offset / (1 << (outer_iteration_count - 1)))) +
        (location_t)(blockDim.x * block_x) + (location_t)(2 * block_y * offset) +
        (location_t)(block_z << N_power);

    location_t omega_addresss =
        idx_x + (location_t)(idx_y * (offset / (1 << (outer_iteration_count - 1)))) +
        (location_t)(blockDim.x * block_x) + (location_t)(block_y * offset);

    location_t shared_addresss = (idx_x + (idx_y * blockDim.x));

    // Load data from global & store to shared
    shared_memory[shared_addresss] = polynomial_in[global_addresss];
    shared_memory[shared_addresss + (blockDim.x * blockDim.y)] =
        polynomial_in[global_addresss + offset];

    int t = 1 << t_;
    int in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
    location_t current_root_index;
    if(not_last_kernel)
    {
#pragma unroll
        for(int lp = 0; lp < outer_iteration_count; lp++)
        {
            __syncthreads();
            if(reduction_poly_check)
            {  // X_N_minus
                current_root_index = (omega_addresss >> t_2);
            }
            else
            {  // X_N_plus
                current_root_index = m + (omega_addresss >> t_2);
            }
            CooleyTukeyUnit(shared_memory[in_shared_address], shared_memory[in_shared_address + t],
                            root_of_unity_table[current_root_index], modulus);

            t = t >> 1;
            t_2 -= 1;
            t_ -= 1;
            m <<= 1;

            in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
            //__syncthreads();
        }
        __syncthreads();
    }
    else
    {
#pragma unroll
        for(int lp = 0; lp < (shared_index - 5); lp++)  // 4 for 512 thread
        {
            __syncthreads();
            if(reduction_poly_check)
            {  // X_N_minus
                current_root_index = (omega_addresss >> t_2);
            }
            else
            {  // X_N_plus
                current_root_index = m + (omega_addresss >> t_2);
            }

            CooleyTukeyUnit(shared_memory[in_shared_address], shared_memory[in_shared_address + t],
                            root_of_unity_table[current_root_index], modulus);

            t = t >> 1;
            t_2 -= 1;
            t_ -= 1;
            m <<= 1;

            in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
            //__syncthreads();
        }
        __syncthreads();

#pragma unroll
        for(int lp = 0; lp < 6; lp++)
        {
            if(reduction_poly_check)
            {  // X_N_minus
                current_root_index = (omega_addresss >> t_2);
            }
            else
            {  // X_N_plus
                current_root_index = m + (omega_addresss >> t_2);
            }
            CooleyTukeyUnit(shared_memory[in_shared_address], shared_memory[in_shared_address + t],
                            root_of_unity_table[current_root_index], modulus);

            t = t >> 1;
            t_2 -= 1;
            t_ -= 1;
            m <<= 1;

            in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
        }
        __syncthreads();
    }

    polynomial_out[global_addresss] = shared_memory[shared_addresss];
    polynomial_out[global_addresss + offset] =
        shared_memory[shared_addresss + (blockDim.x * blockDim.y)];
}

__global__ void ForwardCore(Data* polynomial_in, Data* polynomial_out, Root* root_of_unity_table,
                            Modulus* modulus, int shared_index, int logm, int outer_iteration_count,
                            int N_power, bool zero_padding, bool not_last_kernel,
                            bool reduction_poly_check, int mod_count)
{
    const int idx_x = threadIdx.x;
    const int idx_y = threadIdx.y;
    const int block_x = blockIdx.x;
    const int block_y = blockIdx.y;
    const int block_z = blockIdx.z;

    const int mod_index = block_z % mod_count;

    extern __shared__ Data shared_memory[];

    int t_2 = N_power - logm - 1;
    location_t offset = 1 << (N_power - logm - 1);
    int t_ = shared_index;
    location_t m = (location_t)1 << logm;

    location_t global_addresss =
        idx_x + (location_t)(idx_y * (offset / (1 << (outer_iteration_count - 1)))) +
        (location_t)(blockDim.x * block_x) + (location_t)(2 * block_y * offset) +
        (location_t)(block_z << N_power);

    location_t omega_addresss =
        idx_x + (location_t)(idx_y * (offset / (1 << (outer_iteration_count - 1)))) +
        (location_t)(blockDim.x * block_x) + (location_t)(block_y * offset);

    location_t shared_addresss = (idx_x + (idx_y * blockDim.x));

    // Load data from global & store to shared
    shared_memory[shared_addresss] = polynomial_in[global_addresss];
    shared_memory[shared_addresss + (blockDim.x * blockDim.y)] =
        polynomial_in[global_addresss + offset];

    int t = 1 << t_;
    int in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
    location_t current_root_index;
    if(not_last_kernel)
    {
#pragma unroll
        for(int lp = 0; lp < outer_iteration_count; lp++)
        {
            __syncthreads();
            if(reduction_poly_check)
            {  // X_N_minus
                current_root_index = (omega_addresss >> t_2) + (location_t)(mod_index << N_power);
            }
            else
            {  // X_N_plus
                current_root_index =
                    m + (omega_addresss >> t_2) + (location_t)(mod_index << N_power);
            }
            CooleyTukeyUnit(shared_memory[in_shared_address], shared_memory[in_shared_address + t],
                            root_of_unity_table[current_root_index], modulus[mod_index]);

            t = t >> 1;
            t_2 -= 1;
            t_ -= 1;
            m <<= 1;

            in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
            //__syncthreads();
        }
        __syncthreads();
    }
    else
    {
#pragma unroll
        for(int lp = 0; lp < (shared_index - 5); lp++)  // 4 for 512 thread
        {
            __syncthreads();
            if(reduction_poly_check)
            {  // X_N_minus
                current_root_index = (omega_addresss >> t_2) + (location_t)(mod_index << N_power);
            }
            else
            {  // X_N_plus
                current_root_index =
                    m + (omega_addresss >> t_2) + (location_t)(mod_index << N_power);
            }

            CooleyTukeyUnit(shared_memory[in_shared_address], shared_memory[in_shared_address + t],
                            root_of_unity_table[current_root_index], modulus[mod_index]);

            t = t >> 1;
            t_2 -= 1;
            t_ -= 1;
            m <<= 1;

            in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
            //__syncthreads();
        }
        __syncthreads();

#pragma unroll
        for(int lp = 0; lp < 6; lp++)
        {
            if(reduction_poly_check)
            {  // X_N_minus
                current_root_index = (omega_addresss >> t_2) + (location_t)(mod_index << N_power);
            }
            else
            {  // X_N_plus
                current_root_index =
                    m + (omega_addresss >> t_2) + (location_t)(mod_index << N_power);
            }
            CooleyTukeyUnit(shared_memory[in_shared_address], shared_memory[in_shared_address + t],
                            root_of_unity_table[current_root_index], modulus[mod_index]);

            t = t >> 1;
            t_2 -= 1;
            t_ -= 1;
            m <<= 1;

            in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
        }
        __syncthreads();
    }

    polynomial_out[global_addresss] = shared_memory[shared_addresss];
    polynomial_out[global_addresss + offset] =
        shared_memory[shared_addresss + (blockDim.x * blockDim.y)];
}

__global__ void ForwardCore_(Data* polynomial_in, Data* polynomial_out, Root* root_of_unity_table,
                             Modulus modulus, int shared_index, int logm, int outer_iteration_count,
                             int N_power, bool zero_padding, bool not_last_kernel,
                             bool reduction_poly_check)
{
    const int idx_x = threadIdx.x;
    const int idx_y = threadIdx.y;
    const int block_x = blockIdx.x;
    const int block_y = blockIdx.y;
    const int block_z = blockIdx.z;

    extern __shared__ Data shared_memory[];

    int t_2 = N_power - logm - 1;
    location_t offset = 1 << (N_power - logm - 1);
    int t_ = shared_index;
    location_t m = (location_t)1 << logm;

    location_t global_addresss =
        idx_x + (location_t)(idx_y * (offset / (1 << (outer_iteration_count - 1)))) +
        (location_t)(blockDim.x * block_y) + (location_t)(2 * block_x * offset) +
        (location_t)(block_z << N_power);
    location_t omega_addresss =
        idx_x + (location_t)(idx_y * (offset / (1 << (outer_iteration_count - 1)))) +
        (location_t)(blockDim.x * block_y) + (location_t)(block_x * offset);
    location_t shared_addresss = (idx_x + (idx_y * blockDim.x));

    // Load data from global & store to shared
    shared_memory[shared_addresss] = polynomial_in[global_addresss];
    shared_memory[shared_addresss + (blockDim.x * blockDim.y)] =
        polynomial_in[global_addresss + offset];

    int t = 1 << t_;
    int in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
    location_t current_root_index;
    if(not_last_kernel)
    {
#pragma unroll
        for(int lp = 0; lp < outer_iteration_count; lp++)
        {
            __syncthreads();
            if(reduction_poly_check)
            {  // X_N_minus
                current_root_index = (omega_addresss >> t_2);
            }
            else
            {  // X_N_plus
                current_root_index = m + (omega_addresss >> t_2);
            }
            CooleyTukeyUnit(shared_memory[in_shared_address], shared_memory[in_shared_address + t],
                            root_of_unity_table[current_root_index], modulus);

            t = t >> 1;
            t_2 -= 1;
            t_ -= 1;
            m <<= 1;

            in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
            //__syncthreads();
        }
        __syncthreads();
    }
    else
    {
#pragma unroll
        for(int lp = 0; lp < (shared_index - 5); lp++)
        {
            __syncthreads();
            if(reduction_poly_check)
            {  // X_N_minus
                current_root_index = (omega_addresss >> t_2);
            }
            else
            {  // X_N_plus
                current_root_index = m + (omega_addresss >> t_2);
            }

            CooleyTukeyUnit(shared_memory[in_shared_address], shared_memory[in_shared_address + t],
                            root_of_unity_table[current_root_index], modulus);

            t = t >> 1;
            t_2 -= 1;
            t_ -= 1;
            m <<= 1;

            in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
            //__syncthreads();
        }
        __syncthreads();

#pragma unroll
        for(int lp = 0; lp < 6; lp++)
        {
            if(reduction_poly_check)
            {  // X_N_minus
                current_root_index = (omega_addresss >> t_2);
            }
            else
            {  // X_N_plus
                current_root_index = m + (omega_addresss >> t_2);
            }
            CooleyTukeyUnit(shared_memory[in_shared_address], shared_memory[in_shared_address + t],
                            root_of_unity_table[current_root_index], modulus);

            t = t >> 1;
            t_2 -= 1;
            t_ -= 1;
            m <<= 1;

            in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
        }
        __syncthreads();
    }

    polynomial_out[global_addresss] = shared_memory[shared_addresss];
    polynomial_out[global_addresss + offset] =
        shared_memory[shared_addresss + (blockDim.x * blockDim.y)];
}

__global__ void ForwardCore_(Data* polynomial_in, Data* polynomial_out, Root* root_of_unity_table,
                             Modulus* modulus, int shared_index, int logm,
                             int outer_iteration_count, int N_power, bool zero_padding,
                             bool not_last_kernel, bool reduction_poly_check, int mod_count)
{
    const int idx_x = threadIdx.x;
    const int idx_y = threadIdx.y;
    const int block_x = blockIdx.x;
    const int block_y = blockIdx.y;
    const int block_z = blockIdx.z;

    const int mod_index = block_z % mod_count;

    extern __shared__ Data shared_memory[];

    int t_2 = N_power - logm - 1;
    location_t offset = 1 << (N_power - logm - 1);
    int t_ = shared_index;
    location_t m = (location_t)1 << logm;

    location_t global_addresss =
        idx_x + (location_t)(idx_y * (offset / (1 << (outer_iteration_count - 1)))) +
        (location_t)(blockDim.x * block_y) + (location_t)(2 * block_x * offset) +
        (location_t)(block_z << N_power);
    location_t omega_addresss =
        idx_x + (location_t)(idx_y * (offset / (1 << (outer_iteration_count - 1)))) +
        (location_t)(blockDim.x * block_y) + (location_t)(block_x * offset);
    location_t shared_addresss = (idx_x + (idx_y * blockDim.x));

    // Load data from global & store to shared
    shared_memory[shared_addresss] = polynomial_in[global_addresss];
    shared_memory[shared_addresss + (blockDim.x * blockDim.y)] =
        polynomial_in[global_addresss + offset];

    int t = 1 << t_;
    int in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
    location_t current_root_index;
    if(not_last_kernel)
    {
#pragma unroll
        for(int lp = 0; lp < outer_iteration_count; lp++)
        {
            __syncthreads();
            if(reduction_poly_check)
            {  // X_N_minus
                current_root_index = (omega_addresss >> t_2) + (location_t)(mod_index << N_power);
            }
            else
            {  // X_N_plus
                current_root_index =
                    m + (omega_addresss >> t_2) + (location_t)(mod_index << N_power);
            }
            CooleyTukeyUnit(shared_memory[in_shared_address], shared_memory[in_shared_address + t],
                            root_of_unity_table[current_root_index], modulus[mod_index]);

            t = t >> 1;
            t_2 -= 1;
            t_ -= 1;
            m <<= 1;

            in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
            //__syncthreads();
        }
        __syncthreads();
    }
    else
    {
#pragma unroll
        for(int lp = 0; lp < (shared_index - 5); lp++)
        {
            __syncthreads();
            if(reduction_poly_check)
            {  // X_N_minus
                current_root_index = (omega_addresss >> t_2) + (location_t)(mod_index << N_power);
            }
            else
            {  // X_N_plus
                current_root_index =
                    m + (omega_addresss >> t_2) + (location_t)(mod_index << N_power);
            }
            CooleyTukeyUnit(shared_memory[in_shared_address], shared_memory[in_shared_address + t],
                            root_of_unity_table[current_root_index], modulus[mod_index]);

            t = t >> 1;
            t_2 -= 1;
            t_ -= 1;
            m <<= 1;

            in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
            //__syncthreads();
        }
        __syncthreads();

#pragma unroll
        for(int lp = 0; lp < 6; lp++)
        {
            if(reduction_poly_check)
            {  // X_N_minus
                current_root_index = (omega_addresss >> t_2) + (location_t)(mod_index << N_power);
            }
            else
            {  // X_N_plus
                current_root_index =
                    m + (omega_addresss >> t_2) + (location_t)(mod_index << N_power);
            }
            CooleyTukeyUnit(shared_memory[in_shared_address], shared_memory[in_shared_address + t],
                            root_of_unity_table[current_root_index], modulus[mod_index]);

            t = t >> 1;
            t_2 -= 1;
            t_ -= 1;
            m <<= 1;

            in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
        }
        __syncthreads();
    }

    polynomial_out[global_addresss] = shared_memory[shared_addresss];
    polynomial_out[global_addresss + offset] =
        shared_memory[shared_addresss + (blockDim.x * blockDim.y)];
}

__global__ void InverseCore(Data* polynomial_in, Data* polynomial_out,
                            Root* inverse_root_of_unity_table, Modulus modulus, int shared_index,
                            int logm, int k, int outer_iteration_count, int N_power,
                            Ninverse n_inverse, bool last_kernel, bool reduction_poly_check)
{
    const int idx_x = threadIdx.x;
    const int idx_y = threadIdx.y;
    const int block_x = blockIdx.x;
    const int block_y = blockIdx.y;
    const int block_z = blockIdx.z;

    extern __shared__ Data shared_memory[];

    int t_2 = N_power - logm - 1;
    location_t offset = 1 << (N_power - k - 1);
    // int t_ = 9 - outer_iteration_count;
    int t_ = (shared_index + 1) - outer_iteration_count;
    int loops = outer_iteration_count;
    location_t m = (location_t)1 << logm;

    location_t global_addresss =
        idx_x + (location_t)(idx_y * (offset / (1 << (outer_iteration_count - 1)))) +
        (location_t)(blockDim.x * block_x) + (location_t)(2 * block_y * offset) +
        (location_t)(block_z << N_power);

    location_t omega_addresss =
        idx_x + (location_t)(idx_y * (offset / (1 << (outer_iteration_count - 1)))) +
        (location_t)(blockDim.x * block_x) + (location_t)(block_y * offset);
    location_t shared_addresss = (idx_x + (idx_y * blockDim.x));

    shared_memory[shared_addresss] = polynomial_in[global_addresss];
    shared_memory[shared_addresss + (blockDim.x * blockDim.y)] =
        polynomial_in[global_addresss + offset];

    int t = 1 << t_;
    int in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
    location_t current_root_index;
#pragma unroll
    for(int lp = 0; lp < loops; lp++)
    {
        __syncthreads();
        if(reduction_poly_check)
        {  // X_N_minus
            current_root_index = (omega_addresss >> t_2);
        }
        else
        {  // X_N_plus
            current_root_index = m + (omega_addresss >> t_2);
        }

        GentlemanSandeUnit(shared_memory[in_shared_address], shared_memory[in_shared_address + t],
                           inverse_root_of_unity_table[current_root_index], modulus);

        t = t << 1;
        t_2 += 1;
        t_ += 1;
        m >>= 1;

        in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
    }
    __syncthreads();

    if(last_kernel)
    {
        polynomial_out[global_addresss] =
            VALUE_GPU::mult(shared_memory[shared_addresss], n_inverse, modulus);
        polynomial_out[global_addresss + offset] = VALUE_GPU::mult(
            shared_memory[shared_addresss + (blockDim.x * blockDim.y)], n_inverse, modulus);
    }
    else
    {
        polynomial_out[global_addresss] = shared_memory[shared_addresss];
        polynomial_out[global_addresss + offset] =
            shared_memory[shared_addresss + (blockDim.x * blockDim.y)];
    }
}

__global__ void InverseCore(Data* polynomial_in, Data* polynomial_out,
                            Root* inverse_root_of_unity_table, Modulus* modulus, int shared_index,
                            int logm, int k, int outer_iteration_count, int N_power,
                            Ninverse* n_inverse, bool last_kernel, bool reduction_poly_check,
                            int mod_count)
{
    const int idx_x = threadIdx.x;
    const int idx_y = threadIdx.y;
    const int block_x = blockIdx.x;
    const int block_y = blockIdx.y;
    const int block_z = blockIdx.z;

    const int mod_index = block_z % mod_count;

    extern __shared__ Data shared_memory[];

    int t_2 = N_power - logm - 1;
    location_t offset = 1 << (N_power - k - 1);
    // int t_ = 9 - outer_iteration_count;
    int t_ = (shared_index + 1) - outer_iteration_count;
    int loops = outer_iteration_count;
    location_t m = (location_t)1 << logm;

    location_t global_addresss =
        idx_x + (location_t)(idx_y * (offset / (1 << (outer_iteration_count - 1)))) +
        (location_t)(blockDim.x * block_x) + (location_t)(2 * block_y * offset) +
        (location_t)(block_z << N_power);

    location_t omega_addresss =
        idx_x + (location_t)(idx_y * (offset / (1 << (outer_iteration_count - 1)))) +
        (location_t)(blockDim.x * block_x) + (location_t)(block_y * offset);
    location_t shared_addresss = (idx_x + (idx_y * blockDim.x));

    shared_memory[shared_addresss] = polynomial_in[global_addresss];
    shared_memory[shared_addresss + (blockDim.x * blockDim.y)] =
        polynomial_in[global_addresss + offset];

    int t = 1 << t_;
    int in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
    location_t current_root_index;
#pragma unroll
    for(int lp = 0; lp < loops; lp++)
    {
        __syncthreads();
        if(reduction_poly_check)
        {  // X_N_minus
            current_root_index = (omega_addresss >> t_2) + (location_t)(mod_index << N_power);
        }
        else
        {  // X_N_plus
            current_root_index = m + (omega_addresss >> t_2) + (location_t)(mod_index << N_power);
        }

        GentlemanSandeUnit(shared_memory[in_shared_address], shared_memory[in_shared_address + t],
                           inverse_root_of_unity_table[current_root_index], modulus[mod_index]);

        t = t << 1;
        t_2 += 1;
        t_ += 1;
        m >>= 1;

        in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
    }
    __syncthreads();

    if(last_kernel)
    {
        polynomial_out[global_addresss] = VALUE_GPU::mult(shared_memory[shared_addresss],
                                                          n_inverse[mod_index], modulus[mod_index]);
        polynomial_out[global_addresss + offset] =
            VALUE_GPU::mult(shared_memory[shared_addresss + (blockDim.x * blockDim.y)],
                            n_inverse[mod_index], modulus[mod_index]);
    }
    else
    {
        polynomial_out[global_addresss] = shared_memory[shared_addresss];
        polynomial_out[global_addresss + offset] =
            shared_memory[shared_addresss + (blockDim.x * blockDim.y)];
    }
}

__global__ void InverseCore_(Data* polynomial_in, Data* polynomial_out,
                             Root* inverse_root_of_unity_table, Modulus modulus, int shared_index,
                             int logm, int k, int outer_iteration_count, int N_power,
                             Ninverse n_inverse, bool last_kernel, bool reduction_poly_check)
{
    const int idx_x = threadIdx.x;
    const int idx_y = threadIdx.y;
    const int block_x = blockIdx.x;
    const int block_y = blockIdx.y;
    const int block_z = blockIdx.z;

    extern __shared__ Data shared_memory[];

    int t_2 = N_power - logm - 1;
    location_t offset = 1 << (N_power - k - 1);
    // int t_ = 9 - outer_iteration_count;
    int t_ = (shared_index + 1) - outer_iteration_count;
    int loops = outer_iteration_count;
    location_t m = (location_t)1 << logm;

    location_t global_addresss =
        idx_x + (location_t)(idx_y * (offset / (1 << (outer_iteration_count - 1)))) +
        (location_t)(blockDim.x * block_y) + (location_t)(2 * block_x * offset) +
        (location_t)(block_z << N_power);

    location_t omega_addresss =
        idx_x + (location_t)(idx_y * (offset / (1 << (outer_iteration_count - 1)))) +
        (location_t)(blockDim.x * block_y) + (location_t)(block_x * offset);
    location_t shared_addresss = (idx_x + (idx_y * blockDim.x));

    shared_memory[shared_addresss] = polynomial_in[global_addresss];
    shared_memory[shared_addresss + (blockDim.x * blockDim.y)] =
        polynomial_in[global_addresss + offset];

    int t = 1 << t_;
    int in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
    location_t current_root_index;
#pragma unroll
    for(int lp = 0; lp < loops; lp++)
    {
        __syncthreads();
        if(reduction_poly_check)
        {  // X_N_minus
            current_root_index = (omega_addresss >> t_2);
        }
        else
        {  // X_N_plus
            current_root_index = m + (omega_addresss >> t_2);
        }

        GentlemanSandeUnit(shared_memory[in_shared_address], shared_memory[in_shared_address + t],
                           inverse_root_of_unity_table[current_root_index], modulus);

        t = t << 1;
        t_2 += 1;
        t_ += 1;
        m >>= 1;

        in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
    }
    __syncthreads();

    if(last_kernel)
    {
        polynomial_out[global_addresss] =
            VALUE_GPU::mult(shared_memory[shared_addresss], n_inverse, modulus);
        polynomial_out[global_addresss + offset] = VALUE_GPU::mult(
            shared_memory[shared_addresss + (blockDim.x * blockDim.y)], n_inverse, modulus);
    }
    else
    {
        polynomial_out[global_addresss] = shared_memory[shared_addresss];
        polynomial_out[global_addresss + offset] =
            shared_memory[shared_addresss + (blockDim.x * blockDim.y)];
    }
}

__global__ void InverseCore_(Data* polynomial_in, Data* polynomial_out,
                             Root* inverse_root_of_unity_table, Modulus* modulus, int shared_index,
                             int logm, int k, int outer_iteration_count, int N_power,
                             Ninverse* n_inverse, bool last_kernel, bool reduction_poly_check,
                             int mod_count)
{
    const int idx_x = threadIdx.x;
    const int idx_y = threadIdx.y;
    const int block_x = blockIdx.x;
    const int block_y = blockIdx.y;
    const int block_z = blockIdx.z;

    const int mod_index = block_z % mod_count;

    extern __shared__ Data shared_memory[];

    int t_2 = N_power - logm - 1;
    location_t offset = 1 << (N_power - k - 1);
    // int t_ = 9 - outer_iteration_count;
    int t_ = (shared_index + 1) - outer_iteration_count;
    int loops = outer_iteration_count;
    location_t m = (location_t)1 << logm;

    location_t global_addresss =
        idx_x + (location_t)(idx_y * (offset / (1 << (outer_iteration_count - 1)))) +
        (location_t)(blockDim.x * block_y) + (location_t)(2 * block_x * offset) +
        (location_t)(block_z << N_power);

    location_t omega_addresss =
        idx_x + (location_t)(idx_y * (offset / (1 << (outer_iteration_count - 1)))) +
        (location_t)(blockDim.x * block_y) + (location_t)(block_x * offset);
    location_t shared_addresss = (idx_x + (idx_y * blockDim.x));

    shared_memory[shared_addresss] = polynomial_in[global_addresss];
    shared_memory[shared_addresss + (blockDim.x * blockDim.y)] =
        polynomial_in[global_addresss + offset];

    int t = 1 << t_;
    int in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
    location_t current_root_index;
#pragma unroll
    for(int lp = 0; lp < loops; lp++)
    {
        __syncthreads();
        if(reduction_poly_check)
        {  // X_N_minus
            current_root_index = (omega_addresss >> t_2) + (location_t)(mod_index << N_power);
        }
        else
        {  // X_N_plus
            current_root_index = m + (omega_addresss >> t_2) + (location_t)(mod_index << N_power);
        }

        GentlemanSandeUnit(shared_memory[in_shared_address], shared_memory[in_shared_address + t],
                           inverse_root_of_unity_table[current_root_index], modulus[mod_index]);

        t = t << 1;
        t_2 += 1;
        t_ += 1;
        m >>= 1;

        in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
    }
    __syncthreads();

    if(last_kernel)
    {
        polynomial_out[global_addresss] = VALUE_GPU::mult(shared_memory[shared_addresss],
                                                          n_inverse[mod_index], modulus[mod_index]);
        polynomial_out[global_addresss + offset] =
            VALUE_GPU::mult(shared_memory[shared_addresss + (blockDim.x * blockDim.y)],
                            n_inverse[mod_index], modulus[mod_index]);
    }
    else
    {
        polynomial_out[global_addresss] = shared_memory[shared_addresss];
        polynomial_out[global_addresss + offset] =
            shared_memory[shared_addresss + (blockDim.x * blockDim.y)];
    }
}

__host__ void GPU_NTT(Data* device_in, Data* device_out, Root* root_of_unity_table, Modulus modulus,
                      ntt_configuration cfg, int batch_size)
{
    switch(cfg.ntt_type)
    {
        case FORWARD:
            switch(cfg.n_power)
            {
                case 12:
                    ForwardCore<<<dim3(8, 1, batch_size), dim3(64, 4), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 0, 3, cfg.n_power,
                        cfg.zero_padding, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCore<<<dim3(1, 8, batch_size), dim3(256, 1), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 3, 9, cfg.n_power,
                        false, false, (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 13:
                    ForwardCore<<<dim3(16, 1, batch_size), dim3(32, 8), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 0, 4, cfg.n_power,
                        cfg.zero_padding, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCore<<<dim3(1, 16, batch_size), dim3(256, 1), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 4, 9, cfg.n_power,
                        false, false, (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 14:
                    ForwardCore<<<dim3(32, 1, batch_size), dim3(16, 16), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 0, 5, cfg.n_power,
                        cfg.zero_padding, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCore<<<dim3(1, 32, batch_size), dim3(256, 1), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 5, 9, cfg.n_power,
                        false, false, (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 15:
                    ForwardCore<<<dim3(64, 1, batch_size), dim3(8, 32), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 0, 6, cfg.n_power,
                        cfg.zero_padding, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCore<<<dim3(1, 64, batch_size), dim3(256, 1), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 6, 9, cfg.n_power,
                        false, false, (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 16:
                    ForwardCore<<<dim3(128, 1, batch_size), dim3(4, 64), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 0, 7, cfg.n_power,
                        cfg.zero_padding, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCore<<<dim3(1, 128, batch_size), dim3(256, 1), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 7, 9, cfg.n_power,
                        false, false, (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 17:
                    ForwardCore<<<dim3(256, 1, batch_size), dim3(32, 8), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 0, 4, cfg.n_power,
                        cfg.zero_padding, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCore<<<dim3(16, 16, batch_size), dim3(32, 8), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 4, 4, cfg.n_power,
                        false, true, (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCore<<<dim3(1, 256, batch_size), dim3(256, 1), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 8, 9, cfg.n_power,
                        false, false, (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 18:
                    ForwardCore<<<dim3(512, 1, batch_size), dim3(32, 8), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 0, 4, cfg.n_power,
                        cfg.zero_padding, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCore<<<dim3(32, 16, batch_size), dim3(16, 16), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 4, 5, cfg.n_power,
                        false, true, (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCore<<<dim3(1, 512, batch_size), dim3(256, 1), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 9, 9, cfg.n_power,
                        false, false, (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 19:
                    ForwardCore<<<dim3(1024, 1, batch_size), dim3(16, 16), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 0, 5, cfg.n_power,
                        cfg.zero_padding, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCore<<<dim3(32, 32, batch_size), dim3(16, 16), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 5, 5, cfg.n_power,
                        false, true, (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCore<<<dim3(1, 1024, batch_size), dim3(256, 1), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 10, 9, cfg.n_power,
                        false, false, (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 20:
                    ForwardCore<<<dim3(2048, 1, batch_size), dim3(16, 16), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 0, 5, cfg.n_power,
                        cfg.zero_padding, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCore<<<dim3(64, 32, batch_size), dim3(8, 32), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 5, 6, cfg.n_power,
                        false, true, (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCore<<<dim3(1, 2048, batch_size), dim3(256, 1), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 11, 9, cfg.n_power,
                        false, false, (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 21:
                    ForwardCore<<<dim3(4096, 1, batch_size), dim3(8, 32), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 0, 6, cfg.n_power,
                        cfg.zero_padding, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCore<<<dim3(64, 64, batch_size), dim3(8, 32), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 6, 6, cfg.n_power,
                        false, true, (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCore<<<dim3(1, 4096, batch_size), dim3(256, 1), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 12, 9, cfg.n_power,
                        false, false, (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 22:
                    ForwardCore<<<dim3(8192, 1, batch_size), dim3(8, 32), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 0, 6, cfg.n_power,
                        cfg.zero_padding, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCore<<<dim3(128, 64, batch_size), dim3(4, 64), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 6, 7, cfg.n_power,
                        false, true, (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCore<<<dim3(1, 8192, batch_size), dim3(256, 1), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 13, 9, cfg.n_power,
                        false, false, (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    THROW_IF_CUDA_ERROR(cudaGetLastError());

                    break;
                case 23:
                    ForwardCore<<<dim3(16384, 1, batch_size), dim3(4, 64), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 0, 7, cfg.n_power,
                        cfg.zero_padding, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCore<<<dim3(128, 128, batch_size), dim3(4, 64), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 7, 7, cfg.n_power,
                        false, true, (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCore<<<dim3(1, 16384, batch_size), dim3(256, 1), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 14, 9, cfg.n_power,
                        false, false, (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 24:
                    ForwardCore<<<dim3(16384, 1, batch_size), dim3(8, 64), 1024 * sizeof(Data),
                                  cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 9, 0, 7, cfg.n_power,
                        cfg.zero_padding, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCore<<<dim3(128, 128, batch_size), dim3(8, 64), 1024 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 9, 7, 7, cfg.n_power,
                        false, true, (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCore<<<dim3(1, 16384, batch_size), dim3(512, 1), 1024 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 9, 14, 10,
                        cfg.n_power, false, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 25:
                    ForwardCore<<<dim3(32768, 1, batch_size), dim3(8, 64), 1024 * sizeof(Data),
                                  cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 9, 0, 7, cfg.n_power,
                        cfg.zero_padding, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCore<<<dim3(256, 128, batch_size), dim3(4, 128), 1024 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 9, 7, 8, cfg.n_power,
                        false, true, (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ///////////////////////////////////////////////////////////
                    ForwardCore_<<<dim3(32768, 1, batch_size), dim3(512, 1), 1024 * sizeof(Data),
                                   cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 9, 15, 10,
                        cfg.n_power, false, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 26:
                    ForwardCore<<<dim3(65536, 1, batch_size), dim3(4, 128), 1024 * sizeof(Data),
                                  cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 9, 0, 8, cfg.n_power,
                        cfg.zero_padding, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCore<<<dim3(256, 256, batch_size), dim3(4, 128), 1024 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 9, 8, 8, cfg.n_power,
                        false, true, (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ///////////////////////////////////////////////////////////
                    ForwardCore_<<<dim3(65536, 1, batch_size), dim3(512, 1), 1024 * sizeof(Data),
                                   cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 9, 16, 10,
                        cfg.n_power, false, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 27:
#ifndef CC_89
                    ForwardCore<<<dim3(262144, 1, batch_size), dim3(16, 16), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 0, 5, cfg.n_power,
                        cfg.zero_padding, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCore<<<dim3(8192, 32, batch_size), dim3(8, 32), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 5, 6, cfg.n_power,
                        false, true, (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCore<<<dim3(128, 2048, batch_size), dim3(4, 64), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 11, 7, cfg.n_power,
                        false, true, (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ///////////////////////////////////////////////////////////
                    ForwardCore_<<<dim3(262144, 1, batch_size), dim3(256, 1), 512 * sizeof(Data),
                                   cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 18, 9, cfg.n_power,
                        false, false, (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
#else
                    ForwardCore<<<dim3(131072, 1, batch_size), dim3(4, 128), 1024 * sizeof(Data),
                                  cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 9, 0, 8, cfg.n_power,
                        cfg.zero_padding, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCore<<<dim3(512, 256, batch_size), dim3(2, 256), 1024 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 9, 8, 9, cfg.n_power,
                        false, true, (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ///////////////////////////////////////////////////////////
                    ForwardCore_<<<dim3(131072, 1, batch_size), dim3(512, 1), 1024 * sizeof(Data),
                                   cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 9, 17, 10,
                        cfg.n_power, false, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
#endif
                    break;
                case 28:
#ifndef CC_89
                    ForwardCore<<<dim3(524288, 1, batch_size), dim3(8, 32), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 0, 6, cfg.n_power,
                        cfg.zero_padding, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCore<<<dim3(8192, 64, batch_size), dim3(8, 32), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 6, 6, cfg.n_power,
                        false, true, (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCore<<<dim3(128, 4096, batch_size), dim3(4, 64), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 12, 7, cfg.n_power,
                        false, true, (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ///////////////////////////////////////////////////////////
                    ForwardCore_<<<dim3(524288, 1, batch_size), dim3(256, 1), 512 * sizeof(Data),
                                   cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 19, 9, cfg.n_power,
                        false, false, (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
#else
                    ForwardCore<<<dim3(262144, 1, batch_size), dim3(2, 256), 1024 * sizeof(Data),
                                  cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 9, 0, 9, cfg.n_power,
                        cfg.zero_padding, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCore<<<dim3(512, 512, batch_size), dim3(2, 256), 1024 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 9, 9, 9, cfg.n_power,
                        false, true, (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ///////////////////////////////////////////////////////////
                    ForwardCore_<<<dim3(262144, 1, batch_size), dim3(512, 1), 1024 * sizeof(Data),
                                   cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 9, 18, 10,
                        cfg.n_power, false, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
#endif
                    break;

                default:
                    break;
            }
            break;
        case INVERSE:
            switch(cfg.n_power)
            {
                case 12:
                    InverseCore<<<dim3(1, 8, batch_size), dim3(256, 1), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 11, 3, 9,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCore<<<dim3(8, 1, batch_size), dim3(64, 4), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 2, 0, 3,
                        cfg.n_power, cfg.mod_inverse, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 13:
                    InverseCore<<<dim3(1, 16, batch_size), dim3(256, 1), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 12, 4, 9,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCore<<<dim3(16, 1, batch_size), dim3(32, 8), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 3, 0, 4,
                        cfg.n_power, cfg.mod_inverse, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 14:
                    InverseCore<<<dim3(1, 32, batch_size), dim3(256, 1), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 13, 5, 9,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCore<<<dim3(32, 1, batch_size), dim3(16, 16), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 4, 0, 5,
                        cfg.n_power, cfg.mod_inverse, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 15:
                    InverseCore<<<dim3(1, 64, batch_size), dim3(256, 1), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 14, 6, 9,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCore<<<dim3(64, 1, batch_size), dim3(8, 32), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 5, 0, 6,
                        cfg.n_power, cfg.mod_inverse, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 16:
                    InverseCore<<<dim3(1, 128, batch_size), dim3(256, 1), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 15, 7, 9,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCore<<<dim3(128, 1, batch_size), dim3(4, 64), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 6, 0, 7,
                        cfg.n_power, cfg.mod_inverse, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 17:
                    InverseCore<<<dim3(1, 256, batch_size), dim3(256, 1), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 16, 8, 9,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCore<<<dim3(16, 16, batch_size), dim3(32, 8), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 7, 4, 4,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCore<<<dim3(256, 1, batch_size), dim3(32, 8), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 3, 0, 4,
                        cfg.n_power, cfg.mod_inverse, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 18:
                    InverseCore<<<dim3(1, 512, batch_size), dim3(256, 1), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 17, 9, 9,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCore<<<dim3(32, 16, batch_size), dim3(16, 16), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 8, 4, 5,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCore<<<dim3(512, 1, batch_size), dim3(32, 8), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 3, 0, 4,
                        cfg.n_power, cfg.mod_inverse, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 19:
                    InverseCore<<<dim3(1, 1024, batch_size), dim3(256, 1), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 18, 10, 9,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCore<<<dim3(32, 32, batch_size), dim3(16, 16), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 9, 5, 5,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCore<<<dim3(1024, 1, batch_size), dim3(16, 16), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 4, 0, 5,
                        cfg.n_power, cfg.mod_inverse, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 20:
                    InverseCore<<<dim3(1, 2048, batch_size), dim3(256, 1), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 19, 11, 9,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCore<<<dim3(64, 32, batch_size), dim3(8, 32), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 10, 5, 6,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCore<<<dim3(2048, 1, batch_size), dim3(16, 16), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 4, 0, 5,
                        cfg.n_power, cfg.mod_inverse, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 21:  //
                    InverseCore<<<dim3(1, 4096, batch_size), dim3(256, 1), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 20, 12, 9,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCore<<<dim3(64, 64, batch_size), dim3(8, 32), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 11, 6, 6,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCore<<<dim3(4096, 1, batch_size), dim3(8, 32), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 5, 0, 6,
                        cfg.n_power, cfg.mod_inverse, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 22:
                    InverseCore<<<dim3(1, 8192, batch_size), dim3(256, 1), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 21, 13, 9,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCore<<<dim3(128, 64, batch_size), dim3(4, 64), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 12, 6, 7,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCore<<<dim3(8192, 1, batch_size), dim3(8, 32), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 5, 0, 6,
                        cfg.n_power, cfg.mod_inverse, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 23:
                    InverseCore<<<dim3(1, 16384, batch_size), dim3(256, 1), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 22, 14, 9,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCore<<<dim3(128, 128, batch_size), dim3(4, 64), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 13, 7, 7,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCore<<<dim3(16384, 1, batch_size), dim3(4, 64), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 6, 0, 7,
                        cfg.n_power, cfg.mod_inverse, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 24:
                    InverseCore<<<dim3(1, 16384, batch_size), dim3(512, 1), 1024 * sizeof(Data),
                                  cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 9, 23, 14, 10,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCore<<<dim3(128, 128, batch_size), dim3(8, 64), 1024 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 9, 13, 7, 7,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCore<<<dim3(16384, 1, batch_size), dim3(8, 64), 1024 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 9, 6, 0, 7,
                        cfg.n_power, cfg.mod_inverse, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 25:
                    InverseCore_<<<dim3(32768, 1, batch_size), dim3(512, 1), 1024 * sizeof(Data),
                                   cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 9, 24, 15, 10,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ///////////////////////////////////////////////////////////
                    InverseCore<<<dim3(256, 128, batch_size), dim3(4, 128), 1024 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 9, 14, 7, 8,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCore<<<dim3(32768, 1, batch_size), dim3(8, 64), 1024 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 9, 6, 0, 7,
                        cfg.n_power, cfg.mod_inverse, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 26:
                    InverseCore_<<<dim3(65536, 1, batch_size), dim3(512, 1), 1024 * sizeof(Data),
                                   cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 9, 25, 16, 10,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ///////////////////////////////////////////////////////////
                    InverseCore<<<dim3(256, 256, batch_size), dim3(4, 128), 1024 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 9, 15, 8, 8,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCore<<<dim3(65536, 1, batch_size), dim3(4, 128), 1024 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 9, 7, 0, 8,
                        cfg.n_power, cfg.mod_inverse, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 27:
#ifndef CC_89
                    InverseCore_<<<dim3(262144, 1, batch_size), dim3(256, 1), 512 * sizeof(Data),
                                   cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 26, 18, 9,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ///////////////////////////////////////////////////////////
                    InverseCore<<<dim3(128, 2048, batch_size), dim3(4, 64), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 17, 11, 7,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCore<<<dim3(8192, 32, batch_size), dim3(8, 32), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 10, 5, 6,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCore<<<dim3(262144, 1, batch_size), dim3(16, 16), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 4, 0, 5,
                        cfg.n_power, cfg.mod_inverse, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
#else
                    InverseCore_<<<dim3(131072, 1, batch_size), dim3(512, 1), 1024 * sizeof(Data),
                                   cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 9, 26, 17, 10,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ///////////////////////////////////////////////////////////
                    InverseCore<<<dim3(512, 256, batch_size), dim3(2, 256), 1024 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 9, 16, 8, 9,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCore<<<dim3(131072, 1, batch_size), dim3(4, 128), 1024 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 9, 7, 0, 8,
                        cfg.n_power, cfg.mod_inverse, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
#endif
                    break;
                case 28:
#ifndef CC_89
                    InverseCore_<<<dim3(524288, 1, batch_size), dim3(256, 1), 512 * sizeof(Data),
                                   cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 27, 19, 9,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ///////////////////////////////////////////////////////////
                    InverseCore<<<dim3(128, 4096, batch_size), dim3(4, 64), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 18, 12, 7,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCore<<<dim3(8192, 64, batch_size), dim3(8, 32), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 11, 6, 6,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCore<<<dim3(524288, 1, batch_size), dim3(8, 32), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 5, 0, 6,
                        cfg.n_power, cfg.mod_inverse, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
#else
                    InverseCore_<<<dim3(262144, 1, batch_size), dim3(512, 1), 1024 * sizeof(Data),
                                   cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 9, 27, 18, 10,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ///////////////////////////////////////////////////////////
                    InverseCore<<<dim3(512, 512, batch_size), dim3(2, 256), 1024 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 9, 17, 9, 9,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCore<<<dim3(262144, 1, batch_size), dim3(2, 256), 1024 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 9, 8, 0, 9,
                        cfg.n_power, cfg.mod_inverse, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
#endif
                    break;

                default:
                    break;
            }
            break;

        default:
            break;
    }
}

__host__ void GPU_NTT(Data* device_in, Data* device_out, Root* root_of_unity_table,
                      Modulus* modulus, ntt_rns_configuration cfg, int batch_size, int mod_count)
{
    switch(cfg.ntt_type)
    {
        case FORWARD:
            switch(cfg.n_power)
            {
                case 12:
                    ForwardCore<<<dim3(8, 1, batch_size), dim3(64, 4), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 0, 3, cfg.n_power,
                        cfg.zero_padding, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCore<<<dim3(1, 8, batch_size), dim3(256, 1), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 3, 9, cfg.n_power,
                        false, false, (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                        mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 13:
                    ForwardCore<<<dim3(16, 1, batch_size), dim3(32, 8), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 0, 4, cfg.n_power,
                        cfg.zero_padding, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCore<<<dim3(1, 16, batch_size), dim3(256, 1), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 4, 9, cfg.n_power,
                        false, false, (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                        mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 14:
                    ForwardCore<<<dim3(32, 1, batch_size), dim3(16, 16), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 0, 5, cfg.n_power,
                        cfg.zero_padding, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCore<<<dim3(1, 32, batch_size), dim3(256, 1), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 5, 9, cfg.n_power,
                        false, false, (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                        mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 15:
                    ForwardCore<<<dim3(64, 1, batch_size), dim3(8, 32), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 0, 6, cfg.n_power,
                        cfg.zero_padding, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCore<<<dim3(1, 64, batch_size), dim3(256, 1), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 6, 9, cfg.n_power,
                        false, false, (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                        mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 16:
                    ForwardCore<<<dim3(128, 1, batch_size), dim3(4, 64), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 0, 7, cfg.n_power,
                        cfg.zero_padding, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCore<<<dim3(1, 128, batch_size), dim3(256, 1), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 7, 9, cfg.n_power,
                        false, false, (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                        mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 17:
                    ForwardCore<<<dim3(256, 1, batch_size), dim3(32, 8), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 0, 4, cfg.n_power,
                        cfg.zero_padding, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCore<<<dim3(16, 16, batch_size), dim3(32, 8), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 4, 4, cfg.n_power,
                        false, true, (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                        mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCore<<<dim3(1, 256, batch_size), dim3(256, 1), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 8, 9, cfg.n_power,
                        false, false, (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                        mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 18:
                    ForwardCore<<<dim3(512, 1, batch_size), dim3(32, 8), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 0, 4, cfg.n_power,
                        cfg.zero_padding, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCore<<<dim3(32, 16, batch_size), dim3(16, 16), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 4, 5, cfg.n_power,
                        false, true, (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                        mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCore<<<dim3(1, 512, batch_size), dim3(256, 1), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 9, 9, cfg.n_power,
                        false, false, (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                        mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 19:
                    ForwardCore<<<dim3(1024, 1, batch_size), dim3(16, 16), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 0, 5, cfg.n_power,
                        cfg.zero_padding, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCore<<<dim3(32, 32, batch_size), dim3(16, 16), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 5, 5, cfg.n_power,
                        false, true, (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                        mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCore<<<dim3(1, 1024, batch_size), dim3(256, 1), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 10, 9, cfg.n_power,
                        false, false, (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                        mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 20:
                    ForwardCore<<<dim3(2048, 1, batch_size), dim3(16, 16), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 0, 5, cfg.n_power,
                        cfg.zero_padding, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCore<<<dim3(64, 32, batch_size), dim3(8, 32), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 5, 6, cfg.n_power,
                        false, true, (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                        mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCore<<<dim3(1, 2048, batch_size), dim3(256, 1), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 11, 9, cfg.n_power,
                        false, false, (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                        mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 21:
                    ForwardCore<<<dim3(4096, 1, batch_size), dim3(8, 32), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 0, 6, cfg.n_power,
                        cfg.zero_padding, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCore<<<dim3(64, 64, batch_size), dim3(8, 32), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 6, 6, cfg.n_power,
                        false, true, (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                        mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCore<<<dim3(1, 4096, batch_size), dim3(256, 1), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 12, 9, cfg.n_power,
                        false, false, (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                        mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 22:
                    ForwardCore<<<dim3(8192, 1, batch_size), dim3(8, 32), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 0, 6, cfg.n_power,
                        cfg.zero_padding, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCore<<<dim3(128, 64, batch_size), dim3(4, 64), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 6, 7, cfg.n_power,
                        false, true, (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                        mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCore<<<dim3(1, 8192, batch_size), dim3(256, 1), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 13, 9, cfg.n_power,
                        false, false, (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                        mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());

                    break;
                case 23:
                    ForwardCore<<<dim3(16384, 1, batch_size), dim3(4, 64), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 0, 7, cfg.n_power,
                        cfg.zero_padding, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCore<<<dim3(128, 128, batch_size), dim3(4, 64), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 7, 7, cfg.n_power,
                        false, true, (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                        mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCore<<<dim3(1, 16384, batch_size), dim3(256, 1), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 14, 9, cfg.n_power,
                        false, false, (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                        mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 24:
                    ForwardCore<<<dim3(16384, 1, batch_size), dim3(8, 64), 1024 * sizeof(Data),
                                  cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 9, 0, 7, cfg.n_power,
                        cfg.zero_padding, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCore<<<dim3(128, 128, batch_size), dim3(8, 64), 1024 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 9, 7, 7, cfg.n_power,
                        false, true, (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                        mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCore<<<dim3(1, 16384, batch_size), dim3(512, 1), 1024 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 9, 14, 10,
                        cfg.n_power, false, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 25:
                    ForwardCore<<<dim3(32768, 1, batch_size), dim3(8, 64), 1024 * sizeof(Data),
                                  cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 9, 0, 7, cfg.n_power,
                        cfg.zero_padding, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCore<<<dim3(256, 128, batch_size), dim3(4, 128), 1024 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 9, 7, 8, cfg.n_power,
                        false, true, (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                        mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ///////////////////////////////////////////////////////////
                    ForwardCore_<<<dim3(32768, 1, batch_size), dim3(512, 1), 1024 * sizeof(Data),
                                   cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 9, 15, 10,
                        cfg.n_power, false, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 26:
                    ForwardCore<<<dim3(65536, 1, batch_size), dim3(4, 128), 1024 * sizeof(Data),
                                  cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 9, 0, 8, cfg.n_power,
                        cfg.zero_padding, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCore<<<dim3(256, 256, batch_size), dim3(4, 128), 1024 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 9, 8, 8, cfg.n_power,
                        false, true, (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                        mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ///////////////////////////////////////////////////////////
                    ForwardCore_<<<dim3(65536, 1, batch_size), dim3(512, 1), 1024 * sizeof(Data),
                                   cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 9, 16, 10,
                        cfg.n_power, false, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 27:
#ifndef CC_89
                    ForwardCore<<<dim3(262144, 1, batch_size), dim3(16, 16), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 0, 5, cfg.n_power,
                        cfg.zero_padding, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCore<<<dim3(8192, 32, batch_size), dim3(8, 32), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 5, 6, cfg.n_power,
                        false, true, (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                        mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCore<<<dim3(128, 2048, batch_size), dim3(4, 64), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 11, 7, cfg.n_power,
                        false, true, (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                        mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ///////////////////////////////////////////////////////////
                    ForwardCore_<<<dim3(262144, 1, batch_size), dim3(256, 1), 512 * sizeof(Data),
                                   cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 18, 9, cfg.n_power,
                        false, false, (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                        mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
#else
                    ForwardCore<<<dim3(131072, 1, batch_size), dim3(4, 128), 1024 * sizeof(Data),
                                  cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 9, 0, 8, cfg.n_power,
                        cfg.zero_padding, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCore<<<dim3(512, 256, batch_size), dim3(2, 256), 1024 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 9, 8, 9, cfg.n_power,
                        false, true, (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                        mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ///////////////////////////////////////////////////////////
                    ForwardCore_<<<dim3(131072, 1, batch_size), dim3(512, 1), 1024 * sizeof(Data),
                                   cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 9, 17, 10,
                        cfg.n_power, false, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
#endif
                    break;
                case 28:
#ifndef CC_89
                    ForwardCore<<<dim3(524288, 1, batch_size), dim3(8, 32), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 0, 6, cfg.n_power,
                        cfg.zero_padding, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCore<<<dim3(8192, 64, batch_size), dim3(8, 32), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 6, 6, cfg.n_power,
                        false, true, (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                        mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCore<<<dim3(128, 4096, batch_size), dim3(4, 64), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 12, 7, cfg.n_power,
                        false, true, (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                        mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ///////////////////////////////////////////////////////////
                    ForwardCore_<<<dim3(524288, 1, batch_size), dim3(256, 1), 512 * sizeof(Data),
                                   cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 19, 9, cfg.n_power,
                        false, false, (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                        mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
#else
                    ForwardCore<<<dim3(262144, 1, batch_size), dim3(2, 256), 1024 * sizeof(Data),
                                  cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 9, 0, 9, cfg.n_power,
                        cfg.zero_padding, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCore<<<dim3(512, 512, batch_size), dim3(2, 256), 1024 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 9, 9, 9, cfg.n_power,
                        false, true, (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                        mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ///////////////////////////////////////////////////////////
                    ForwardCore_<<<dim3(262144, 1, batch_size), dim3(512, 1), 1024 * sizeof(Data),
                                   cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 9, 18, 10,
                        cfg.n_power, false, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
#endif
                    break;

                default:
                    break;
            }
            break;
        case INVERSE:
            switch(cfg.n_power)
            {
                case 12:
                    InverseCore<<<dim3(1, 8, batch_size), dim3(256, 1), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 11, 3, 9,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCore<<<dim3(8, 1, batch_size), dim3(64, 4), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 2, 0, 3,
                        cfg.n_power, cfg.mod_inverse, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 13:
                    InverseCore<<<dim3(1, 16, batch_size), dim3(256, 1), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 12, 4, 9,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCore<<<dim3(16, 1, batch_size), dim3(32, 8), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 3, 0, 4,
                        cfg.n_power, cfg.mod_inverse, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 14:
                    InverseCore<<<dim3(1, 32, batch_size), dim3(256, 1), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 13, 5, 9,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCore<<<dim3(32, 1, batch_size), dim3(16, 16), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 4, 0, 5,
                        cfg.n_power, cfg.mod_inverse, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 15:
                    InverseCore<<<dim3(1, 64, batch_size), dim3(256, 1), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 14, 6, 9,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCore<<<dim3(64, 1, batch_size), dim3(8, 32), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 5, 0, 6,
                        cfg.n_power, cfg.mod_inverse, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 16:
                    InverseCore<<<dim3(1, 128, batch_size), dim3(256, 1), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 15, 7, 9,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCore<<<dim3(128, 1, batch_size), dim3(4, 64), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 6, 0, 7,
                        cfg.n_power, cfg.mod_inverse, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 17:
                    InverseCore<<<dim3(1, 256, batch_size), dim3(256, 1), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 16, 8, 9,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCore<<<dim3(16, 16, batch_size), dim3(32, 8), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 7, 4, 4,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCore<<<dim3(256, 1, batch_size), dim3(32, 8), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 3, 0, 4,
                        cfg.n_power, cfg.mod_inverse, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 18:
                    InverseCore<<<dim3(1, 512, batch_size), dim3(256, 1), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 17, 9, 9,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCore<<<dim3(32, 16, batch_size), dim3(16, 16), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 8, 4, 5,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCore<<<dim3(512, 1, batch_size), dim3(32, 8), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 3, 0, 4,
                        cfg.n_power, cfg.mod_inverse, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 19:
                    InverseCore<<<dim3(1, 1024, batch_size), dim3(256, 1), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 18, 10, 9,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCore<<<dim3(32, 32, batch_size), dim3(16, 16), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 9, 5, 5,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCore<<<dim3(1024, 1, batch_size), dim3(16, 16), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 4, 0, 5,
                        cfg.n_power, cfg.mod_inverse, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 20:
                    InverseCore<<<dim3(1, 2048, batch_size), dim3(256, 1), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 19, 11, 9,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCore<<<dim3(64, 32, batch_size), dim3(8, 32), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 10, 5, 6,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCore<<<dim3(2048, 1, batch_size), dim3(16, 16), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 4, 0, 5,
                        cfg.n_power, cfg.mod_inverse, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 21:  //
                    InverseCore<<<dim3(1, 4096, batch_size), dim3(256, 1), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 20, 12, 9,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCore<<<dim3(64, 64, batch_size), dim3(8, 32), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 11, 6, 6,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCore<<<dim3(4096, 1, batch_size), dim3(8, 32), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 5, 0, 6,
                        cfg.n_power, cfg.mod_inverse, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 22:
                    InverseCore<<<dim3(1, 8192, batch_size), dim3(256, 1), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 21, 13, 9,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCore<<<dim3(128, 64, batch_size), dim3(4, 64), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 12, 6, 7,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCore<<<dim3(8192, 1, batch_size), dim3(8, 32), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 5, 0, 6,
                        cfg.n_power, cfg.mod_inverse, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 23:
                    InverseCore<<<dim3(1, 16384, batch_size), dim3(256, 1), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 22, 14, 9,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCore<<<dim3(128, 128, batch_size), dim3(4, 64), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 13, 7, 7,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCore<<<dim3(16384, 1, batch_size), dim3(4, 64), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 6, 0, 7,
                        cfg.n_power, cfg.mod_inverse, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 24:
                    InverseCore<<<dim3(1, 16384, batch_size), dim3(512, 1), 1024 * sizeof(Data),
                                  cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 9, 23, 14, 10,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCore<<<dim3(128, 128, batch_size), dim3(8, 64), 1024 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 9, 13, 7, 7,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCore<<<dim3(16384, 1, batch_size), dim3(8, 64), 1024 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 9, 6, 0, 7,
                        cfg.n_power, cfg.mod_inverse, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 25:
                    InverseCore_<<<dim3(32768, 1, batch_size), dim3(512, 1), 1024 * sizeof(Data),
                                   cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 9, 24, 15, 10,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ///////////////////////////////////////////////////////////
                    InverseCore<<<dim3(256, 128, batch_size), dim3(4, 128), 1024 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 9, 14, 7, 8,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCore<<<dim3(32768, 1, batch_size), dim3(8, 64), 1024 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 9, 6, 0, 7,
                        cfg.n_power, cfg.mod_inverse, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 26:
                    InverseCore_<<<dim3(65536, 1, batch_size), dim3(512, 1), 1024 * sizeof(Data),
                                   cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 9, 25, 16, 10,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ///////////////////////////////////////////////////////////
                    InverseCore<<<dim3(256, 256, batch_size), dim3(4, 128), 1024 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 9, 15, 8, 8,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCore<<<dim3(65536, 1, batch_size), dim3(4, 128), 1024 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 9, 7, 0, 8,
                        cfg.n_power, cfg.mod_inverse, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 27:
#ifndef CC_89
                    InverseCore_<<<dim3(262144, 1, batch_size), dim3(256, 1), 512 * sizeof(Data),
                                   cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 26, 18, 9,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ///////////////////////////////////////////////////////////
                    InverseCore<<<dim3(128, 2048, batch_size), dim3(4, 64), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 17, 11, 7,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCore<<<dim3(8192, 32, batch_size), dim3(8, 32), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 10, 5, 6,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCore<<<dim3(262144, 1, batch_size), dim3(16, 16), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 4, 0, 5,
                        cfg.n_power, cfg.mod_inverse, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count);
#else
                    InverseCore_<<<dim3(131072, 1, batch_size), dim3(512, 1), 1024 * sizeof(Data),
                                   cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 9, 26, 17, 10,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ///////////////////////////////////////////////////////////
                    InverseCore<<<dim3(512, 256, batch_size), dim3(2, 256), 1024 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 9, 16, 8, 9,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCore<<<dim3(131072, 1, batch_size), dim3(4, 128), 1024 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 9, 7, 0, 8,
                        cfg.n_power, cfg.mod_inverse, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count);
#endif
                    break;
                case 28:
#ifndef CC_89
                    InverseCore_<<<dim3(524288, 1, batch_size), dim3(256, 1), 512 * sizeof(Data),
                                   cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 27, 19, 9,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ///////////////////////////////////////////////////////////
                    InverseCore<<<dim3(128, 4096, batch_size), dim3(4, 64), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 18, 12, 7,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCore<<<dim3(8192, 64, batch_size), dim3(8, 32), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 11, 6, 6,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCore<<<dim3(524288, 1, batch_size), dim3(8, 32), 512 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 5, 0, 6,
                        cfg.n_power, cfg.mod_inverse, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count);
#else
                    InverseCore_<<<dim3(262144, 1, batch_size), dim3(512, 1), 1024 * sizeof(Data),
                                   cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 9, 27, 18, 10,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ///////////////////////////////////////////////////////////
                    InverseCore<<<dim3(512, 512, batch_size), dim3(2, 256), 1024 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 9, 17, 9, 9,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCore<<<dim3(262144, 1, batch_size), dim3(2, 256), 1024 * sizeof(Data),
                                  cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 9, 8, 0, 9,
                        cfg.n_power, cfg.mod_inverse, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
#endif
                    break;

                default:
                    break;
            }
            break;

        default:
            break;
    }
}

__host__ void GPU_NTT_Inplace(Data* device_inout, Root* root_of_unity_table, Modulus modulus,
                              ntt_configuration cfg, int batch_size)
{
    GPU_NTT(device_inout, device_inout, root_of_unity_table, modulus, cfg, batch_size);
}

__host__ void GPU_NTT_Inplace(Data* device_inout, Root* root_of_unity_table, Modulus* modulus,
                              ntt_rns_configuration cfg, int batch_size, int mod_count)
{
    GPU_NTT(device_inout, device_inout, root_of_unity_table, modulus, cfg, batch_size, mod_count);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Modulus Ordered

__global__ void ForwardCoreModulusOrdered(Data* polynomial_in, Data* polynomial_out,
                                          Root* root_of_unity_table, Modulus* modulus,
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

    extern __shared__ Data shared_memory[];

    int t_2 = N_power - logm - 1;
    location_t offset = 1 << (N_power - logm - 1);
    int t_ = shared_index;
    location_t m = (location_t)1 << logm;

    location_t global_addresss =
        idx_x + (location_t)(idx_y * (offset / (1 << (outer_iteration_count - 1)))) +
        (location_t)(blockDim.x * block_x) + (location_t)(2 * block_y * offset) +
        (location_t)(block_z << N_power);

    location_t omega_addresss =
        idx_x + (location_t)(idx_y * (offset / (1 << (outer_iteration_count - 1)))) +
        (location_t)(blockDim.x * block_x) + (location_t)(block_y * offset);

    location_t shared_addresss = (idx_x + (idx_y * blockDim.x));

    // Load data from global & store to shared
    shared_memory[shared_addresss] = polynomial_in[global_addresss];
    shared_memory[shared_addresss + (blockDim.x * blockDim.y)] =
        polynomial_in[global_addresss + offset];

    int t = 1 << t_;
    int in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
    location_t current_root_index;
    if(not_last_kernel)
    {
#pragma unroll
        for(int lp = 0; lp < outer_iteration_count; lp++)
        {
            __syncthreads();
            if(reduction_poly_check)
            {  // X_N_minus
                current_root_index = (omega_addresss >> t_2) + (location_t)(prime_index << N_power);
            }
            else
            {  // X_N_plus
                current_root_index =
                    m + (omega_addresss >> t_2) + (location_t)(prime_index << N_power);
            }
            CooleyTukeyUnit(shared_memory[in_shared_address], shared_memory[in_shared_address + t],
                            root_of_unity_table[current_root_index], modulus[prime_index]);

            t = t >> 1;
            t_2 -= 1;
            t_ -= 1;
            m <<= 1;

            in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
            //__syncthreads();
        }
        __syncthreads();
    }
    else
    {
#pragma unroll
        for(int lp = 0; lp < (shared_index - 5); lp++)  // 4 for 512 thread
        {
            __syncthreads();
            if(reduction_poly_check)
            {  // X_N_minus
                current_root_index = (omega_addresss >> t_2) + (location_t)(prime_index << N_power);
            }
            else
            {  // X_N_plus
                current_root_index =
                    m + (omega_addresss >> t_2) + (location_t)(prime_index << N_power);
            }

            CooleyTukeyUnit(shared_memory[in_shared_address], shared_memory[in_shared_address + t],
                            root_of_unity_table[current_root_index], modulus[prime_index]);

            t = t >> 1;
            t_2 -= 1;
            t_ -= 1;
            m <<= 1;

            in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
            //__syncthreads();
        }
        __syncthreads();

#pragma unroll
        for(int lp = 0; lp < 6; lp++)
        {
            if(reduction_poly_check)
            {  // X_N_minus
                current_root_index = (omega_addresss >> t_2) + (location_t)(prime_index << N_power);
            }
            else
            {  // X_N_plus
                current_root_index =
                    m + (omega_addresss >> t_2) + (location_t)(prime_index << N_power);
            }
            CooleyTukeyUnit(shared_memory[in_shared_address], shared_memory[in_shared_address + t],
                            root_of_unity_table[current_root_index], modulus[prime_index]);

            t = t >> 1;
            t_2 -= 1;
            t_ -= 1;
            m <<= 1;

            in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
        }
        __syncthreads();
    }

    polynomial_out[global_addresss] = shared_memory[shared_addresss];
    polynomial_out[global_addresss + offset] =
        shared_memory[shared_addresss + (blockDim.x * blockDim.y)];
}

__global__ void ForwardCoreModulusOrdered_(Data* polynomial_in, Data* polynomial_out,
                                           Root* root_of_unity_table, Modulus* modulus,
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

    extern __shared__ Data shared_memory[];

    int t_2 = N_power - logm - 1;
    location_t offset = 1 << (N_power - logm - 1);
    int t_ = shared_index;
    location_t m = (location_t)1 << logm;

    location_t global_addresss =
        idx_x + (location_t)(idx_y * (offset / (1 << (outer_iteration_count - 1)))) +
        (location_t)(blockDim.x * block_y) + (location_t)(2 * block_x * offset) +
        (location_t)(block_z << N_power);
    location_t omega_addresss =
        idx_x + (location_t)(idx_y * (offset / (1 << (outer_iteration_count - 1)))) +
        (location_t)(blockDim.x * block_y) + (location_t)(block_x * offset);
    location_t shared_addresss = (idx_x + (idx_y * blockDim.x));

    // Load data from global & store to shared
    shared_memory[shared_addresss] = polynomial_in[global_addresss];
    shared_memory[shared_addresss + (blockDim.x * blockDim.y)] =
        polynomial_in[global_addresss + offset];

    int t = 1 << t_;
    int in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
    location_t current_root_index;
    if(not_last_kernel)
    {
#pragma unroll
        for(int lp = 0; lp < outer_iteration_count; lp++)
        {
            __syncthreads();
            if(reduction_poly_check)
            {  // X_N_minus
                current_root_index = (omega_addresss >> t_2) + (location_t)(prime_index << N_power);
            }
            else
            {  // X_N_plus
                current_root_index =
                    m + (omega_addresss >> t_2) + (location_t)(prime_index << N_power);
            }
            CooleyTukeyUnit(shared_memory[in_shared_address], shared_memory[in_shared_address + t],
                            root_of_unity_table[current_root_index], modulus[prime_index]);

            t = t >> 1;
            t_2 -= 1;
            t_ -= 1;
            m <<= 1;

            in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
            //__syncthreads();
        }
        __syncthreads();
    }
    else
    {
#pragma unroll
        for(int lp = 0; lp < (shared_index - 5); lp++)
        {
            __syncthreads();
            if(reduction_poly_check)
            {  // X_N_minus
                current_root_index = (omega_addresss >> t_2) + (location_t)(prime_index << N_power);
            }
            else
            {  // X_N_plus
                current_root_index =
                    m + (omega_addresss >> t_2) + (location_t)(prime_index << N_power);
            }
            CooleyTukeyUnit(shared_memory[in_shared_address], shared_memory[in_shared_address + t],
                            root_of_unity_table[current_root_index], modulus[prime_index]);

            t = t >> 1;
            t_2 -= 1;
            t_ -= 1;
            m <<= 1;

            in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
            //__syncthreads();
        }
        __syncthreads();

#pragma unroll
        for(int lp = 0; lp < 6; lp++)
        {
            if(reduction_poly_check)
            {  // X_N_minus
                current_root_index = (omega_addresss >> t_2) + (location_t)(prime_index << N_power);
            }
            else
            {  // X_N_plus
                current_root_index =
                    m + (omega_addresss >> t_2) + (location_t)(prime_index << N_power);
            }
            CooleyTukeyUnit(shared_memory[in_shared_address], shared_memory[in_shared_address + t],
                            root_of_unity_table[current_root_index], modulus[prime_index]);

            t = t >> 1;
            t_2 -= 1;
            t_ -= 1;
            m <<= 1;

            in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
        }
        __syncthreads();
    }

    polynomial_out[global_addresss] = shared_memory[shared_addresss];
    polynomial_out[global_addresss + offset] =
        shared_memory[shared_addresss + (blockDim.x * blockDim.y)];
}

__global__ void InverseCoreModulusOrdered(Data* polynomial_in, Data* polynomial_out,
                                          Root* inverse_root_of_unity_table, Modulus* modulus,
                                          int shared_index, int logm, int k,
                                          int outer_iteration_count, int N_power,
                                          Ninverse* n_inverse, bool last_kernel,
                                          bool reduction_poly_check, int mod_count, int* order)
{
    const int idx_x = threadIdx.x;
    const int idx_y = threadIdx.y;
    const int block_x = blockIdx.x;
    const int block_y = blockIdx.y;
    const int block_z = blockIdx.z;

    const int mod_index = block_z % mod_count;
    const int prime_index = order[mod_index];

    extern __shared__ Data shared_memory[];

    int t_2 = N_power - logm - 1;
    location_t offset = 1 << (N_power - k - 1);
    // int t_ = 9 - outer_iteration_count;
    int t_ = (shared_index + 1) - outer_iteration_count;
    int loops = outer_iteration_count;
    location_t m = (location_t)1 << logm;

    location_t global_addresss =
        idx_x + (location_t)(idx_y * (offset / (1 << (outer_iteration_count - 1)))) +
        (location_t)(blockDim.x * block_x) + (location_t)(2 * block_y * offset) +
        (location_t)(block_z << N_power);

    location_t omega_addresss =
        idx_x + (location_t)(idx_y * (offset / (1 << (outer_iteration_count - 1)))) +
        (location_t)(blockDim.x * block_x) + (location_t)(block_y * offset);
    location_t shared_addresss = (idx_x + (idx_y * blockDim.x));

    shared_memory[shared_addresss] = polynomial_in[global_addresss];
    shared_memory[shared_addresss + (blockDim.x * blockDim.y)] =
        polynomial_in[global_addresss + offset];

    int t = 1 << t_;
    int in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
    location_t current_root_index;
#pragma unroll
    for(int lp = 0; lp < loops; lp++)
    {
        __syncthreads();
        if(reduction_poly_check)
        {  // X_N_minus
            current_root_index = (omega_addresss >> t_2) + (location_t)(prime_index << N_power);
        }
        else
        {  // X_N_plus
            current_root_index = m + (omega_addresss >> t_2) + (location_t)(prime_index << N_power);
        }

        GentlemanSandeUnit(shared_memory[in_shared_address], shared_memory[in_shared_address + t],
                           inverse_root_of_unity_table[current_root_index], modulus[prime_index]);

        t = t << 1;
        t_2 += 1;
        t_ += 1;
        m >>= 1;

        in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
    }
    __syncthreads();

    if(last_kernel)
    {
        polynomial_out[global_addresss] = VALUE_GPU::mult(
            shared_memory[shared_addresss], n_inverse[prime_index], modulus[prime_index]);
        polynomial_out[global_addresss + offset] =
            VALUE_GPU::mult(shared_memory[shared_addresss + (blockDim.x * blockDim.y)],
                            n_inverse[prime_index], modulus[prime_index]);
    }
    else
    {
        polynomial_out[global_addresss] = shared_memory[shared_addresss];
        polynomial_out[global_addresss + offset] =
            shared_memory[shared_addresss + (blockDim.x * blockDim.y)];
    }
}

__global__ void InverseCoreModulusOrdered_(Data* polynomial_in, Data* polynomial_out,
                                           Root* inverse_root_of_unity_table, Modulus* modulus,
                                           int shared_index, int logm, int k,
                                           int outer_iteration_count, int N_power,
                                           Ninverse* n_inverse, bool last_kernel,
                                           bool reduction_poly_check, int mod_count, int* order)
{
    const int idx_x = threadIdx.x;
    const int idx_y = threadIdx.y;
    const int block_x = blockIdx.x;
    const int block_y = blockIdx.y;
    const int block_z = blockIdx.z;

    const int mod_index = block_z % mod_count;
    const int prime_index = order[mod_index];

    extern __shared__ Data shared_memory[];

    int t_2 = N_power - logm - 1;
    location_t offset = 1 << (N_power - k - 1);
    // int t_ = 9 - outer_iteration_count;
    int t_ = (shared_index + 1) - outer_iteration_count;
    int loops = outer_iteration_count;
    location_t m = (location_t)1 << logm;

    location_t global_addresss =
        idx_x + (location_t)(idx_y * (offset / (1 << (outer_iteration_count - 1)))) +
        (location_t)(blockDim.x * block_y) + (location_t)(2 * block_x * offset) +
        (location_t)(block_z << N_power);

    location_t omega_addresss =
        idx_x + (location_t)(idx_y * (offset / (1 << (outer_iteration_count - 1)))) +
        (location_t)(blockDim.x * block_y) + (location_t)(block_x * offset);
    location_t shared_addresss = (idx_x + (idx_y * blockDim.x));

    shared_memory[shared_addresss] = polynomial_in[global_addresss];
    shared_memory[shared_addresss + (blockDim.x * blockDim.y)] =
        polynomial_in[global_addresss + offset];

    int t = 1 << t_;
    int in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
    location_t current_root_index;
#pragma unroll
    for(int lp = 0; lp < loops; lp++)
    {
        __syncthreads();
        if(reduction_poly_check)
        {  // X_N_minus
            current_root_index = (omega_addresss >> t_2) + (location_t)(prime_index << N_power);
        }
        else
        {  // X_N_plus
            current_root_index = m + (omega_addresss >> t_2) + (location_t)(prime_index << N_power);
        }

        GentlemanSandeUnit(shared_memory[in_shared_address], shared_memory[in_shared_address + t],
                           inverse_root_of_unity_table[current_root_index], modulus[prime_index]);

        t = t << 1;
        t_2 += 1;
        t_ += 1;
        m >>= 1;

        in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
    }
    __syncthreads();

    if(last_kernel)
    {
        polynomial_out[global_addresss] = VALUE_GPU::mult(
            shared_memory[shared_addresss], n_inverse[prime_index], modulus[prime_index]);
        polynomial_out[global_addresss + offset] =
            VALUE_GPU::mult(shared_memory[shared_addresss + (blockDim.x * blockDim.y)],
                            n_inverse[prime_index], modulus[prime_index]);
    }
    else
    {
        polynomial_out[global_addresss] = shared_memory[shared_addresss];
        polynomial_out[global_addresss + offset] =
            shared_memory[shared_addresss + (blockDim.x * blockDim.y)];
    }
}

__host__ void GPU_NTT_Modulus_Ordered(Data* device_in, Data* device_out, Root* root_of_unity_table,
                                      Modulus* modulus, ntt_rns_configuration cfg, int batch_size,
                                      int mod_count, int* order)
{
    switch(cfg.ntt_type)
    {
        case FORWARD:
            switch(cfg.n_power)
            {
                case 12:
                    ForwardCoreModulusOrdered<<<dim3(8, 1, batch_size), dim3(64, 4),
                                                512 * sizeof(Data), cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 0, 3, cfg.n_power,
                        cfg.zero_padding, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCoreModulusOrdered<<<dim3(1, 8, batch_size), dim3(256, 1),
                                                512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 3, 9, cfg.n_power,
                        false, false, (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                        mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 13:
                    ForwardCoreModulusOrdered<<<dim3(16, 1, batch_size), dim3(32, 8),
                                                512 * sizeof(Data), cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 0, 4, cfg.n_power,
                        cfg.zero_padding, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCoreModulusOrdered<<<dim3(1, 16, batch_size), dim3(256, 1),
                                                512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 4, 9, cfg.n_power,
                        false, false, (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                        mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 14:
                    ForwardCoreModulusOrdered<<<dim3(32, 1, batch_size), dim3(16, 16),
                                                512 * sizeof(Data), cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 0, 5, cfg.n_power,
                        cfg.zero_padding, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCoreModulusOrdered<<<dim3(1, 32, batch_size), dim3(256, 1),
                                                512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 5, 9, cfg.n_power,
                        false, false, (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                        mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 15:
                    ForwardCoreModulusOrdered<<<dim3(64, 1, batch_size), dim3(8, 32),
                                                512 * sizeof(Data), cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 0, 6, cfg.n_power,
                        cfg.zero_padding, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCoreModulusOrdered<<<dim3(1, 64, batch_size), dim3(256, 1),
                                                512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 6, 9, cfg.n_power,
                        false, false, (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                        mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 16:
                    ForwardCoreModulusOrdered<<<dim3(128, 1, batch_size), dim3(4, 64),
                                                512 * sizeof(Data), cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 0, 7, cfg.n_power,
                        cfg.zero_padding, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCoreModulusOrdered<<<dim3(1, 128, batch_size), dim3(256, 1),
                                                512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 7, 9, cfg.n_power,
                        false, false, (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                        mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 17:
                    ForwardCoreModulusOrdered<<<dim3(256, 1, batch_size), dim3(32, 8),
                                                512 * sizeof(Data), cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 0, 4, cfg.n_power,
                        cfg.zero_padding, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCoreModulusOrdered<<<dim3(16, 16, batch_size), dim3(32, 8),
                                                512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 4, 4, cfg.n_power,
                        false, true, (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                        mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCoreModulusOrdered<<<dim3(1, 256, batch_size), dim3(256, 1),
                                                512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 8, 9, cfg.n_power,
                        false, false, (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                        mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 18:
                    ForwardCoreModulusOrdered<<<dim3(512, 1, batch_size), dim3(32, 8),
                                                512 * sizeof(Data), cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 0, 4, cfg.n_power,
                        cfg.zero_padding, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCoreModulusOrdered<<<dim3(32, 16, batch_size), dim3(16, 16),
                                                512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 4, 5, cfg.n_power,
                        false, true, (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                        mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCoreModulusOrdered<<<dim3(1, 512, batch_size), dim3(256, 1),
                                                512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 9, 9, cfg.n_power,
                        false, false, (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                        mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 19:
                    ForwardCoreModulusOrdered<<<dim3(1024, 1, batch_size), dim3(16, 16),
                                                512 * sizeof(Data), cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 0, 5, cfg.n_power,
                        cfg.zero_padding, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCoreModulusOrdered<<<dim3(32, 32, batch_size), dim3(16, 16),
                                                512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 5, 5, cfg.n_power,
                        false, true, (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                        mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCoreModulusOrdered<<<dim3(1, 1024, batch_size), dim3(256, 1),
                                                512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 10, 9, cfg.n_power,
                        false, false, (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                        mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 20:
                    ForwardCoreModulusOrdered<<<dim3(2048, 1, batch_size), dim3(16, 16),
                                                512 * sizeof(Data), cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 0, 5, cfg.n_power,
                        cfg.zero_padding, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCoreModulusOrdered<<<dim3(64, 32, batch_size), dim3(8, 32),
                                                512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 5, 6, cfg.n_power,
                        false, true, (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                        mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCoreModulusOrdered<<<dim3(1, 2048, batch_size), dim3(256, 1),
                                                512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 11, 9, cfg.n_power,
                        false, false, (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                        mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 21:
                    ForwardCoreModulusOrdered<<<dim3(4096, 1, batch_size), dim3(8, 32),
                                                512 * sizeof(Data), cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 0, 6, cfg.n_power,
                        cfg.zero_padding, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCoreModulusOrdered<<<dim3(64, 64, batch_size), dim3(8, 32),
                                                512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 6, 6, cfg.n_power,
                        false, true, (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                        mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCoreModulusOrdered<<<dim3(1, 4096, batch_size), dim3(256, 1),
                                                512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 12, 9, cfg.n_power,
                        false, false, (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                        mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 22:
                    ForwardCoreModulusOrdered<<<dim3(8192, 1, batch_size), dim3(8, 32),
                                                512 * sizeof(Data), cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 0, 6, cfg.n_power,
                        cfg.zero_padding, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCoreModulusOrdered<<<dim3(128, 64, batch_size), dim3(4, 64),
                                                512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 6, 7, cfg.n_power,
                        false, true, (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                        mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCoreModulusOrdered<<<dim3(1, 8192, batch_size), dim3(256, 1),
                                                512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 13, 9, cfg.n_power,
                        false, false, (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                        mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());

                    break;
                case 23:
                    ForwardCoreModulusOrdered<<<dim3(16384, 1, batch_size), dim3(4, 64),
                                                512 * sizeof(Data), cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 0, 7, cfg.n_power,
                        cfg.zero_padding, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCoreModulusOrdered<<<dim3(128, 128, batch_size), dim3(4, 64),
                                                512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 7, 7, cfg.n_power,
                        false, true, (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                        mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCoreModulusOrdered<<<dim3(1, 16384, batch_size), dim3(256, 1),
                                                512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 14, 9, cfg.n_power,
                        false, false, (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                        mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 24:
                    ForwardCoreModulusOrdered<<<dim3(16384, 1, batch_size), dim3(8, 64),
                                                1024 * sizeof(Data), cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 9, 0, 7, cfg.n_power,
                        cfg.zero_padding, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCoreModulusOrdered<<<dim3(128, 128, batch_size), dim3(8, 64),
                                                1024 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 9, 7, 7, cfg.n_power,
                        false, true, (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                        mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCoreModulusOrdered<<<dim3(1, 16384, batch_size), dim3(512, 1),
                                                1024 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 9, 14, 10,
                        cfg.n_power, false, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 25:
                    ForwardCoreModulusOrdered<<<dim3(32768, 1, batch_size), dim3(8, 64),
                                                1024 * sizeof(Data), cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 9, 0, 7, cfg.n_power,
                        cfg.zero_padding, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCoreModulusOrdered<<<dim3(256, 128, batch_size), dim3(4, 128),
                                                1024 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 9, 7, 8, cfg.n_power,
                        false, true, (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                        mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ///////////////////////////////////////////////////////////
                    ForwardCoreModulusOrdered_<<<dim3(32768, 1, batch_size), dim3(512, 1),
                                                 1024 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 9, 15, 10,
                        cfg.n_power, false, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 26:
                    ForwardCoreModulusOrdered<<<dim3(65536, 1, batch_size), dim3(4, 128),
                                                1024 * sizeof(Data), cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 9, 0, 8, cfg.n_power,
                        cfg.zero_padding, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCoreModulusOrdered<<<dim3(256, 256, batch_size), dim3(4, 128),
                                                1024 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 9, 8, 8, cfg.n_power,
                        false, true, (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                        mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ///////////////////////////////////////////////////////////
                    ForwardCoreModulusOrdered_<<<dim3(65536, 1, batch_size), dim3(512, 1),
                                                 1024 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 9, 16, 10,
                        cfg.n_power, false, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 27:
#ifndef CC_89
                    ForwardCoreModulusOrdered<<<dim3(262144, 1, batch_size), dim3(16, 16),
                                                512 * sizeof(Data), cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 0, 5, cfg.n_power,
                        cfg.zero_padding, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCoreModulusOrdered<<<dim3(8192, 32, batch_size), dim3(8, 32),
                                                512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 5, 6, cfg.n_power,
                        false, true, (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                        mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCoreModulusOrdered<<<dim3(128, 2048, batch_size), dim3(4, 64),
                                                512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 11, 7, cfg.n_power,
                        false, true, (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                        mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ///////////////////////////////////////////////////////////
                    ForwardCoreModulusOrdered_<<<dim3(262144, 1, batch_size), dim3(256, 1),
                                                 512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 18, 9, cfg.n_power,
                        false, false, (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                        mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
#else
                    ForwardCoreModulusOrdered<<<dim3(131072, 1, batch_size), dim3(4, 128),
                                                1024 * sizeof(Data), cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 9, 0, 8, cfg.n_power,
                        cfg.zero_padding, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCoreModulusOrdered<<<dim3(512, 256, batch_size), dim3(2, 256),
                                                1024 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 9, 8, 9, cfg.n_power,
                        false, true, (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                        mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ///////////////////////////////////////////////////////////
                    ForwardCoreModulusOrdered_<<<dim3(131072, 1, batch_size), dim3(512, 1),
                                                 1024 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 9, 17, 10,
                        cfg.n_power, false, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
#endif
                    break;
                case 28:
#ifndef CC_89
                    ForwardCoreModulusOrdered<<<dim3(524288, 1, batch_size), dim3(8, 32),
                                                512 * sizeof(Data), cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 0, 6, cfg.n_power,
                        cfg.zero_padding, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCoreModulusOrdered<<<dim3(8192, 64, batch_size), dim3(8, 32),
                                                512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 6, 6, cfg.n_power,
                        false, true, (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                        mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCoreModulusOrdered<<<dim3(128, 4096, batch_size), dim3(4, 64),
                                                512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 12, 7, cfg.n_power,
                        false, true, (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                        mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ///////////////////////////////////////////////////////////
                    ForwardCoreModulusOrdered_<<<dim3(524288, 1, batch_size), dim3(256, 1),
                                                 512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 19, 9, cfg.n_power,
                        false, false, (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                        mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
#else
                    ForwardCoreModulusOrdered<<<dim3(262144, 1, batch_size), dim3(2, 256),
                                                1024 * sizeof(Data), cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 9, 0, 9, cfg.n_power,
                        cfg.zero_padding, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCoreModulusOrdered<<<dim3(512, 512, batch_size), dim3(2, 256),
                                                1024 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 9, 9, 9, cfg.n_power,
                        false, true, (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                        mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ///////////////////////////////////////////////////////////
                    ForwardCoreModulusOrdered_<<<dim3(262144, 1, batch_size), dim3(512, 1),
                                                 1024 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 9, 18, 10,
                        cfg.n_power, false, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
#endif
                    break;

                default:
                    break;
            }
            break;
        case INVERSE:
            switch(cfg.n_power)
            {
                case 12:
                    InverseCoreModulusOrdered<<<dim3(1, 8, batch_size), dim3(256, 1),
                                                512 * sizeof(Data), cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 11, 3, 9,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCoreModulusOrdered<<<dim3(8, 1, batch_size), dim3(64, 4),
                                                512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 2, 0, 3,
                        cfg.n_power, cfg.mod_inverse, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 13:
                    InverseCoreModulusOrdered<<<dim3(1, 16, batch_size), dim3(256, 1),
                                                512 * sizeof(Data), cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 12, 4, 9,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCoreModulusOrdered<<<dim3(16, 1, batch_size), dim3(32, 8),
                                                512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 3, 0, 4,
                        cfg.n_power, cfg.mod_inverse, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 14:
                    InverseCoreModulusOrdered<<<dim3(1, 32, batch_size), dim3(256, 1),
                                                512 * sizeof(Data), cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 13, 5, 9,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCoreModulusOrdered<<<dim3(32, 1, batch_size), dim3(16, 16),
                                                512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 4, 0, 5,
                        cfg.n_power, cfg.mod_inverse, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 15:
                    InverseCoreModulusOrdered<<<dim3(1, 64, batch_size), dim3(256, 1),
                                                512 * sizeof(Data), cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 14, 6, 9,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCoreModulusOrdered<<<dim3(64, 1, batch_size), dim3(8, 32),
                                                512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 5, 0, 6,
                        cfg.n_power, cfg.mod_inverse, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 16:
                    InverseCoreModulusOrdered<<<dim3(1, 128, batch_size), dim3(256, 1),
                                                512 * sizeof(Data), cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 15, 7, 9,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCoreModulusOrdered<<<dim3(128, 1, batch_size), dim3(4, 64),
                                                512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 6, 0, 7,
                        cfg.n_power, cfg.mod_inverse, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 17:
                    InverseCoreModulusOrdered<<<dim3(1, 256, batch_size), dim3(256, 1),
                                                512 * sizeof(Data), cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 16, 8, 9,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCoreModulusOrdered<<<dim3(16, 16, batch_size), dim3(32, 8),
                                                512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 7, 4, 4,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCoreModulusOrdered<<<dim3(256, 1, batch_size), dim3(32, 8),
                                                512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 3, 0, 4,
                        cfg.n_power, cfg.mod_inverse, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 18:
                    InverseCoreModulusOrdered<<<dim3(1, 512, batch_size), dim3(256, 1),
                                                512 * sizeof(Data), cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 17, 9, 9,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCoreModulusOrdered<<<dim3(32, 16, batch_size), dim3(16, 16),
                                                512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 8, 4, 5,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCoreModulusOrdered<<<dim3(512, 1, batch_size), dim3(32, 8),
                                                512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 3, 0, 4,
                        cfg.n_power, cfg.mod_inverse, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 19:
                    InverseCoreModulusOrdered<<<dim3(1, 1024, batch_size), dim3(256, 1),
                                                512 * sizeof(Data), cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 18, 10, 9,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCoreModulusOrdered<<<dim3(32, 32, batch_size), dim3(16, 16),
                                                512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 9, 5, 5,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCoreModulusOrdered<<<dim3(1024, 1, batch_size), dim3(16, 16),
                                                512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 4, 0, 5,
                        cfg.n_power, cfg.mod_inverse, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 20:
                    InverseCoreModulusOrdered<<<dim3(1, 2048, batch_size), dim3(256, 1),
                                                512 * sizeof(Data), cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 19, 11, 9,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCoreModulusOrdered<<<dim3(64, 32, batch_size), dim3(8, 32),
                                                512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 10, 5, 6,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCoreModulusOrdered<<<dim3(2048, 1, batch_size), dim3(16, 16),
                                                512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 4, 0, 5,
                        cfg.n_power, cfg.mod_inverse, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 21:  //
                    InverseCoreModulusOrdered<<<dim3(1, 4096, batch_size), dim3(256, 1),
                                                512 * sizeof(Data), cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 20, 12, 9,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCoreModulusOrdered<<<dim3(64, 64, batch_size), dim3(8, 32),
                                                512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 11, 6, 6,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCoreModulusOrdered<<<dim3(4096, 1, batch_size), dim3(8, 32),
                                                512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 5, 0, 6,
                        cfg.n_power, cfg.mod_inverse, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 22:
                    InverseCoreModulusOrdered<<<dim3(1, 8192, batch_size), dim3(256, 1),
                                                512 * sizeof(Data), cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 21, 13, 9,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCoreModulusOrdered<<<dim3(128, 64, batch_size), dim3(4, 64),
                                                512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 12, 6, 7,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCoreModulusOrdered<<<dim3(8192, 1, batch_size), dim3(8, 32),
                                                512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 5, 0, 6,
                        cfg.n_power, cfg.mod_inverse, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 23:
                    InverseCoreModulusOrdered<<<dim3(1, 16384, batch_size), dim3(256, 1),
                                                512 * sizeof(Data), cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 22, 14, 9,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCoreModulusOrdered<<<dim3(128, 128, batch_size), dim3(4, 64),
                                                512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 13, 7, 7,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCoreModulusOrdered<<<dim3(16384, 1, batch_size), dim3(4, 64),
                                                512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 6, 0, 7,
                        cfg.n_power, cfg.mod_inverse, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 24:
                    InverseCoreModulusOrdered<<<dim3(1, 16384, batch_size), dim3(512, 1),
                                                1024 * sizeof(Data), cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 9, 23, 14, 10,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCoreModulusOrdered<<<dim3(128, 128, batch_size), dim3(8, 64),
                                                1024 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 9, 13, 7, 7,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCoreModulusOrdered<<<dim3(16384, 1, batch_size), dim3(8, 64),
                                                1024 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 9, 6, 0, 7,
                        cfg.n_power, cfg.mod_inverse, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 25:
                    InverseCoreModulusOrdered_<<<dim3(32768, 1, batch_size), dim3(512, 1),
                                                 1024 * sizeof(Data), cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 9, 24, 15, 10,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ///////////////////////////////////////////////////////////
                    InverseCoreModulusOrdered<<<dim3(256, 128, batch_size), dim3(4, 128),
                                                1024 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 9, 14, 7, 8,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCoreModulusOrdered<<<dim3(32768, 1, batch_size), dim3(8, 64),
                                                1024 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 9, 6, 0, 7,
                        cfg.n_power, cfg.mod_inverse, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 26:
                    InverseCoreModulusOrdered_<<<dim3(65536, 1, batch_size), dim3(512, 1),
                                                 1024 * sizeof(Data), cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 9, 25, 16, 10,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ///////////////////////////////////////////////////////////
                    InverseCoreModulusOrdered<<<dim3(256, 256, batch_size), dim3(4, 128),
                                                1024 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 9, 15, 8, 8,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCoreModulusOrdered<<<dim3(65536, 1, batch_size), dim3(4, 128),
                                                1024 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 9, 7, 0, 8,
                        cfg.n_power, cfg.mod_inverse, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 27:
#ifndef CC_89
                    InverseCoreModulusOrdered_<<<dim3(262144, 1, batch_size), dim3(256, 1),
                                                 512 * sizeof(Data), cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 26, 18, 9,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ///////////////////////////////////////////////////////////
                    InverseCoreModulusOrdered<<<dim3(128, 2048, batch_size), dim3(4, 64),
                                                512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 17, 11, 7,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCoreModulusOrdered<<<dim3(8192, 32, batch_size), dim3(8, 32),
                                                512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 10, 5, 6,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCoreModulusOrdered<<<dim3(262144, 1, batch_size), dim3(16, 16),
                                                512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 4, 0, 5,
                        cfg.n_power, cfg.mod_inverse, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
#else
                    InverseCoreModulusOrdered_<<<dim3(131072, 1, batch_size), dim3(512, 1),
                                                 1024 * sizeof(Data), cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 9, 26, 17, 10,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ///////////////////////////////////////////////////////////
                    InverseCoreModulusOrdered<<<dim3(512, 256, batch_size), dim3(2, 256),
                                                1024 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 9, 16, 8, 9,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCoreModulusOrdered<<<dim3(131072, 1, batch_size), dim3(4, 128),
                                                1024 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 9, 7, 0, 8,
                        cfg.n_power, cfg.mod_inverse, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
#endif
                    break;
                case 28:
#ifndef CC_89
                    InverseCoreModulusOrdered_<<<dim3(524288, 1, batch_size), dim3(256, 1),
                                                 512 * sizeof(Data), cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 27, 19, 9,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ///////////////////////////////////////////////////////////
                    InverseCoreModulusOrdered<<<dim3(128, 4096, batch_size), dim3(4, 64),
                                                512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 18, 12, 7,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCoreModulusOrdered<<<dim3(8192, 64, batch_size), dim3(8, 32),
                                                512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 11, 6, 6,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCoreModulusOrdered<<<dim3(524288, 1, batch_size), dim3(8, 32),
                                                512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 5, 0, 6,
                        cfg.n_power, cfg.mod_inverse, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
#else
                    InverseCoreModulusOrdered_<<<dim3(262144, 1, batch_size), dim3(512, 1),
                                                 1024 * sizeof(Data), cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 9, 27, 18, 10,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ///////////////////////////////////////////////////////////
                    InverseCoreModulusOrdered<<<dim3(512, 512, batch_size), dim3(2, 256),
                                                1024 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 9, 17, 9, 9,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCoreModulusOrdered<<<dim3(262144, 1, batch_size), dim3(2, 256),
                                                1024 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 9, 8, 0, 9,
                        cfg.n_power, cfg.mod_inverse, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
#endif
                    break;

                default:
                    break;
            }
            break;

        default:
            break;
    }
}

__host__ void GPU_NTT_Modulus_Ordered_Inplace(Data* device_inout, Root* root_of_unity_table,
                                              Modulus* modulus, ntt_rns_configuration cfg,
                                              int batch_size, int mod_count, int* order)
{
    GPU_NTT_Modulus_Ordered(device_inout, device_inout, root_of_unity_table, modulus, cfg,
                            batch_size, mod_count, order);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Poly Ordered

__global__ void ForwardCorePolyOrdered(Data* polynomial_in, Data* polynomial_out,
                                       Root* root_of_unity_table, Modulus* modulus,
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

    extern __shared__ Data shared_memory[];

    int t_2 = N_power - logm - 1;
    location_t offset = 1 << (N_power - logm - 1);
    int t_ = shared_index;
    location_t m = (location_t)1 << logm;

    location_t global_addresss =
        idx_x + (location_t)(idx_y * (offset / (1 << (outer_iteration_count - 1)))) +
        (location_t)(blockDim.x * block_x) + (location_t)(2 * block_y * offset) +
        (location_t)(input_index << N_power);

    location_t omega_addresss =
        idx_x + (location_t)(idx_y * (offset / (1 << (outer_iteration_count - 1)))) +
        (location_t)(blockDim.x * block_x) + (location_t)(block_y * offset);

    location_t shared_addresss = (idx_x + (idx_y * blockDim.x));

    // Load data from global & store to shared
    shared_memory[shared_addresss] = polynomial_in[global_addresss];
    shared_memory[shared_addresss + (blockDim.x * blockDim.y)] =
        polynomial_in[global_addresss + offset];

    int t = 1 << t_;
    int in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
    location_t current_root_index;
    if(not_last_kernel)
    {
#pragma unroll
        for(int lp = 0; lp < outer_iteration_count; lp++)
        {
            __syncthreads();
            if(reduction_poly_check)
            {  // X_N_minus
                current_root_index = (omega_addresss >> t_2) + (location_t)(mod_index << N_power);
            }
            else
            {  // X_N_plus
                current_root_index =
                    m + (omega_addresss >> t_2) + (location_t)(mod_index << N_power);
            }
            CooleyTukeyUnit(shared_memory[in_shared_address], shared_memory[in_shared_address + t],
                            root_of_unity_table[current_root_index], modulus[mod_index]);

            t = t >> 1;
            t_2 -= 1;
            t_ -= 1;
            m <<= 1;

            in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
            //__syncthreads();
        }
        __syncthreads();
    }
    else
    {
#pragma unroll
        for(int lp = 0; lp < (shared_index - 5); lp++)  // 4 for 512 thread
        {
            __syncthreads();
            if(reduction_poly_check)
            {  // X_N_minus
                current_root_index = (omega_addresss >> t_2) + (location_t)(mod_index << N_power);
            }
            else
            {  // X_N_plus
                current_root_index =
                    m + (omega_addresss >> t_2) + (location_t)(mod_index << N_power);
            }

            CooleyTukeyUnit(shared_memory[in_shared_address], shared_memory[in_shared_address + t],
                            root_of_unity_table[current_root_index], modulus[mod_index]);

            t = t >> 1;
            t_2 -= 1;
            t_ -= 1;
            m <<= 1;

            in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
            //__syncthreads();
        }
        __syncthreads();

#pragma unroll
        for(int lp = 0; lp < 6; lp++)
        {
            if(reduction_poly_check)
            {  // X_N_minus
                current_root_index = (omega_addresss >> t_2) + (location_t)(mod_index << N_power);
            }
            else
            {  // X_N_plus
                current_root_index =
                    m + (omega_addresss >> t_2) + (location_t)(mod_index << N_power);
            }
            CooleyTukeyUnit(shared_memory[in_shared_address], shared_memory[in_shared_address + t],
                            root_of_unity_table[current_root_index], modulus[mod_index]);

            t = t >> 1;
            t_2 -= 1;
            t_ -= 1;
            m <<= 1;

            in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
        }
        __syncthreads();
    }

    polynomial_out[global_addresss] = shared_memory[shared_addresss];
    polynomial_out[global_addresss + offset] =
        shared_memory[shared_addresss + (blockDim.x * blockDim.y)];
}

__global__ void ForwardCorePolyOrdered_(Data* polynomial_in, Data* polynomial_out,
                                        Root* root_of_unity_table, Modulus* modulus,
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

    extern __shared__ Data shared_memory[];

    int t_2 = N_power - logm - 1;
    location_t offset = 1 << (N_power - logm - 1);
    int t_ = shared_index;
    location_t m = (location_t)1 << logm;

    location_t global_addresss =
        idx_x + (location_t)(idx_y * (offset / (1 << (outer_iteration_count - 1)))) +
        (location_t)(blockDim.x * block_y) + (location_t)(2 * block_x * offset) +
        (location_t)(input_index << N_power);
    location_t omega_addresss =
        idx_x + (location_t)(idx_y * (offset / (1 << (outer_iteration_count - 1)))) +
        (location_t)(blockDim.x * block_y) + (location_t)(block_x * offset);
    location_t shared_addresss = (idx_x + (idx_y * blockDim.x));

    // Load data from global & store to shared
    shared_memory[shared_addresss] = polynomial_in[global_addresss];
    shared_memory[shared_addresss + (blockDim.x * blockDim.y)] =
        polynomial_in[global_addresss + offset];

    int t = 1 << t_;
    int in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
    location_t current_root_index;
    if(not_last_kernel)
    {
#pragma unroll
        for(int lp = 0; lp < outer_iteration_count; lp++)
        {
            __syncthreads();
            if(reduction_poly_check)
            {  // X_N_minus
                current_root_index = (omega_addresss >> t_2) + (location_t)(mod_index << N_power);
            }
            else
            {  // X_N_plus
                current_root_index =
                    m + (omega_addresss >> t_2) + (location_t)(mod_index << N_power);
            }
            CooleyTukeyUnit(shared_memory[in_shared_address], shared_memory[in_shared_address + t],
                            root_of_unity_table[current_root_index], modulus[mod_index]);

            t = t >> 1;
            t_2 -= 1;
            t_ -= 1;
            m <<= 1;

            in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
            //__syncthreads();
        }
        __syncthreads();
    }
    else
    {
#pragma unroll
        for(int lp = 0; lp < (shared_index - 5); lp++)
        {
            __syncthreads();
            if(reduction_poly_check)
            {  // X_N_minus
                current_root_index = (omega_addresss >> t_2) + (location_t)(mod_index << N_power);
            }
            else
            {  // X_N_plus
                current_root_index =
                    m + (omega_addresss >> t_2) + (location_t)(mod_index << N_power);
            }
            CooleyTukeyUnit(shared_memory[in_shared_address], shared_memory[in_shared_address + t],
                            root_of_unity_table[current_root_index], modulus[mod_index]);

            t = t >> 1;
            t_2 -= 1;
            t_ -= 1;
            m <<= 1;

            in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
            //__syncthreads();
        }
        __syncthreads();

#pragma unroll
        for(int lp = 0; lp < 6; lp++)
        {
            if(reduction_poly_check)
            {  // X_N_minus
                current_root_index = (omega_addresss >> t_2) + (location_t)(mod_index << N_power);
            }
            else
            {  // X_N_plus
                current_root_index =
                    m + (omega_addresss >> t_2) + (location_t)(mod_index << N_power);
            }
            CooleyTukeyUnit(shared_memory[in_shared_address], shared_memory[in_shared_address + t],
                            root_of_unity_table[current_root_index], modulus[mod_index]);

            t = t >> 1;
            t_2 -= 1;
            t_ -= 1;
            m <<= 1;

            in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
        }
        __syncthreads();
    }

    polynomial_out[global_addresss] = shared_memory[shared_addresss];
    polynomial_out[global_addresss + offset] =
        shared_memory[shared_addresss + (blockDim.x * blockDim.y)];
}

__global__ void InverseCorePolyOrdered(Data* polynomial_in, Data* polynomial_out,
                                       Root* inverse_root_of_unity_table, Modulus* modulus,
                                       int shared_index, int logm, int k, int outer_iteration_count,
                                       int N_power, Ninverse* n_inverse, bool last_kernel,
                                       bool reduction_poly_check, int mod_count, int* order)
{
    const int idx_x = threadIdx.x;
    const int idx_y = threadIdx.y;
    const int block_x = blockIdx.x;
    const int block_y = blockIdx.y;
    const int block_z = blockIdx.z;

    const int mod_index = block_z % mod_count;
    const int input_index = order[block_z];

    extern __shared__ Data shared_memory[];

    int t_2 = N_power - logm - 1;
    location_t offset = 1 << (N_power - k - 1);
    // int t_ = 9 - outer_iteration_count;
    int t_ = (shared_index + 1) - outer_iteration_count;
    int loops = outer_iteration_count;
    location_t m = (location_t)1 << logm;

    location_t global_addresss =
        idx_x + (location_t)(idx_y * (offset / (1 << (outer_iteration_count - 1)))) +
        (location_t)(blockDim.x * block_x) + (location_t)(2 * block_y * offset) +
        (location_t)(input_index << N_power);

    location_t omega_addresss =
        idx_x + (location_t)(idx_y * (offset / (1 << (outer_iteration_count - 1)))) +
        (location_t)(blockDim.x * block_x) + (location_t)(block_y * offset);
    location_t shared_addresss = (idx_x + (idx_y * blockDim.x));

    shared_memory[shared_addresss] = polynomial_in[global_addresss];
    shared_memory[shared_addresss + (blockDim.x * blockDim.y)] =
        polynomial_in[global_addresss + offset];

    int t = 1 << t_;
    int in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
    location_t current_root_index;
#pragma unroll
    for(int lp = 0; lp < loops; lp++)
    {
        __syncthreads();
        if(reduction_poly_check)
        {  // X_N_minus
            current_root_index = (omega_addresss >> t_2) + (location_t)(mod_index << N_power);
        }
        else
        {  // X_N_plus
            current_root_index = m + (omega_addresss >> t_2) + (location_t)(mod_index << N_power);
        }

        GentlemanSandeUnit(shared_memory[in_shared_address], shared_memory[in_shared_address + t],
                           inverse_root_of_unity_table[current_root_index], modulus[mod_index]);

        t = t << 1;
        t_2 += 1;
        t_ += 1;
        m >>= 1;

        in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
    }
    __syncthreads();

    if(last_kernel)
    {
        polynomial_out[global_addresss] = VALUE_GPU::mult(shared_memory[shared_addresss],
                                                          n_inverse[mod_index], modulus[mod_index]);
        polynomial_out[global_addresss + offset] =
            VALUE_GPU::mult(shared_memory[shared_addresss + (blockDim.x * blockDim.y)],
                            n_inverse[mod_index], modulus[mod_index]);
    }
    else
    {
        polynomial_out[global_addresss] = shared_memory[shared_addresss];
        polynomial_out[global_addresss + offset] =
            shared_memory[shared_addresss + (blockDim.x * blockDim.y)];
    }
}

__global__ void InverseCorePolyOrdered_(Data* polynomial_in, Data* polynomial_out,
                                        Root* inverse_root_of_unity_table, Modulus* modulus,
                                        int shared_index, int logm, int k,
                                        int outer_iteration_count, int N_power, Ninverse* n_inverse,
                                        bool last_kernel, bool reduction_poly_check, int mod_count,
                                        int* order)
{
    const int idx_x = threadIdx.x;
    const int idx_y = threadIdx.y;
    const int block_x = blockIdx.x;
    const int block_y = blockIdx.y;
    const int block_z = blockIdx.z;

    const int mod_index = block_z % mod_count;
    const int input_index = order[block_z];

    extern __shared__ Data shared_memory[];

    int t_2 = N_power - logm - 1;
    location_t offset = 1 << (N_power - k - 1);
    // int t_ = 9 - outer_iteration_count;
    int t_ = (shared_index + 1) - outer_iteration_count;
    int loops = outer_iteration_count;
    location_t m = (location_t)1 << logm;

    location_t global_addresss =
        idx_x + (location_t)(idx_y * (offset / (1 << (outer_iteration_count - 1)))) +
        (location_t)(blockDim.x * block_y) + (location_t)(2 * block_x * offset) +
        (location_t)(input_index << N_power);

    location_t omega_addresss =
        idx_x + (location_t)(idx_y * (offset / (1 << (outer_iteration_count - 1)))) +
        (location_t)(blockDim.x * block_y) + (location_t)(block_x * offset);
    location_t shared_addresss = (idx_x + (idx_y * blockDim.x));

    shared_memory[shared_addresss] = polynomial_in[global_addresss];
    shared_memory[shared_addresss + (blockDim.x * blockDim.y)] =
        polynomial_in[global_addresss + offset];

    int t = 1 << t_;
    int in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
    location_t current_root_index;
#pragma unroll
    for(int lp = 0; lp < loops; lp++)
    {
        __syncthreads();
        if(reduction_poly_check)
        {  // X_N_minus
            current_root_index = (omega_addresss >> t_2) + (location_t)(mod_index << N_power);
        }
        else
        {  // X_N_plus
            current_root_index = m + (omega_addresss >> t_2) + (location_t)(mod_index << N_power);
        }

        GentlemanSandeUnit(shared_memory[in_shared_address], shared_memory[in_shared_address + t],
                           inverse_root_of_unity_table[current_root_index], modulus[mod_index]);

        t = t << 1;
        t_2 += 1;
        t_ += 1;
        m >>= 1;

        in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
    }
    __syncthreads();

    if(last_kernel)
    {
        polynomial_out[global_addresss] = VALUE_GPU::mult(shared_memory[shared_addresss],
                                                          n_inverse[mod_index], modulus[mod_index]);
        polynomial_out[global_addresss + offset] =
            VALUE_GPU::mult(shared_memory[shared_addresss + (blockDim.x * blockDim.y)],
                            n_inverse[mod_index], modulus[mod_index]);
    }
    else
    {
        polynomial_out[global_addresss] = shared_memory[shared_addresss];
        polynomial_out[global_addresss + offset] =
            shared_memory[shared_addresss + (blockDim.x * blockDim.y)];
    }
}

__host__ void GPU_NTT_Poly_Ordered(Data* device_in, Data* device_out, Root* root_of_unity_table,
                                   Modulus* modulus, ntt_rns_configuration cfg, int batch_size,
                                   int mod_count, int* order)
{
    switch(cfg.ntt_type)
    {
        case FORWARD:
            switch(cfg.n_power)
            {
                case 12:
                    ForwardCorePolyOrdered<<<dim3(8, 1, batch_size), dim3(64, 4),
                                             512 * sizeof(Data), cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 0, 3, cfg.n_power,
                        cfg.zero_padding, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCorePolyOrdered<<<dim3(1, 8, batch_size), dim3(256, 1),
                                             512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 3, 9, cfg.n_power,
                        false, false, (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                        mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 13:
                    ForwardCorePolyOrdered<<<dim3(16, 1, batch_size), dim3(32, 8),
                                             512 * sizeof(Data), cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 0, 4, cfg.n_power,
                        cfg.zero_padding, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCorePolyOrdered<<<dim3(1, 16, batch_size), dim3(256, 1),
                                             512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 4, 9, cfg.n_power,
                        false, false, (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                        mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 14:
                    ForwardCorePolyOrdered<<<dim3(32, 1, batch_size), dim3(16, 16),
                                             512 * sizeof(Data), cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 0, 5, cfg.n_power,
                        cfg.zero_padding, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCorePolyOrdered<<<dim3(1, 32, batch_size), dim3(256, 1),
                                             512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 5, 9, cfg.n_power,
                        false, false, (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                        mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 15:
                    ForwardCorePolyOrdered<<<dim3(64, 1, batch_size), dim3(8, 32),
                                             512 * sizeof(Data), cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 0, 6, cfg.n_power,
                        cfg.zero_padding, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCorePolyOrdered<<<dim3(1, 64, batch_size), dim3(256, 1),
                                             512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 6, 9, cfg.n_power,
                        false, false, (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                        mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 16:
                    ForwardCorePolyOrdered<<<dim3(128, 1, batch_size), dim3(4, 64),
                                             512 * sizeof(Data), cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 0, 7, cfg.n_power,
                        cfg.zero_padding, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCorePolyOrdered<<<dim3(1, 128, batch_size), dim3(256, 1),
                                             512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 7, 9, cfg.n_power,
                        false, false, (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                        mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 17:
                    ForwardCorePolyOrdered<<<dim3(256, 1, batch_size), dim3(32, 8),
                                             512 * sizeof(Data), cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 0, 4, cfg.n_power,
                        cfg.zero_padding, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCorePolyOrdered<<<dim3(16, 16, batch_size), dim3(32, 8),
                                             512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 4, 4, cfg.n_power,
                        false, true, (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                        mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCorePolyOrdered<<<dim3(1, 256, batch_size), dim3(256, 1),
                                             512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 8, 9, cfg.n_power,
                        false, false, (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                        mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 18:
                    ForwardCorePolyOrdered<<<dim3(512, 1, batch_size), dim3(32, 8),
                                             512 * sizeof(Data), cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 0, 4, cfg.n_power,
                        cfg.zero_padding, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCorePolyOrdered<<<dim3(32, 16, batch_size), dim3(16, 16),
                                             512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 4, 5, cfg.n_power,
                        false, true, (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                        mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCorePolyOrdered<<<dim3(1, 512, batch_size), dim3(256, 1),
                                             512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 9, 9, cfg.n_power,
                        false, false, (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                        mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 19:
                    ForwardCorePolyOrdered<<<dim3(1024, 1, batch_size), dim3(16, 16),
                                             512 * sizeof(Data), cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 0, 5, cfg.n_power,
                        cfg.zero_padding, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCorePolyOrdered<<<dim3(32, 32, batch_size), dim3(16, 16),
                                             512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 5, 5, cfg.n_power,
                        false, true, (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                        mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCorePolyOrdered<<<dim3(1, 1024, batch_size), dim3(256, 1),
                                             512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 10, 9, cfg.n_power,
                        false, false, (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                        mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 20:
                    ForwardCorePolyOrdered<<<dim3(2048, 1, batch_size), dim3(16, 16),
                                             512 * sizeof(Data), cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 0, 5, cfg.n_power,
                        cfg.zero_padding, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCorePolyOrdered<<<dim3(64, 32, batch_size), dim3(8, 32),
                                             512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 5, 6, cfg.n_power,
                        false, true, (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                        mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCorePolyOrdered<<<dim3(1, 2048, batch_size), dim3(256, 1),
                                             512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 11, 9, cfg.n_power,
                        false, false, (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                        mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 21:
                    ForwardCorePolyOrdered<<<dim3(4096, 1, batch_size), dim3(8, 32),
                                             512 * sizeof(Data), cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 0, 6, cfg.n_power,
                        cfg.zero_padding, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCorePolyOrdered<<<dim3(64, 64, batch_size), dim3(8, 32),
                                             512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 6, 6, cfg.n_power,
                        false, true, (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                        mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCorePolyOrdered<<<dim3(1, 4096, batch_size), dim3(256, 1),
                                             512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 12, 9, cfg.n_power,
                        false, false, (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                        mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 22:
                    ForwardCorePolyOrdered<<<dim3(8192, 1, batch_size), dim3(8, 32),
                                             512 * sizeof(Data), cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 0, 6, cfg.n_power,
                        cfg.zero_padding, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCorePolyOrdered<<<dim3(128, 64, batch_size), dim3(4, 64),
                                             512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 6, 7, cfg.n_power,
                        false, true, (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                        mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCorePolyOrdered<<<dim3(1, 8192, batch_size), dim3(256, 1),
                                             512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 13, 9, cfg.n_power,
                        false, false, (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                        mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());

                    break;
                case 23:
                    ForwardCorePolyOrdered<<<dim3(16384, 1, batch_size), dim3(4, 64),
                                             512 * sizeof(Data), cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 0, 7, cfg.n_power,
                        cfg.zero_padding, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCorePolyOrdered<<<dim3(128, 128, batch_size), dim3(4, 64),
                                             512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 7, 7, cfg.n_power,
                        false, true, (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                        mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCorePolyOrdered<<<dim3(1, 16384, batch_size), dim3(256, 1),
                                             512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 14, 9, cfg.n_power,
                        false, false, (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                        mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 24:
                    ForwardCorePolyOrdered<<<dim3(16384, 1, batch_size), dim3(8, 64),
                                             1024 * sizeof(Data), cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 9, 0, 7, cfg.n_power,
                        cfg.zero_padding, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCorePolyOrdered<<<dim3(128, 128, batch_size), dim3(8, 64),
                                             1024 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 9, 7, 7, cfg.n_power,
                        false, true, (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                        mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCorePolyOrdered<<<dim3(1, 16384, batch_size), dim3(512, 1),
                                             1024 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 9, 14, 10,
                        cfg.n_power, false, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 25:
                    ForwardCorePolyOrdered<<<dim3(32768, 1, batch_size), dim3(8, 64),
                                             1024 * sizeof(Data), cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 9, 0, 7, cfg.n_power,
                        cfg.zero_padding, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCorePolyOrdered<<<dim3(256, 128, batch_size), dim3(4, 128),
                                             1024 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 9, 7, 8, cfg.n_power,
                        false, true, (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                        mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ///////////////////////////////////////////////////////////
                    ForwardCorePolyOrdered_<<<dim3(32768, 1, batch_size), dim3(512, 1),
                                              1024 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 9, 15, 10,
                        cfg.n_power, false, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 26:
                    ForwardCorePolyOrdered<<<dim3(65536, 1, batch_size), dim3(4, 128),
                                             1024 * sizeof(Data), cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 9, 0, 8, cfg.n_power,
                        cfg.zero_padding, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCorePolyOrdered<<<dim3(256, 256, batch_size), dim3(4, 128),
                                             1024 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 9, 8, 8, cfg.n_power,
                        false, true, (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                        mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ///////////////////////////////////////////////////////////
                    ForwardCorePolyOrdered_<<<dim3(65536, 1, batch_size), dim3(512, 1),
                                              1024 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 9, 16, 10,
                        cfg.n_power, false, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 27:
#ifndef CC_89
                    ForwardCorePolyOrdered<<<dim3(262144, 1, batch_size), dim3(16, 16),
                                             512 * sizeof(Data), cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 0, 5, cfg.n_power,
                        cfg.zero_padding, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCorePolyOrdered<<<dim3(8192, 32, batch_size), dim3(8, 32),
                                             512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 5, 6, cfg.n_power,
                        false, true, (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                        mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCorePolyOrdered<<<dim3(128, 2048, batch_size), dim3(4, 64),
                                             512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 11, 7, cfg.n_power,
                        false, true, (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                        mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ///////////////////////////////////////////////////////////
                    ForwardCorePolyOrdered_<<<dim3(262144, 1, batch_size), dim3(256, 1),
                                              512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 18, 9, cfg.n_power,
                        false, false, (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                        mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
#else
                    ForwardCorePolyOrdered<<<dim3(131072, 1, batch_size), dim3(4, 128),
                                             1024 * sizeof(Data), cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 9, 0, 8, cfg.n_power,
                        cfg.zero_padding, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCorePolyOrdered<<<dim3(512, 256, batch_size), dim3(2, 256),
                                             1024 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 9, 8, 9, cfg.n_power,
                        false, true, (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                        mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ///////////////////////////////////////////////////////////
                    ForwardCorePolyOrdered_<<<dim3(131072, 1, batch_size), dim3(512, 1),
                                              1024 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 9, 17, 10,
                        cfg.n_power, false, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
#endif
                    break;
                case 28:
#ifndef CC_89
                    ForwardCorePolyOrdered<<<dim3(524288, 1, batch_size), dim3(8, 32),
                                             512 * sizeof(Data), cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 0, 6, cfg.n_power,
                        cfg.zero_padding, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCorePolyOrdered<<<dim3(8192, 64, batch_size), dim3(8, 32),
                                             512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 6, 6, cfg.n_power,
                        false, true, (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                        mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCorePolyOrdered<<<dim3(128, 4096, batch_size), dim3(4, 64),
                                             512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 12, 7, cfg.n_power,
                        false, true, (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                        mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ///////////////////////////////////////////////////////////
                    ForwardCorePolyOrdered_<<<dim3(524288, 1, batch_size), dim3(256, 1),
                                              512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 19, 9, cfg.n_power,
                        false, false, (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                        mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
#else
                    ForwardCorePolyOrdered<<<dim3(262144, 1, batch_size), dim3(2, 256),
                                             1024 * sizeof(Data), cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 9, 0, 9, cfg.n_power,
                        cfg.zero_padding, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCorePolyOrdered<<<dim3(512, 512, batch_size), dim3(2, 256),
                                             1024 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 9, 9, 9, cfg.n_power,
                        false, true, (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                        mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ///////////////////////////////////////////////////////////
                    ForwardCorePolyOrdered_<<<dim3(262144, 1, batch_size), dim3(512, 1),
                                              1024 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 9, 18, 10,
                        cfg.n_power, false, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
#endif
                    break;

                default:
                    break;
            }
            break;
        case INVERSE:
            switch(cfg.n_power)
            {
                case 12:
                    InverseCorePolyOrdered<<<dim3(1, 8, batch_size), dim3(256, 1),
                                             512 * sizeof(Data), cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 11, 3, 9,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCorePolyOrdered<<<dim3(8, 1, batch_size), dim3(64, 4),
                                             512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 2, 0, 3,
                        cfg.n_power, cfg.mod_inverse, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 13:
                    InverseCorePolyOrdered<<<dim3(1, 16, batch_size), dim3(256, 1),
                                             512 * sizeof(Data), cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 12, 4, 9,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCorePolyOrdered<<<dim3(16, 1, batch_size), dim3(32, 8),
                                             512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 3, 0, 4,
                        cfg.n_power, cfg.mod_inverse, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 14:
                    InverseCorePolyOrdered<<<dim3(1, 32, batch_size), dim3(256, 1),
                                             512 * sizeof(Data), cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 13, 5, 9,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCorePolyOrdered<<<dim3(32, 1, batch_size), dim3(16, 16),
                                             512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 4, 0, 5,
                        cfg.n_power, cfg.mod_inverse, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 15:
                    InverseCorePolyOrdered<<<dim3(1, 64, batch_size), dim3(256, 1),
                                             512 * sizeof(Data), cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 14, 6, 9,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCorePolyOrdered<<<dim3(64, 1, batch_size), dim3(8, 32),
                                             512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 5, 0, 6,
                        cfg.n_power, cfg.mod_inverse, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 16:
                    InverseCorePolyOrdered<<<dim3(1, 128, batch_size), dim3(256, 1),
                                             512 * sizeof(Data), cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 15, 7, 9,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCorePolyOrdered<<<dim3(128, 1, batch_size), dim3(4, 64),
                                             512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 6, 0, 7,
                        cfg.n_power, cfg.mod_inverse, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 17:
                    InverseCorePolyOrdered<<<dim3(1, 256, batch_size), dim3(256, 1),
                                             512 * sizeof(Data), cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 16, 8, 9,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCorePolyOrdered<<<dim3(16, 16, batch_size), dim3(32, 8),
                                             512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 7, 4, 4,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCorePolyOrdered<<<dim3(256, 1, batch_size), dim3(32, 8),
                                             512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 3, 0, 4,
                        cfg.n_power, cfg.mod_inverse, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 18:
                    InverseCorePolyOrdered<<<dim3(1, 512, batch_size), dim3(256, 1),
                                             512 * sizeof(Data), cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 17, 9, 9,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCorePolyOrdered<<<dim3(32, 16, batch_size), dim3(16, 16),
                                             512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 8, 4, 5,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCorePolyOrdered<<<dim3(512, 1, batch_size), dim3(32, 8),
                                             512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 3, 0, 4,
                        cfg.n_power, cfg.mod_inverse, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 19:
                    InverseCorePolyOrdered<<<dim3(1, 1024, batch_size), dim3(256, 1),
                                             512 * sizeof(Data), cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 18, 10, 9,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCorePolyOrdered<<<dim3(32, 32, batch_size), dim3(16, 16),
                                             512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 9, 5, 5,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCorePolyOrdered<<<dim3(1024, 1, batch_size), dim3(16, 16),
                                             512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 4, 0, 5,
                        cfg.n_power, cfg.mod_inverse, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 20:
                    InverseCorePolyOrdered<<<dim3(1, 2048, batch_size), dim3(256, 1),
                                             512 * sizeof(Data), cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 19, 11, 9,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCorePolyOrdered<<<dim3(64, 32, batch_size), dim3(8, 32),
                                             512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 10, 5, 6,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCorePolyOrdered<<<dim3(2048, 1, batch_size), dim3(16, 16),
                                             512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 4, 0, 5,
                        cfg.n_power, cfg.mod_inverse, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 21:  //
                    InverseCorePolyOrdered<<<dim3(1, 4096, batch_size), dim3(256, 1),
                                             512 * sizeof(Data), cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 20, 12, 9,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCorePolyOrdered<<<dim3(64, 64, batch_size), dim3(8, 32),
                                             512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 11, 6, 6,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCorePolyOrdered<<<dim3(4096, 1, batch_size), dim3(8, 32),
                                             512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 5, 0, 6,
                        cfg.n_power, cfg.mod_inverse, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 22:
                    InverseCorePolyOrdered<<<dim3(1, 8192, batch_size), dim3(256, 1),
                                             512 * sizeof(Data), cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 21, 13, 9,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCorePolyOrdered<<<dim3(128, 64, batch_size), dim3(4, 64),
                                             512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 12, 6, 7,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCorePolyOrdered<<<dim3(8192, 1, batch_size), dim3(8, 32),
                                             512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 5, 0, 6,
                        cfg.n_power, cfg.mod_inverse, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 23:
                    InverseCorePolyOrdered<<<dim3(1, 16384, batch_size), dim3(256, 1),
                                             512 * sizeof(Data), cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 22, 14, 9,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCorePolyOrdered<<<dim3(128, 128, batch_size), dim3(4, 64),
                                             512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 13, 7, 7,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCorePolyOrdered<<<dim3(16384, 1, batch_size), dim3(4, 64),
                                             512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 6, 0, 7,
                        cfg.n_power, cfg.mod_inverse, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 24:
                    InverseCorePolyOrdered<<<dim3(1, 16384, batch_size), dim3(512, 1),
                                             1024 * sizeof(Data), cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 9, 23, 14, 10,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCorePolyOrdered<<<dim3(128, 128, batch_size), dim3(8, 64),
                                             1024 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 9, 13, 7, 7,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCorePolyOrdered<<<dim3(16384, 1, batch_size), dim3(8, 64),
                                             1024 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 9, 6, 0, 7,
                        cfg.n_power, cfg.mod_inverse, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 25:
                    InverseCorePolyOrdered_<<<dim3(32768, 1, batch_size), dim3(512, 1),
                                              1024 * sizeof(Data), cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 9, 24, 15, 10,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ///////////////////////////////////////////////////////////
                    InverseCorePolyOrdered<<<dim3(256, 128, batch_size), dim3(4, 128),
                                             1024 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 9, 14, 7, 8,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCorePolyOrdered<<<dim3(32768, 1, batch_size), dim3(8, 64),
                                             1024 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 9, 6, 0, 7,
                        cfg.n_power, cfg.mod_inverse, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 26:
                    InverseCorePolyOrdered_<<<dim3(65536, 1, batch_size), dim3(512, 1),
                                              1024 * sizeof(Data), cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 9, 25, 16, 10,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ///////////////////////////////////////////////////////////
                    InverseCorePolyOrdered<<<dim3(256, 256, batch_size), dim3(4, 128),
                                             1024 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 9, 15, 8, 8,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCorePolyOrdered<<<dim3(65536, 1, batch_size), dim3(4, 128),
                                             1024 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 9, 7, 0, 8,
                        cfg.n_power, cfg.mod_inverse, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 27:
#ifndef CC_89
                    InverseCorePolyOrdered_<<<dim3(262144, 1, batch_size), dim3(256, 1),
                                              512 * sizeof(Data), cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 26, 18, 9,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ///////////////////////////////////////////////////////////
                    InverseCorePolyOrdered<<<dim3(128, 2048, batch_size), dim3(4, 64),
                                             512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 17, 11, 7,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCorePolyOrdered<<<dim3(8192, 32, batch_size), dim3(8, 32),
                                             512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 10, 5, 6,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCorePolyOrdered<<<dim3(262144, 1, batch_size), dim3(16, 16),
                                             512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 4, 0, 5,
                        cfg.n_power, cfg.mod_inverse, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
#else
                    InverseCorePolyOrdered_<<<dim3(131072, 1, batch_size), dim3(512, 1),
                                              1024 * sizeof(Data), cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 9, 26, 17, 10,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ///////////////////////////////////////////////////////////
                    InverseCorePolyOrdered<<<dim3(512, 256, batch_size), dim3(2, 256),
                                             1024 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 9, 16, 8, 9,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCorePolyOrdered<<<dim3(131072, 1, batch_size), dim3(4, 128),
                                             1024 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 9, 7, 0, 8,
                        cfg.n_power, cfg.mod_inverse, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
#endif
                    break;
                case 28:
#ifndef CC_89
                    InverseCorePolyOrdered_<<<dim3(524288, 1, batch_size), dim3(256, 1),
                                              512 * sizeof(Data), cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 8, 27, 19, 9,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ///////////////////////////////////////////////////////////
                    InverseCorePolyOrdered<<<dim3(128, 4096, batch_size), dim3(4, 64),
                                             512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 18, 12, 7,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCorePolyOrdered<<<dim3(8192, 64, batch_size), dim3(8, 32),
                                             512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 11, 6, 6,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCorePolyOrdered<<<dim3(524288, 1, batch_size), dim3(8, 32),
                                             512 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 8, 5, 0, 6,
                        cfg.n_power, cfg.mod_inverse, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
#else
                    InverseCorePolyOrdered_<<<dim3(262144, 1, batch_size), dim3(512, 1),
                                              1024 * sizeof(Data), cfg.stream>>>(
                        device_in, device_out, root_of_unity_table, modulus, 9, 27, 18, 10,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ///////////////////////////////////////////////////////////
                    InverseCorePolyOrdered<<<dim3(512, 512, batch_size), dim3(2, 256),
                                             1024 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 9, 17, 9, 9,
                        cfg.n_power, cfg.mod_inverse, false,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    InverseCorePolyOrdered<<<dim3(262144, 1, batch_size), dim3(2, 256),
                                             1024 * sizeof(Data), cfg.stream>>>(
                        device_out, device_out, root_of_unity_table, modulus, 9, 8, 0, 9,
                        cfg.n_power, cfg.mod_inverse, true,
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count, order);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
#endif
                    break;

                default:
                    break;
            }
            break;

        default:
            break;
    }
}

__host__ void GPU_NTT_Poly_Ordered_Inplace(Data* device_inout, Root* root_of_unity_table,
                                           Modulus* modulus, ntt_rns_configuration cfg,
                                           int batch_size, int mod_count, int* order)
{
    GPU_NTT_Poly_Ordered(device_inout, device_inout, root_of_unity_table, modulus, cfg, batch_size,
                         mod_count, order);
}
