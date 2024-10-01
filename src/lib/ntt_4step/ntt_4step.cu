// (C) Ulvetanna Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan
// Paper: https://eprint.iacr.org/2023/1410

#include "ntt_4step.cuh"

__device__ void CooleyTukeyUnit_(Data& U, Data& V, Root& root, Modulus& modulus)
{
    Data u_ = U;
    Data v_ = VALUE_GPU::mult(V, root, modulus);

    U = VALUE_GPU::add(u_, v_, modulus);
    V = VALUE_GPU::sub(u_, v_, modulus);
}

__device__ void GentlemanSandeUnit_(Data& U, Data& V, Root& root,
                                    Modulus& modulus)
{
    Data u_ = U;
    Data v_ = V;

    U = VALUE_GPU::add(u_, v_, modulus);

    v_ = VALUE_GPU::sub(u_, v_, modulus);
    V = VALUE_GPU::mult(v_, root, modulus);
}

__global__ void Transpose_Batch(Data* polynomial_in, Data* polynomial_out,
                                const int row, const int col, int n_power)
{
    int idx_x = threadIdx.x; // 16
    int idx_y = threadIdx.y; // 16

    int block_x = blockIdx.x * blockDim.x;
    int block_y = blockIdx.y * blockDim.y;

    int divindex = blockIdx.z << n_power;

    __shared__ Data sharedmemorys[16][16];

    sharedmemorys[idx_y][idx_x] =
        polynomial_in[((block_y + idx_y) * col) + block_x + idx_x + divindex];
    __syncthreads();

    polynomial_out[((block_x + idx_y) * row) + block_y + idx_x + divindex] =
        sharedmemorys[idx_x][idx_y];
}

__host__ void GPU_Transpose(Data* input, Data* output, const int row,
                            const int col, const int n_power,
                            const int batch_size)
{
    Transpose_Batch<<<dim3(col >> 4, row >> 4, batch_size), dim3(16, 16)>>>(
        input, output, row, col, n_power);
    THROW_IF_CUDA_ERROR(cudaGetLastError());
}

__global__ void FourStepForwardCoreT1(Data* polynomial_in, Data* polynomial_out,
                                      Root* n1_root_of_unity_table,
                                      Modulus* modulus, int index1, int index2,
                                      int n_power, int mod_count)
{
    int idx_x = threadIdx.x;
    int idx_y = threadIdx.y;
    int block_x = blockIdx.x;
    int block_y = blockIdx.y;

    __shared__ Data sharedmemorys[32][32 + 1];

    int q_index = block_y % mod_count;
    Modulus q_thread = modulus[q_index];

    int idx_index = idx_x + (idx_y << 5);
    int global_addresss = idx_index + (block_x << 10);
    int divindex = block_y << n_power;

    // Load data from global & store to shared
    sharedmemorys[idx_y][idx_x] = polynomial_in[global_addresss + divindex];
    sharedmemorys[idx_y + 8][idx_x] =
        polynomial_in[global_addresss + 256 + divindex];
    sharedmemorys[idx_y + 16][idx_x] =
        polynomial_in[global_addresss + 512 + divindex];
    sharedmemorys[idx_y + 24][idx_x] =
        polynomial_in[global_addresss + 768 + divindex];
    __syncthreads();

    int global_index1 = idx_x >> 4;
    int global_index2 = idx_x % 16;

    int t_ = 4;
    int t = 1 << t_;
    int in_shared_address = ((global_index2 >> t_) << t_) + global_index2;

    CooleyTukeyUnit_(
        sharedmemorys[(idx_y << 1) + global_index1][in_shared_address],
        sharedmemorys[(idx_y << 1) + global_index1][in_shared_address + t],
        n1_root_of_unity_table[(global_index2 >> t_)], q_thread);
    CooleyTukeyUnit_(
        sharedmemorys[(idx_y << 1) + global_index1 + 16][in_shared_address],
        sharedmemorys[(idx_y << 1) + global_index1 + 16][in_shared_address + t],
        n1_root_of_unity_table[(global_index2 >> t_)], q_thread);
    __syncthreads();

    for (int i = 0; i < 4; i++)
    {
        t = t >> 1;
        t_ -= 1;

        in_shared_address = ((global_index2 >> t_) << t_) + global_index2;

        CooleyTukeyUnit_(
            sharedmemorys[(idx_y << 1) + global_index1][in_shared_address],
            sharedmemorys[(idx_y << 1) + global_index1][in_shared_address + t],
            n1_root_of_unity_table[(global_index2 >> t_)], q_thread);
        CooleyTukeyUnit_(
            sharedmemorys[(idx_y << 1) + global_index1 + 16][in_shared_address],
            sharedmemorys[(idx_y << 1) + global_index1 + 16]
                         [in_shared_address + t],
            n1_root_of_unity_table[(global_index2 >> t_)], q_thread);
        __syncthreads();
    }
    __syncthreads();

    polynomial_out[idx_x + (idx_y << index1) + (block_x << 5) + divindex] =
        sharedmemorys[idx_x][idx_y];
    polynomial_out[idx_x + (idx_y << index1) + (block_x << 5) + index2 +
                   divindex] = sharedmemorys[idx_x][idx_y + 8];
    polynomial_out[idx_x + (idx_y << index1) + (block_x << 5) + (index2 * 2) +
                   divindex] = sharedmemorys[idx_x][idx_y + 16];
    polynomial_out[idx_x + (idx_y << index1) + (block_x << 5) + (index2 * 3) +
                   divindex] = sharedmemorys[idx_x][idx_y + 24];
}

__global__ void FourStepForwardCoreT1(Data* polynomial_in, Data* polynomial_out,
                                      Root* n1_root_of_unity_table,
                                      Modulus modulus, int index1, int index2,
                                      int n_power)
{
    int idx_x = threadIdx.x;
    int idx_y = threadIdx.y;
    int block_x = blockIdx.x;
    int block_y = blockIdx.y;

    __shared__ Data sharedmemorys[32][32 + 1];

    Modulus q_thread = modulus;

    int idx_index = idx_x + (idx_y << 5);
    int global_addresss = idx_index + (block_x << 10);
    int divindex = block_y << n_power;

    // Load data from global & store to shared
    sharedmemorys[idx_y][idx_x] = polynomial_in[global_addresss + divindex];
    sharedmemorys[idx_y + 8][idx_x] =
        polynomial_in[global_addresss + 256 + divindex];
    sharedmemorys[idx_y + 16][idx_x] =
        polynomial_in[global_addresss + 512 + divindex];
    sharedmemorys[idx_y + 24][idx_x] =
        polynomial_in[global_addresss + 768 + divindex];
    __syncthreads();

    int global_index1 = idx_x >> 4;
    int global_index2 = idx_x % 16;

    int t_ = 4;
    int t = 1 << t_;
    int in_shared_address = ((global_index2 >> t_) << t_) + global_index2;

    CooleyTukeyUnit_(
        sharedmemorys[(idx_y << 1) + global_index1][in_shared_address],
        sharedmemorys[(idx_y << 1) + global_index1][in_shared_address + t],
        n1_root_of_unity_table[(global_index2 >> t_)], q_thread);
    CooleyTukeyUnit_(
        sharedmemorys[(idx_y << 1) + global_index1 + 16][in_shared_address],
        sharedmemorys[(idx_y << 1) + global_index1 + 16][in_shared_address + t],
        n1_root_of_unity_table[(global_index2 >> t_)], q_thread);
    __syncthreads();

    for (int i = 0; i < 4; i++)
    {
        t = t >> 1;
        t_ -= 1;

        in_shared_address = ((global_index2 >> t_) << t_) + global_index2;

        CooleyTukeyUnit_(
            sharedmemorys[(idx_y << 1) + global_index1][in_shared_address],
            sharedmemorys[(idx_y << 1) + global_index1][in_shared_address + t],
            n1_root_of_unity_table[(global_index2 >> t_)], q_thread);
        CooleyTukeyUnit_(
            sharedmemorys[(idx_y << 1) + global_index1 + 16][in_shared_address],
            sharedmemorys[(idx_y << 1) + global_index1 + 16]
                         [in_shared_address + t],
            n1_root_of_unity_table[(global_index2 >> t_)], q_thread);
        __syncthreads();
    }
    __syncthreads();

    polynomial_out[idx_x + (idx_y << index1) + (block_x << 5) + divindex] =
        sharedmemorys[idx_x][idx_y];
    polynomial_out[idx_x + (idx_y << index1) + (block_x << 5) + index2 +
                   divindex] = sharedmemorys[idx_x][idx_y + 8];
    polynomial_out[idx_x + (idx_y << index1) + (block_x << 5) + (index2 * 2) +
                   divindex] = sharedmemorys[idx_x][idx_y + 16];
    polynomial_out[idx_x + (idx_y << index1) + (block_x << 5) + (index2 * 3) +
                   divindex] = sharedmemorys[idx_x][idx_y + 24];
}

__global__ void FourStepForwardCoreT2(Data* polynomial_in, Data* polynomial_out,
                                      Root* n1_root_of_unity_table,
                                      Modulus* modulus, int index1, int index2,
                                      int index3, int n_power, int mod_count)
{
    int idx_x = threadIdx.x;
    int idx_y = threadIdx.y;
    int block_x = blockIdx.x;
    int block_y = blockIdx.y;

    __shared__ Data sharedmemorys[16][64 + 1];

    int q_index = block_y % mod_count;
    Modulus q_thread = modulus[q_index];

    int idx_index = idx_x + (idx_y << 5);
    int global_addresss = idx_index + (block_x << 10);
    int divindex = block_y << n_power;

    int shr_index1 = idx_y >> 1;
    int shr_index2 = idx_y % 2;

    // Load data from global & store to shared
    sharedmemorys[shr_index1][idx_x + (shr_index2 << 5)] =
        polynomial_in[global_addresss + divindex];
    sharedmemorys[shr_index1 + 4][idx_x + (shr_index2 << 5)] =
        polynomial_in[global_addresss + 256 + divindex];
    sharedmemorys[shr_index1 + 8][idx_x + (shr_index2 << 5)] =
        polynomial_in[global_addresss + 512 + divindex];
    sharedmemorys[shr_index1 + 12][idx_x + (shr_index2 << 5)] =
        polynomial_in[global_addresss + 768 + divindex];
    __syncthreads();

    int t_ = 5;
    int t = 1 << t_;
    int in_shared_address = ((idx_x >> t_) << t_) + idx_x;

    CooleyTukeyUnit_(sharedmemorys[idx_y][in_shared_address],
                     sharedmemorys[idx_y][in_shared_address + t],
                     n1_root_of_unity_table[(idx_x >> t_)], q_thread);
    CooleyTukeyUnit_(sharedmemorys[idx_y + 8][in_shared_address],
                     sharedmemorys[idx_y + 8][in_shared_address + t],
                     n1_root_of_unity_table[(idx_x >> t_)], q_thread);
    __syncthreads();

    for (int i = 0; i < 5; i++)
    {
        t = t >> 1;
        t_ -= 1;

        in_shared_address = ((idx_x >> t_) << t_) + idx_x;
        ;

        CooleyTukeyUnit_(sharedmemorys[idx_y][in_shared_address],
                         sharedmemorys[idx_y][in_shared_address + t],
                         n1_root_of_unity_table[(idx_x >> t_)], q_thread);
        CooleyTukeyUnit_(sharedmemorys[idx_y + 8][in_shared_address],
                         sharedmemorys[idx_y + 8][in_shared_address + t],
                         n1_root_of_unity_table[(idx_x >> t_)], q_thread);
        __syncthreads();
    }
    __syncthreads();

    int global_index1 = idx_x >> 4;
    int global_index2 = idx_x % 16;

    polynomial_out[global_index2 + (global_index1 << index1) +
                   (idx_y << index2) + (block_x << 4) + divindex] =
        sharedmemorys[global_index2][global_index1 + (idx_y << 1)];
    polynomial_out[global_index2 + (global_index1 << index1) +
                   (idx_y << index2) + (block_x << 4) + index3 + divindex] =
        sharedmemorys[global_index2][global_index1 + (idx_y << 1) + 16];
    polynomial_out[global_index2 + (global_index1 << index1) +
                   (idx_y << index2) + (block_x << 4) + (index3 * 2) +
                   divindex] =
        sharedmemorys[global_index2][global_index1 + (idx_y << 1) + 32];
    polynomial_out[global_index2 + (global_index1 << index1) +
                   (idx_y << index2) + (block_x << 4) + (index3 * 3) +
                   divindex] =
        sharedmemorys[global_index2][global_index1 + (idx_y << 1) + 48];
}

__global__ void FourStepForwardCoreT2(Data* polynomial_in, Data* polynomial_out,
                                      Root* n1_root_of_unity_table,
                                      Modulus modulus, int index1, int index2,
                                      int index3, int n_power)
{
    int idx_x = threadIdx.x;
    int idx_y = threadIdx.y;
    int block_x = blockIdx.x;
    int block_y = blockIdx.y;

    __shared__ Data sharedmemorys[16][64 + 1];

    Modulus q_thread = modulus;

    int idx_index = idx_x + (idx_y << 5);
    int global_addresss = idx_index + (block_x << 10);
    int divindex = block_y << n_power;

    int shr_index1 = idx_y >> 1;
    int shr_index2 = idx_y % 2;

    // Load data from global & store to shared
    sharedmemorys[shr_index1][idx_x + (shr_index2 << 5)] =
        polynomial_in[global_addresss + divindex];
    sharedmemorys[shr_index1 + 4][idx_x + (shr_index2 << 5)] =
        polynomial_in[global_addresss + 256 + divindex];
    sharedmemorys[shr_index1 + 8][idx_x + (shr_index2 << 5)] =
        polynomial_in[global_addresss + 512 + divindex];
    sharedmemorys[shr_index1 + 12][idx_x + (shr_index2 << 5)] =
        polynomial_in[global_addresss + 768 + divindex];
    __syncthreads();

    int t_ = 5;
    int t = 1 << t_;
    int in_shared_address = ((idx_x >> t_) << t_) + idx_x;

    CooleyTukeyUnit_(sharedmemorys[idx_y][in_shared_address],
                     sharedmemorys[idx_y][in_shared_address + t],
                     n1_root_of_unity_table[(idx_x >> t_)], q_thread);
    CooleyTukeyUnit_(sharedmemorys[idx_y + 8][in_shared_address],
                     sharedmemorys[idx_y + 8][in_shared_address + t],
                     n1_root_of_unity_table[(idx_x >> t_)], q_thread);
    __syncthreads();

    for (int i = 0; i < 5; i++)
    {
        t = t >> 1;
        t_ -= 1;

        in_shared_address = ((idx_x >> t_) << t_) + idx_x;
        ;

        CooleyTukeyUnit_(sharedmemorys[idx_y][in_shared_address],
                         sharedmemorys[idx_y][in_shared_address + t],
                         n1_root_of_unity_table[(idx_x >> t_)], q_thread);
        CooleyTukeyUnit_(sharedmemorys[idx_y + 8][in_shared_address],
                         sharedmemorys[idx_y + 8][in_shared_address + t],
                         n1_root_of_unity_table[(idx_x >> t_)], q_thread);
        __syncthreads();
    }
    __syncthreads();

    int global_index1 = idx_x >> 4;
    int global_index2 = idx_x % 16;

    polynomial_out[global_index2 + (global_index1 << index1) +
                   (idx_y << index2) + (block_x << 4) + divindex] =
        sharedmemorys[global_index2][global_index1 + (idx_y << 1)];
    polynomial_out[global_index2 + (global_index1 << index1) +
                   (idx_y << index2) + (block_x << 4) + index3 + divindex] =
        sharedmemorys[global_index2][global_index1 + (idx_y << 1) + 16];
    polynomial_out[global_index2 + (global_index1 << index1) +
                   (idx_y << index2) + (block_x << 4) + (index3 * 2) +
                   divindex] =
        sharedmemorys[global_index2][global_index1 + (idx_y << 1) + 32];
    polynomial_out[global_index2 + (global_index1 << index1) +
                   (idx_y << index2) + (block_x << 4) + (index3 * 3) +
                   divindex] =
        sharedmemorys[global_index2][global_index1 + (idx_y << 1) + 48];
}

__global__ void FourStepForwardCoreT3(Data* polynomial_in, Data* polynomial_out,
                                      Root* n1_root_of_unity_table,
                                      Modulus* modulus, int index1, int index2,
                                      int index3, int n_power, int mod_count)
{
    int idx_x = threadIdx.x;
    int idx_y = threadIdx.y;
    int block_x = blockIdx.x;
    int block_y = blockIdx.y;

    __shared__ Data sharedmemorys[8][128 + 1];

    int q_index = block_y % mod_count;
    Modulus q_thread = modulus[q_index];

    int idx_index = idx_x + (idx_y << 5);
    int global_addresss = idx_index + (block_x << 10);
    int divindex = block_y << n_power;

    int shr_index1 = idx_y >> 2;
    int shr_index2 = idx_y % 4;

    // Load data from global & store to shared
    sharedmemorys[shr_index1][idx_x + (shr_index2 << 5)] =
        polynomial_in[global_addresss + divindex];
    sharedmemorys[shr_index1 + 2][idx_x + (shr_index2 << 5)] =
        polynomial_in[global_addresss + 256 + divindex];
    sharedmemorys[shr_index1 + 4][idx_x + (shr_index2 << 5)] =
        polynomial_in[global_addresss + 512 + divindex];
    sharedmemorys[shr_index1 + 6][idx_x + (shr_index2 << 5)] =
        polynomial_in[global_addresss + 768 + divindex];
    __syncthreads();

    int shr_address = idx_x + ((idx_y % 2) << 5);
    int shr_in = idx_y >> 1;

    int t_ = 6;
    int t = 1 << t_;
    int in_shared_address = ((shr_address >> t_) << t_) + shr_address;

    CooleyTukeyUnit_(sharedmemorys[shr_in][in_shared_address],
                     sharedmemorys[shr_in][in_shared_address + t],
                     n1_root_of_unity_table[(shr_address >> t_)], q_thread);
    CooleyTukeyUnit_(sharedmemorys[shr_in + 4][in_shared_address],
                     sharedmemorys[shr_in + 4][in_shared_address + t],
                     n1_root_of_unity_table[(shr_address >> t_)], q_thread);
    __syncthreads();

    for (int i = 0; i < 6; i++)
    {
        t = t >> 1;
        t_ -= 1;

        in_shared_address = ((shr_address >> t_) << t_) + shr_address;

        CooleyTukeyUnit_(sharedmemorys[shr_in][in_shared_address],
                         sharedmemorys[shr_in][in_shared_address + t],
                         n1_root_of_unity_table[(shr_address >> t_)], q_thread);
        CooleyTukeyUnit_(sharedmemorys[shr_in + 4][in_shared_address],
                         sharedmemorys[shr_in + 4][in_shared_address + t],
                         n1_root_of_unity_table[(shr_address >> t_)], q_thread);
        __syncthreads();
    }
    __syncthreads();

    int global_index1 = idx_x >> 3;
    int global_index2 = idx_x % 8;

    polynomial_out[global_index2 + (global_index1 << index1) +
                   (idx_y << index2) + (block_x << 3) + divindex] =
        sharedmemorys[global_index2][global_index1 + (idx_y << 2)];
    polynomial_out[global_index2 + (global_index1 << index1) +
                   (idx_y << index2) + (block_x << 3) + index3 + divindex] =
        sharedmemorys[global_index2][global_index1 + (idx_y << 2) + 32];
    polynomial_out[global_index2 + (global_index1 << index1) +
                   (idx_y << index2) + (block_x << 3) + (index3 * 2) +
                   divindex] =
        sharedmemorys[global_index2][global_index1 + (idx_y << 2) + 64];
    polynomial_out[global_index2 + (global_index1 << index1) +
                   (idx_y << index2) + (block_x << 3) + (index3 * 3) +
                   divindex] =
        sharedmemorys[global_index2][global_index1 + (idx_y << 2) + 96];
}

__global__ void FourStepForwardCoreT3(Data* polynomial_in, Data* polynomial_out,
                                      Root* n1_root_of_unity_table,
                                      Modulus modulus, int index1, int index2,
                                      int index3, int n_power)
{
    int idx_x = threadIdx.x;
    int idx_y = threadIdx.y;
    int block_x = blockIdx.x;
    int block_y = blockIdx.y;

    __shared__ Data sharedmemorys[8][128 + 1];

    Modulus q_thread = modulus;

    int idx_index = idx_x + (idx_y << 5);
    int global_addresss = idx_index + (block_x << 10);
    int divindex = block_y << n_power;

    int shr_index1 = idx_y >> 2;
    int shr_index2 = idx_y % 4;

    // Load data from global & store to shared
    sharedmemorys[shr_index1][idx_x + (shr_index2 << 5)] =
        polynomial_in[global_addresss + divindex];
    sharedmemorys[shr_index1 + 2][idx_x + (shr_index2 << 5)] =
        polynomial_in[global_addresss + 256 + divindex];
    sharedmemorys[shr_index1 + 4][idx_x + (shr_index2 << 5)] =
        polynomial_in[global_addresss + 512 + divindex];
    sharedmemorys[shr_index1 + 6][idx_x + (shr_index2 << 5)] =
        polynomial_in[global_addresss + 768 + divindex];
    __syncthreads();

    int shr_address = idx_x + ((idx_y % 2) << 5);
    int shr_in = idx_y >> 1;

    int t_ = 6;
    int t = 1 << t_;
    int in_shared_address = ((shr_address >> t_) << t_) + shr_address;

    CooleyTukeyUnit_(sharedmemorys[shr_in][in_shared_address],
                     sharedmemorys[shr_in][in_shared_address + t],
                     n1_root_of_unity_table[(shr_address >> t_)], q_thread);
    CooleyTukeyUnit_(sharedmemorys[shr_in + 4][in_shared_address],
                     sharedmemorys[shr_in + 4][in_shared_address + t],
                     n1_root_of_unity_table[(shr_address >> t_)], q_thread);
    __syncthreads();

    for (int i = 0; i < 6; i++)
    {
        t = t >> 1;
        t_ -= 1;

        in_shared_address = ((shr_address >> t_) << t_) + shr_address;

        CooleyTukeyUnit_(sharedmemorys[shr_in][in_shared_address],
                         sharedmemorys[shr_in][in_shared_address + t],
                         n1_root_of_unity_table[(shr_address >> t_)], q_thread);
        CooleyTukeyUnit_(sharedmemorys[shr_in + 4][in_shared_address],
                         sharedmemorys[shr_in + 4][in_shared_address + t],
                         n1_root_of_unity_table[(shr_address >> t_)], q_thread);
        __syncthreads();
    }
    __syncthreads();

    int global_index1 = idx_x >> 3;
    int global_index2 = idx_x % 8;

    polynomial_out[global_index2 + (global_index1 << index1) +
                   (idx_y << index2) + (block_x << 3) + divindex] =
        sharedmemorys[global_index2][global_index1 + (idx_y << 2)];
    polynomial_out[global_index2 + (global_index1 << index1) +
                   (idx_y << index2) + (block_x << 3) + index3 + divindex] =
        sharedmemorys[global_index2][global_index1 + (idx_y << 2) + 32];
    polynomial_out[global_index2 + (global_index1 << index1) +
                   (idx_y << index2) + (block_x << 3) + (index3 * 2) +
                   divindex] =
        sharedmemorys[global_index2][global_index1 + (idx_y << 2) + 64];
    polynomial_out[global_index2 + (global_index1 << index1) +
                   (idx_y << index2) + (block_x << 3) + (index3 * 3) +
                   divindex] =
        sharedmemorys[global_index2][global_index1 + (idx_y << 2) + 96];
}

__global__ void FourStepForwardCoreT4(Data* polynomial_in, Data* polynomial_out,
                                      Root* n1_root_of_unity_table,
                                      Modulus* modulus, int index1, int index2,
                                      int index3, int n_power, int mod_count)
{
    int idx_x = threadIdx.x;
    int idx_y = threadIdx.y;
    int block_x = blockIdx.x;
    int block_y = blockIdx.y;

    __shared__ Data sharedmemorys[4][256 + 1];

    int q_index = block_y % mod_count;
    Modulus q_thread = modulus[q_index];

    int idx_index = idx_x + (idx_y << 5);
    int global_addresss = idx_index + (block_x << 10);
    int divindex = block_y << n_power;

    int shr_index1 = idx_y >> 3;
    int shr_index2 = idx_y % 8;

    // Load data from global & store to shared
    sharedmemorys[shr_index1][idx_x + (shr_index2 << 5)] =
        polynomial_in[global_addresss + divindex];
    sharedmemorys[shr_index1 + 1][idx_x + (shr_index2 << 5)] =
        polynomial_in[global_addresss + 256 + divindex];
    sharedmemorys[shr_index1 + 2][idx_x + (shr_index2 << 5)] =
        polynomial_in[global_addresss + 512 + divindex];
    sharedmemorys[shr_index1 + 3][idx_x + (shr_index2 << 5)] =
        polynomial_in[global_addresss + 768 + divindex];
    __syncthreads();

    int shr_address = idx_x + ((idx_y % 4) << 5);
    int shr_in = idx_y >> 2;

    int t_ = 7;
    int t = 1 << t_;
    int in_shared_address = ((shr_address >> t_) << t_) + shr_address;

    CooleyTukeyUnit_(sharedmemorys[shr_in][in_shared_address],
                     sharedmemorys[shr_in][in_shared_address + t],
                     n1_root_of_unity_table[(shr_address >> t_)], q_thread);
    CooleyTukeyUnit_(sharedmemorys[shr_in + 2][in_shared_address],
                     sharedmemorys[shr_in + 2][in_shared_address + t],
                     n1_root_of_unity_table[(shr_address >> t_)], q_thread);
    __syncthreads();

    for (int i = 0; i < 7; i++)
    {
        t = t >> 1;
        t_ -= 1;

        in_shared_address = ((shr_address >> t_) << t_) + shr_address;

        CooleyTukeyUnit_(sharedmemorys[shr_in][in_shared_address],
                         sharedmemorys[shr_in][in_shared_address + t],
                         n1_root_of_unity_table[(shr_address >> t_)], q_thread);
        CooleyTukeyUnit_(sharedmemorys[shr_in + 2][in_shared_address],
                         sharedmemorys[shr_in + 2][in_shared_address + t],
                         n1_root_of_unity_table[(shr_address >> t_)], q_thread);
        __syncthreads();
    }
    __syncthreads();

    int global_index1 = idx_x >> 2;
    int global_index2 = idx_x % 4;

    polynomial_out[global_index2 + (global_index1 << index1) +
                   (idx_y << index2) + (block_x << 2) + divindex] =
        sharedmemorys[global_index2][global_index1 + (idx_y << 3)];
    polynomial_out[global_index2 + (global_index1 << index1) +
                   (idx_y << index2) + (block_x << 2) + index3 + divindex] =
        sharedmemorys[global_index2][global_index1 + (idx_y << 3) + 64];
    polynomial_out[global_index2 + (global_index1 << index1) +
                   (idx_y << index2) + (block_x << 2) + (index3 * 2) +
                   divindex] =
        sharedmemorys[global_index2][global_index1 + (idx_y << 3) + 128];
    polynomial_out[global_index2 + (global_index1 << index1) +
                   (idx_y << index2) + (block_x << 2) + (index3 * 3) +
                   divindex] =
        sharedmemorys[global_index2][global_index1 + (idx_y << 3) + 192];
}

__global__ void FourStepForwardCoreT4(Data* polynomial_in, Data* polynomial_out,
                                      Root* n1_root_of_unity_table,
                                      Modulus modulus, int index1, int index2,
                                      int index3, int n_power)
{
    int idx_x = threadIdx.x;
    int idx_y = threadIdx.y;
    int block_x = blockIdx.x;
    int block_y = blockIdx.y;

    __shared__ Data sharedmemorys[4][256 + 1];

    Modulus q_thread = modulus;

    int idx_index = idx_x + (idx_y << 5);
    int global_addresss = idx_index + (block_x << 10);
    int divindex = block_y << n_power;

    int shr_index1 = idx_y >> 3;
    int shr_index2 = idx_y % 8;

    // Load data from global & store to shared
    sharedmemorys[shr_index1][idx_x + (shr_index2 << 5)] =
        polynomial_in[global_addresss + divindex];
    sharedmemorys[shr_index1 + 1][idx_x + (shr_index2 << 5)] =
        polynomial_in[global_addresss + 256 + divindex];
    sharedmemorys[shr_index1 + 2][idx_x + (shr_index2 << 5)] =
        polynomial_in[global_addresss + 512 + divindex];
    sharedmemorys[shr_index1 + 3][idx_x + (shr_index2 << 5)] =
        polynomial_in[global_addresss + 768 + divindex];
    __syncthreads();

    int shr_address = idx_x + ((idx_y % 4) << 5);
    int shr_in = idx_y >> 2;

    int t_ = 7;
    int t = 1 << t_;
    int in_shared_address = ((shr_address >> t_) << t_) + shr_address;

    CooleyTukeyUnit_(sharedmemorys[shr_in][in_shared_address],
                     sharedmemorys[shr_in][in_shared_address + t],
                     n1_root_of_unity_table[(shr_address >> t_)], q_thread);
    CooleyTukeyUnit_(sharedmemorys[shr_in + 2][in_shared_address],
                     sharedmemorys[shr_in + 2][in_shared_address + t],
                     n1_root_of_unity_table[(shr_address >> t_)], q_thread);
    __syncthreads();

    for (int i = 0; i < 7; i++)
    {
        t = t >> 1;
        t_ -= 1;

        in_shared_address = ((shr_address >> t_) << t_) + shr_address;

        CooleyTukeyUnit_(sharedmemorys[shr_in][in_shared_address],
                         sharedmemorys[shr_in][in_shared_address + t],
                         n1_root_of_unity_table[(shr_address >> t_)], q_thread);
        CooleyTukeyUnit_(sharedmemorys[shr_in + 2][in_shared_address],
                         sharedmemorys[shr_in + 2][in_shared_address + t],
                         n1_root_of_unity_table[(shr_address >> t_)], q_thread);
        __syncthreads();
    }
    __syncthreads();

    int global_index1 = idx_x >> 2;
    int global_index2 = idx_x % 4;

    polynomial_out[global_index2 + (global_index1 << index1) +
                   (idx_y << index2) + (block_x << 2) + divindex] =
        sharedmemorys[global_index2][global_index1 + (idx_y << 3)];
    polynomial_out[global_index2 + (global_index1 << index1) +
                   (idx_y << index2) + (block_x << 2) + index3 + divindex] =
        sharedmemorys[global_index2][global_index1 + (idx_y << 3) + 64];
    polynomial_out[global_index2 + (global_index1 << index1) +
                   (idx_y << index2) + (block_x << 2) + (index3 * 2) +
                   divindex] =
        sharedmemorys[global_index2][global_index1 + (idx_y << 3) + 128];
    polynomial_out[global_index2 + (global_index1 << index1) +
                   (idx_y << index2) + (block_x << 2) + (index3 * 3) +
                   divindex] =
        sharedmemorys[global_index2][global_index1 + (idx_y << 3) + 192];
}

__global__ void FourStepPartialForwardCore1(Data* polynomial_in,
                                            Root* n2_root_of_unity_table,
                                            Root* w_root_of_unity_table,
                                            Modulus* modulus, int small_npower,
                                            int loc1, int loc2, int loop,
                                            int n_power, int mod_count)
{
    int idx_x = threadIdx.x;
    int idx_y = threadIdx.y;
    int block_x = blockIdx.x;
    int block_y = blockIdx.y;
    int block_z = blockIdx.z;

    __shared__ Data sharedmemorys[512];

    int n_power__ = small_npower;
    int t_2 = n_power__ - 1;

    int q_index = block_z % mod_count;
    Modulus q_thread = modulus[q_index];

    int grid = (block_y << n_power__);
    int divindex = block_z << n_power;

    int global_addresss = (idx_y << 9) + idx_x + (block_x << loc1);
    int shared_addresss = (idx_x + (idx_y << loc1));

    int load_store_address = global_addresss + grid;

    Data mult_1 = polynomial_in[load_store_address + divindex];
    Data mult_2 = polynomial_in[load_store_address + loc2 + divindex];

    mult_1 = VALUE_GPU::mult(mult_1, w_root_of_unity_table[load_store_address],
                             q_thread);
    mult_2 = VALUE_GPU::mult(
        mult_2, w_root_of_unity_table[load_store_address + loc2], q_thread);

    sharedmemorys[shared_addresss] = mult_1;
    sharedmemorys[shared_addresss + 256] = mult_2;

    int t_ = 8;
    int t = 1 << t_;
    int in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;

    for (int lp = 0; lp < loop; lp++)
    {
        CooleyTukeyUnit_(sharedmemorys[in_shared_address],
                         sharedmemorys[in_shared_address + t],
                         n2_root_of_unity_table[(global_addresss >> t_2)],
                         q_thread);
        __syncthreads();

        t = t >> 1;
        t_2 -= 1;
        t_ -= 1;

        in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
    }
    __syncthreads();

    // Load data from shared & store to global
    polynomial_in[load_store_address + divindex] =
        sharedmemorys[shared_addresss];
    polynomial_in[load_store_address + loc2 + divindex] =
        sharedmemorys[shared_addresss + 256];
}

__global__ void FourStepPartialForwardCore1(Data* polynomial_in,
                                            Root* n2_root_of_unity_table,
                                            Root* w_root_of_unity_table,
                                            Modulus modulus, int small_npower,
                                            int loc1, int loc2, int loop,
                                            int n_power)
{
    int idx_x = threadIdx.x;
    int idx_y = threadIdx.y;
    int block_x = blockIdx.x;
    int block_y = blockIdx.y;
    int block_z = blockIdx.z;

    __shared__ Data sharedmemorys[512];

    int n_power__ = small_npower;
    int t_2 = n_power__ - 1;

    Modulus q_thread = modulus;

    int grid = (block_y << n_power__);
    int divindex = block_z << n_power;

    int global_addresss = (idx_y << 9) + idx_x + (block_x << loc1);
    int shared_addresss = (idx_x + (idx_y << loc1));

    int load_store_address = global_addresss + grid;

    Data mult_1 = polynomial_in[load_store_address + divindex];
    Data mult_2 = polynomial_in[load_store_address + loc2 + divindex];

    mult_1 = VALUE_GPU::mult(mult_1, w_root_of_unity_table[load_store_address],
                             q_thread);
    mult_2 = VALUE_GPU::mult(
        mult_2, w_root_of_unity_table[load_store_address + loc2], q_thread);

    sharedmemorys[shared_addresss] = mult_1;
    sharedmemorys[shared_addresss + 256] = mult_2;

    int t_ = 8;
    int t = 1 << t_;
    int in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;

    for (int lp = 0; lp < loop; lp++)
    {
        CooleyTukeyUnit_(sharedmemorys[in_shared_address],
                         sharedmemorys[in_shared_address + t],
                         n2_root_of_unity_table[(global_addresss >> t_2)],
                         q_thread);
        __syncthreads();

        t = t >> 1;
        t_2 -= 1;
        t_ -= 1;

        in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
    }
    __syncthreads();

    // Load data from shared & store to global
    polynomial_in[load_store_address + divindex] =
        sharedmemorys[shared_addresss];
    polynomial_in[load_store_address + loc2 + divindex] =
        sharedmemorys[shared_addresss + 256];
}

__global__ void FourStepPartialForwardCore2(Data* polynomial_in,
                                            Root* n2_root_of_unity_table,
                                            Modulus* modulus, int small_npower,
                                            int n_power, int mod_count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int block_y = blockIdx.y;
    int block_z = blockIdx.z;

    int local_idx = threadIdx.x;

    __shared__ Data sharedmemorys[512];

    int n_power__ = small_npower;
    int t_2 = 8;
    int t = 1 << t_2;

    int dividx = (block_y << n_power__);
    int divindex = block_z << n_power;

    int q_index = block_z % mod_count;
    Modulus q_thread = modulus[q_index];

    int address = dividx + ((idx >> t_2) << t_2) + idx;

    int shrd_dixidx_t = (local_idx >> t_2) << t_2;
    int shrd_address = shrd_dixidx_t + local_idx;

    sharedmemorys[shrd_address] = polynomial_in[address + divindex];
    sharedmemorys[shrd_address + t] = polynomial_in[address + t + divindex];

#pragma unroll
    for (int loop_dep = 0; loop_dep < 3; loop_dep++)
    {
        CooleyTukeyUnit_(sharedmemorys[shrd_address],
                         sharedmemorys[shrd_address + t],
                         n2_root_of_unity_table[(idx >> t_2)], q_thread);

        t = t >> 1;
        t_2 -= 1;

        shrd_dixidx_t = (local_idx >> t_2) << t_2;
        shrd_address = shrd_dixidx_t + local_idx;
        __syncthreads();
    }

#pragma unroll
    for (int loop_indep = 0; loop_indep < 5; loop_indep++)
    {
        CooleyTukeyUnit_(sharedmemorys[shrd_address],
                         sharedmemorys[shrd_address + t],
                         n2_root_of_unity_table[(idx >> t_2)], q_thread);

        t = t >> 1;
        t_2 -= 1;

        shrd_dixidx_t = (local_idx >> t_2) << t_2;
        shrd_address = shrd_dixidx_t + local_idx;
    }

    address = dividx + ((idx >> t_2) << t_2) + idx;

    CooleyTukeyUnit_(sharedmemorys[shrd_address],
                     sharedmemorys[shrd_address + t],
                     n2_root_of_unity_table[(idx >> t_2)], q_thread);
    polynomial_in[address + divindex] = sharedmemorys[shrd_address];
    polynomial_in[address + t + divindex] = sharedmemorys[shrd_address + t];
}

__global__ void FourStepPartialForwardCore2(Data* polynomial_in,
                                            Root* n2_root_of_unity_table,
                                            Modulus modulus, int small_npower,
                                            int n_power)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int block_y = blockIdx.y;
    int block_z = blockIdx.z;

    int local_idx = threadIdx.x;

    __shared__ Data sharedmemorys[512];

    int n_power__ = small_npower;
    int t_2 = 8;
    int t = 1 << t_2;

    int dividx = (block_y << n_power__);
    int divindex = block_z << n_power;

    Modulus q_thread = modulus;

    int address = dividx + ((idx >> t_2) << t_2) + idx;

    int shrd_dixidx_t = (local_idx >> t_2) << t_2;
    int shrd_address = shrd_dixidx_t + local_idx;

    sharedmemorys[shrd_address] = polynomial_in[address + divindex];
    sharedmemorys[shrd_address + t] = polynomial_in[address + t + divindex];

#pragma unroll
    for (int loop_dep = 0; loop_dep < 3; loop_dep++)
    {
        CooleyTukeyUnit_(sharedmemorys[shrd_address],
                         sharedmemorys[shrd_address + t],
                         n2_root_of_unity_table[(idx >> t_2)], q_thread);

        t = t >> 1;
        t_2 -= 1;

        shrd_dixidx_t = (local_idx >> t_2) << t_2;
        shrd_address = shrd_dixidx_t + local_idx;
        __syncthreads();
    }

#pragma unroll
    for (int loop_indep = 0; loop_indep < 5; loop_indep++)
    {
        CooleyTukeyUnit_(sharedmemorys[shrd_address],
                         sharedmemorys[shrd_address + t],
                         n2_root_of_unity_table[(idx >> t_2)], q_thread);

        t = t >> 1;
        t_2 -= 1;

        shrd_dixidx_t = (local_idx >> t_2) << t_2;
        shrd_address = shrd_dixidx_t + local_idx;
    }

    address = dividx + ((idx >> t_2) << t_2) + idx;

    CooleyTukeyUnit_(sharedmemorys[shrd_address],
                     sharedmemorys[shrd_address + t],
                     n2_root_of_unity_table[(idx >> t_2)], q_thread);
    polynomial_in[address + divindex] = sharedmemorys[shrd_address];
    polynomial_in[address + t + divindex] = sharedmemorys[shrd_address + t];
}

__global__ void FourStepPartialForwardCore(Data* polynomial_in,
                                           Root* n2_root_of_unity_table,
                                           Root* w_root_of_unity_table,
                                           Modulus* modulus, int small_npower,
                                           int T, int LOOP, int n_power,
                                           int mod_count)
{
    int local_idx = threadIdx.x;
    int block_x = blockIdx.x;
    int block_y = blockIdx.y;

    __shared__ Data sharedmemorys[512];

    int n_power__ = small_npower;
    int t_2 = T;
    int t = 1 << t_2;

    int dividx = (block_x << n_power__);
    int divindex = block_y << n_power;

    int q_index = block_y % mod_count;
    Modulus q_thread = modulus[q_index];

    int address = dividx + ((local_idx >> t_2) << t_2) + local_idx;

    int shrd_dixidx_t = (local_idx >> t_2) << t_2;
    int shrd_address = shrd_dixidx_t + local_idx;

    Data mult_1 = polynomial_in[address + divindex];
    Data mult_2 = polynomial_in[address + t + divindex];

    mult_1 = VALUE_GPU::mult(mult_1, w_root_of_unity_table[address], q_thread);
    mult_2 =
        VALUE_GPU::mult(mult_2, w_root_of_unity_table[address + t], q_thread);

    sharedmemorys[shrd_address] = mult_1;
    sharedmemorys[shrd_address + t] = mult_2;

#pragma unroll
    for (int loop_dep = 0; loop_dep < LOOP; loop_dep++)
    {
        CooleyTukeyUnit_(sharedmemorys[shrd_address],
                         sharedmemorys[shrd_address + t],
                         n2_root_of_unity_table[(local_idx >> t_2)], q_thread);

        t = t >> 1;
        t_2 -= 1;

        shrd_dixidx_t = (local_idx >> t_2) << t_2;
        shrd_address = shrd_dixidx_t + local_idx;
        __syncthreads();
    }

#pragma unroll
    for (int loop_indep = 0; loop_indep < 5; loop_indep++)
    {
        CooleyTukeyUnit_(sharedmemorys[shrd_address],
                         sharedmemorys[shrd_address + t],
                         n2_root_of_unity_table[(local_idx >> t_2)], q_thread);

        t = t >> 1;
        t_2 -= 1;
        shrd_dixidx_t = (local_idx >> t_2) << t_2;
        shrd_address = shrd_dixidx_t + local_idx;
    }

    address = dividx + ((local_idx >> t_2) << t_2) + local_idx;

    CooleyTukeyUnit_(sharedmemorys[shrd_address],
                     sharedmemorys[shrd_address + t],
                     n2_root_of_unity_table[(local_idx >> t_2)], q_thread);
    polynomial_in[address + divindex] = sharedmemorys[shrd_address];
    polynomial_in[address + t + divindex] = sharedmemorys[shrd_address + t];
}

__global__ void FourStepPartialForwardCore(Data* polynomial_in,
                                           Root* n2_root_of_unity_table,
                                           Root* w_root_of_unity_table,
                                           Modulus modulus, int small_npower,
                                           int T, int LOOP, int n_power)
{
    int local_idx = threadIdx.x;
    int block_x = blockIdx.x;
    int block_y = blockIdx.y;

    __shared__ Data sharedmemorys[512];

    int n_power__ = small_npower;
    int t_2 = T;
    int t = 1 << t_2;

    int dividx = (block_x << n_power__);
    int divindex = block_y << n_power;

    Modulus q_thread = modulus;

    int address = dividx + ((local_idx >> t_2) << t_2) + local_idx;

    int shrd_dixidx_t = (local_idx >> t_2) << t_2;
    int shrd_address = shrd_dixidx_t + local_idx;

    Data mult_1 = polynomial_in[address + divindex];
    Data mult_2 = polynomial_in[address + t + divindex];

    mult_1 = VALUE_GPU::mult(mult_1, w_root_of_unity_table[address], q_thread);
    mult_2 =
        VALUE_GPU::mult(mult_2, w_root_of_unity_table[address + t], q_thread);

    sharedmemorys[shrd_address] = mult_1;
    sharedmemorys[shrd_address + t] = mult_2;

#pragma unroll
    for (int loop_dep = 0; loop_dep < LOOP; loop_dep++)
    {
        CooleyTukeyUnit_(sharedmemorys[shrd_address],
                         sharedmemorys[shrd_address + t],
                         n2_root_of_unity_table[(local_idx >> t_2)], q_thread);

        t = t >> 1;
        t_2 -= 1;

        shrd_dixidx_t = (local_idx >> t_2) << t_2;
        shrd_address = shrd_dixidx_t + local_idx;
        __syncthreads();
    }

#pragma unroll
    for (int loop_indep = 0; loop_indep < 5; loop_indep++)
    {
        CooleyTukeyUnit_(sharedmemorys[shrd_address],
                         sharedmemorys[shrd_address + t],
                         n2_root_of_unity_table[(local_idx >> t_2)], q_thread);

        t = t >> 1;
        t_2 -= 1;
        shrd_dixidx_t = (local_idx >> t_2) << t_2;
        shrd_address = shrd_dixidx_t + local_idx;
    }

    address = dividx + ((local_idx >> t_2) << t_2) + local_idx;

    CooleyTukeyUnit_(sharedmemorys[shrd_address],
                     sharedmemorys[shrd_address + t],
                     n2_root_of_unity_table[(local_idx >> t_2)], q_thread);
    polynomial_in[address + divindex] = sharedmemorys[shrd_address];
    polynomial_in[address + t + divindex] = sharedmemorys[shrd_address + t];
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// INTT PART

__global__ void FourStepInverseCoreT1(Data* polynomial_in, Data* polynomial_out,
                                      Root* n1_root_of_unity_table,
                                      Modulus* modulus, int index1, int index2,
                                      int n_power, int mod_count)
{
    int idx_x = threadIdx.x;
    int idx_y = threadIdx.y;
    int block_x = blockIdx.x;
    int block_y = blockIdx.y;

    __shared__ Data sharedmemorys[32][32 + 1];

    int divindex = block_y << n_power;

    int q_index = block_y % mod_count;
    Modulus q_thread = modulus[q_index];

    int idx_index = idx_x + (idx_y << 5);
    int global_addresss = idx_index + (block_x << 10);

    // Load data from global & store to shared
    sharedmemorys[idx_y][idx_x] = polynomial_in[global_addresss + divindex];
    sharedmemorys[idx_y + 8][idx_x] =
        polynomial_in[global_addresss + 256 + divindex];
    sharedmemorys[idx_y + 16][idx_x] =
        polynomial_in[global_addresss + 512 + divindex];
    sharedmemorys[idx_y + 24][idx_x] =
        polynomial_in[global_addresss + 768 + divindex];
    __syncthreads();

    int global_index1 = idx_x >> 4;
    int global_index2 = idx_x % 16;

    int t_ = 0;
    int t = 1 << t_;
    int in_shared_address = ((global_index2 >> t_) << t_) + global_index2;

    GentlemanSandeUnit_(
        sharedmemorys[(idx_y << 1) + global_index1][in_shared_address],
        sharedmemorys[(idx_y << 1) + global_index1][in_shared_address + t],
        n1_root_of_unity_table[(global_index2 >> t_)], q_thread);
    GentlemanSandeUnit_(
        sharedmemorys[(idx_y << 1) + global_index1 + 16][in_shared_address],
        sharedmemorys[(idx_y << 1) + global_index1 + 16][in_shared_address + t],
        n1_root_of_unity_table[(global_index2 >> t_)], q_thread);
    __syncthreads();

    for (int i = 0; i < 4; i++)
    {
        t = t << 1;
        t_ += 1;

        in_shared_address = ((global_index2 >> t_) << t_) + global_index2;
        ;

        GentlemanSandeUnit_(
            sharedmemorys[(idx_y << 1) + global_index1][in_shared_address],
            sharedmemorys[(idx_y << 1) + global_index1][in_shared_address + t],
            n1_root_of_unity_table[(global_index2 >> t_)], q_thread);
        GentlemanSandeUnit_(
            sharedmemorys[(idx_y << 1) + global_index1 + 16][in_shared_address],
            sharedmemorys[(idx_y << 1) + global_index1 + 16]
                         [in_shared_address + t],
            n1_root_of_unity_table[(global_index2 >> t_)], q_thread);
        __syncthreads();
    }
    __syncthreads();

    polynomial_out[idx_x + (idx_y << index1) + (block_x << 5) + divindex] =
        sharedmemorys[idx_x][idx_y];
    polynomial_out[idx_x + (idx_y << index1) + (block_x << 5) + index2 +
                   divindex] = sharedmemorys[idx_x][idx_y + 8];
    polynomial_out[idx_x + (idx_y << index1) + (block_x << 5) + (index2 * 2) +
                   divindex] = sharedmemorys[idx_x][idx_y + 16];
    polynomial_out[idx_x + (idx_y << index1) + (block_x << 5) + (index2 * 3) +
                   divindex] = sharedmemorys[idx_x][idx_y + 24];
}

__global__ void FourStepInverseCoreT1(Data* polynomial_in, Data* polynomial_out,
                                      Root* n1_root_of_unity_table,
                                      Modulus modulus, int index1, int index2,
                                      int n_power)
{
    int idx_x = threadIdx.x;
    int idx_y = threadIdx.y;
    int block_x = blockIdx.x;
    int block_y = blockIdx.y;

    __shared__ Data sharedmemorys[32][32 + 1];

    int divindex = block_y << n_power;

    Modulus q_thread = modulus;

    int idx_index = idx_x + (idx_y << 5);
    int global_addresss = idx_index + (block_x << 10);

    // Load data from global & store to shared
    sharedmemorys[idx_y][idx_x] = polynomial_in[global_addresss + divindex];
    sharedmemorys[idx_y + 8][idx_x] =
        polynomial_in[global_addresss + 256 + divindex];
    sharedmemorys[idx_y + 16][idx_x] =
        polynomial_in[global_addresss + 512 + divindex];
    sharedmemorys[idx_y + 24][idx_x] =
        polynomial_in[global_addresss + 768 + divindex];
    __syncthreads();

    int global_index1 = idx_x >> 4;
    int global_index2 = idx_x % 16;

    int t_ = 0;
    int t = 1 << t_;
    int in_shared_address = ((global_index2 >> t_) << t_) + global_index2;

    GentlemanSandeUnit_(
        sharedmemorys[(idx_y << 1) + global_index1][in_shared_address],
        sharedmemorys[(idx_y << 1) + global_index1][in_shared_address + t],
        n1_root_of_unity_table[(global_index2 >> t_)], q_thread);
    GentlemanSandeUnit_(
        sharedmemorys[(idx_y << 1) + global_index1 + 16][in_shared_address],
        sharedmemorys[(idx_y << 1) + global_index1 + 16][in_shared_address + t],
        n1_root_of_unity_table[(global_index2 >> t_)], q_thread);
    __syncthreads();

    for (int i = 0; i < 4; i++)
    {
        t = t << 1;
        t_ += 1;

        in_shared_address = ((global_index2 >> t_) << t_) + global_index2;
        ;

        GentlemanSandeUnit_(
            sharedmemorys[(idx_y << 1) + global_index1][in_shared_address],
            sharedmemorys[(idx_y << 1) + global_index1][in_shared_address + t],
            n1_root_of_unity_table[(global_index2 >> t_)], q_thread);
        GentlemanSandeUnit_(
            sharedmemorys[(idx_y << 1) + global_index1 + 16][in_shared_address],
            sharedmemorys[(idx_y << 1) + global_index1 + 16]
                         [in_shared_address + t],
            n1_root_of_unity_table[(global_index2 >> t_)], q_thread);
        __syncthreads();
    }
    __syncthreads();

    polynomial_out[idx_x + (idx_y << index1) + (block_x << 5) + divindex] =
        sharedmemorys[idx_x][idx_y];
    polynomial_out[idx_x + (idx_y << index1) + (block_x << 5) + index2 +
                   divindex] = sharedmemorys[idx_x][idx_y + 8];
    polynomial_out[idx_x + (idx_y << index1) + (block_x << 5) + (index2 * 2) +
                   divindex] = sharedmemorys[idx_x][idx_y + 16];
    polynomial_out[idx_x + (idx_y << index1) + (block_x << 5) + (index2 * 3) +
                   divindex] = sharedmemorys[idx_x][idx_y + 24];
}

__global__ void FourStepInverseCoreT2(Data* polynomial_in, Data* polynomial_out,
                                      Root* n1_root_of_unity_table,
                                      Modulus* modulus, int index1, int index2,
                                      int index3, int n_power, int mod_count)
{
    int idx_x = threadIdx.x;
    int idx_y = threadIdx.y;
    int block_x = blockIdx.x;
    int block_y = blockIdx.y;

    __shared__ Data sharedmemorys[16][64 + 1];

    int divindex = block_y << n_power;

    int q_index = block_y % mod_count;
    Modulus q_thread = modulus[q_index];

    int idx_index = idx_x + (idx_y << 5);
    int global_addresss = idx_index + (block_x << 10);

    int shr_index1 = idx_y >> 1;
    int shr_index2 = idx_y % 2;

    // Load data from global & store to shared
    sharedmemorys[shr_index1][idx_x + (shr_index2 << 5)] =
        polynomial_in[global_addresss + divindex];
    sharedmemorys[shr_index1 + 4][idx_x + (shr_index2 << 5)] =
        polynomial_in[global_addresss + 256 + divindex];
    sharedmemorys[shr_index1 + 8][idx_x + (shr_index2 << 5)] =
        polynomial_in[global_addresss + 512 + divindex];
    sharedmemorys[shr_index1 + 12][idx_x + (shr_index2 << 5)] =
        polynomial_in[global_addresss + 768 + divindex];
    __syncthreads();

    int t_ = 0;
    int t = 1 << t_;
    int in_shared_address = ((idx_x >> t_) << t_) + idx_x;

    GentlemanSandeUnit_(sharedmemorys[idx_y][in_shared_address],
                        sharedmemorys[idx_y][in_shared_address + t],
                        n1_root_of_unity_table[(idx_x >> t_)], q_thread);
    GentlemanSandeUnit_(sharedmemorys[idx_y + 8][in_shared_address],
                        sharedmemorys[idx_y + 8][in_shared_address + t],
                        n1_root_of_unity_table[(idx_x >> t_)], q_thread);
    __syncthreads();

    for (int i = 0; i < 5; i++)
    {
        t = t << 1;
        t_ += 1;

        in_shared_address = ((idx_x >> t_) << t_) + idx_x;

        GentlemanSandeUnit_(sharedmemorys[idx_y][in_shared_address],
                            sharedmemorys[idx_y][in_shared_address + t],
                            n1_root_of_unity_table[(idx_x >> t_)], q_thread);
        GentlemanSandeUnit_(sharedmemorys[idx_y + 8][in_shared_address],
                            sharedmemorys[idx_y + 8][in_shared_address + t],
                            n1_root_of_unity_table[(idx_x >> t_)], q_thread);
        __syncthreads();
    }
    __syncthreads();

    int global_index1 = idx_x >> 4;
    int global_index2 = idx_x % 16;

    polynomial_out[global_index2 + (global_index1 << index1) +
                   (idx_y << index2) + (block_x << 4) + divindex] =
        sharedmemorys[global_index2][global_index1 + (idx_y << 1)];
    polynomial_out[global_index2 + (global_index1 << index1) +
                   (idx_y << index2) + (block_x << 4) + index3 + divindex] =
        sharedmemorys[global_index2][global_index1 + (idx_y << 1) + 16];
    polynomial_out[global_index2 + (global_index1 << index1) +
                   (idx_y << index2) + (block_x << 4) + (index3 * 2) +
                   divindex] =
        sharedmemorys[global_index2][global_index1 + (idx_y << 1) + 32];
    polynomial_out[global_index2 + (global_index1 << index1) +
                   (idx_y << index2) + (block_x << 4) + (index3 * 3) +
                   divindex] =
        sharedmemorys[global_index2][global_index1 + (idx_y << 1) + 48];
}

__global__ void FourStepInverseCoreT2(Data* polynomial_in, Data* polynomial_out,
                                      Root* n1_root_of_unity_table,
                                      Modulus modulus, int index1, int index2,
                                      int index3, int n_power)
{
    int idx_x = threadIdx.x;
    int idx_y = threadIdx.y;
    int block_x = blockIdx.x;
    int block_y = blockIdx.y;

    __shared__ Data sharedmemorys[16][64 + 1];

    int divindex = block_y << n_power;

    Modulus q_thread = modulus;

    int idx_index = idx_x + (idx_y << 5);
    int global_addresss = idx_index + (block_x << 10);

    int shr_index1 = idx_y >> 1;
    int shr_index2 = idx_y % 2;

    // Load data from global & store to shared
    sharedmemorys[shr_index1][idx_x + (shr_index2 << 5)] =
        polynomial_in[global_addresss + divindex];
    sharedmemorys[shr_index1 + 4][idx_x + (shr_index2 << 5)] =
        polynomial_in[global_addresss + 256 + divindex];
    sharedmemorys[shr_index1 + 8][idx_x + (shr_index2 << 5)] =
        polynomial_in[global_addresss + 512 + divindex];
    sharedmemorys[shr_index1 + 12][idx_x + (shr_index2 << 5)] =
        polynomial_in[global_addresss + 768 + divindex];
    __syncthreads();

    int t_ = 0;
    int t = 1 << t_;
    int in_shared_address = ((idx_x >> t_) << t_) + idx_x;

    GentlemanSandeUnit_(sharedmemorys[idx_y][in_shared_address],
                        sharedmemorys[idx_y][in_shared_address + t],
                        n1_root_of_unity_table[(idx_x >> t_)], q_thread);
    GentlemanSandeUnit_(sharedmemorys[idx_y + 8][in_shared_address],
                        sharedmemorys[idx_y + 8][in_shared_address + t],
                        n1_root_of_unity_table[(idx_x >> t_)], q_thread);
    __syncthreads();

    for (int i = 0; i < 5; i++)
    {
        t = t << 1;
        t_ += 1;

        in_shared_address = ((idx_x >> t_) << t_) + idx_x;

        GentlemanSandeUnit_(sharedmemorys[idx_y][in_shared_address],
                            sharedmemorys[idx_y][in_shared_address + t],
                            n1_root_of_unity_table[(idx_x >> t_)], q_thread);
        GentlemanSandeUnit_(sharedmemorys[idx_y + 8][in_shared_address],
                            sharedmemorys[idx_y + 8][in_shared_address + t],
                            n1_root_of_unity_table[(idx_x >> t_)], q_thread);
        __syncthreads();
    }
    __syncthreads();

    int global_index1 = idx_x >> 4;
    int global_index2 = idx_x % 16;

    polynomial_out[global_index2 + (global_index1 << index1) +
                   (idx_y << index2) + (block_x << 4) + divindex] =
        sharedmemorys[global_index2][global_index1 + (idx_y << 1)];
    polynomial_out[global_index2 + (global_index1 << index1) +
                   (idx_y << index2) + (block_x << 4) + index3 + divindex] =
        sharedmemorys[global_index2][global_index1 + (idx_y << 1) + 16];
    polynomial_out[global_index2 + (global_index1 << index1) +
                   (idx_y << index2) + (block_x << 4) + (index3 * 2) +
                   divindex] =
        sharedmemorys[global_index2][global_index1 + (idx_y << 1) + 32];
    polynomial_out[global_index2 + (global_index1 << index1) +
                   (idx_y << index2) + (block_x << 4) + (index3 * 3) +
                   divindex] =
        sharedmemorys[global_index2][global_index1 + (idx_y << 1) + 48];
}

__global__ void FourStepInverseCoreT3(Data* polynomial_in, Data* polynomial_out,
                                      Root* n1_root_of_unity_table,
                                      Modulus* modulus, int index1, int index2,
                                      int index3, int n_power, int mod_count)
{
    int idx_x = threadIdx.x;
    int idx_y = threadIdx.y;
    int block_x = blockIdx.x;
    int block_y = blockIdx.y;

    __shared__ Data sharedmemorys[8][128 + 1];

    int divindex = block_y << n_power;

    int q_index = block_y % mod_count;
    Modulus q_thread = modulus[q_index];

    int idx_index = idx_x + (idx_y << 5);
    int global_addresss = idx_index + (block_x << 10);

    int shr_index1 = idx_y >> 2;
    int shr_index2 = idx_y % 4;

    // Load data from global & store to shared
    sharedmemorys[shr_index1][idx_x + (shr_index2 << 5)] =
        polynomial_in[global_addresss + divindex];
    sharedmemorys[shr_index1 + 2][idx_x + (shr_index2 << 5)] =
        polynomial_in[global_addresss + 256 + divindex];
    sharedmemorys[shr_index1 + 4][idx_x + (shr_index2 << 5)] =
        polynomial_in[global_addresss + 512 + divindex];
    sharedmemorys[shr_index1 + 6][idx_x + (shr_index2 << 5)] =
        polynomial_in[global_addresss + 768 + divindex];
    __syncthreads();

    int shr_address = idx_x + ((idx_y % 2) << 5);
    int shr_in = idx_y >> 1;

    int t_ = 0;
    int t = 1 << t_;
    int in_shared_address = ((shr_address >> t_) << t_) + shr_address;

    GentlemanSandeUnit_(sharedmemorys[shr_in][in_shared_address],
                        sharedmemorys[shr_in][in_shared_address + t],
                        n1_root_of_unity_table[(shr_address >> t_)], q_thread);
    GentlemanSandeUnit_(sharedmemorys[shr_in + 4][in_shared_address],
                        sharedmemorys[shr_in + 4][in_shared_address + t],
                        n1_root_of_unity_table[(shr_address >> t_)], q_thread);
    __syncthreads();

    for (int i = 0; i < 6; i++)
    {
        t = t << 1;
        t_ += 1;

        in_shared_address = ((shr_address >> t_) << t_) + shr_address;

        GentlemanSandeUnit_(sharedmemorys[shr_in][in_shared_address],
                            sharedmemorys[shr_in][in_shared_address + t],
                            n1_root_of_unity_table[(shr_address >> t_)],
                            q_thread);
        GentlemanSandeUnit_(sharedmemorys[shr_in + 4][in_shared_address],
                            sharedmemorys[shr_in + 4][in_shared_address + t],
                            n1_root_of_unity_table[(shr_address >> t_)],
                            q_thread);
        __syncthreads();
    }
    __syncthreads();

    int global_index1 = idx_x >> 3;
    int global_index2 = idx_x % 8;

    polynomial_out[global_index2 + (global_index1 << index1) +
                   (idx_y << index2) + (block_x << 3) + divindex] =
        sharedmemorys[global_index2][global_index1 + (idx_y << 2)];
    polynomial_out[global_index2 + (global_index1 << index1) +
                   (idx_y << index2) + (block_x << 3) + index3 + divindex] =
        sharedmemorys[global_index2][global_index1 + (idx_y << 2) + 32];
    polynomial_out[global_index2 + (global_index1 << index1) +
                   (idx_y << index2) + (block_x << 3) + (index3 * 2) +
                   divindex] =
        sharedmemorys[global_index2][global_index1 + (idx_y << 2) + 64];
    polynomial_out[global_index2 + (global_index1 << index1) +
                   (idx_y << index2) + (block_x << 3) + (index3 * 3) +
                   divindex] =
        sharedmemorys[global_index2][global_index1 + (idx_y << 2) + 96];
}

__global__ void FourStepInverseCoreT3(Data* polynomial_in, Data* polynomial_out,
                                      Root* n1_root_of_unity_table,
                                      Modulus modulus, int index1, int index2,
                                      int index3, int n_power)
{
    int idx_x = threadIdx.x;
    int idx_y = threadIdx.y;
    int block_x = blockIdx.x;
    int block_y = blockIdx.y;

    __shared__ Data sharedmemorys[8][128 + 1];

    int divindex = block_y << n_power;

    Modulus q_thread = modulus;

    int idx_index = idx_x + (idx_y << 5);
    int global_addresss = idx_index + (block_x << 10);

    int shr_index1 = idx_y >> 2;
    int shr_index2 = idx_y % 4;

    // Load data from global & store to shared
    sharedmemorys[shr_index1][idx_x + (shr_index2 << 5)] =
        polynomial_in[global_addresss + divindex];
    sharedmemorys[shr_index1 + 2][idx_x + (shr_index2 << 5)] =
        polynomial_in[global_addresss + 256 + divindex];
    sharedmemorys[shr_index1 + 4][idx_x + (shr_index2 << 5)] =
        polynomial_in[global_addresss + 512 + divindex];
    sharedmemorys[shr_index1 + 6][idx_x + (shr_index2 << 5)] =
        polynomial_in[global_addresss + 768 + divindex];
    __syncthreads();

    int shr_address = idx_x + ((idx_y % 2) << 5);
    int shr_in = idx_y >> 1;

    int t_ = 0;
    int t = 1 << t_;
    int in_shared_address = ((shr_address >> t_) << t_) + shr_address;

    GentlemanSandeUnit_(sharedmemorys[shr_in][in_shared_address],
                        sharedmemorys[shr_in][in_shared_address + t],
                        n1_root_of_unity_table[(shr_address >> t_)], q_thread);
    GentlemanSandeUnit_(sharedmemorys[shr_in + 4][in_shared_address],
                        sharedmemorys[shr_in + 4][in_shared_address + t],
                        n1_root_of_unity_table[(shr_address >> t_)], q_thread);
    __syncthreads();

    for (int i = 0; i < 6; i++)
    {
        t = t << 1;
        t_ += 1;

        in_shared_address = ((shr_address >> t_) << t_) + shr_address;

        GentlemanSandeUnit_(sharedmemorys[shr_in][in_shared_address],
                            sharedmemorys[shr_in][in_shared_address + t],
                            n1_root_of_unity_table[(shr_address >> t_)],
                            q_thread);
        GentlemanSandeUnit_(sharedmemorys[shr_in + 4][in_shared_address],
                            sharedmemorys[shr_in + 4][in_shared_address + t],
                            n1_root_of_unity_table[(shr_address >> t_)],
                            q_thread);
        __syncthreads();
    }
    __syncthreads();

    int global_index1 = idx_x >> 3;
    int global_index2 = idx_x % 8;

    polynomial_out[global_index2 + (global_index1 << index1) +
                   (idx_y << index2) + (block_x << 3) + divindex] =
        sharedmemorys[global_index2][global_index1 + (idx_y << 2)];
    polynomial_out[global_index2 + (global_index1 << index1) +
                   (idx_y << index2) + (block_x << 3) + index3 + divindex] =
        sharedmemorys[global_index2][global_index1 + (idx_y << 2) + 32];
    polynomial_out[global_index2 + (global_index1 << index1) +
                   (idx_y << index2) + (block_x << 3) + (index3 * 2) +
                   divindex] =
        sharedmemorys[global_index2][global_index1 + (idx_y << 2) + 64];
    polynomial_out[global_index2 + (global_index1 << index1) +
                   (idx_y << index2) + (block_x << 3) + (index3 * 3) +
                   divindex] =
        sharedmemorys[global_index2][global_index1 + (idx_y << 2) + 96];
}

__global__ void FourStepInverseCoreT4(Data* polynomial_in, Data* polynomial_out,
                                      Root* n1_root_of_unity_table,
                                      Modulus* modulus, int index1, int index2,
                                      int index3, int n_power, int mod_count)
{
    int idx_x = threadIdx.x;
    int idx_y = threadIdx.y;
    int block_x = blockIdx.x;
    int block_y = blockIdx.y;

    __shared__ Data sharedmemorys[4][256 + 1];

    int divindex = block_y << n_power;

    int q_index = block_y % mod_count;
    Modulus q_thread = modulus[q_index];

    int idx_index = idx_x + (idx_y << 5);
    int global_addresss = idx_index + (block_x << 10);

    int shr_index1 = idx_y >> 3;
    int shr_index2 = idx_y % 8;

    // Load data from global & store to shared
    sharedmemorys[shr_index1][idx_x + (shr_index2 << 5)] =
        polynomial_in[global_addresss + divindex];
    sharedmemorys[shr_index1 + 1][idx_x + (shr_index2 << 5)] =
        polynomial_in[global_addresss + 256 + divindex];
    sharedmemorys[shr_index1 + 2][idx_x + (shr_index2 << 5)] =
        polynomial_in[global_addresss + 512 + divindex];
    sharedmemorys[shr_index1 + 3][idx_x + (shr_index2 << 5)] =
        polynomial_in[global_addresss + 768 + divindex];
    __syncthreads();

    int shr_address = idx_x + ((idx_y % 4) << 5);
    int shr_in = idx_y >> 2;

    int t_ = 0;
    int t = 1 << t_;
    int in_shared_address = ((shr_address >> t_) << t_) + shr_address;

    GentlemanSandeUnit_(sharedmemorys[shr_in][in_shared_address],
                        sharedmemorys[shr_in][in_shared_address + t],
                        n1_root_of_unity_table[(shr_address >> t_)], q_thread);
    GentlemanSandeUnit_(sharedmemorys[shr_in + 2][in_shared_address],
                        sharedmemorys[shr_in + 2][in_shared_address + t],
                        n1_root_of_unity_table[(shr_address >> t_)], q_thread);
    __syncthreads();

    for (int i = 0; i < 7; i++)
    {
        t = t << 1;
        t_ += 1;

        in_shared_address = ((shr_address >> t_) << t_) + shr_address;

        GentlemanSandeUnit_(sharedmemorys[shr_in][in_shared_address],
                            sharedmemorys[shr_in][in_shared_address + t],
                            n1_root_of_unity_table[(shr_address >> t_)],
                            q_thread);
        GentlemanSandeUnit_(sharedmemorys[shr_in + 2][in_shared_address],
                            sharedmemorys[shr_in + 2][in_shared_address + t],
                            n1_root_of_unity_table[(shr_address >> t_)],
                            q_thread);
        __syncthreads();
    }
    __syncthreads();

    int global_index1 = idx_x >> 2;
    int global_index2 = idx_x % 4;

    polynomial_out[global_index2 + (global_index1 << index1) +
                   (idx_y << index2) + (block_x << 2) + divindex] =
        sharedmemorys[global_index2][global_index1 + (idx_y << 3)];
    polynomial_out[global_index2 + (global_index1 << index1) +
                   (idx_y << index2) + (block_x << 2) + index3 + divindex] =
        sharedmemorys[global_index2][global_index1 + (idx_y << 3) + 64];
    polynomial_out[global_index2 + (global_index1 << index1) +
                   (idx_y << index2) + (block_x << 2) + (index3 * 2) +
                   divindex] =
        sharedmemorys[global_index2][global_index1 + (idx_y << 3) + 128];
    polynomial_out[global_index2 + (global_index1 << index1) +
                   (idx_y << index2) + (block_x << 2) + (index3 * 3) +
                   divindex] =
        sharedmemorys[global_index2][global_index1 + (idx_y << 3) + 192];
}

__global__ void FourStepInverseCoreT4(Data* polynomial_in, Data* polynomial_out,
                                      Root* n1_root_of_unity_table,
                                      Modulus modulus, int index1, int index2,
                                      int index3, int n_power)
{
    int idx_x = threadIdx.x;
    int idx_y = threadIdx.y;
    int block_x = blockIdx.x;
    int block_y = blockIdx.y;

    __shared__ Data sharedmemorys[4][256 + 1];

    int divindex = block_y << n_power;

    Modulus q_thread = modulus;

    int idx_index = idx_x + (idx_y << 5);
    int global_addresss = idx_index + (block_x << 10);

    int shr_index1 = idx_y >> 3;
    int shr_index2 = idx_y % 8;

    // Load data from global & store to shared
    sharedmemorys[shr_index1][idx_x + (shr_index2 << 5)] =
        polynomial_in[global_addresss + divindex];
    sharedmemorys[shr_index1 + 1][idx_x + (shr_index2 << 5)] =
        polynomial_in[global_addresss + 256 + divindex];
    sharedmemorys[shr_index1 + 2][idx_x + (shr_index2 << 5)] =
        polynomial_in[global_addresss + 512 + divindex];
    sharedmemorys[shr_index1 + 3][idx_x + (shr_index2 << 5)] =
        polynomial_in[global_addresss + 768 + divindex];
    __syncthreads();

    int shr_address = idx_x + ((idx_y % 4) << 5);
    int shr_in = idx_y >> 2;

    int t_ = 0;
    int t = 1 << t_;
    int in_shared_address = ((shr_address >> t_) << t_) + shr_address;

    GentlemanSandeUnit_(sharedmemorys[shr_in][in_shared_address],
                        sharedmemorys[shr_in][in_shared_address + t],
                        n1_root_of_unity_table[(shr_address >> t_)], q_thread);
    GentlemanSandeUnit_(sharedmemorys[shr_in + 2][in_shared_address],
                        sharedmemorys[shr_in + 2][in_shared_address + t],
                        n1_root_of_unity_table[(shr_address >> t_)], q_thread);
    __syncthreads();

    for (int i = 0; i < 7; i++)
    {
        t = t << 1;
        t_ += 1;

        in_shared_address = ((shr_address >> t_) << t_) + shr_address;

        GentlemanSandeUnit_(sharedmemorys[shr_in][in_shared_address],
                            sharedmemorys[shr_in][in_shared_address + t],
                            n1_root_of_unity_table[(shr_address >> t_)],
                            q_thread);
        GentlemanSandeUnit_(sharedmemorys[shr_in + 2][in_shared_address],
                            sharedmemorys[shr_in + 2][in_shared_address + t],
                            n1_root_of_unity_table[(shr_address >> t_)],
                            q_thread);
        __syncthreads();
    }
    __syncthreads();

    int global_index1 = idx_x >> 2;
    int global_index2 = idx_x % 4;

    polynomial_out[global_index2 + (global_index1 << index1) +
                   (idx_y << index2) + (block_x << 2) + divindex] =
        sharedmemorys[global_index2][global_index1 + (idx_y << 3)];
    polynomial_out[global_index2 + (global_index1 << index1) +
                   (idx_y << index2) + (block_x << 2) + index3 + divindex] =
        sharedmemorys[global_index2][global_index1 + (idx_y << 3) + 64];
    polynomial_out[global_index2 + (global_index1 << index1) +
                   (idx_y << index2) + (block_x << 2) + (index3 * 2) +
                   divindex] =
        sharedmemorys[global_index2][global_index1 + (idx_y << 3) + 128];
    polynomial_out[global_index2 + (global_index1 << index1) +
                   (idx_y << index2) + (block_x << 2) + (index3 * 3) +
                   divindex] =
        sharedmemorys[global_index2][global_index1 + (idx_y << 3) + 192];
}

__global__ void FourStepPartialInverseCore(Data* polynomial_in,
                                           Root* n2_root_of_unity_table,
                                           Root* w_root_of_unity_table,
                                           Modulus* modulus, int small_npower,
                                           int LOOP, Ninverse* inverse,
                                           int poly_n_power, int mod_count)
{
    int local_idx = threadIdx.x;
    int block_x = blockIdx.x;
    int block_y = blockIdx.y;

    __shared__ Data sharedmemorys[512];

    int small_npower_ = small_npower;
    int t_2 = 0;
    int t = 1 << t_2;

    int dividx = (block_x << small_npower_);

    int divindex = block_y << poly_n_power;

    int q_index = block_y % mod_count;
    Modulus q_thread = modulus[q_index];

    int address = dividx + ((local_idx >> t_2) << t_2) + local_idx;

    int shrd_dixidx_t = (local_idx >> t_2) << t_2;
    int shrd_address = shrd_dixidx_t + local_idx;

    Data mult_1 = polynomial_in[address + divindex];
    Data mult_2 = polynomial_in[address + t + divindex];

    mult_1 = VALUE_GPU::mult(mult_1, w_root_of_unity_table[address], q_thread);
    mult_2 =
        VALUE_GPU::mult(mult_2, w_root_of_unity_table[address + t], q_thread);

    sharedmemorys[shrd_address] = mult_1;
    sharedmemorys[shrd_address + t] = mult_2;

#pragma unroll
    for (int loop_dep = 0; loop_dep < LOOP; loop_dep++)
    {
        GentlemanSandeUnit_(
            sharedmemorys[shrd_address], sharedmemorys[shrd_address + t],
            n2_root_of_unity_table[(local_idx >> t_2)], q_thread);

        t = t << 1;
        t_2 += 1;

        shrd_dixidx_t = (local_idx >> t_2) << t_2;
        shrd_address = shrd_dixidx_t + local_idx;
        __syncthreads();
    }

    address = dividx + ((local_idx >> t_2) << t_2) + local_idx;

    GentlemanSandeUnit_(sharedmemorys[shrd_address],
                        sharedmemorys[shrd_address + t],
                        n2_root_of_unity_table[(local_idx >> t_2)], q_thread);
    __syncthreads();

    Data temp1 =
        VALUE_GPU::mult(sharedmemorys[shrd_address], inverse[0], q_thread);
    polynomial_in[address + divindex] = temp1;

    Data temp2 =
        VALUE_GPU::mult(sharedmemorys[shrd_address + t], inverse[0], q_thread);
    polynomial_in[address + t + divindex] = temp2;
}

__global__ void FourStepPartialInverseCore(Data* polynomial_in,
                                           Root* n2_root_of_unity_table,
                                           Root* w_root_of_unity_table,
                                           Modulus modulus, int small_npower,
                                           int LOOP, Ninverse inverse,
                                           int poly_n_power)
{
    int local_idx = threadIdx.x;
    int block_x = blockIdx.x;
    int block_y = blockIdx.y;

    __shared__ Data sharedmemorys[512];

    int small_npower_ = small_npower;
    int t_2 = 0;
    int t = 1 << t_2;

    int dividx = (block_x << small_npower_);

    int divindex = block_y << poly_n_power;

    Modulus q_thread = modulus;

    int address = dividx + ((local_idx >> t_2) << t_2) + local_idx;

    int shrd_dixidx_t = (local_idx >> t_2) << t_2;
    int shrd_address = shrd_dixidx_t + local_idx;

    Data mult_1 = polynomial_in[address + divindex];
    Data mult_2 = polynomial_in[address + t + divindex];

    mult_1 = VALUE_GPU::mult(mult_1, w_root_of_unity_table[address], q_thread);
    mult_2 =
        VALUE_GPU::mult(mult_2, w_root_of_unity_table[address + t], q_thread);

    sharedmemorys[shrd_address] = mult_1;
    sharedmemorys[shrd_address + t] = mult_2;

#pragma unroll
    for (int loop_dep = 0; loop_dep < LOOP; loop_dep++)
    {
        GentlemanSandeUnit_(
            sharedmemorys[shrd_address], sharedmemorys[shrd_address + t],
            n2_root_of_unity_table[(local_idx >> t_2)], q_thread);

        t = t << 1;
        t_2 += 1;

        shrd_dixidx_t = (local_idx >> t_2) << t_2;
        shrd_address = shrd_dixidx_t + local_idx;
        __syncthreads();
    }

    address = dividx + ((local_idx >> t_2) << t_2) + local_idx;

    GentlemanSandeUnit_(sharedmemorys[shrd_address],
                        sharedmemorys[shrd_address + t],
                        n2_root_of_unity_table[(local_idx >> t_2)], q_thread);
    __syncthreads();

    Data temp1 =
        VALUE_GPU::mult(sharedmemorys[shrd_address], inverse, q_thread);
    polynomial_in[address + divindex] = temp1;

    Data temp2 =
        VALUE_GPU::mult(sharedmemorys[shrd_address + t], inverse, q_thread);
    polynomial_in[address + t + divindex] = temp2;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void FourStepPartialInverseCore1(Data* polynomial_in,
                                            Root* n2_root_of_unity_table,
                                            Root* w_root_of_unity_table,
                                            Modulus* modulus, int small_npower,
                                            int poly_n_power, int mod_count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int block_y = blockIdx.y;
    int block_z = blockIdx.z;

    int local_idx = threadIdx.x;

    __shared__ Data sharedmemorys[512];

    int small_npower_ = small_npower;
    int t_2 = 0;
    int t = 1 << t_2;

    int dividx = block_y << small_npower_;

    int divindex = block_z << poly_n_power;

    int q_index = block_z % mod_count;
    Modulus q_thread = modulus[q_index];

    int address = dividx + ((idx >> t_2) << t_2) + idx;

    int shrd_dixidx_t = (local_idx >> t_2) << t_2;
    int shrd_address = shrd_dixidx_t + local_idx;

    Data mult_1 = polynomial_in[address + divindex];
    Data mult_2 = polynomial_in[address + t + divindex];

    mult_1 = VALUE_GPU::mult(mult_1, w_root_of_unity_table[address], q_thread);
    mult_2 =
        VALUE_GPU::mult(mult_2, w_root_of_unity_table[address + t], q_thread);

    sharedmemorys[shrd_address] = mult_1;
    sharedmemorys[shrd_address + t] = mult_2;

    for (int loop = 0; loop < 8; loop++)
    {
        GentlemanSandeUnit_(sharedmemorys[shrd_address],
                            sharedmemorys[shrd_address + t],
                            n2_root_of_unity_table[(idx >> t_2)], q_thread);

        t = t << 1;
        t_2 += 1;

        shrd_dixidx_t = (local_idx >> t_2) << t_2;
        shrd_address = shrd_dixidx_t + local_idx;
        __syncthreads();
    }

    address = dividx + ((idx >> t_2) << t_2) + idx;

    GentlemanSandeUnit_(sharedmemorys[shrd_address],
                        sharedmemorys[shrd_address + t],
                        n2_root_of_unity_table[(idx >> t_2)], q_thread);

    polynomial_in[address + divindex] = sharedmemorys[shrd_address];
    polynomial_in[address + t + divindex] = sharedmemorys[shrd_address + t];
}

__global__ void FourStepPartialInverseCore1(Data* polynomial_in,
                                            Root* n2_root_of_unity_table,
                                            Root* w_root_of_unity_table,
                                            Modulus modulus, int small_npower,
                                            int poly_n_power)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int block_y = blockIdx.y;
    int block_z = blockIdx.z;

    int local_idx = threadIdx.x;

    __shared__ Data sharedmemorys[512];

    int small_npower_ = small_npower;
    int t_2 = 0;
    int t = 1 << t_2;

    int dividx = block_y << small_npower_;

    int divindex = block_z << poly_n_power;

    Modulus q_thread = modulus;

    int address = dividx + ((idx >> t_2) << t_2) + idx;

    int shrd_dixidx_t = (local_idx >> t_2) << t_2;
    int shrd_address = shrd_dixidx_t + local_idx;

    Data mult_1 = polynomial_in[address + divindex];
    Data mult_2 = polynomial_in[address + t + divindex];

    mult_1 = VALUE_GPU::mult(mult_1, w_root_of_unity_table[address], q_thread);
    mult_2 =
        VALUE_GPU::mult(mult_2, w_root_of_unity_table[address + t], q_thread);

    sharedmemorys[shrd_address] = mult_1;
    sharedmemorys[shrd_address + t] = mult_2;

    for (int loop = 0; loop < 8; loop++)
    {
        GentlemanSandeUnit_(sharedmemorys[shrd_address],
                            sharedmemorys[shrd_address + t],
                            n2_root_of_unity_table[(idx >> t_2)], q_thread);

        t = t << 1;
        t_2 += 1;

        shrd_dixidx_t = (local_idx >> t_2) << t_2;
        shrd_address = shrd_dixidx_t + local_idx;
        __syncthreads();
    }

    address = dividx + ((idx >> t_2) << t_2) + idx;

    GentlemanSandeUnit_(sharedmemorys[shrd_address],
                        sharedmemorys[shrd_address + t],
                        n2_root_of_unity_table[(idx >> t_2)], q_thread);

    polynomial_in[address + divindex] = sharedmemorys[shrd_address];
    polynomial_in[address + t + divindex] = sharedmemorys[shrd_address + t];
}

__global__ void FourStepPartialInverseCore2(Data* polynomial_in,
                                            Root* n2_root_of_unity_table,
                                            Modulus* modulus, int small_npower,
                                            int T, int loc1, int loc2, int loc3,
                                            int loop, Ninverse* inverse,
                                            int poly_n_power, int mod_count)
{
    int idx_x = threadIdx.x;
    int idx_y = threadIdx.y;
    int block_x = blockIdx.x;
    int block_y = blockIdx.y;
    int block_z = blockIdx.z;

    __shared__ Data sharedmemorys[512];

    int small_npower_ = small_npower;
    int t_2 = T;

    int divindex = block_z << poly_n_power;

    int q_index = block_z % mod_count;
    Modulus q_thread = modulus[q_index];

    int grid = block_y << small_npower_;

    int global_addresss = (idx_y << loc3) + idx_x + (block_x << loc1);
    int shared_addresss = (idx_x + (idx_y << loc1));

    int load_store_address = global_addresss + grid;

    // Load data from global & store to shared
    sharedmemorys[shared_addresss] =
        polynomial_in[load_store_address + divindex];
    sharedmemorys[shared_addresss + 256] =
        polynomial_in[load_store_address + loc2 + divindex];
    __syncthreads();

    int t_ = loc1;
    int t = 1 << t_;
    int in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;

    GentlemanSandeUnit_(
        sharedmemorys[in_shared_address], sharedmemorys[in_shared_address + t],
        n2_root_of_unity_table[(global_addresss >> t_2)], q_thread);
    __syncthreads();

    for (int lp = 0; lp < loop; lp++)
    {
        t = t << 1;
        t_2 += 1;
        t_ += 1;

        in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;

        GentlemanSandeUnit_(sharedmemorys[in_shared_address],
                            sharedmemorys[in_shared_address + t],
                            n2_root_of_unity_table[(global_addresss >> t_2)],
                            q_thread);

        __syncthreads();
    }

    Data temp1 =
        VALUE_GPU::mult(sharedmemorys[shared_addresss], inverse[0], q_thread);
    polynomial_in[load_store_address + divindex] = temp1;

    Data temp2 = VALUE_GPU::mult(sharedmemorys[shared_addresss + 256],
                                 inverse[0], q_thread);
    polynomial_in[load_store_address + loc2 + divindex] = temp2;
}

__global__ void FourStepPartialInverseCore2(Data* polynomial_in,
                                            Root* n2_root_of_unity_table,
                                            Modulus modulus, int small_npower,
                                            int T, int loc1, int loc2, int loc3,
                                            int loop, Ninverse inverse,
                                            int poly_n_power)
{
    int idx_x = threadIdx.x;
    int idx_y = threadIdx.y;
    int block_x = blockIdx.x;
    int block_y = blockIdx.y;
    int block_z = blockIdx.z;

    __shared__ Data sharedmemorys[512];

    int small_npower_ = small_npower;
    int t_2 = T;

    int divindex = block_z << poly_n_power;

    Modulus q_thread = modulus;

    int grid = block_y << small_npower_;

    int global_addresss = (idx_y << loc3) + idx_x + (block_x << loc1);
    int shared_addresss = (idx_x + (idx_y << loc1));

    int load_store_address = global_addresss + grid;

    // Load data from global & store to shared
    sharedmemorys[shared_addresss] =
        polynomial_in[load_store_address + divindex];
    sharedmemorys[shared_addresss + 256] =
        polynomial_in[load_store_address + loc2 + divindex];
    __syncthreads();

    int t_ = loc1;
    int t = 1 << t_;
    int in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;

    GentlemanSandeUnit_(
        sharedmemorys[in_shared_address], sharedmemorys[in_shared_address + t],
        n2_root_of_unity_table[(global_addresss >> t_2)], q_thread);
    __syncthreads();

    for (int lp = 0; lp < loop; lp++)
    {
        t = t << 1;
        t_2 += 1;
        t_ += 1;

        in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;

        GentlemanSandeUnit_(sharedmemorys[in_shared_address],
                            sharedmemorys[in_shared_address + t],
                            n2_root_of_unity_table[(global_addresss >> t_2)],
                            q_thread);

        __syncthreads();
    }

    Data temp1 =
        VALUE_GPU::mult(sharedmemorys[shared_addresss], inverse, q_thread);
    polynomial_in[load_store_address + divindex] = temp1;

    Data temp2 = VALUE_GPU::mult(sharedmemorys[shared_addresss + 256], inverse,
                                 q_thread);
    polynomial_in[load_store_address + loc2 + divindex] = temp2;
}

__host__ void GPU_4STEP_NTT(Data* device_in, Data* device_out,
                            Root* n1_root_of_unity_table,
                            Root* n2_root_of_unity_table,
                            Root* W_root_of_unity_table, Modulus* modulus,
                            ntt4step_rns_configuration cfg, int batch_size,
                            int mod_count)
{
    switch (cfg.ntt_type)
    {
        case FORWARD:
            switch (cfg.n_power)
            {
                case 12:

                    FourStepForwardCoreT1<<<dim3(4, batch_size), dim3(32, 8)>>>(
                        device_in, device_out, n1_root_of_unity_table, modulus,
                        7, 1024, 12, mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialForwardCore<<<dim3(32, batch_size), 64>>>(
                        device_out, n2_root_of_unity_table,
                        W_root_of_unity_table, modulus, 7, 6, 1, 12, mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 13:

                    FourStepForwardCoreT1<<<dim3(8, batch_size), dim3(32, 8)>>>(
                        device_in, device_out, n1_root_of_unity_table, modulus,
                        8, 2048, 13, mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialForwardCore<<<dim3(32, batch_size), 128>>>(
                        device_out, n2_root_of_unity_table,
                        W_root_of_unity_table, modulus, 8, 7, 2, 13, mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 14:

                    FourStepForwardCoreT1<<<dim3(16, batch_size),
                                            dim3(32, 8)>>>(
                        device_in, device_out, n1_root_of_unity_table, modulus,
                        9, 4096, 14, mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialForwardCore<<<dim3(32, batch_size), 256>>>(
                        device_out, n2_root_of_unity_table,
                        W_root_of_unity_table, modulus, 9, 8, 3, 14, mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 15:

                    FourStepForwardCoreT2<<<dim3(32, batch_size),
                                            dim3(32, 8)>>>(
                        device_in, device_out, n1_root_of_unity_table, modulus,
                        9, 10, 8192, 15, mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialForwardCore<<<dim3(64, batch_size), 256>>>(
                        device_out, n2_root_of_unity_table,
                        W_root_of_unity_table, modulus, 9, 8, 3, 15, mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 16:

                    FourStepForwardCoreT3<<<dim3(64, batch_size),
                                            dim3(32, 8)>>>(
                        device_in, device_out, n1_root_of_unity_table, modulus,
                        9, 11, 16384, 16, mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialForwardCore<<<dim3(128, batch_size), 256>>>(
                        device_out, n2_root_of_unity_table,
                        W_root_of_unity_table, modulus, 9, 8, 3, 16, mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 17:

                    FourStepForwardCoreT1<<<dim3(128, batch_size),
                                            dim3(32, 8)>>>(
                        device_in, device_out, n1_root_of_unity_table, modulus,
                        12, 32768, 17, mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialForwardCore1<<<dim3(8, 32, batch_size),
                                                  dim3(64, 4)>>>(
                        device_out, n2_root_of_unity_table,
                        W_root_of_unity_table, modulus, 12, 6, 2048, 3, 17,
                        mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialForwardCore2<<<dim3(8, 32, batch_size),
                                                  256>>>(
                        device_out, n2_root_of_unity_table, modulus, 12, 17,
                        mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 18:

                    FourStepForwardCoreT1<<<dim3(256, batch_size),
                                            dim3(32, 8)>>>(
                        device_in, device_out, n1_root_of_unity_table, modulus,
                        13, 65536, 18, mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialForwardCore1<<<dim3(16, 32, batch_size),
                                                  dim3(32, 8)>>>(
                        device_out, n2_root_of_unity_table,
                        W_root_of_unity_table, modulus, 13, 5, 4096, 4, 18,
                        mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialForwardCore2<<<dim3(16, 32, batch_size),
                                                  256>>>(
                        device_out, n2_root_of_unity_table, modulus, 13, 18,
                        mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 19:

                    FourStepForwardCoreT1<<<dim3(512, batch_size),
                                            dim3(32, 8)>>>(
                        device_in, device_out, n1_root_of_unity_table, modulus,
                        14, 131072, 19, mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialForwardCore1<<<dim3(32, 32, batch_size),
                                                  dim3(16, 16)>>>(
                        device_out, n2_root_of_unity_table,
                        W_root_of_unity_table, modulus, 14, 4, 8192, 5, 19,
                        mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialForwardCore2<<<dim3(32, 32, batch_size),
                                                  256>>>(
                        device_out, n2_root_of_unity_table, modulus, 14, 19,
                        mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 20:

                    FourStepForwardCoreT1<<<dim3(1024, batch_size),
                                            dim3(32, 8)>>>(
                        device_in, device_out, n1_root_of_unity_table, modulus,
                        15, 262144, 20, mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialForwardCore1<<<dim3(64, 32, batch_size),
                                                  dim3(8, 32)>>>(
                        device_out, n2_root_of_unity_table,
                        W_root_of_unity_table, modulus, 15, 3, 16384, 6, 20,
                        mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialForwardCore2<<<dim3(64, 32, batch_size),
                                                  256>>>(
                        device_out, n2_root_of_unity_table, modulus, 15, 20,
                        mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 21:

                    FourStepForwardCoreT2<<<dim3(2048, batch_size),
                                            dim3(32, 8)>>>(
                        device_in, device_out, n1_root_of_unity_table, modulus,
                        15, 16, 524288, 21, mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialForwardCore1<<<dim3(64, 64, batch_size),
                                                  dim3(8, 32)>>>(
                        device_out, n2_root_of_unity_table,
                        W_root_of_unity_table, modulus, 15, 3, 16384, 6, 21,
                        mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialForwardCore2<<<dim3(64, 64, batch_size),
                                                  256>>>(
                        device_out, n2_root_of_unity_table, modulus, 15, 21,
                        mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 22:

                    FourStepForwardCoreT3<<<dim3(4096, batch_size),
                                            dim3(32, 8)>>>(
                        device_in, device_out, n1_root_of_unity_table, modulus,
                        15, 17, 1048576, 22, mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialForwardCore1<<<dim3(64, 128, batch_size),
                                                  dim3(8, 32)>>>(
                        device_out, n2_root_of_unity_table,
                        W_root_of_unity_table, modulus, 15, 3, 16384, 6, 22,
                        mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialForwardCore2<<<dim3(64, 128, batch_size),
                                                  256>>>(
                        device_out, n2_root_of_unity_table, modulus, 15, 22,
                        mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 23:

                    FourStepForwardCoreT3<<<dim3(8192, batch_size),
                                            dim3(32, 8)>>>(
                        device_in, device_out, n1_root_of_unity_table, modulus,
                        16, 18, 2097152, 23, mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialForwardCore1<<<dim3(128, 128, batch_size),
                                                  dim3(4, 64)>>>(
                        device_out, n2_root_of_unity_table,
                        W_root_of_unity_table, modulus, 16, 2, 32768, 7, 23,
                        mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialForwardCore2<<<dim3(128, 128, batch_size),
                                                  256>>>(
                        device_out, n2_root_of_unity_table, modulus, 16, 23,
                        mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 24:

                    FourStepForwardCoreT4<<<dim3(16384, batch_size),
                                            dim3(32, 8)>>>(
                        device_in, device_out, n1_root_of_unity_table, modulus,
                        16, 19, 4194304, 24, mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialForwardCore1<<<dim3(128, 256, batch_size),
                                                  dim3(4, 64)>>>(
                        device_out, n2_root_of_unity_table,
                        W_root_of_unity_table, modulus, 16, 2, 32768, 7, 24,
                        mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialForwardCore2<<<dim3(128, 256, batch_size),
                                                  256>>>(
                        device_out, n2_root_of_unity_table, modulus, 16, 24,
                        mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;

                default:
                    std::cout << "This ring size is not supported!"
                              << std::endl;
                    break;
            }
            // ss
            break;
        case INVERSE:
            switch (cfg.n_power)
            {
                case 12:

                    FourStepInverseCoreT1<<<dim3(4, batch_size), dim3(32, 8)>>>(
                        device_in, device_out, n1_root_of_unity_table, modulus,
                        7, 1024, 12, mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialInverseCore<<<dim3(32, batch_size), 64>>>(
                        device_out, n2_root_of_unity_table,
                        W_root_of_unity_table, modulus, 7, 6, cfg.mod_inverse,
                        12, mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 13:

                    FourStepInverseCoreT1<<<dim3(8, batch_size), dim3(32, 8)>>>(
                        device_in, device_out, n1_root_of_unity_table, modulus,
                        8, 2048, 13, mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialInverseCore<<<dim3(32, batch_size), 128>>>(
                        device_out, n2_root_of_unity_table,
                        W_root_of_unity_table, modulus, 8, 7, cfg.mod_inverse,
                        13, mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 14:

                    FourStepInverseCoreT1<<<dim3(16, batch_size),
                                            dim3(32, 8)>>>(
                        device_in, device_out, n1_root_of_unity_table, modulus,
                        9, 4096, 14, mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialInverseCore<<<dim3(32, batch_size), 256>>>(
                        device_out, n2_root_of_unity_table,
                        W_root_of_unity_table, modulus, 9, 8, cfg.mod_inverse,
                        14, mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 15:

                    FourStepInverseCoreT2<<<dim3(32, batch_size),
                                            dim3(32, 8)>>>(
                        device_in, device_out, n1_root_of_unity_table, modulus,
                        9, 10, 8192, 15, mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialInverseCore<<<dim3(64, batch_size), 256>>>(
                        device_out, n2_root_of_unity_table,
                        W_root_of_unity_table, modulus, 9, 8, cfg.mod_inverse,
                        15, mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 16:

                    FourStepInverseCoreT3<<<dim3(64, batch_size),
                                            dim3(32, 8)>>>(
                        device_in, device_out, n1_root_of_unity_table, modulus,
                        9, 11, 16384, 16, mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialInverseCore<<<dim3(128, batch_size), 256>>>(
                        device_out, n2_root_of_unity_table,
                        W_root_of_unity_table, modulus, 9, 8, cfg.mod_inverse,
                        16, mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 17:

                    FourStepInverseCoreT1<<<dim3(128, batch_size),
                                            dim3(32, 8)>>>(
                        device_in, device_out, n1_root_of_unity_table, modulus,
                        12, 32768, 17, mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialInverseCore1<<<dim3(8, 32, batch_size),
                                                  256>>>(
                        device_out, n2_root_of_unity_table,
                        W_root_of_unity_table, modulus, 12, 17, mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialInverseCore2<<<dim3(8, 32, batch_size),
                                                  dim3(64, 4)>>>(
                        device_out, n2_root_of_unity_table, modulus, 12, 9, 6,
                        2048, 9, 2, cfg.mod_inverse, 17, mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 18:

                    FourStepInverseCoreT1<<<dim3(256, batch_size),
                                            dim3(32, 8)>>>(
                        device_in, device_out, n1_root_of_unity_table, modulus,
                        13, 65536, 18, mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialInverseCore1<<<dim3(16, 32, batch_size),
                                                  256>>>(
                        device_out, n2_root_of_unity_table,
                        W_root_of_unity_table, modulus, 13, 18, mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialInverseCore2<<<dim3(16, 32, batch_size),
                                                  dim3(32, 8)>>>(
                        device_out, n2_root_of_unity_table, modulus, 13, 9, 5,
                        4096, 9, 3, cfg.mod_inverse, 18, mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 19:

                    FourStepInverseCoreT1<<<dim3(512, batch_size),
                                            dim3(32, 8)>>>(
                        device_in, device_out, n1_root_of_unity_table, modulus,
                        14, 131072, 19, mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialInverseCore1<<<dim3(32, 32, batch_size),
                                                  256>>>(
                        device_out, n2_root_of_unity_table,
                        W_root_of_unity_table, modulus, 14, 19, mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialInverseCore2<<<dim3(32, 32, batch_size),
                                                  dim3(16, 16)>>>(
                        device_out, n2_root_of_unity_table, modulus, 14, 9, 4,
                        8192, 9, 4, cfg.mod_inverse, 19, mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 20:

                    FourStepInverseCoreT1<<<dim3(1024, batch_size),
                                            dim3(32, 8)>>>(
                        device_in, device_out, n1_root_of_unity_table, modulus,
                        15, 262144, 20, mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialInverseCore1<<<dim3(64, 32, batch_size),
                                                  256>>>(
                        device_out, n2_root_of_unity_table,
                        W_root_of_unity_table, modulus, 15, 20, mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialInverseCore2<<<dim3(64, 32, batch_size),
                                                  dim3(8, 32)>>>(
                        device_out, n2_root_of_unity_table, modulus, 15, 9, 3,
                        16384, 9, 5, cfg.mod_inverse, 20, mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 21:

                    FourStepInverseCoreT2<<<dim3(2048, batch_size),
                                            dim3(32, 8)>>>(
                        device_in, device_out, n1_root_of_unity_table, modulus,
                        15, 16, 524288, 21, mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialInverseCore1<<<dim3(64, 64, batch_size),
                                                  256>>>(
                        device_out, n2_root_of_unity_table,
                        W_root_of_unity_table, modulus, 15, 21, mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialInverseCore2<<<dim3(64, 64, batch_size),
                                                  dim3(8, 32)>>>(
                        device_out, n2_root_of_unity_table, modulus, 15, 9, 3,
                        16384, 9, 5, cfg.mod_inverse, 21, mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 22:

                    FourStepInverseCoreT3<<<dim3(4096, batch_size),
                                            dim3(32, 8)>>>(
                        device_in, device_out, n1_root_of_unity_table, modulus,
                        15, 17, 1048576, 22, mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialInverseCore1<<<dim3(64, 128, batch_size),
                                                  256>>>(
                        device_out, n2_root_of_unity_table,
                        W_root_of_unity_table, modulus, 15, 22, mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialInverseCore2<<<dim3(64, 128, batch_size),
                                                  dim3(8, 32)>>>(
                        device_out, n2_root_of_unity_table, modulus, 15, 9, 3,
                        16384, 9, 5, cfg.mod_inverse, 22, mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 23:

                    FourStepInverseCoreT3<<<dim3(8192, batch_size),
                                            dim3(32, 8)>>>(
                        device_in, device_out, n1_root_of_unity_table, modulus,
                        16, 18, 2097152, 23, mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialInverseCore1<<<dim3(128, 128, batch_size),
                                                  256>>>(
                        device_out, n2_root_of_unity_table,
                        W_root_of_unity_table, modulus, 16, 23, mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialInverseCore2<<<dim3(128, 128, batch_size),
                                                  dim3(4, 64)>>>(
                        device_out, n2_root_of_unity_table, modulus, 16, 9, 2,
                        32768, 9, 6, cfg.mod_inverse, 23, mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 24:

                    FourStepInverseCoreT4<<<dim3(16384, batch_size),
                                            dim3(32, 8)>>>(
                        device_in, device_out, n1_root_of_unity_table, modulus,
                        16, 19, 4194304, 24, mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialInverseCore1<<<dim3(128, 256, batch_size),
                                                  256>>>(
                        device_out, n2_root_of_unity_table,
                        W_root_of_unity_table, modulus, 16, 24, mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialInverseCore2<<<dim3(128, 256, batch_size),
                                                  dim3(4, 64)>>>(
                        device_out, n2_root_of_unity_table, modulus, 16, 9, 2,
                        32768, 9, 6, cfg.mod_inverse, 24, mod_count);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;

                default:
                    std::cout << "This ring size is not supported!"
                              << std::endl;
                    break;
            }

            break;

        default:
            break;
    }
}

__host__ void GPU_4STEP_NTT(Data* device_in, Data* device_out,
                            Root* n1_root_of_unity_table,
                            Root* n2_root_of_unity_table,
                            Root* W_root_of_unity_table, Modulus modulus,
                            ntt4step_configuration cfg, int batch_size)
{
    switch (cfg.ntt_type)
    {
        case FORWARD:
            switch (cfg.n_power)
            {
                case 12:

                    FourStepForwardCoreT1<<<dim3(4, batch_size), dim3(32, 8)>>>(
                        device_in, device_out, n1_root_of_unity_table, modulus,
                        7, 1024, 12);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialForwardCore<<<dim3(32, batch_size), 64>>>(
                        device_out, n2_root_of_unity_table,
                        W_root_of_unity_table, modulus, 7, 6, 1, 12);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 13:

                    FourStepForwardCoreT1<<<dim3(8, batch_size), dim3(32, 8)>>>(
                        device_in, device_out, n1_root_of_unity_table, modulus,
                        8, 2048, 13);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialForwardCore<<<dim3(32, batch_size), 128>>>(
                        device_out, n2_root_of_unity_table,
                        W_root_of_unity_table, modulus, 8, 7, 2, 13);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 14:

                    FourStepForwardCoreT1<<<dim3(16, batch_size),
                                            dim3(32, 8)>>>(
                        device_in, device_out, n1_root_of_unity_table, modulus,
                        9, 4096, 14);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialForwardCore<<<dim3(32, batch_size), 256>>>(
                        device_out, n2_root_of_unity_table,
                        W_root_of_unity_table, modulus, 9, 8, 3, 14);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 15:

                    FourStepForwardCoreT2<<<dim3(32, batch_size),
                                            dim3(32, 8)>>>(
                        device_in, device_out, n1_root_of_unity_table, modulus,
                        9, 10, 8192, 15);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialForwardCore<<<dim3(64, batch_size), 256>>>(
                        device_out, n2_root_of_unity_table,
                        W_root_of_unity_table, modulus, 9, 8, 3, 15);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 16:

                    FourStepForwardCoreT3<<<dim3(64, batch_size),
                                            dim3(32, 8)>>>(
                        device_in, device_out, n1_root_of_unity_table, modulus,
                        9, 11, 16384, 16);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialForwardCore<<<dim3(128, batch_size), 256>>>(
                        device_out, n2_root_of_unity_table,
                        W_root_of_unity_table, modulus, 9, 8, 3, 16);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 17:

                    FourStepForwardCoreT1<<<dim3(128, batch_size),
                                            dim3(32, 8)>>>(
                        device_in, device_out, n1_root_of_unity_table, modulus,
                        12, 32768, 17);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialForwardCore1<<<dim3(8, 32, batch_size),
                                                  dim3(64, 4)>>>(
                        device_out, n2_root_of_unity_table,
                        W_root_of_unity_table, modulus, 12, 6, 2048, 3, 17);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialForwardCore2<<<dim3(8, 32, batch_size),
                                                  256>>>(
                        device_out, n2_root_of_unity_table, modulus, 12, 17);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 18:

                    FourStepForwardCoreT1<<<dim3(256, batch_size),
                                            dim3(32, 8)>>>(
                        device_in, device_out, n1_root_of_unity_table, modulus,
                        13, 65536, 18);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialForwardCore1<<<dim3(16, 32, batch_size),
                                                  dim3(32, 8)>>>(
                        device_out, n2_root_of_unity_table,
                        W_root_of_unity_table, modulus, 13, 5, 4096, 4, 18);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialForwardCore2<<<dim3(16, 32, batch_size),
                                                  256>>>(
                        device_out, n2_root_of_unity_table, modulus, 13, 18);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 19:

                    FourStepForwardCoreT1<<<dim3(512, batch_size),
                                            dim3(32, 8)>>>(
                        device_in, device_out, n1_root_of_unity_table, modulus,
                        14, 131072, 19);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialForwardCore1<<<dim3(32, 32, batch_size),
                                                  dim3(16, 16)>>>(
                        device_out, n2_root_of_unity_table,
                        W_root_of_unity_table, modulus, 14, 4, 8192, 5, 19);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialForwardCore2<<<dim3(32, 32, batch_size),
                                                  256>>>(
                        device_out, n2_root_of_unity_table, modulus, 14, 19);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 20:

                    FourStepForwardCoreT1<<<dim3(1024, batch_size),
                                            dim3(32, 8)>>>(
                        device_in, device_out, n1_root_of_unity_table, modulus,
                        15, 262144, 20);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialForwardCore1<<<dim3(64, 32, batch_size),
                                                  dim3(8, 32)>>>(
                        device_out, n2_root_of_unity_table,
                        W_root_of_unity_table, modulus, 15, 3, 16384, 6, 20);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialForwardCore2<<<dim3(64, 32, batch_size),
                                                  256>>>(
                        device_out, n2_root_of_unity_table, modulus, 15, 20);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 21:

                    FourStepForwardCoreT2<<<dim3(2048, batch_size),
                                            dim3(32, 8)>>>(
                        device_in, device_out, n1_root_of_unity_table, modulus,
                        15, 16, 524288, 21);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialForwardCore1<<<dim3(64, 64, batch_size),
                                                  dim3(8, 32)>>>(
                        device_out, n2_root_of_unity_table,
                        W_root_of_unity_table, modulus, 15, 3, 16384, 6, 21);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialForwardCore2<<<dim3(64, 64, batch_size),
                                                  256>>>(
                        device_out, n2_root_of_unity_table, modulus, 15, 21);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 22:

                    FourStepForwardCoreT3<<<dim3(4096, batch_size),
                                            dim3(32, 8)>>>(
                        device_in, device_out, n1_root_of_unity_table, modulus,
                        15, 17, 1048576, 22);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialForwardCore1<<<dim3(64, 128, batch_size),
                                                  dim3(8, 32)>>>(
                        device_out, n2_root_of_unity_table,
                        W_root_of_unity_table, modulus, 15, 3, 16384, 6, 22);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialForwardCore2<<<dim3(64, 128, batch_size),
                                                  256>>>(
                        device_out, n2_root_of_unity_table, modulus, 15, 22);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 23:

                    FourStepForwardCoreT3<<<dim3(8192, batch_size),
                                            dim3(32, 8)>>>(
                        device_in, device_out, n1_root_of_unity_table, modulus,
                        16, 18, 2097152, 23);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialForwardCore1<<<dim3(128, 128, batch_size),
                                                  dim3(4, 64)>>>(
                        device_out, n2_root_of_unity_table,
                        W_root_of_unity_table, modulus, 16, 2, 32768, 7, 23);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialForwardCore2<<<dim3(128, 128, batch_size),
                                                  256>>>(
                        device_out, n2_root_of_unity_table, modulus, 16, 23);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 24:

                    FourStepForwardCoreT4<<<dim3(16384, batch_size),
                                            dim3(32, 8)>>>(
                        device_in, device_out, n1_root_of_unity_table, modulus,
                        16, 19, 4194304, 24);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialForwardCore1<<<dim3(128, 256, batch_size),
                                                  dim3(4, 64)>>>(
                        device_out, n2_root_of_unity_table,
                        W_root_of_unity_table, modulus, 16, 2, 32768, 7, 24);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialForwardCore2<<<dim3(128, 256, batch_size),
                                                  256>>>(
                        device_out, n2_root_of_unity_table, modulus, 16, 24);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;

                default:
                    std::cout << "This ring size is not supported!"
                              << std::endl;
                    break;
            }
            // ss
            break;
        case INVERSE:
            switch (cfg.n_power)
            {
                case 12:

                    FourStepInverseCoreT1<<<dim3(4, batch_size), dim3(32, 8)>>>(
                        device_in, device_out, n1_root_of_unity_table, modulus,
                        7, 1024, 12);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialInverseCore<<<dim3(32, batch_size), 64>>>(
                        device_out, n2_root_of_unity_table,
                        W_root_of_unity_table, modulus, 7, 6, cfg.mod_inverse,
                        12);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 13:

                    FourStepInverseCoreT1<<<dim3(8, batch_size), dim3(32, 8)>>>(
                        device_in, device_out, n1_root_of_unity_table, modulus,
                        8, 2048, 13);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialInverseCore<<<dim3(32, batch_size), 128>>>(
                        device_out, n2_root_of_unity_table,
                        W_root_of_unity_table, modulus, 8, 7, cfg.mod_inverse,
                        13);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 14:

                    FourStepInverseCoreT1<<<dim3(16, batch_size),
                                            dim3(32, 8)>>>(
                        device_in, device_out, n1_root_of_unity_table, modulus,
                        9, 4096, 14);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialInverseCore<<<dim3(32, batch_size), 256>>>(
                        device_out, n2_root_of_unity_table,
                        W_root_of_unity_table, modulus, 9, 8, cfg.mod_inverse,
                        14);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 15:

                    FourStepInverseCoreT2<<<dim3(32, batch_size),
                                            dim3(32, 8)>>>(
                        device_in, device_out, n1_root_of_unity_table, modulus,
                        9, 10, 8192, 15);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialInverseCore<<<dim3(64, batch_size), 256>>>(
                        device_out, n2_root_of_unity_table,
                        W_root_of_unity_table, modulus, 9, 8, cfg.mod_inverse,
                        15);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 16:

                    FourStepInverseCoreT3<<<dim3(64, batch_size),
                                            dim3(32, 8)>>>(
                        device_in, device_out, n1_root_of_unity_table, modulus,
                        9, 11, 16384, 16);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialInverseCore<<<dim3(128, batch_size), 256>>>(
                        device_out, n2_root_of_unity_table,
                        W_root_of_unity_table, modulus, 9, 8, cfg.mod_inverse,
                        16);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 17:

                    FourStepInverseCoreT1<<<dim3(128, batch_size),
                                            dim3(32, 8)>>>(
                        device_in, device_out, n1_root_of_unity_table, modulus,
                        12, 32768, 17);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialInverseCore1<<<dim3(8, 32, batch_size),
                                                  256>>>(
                        device_out, n2_root_of_unity_table,
                        W_root_of_unity_table, modulus, 12, 17);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialInverseCore2<<<dim3(8, 32, batch_size),
                                                  dim3(64, 4)>>>(
                        device_out, n2_root_of_unity_table, modulus, 12, 9, 6,
                        2048, 9, 2, cfg.mod_inverse, 17);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 18:

                    FourStepInverseCoreT1<<<dim3(256, batch_size),
                                            dim3(32, 8)>>>(
                        device_in, device_out, n1_root_of_unity_table, modulus,
                        13, 65536, 18);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialInverseCore1<<<dim3(16, 32, batch_size),
                                                  256>>>(
                        device_out, n2_root_of_unity_table,
                        W_root_of_unity_table, modulus, 13, 18);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialInverseCore2<<<dim3(16, 32, batch_size),
                                                  dim3(32, 8)>>>(
                        device_out, n2_root_of_unity_table, modulus, 13, 9, 5,
                        4096, 9, 3, cfg.mod_inverse, 18);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 19:

                    FourStepInverseCoreT1<<<dim3(512, batch_size),
                                            dim3(32, 8)>>>(
                        device_in, device_out, n1_root_of_unity_table, modulus,
                        14, 131072, 19);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialInverseCore1<<<dim3(32, 32, batch_size),
                                                  256>>>(
                        device_out, n2_root_of_unity_table,
                        W_root_of_unity_table, modulus, 14, 19);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialInverseCore2<<<dim3(32, 32, batch_size),
                                                  dim3(16, 16)>>>(
                        device_out, n2_root_of_unity_table, modulus, 14, 9, 4,
                        8192, 9, 4, cfg.mod_inverse, 19);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 20:

                    FourStepInverseCoreT1<<<dim3(1024, batch_size),
                                            dim3(32, 8)>>>(
                        device_in, device_out, n1_root_of_unity_table, modulus,
                        15, 262144, 20);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialInverseCore1<<<dim3(64, 32, batch_size),
                                                  256>>>(
                        device_out, n2_root_of_unity_table,
                        W_root_of_unity_table, modulus, 15, 20);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialInverseCore2<<<dim3(64, 32, batch_size),
                                                  dim3(8, 32)>>>(
                        device_out, n2_root_of_unity_table, modulus, 15, 9, 3,
                        16384, 9, 5, cfg.mod_inverse, 20);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 21:

                    FourStepInverseCoreT2<<<dim3(2048, batch_size),
                                            dim3(32, 8)>>>(
                        device_in, device_out, n1_root_of_unity_table, modulus,
                        15, 16, 524288, 21);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialInverseCore1<<<dim3(64, 64, batch_size),
                                                  256>>>(
                        device_out, n2_root_of_unity_table,
                        W_root_of_unity_table, modulus, 15, 21);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialInverseCore2<<<dim3(64, 64, batch_size),
                                                  dim3(8, 32)>>>(
                        device_out, n2_root_of_unity_table, modulus, 15, 9, 3,
                        16384, 9, 5, cfg.mod_inverse, 21);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 22:

                    FourStepInverseCoreT3<<<dim3(4096, batch_size),
                                            dim3(32, 8)>>>(
                        device_in, device_out, n1_root_of_unity_table, modulus,
                        15, 17, 1048576, 22);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialInverseCore1<<<dim3(64, 128, batch_size),
                                                  256>>>(
                        device_out, n2_root_of_unity_table,
                        W_root_of_unity_table, modulus, 15, 22);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialInverseCore2<<<dim3(64, 128, batch_size),
                                                  dim3(8, 32)>>>(
                        device_out, n2_root_of_unity_table, modulus, 15, 9, 3,
                        16384, 9, 5, cfg.mod_inverse, 22);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 23:

                    FourStepInverseCoreT3<<<dim3(8192, batch_size),
                                            dim3(32, 8)>>>(
                        device_in, device_out, n1_root_of_unity_table, modulus,
                        16, 18, 2097152, 23);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialInverseCore1<<<dim3(128, 128, batch_size),
                                                  256>>>(
                        device_out, n2_root_of_unity_table,
                        W_root_of_unity_table, modulus, 16, 23);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialInverseCore2<<<dim3(128, 128, batch_size),
                                                  dim3(4, 64)>>>(
                        device_out, n2_root_of_unity_table, modulus, 16, 9, 2,
                        32768, 9, 6, cfg.mod_inverse, 23);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;
                case 24:

                    FourStepInverseCoreT4<<<dim3(16384, batch_size),
                                            dim3(32, 8)>>>(
                        device_in, device_out, n1_root_of_unity_table, modulus,
                        16, 19, 4194304, 24);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialInverseCore1<<<dim3(128, 256, batch_size),
                                                  256>>>(
                        device_out, n2_root_of_unity_table,
                        W_root_of_unity_table, modulus, 16, 24);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    FourStepPartialInverseCore2<<<dim3(128, 256, batch_size),
                                                  dim3(4, 64)>>>(
                        device_out, n2_root_of_unity_table, modulus, 16, 9, 2,
                        32768, 9, 6, cfg.mod_inverse, 24);
                    THROW_IF_CUDA_ERROR(cudaGetLastError());
                    break;

                default:
                    std::cout << "This ring size is not supported!"
                              << std::endl;
                    break;
            }

            break;

        default:
            break;
    }
}