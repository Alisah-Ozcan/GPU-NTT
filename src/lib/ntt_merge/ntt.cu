// Copyright 2022 Ulvetanna Inc.
// Copyright 2023-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan
// Paper: https://eprint.iacr.org/2023/1410

#include "gpuntt/ntt_merge/ntt.cuh"
#include "gpuntt/ntt_merge/descriptor.cuh"

namespace gpuntt
{
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
    // and the modulus ordered and poly ordered variants) run one stage of a
    // transform that spans several kernels; each stage is split into load,
    // execute and store steps. The variants differ only in the modulus and
    // polynomial policies and in SwapGridAxes: the _ variants swap the roles
    // of blockIdx.x and blockIdx.y for large grids.

    // Per-thread state of one stage: thread (x, y) owns the shared memory
    // entries shared_address and shared_address + blockDim.x * blockDim.y,
    // read from and written to global_address and global_address + offset.
    template <typename TU> struct StageThread
    {
        int prime_index;
        Modulus<TU> modulus;
        Ninverse<TU> n_inverse;
        location_t offset;
        location_t global_address;
        location_t omega_address;
        location_t shared_address;
    };

    // The distance between the two coefficients of a thread is
    // 2^(N_power - level - 1), with level = logm for the forward and k for
    // the inverse transform. n_inverse is only read when WithNinverse is set.
    template <bool SwapGridAxes, bool WithNinverse, typename TU,
              typename ModulusPolicy, typename PolyPolicy>
    __device__ __forceinline__ StageThread<TU>
    StageThreadState(ModulusPolicy modulus_policy, PolyPolicy poly_policy,
                     int level, int outer_iteration_count, int N_power)
    {
        const int idx_x = threadIdx.x;
        const int idx_y = threadIdx.y;
        const int block_x = SwapGridAxes ? blockIdx.y : blockIdx.x;
        const int block_y = SwapGridAxes ? blockIdx.x : blockIdx.y;
        const int block_z = blockIdx.z;

        const int prime_index = modulus_policy.prime_index(block_z);
        const int poly_index = poly_policy.poly_index(block_z);

        const Modulus<TU> modulus = modulus_policy.get_modulus(prime_index);
        Ninverse<TU> n_inverse = 0;
        if constexpr (WithNinverse)
        {
            n_inverse = modulus_policy.get_n_inverse(prime_index);
        }

        const location_t offset = 1 << (N_power - level - 1);

        const location_t global_address =
            idx_x +
            (location_t) (idx_y *
                          (offset / (1 << (outer_iteration_count - 1)))) +
            (location_t) (blockDim.x * block_x) +
            (location_t) (2 * block_y * offset) +
            (location_t) (poly_index << N_power);

        const location_t omega_address =
            idx_x +
            (location_t) (idx_y *
                          (offset / (1 << (outer_iteration_count - 1)))) +
            (location_t) (blockDim.x * block_x) +
            (location_t) (block_y * offset);

        const location_t shared_address = (idx_x + (idx_y * blockDim.x));

        // Modulus has no device default constructor, so the state is built
        // in one aggregate initialization.
        return StageThread<TU>{prime_index,   modulus,        n_inverse,
                               offset,        global_address, omega_address,
                               shared_address};
    }

    // Copies the thread's two coefficients to shared memory; signed input is
    // reduced to [0, q).
    template <typename T>
    __device__ __forceinline__ void
    StageLoad(const StageThread<typename std::make_unsigned<T>::type>& thread,
              typename std::make_unsigned<T>::type* shared_memory,
              T* polynomial_in)
    {
        using TU = typename std::make_unsigned<T>::type;

        if constexpr (std::is_signed<T>::value)
        {
            T input1_reg = polynomial_in[thread.global_address];
            T input2_reg = polynomial_in[thread.global_address + thread.offset];
            shared_memory[thread.shared_address] =
                OPERATOR_GPU<TU>::reduce(input1_reg, thread.modulus);
            shared_memory[thread.shared_address + (blockDim.x * blockDim.y)] =
                OPERATOR_GPU<TU>::reduce(input2_reg, thread.modulus);
        }
        else
        {
            shared_memory[thread.shared_address] =
                polynomial_in[thread.global_address];
            shared_memory[thread.shared_address + (blockDim.x * blockDim.y)] =
                polynomial_in[thread.global_address + thread.offset];
        }
    }

    // Forward butterflies of one stage in shared memory. Ends with a block
    // barrier, so the result can be read by any thread of the block.
    template <typename TU>
    __device__ __forceinline__ void
    ForwardStageExecute(const StageThread<TU>& thread, TU* shared_memory,
                        const Root<TU>* root_of_unity_table, int shared_index,
                        int logm, int outer_iteration_count, int N_power,
                        bool not_last_kernel, bool reduction_poly_check)
    {
        const location_t shared_address = thread.shared_address;
        const location_t omega_address = thread.omega_address;

        int t_2 = N_power - logm - 1;
        int t_ = shared_index;
        location_t m = (location_t) 1 << logm;

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
                CooleyTukeyUnit(shared_memory[in_shared_address],
                                shared_memory[in_shared_address + t],
                                root_of_unity_table[current_root_index],
                                thread.modulus);

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

                CooleyTukeyUnit(shared_memory[in_shared_address],
                                shared_memory[in_shared_address + t],
                                root_of_unity_table[current_root_index],
                                thread.modulus);

                t = t >> 1;
                t_2 -= 1;
                t_ -= 1;
                m <<= 1;

                in_shared_address =
                    ((shared_address >> t_) << t_) + shared_address;
            }

            // In the last 6 butterfly levels (t = 32 ... 1) warp w only
            // touches shared memory entries 64 * w ... 64 * w + 63, so a warp
            // barrier between the levels is enough. The warp barriers also
            // order global memory accesses, so the roots of these levels are
            // loaded up front to keep the loads overlapped.
            Root<TU> last_roots[6];
#pragma unroll
            for (int lp = 0; lp < 6; lp++)
            {
                if (reduction_poly_check)
                { // X_N_minus
                    current_root_index =
                        (omega_address >> (t_2 - lp)) +
                        (location_t) (thread.prime_index << N_power);
                }
                else
                { // X_N_plus
                    current_root_index =
                        (m << lp) + (omega_address >> (t_2 - lp)) +
                        (location_t) (thread.prime_index << N_power);
                }
                last_roots[lp] = root_of_unity_table[current_root_index];
            }
            __syncthreads();

#pragma unroll
            for (int lp = 0; lp < 6; lp++)
            {
                CooleyTukeyUnit(shared_memory[in_shared_address],
                                shared_memory[in_shared_address + t],
                                last_roots[lp], thread.modulus);

                t = t >> 1;
                t_ -= 1;

                in_shared_address =
                    ((shared_address >> t_) << t_) + shared_address;
                __syncwarp();
            }
            __syncthreads();
        }
    }

    // Inverse butterflies of one stage in shared memory, without the scaling
    // by N^-1 (done by InverseStageStore in the last stage). Ends with a
    // block barrier.
    template <typename TU>
    __device__ __forceinline__ void
    InverseStageExecute(const StageThread<TU>& thread, TU* shared_memory,
                        const Root<TU>* inverse_root_of_unity_table,
                        int shared_index, int logm, int outer_iteration_count,
                        int N_power, bool reduction_poly_check)
    {
        const location_t shared_address = thread.shared_address;
        const location_t omega_address = thread.omega_address;

        int t_2 = N_power - logm - 1;
        int t_ = (shared_index + 1) - outer_iteration_count;
        int loops = outer_iteration_count;
        location_t m = (location_t) 1 << logm;

        int t = 1 << t_;
        int in_shared_address = ((shared_address >> t_) << t_) + shared_address;
        location_t current_root_index;
#pragma unroll
        for (int lp = 0; lp < loops; lp++)
        {
            __syncthreads();
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

            GentlemanSandeUnit(shared_memory[in_shared_address],
                               shared_memory[in_shared_address + t],
                               inverse_root_of_unity_table[current_root_index],
                               thread.modulus);

            t = t << 1;
            t_2 += 1;
            t_ += 1;
            m >>= 1;

            in_shared_address = ((shared_address >> t_) << t_) + shared_address;
        }
        __syncthreads();
    }

    // Copies the thread's two coefficients back to global memory.
    template <typename TU>
    __device__ __forceinline__ void
    ForwardStageStore(const StageThread<TU>& thread, TU* shared_memory,
                      TU* polynomial_out)
    {
        polynomial_out[thread.global_address] =
            shared_memory[thread.shared_address];
        polynomial_out[thread.global_address + thread.offset] =
            shared_memory[thread.shared_address + (blockDim.x * blockDim.y)];
    }

    // Copies the thread's two coefficients back to global memory. In the
    // last stage they are scaled by N^-1 and signed output is mapped to the
    // centered range.
    template <typename T>
    __device__ __forceinline__ void InverseStageStore(
        const StageThread<typename std::make_unsigned<T>::type>& thread,
        typename std::make_unsigned<T>::type* shared_memory, T* polynomial_out,
        bool last_kernel)
    {
        using TU = typename std::make_unsigned<T>::type;

        if (last_kernel)
        {
            TU output1_reg =
                OPERATOR_GPU<TU>::mult(shared_memory[thread.shared_address],
                                       thread.n_inverse, thread.modulus);
            TU output2_reg =
                OPERATOR_GPU<TU>::mult(shared_memory[thread.shared_address +
                                                     (blockDim.x * blockDim.y)],
                                       thread.n_inverse, thread.modulus);

            if constexpr (std::is_signed<T>::value)
            {
                polynomial_out[thread.global_address] =
                    OPERATOR_GPU<TU>::centered_reduction(output1_reg,
                                                         thread.modulus);
                polynomial_out[thread.global_address + thread.offset] =
                    OPERATOR_GPU<TU>::centered_reduction(output2_reg,
                                                         thread.modulus);
            }
            else
            {
                polynomial_out[thread.global_address] = output1_reg;
                polynomial_out[thread.global_address + thread.offset] =
                    output2_reg;
            }
        }
        else
        {
            polynomial_out[thread.global_address] =
                shared_memory[thread.shared_address];
            polynomial_out[thread.global_address + thread.offset] =
                shared_memory[thread.shared_address +
                              (blockDim.x * blockDim.y)];
        }
    }

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

        extern __shared__ char shared_memory_typed[];
        TU* shared_memory = reinterpret_cast<TU*>(shared_memory_typed);

        const StageThread<TU> thread =
            StageThreadState<SwapGridAxes, false, TU>(
                modulus_policy, poly_policy, logm, outer_iteration_count,
                N_power);
        StageLoad(thread, shared_memory, polynomial_in);
        ForwardStageExecute(thread, shared_memory, root_of_unity_table,
                            shared_index, logm, outer_iteration_count, N_power,
                            not_last_kernel, reduction_poly_check);
        ForwardStageStore(thread, shared_memory, polynomial_out);
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

        extern __shared__ char shared_memory_typed[];
        TU* shared_memory = reinterpret_cast<TU*>(shared_memory_typed);

        const StageThread<TU> thread = StageThreadState<SwapGridAxes, true, TU>(
            modulus_policy, poly_policy, k, outer_iteration_count, N_power);
        StageLoad(thread, shared_memory, polynomial_in);
        InverseStageExecute(thread, shared_memory, inverse_root_of_unity_table,
                            shared_index, logm, outer_iteration_count, N_power,
                            reduction_poly_check);
        InverseStageStore(thread, shared_memory, polynomial_out, last_kernel);
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

    // The transpose kernels run the PerCoefficient NTT: the input is a
    // 2^log_row x 2^log_column matrix and every column is one polynomial.
    // Block x handles blockDim.x adjacent columns; thread (x, y) loads rows y
    // and y + blockDim.y of column x into shared memory transposed, so that
    // each column becomes 2^log_row contiguous coefficients. The butterflies
    // of a column are then run by 2^(log_row - 1) consecutive threads of the
    // block, so a thread may compute in a different column than the one it
    // loads and stores. The RNS variant uses the modulus of each column and
    // keeps one root table per column in shared memory; the single modulus
    // variant keeps a single root table. Apart from loading the root tables,
    // both variants share the load, execute and store steps.

    // The 512 threads of a block hold 1024 coefficients in shared memory;
    // the root tables follow them.
    constexpr int kTransposeRootTableOffset = 1024;

    template <typename TU> struct TransposeThread
    {
        // Modulus of the column the thread computes in.
        Modulus<TU> modulus;
        // Modulus and N^-1 of the column the thread loads and stores.
        int io_prime_index;
        Modulus<TU> io_modulus;
        Ninverse<TU> io_n_inverse;
        location_t global_address;
        location_t global_offset;
        int transpose_block;
        int ntt_thread_index;
        // Start of the computed column in shared memory and of its root
        // table in the root table area.
        int offset;
        int root_offset;
    };

    // PerColumnRoots is set when every column has its own root table.
    // io_n_inverse is only read when WithNinverse is set (inverse transform).
    template <bool PerColumnRoots, bool WithNinverse, typename TU,
              typename ModulusPolicy>
    __device__ __forceinline__ TransposeThread<TU>
    TransposeThreadState(ModulusPolicy modulus_policy, int log_row,
                         int log_column)
    {
        const int idx_x = threadIdx.x;
        const int idx_y = threadIdx.y;
        const int block_x = blockIdx.x;
        const int column_base = blockDim.x * block_x;

        const int block_thread_index = idx_x + (idx_y * blockDim.x);
        const int ntt_thread_index =
            block_thread_index & ((1 << (log_row - 1)) - 1);
        const int ntt_block_index = block_thread_index >> (log_row - 1);
        const int offset = ntt_block_index << log_row;

        const int prime_index =
            modulus_policy.prime_index(column_base + ntt_block_index);
        const int io_prime_index =
            modulus_policy.prime_index(column_base + idx_x);
        const Modulus<TU> modulus = modulus_policy.get_modulus(prime_index);
        const Modulus<TU> io_modulus =
            modulus_policy.get_modulus(io_prime_index);
        Ninverse<TU> io_n_inverse = 0;
        if constexpr (WithNinverse)
        {
            io_n_inverse = modulus_policy.get_n_inverse(io_prime_index);
        }

        const location_t global_address =
            (idx_y << log_column) + (blockDim.x * block_x) + idx_x;
        const location_t global_offset = (blockDim.y << log_column);
        const int transpose_block = idx_y + (idx_x * (2 * blockDim.y));

        // Modulus has no device default constructor, so the state is built
        // in one aggregate initialization.
        return TransposeThread<TU>{
            modulus,         io_prime_index,
            io_modulus,      io_n_inverse,
            global_address,  global_offset,
            transpose_block, ntt_thread_index,
            offset,          PerColumnRoots ? offset : 0};
    }

    // Single modulus: the threads of column 0 load the one root table.
    template <typename TU>
    __device__ __forceinline__ void
    TransposeLoadRoots(TU* shared_memory, const Root<TU>* root_of_unity_table,
                       bool reduction_poly_check)
    {
        const int idx_x = threadIdx.x;
        const int idx_y = threadIdx.y;

        TU* root_shared_memory = shared_memory + kTransposeRootTableOffset;

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
    }

    // RNS: the threads of column x load the root table of the column's
    // modulus to root_shared_memory[x << log_row].
    template <typename TU>
    __device__ __forceinline__ void
    TransposeLoadColumnRoots(const TransposeThread<TU>& thread,
                             TU* shared_memory,
                             const Root<TU>* root_of_unity_table, int log_row,
                             bool reduction_poly_check)
    {
        const int idx_x = threadIdx.x;
        const int idx_y = threadIdx.y;

        TU* root_shared_memory = shared_memory + kTransposeRootTableOffset;

        const location_t root_src = (location_t) thread.io_prime_index
                                    << log_row;
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

    // Copies the thread's two coefficients to shared memory, transposed;
    // signed input is reduced to [0, q).
    template <typename T>
    __device__ __forceinline__ void TransposeLoad(
        const TransposeThread<typename std::make_unsigned<T>::type>& thread,
        typename std::make_unsigned<T>::type* shared_memory, T* polynomial_in)
    {
        using TU = typename std::make_unsigned<T>::type;

        if constexpr (std::is_signed<T>::value)
        {
            T input1_reg = polynomial_in[thread.global_address];
            T input2_reg =
                polynomial_in[thread.global_address + thread.global_offset];
            shared_memory[thread.transpose_block] =
                OPERATOR_GPU<TU>::reduce(input1_reg, thread.io_modulus);
            shared_memory[thread.transpose_block + blockDim.y] =
                OPERATOR_GPU<TU>::reduce(input2_reg, thread.io_modulus);
        }
        else
        {
            shared_memory[thread.transpose_block] =
                polynomial_in[thread.global_address];
            shared_memory[thread.transpose_block + blockDim.y] =
                polynomial_in[thread.global_address + thread.global_offset];
        }
    }

    // Forward NTT of the block's columns in shared memory. Ends with a
    // block barrier.
    template <typename TU>
    __device__ __forceinline__ void
    ForwardTransposeExecute(const TransposeThread<TU>& thread,
                            TU* shared_memory, int log_row,
                            bool reduction_poly_check)
    {
        const TU* root_shared_memory =
            shared_memory + kTransposeRootTableOffset;

        int t_ = log_row - 1;
        int m = 1;

        int shared_address = thread.ntt_thread_index;
        int omega_address = thread.ntt_thread_index;
        int t = 1 << t_;
        int in_shared_address = ((shared_address >> t_) << t_) + shared_address;
        location_t current_root_index;
        __syncthreads();

        int loop = ((log_row - 6) <= 0) ? log_row : 6;

#pragma unroll
        for (int lp = 0; lp < (log_row - 6); lp++)
        {
            int group_in_shared_address = in_shared_address + thread.offset;

            if (reduction_poly_check)
            { // X_N_minus
                current_root_index = (omega_address >> t_) + thread.root_offset;
            }
            else
            { // X_N_plus
                current_root_index =
                    m + (omega_address >> t_) + thread.root_offset;
            }

            Root<TU> root = root_shared_memory[current_root_index];

            CooleyTukeyUnit(shared_memory[group_in_shared_address],
                            shared_memory[group_in_shared_address + t], root,
                            thread.modulus);

            t = t >> 1;
            t_ -= 1;
            m <<= 1;

            in_shared_address = ((shared_address >> t_) << t_) + shared_address;
            __syncthreads();
        }

        // In the last levels (t <= 32) each group of 64 entries is computed
        // by one warp: by warp w at offset 64 * w if log_row >= 6, otherwise
        // a whole column of at most 32 entries by at most 16 consecutive
        // threads. A warp barrier between the levels is therefore enough.
#pragma unroll
        for (int lp = 0; lp < loop; lp++)
        {
            int group_in_shared_address = in_shared_address + thread.offset;

            if (reduction_poly_check)
            { // X_N_minus
                current_root_index = (omega_address >> t_) + thread.root_offset;
            }
            else
            { // X_N_plus
                current_root_index =
                    m + (omega_address >> t_) + thread.root_offset;
            }

            Root<TU> root = root_shared_memory[current_root_index];

            CooleyTukeyUnit(shared_memory[group_in_shared_address],
                            shared_memory[group_in_shared_address + t], root,
                            thread.modulus);

            t = t >> 1;
            t_ -= 1;
            m <<= 1;

            in_shared_address = ((shared_address >> t_) << t_) + shared_address;
            __syncwarp();
        }

        __syncthreads();
    }

    // Inverse NTT of the block's columns in shared memory, without the
    // scaling by N^-1 (done by InverseTransposeStore). Ends with a block
    // barrier.
    template <typename TU>
    __device__ __forceinline__ void
    InverseTransposeExecute(const TransposeThread<TU>& thread,
                            TU* shared_memory, int log_row,
                            bool reduction_poly_check)
    {
        const TU* root_shared_memory =
            shared_memory + kTransposeRootTableOffset;

        int t_ = 0;
        int loops = log_row;
        int m = (int) 1 << (log_row - 1);

        int omega_address = thread.ntt_thread_index;
        int shared_address = thread.ntt_thread_index;
        int t = 1 << t_;
        int in_shared_address = ((shared_address >> t_) << t_) + shared_address;
        location_t current_root_index;
        __syncthreads();

#pragma unroll
        for (int lp = 0; lp < loops; lp++)
        {
            int group_in_shared_address = in_shared_address + thread.offset;

            if (reduction_poly_check)
            { // X_N_minus
                current_root_index = (omega_address >> t_) + thread.root_offset;
            }
            else
            { // X_N_plus
                current_root_index =
                    m + (omega_address >> t_) + thread.root_offset;
            }

            Root<TU> root = root_shared_memory[current_root_index];

            GentlemanSandeUnit(shared_memory[group_in_shared_address],
                               shared_memory[group_in_shared_address + t], root,
                               thread.modulus);

            t = t << 1;
            t_ += 1;
            m >>= 1;

            in_shared_address = ((shared_address >> t_) << t_) + shared_address;
            __syncthreads();
        }
    }

    // Copies the thread's two coefficients back to global memory.
    template <typename TU>
    __device__ __forceinline__ void
    ForwardTransposeStore(const TransposeThread<TU>& thread, TU* shared_memory,
                          TU* polynomial_out)
    {
        polynomial_out[thread.global_address] =
            shared_memory[thread.transpose_block];
        polynomial_out[thread.global_address + thread.global_offset] =
            shared_memory[thread.transpose_block + blockDim.y];
    }

    // Scales the thread's two coefficients by N^-1 of their column and
    // writes them to global memory; signed output is mapped to the centered
    // range.
    template <typename T>
    __device__ __forceinline__ void InverseTransposeStore(
        const TransposeThread<typename std::make_unsigned<T>::type>& thread,
        typename std::make_unsigned<T>::type* shared_memory, T* polynomial_out)
    {
        using TU = typename std::make_unsigned<T>::type;

        TU output1_reg =
            OPERATOR_GPU<TU>::mult(shared_memory[thread.transpose_block],
                                   thread.io_n_inverse, thread.io_modulus);
        TU output2_reg = OPERATOR_GPU<TU>::mult(
            shared_memory[thread.transpose_block + blockDim.y],
            thread.io_n_inverse, thread.io_modulus);

        if constexpr (std::is_signed<T>::value)
        {
            polynomial_out[thread.global_address] =
                OPERATOR_GPU<TU>::centered_reduction(output1_reg,
                                                     thread.io_modulus);
            polynomial_out[thread.global_address + thread.global_offset] =
                OPERATOR_GPU<TU>::centered_reduction(output2_reg,
                                                     thread.io_modulus);
        }
        else
        {
            polynomial_out[thread.global_address] = output1_reg;
            polynomial_out[thread.global_address + thread.global_offset] =
                output2_reg;
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

        extern __shared__ char shared_memory_typed[];
        TU* shared_memory = reinterpret_cast<TU*>(shared_memory_typed);

        const TransposeThread<TU> thread =
            TransposeThreadState<false, false, TU>(SingleModulus<TU>{modulus},
                                                   log_row, log_column);
        TransposeLoadRoots(shared_memory, root_of_unity_table,
                           reduction_poly_check);
        TransposeLoad(thread, shared_memory, polynomial_in);
        ForwardTransposeExecute(thread, shared_memory, log_row,
                                reduction_poly_check);
        ForwardTransposeStore(thread, shared_memory, polynomial_out);
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

        extern __shared__ char shared_memory_typed[];
        TU* shared_memory = reinterpret_cast<TU*>(shared_memory_typed);

        const TransposeThread<TU> thread =
            TransposeThreadState<true, false, TU>(
                RnsModulus<TU>{modulus, mod_count}, log_row, log_column);
        TransposeLoadColumnRoots(thread, shared_memory, root_of_unity_table,
                                 log_row, reduction_poly_check);
        TransposeLoad(thread, shared_memory, polynomial_in);
        ForwardTransposeExecute(thread, shared_memory, log_row,
                                reduction_poly_check);
        ForwardTransposeStore(thread, shared_memory, polynomial_out);
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

        extern __shared__ char shared_memory_typed[];
        TU* shared_memory = reinterpret_cast<TU*>(shared_memory_typed);

        const TransposeThread<TU> thread =
            TransposeThreadState<false, true, TU>(
                SingleModulus<TU>{modulus, n_inverse}, log_row, log_column);
        TransposeLoadRoots(shared_memory, inverse_root_of_unity_table,
                           reduction_poly_check);
        TransposeLoad(thread, shared_memory, polynomial_in);
        InverseTransposeExecute(thread, shared_memory, log_row,
                                reduction_poly_check);
        InverseTransposeStore(thread, shared_memory, polynomial_out);
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

        extern __shared__ char shared_memory_typed[];
        TU* shared_memory = reinterpret_cast<TU*>(shared_memory_typed);

        const TransposeThread<TU> thread = TransposeThreadState<true, true, TU>(
            RnsModulus<TU>{modulus, mod_count, n_inverse}, log_row, log_column);
        TransposeLoadColumnRoots(thread, shared_memory,
                                 inverse_root_of_unity_table, log_row,
                                 reduction_poly_check);
        TransposeLoad(thread, shared_memory, polynomial_in);
        InverseTransposeExecute(thread, shared_memory, log_row,
                                reduction_poly_check);
        InverseTransposeStore(thread, shared_memory, polynomial_out);
    }

    // n_power must be in [1, 28]; the host API checks it before the lookup.
    template <typename T>
    const KernelStages& forward_kernel_configs(int n_power)
    {
        return forward_kernel_table<T>[n_power];
    }

    template <typename T>
    const KernelStages& inverse_kernel_configs(int n_power)
    {
        return inverse_kernel_table<T>[n_power];
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

        const KernelStages& kernels = forward_kernel_configs<TU>(n_power);
        const bool swap_last = (n_power >= 25);
        const bool x_n_minus =
            (reduction_poly == ReductionPolynomial::X_N_minus);

        for (int i = 0; i < kernels.size(); i++)
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
        const KernelStages& kernels = inverse_kernel_configs<TU>(n_power);
        const bool swap_first = (n_power >= 25);
        const bool x_n_minus =
            (reduction_poly == ReductionPolynomial::X_N_minus);

        for (int i = 0; i < kernels.size(); i++)
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
