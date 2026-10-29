// Copyright 2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

// Tests the device API (NTT().load / execute / store, the per stage
// NTT::stage<I>() functions and NTT::launch) against the host API: kernels
// written with a description must produce exactly the output of GPU_NTT /
// GPU_INTT on the same data. The data and root tables
// are random, so the comparison covers every butterfly. A fused kernel
// (NTT, pointwise product, INTT) is checked against the same steps run
// through the host API.

#include "gpuntt/ntt_merge/descriptor.cuh"
#include "gpuntt/ntt_merge/ntt.cuh"

#include <algorithm>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace gpuntt;

namespace
{
    int failures = 0;
    int cases = 0;
    std::mt19937_64 rng(2026);

    void check_cuda(cudaError_t error, const char* what)
    {
        if (error != cudaSuccess)
        {
            std::printf("CUDA error in %s: %s\n", what,
                        cudaGetErrorString(error));
            std::exit(1);
        }
    }

    template <typename T> struct DeviceBuffer
    {
        T* data = nullptr;
        size_t count = 0;

        explicit DeviceBuffer(size_t n) : count(n)
        {
            check_cuda(cudaMalloc(&data, n * sizeof(T)), "cudaMalloc");
        }

        explicit DeviceBuffer(const std::vector<T>& values)
            : DeviceBuffer(values.size())
        {
            check_cuda(cudaMemcpy(data, values.data(), count * sizeof(T),
                                  cudaMemcpyHostToDevice),
                       "cudaMemcpy");
        }

        ~DeviceBuffer() { cudaFree(data); }

        std::vector<T> read() const
        {
            std::vector<T> values(count);
            check_cuda(cudaDeviceSynchronize(), "kernel");
            check_cuda(cudaMemcpy(values.data(), data, count * sizeof(T),
                                  cudaMemcpyDeviceToHost),
                       "cudaMemcpy");
            return values;
        }
    };

    template <typename NTT, typename TIn>
    __global__ void forward_kernel(TIn* in, typename NTT::value_type* out,
                                   const typename NTT::value_type* roots,
                                   Modulus<typename NTT::value_type> modulus,
                                   int batch)
    {
        using T = typename NTT::value_type;
        extern __shared__ char shared_memory_raw[];
        T* shared_memory = reinterpret_cast<T*>(shared_memory_raw);

        const int first = blockIdx.x * NTT::batches_per_block;
        const int num_polys = min(NTT::batches_per_block, batch - first);
        const size_t offset = size_t(first) * NTT::size;

        NTT().load(shared_memory, in + offset, modulus, num_polys);
        NTT().execute(shared_memory, roots, modulus);
        NTT().store(shared_memory, out + offset, num_polys);
    }

    template <typename NTT, typename TOut>
    __global__ void inverse_kernel(typename NTT::value_type* in, TOut* out,
                                   const typename NTT::value_type* roots,
                                   Modulus<typename NTT::value_type> modulus,
                                   typename NTT::value_type n_inverse,
                                   int batch)
    {
        using T = typename NTT::value_type;
        extern __shared__ char shared_memory_raw[];
        T* shared_memory = reinterpret_cast<T*>(shared_memory_raw);

        const int first = blockIdx.x * NTT::batches_per_block;
        const int num_polys = min(NTT::batches_per_block, batch - first);
        const size_t offset = size_t(first) * NTT::size;

        NTT().load(shared_memory, in + offset, modulus, num_polys);
        NTT().execute(shared_memory, roots, modulus);
        NTT().store(shared_memory, out + offset, modulus, n_inverse, num_polys);
    }

    // Polynomial multiplication in one kernel: forward NTT of a, pointwise
    // product with b_hat (b in the NTT domain) in shared memory, inverse NTT.
    template <typename Fwd, typename Inv>
    __global__ void fused_multiply_kernel(
        typename Fwd::value_type* a, const typename Fwd::value_type* b_hat,
        typename Fwd::value_type* out, const typename Fwd::value_type* roots,
        const typename Fwd::value_type* inverse_roots,
        Modulus<typename Fwd::value_type> modulus,
        typename Fwd::value_type n_inverse, int batch)
    {
        using T = typename Fwd::value_type;
        extern __shared__ char shared_memory_raw[];
        T* shared_memory = reinterpret_cast<T*>(shared_memory_raw);

        const int first = blockIdx.x * Fwd::batches_per_block;
        const int num_polys = min(Fwd::batches_per_block, batch - first);
        const size_t offset = size_t(first) * Fwd::size;

        Fwd().load(shared_memory, a + offset, modulus, num_polys);
        Fwd().execute(shared_memory, roots, modulus);

        const int threads = blockDim.x * blockDim.y;
        for (int i = threadIdx.x + threadIdx.y * blockDim.x;
             i < num_polys * Fwd::size; i += threads)
        {
            shared_memory[i] = OPERATOR_GPU<T>::mult(
                shared_memory[i], b_hat[offset + i], modulus);
        }
        __syncthreads();

        Inv().execute(shared_memory, inverse_roots, modulus);
        Inv().store(shared_memory, out + offset, modulus, n_inverse, num_polys);
    }

    // The same with the helpers: shared_memory, block_data, num_polys and
    // for_each.
    template <typename Fwd, typename Inv>
    __global__ void fused_multiply_helpers_kernel(
        typename Fwd::value_type* a, const typename Fwd::value_type* b_hat,
        typename Fwd::value_type* out, const typename Fwd::value_type* roots,
        const typename Fwd::value_type* inverse_roots,
        Modulus<typename Fwd::value_type> modulus,
        typename Fwd::value_type n_inverse, int batch)
    {
        using T = typename Fwd::value_type;
        T* shared_memory = Fwd::shared_memory();
        const T* block_b_hat = Fwd::block_data(b_hat);
        const int num_polys = Fwd::num_polys(batch);

        Fwd().load(shared_memory, Fwd::block_data(a), modulus, num_polys);
        Fwd().execute(shared_memory, roots, modulus);
        Fwd().for_each(
            shared_memory,
            [&](T& value, size_t index) {
                value =
                    OPERATOR_GPU<T>::mult(value, block_b_hat[index], modulus);
            },
            num_polys);
        Inv().execute(shared_memory, inverse_roots, modulus);
        Inv().store(shared_memory, Fwd::block_data(out), modulus, n_inverse,
                    num_polys);
    }

    // The same with entries: each thread reads its two b_hat values into
    // registers before the forward execute and multiplies its own entries
    // after it.
    template <typename Fwd, typename Inv>
    __global__ void fused_multiply_entries_kernel(
        typename Fwd::value_type* a, const typename Fwd::value_type* b_hat,
        typename Fwd::value_type* out, const typename Fwd::value_type* roots,
        const typename Fwd::value_type* inverse_roots,
        Modulus<typename Fwd::value_type> modulus,
        typename Fwd::value_type n_inverse, int batch)
    {
        using T = typename Fwd::value_type;
        T* shared_memory = Fwd::shared_memory();
        const T* block_b_hat = Fwd::block_data(b_hat);
        const int num_polys = Fwd::num_polys(batch);
        const ThreadEntries entries = Fwd().entries(num_polys);

        Fwd().load(shared_memory, Fwd::block_data(a), modulus, num_polys);
        T b_values[ThreadEntries::count] = {};
        if (entries.valid)
        {
            for (int k = 0; k < ThreadEntries::count; k++)
            {
                b_values[k] = block_b_hat[entries.index[k]];
            }
        }
        Fwd().execute(shared_memory, roots, modulus);
        if (entries.valid)
        {
            for (int k = 0; k < ThreadEntries::count; k++)
            {
                T& value = shared_memory[entries.position[k]];
                value = OPERATOR_GPU<T>::mult(value, b_values[k], modulus);
            }
        }
        __syncthreads();
        Inv().execute(shared_memory, inverse_roots, modulus);
        Inv().store(shared_memory, Fwd::block_data(out), modulus, n_inverse,
                    num_polys);
    }

    // NTT(a) + NTT(b) in one kernel without extra shared memory: each thread
    // keeps its two entries of NTT(b) in registers while the same shared
    // memory transforms a.
    template <typename Fwd>
    __global__ void two_ntt_sum_kernel(
        const typename Fwd::value_type* a, const typename Fwd::value_type* b,
        typename Fwd::value_type* out, const typename Fwd::value_type* roots,
        Modulus<typename Fwd::value_type> modulus, int batch)
    {
        using T = typename Fwd::value_type;
        T* shared_memory = Fwd::shared_memory();
        const int num_polys = Fwd::num_polys(batch);
        const ThreadEntries entries = Fwd().entries(num_polys);

        Fwd().load(shared_memory, Fwd::block_data(b), modulus, num_polys);
        Fwd().execute(shared_memory, roots, modulus);
        T b_hat[ThreadEntries::count];
        for (int k = 0; k < ThreadEntries::count; k++)
        {
            b_hat[k] = shared_memory[entries.position[k]];
        }
        __syncthreads();

        Fwd().load(shared_memory, Fwd::block_data(a), modulus, num_polys);
        Fwd().execute(shared_memory, roots, modulus);
        for (int k = 0; k < ThreadEntries::count; k++)
        {
            T& value = shared_memory[entries.position[k]];
            value = OPERATOR_GPU<T>::add(value, b_hat[k], modulus);
        }
        __syncthreads();
        Fwd().store(shared_memory, Fwd::block_data(out), num_polys);
    }

    // Own use of shared memory: the block's polynomials are written to
    // shared memory one after another in the order of data instead of load,
    // and read back the same way instead of store. The inverse execute
    // leaves N * INTT(x), so reading it means multiplying by n_inverse.
    template <typename NTT>
    __global__ void own_io_kernel(const typename NTT::value_type* in,
                                  typename NTT::value_type* out,
                                  const typename NTT::value_type* roots,
                                  Modulus<typename NTT::value_type> modulus,
                                  typename NTT::value_type n_inverse, int batch)
    {
        using T = typename NTT::value_type;
        T* shared_memory = NTT::shared_memory();
        const T* block_in = NTT::block_data(in);
        T* block_out = NTT::block_data(out);
        const int count = NTT::num_polys(batch) * NTT::size;
        const int threads = blockDim.x * blockDim.y;
        const int thread = threadIdx.x + threadIdx.y * blockDim.x;

        for (int i = thread; i < count; i += threads)
        {
            shared_memory[i] = block_in[i];
        }
        // Other threads' entries were written, so execute needs a barrier.
        __syncthreads();
        NTT().execute(shared_memory, roots, modulus);
        for (int i = thread; i < count; i += threads)
        {
            block_out[i] = (NTT::direction == FORWARD)
                               ? shared_memory[i]
                               : OPERATOR_GPU<T>::mult(shared_memory[i],
                                                       n_inverse, modulus);
        }
    }

    // The same for stage I: shared memory is written and read through
    // for_each, which gives each entry with its index in data.
    template <typename NTT, int I>
    __global__ void
    own_io_stage_kernel(const typename NTT::value_type* in,
                        typename NTT::value_type* out,
                        const typename NTT::value_type* roots,
                        Modulus<typename NTT::value_type> modulus,
                        typename NTT::value_type n_inverse)
    {
        using T = typename NTT::value_type;
        using Stage = typename NTT::template stage<I>;
        T* shared_memory = Stage::shared_memory();

        Stage().for_each(shared_memory,
                         [&](T& value, size_t index) { value = in[index]; });
        Stage().execute(shared_memory, roots, modulus);
        Stage().for_each(shared_memory,
                         [&](T& value, size_t index)
                         {
                             out[index] =
                                 (NTT::direction == INVERSE && Stage::last)
                                     ? OPERATOR_GPU<T>::mult(value, n_inverse,
                                                             modulus)
                                     : value;
                         });
    }

    template <typename NTT>
    void launch_own_io_stages(const typename NTT::value_type* in,
                              typename NTT::value_type* out,
                              const typename NTT::value_type* roots,
                              Modulus<typename NTT::value_type> modulus,
                              typename NTT::value_type n_inverse, int batch)
    {
        NTT::for_each_stage(
            [&](auto I)
            {
                using Stage = typename NTT::template stage<I>;
                own_io_stage_kernel<NTT, I>
                    <<<Stage::grid(batch), Stage::block_dim,
                       Stage::shared_memory_size>>>(I == 0 ? in : out, out,
                                                    roots, modulus, n_inverse);
            });
        check_cuda(cudaGetLastError(), "launch");
    }

    // a * b in one kernel with b_hat copied to shared memory after the
    // description's shared_memory_size bytes; launched with the extra bytes.
    template <typename Fwd, typename Inv>
    __global__ void fused_multiply_extra_shared_kernel(
        typename Fwd::value_type* a, const typename Fwd::value_type* b_hat,
        typename Fwd::value_type* out, const typename Fwd::value_type* roots,
        const typename Fwd::value_type* inverse_roots,
        Modulus<typename Fwd::value_type> modulus,
        typename Fwd::value_type n_inverse, int batch)
    {
        using T = typename Fwd::value_type;
        T* shared_memory = Fwd::shared_memory();
        T* b_shared = shared_memory + Fwd::shared_memory_size / sizeof(T);
        const T* block_b_hat = Fwd::block_data(b_hat);
        const int num_polys = Fwd::num_polys(batch);
        const int count = num_polys * Fwd::size;

        for (int i = threadIdx.x + threadIdx.y * blockDim.x; i < count;
             i += blockDim.x * blockDim.y)
        {
            b_shared[i] = block_b_hat[i];
        }
        Fwd().load(shared_memory, Fwd::block_data(a), modulus, num_polys);
        // execute ends with a block barrier, so b_shared is complete after it.
        Fwd().execute(shared_memory, roots, modulus);
        Fwd().for_each(
            shared_memory,
            [&](T& value, size_t index)
            { value = OPERATOR_GPU<T>::mult(value, b_shared[index], modulus); },
            num_polys);
        Inv().execute(shared_memory, inverse_roots, modulus);
        Inv().store(shared_memory, Fwd::block_data(out), modulus, n_inverse,
                    num_polys);
    }

    // a * b in one kernel with both NTTs in shared memory: b is loaded into a
    // second buffer after the first (launched with 2 * shared_memory_size).
    // Both buffers keep an entry at the same position.
    template <typename Fwd, typename Inv>
    __global__ void fused_multiply_two_buffers_kernel(
        typename Fwd::value_type* a, const typename Fwd::value_type* b,
        typename Fwd::value_type* out, const typename Fwd::value_type* roots,
        const typename Fwd::value_type* inverse_roots,
        Modulus<typename Fwd::value_type> modulus,
        typename Fwd::value_type n_inverse, int batch)
    {
        using T = typename Fwd::value_type;
        T* a_shared = Fwd::shared_memory();
        T* b_shared = a_shared + Fwd::shared_memory_size / sizeof(T);
        const int num_polys = Fwd::num_polys(batch);
        const ThreadEntries entries = Fwd().entries(num_polys);

        Fwd().load(a_shared, Fwd::block_data(a), modulus, num_polys);
        Fwd().load(b_shared, Fwd::block_data(b), modulus, num_polys);
        Fwd().execute(a_shared, roots, modulus);
        Fwd().execute(b_shared, roots, modulus);
        if (entries.valid)
        {
            for (int k = 0; k < ThreadEntries::count; k++)
            {
                T& value = a_shared[entries.position[k]];
                value = OPERATOR_GPU<T>::mult(
                    value, b_shared[entries.position[k]], modulus);
            }
        }
        __syncthreads();
        Inv().execute(a_shared, inverse_roots, modulus);
        Inv().store(a_shared, Fwd::block_data(out), modulus, n_inverse,
                    num_polys);
    }

    template <typename T>
    __global__ void multiply_kernel(T* a, const T* b, Modulus<T> modulus,
                                    size_t total)
    {
        const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i < total)
        {
            a[i] = OPERATOR_GPU<T>::mult(a[i], b[i], modulus);
        }
    }

    // Kernel I of a multi-kernel NTT. Kernel 0 reads the input; the others
    // read and write the output in place. The last inverse kernel scales by
    // N^-1 and may write signed output.
    template <typename NTT, int I, typename TIn, typename TOut>
    __global__ void stage_kernel(TIn* in, TOut* out,
                                 const typename NTT::value_type* roots,
                                 Modulus<typename NTT::value_type> modulus,
                                 typename NTT::value_type n_inverse)
    {
        using T = typename NTT::value_type;
        using Stage = typename NTT::template stage<I>;
        extern __shared__ char shared_memory_raw[];
        T* shared_memory = reinterpret_cast<T*>(shared_memory_raw);

        Stage().load(shared_memory, in, modulus);
        Stage().execute(shared_memory, roots, modulus);
        if constexpr (NTT::direction == INVERSE && Stage::last)
        {
            Stage().store(shared_memory, out, modulus, n_inverse);
        }
        else
        {
            Stage().store(shared_memory, out);
        }
    }

    template <typename NTT, int I, typename TIn, typename TOut>
    void launch_stage(TIn* in, TOut* out, const typename NTT::value_type* roots,
                      Modulus<typename NTT::value_type> modulus,
                      typename NTT::value_type n_inverse, int batch)
    {
        using T = typename NTT::value_type;
        using Stage = typename NTT::template stage<I>;
        T* out_unsigned = reinterpret_cast<T*>(out);
        auto stage_in = [&]
        {
            if constexpr (I == 0)
            {
                return in;
            }
            else
            {
                return out_unsigned;
            }
        }();
        auto stage_out = [&]
        {
            if constexpr (NTT::direction == INVERSE && Stage::last)
            {
                return out;
            }
            else
            {
                return out_unsigned;
            }
        }();
        stage_kernel<NTT, I><<<Stage::grid(batch), Stage::block_dim,
                               Stage::shared_memory_size>>>(
            stage_in, stage_out, roots, modulus, n_inverse);
        check_cuda(cudaGetLastError(), "launch");
    }

    // Forward stage I with the shared_memory helper; launched by
    // for_each_stage.
    template <typename NTT, int I>
    __global__ void
    forward_stage_helpers_kernel(const typename NTT::value_type* in,
                                 typename NTT::value_type* out,
                                 const typename NTT::value_type* roots,
                                 Modulus<typename NTT::value_type> modulus)
    {
        using Stage = typename NTT::template stage<I>;
        typename NTT::value_type* shared_memory = Stage::shared_memory();
        Stage().load(shared_memory, in, modulus);
        Stage().execute(shared_memory, roots, modulus);
        Stage().store(shared_memory, out);
    }

    // The last forward stage, a pointwise product with b_hat and the first
    // inverse stage in one kernel: the product runs on the entries of
    // for_each, without a store and load in between.
    template <typename Fwd, typename Inv>
    __global__ void
    fused_stage_kernel(typename Fwd::value_type* data,
                       const typename Fwd::value_type* b_hat,
                       const typename Fwd::value_type* roots,
                       const typename Fwd::value_type* inverse_roots,
                       Modulus<typename Fwd::value_type> modulus,
                       typename Fwd::value_type n_inverse)
    {
        using T = typename Fwd::value_type;
        using Last = typename Fwd::template stage<Fwd::num_stages - 1>;
        using First = typename Inv::template stage<0>;
        extern __shared__ char shared_memory_raw[];
        T* shared_memory = reinterpret_cast<T*>(shared_memory_raw);

        Last().load(shared_memory, data, modulus);
        Last().execute(shared_memory, roots, modulus);
        Last().for_each(
            shared_memory, [&](T& value, size_t index)
            { value = OPERATOR_GPU<T>::mult(value, b_hat[index], modulus); });
        First().execute(shared_memory, inverse_roots, modulus);
        if constexpr (First::last)
        {
            First().store(shared_memory, data, modulus, n_inverse);
        }
        else
        {
            First().store(shared_memory, data);
        }
    }

    // The same with two shared memory buffers: the last forward stage of a
    // and of b, their product and the first inverse stage of the product.
    template <typename Fwd, typename Inv>
    __global__ void fused_stage_two_buffers_kernel(
        typename Fwd::value_type* a, const typename Fwd::value_type* b,
        const typename Fwd::value_type* roots,
        const typename Fwd::value_type* inverse_roots,
        Modulus<typename Fwd::value_type> modulus,
        typename Fwd::value_type n_inverse)
    {
        using T = typename Fwd::value_type;
        using Last = typename Fwd::template stage<Fwd::num_stages - 1>;
        using First = typename Inv::template stage<0>;
        T* a_shared = Last::shared_memory();
        T* b_shared = a_shared + Last::shared_memory_size / sizeof(T);
        const ThreadEntries entries = Last().entries();

        Last().load(a_shared, a, modulus);
        Last().load(b_shared, b, modulus);
        Last().execute(a_shared, roots, modulus);
        Last().execute(b_shared, roots, modulus);
        for (int k = 0; k < ThreadEntries::count; k++)
        {
            T& value = a_shared[entries.position[k]];
            value = OPERATOR_GPU<T>::mult(value, b_shared[entries.position[k]],
                                          modulus);
        }
        __syncthreads();
        First().execute(a_shared, inverse_roots, modulus);
        if constexpr (First::last)
        {
            First().store(a_shared, a, modulus, n_inverse);
        }
        else
        {
            First().store(a_shared, a);
        }
    }

    // The same with entries: the b_hat values are read into registers before
    // the last forward execute.
    template <typename Fwd, typename Inv>
    __global__ void
    fused_stage_entries_kernel(typename Fwd::value_type* data,
                               const typename Fwd::value_type* b_hat,
                               const typename Fwd::value_type* roots,
                               const typename Fwd::value_type* inverse_roots,
                               Modulus<typename Fwd::value_type> modulus,
                               typename Fwd::value_type n_inverse)
    {
        using T = typename Fwd::value_type;
        using Last = typename Fwd::template stage<Fwd::num_stages - 1>;
        using First = typename Inv::template stage<0>;
        T* shared_memory = Last::shared_memory();
        const ThreadEntries entries = Last().entries();

        Last().load(shared_memory, data, modulus);
        T b_values[ThreadEntries::count];
        for (int k = 0; k < ThreadEntries::count; k++)
        {
            b_values[k] = b_hat[entries.index[k]];
        }
        Last().execute(shared_memory, roots, modulus);
        for (int k = 0; k < ThreadEntries::count; k++)
        {
            T& value = shared_memory[entries.position[k]];
            value = OPERATOR_GPU<T>::mult(value, b_values[k], modulus);
        }
        __syncthreads();
        First().execute(shared_memory, inverse_roots, modulus);
        if constexpr (First::last)
        {
            First().store(shared_memory, data, modulus, n_inverse);
        }
        else
        {
            First().store(shared_memory, data);
        }
    }

    template <typename NTT, typename TIn, typename TOut, int... I>
    void launch_stages(TIn* in, TOut* out,
                       const typename NTT::value_type* roots,
                       Modulus<typename NTT::value_type> modulus,
                       typename NTT::value_type n_inverse, int batch,
                       std::integer_sequence<int, I...>)
    {
        (launch_stage<NTT, I>(in, out, roots, modulus, n_inverse, batch), ...);
    }

    // Kernels First, First + 1, ... of a multi-kernel NTT.
    template <typename NTT, int First, typename TIn, typename TOut, int... I>
    void launch_stages_from(TIn* in, TOut* out,
                            const typename NTT::value_type* roots,
                            Modulus<typename NTT::value_type> modulus,
                            typename NTT::value_type n_inverse, int batch,
                            std::integer_sequence<int, I...>)
    {
        (launch_stage<NTT, First + I>(in, out, roots, modulus, n_inverse,
                                      batch),
         ...);
    }

    // RNS: polynomial z uses modulus z % mod_count and the roots at
    // roots + (prime << log2(N)), picked by the kernel itself.
    template <typename NTT>
    __global__ void
    rns_single_kernel(typename NTT::value_type* in,
                      typename NTT::value_type* out,
                      const typename NTT::value_type* roots,
                      const Modulus<typename NTT::value_type>* moduli,
                      const typename NTT::value_type* n_inverses, int mod_count)
    {
        using T = typename NTT::value_type;
        static_assert(NTT::batches_per_block == 1);
        extern __shared__ char shared_memory_raw[];
        T* shared_memory = reinterpret_cast<T*>(shared_memory_raw);

        const int prime = blockIdx.x % mod_count;
        const Modulus<T> modulus = moduli[prime];
        const T* prime_roots = roots + (size_t(prime) << NTT::log_size);
        const size_t offset = size_t(blockIdx.x) * NTT::size;

        NTT().load(shared_memory, in + offset, modulus);
        NTT().execute(shared_memory, prime_roots, modulus);
        if constexpr (NTT::direction == FORWARD)
        {
            NTT().store(shared_memory, out + offset);
        }
        else
        {
            NTT().store(shared_memory, out + offset, modulus,
                        n_inverses[prime]);
        }
    }

    template <typename NTT, int I>
    __global__ void
    rns_stage_kernel(typename NTT::value_type* in,
                     typename NTT::value_type* out,
                     const typename NTT::value_type* roots,
                     const Modulus<typename NTT::value_type>* moduli,
                     const typename NTT::value_type* n_inverses, int mod_count)
    {
        using T = typename NTT::value_type;
        using Stage = typename NTT::template stage<I>;
        extern __shared__ char shared_memory_raw[];
        T* shared_memory = reinterpret_cast<T*>(shared_memory_raw);

        const int prime = blockIdx.z % mod_count;
        const Modulus<T> modulus = moduli[prime];
        const T* prime_roots = roots + (size_t(prime) << NTT::log_size);

        Stage().load(shared_memory, in, modulus);
        Stage().execute(shared_memory, prime_roots, modulus);
        if constexpr (NTT::direction == INVERSE && Stage::last)
        {
            Stage().store(shared_memory, out, modulus, n_inverses[prime]);
        }
        else
        {
            Stage().store(shared_memory, out);
        }
    }

    template <typename NTT, int... I>
    void launch_rns_stages(typename NTT::value_type* in,
                           typename NTT::value_type* out,
                           const typename NTT::value_type* roots,
                           const Modulus<typename NTT::value_type>* moduli,
                           const typename NTT::value_type* n_inverses,
                           int mod_count, int batch,
                           std::integer_sequence<int, I...>)
    {
        ((rns_stage_kernel<NTT, I>
          <<<NTT::template stage<I>::grid(batch),
             NTT::template stage<I>::block_dim,
             NTT::template stage<I>::shared_memory_size>>>(
              I == 0 ? in : out, out, roots, moduli, n_inverses, mod_count)),
         ...);
        check_cuda(cudaGetLastError(), "launch");
    }

    template <typename T> std::vector<T> random_below(size_t n, T bound)
    {
        std::vector<T> values(n);
        for (T& v : values)
        {
            v = static_cast<T>(rng() % bound);
        }
        return values;
    }

    // Block(), with BlockDim<Threads>() if Threads is not 0.
    template <int Threads> constexpr auto block_operators()
    {
        if constexpr (Threads == 0)
        {
            return Block();
        }
        else
        {
            return Block() + BlockDim<Threads>();
        }
    }

    // Runs one description on random data and compares it with the host
    // API. TIO is the signed or unsigned type of the input (forward) or the
    // output (inverse). Threads selects BlockDim<Threads> (0: none).
    template <bool Staged, typename T, typename TIO, int LogN, type D,
              ReductionPolynomial R, int Threads = 0>
    void run_case()
    {
        using NTT = decltype(Size<(1 << LogN)>() + Direction<D>() +
                             Precision<T>() + ReductionPoly<R>() +
                             block_operators<Threads>());
        constexpr int n = 1 << LogN;
        const int batch = 2 * NTT::batches_per_block + 1;
        const size_t total = size_t(batch) * n;

        const int max_bits = std::is_same<T, Data32>::value ? 30 : 61;
        const T top = T(1) << (max_bits - 1);
        const T p = top + (static_cast<T>(rng() % top) | 1);
        const Modulus<T> modulus(p);
        // NTT::launch multiplies by n_inv(modulus).
        const T n_inverse = NTT::n_inv(modulus);

        DeviceBuffer<T> roots(random_below<T>(n, p));
        std::vector<T> input = random_below<T>(total, p);

        ntt_configuration<T> cfg = {.n_power = LogN,
                                    .ntt_type = D,
                                    .ntt_layout = PerPolynomial,
                                    .reduction_poly = R,
                                    .mod_inverse = n_inverse,
                                    .stream = 0};

        std::vector<TIO> expected, actual, launched, in_place, own_io;
        if constexpr (D == FORWARD)
        {
            std::vector<TIO> signed_input(total);
            for (size_t i = 0; i < total; i++)
            {
                signed_input[i] = std::is_signed<TIO>::value
                                      ? static_cast<TIO>(input[i] % p) -
                                            static_cast<TIO>(p >> 1)
                                      : static_cast<TIO>(input[i]);
            }
            DeviceBuffer<TIO> in(signed_input);
            DeviceBuffer<T> reference(total), out(total);
            GPU_NTT(in.data, reference.data, roots.data, modulus, cfg, batch);
            if constexpr (Staged)
            {
                launch_stages<NTT>(
                    in.data, out.data, roots.data, modulus, n_inverse, batch,
                    std::make_integer_sequence<int, NTT::num_stages>());
            }
            else
            {
                forward_kernel<NTT><<<NTT::grid(batch), NTT::block_dim,
                                      NTT::shared_memory_size>>>(
                    in.data, out.data, roots.data, modulus, batch);
                check_cuda(cudaGetLastError(), "launch");
            }
            const std::vector<T> e = reference.read(), a = out.read();
            expected.assign(e.begin(), e.end());
            actual.assign(a.begin(), a.end());

            DeviceBuffer<T> out_launch(total);
            NTT::launch(in.data, out_launch.data, roots.data, modulus, batch);
            const std::vector<T> l = out_launch.read();
            launched.assign(l.begin(), l.end());
            if constexpr (std::is_same<TIO, T>::value)
            {
                DeviceBuffer<T> inout(input);
                NTT::launch(inout.data, inout.data, roots.data, modulus, batch);
                in_place = inout.read();
            }
        }
        else
        {
            DeviceBuffer<T> in(input);
            DeviceBuffer<TIO> reference(total), out(total);
            GPU_INTT(in.data, reference.data, roots.data, modulus, cfg, batch);
            if constexpr (Staged)
            {
                launch_stages<NTT>(
                    in.data, out.data, roots.data, modulus, n_inverse, batch,
                    std::make_integer_sequence<int, NTT::num_stages>());
            }
            else
            {
                inverse_kernel<NTT><<<NTT::grid(batch), NTT::block_dim,
                                      NTT::shared_memory_size>>>(
                    in.data, out.data, roots.data, modulus, n_inverse, batch);
                check_cuda(cudaGetLastError(), "launch");
            }
            expected = reference.read();
            actual = out.read();

            DeviceBuffer<TIO> out_launch(total);
            NTT::launch(in.data, out_launch.data, roots.data, modulus, batch);
            launched = out_launch.read();
            if constexpr (std::is_same<TIO, T>::value)
            {
                DeviceBuffer<T> inout(input);
                NTT::launch(inout.data, inout.data, roots.data, modulus, batch);
                in_place = inout.read();
            }
        }

        // The kernel writes and reads shared memory itself instead of load
        // and store.
        if constexpr (std::is_same<TIO, T>::value)
        {
            DeviceBuffer<T> in(input), out(total);
            if constexpr (Staged)
            {
                launch_own_io_stages<NTT>(in.data, out.data, roots.data,
                                          modulus, n_inverse, batch);
            }
            else
            {
                own_io_kernel<NTT><<<NTT::grid(batch), NTT::block_dim,
                                     NTT::shared_memory_size>>>(
                    in.data, out.data, roots.data, modulus, n_inverse, batch);
                check_cuda(cudaGetLastError(), "launch");
            }
            own_io = out.read();
        }

        const char* const variants[] = {"", " launch", " launch in place",
                                        " own shared memory"};
        const std::vector<TIO>* results[] = {&actual, &launched, &in_place,
                                             &own_io};
        for (int v = 0; v < 4; v++)
        {
            // Signed data has no in place launch and no own shared memory
            // variant.
            if (v >= 2 && !std::is_same<TIO, T>::value)
            {
                continue;
            }
            cases++;
            if (*results[v] != expected)
            {
                failures++;
                std::printf(
                    "FAIL %s%s %s logn=%d threads=%d %s %s%s batch=%d\n",
                    Staged ? "stages" : "single kernel", variants[v],
                    sizeof(T) == 4 ? "u32" : "u64", LogN, Threads,
                    D == FORWARD ? "forward" : "inverse",
                    R == X_N_minus ? "X^N-1" : "X^N+1",
                    std::is_signed<TIO>::value ? " signed" : "", batch);
            }
        }
    }

    // The fused kernel must match GPU_NTT, a pointwise product and GPU_INTT
    // run as separate steps.
    template <typename T, int LogN, ReductionPolynomial R> void run_fused()
    {
        using Fwd = decltype(Size<(1 << LogN)>() + Direction<FORWARD>() +
                             Precision<T>() + ReductionPoly<R>() + Block());
        using Inv = decltype(Size<(1 << LogN)>() + Direction<INVERSE>() +
                             Precision<T>() + ReductionPoly<R>() + Block());
        static_assert(Fwd::batches_per_block == Inv::batches_per_block &&
                      Fwd::shared_memory_size == Inv::shared_memory_size);

        constexpr int n = 1 << LogN;
        const int batch = 2 * Fwd::batches_per_block + 1;
        const size_t total = size_t(batch) * n;

        const int max_bits = std::is_same<T, Data32>::value ? 30 : 61;
        const T top = T(1) << (max_bits - 1);
        const T p = top + (static_cast<T>(rng() % top) | 1);
        const Modulus<T> modulus(p);
        const T n_inverse = static_cast<T>(rng() % p);

        DeviceBuffer<T> roots(random_below<T>(n, p));
        DeviceBuffer<T> inverse_roots(random_below<T>(n, p));
        DeviceBuffer<T> a(random_below<T>(total, p));
        DeviceBuffer<T> b(random_below<T>(total, p));

        ntt_configuration<T> fwd_cfg = {.n_power = LogN,
                                        .ntt_type = FORWARD,
                                        .ntt_layout = PerPolynomial,
                                        .reduction_poly = R,
                                        .mod_inverse = n_inverse,
                                        .stream = 0};
        ntt_configuration<T> inv_cfg = fwd_cfg;
        inv_cfg.ntt_type = INVERSE;

        DeviceBuffer<T> b_hat(total), a_hat(total), expected(total),
            actual(total);
        GPU_NTT(b.data, b_hat.data, roots.data, modulus, fwd_cfg, batch);
        GPU_NTT(a.data, a_hat.data, roots.data, modulus, fwd_cfg, batch);
        multiply_kernel<<<(total + 255) / 256, 256>>>(a_hat.data, b_hat.data,
                                                      modulus, total);
        GPU_INTT(a_hat.data, expected.data, inverse_roots.data, modulus,
                 inv_cfg, batch);

        fused_multiply_kernel<Fwd, Inv>
            <<<Fwd::grid(batch), Fwd::block_dim, Fwd::shared_memory_size>>>(
                a.data, b_hat.data, actual.data, roots.data, inverse_roots.data,
                modulus, n_inverse, batch);
        check_cuda(cudaGetLastError(), "launch");

        DeviceBuffer<T> with_helpers(total);
        fused_multiply_helpers_kernel<Fwd, Inv>
            <<<Fwd::grid(batch), Fwd::block_dim, Fwd::shared_memory_size>>>(
                a.data, b_hat.data, with_helpers.data, roots.data,
                inverse_roots.data, modulus, n_inverse, batch);
        check_cuda(cudaGetLastError(), "launch");

        DeviceBuffer<T> with_entries(total);
        fused_multiply_entries_kernel<Fwd, Inv>
            <<<Fwd::grid(batch), Fwd::block_dim, Fwd::shared_memory_size>>>(
                a.data, b_hat.data, with_entries.data, roots.data,
                inverse_roots.data, modulus, n_inverse, batch);
        check_cuda(cudaGetLastError(), "launch");

        DeviceBuffer<T> with_extra_shared(total);
        fused_multiply_extra_shared_kernel<Fwd, Inv>
            <<<Fwd::grid(batch), Fwd::block_dim,
               Fwd::shared_memory_size +
                   Fwd::batches_per_block * Fwd::size * sizeof(T)>>>(
                a.data, b_hat.data, with_extra_shared.data, roots.data,
                inverse_roots.data, modulus, n_inverse, batch);
        check_cuda(cudaGetLastError(), "launch");

        DeviceBuffer<T> with_two_buffers(total);
        fused_multiply_two_buffers_kernel<Fwd, Inv>
            <<<Fwd::grid(batch), Fwd::block_dim, 2 * Fwd::shared_memory_size>>>(
                a.data, b.data, with_two_buffers.data, roots.data,
                inverse_roots.data, modulus, n_inverse, batch);
        check_cuda(cudaGetLastError(), "launch");

        const std::vector<T> expected_values = expected.read();
        const char* const variants[] = {"", " with helpers", " with entries",
                                        " with extra shared memory",
                                        " with two buffers"};
        const std::vector<T> results[] = {
            actual.read(), with_helpers.read(), with_entries.read(),
            with_extra_shared.read(), with_two_buffers.read()};
        for (int v = 0; v < 5; v++)
        {
            cases++;
            if (results[v] != expected_values)
            {
                failures++;
                std::printf("FAIL fused multiply%s %s logn=%d %s batch=%d\n",
                            variants[v], sizeof(T) == 4 ? "u32" : "u64", LogN,
                            R == X_N_minus ? "X^N-1" : "X^N+1", batch);
            }
        }
    }

    // The single kernel API needs one polynomial (one modulus) per block.
    template <bool Staged, typename Base> struct RnsDescription
    {
        using type = Base;
    };

    template <typename Base> struct RnsDescription<false, Base>
    {
        using type = decltype(Base() + BatchesPerBlock<1>());
    };

    // The RNS usage described in the README must match the host API's RNS
    // variant (mod_count moduli, polynomial z uses modulus z % mod_count).
    template <bool Staged, typename T, int LogN, type D, ReductionPolynomial R>
    void run_rns()
    {
        using Base = decltype(Size<(1 << LogN)>() + Direction<D>() +
                              Precision<T>() + ReductionPoly<R>() + Block());
        using NTT = typename RnsDescription<Staged, Base>::type;
        constexpr int n = 1 << LogN;
        const int mod_count = 3;
        const int batch = 2 * mod_count + 1;
        const size_t total = size_t(batch) * n;

        const int max_bits = std::is_same<T, Data32>::value ? 30 : 61;
        const T top = T(1) << (max_bits - 1);
        std::vector<Modulus<T>> moduli;
        std::vector<T> n_inverses;
        T p_min = ~T(0);
        for (int i = 0; i < mod_count; i++)
        {
            const T p = top + (static_cast<T>(rng() % top) | 1);
            moduli.push_back(Modulus<T>(p));
            n_inverses.push_back(static_cast<T>(rng() % p));
            p_min = std::min(p_min, p);
        }

        DeviceBuffer<Modulus<T>> moduli_device(moduli);
        DeviceBuffer<T> n_inverses_device(n_inverses);
        DeviceBuffer<T> roots(random_below<T>(size_t(mod_count) * n, p_min));
        DeviceBuffer<T> in(random_below<T>(total, p_min));
        DeviceBuffer<T> expected(total), actual(total);

        ntt_rns_configuration<T> cfg = {.n_power = LogN,
                                        .ntt_type = D,
                                        .ntt_layout = PerPolynomial,
                                        .reduction_poly = R,
                                        .mod_inverse = n_inverses_device.data,
                                        .stream = 0};
        if constexpr (D == FORWARD)
        {
            GPU_NTT(in.data, expected.data, roots.data, moduli_device.data, cfg,
                    batch, mod_count);
        }
        else
        {
            GPU_INTT(in.data, expected.data, roots.data, moduli_device.data,
                     cfg, batch, mod_count);
        }

        if constexpr (Staged)
        {
            launch_rns_stages<NTT>(
                in.data, actual.data, roots.data, moduli_device.data,
                n_inverses_device.data, mod_count, batch,
                std::make_integer_sequence<int, NTT::num_stages>());
        }
        else
        {
            rns_single_kernel<NTT>
                <<<NTT::grid(batch), NTT::block_dim, NTT::shared_memory_size>>>(
                    in.data, actual.data, roots.data, moduli_device.data,
                    n_inverses_device.data, mod_count);
            check_cuda(cudaGetLastError(), "launch");
        }

        cases++;
        if (expected.read() != actual.read())
        {
            failures++;
            std::printf("FAIL rns %s %s logn=%d %s %s\n",
                        Staged ? "stages" : "single kernel",
                        sizeof(T) == 4 ? "u32" : "u64", LogN,
                        D == FORWARD ? "forward" : "inverse",
                        R == X_N_minus ? "X^N-1" : "X^N+1");
        }
    }

    template <typename T, typename TIO, int LogN, bool Staged = false,
              int Threads = 0>
    void run_size()
    {
        run_case<Staged, T, TIO, LogN, FORWARD, X_N_minus, Threads>();
        run_case<Staged, T, TIO, LogN, FORWARD, X_N_plus, Threads>();
        run_case<Staged, T, TIO, LogN, INVERSE, X_N_minus, Threads>();
        run_case<Staged, T, TIO, LogN, INVERSE, X_N_plus, Threads>();
    }

    template <typename T, int First, bool Staged, int... LogN>
    void run_sizes(std::integer_sequence<int, LogN...>)
    {
        (run_size<T, T, First + LogN, Staged>(), ...);
    }

    // The same product for ring sizes with stage kernels: forward stages
    // 0 ... S - 2, fused_stage_kernel, inverse stages 1 ... S - 1.
    template <typename T, int LogN, ReductionPolynomial R, int Threads = 0>
    void run_fused_stages(int batch)
    {
        using Fwd = decltype(Size<(1 << LogN)>() + Direction<FORWARD>() +
                             Precision<T>() + ReductionPoly<R>() +
                             block_operators<Threads>());
        using Inv = decltype(Size<(1 << LogN)>() + Direction<INVERSE>() +
                             Precision<T>() + ReductionPoly<R>() +
                             block_operators<Threads>());
        using Last = typename Fwd::template stage<Fwd::num_stages - 1>;
        static_assert(Fwd::num_stages >= 2);

        constexpr int n = 1 << LogN;
        const size_t total = size_t(batch) * n;

        const int max_bits = std::is_same<T, Data32>::value ? 30 : 61;
        const T top = T(1) << (max_bits - 1);
        const T p = top + (static_cast<T>(rng() % top) | 1);
        const Modulus<T> modulus(p);
        const T n_inverse = static_cast<T>(rng() % p);

        DeviceBuffer<T> roots(random_below<T>(n, p));
        DeviceBuffer<T> inverse_roots(random_below<T>(n, p));
        DeviceBuffer<T> a(random_below<T>(total, p));
        DeviceBuffer<T> b_hat(random_below<T>(total, p));

        ntt_configuration<T> fwd_cfg = {.n_power = LogN,
                                        .ntt_type = FORWARD,
                                        .ntt_layout = PerPolynomial,
                                        .reduction_poly = R,
                                        .mod_inverse = n_inverse,
                                        .stream = 0};
        ntt_configuration<T> inv_cfg = fwd_cfg;
        inv_cfg.ntt_type = INVERSE;

        // Forward NTT with all stages launched by for_each_stage.
        DeviceBuffer<T> expected(total), actual(total), staged(total);
        GPU_NTT(a.data, expected.data, roots.data, modulus, fwd_cfg, batch);
        Fwd::for_each_stage(
            [&](auto I)
            {
                using Stage = typename Fwd::template stage<I>;
                forward_stage_helpers_kernel<Fwd, I>
                    <<<Stage::grid(batch), Stage::block_dim,
                       Stage::shared_memory_size>>>(
                        I == 0 ? a.data : staged.data, staged.data, roots.data,
                        modulus);
            });
        check_cuda(cudaGetLastError(), "launch");
        cases++;
        if (expected.read() != staged.read())
        {
            failures++;
            std::printf("FAIL for_each_stage %s logn=%d threads=%d %s "
                        "batch=%d\n",
                        sizeof(T) == 4 ? "u32" : "u64", LogN, Threads,
                        R == X_N_minus ? "X^N-1" : "X^N+1", batch);
        }
        multiply_kernel<<<(total + 255) / 256, 256>>>(expected.data, b_hat.data,
                                                      modulus, total);
        GPU_INTT_Inplace(expected.data, inverse_roots.data, modulus, inv_cfg,
                         batch);

        launch_stages_from<Fwd, 0>(
            a.data, actual.data, roots.data, modulus, n_inverse, batch,
            std::make_integer_sequence<int, Fwd::num_stages - 1>());
        fused_stage_kernel<Fwd, Inv>
            <<<Last::grid(batch), Last::block_dim, Last::shared_memory_size>>>(
                actual.data, b_hat.data, roots.data, inverse_roots.data,
                modulus, n_inverse);
        check_cuda(cudaGetLastError(), "launch");
        launch_stages_from<Inv, 1>(
            actual.data, actual.data, inverse_roots.data, modulus, n_inverse,
            batch, std::make_integer_sequence<int, Inv::num_stages - 1>());

        DeviceBuffer<T> with_entries(total);
        launch_stages_from<Fwd, 0>(
            a.data, with_entries.data, roots.data, modulus, n_inverse, batch,
            std::make_integer_sequence<int, Fwd::num_stages - 1>());
        fused_stage_entries_kernel<Fwd, Inv>
            <<<Last::grid(batch), Last::block_dim, Last::shared_memory_size>>>(
                with_entries.data, b_hat.data, roots.data, inverse_roots.data,
                modulus, n_inverse);
        check_cuda(cudaGetLastError(), "launch");
        launch_stages_from<Inv, 1>(
            with_entries.data, with_entries.data, inverse_roots.data, modulus,
            n_inverse, batch,
            std::make_integer_sequence<int, Inv::num_stages - 1>());

        const std::vector<T> expected_values = expected.read();
        const char* const variants[] = {"", " with entries"};
        const std::vector<T> results[] = {actual.read(), with_entries.read()};
        for (int v = 0; v < 2; v++)
        {
            cases++;
            if (results[v] != expected_values)
            {
                failures++;
                std::printf("FAIL fused stages%s %s logn=%d threads=%d %s "
                            "batch=%d\n",
                            variants[v], sizeof(T) == 4 ? "u32" : "u64", LogN,
                            Threads, R == X_N_minus ? "X^N-1" : "X^N+1",
                            batch);
            }
        }
    }

    // NTT(a) + NTT(b) with two_ntt_sum_kernel against two GPU_NTT calls.
    template <typename T, int LogN, ReductionPolynomial R> void run_two_ntts()
    {
        using Fwd = decltype(Size<(1 << LogN)>() + Direction<FORWARD>() +
                             Precision<T>() + ReductionPoly<R>() + Block());
        constexpr int n = 1 << LogN;
        const int batch = 2 * Fwd::batches_per_block + 1;
        const size_t total = size_t(batch) * n;

        const int max_bits = std::is_same<T, Data32>::value ? 30 : 61;
        const T top = T(1) << (max_bits - 1);
        const T p = top + (static_cast<T>(rng() % top) | 1);
        const Modulus<T> modulus(p);

        DeviceBuffer<T> roots(random_below<T>(n, p));
        DeviceBuffer<T> a(random_below<T>(total, p));
        DeviceBuffer<T> b(random_below<T>(total, p));
        DeviceBuffer<T> a_hat(total), b_hat(total), actual(total);

        ntt_configuration<T> cfg = {.n_power = LogN,
                                    .ntt_type = FORWARD,
                                    .ntt_layout = PerPolynomial,
                                    .reduction_poly = R,
                                    .mod_inverse = 1,
                                    .stream = 0};
        GPU_NTT(a.data, a_hat.data, roots.data, modulus, cfg, batch);
        GPU_NTT(b.data, b_hat.data, roots.data, modulus, cfg, batch);
        two_ntt_sum_kernel<Fwd>
            <<<Fwd::grid(batch), Fwd::block_dim, Fwd::shared_memory_size>>>(
                a.data, b.data, actual.data, roots.data, modulus, batch);
        check_cuda(cudaGetLastError(), "launch");

        const std::vector<T> a_values = a_hat.read();
        const std::vector<T> b_values = b_hat.read();
        std::vector<T> expected(total);
        for (size_t i = 0; i < total; i++)
        {
            expected[i] = OPERATOR<T>::add(a_values[i], b_values[i], modulus);
        }

        cases++;
        if (actual.read() != expected)
        {
            failures++;
            std::printf("FAIL two NTTs with entries %s logn=%d %s batch=%d\n",
                        sizeof(T) == 4 ? "u32" : "u64", LogN,
                        R == X_N_minus ? "X^N-1" : "X^N+1", batch);
        }
    }

    // a * b with stage kernels and two shared memory buffers in the middle
    // kernel (fused_stage_two_buffers_kernel), against GPU_NTT of a and b,
    // a pointwise product and GPU_INTT.
    template <typename T, int LogN, ReductionPolynomial R>
    void run_two_buffers_stages(int batch)
    {
        using Fwd = decltype(Size<(1 << LogN)>() + Direction<FORWARD>() +
                             Precision<T>() + ReductionPoly<R>() + Block());
        using Inv = decltype(Size<(1 << LogN)>() + Direction<INVERSE>() +
                             Precision<T>() + ReductionPoly<R>() + Block());
        using Last = typename Fwd::template stage<Fwd::num_stages - 1>;
        static_assert(Fwd::num_stages >= 2);

        constexpr int n = 1 << LogN;
        const size_t total = size_t(batch) * n;

        const int max_bits = std::is_same<T, Data32>::value ? 30 : 61;
        const T top = T(1) << (max_bits - 1);
        const T p = top + (static_cast<T>(rng() % top) | 1);
        const Modulus<T> modulus(p);
        const T n_inverse = static_cast<T>(rng() % p);

        DeviceBuffer<T> roots(random_below<T>(n, p));
        DeviceBuffer<T> inverse_roots(random_below<T>(n, p));
        DeviceBuffer<T> a(random_below<T>(total, p));
        DeviceBuffer<T> b(random_below<T>(total, p));

        ntt_configuration<T> fwd_cfg = {.n_power = LogN,
                                        .ntt_type = FORWARD,
                                        .ntt_layout = PerPolynomial,
                                        .reduction_poly = R,
                                        .mod_inverse = n_inverse,
                                        .stream = 0};
        ntt_configuration<T> inv_cfg = fwd_cfg;
        inv_cfg.ntt_type = INVERSE;

        DeviceBuffer<T> expected(total), b_hat(total);
        GPU_NTT(a.data, expected.data, roots.data, modulus, fwd_cfg, batch);
        GPU_NTT(b.data, b_hat.data, roots.data, modulus, fwd_cfg, batch);
        multiply_kernel<<<(total + 255) / 256, 256>>>(expected.data, b_hat.data,
                                                      modulus, total);
        GPU_INTT_Inplace(expected.data, inverse_roots.data, modulus, inv_cfg,
                         batch);

        DeviceBuffer<T> actual(total), b_partial(total);
        launch_stages_from<Fwd, 0>(
            a.data, actual.data, roots.data, modulus, n_inverse, batch,
            std::make_integer_sequence<int, Fwd::num_stages - 1>());
        launch_stages_from<Fwd, 0>(
            b.data, b_partial.data, roots.data, modulus, n_inverse, batch,
            std::make_integer_sequence<int, Fwd::num_stages - 1>());
        fused_stage_two_buffers_kernel<Fwd, Inv>
            <<<Last::grid(batch), Last::block_dim,
               2 * Last::shared_memory_size>>>(actual.data, b_partial.data,
                                               roots.data, inverse_roots.data,
                                               modulus, n_inverse);
        check_cuda(cudaGetLastError(), "launch");
        launch_stages_from<Inv, 1>(
            actual.data, actual.data, inverse_roots.data, modulus, n_inverse,
            batch, std::make_integer_sequence<int, Inv::num_stages - 1>());

        cases++;
        if (expected.read() != actual.read())
        {
            failures++;
            std::printf("FAIL fused stages with two buffers %s logn=%d %s "
                        "batch=%d\n",
                        sizeof(T) == 4 ? "u32" : "u64", LogN,
                        R == X_N_minus ? "X^N-1" : "X^N+1", batch);
        }
    }

} // namespace

int main()
{
    // Every single kernel ring size, 2^1 ... 2^10.
    run_sizes<Data32, 1, false>(std::make_integer_sequence<int, 10>());
    run_sizes<Data64, 1, false>(std::make_integer_sequence<int, 10>());

    // Stage kernels, 2^10 ... 2^20 and 2^25 (whose first or last kernel
    // swaps the grid axes).
    run_sizes<Data32, 10, true>(std::make_integer_sequence<int, 11>());
    run_sizes<Data64, 10, true>(std::make_integer_sequence<int, 11>());
    run_size<Data32, Data32, 25, true>();
    run_size<Data64, Data64, 25, true>();
    run_size<Data32, Data32s, 12, true>();
    run_size<Data64, Data64s, 25, true>();

    // Signed input (forward) and output (inverse).
    run_size<Data32, Data32s, 4>();
    run_size<Data32, Data32s, 9>();
    run_size<Data32, Data32s, 10>();
    run_size<Data64, Data64s, 4>();
    run_size<Data64, Data64s, 9>();
    run_size<Data64, Data64s, 10>();

    // RNS with the modulus and roots picked in the kernel.
    run_rns<false, Data32, 8, FORWARD, X_N_minus>();
    run_rns<false, Data32, 8, INVERSE, X_N_plus>();
    run_rns<false, Data64, 8, FORWARD, X_N_plus>();
    run_rns<false, Data64, 8, INVERSE, X_N_minus>();
    run_rns<true, Data32, 12, FORWARD, X_N_plus>();
    run_rns<true, Data32, 12, INVERSE, X_N_minus>();
    run_rns<true, Data64, 12, FORWARD, X_N_minus>();
    run_rns<true, Data64, 12, INVERSE, X_N_plus>();

    // NTT, pointwise product and INTT fused in one kernel.
    run_fused<Data32, 4, X_N_minus>();
    run_fused<Data32, 8, X_N_plus>();
    run_fused<Data32, 10, X_N_minus>();
    run_fused<Data64, 4, X_N_plus>();
    run_fused<Data64, 8, X_N_minus>();
    run_fused<Data64, 10, X_N_plus>();

    // NTT(a) + NTT(b) in one kernel, NTT(b) kept in registers.
    run_two_ntts<Data32, 4, X_N_plus>();
    run_two_ntts<Data32, 9, X_N_minus>();
    run_two_ntts<Data64, 8, X_N_plus>();
    run_two_ntts<Data64, 10, X_N_minus>();

    // The same with stage kernels; the last forward and the first inverse
    // stage run in one kernel. 2^25 swaps the grid axes in both.
    run_fused_stages<Data32, 11, X_N_plus>(3);
    run_fused_stages<Data32, 16, X_N_minus>(2);
    run_fused_stages<Data32, 20, X_N_plus>(2);
    run_fused_stages<Data64, 12, X_N_minus>(3);
    run_fused_stages<Data64, 17, X_N_plus>(2);
    run_fused_stages<Data64, 25, X_N_minus>(1);

    // BlockDim<T> sets the threads of every kernel for Size above 2^9: for
    // each T the smallest ring size, a 1-level stage (2^10 with 256), last
    // kernels that swap the grid axes (2^25 with 128), five kernels (2^27
    // with 128) and a single kernel (2^11 with 1024).
    run_size<Data32, Data32, 10, true, 128>();
    run_size<Data64, Data64, 16, true, 128>();
    run_size<Data32, Data32, 25, true, 128>();
    run_case<true, Data32, Data32, 27, FORWARD, X_N_plus, 128>();
    run_case<true, Data32, Data32, 27, INVERSE, X_N_minus, 128>();
    run_size<Data64, Data64, 22, true, 128>();
    run_size<Data32, Data32, 10, true, 256>();
    run_size<Data64, Data64, 20, true, 256>();
    run_size<Data64, Data64, 10, true, 512>();
    run_size<Data32, Data32, 13, true, 512>();
    run_size<Data32, Data32, 11, true, 1024>();
    run_size<Data64, Data64, 11, false, 1024>();
    run_size<Data64, Data64, 18, true, 1024>();
    run_size<Data64, Data64s, 14, true, 256>();
    run_fused_stages<Data32, 12, X_N_minus, 128>(3);
    run_fused_stages<Data64, 16, X_N_plus, 1024>(2);
    run_fused_stages<Data64, 25, X_N_minus, 128>(1);

    // The same with a and b in two shared memory buffers.
    run_two_buffers_stages<Data32, 11, X_N_minus>(3);
    run_two_buffers_stages<Data32, 17, X_N_plus>(2);
    run_two_buffers_stages<Data64, 12, X_N_plus>(3);
    run_two_buffers_stages<Data64, 25, X_N_minus>(1);

    // NTT::launch rejects the batch sizes the host API rejects.
    {
        using NTT = decltype(Size<(1 << 12)>() + Direction<FORWARD>() +
                             Precision<Data32>() + ReductionPoly<X_N_minus>() +
                             Block());
        const int batches[] = {0, -1, (1 << 19) + 1};
        for (int batch : batches)
        {
            cases++;
            try
            {
                NTT::launch<Data32, Data32>(nullptr, nullptr, nullptr,
                                            Modulus<Data32>(1073479681), batch);
                failures++;
                std::printf("FAIL launch accepts batch=%d\n", batch);
            }
            catch (const std::invalid_argument&)
            {
            }
        }
    }

    std::printf("ntt_device: %d of %d cases match the host API\n",
                cases - failures, cases);
    return failures == 0 ? 0 : 1;
}
