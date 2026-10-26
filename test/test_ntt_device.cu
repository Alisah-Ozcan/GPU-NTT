// Copyright 2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

// Tests the single kernel device API (NTT().load / execute / store) against
// the host API: a kernel written with a description must produce exactly the
// output of GPU_NTT / GPU_INTT on the same data. The data and root tables
// are random, so the comparison covers every butterfly. A fused kernel
// (NTT, pointwise product, INTT) is checked against the same steps run
// through the host API.

#include "gpuntt/ntt_merge/descriptor.cuh"
#include "gpuntt/ntt_merge/ntt.cuh"

#include <cstdio>
#include <random>
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
                                   const Root<typename NTT::value_type>* roots,
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
                                   const Root<typename NTT::value_type>* roots,
                                   Modulus<typename NTT::value_type> modulus,
                                   Ninverse<typename NTT::value_type> n_inverse,
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
        typename Fwd::value_type* out,
        const Root<typename Fwd::value_type>* roots,
        const Root<typename Fwd::value_type>* inverse_roots,
        Modulus<typename Fwd::value_type> modulus,
        Ninverse<typename Fwd::value_type> n_inverse, int batch)
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

    template <typename T> std::vector<T> random_below(size_t n, T bound)
    {
        std::vector<T> values(n);
        for (T& v : values)
        {
            v = static_cast<T>(rng() % bound);
        }
        return values;
    }

    // Runs one description on random data and compares it with the host
    // API. TIO is the signed or unsigned type of the input (forward) or the
    // output (inverse).
    template <typename T, typename TIO, int LogN, type D, ReductionPolynomial R>
    void run_case()
    {
        using NTT = decltype(Size<(1 << LogN)>() + Direction<D>() +
                             Precision<T>() + ReductionPoly<R>() + Block());
        constexpr int n = 1 << LogN;
        const int batch = 2 * NTT::batches_per_block + 1;
        const size_t total = size_t(batch) * n;

        const int max_bits = std::is_same<T, Data32>::value ? 30 : 61;
        const T top = T(1) << (max_bits - 1);
        const T p = top + (static_cast<T>(rng() % top) | 1);
        const Modulus<T> modulus(p);
        const T n_inverse = static_cast<T>(rng() % p);

        DeviceBuffer<T> roots(random_below<T>(n, p));
        std::vector<T> input = random_below<T>(total, p);

        ntt_configuration<T> cfg = {.n_power = LogN,
                                    .ntt_type = D,
                                    .ntt_layout = PerPolynomial,
                                    .reduction_poly = R,
                                    .mod_inverse = n_inverse,
                                    .stream = 0};

        std::vector<TIO> expected, actual;
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
            forward_kernel<NTT>
                <<<NTT::grid(batch), NTT::block_dim, NTT::shared_memory_size>>>(
                    in.data, out.data, roots.data, modulus, batch);
            check_cuda(cudaGetLastError(), "launch");
            const std::vector<T> e = reference.read(), a = out.read();
            expected.assign(e.begin(), e.end());
            actual.assign(a.begin(), a.end());
        }
        else
        {
            DeviceBuffer<T> in(input);
            DeviceBuffer<TIO> reference(total), out(total);
            GPU_INTT(in.data, reference.data, roots.data, modulus, cfg, batch);
            inverse_kernel<NTT>
                <<<NTT::grid(batch), NTT::block_dim, NTT::shared_memory_size>>>(
                    in.data, out.data, roots.data, modulus, n_inverse, batch);
            check_cuda(cudaGetLastError(), "launch");
            expected = reference.read();
            actual = out.read();
        }

        cases++;
        if (expected != actual)
        {
            failures++;
            std::printf("FAIL %s logn=%d %s %s%s batch=%d\n",
                        sizeof(T) == 4 ? "u32" : "u64", LogN,
                        D == FORWARD ? "forward" : "inverse",
                        R == X_N_minus ? "X^N-1" : "X^N+1",
                        std::is_signed<TIO>::value ? " signed" : "", batch);
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

        cases++;
        if (expected.read() != actual.read())
        {
            failures++;
            std::printf("FAIL fused multiply %s logn=%d %s batch=%d\n",
                        sizeof(T) == 4 ? "u32" : "u64", LogN,
                        R == X_N_minus ? "X^N-1" : "X^N+1", batch);
        }
    }

    template <typename T, typename TIO, int LogN> void run_size()
    {
        run_case<T, TIO, LogN, FORWARD, X_N_minus>();
        run_case<T, TIO, LogN, FORWARD, X_N_plus>();
        run_case<T, TIO, LogN, INVERSE, X_N_minus>();
        run_case<T, TIO, LogN, INVERSE, X_N_plus>();
    }

    template <typename T, int... LogN>
    void run_sizes(std::integer_sequence<int, LogN...>)
    {
        (run_size<T, T, LogN + 1>(), ...);
    }
} // namespace

int main()
{
    // Every single kernel ring size, 2^1 ... 2^10.
    run_sizes<Data32>(std::make_integer_sequence<int, 10>());
    run_sizes<Data64>(std::make_integer_sequence<int, 10>());

    // Signed input (forward) and output (inverse).
    run_size<Data32, Data32s, 4>();
    run_size<Data32, Data32s, 9>();
    run_size<Data32, Data32s, 10>();
    run_size<Data64, Data64s, 4>();
    run_size<Data64, Data64s, 9>();
    run_size<Data64, Data64s, 10>();

    // NTT, pointwise product and INTT fused in one kernel.
    run_fused<Data32, 4, X_N_minus>();
    run_fused<Data32, 8, X_N_plus>();
    run_fused<Data32, 10, X_N_minus>();
    run_fused<Data64, 4, X_N_plus>();
    run_fused<Data64, 8, X_N_minus>();
    run_fused<Data64, 10, X_N_plus>();

    std::printf("ntt_device: %d of %d cases match the host API\n",
                cases - failures, cases);
    return failures == 0 ? 0 : 1;
}
