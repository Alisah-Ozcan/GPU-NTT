// Copyright 2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

// Tests the device API (NTT().load / execute / store and the per stage
// NTT::stage<I>() functions) against the host API: kernels written with a
// description must produce exactly the output of GPU_NTT / GPU_INTT on the
// same data. The data and root tables
// are random, so the comparison covers every butterfly. A fused kernel
// (NTT, pointwise product, INTT) is checked against the same steps run
// through the host API.

#include "gpuntt/ntt_merge/descriptor.cuh"
#include "gpuntt/ntt_merge/ntt.cuh"

#include <algorithm>
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

    // Runs one description on random data and compares it with the host
    // API. TIO is the signed or unsigned type of the input (forward) or the
    // output (inverse).
    template <bool Staged, typename T, typename TIO, int LogN, type D,
              ReductionPolynomial R>
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
        }

        cases++;
        if (expected != actual)
        {
            failures++;
            std::printf("FAIL %s %s logn=%d %s %s%s batch=%d\n",
                        Staged ? "stages" : "single kernel",
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

    template <typename T, typename TIO, int LogN, bool Staged = false>
    void run_size()
    {
        run_case<Staged, T, TIO, LogN, FORWARD, X_N_minus>();
        run_case<Staged, T, TIO, LogN, FORWARD, X_N_plus>();
        run_case<Staged, T, TIO, LogN, INVERSE, X_N_minus>();
        run_case<Staged, T, TIO, LogN, INVERSE, X_N_plus>();
    }

    template <typename T, int First, bool Staged, int... LogN>
    void run_sizes(std::integer_sequence<int, LogN...>)
    {
        (run_size<T, T, First + LogN, Staged>(), ...);
    }
    // The same product for ring sizes with stage kernels: forward stages
    // 0 ... S - 2, fused_stage_kernel, inverse stages 1 ... S - 1.
    template <typename T, int LogN, ReductionPolynomial R>
    void run_fused_stages(int batch)
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
        DeviceBuffer<T> b_hat(random_below<T>(total, p));

        ntt_configuration<T> fwd_cfg = {.n_power = LogN,
                                        .ntt_type = FORWARD,
                                        .ntt_layout = PerPolynomial,
                                        .reduction_poly = R,
                                        .mod_inverse = n_inverse,
                                        .stream = 0};
        ntt_configuration<T> inv_cfg = fwd_cfg;
        inv_cfg.ntt_type = INVERSE;

        DeviceBuffer<T> expected(total), actual(total);
        GPU_NTT(a.data, expected.data, roots.data, modulus, fwd_cfg, batch);
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

        cases++;
        if (expected.read() != actual.read())
        {
            failures++;
            std::printf("FAIL fused stages %s logn=%d %s batch=%d\n",
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

    // The same with stage kernels; the last forward and the first inverse
    // stage run in one kernel. 2^25 swaps the grid axes in both.
    run_fused_stages<Data32, 11, X_N_plus>(3);
    run_fused_stages<Data32, 16, X_N_minus>(2);
    run_fused_stages<Data32, 20, X_N_plus>(2);
    run_fused_stages<Data64, 12, X_N_minus>(3);
    run_fused_stages<Data64, 17, X_N_plus>(2);
    run_fused_stages<Data64, 25, X_N_minus>(1);

    std::printf("ntt_device: %d of %d cases match the host API\n",
                cases - failures, cases);
    return failures == 0 ? 0 : 1;
}
