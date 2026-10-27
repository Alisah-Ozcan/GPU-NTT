// Copyright 2023-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include "bench_util.cuh"

using namespace gpuntt;

namespace
{
    using DataTypes = nvbench::type_list<Data32, Data64>;

    template <typename T> struct centered_functor
    {
        T prime;

        __host__ __device__ typename std::make_signed<T>::type
        operator()(const T value) const
        {
            using S = typename std::make_signed<T>::type;
            return (value > (prime >> 1))
                       ? static_cast<S>(value) - static_cast<S>(prime)
                       : static_cast<S>(value);
        }
    };

    type direction_of(nvbench::state& state)
    {
        return state.get_string("Direction") == "forward" ? FORWARD : INVERSE;
    }

    template <typename T>
    void add_memory_traffic(nvbench::state& state, size_t coefficients,
                            size_t roots)
    {
        state.add_element_count(coefficients, "Coefficients");
        state.add_global_memory_reads<T>(coefficients + roots, "Read");
        state.add_global_memory_writes<T>(coefficients, "Write");
    }

    // Single modulus, row-wise, in-place NTT/INTT.
    template <typename T>
    void merge_ntt(nvbench::state& state, nvbench::type_list<T>)
    {
        const int logn = static_cast<int>(state.get_int64("LogN"));
        const size_t n = size_t(1) << logn;
        const int batch =
            (state.get_string("Workload") == "bulk") ? bulk_batch(logn, 1) : 1;
        const type dir = direction_of(state);
        const ReductionPolynomial poly = (state.get_string("Poly") == "X^N-1")
                                             ? ReductionPolynomial::X_N_minus
                                             : ReductionPolynomial::X_N_plus;
        const T prime = bench_prime<T>();
        const size_t root_count =
            (poly == ReductionPolynomial::X_N_minus) ? n / 2 : n;

        auto data = random_vector<T>(n * batch, prime, 1234);
        auto roots = random_vector<T>(root_count, prime, 4321);
        const Modulus<T> modulus(prime);
        add_memory_traffic<T>(state, n * batch, root_count);

        BenchStream stream(state);
        ntt_configuration<T> cfg = {.n_power = logn,
                                    .ntt_type = dir,
                                    .ntt_layout = PerPolynomial,
                                    .reduction_poly = poly,
                                    .mod_inverse = static_cast<T>(12345),
                                    .stream = stream.get()};

        state.exec(
            [&](nvbench::launch&)
            {
                if (dir == FORWARD)
                    GPU_NTT_Inplace(ptr(data), ptr(roots), modulus, cfg, batch);
                else
                    GPU_INTT_Inplace(ptr(data), ptr(roots), modulus, cfg,
                                     batch);
            });
    }

    // RNS, signed input/output and ordered variants (X^N + 1, bulk workload).
    template <typename T>
    void merge_ntt_variants(nvbench::state& state, nvbench::type_list<T>)
    {
        using S = typename std::make_signed<T>::type;

        const std::string variant = state.get_string("Variant");
        const int logn = static_cast<int>(state.get_int64("LogN"));
        const size_t n = size_t(1) << logn;
        const type dir = direction_of(state);
        const bool ordered =
            (variant == "modulus_ordered" || variant == "poly_ordered");
        if (ordered && logn < 10)
        {
            state.skip("Ordered NTT requires LogN >= 10.");
            return;
        }

        const int mod_count = (variant == "signed") ? 1 : 3;
        const int prime_count = 4;
        const int batch = bulk_batch(logn, 2 * mod_count);
        const size_t total = n * batch;
        const T prime = bench_prime<T>();

        auto data = random_vector<T>(total, prime, 1234);
        auto roots = random_vector<T>(prime_count * n, prime, 4321);
        thrust::device_vector<Modulus<T>> moduli(prime_count,
                                                 Modulus<T>(prime));
        thrust::device_vector<T> n_inverse(prime_count, static_cast<T>(12345));

        std::vector<int> host_order;
        if (variant == "modulus_ordered")
        {
            host_order = {2, 0, 3};
        }
        else
        {
            for (int z = 0; z < batch; z++)
            {
                host_order.push_back(batch - 1 - z);
            }
        }
        thrust::device_vector<int> order(host_order);
        add_memory_traffic<T>(state, total, mod_count * n);

        BenchStream stream(state);
        ntt_configuration<T> cfg = {.n_power = logn,
                                    .ntt_type = dir,
                                    .ntt_layout = PerPolynomial,
                                    .reduction_poly =
                                        ReductionPolynomial::X_N_plus,
                                    .mod_inverse = static_cast<T>(12345),
                                    .stream = stream.get()};
        ntt_rns_configuration<T> cfg_rns = {.n_power = logn,
                                            .ntt_type = dir,
                                            .ntt_layout = PerPolynomial,
                                            .reduction_poly =
                                                ReductionPolynomial::X_N_plus,
                                            .mod_inverse = ptr(n_inverse),
                                            .stream = stream.get()};

        if (variant == "signed" || variant == "signed_rns")
        {
            const bool use_rns = (variant == "signed_rns");
            thrust::device_vector<S> signed_data(total);
            thrust::transform(data.begin(), data.end(), signed_data.begin(),
                              centered_functor<T>{prime});
            const Modulus<T> modulus(prime);
            state.exec(
                [&](nvbench::launch&)
                {
                    if (dir == FORWARD && use_rns)
                        GPU_NTT(ptr(signed_data), ptr(data), ptr(roots),
                                ptr(moduli), cfg_rns, batch, mod_count);
                    else if (dir == FORWARD)
                        GPU_NTT(ptr(signed_data), ptr(data), ptr(roots),
                                modulus, cfg, batch);
                    else if (use_rns)
                        GPU_INTT(ptr(data), ptr(signed_data), ptr(roots),
                                 ptr(moduli), cfg_rns, batch, mod_count);
                    else
                        GPU_INTT(ptr(data), ptr(signed_data), ptr(roots),
                                 modulus, cfg, batch);
                });
        }
        else if (variant == "rns")
        {
            state.exec(
                [&](nvbench::launch&)
                {
                    if (dir == FORWARD)
                        GPU_NTT_Inplace(ptr(data), ptr(roots), ptr(moduli),
                                        cfg_rns, batch, mod_count);
                    else
                        GPU_INTT_Inplace(ptr(data), ptr(roots), ptr(moduli),
                                         cfg_rns, batch, mod_count);
                });
        }
        else if (variant == "modulus_ordered")
        {
            state.exec(
                [&](nvbench::launch&)
                {
                    GPU_NTT_Modulus_Ordered_Inplace(ptr(data), ptr(roots),
                                                    ptr(moduli), cfg_rns, batch,
                                                    mod_count, ptr(order));
                });
        }
        else
        {
            state.exec(
                [&](nvbench::launch&)
                {
                    GPU_NTT_Poly_Ordered_Inplace(ptr(data), ptr(roots),
                                                 ptr(moduli), cfg_rns, batch,
                                                 mod_count, ptr(order));
                });
        }
    }

    // Column-wise NTT/INTT (X^N + 1, about 2^22 coefficients).
    template <typename T>
    void merge_ntt_per_coefficient(nvbench::state& state, nvbench::type_list<T>)
    {
        const int logn = static_cast<int>(state.get_int64("LogN"));
        const size_t n = size_t(1) << logn;
        const int batch = 1 << (22 - logn);
        const type dir = direction_of(state);
        const bool use_rns = (state.get_string("Moduli") == "rns");
        const int mod_count = use_rns ? 3 : 1;
        const T prime = bench_prime<T>();

        auto data = random_vector<T>(n * batch, prime, 1234);
        auto roots = random_vector<T>(mod_count * n, prime, 4321);
        const Modulus<T> modulus(prime);
        thrust::device_vector<Modulus<T>> moduli(mod_count, modulus);
        thrust::device_vector<T> n_inverse(mod_count, static_cast<T>(12345));
        add_memory_traffic<T>(state, n * batch, mod_count * n);

        BenchStream stream(state);
        ntt_configuration<T> cfg = {.n_power = logn,
                                    .ntt_type = dir,
                                    .ntt_layout = PerCoefficient,
                                    .reduction_poly =
                                        ReductionPolynomial::X_N_plus,
                                    .mod_inverse = static_cast<T>(12345),
                                    .stream = stream.get()};
        ntt_rns_configuration<T> cfg_rns = {.n_power = logn,
                                            .ntt_type = dir,
                                            .ntt_layout = PerCoefficient,
                                            .reduction_poly =
                                                ReductionPolynomial::X_N_plus,
                                            .mod_inverse = ptr(n_inverse),
                                            .stream = stream.get()};

        state.exec(
            [&](nvbench::launch&)
            {
                if (dir == FORWARD && use_rns)
                    GPU_NTT_Inplace(ptr(data), ptr(roots), ptr(moduli), cfg_rns,
                                    batch, mod_count);
                else if (dir == FORWARD)
                    GPU_NTT_Inplace(ptr(data), ptr(roots), modulus, cfg, batch);
                else if (use_rns)
                    GPU_INTT_Inplace(ptr(data), ptr(roots), ptr(moduli),
                                     cfg_rns, batch, mod_count);
                else
                    GPU_INTT_Inplace(ptr(data), ptr(roots), modulus, cfg,
                                     batch);
            });
    }

} // namespace

NVBENCH_BENCH_TYPES(merge_ntt, NVBENCH_TYPE_AXES(DataTypes))
    .set_type_axes_names({"Data"})
    .add_int64_axis("LogN", nvbench::range(1, 28))
    .add_string_axis("Workload", {"single", "bulk"})
    .add_string_axis("Direction", {"forward", "inverse"})
    .add_string_axis("Poly", {"X^N-1", "X^N+1"})
    .set_timeout(1);

NVBENCH_BENCH_TYPES(merge_ntt_variants, NVBENCH_TYPE_AXES(DataTypes))
    .set_type_axes_names({"Data"})
    .add_string_axis("Variant", {"rns", "signed", "signed_rns",
                                 "modulus_ordered", "poly_ordered"})
    .add_int64_axis("LogN", {5, 9, 10, 12, 14, 16, 20, 24, 25, 26})
    .add_string_axis("Direction", {"forward", "inverse"})
    .set_timeout(1);

NVBENCH_BENCH_TYPES(merge_ntt_per_coefficient, NVBENCH_TYPE_AXES(DataTypes))
    .set_type_axes_names({"Data"})
    .add_int64_axis("LogN", nvbench::range(1, 9))
    .add_string_axis("Moduli", {"single", "rns"})
    .add_string_axis("Direction", {"forward", "inverse"})
    .set_timeout(1);
