// Copyright 2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

// Correctness tests of the merge NTT host API against the CPU reference.
//
// Usage: merge_ntt_tests --bits <32|64> --suite <name|all> --max-logn <N>
// Suites: single, pool_parameters, rns, signed, per_coefficient,
//         per_coefficient_rns, modulus_ordered, poly_ordered

#include <algorithm>
#include <cstring>
#include <functional>
#include <numeric>

#include "test_util.cuh"

using namespace gpuntt_test;

namespace
{
    const ReductionPolynomial kPolys[] = {ReductionPolynomial::X_N_minus,
                                          ReductionPolynomial::X_N_plus};
    const type kDirections[] = {FORWARD, INVERSE};

    std::string case_name(const std::string& variant, type dir, int logn,
                          ReductionPolynomial poly, int batch)
    {
        return variant + (dir == FORWARD ? " ntt" : " intt") +
               " logn=" + std::to_string(logn) +
               (poly == ReductionPolynomial::X_N_minus ? " X^N-1" : " X^N+1") +
               " batch=" + std::to_string(batch);
    }

    // About 2^20 coefficients per case. 37 is not a multiple of the number of
    // polynomials processed by a single block, so partial blocks are covered.
    int batch_for(int logn, int at_least)
    {
        int batch = std::min(37, std::max(1, (1 << 20) >> logn));
        return std::max(batch, at_least);
    }

    void sync() { GPUNTT_CUDA_CHECK(cudaDeviceSynchronize()); }

    template <typename T> class MergeNTTTests
    {
        using S = typename std::make_signed<T>::type;
        using CpuOf = std::function<NTTCPU<T>&(int)>;

      public:
        explicit MergeNTTTests(int max_logn)
            : fields_(make_fields<T>(4)),
              max_logn_(std::min(max_logn, fields_[0].max_logn())), rng_(2026)
        {
        }

        int run(const std::string& suite)
        {
            if (suite == "single")
                return single();
            if (suite == "pool_parameters")
                return pool_parameters();
            if (suite == "rns")
                return rns();
            if (suite == "signed")
                return signed_io();
            if (suite == "per_coefficient")
                return per_coefficient(false);
            if (suite == "per_coefficient_rns")
                return per_coefficient(true);
            if (suite == "modulus_ordered")
                return modulus_ordered();
            if (suite == "poly_ordered")
                return poly_ordered();
            if (suite == "all")
            {
                int result = 0;
                for (const char* s :
                     {"single", "pool_parameters", "rns", "signed",
                      "per_coefficient", "per_coefficient_rns",
                      "modulus_ordered", "poly_ordered"})
                {
                    result |= run(s);
                }
                return result;
            }
            throw std::invalid_argument("Unknown suite: " + suite);
        }

      private:
        std::string suite_name(const std::string& suite) const
        {
            return std::string(std::is_same<T, Data32>::value ? "Data32"
                                                              : "Data64") +
                   "." + suite;
        }

        std::vector<NTTParameters<T>> parameters(int count, int logn,
                                                 ReductionPolynomial poly) const
        {
            std::vector<NTTParameters<T>> result;
            for (int m = 0; m < count; m++)
            {
                result.push_back(fields_[m].parameters(logn, poly));
            }
            return result;
        }

        static std::vector<NTTCPU<T>>
        cpu_references(const std::vector<NTTParameters<T>>& params)
        {
            std::vector<NTTCPU<T>> result;
            for (const auto& p : params)
            {
                result.emplace_back(p);
            }
            return result;
        }

        static std::vector<T> transform(NTTCPU<T>& cpu, type dir,
                                        std::vector<T> poly)
        {
            return (dir == FORWARD) ? cpu.ntt(poly) : cpu.intt(poly);
        }

        // Polynomial j occupies [j * n, (j + 1) * n) and is transformed with
        // cpu_of(j).
        static std::vector<T> reference(const std::vector<T>& input, int logn,
                                        type dir, const CpuOf& cpu_of)
        {
            const size_t n = size_t(1) << logn;
            std::vector<T> output(input.size());
            for (size_t j = 0; j < input.size() / n; j++)
            {
                std::vector<T> poly(input.begin() + j * n,
                                    input.begin() + (j + 1) * n);
                std::vector<T> result = transform(cpu_of(j), dir, poly);
                std::copy(result.begin(), result.end(), output.begin() + j * n);
            }
            return output;
        }

        // Coefficient i of polynomial j is at i * batch + j.
        static std::vector<T> reference_per_coefficient(
            const std::vector<T>& input, int logn, int batch, type dir,
            const CpuOf& cpu_of)
        {
            const size_t n = size_t(1) << logn;
            std::vector<T> output(input.size());
            for (int j = 0; j < batch; j++)
            {
                std::vector<T> poly(n);
                for (size_t i = 0; i < n; i++)
                {
                    poly[i] = input[i * batch + j];
                }
                std::vector<T> result = transform(cpu_of(j), dir, poly);
                for (size_t i = 0; i < n; i++)
                {
                    output[i * batch + j] = result[i];
                }
            }
            return output;
        }

        std::vector<T> random_input(size_t count)
        {
            return random_values<T>(rng_, count, fields_[0].prime);
        }

        // Random input reduced by each polynomial's own modulus.
        std::vector<T> random_input(size_t count, size_t n,
                                    const std::function<T(int)>& prime_of)
        {
            std::vector<T> values = random_input(count);
            for (size_t i = 0; i < count; i++)
            {
                values[i] %= prime_of(static_cast<int>(i / n));
            }
            return values;
        }

        // Single modulus NTT/INTT, out-of-place and in-place.
        void single_case(TestReport& report, NTTParameters<T>& params,
                         const std::string& variant, type dir)
        {
            const int logn = params.logn;
            const ReductionPolynomial poly = params.poly_reduction;
            const int batch = batch_for(logn, 1);
            const size_t total = size_t(batch) << logn;
            NTTCPU<T> cpu(params);

            std::vector<T> input =
                random_values<T>(rng_, total, params.modulus.value);
            std::vector<T> expected = reference(
                input, logn, dir, [&](int) -> NTTCPU<T>& { return cpu; });

            ntt_configuration<T> cfg = {.n_power = logn,
                                        .ntt_type = dir,
                                        .ntt_layout = PerPolynomial,
                                        .reduction_poly = poly,
                                        .zero_padding = false,
                                        .mod_inverse = params.n_inv,
                                        .stream = 0};
            const std::string name = case_name(variant, dir, logn, poly, batch);
            try
            {
                DeviceBuffer<Root<T>> roots(
                    params.gpu_root_of_unity_table_generator(
                        dir == FORWARD ? params.forward_root_of_unity_table
                                       : params.inverse_root_of_unity_table));

                DeviceBuffer<T> in(input);
                DeviceBuffer<T> out(total);
                if (dir == FORWARD)
                    GPU_NTT(in.get(), out.get(), roots.get(), params.modulus,
                            cfg, batch);
                else
                    GPU_INTT(in.get(), out.get(), roots.get(), params.modulus,
                             cfg, batch);
                sync();
                report.check(name + " out-of-place", out.download(), expected);
                report.check(name + " out-of-place keeps input", in.download(),
                             input);

                DeviceBuffer<T> inout(input);
                if (dir == FORWARD)
                    GPU_NTT_Inplace(inout.get(), roots.get(), params.modulus,
                                    cfg, batch);
                else
                    GPU_INTT_Inplace(inout.get(), roots.get(), params.modulus,
                                     cfg, batch);
                sync();
                report.check(name + " in-place", inout.download(), expected);
            }
            catch (const std::exception& e)
            {
                report.error(name, e.what());
            }
        }

        int single()
        {
            TestReport report(suite_name("single"));
            for (int logn = 1; logn <= max_logn_; logn++)
                for (ReductionPolynomial poly : kPolys)
                    for (type dir : kDirections)
                    {
                        NTTParameters<T> params = fields_[0].parameters(logn, poly);
                        single_case(report, params, "single", dir);
                    }
            return report.finish();
        }

        // NTTParameters built from the default modulus pool (as used by the
        // examples): out-of-range ring sizes are rejected, valid ones give
        // correct transforms.
        int pool_parameters()
        {
            TestReport report(suite_name("pool_parameters"));
            const int pool_max_logn = std::is_same<T, Data32>::value ? 25 : 28;
            for (int logn : {0, pool_max_logn + 1, pool_max_logn + 3})
                for (ReductionPolynomial poly : kPolys)
                {
                    bool rejected = false;
                    try
                    {
                        NTTParameters<T> params(logn, poly);
                    }
                    catch (const std::invalid_argument&)
                    {
                        rejected = true;
                    }
                    report.expect(rejected,
                                  "pool logn=" + std::to_string(logn) +
                                      (poly == ReductionPolynomial::X_N_minus
                                           ? " X^N-1"
                                           : " X^N+1"),
                                  "out-of-range ring size was not rejected");
                }

            for (int logn = 1; logn <= std::min(max_logn_, pool_max_logn); logn++)
                for (ReductionPolynomial poly : kPolys)
                    for (type dir : kDirections)
                    {
                        NTTParameters<T> params(logn, poly);
                        single_case(report, params, "pool", dir);
                    }
            return report.finish();
        }

        int rns()
        {
            TestReport report(suite_name("rns"));
            const int mod_count = 3;
            for (int logn = 1; logn <= max_logn_; logn++)
                for (ReductionPolynomial poly : kPolys)
                    for (type dir : kDirections)
                    {
                        const int batch = batch_for(logn, mod_count);
                        const size_t n = size_t(1) << logn;
                        const size_t total = batch * n;
                        auto params = parameters(mod_count, logn, poly);
                        auto cpus = cpu_references(params);
                        auto cpu_of = [&](int j) -> NTTCPU<T>& {
                            return cpus[j % mod_count];
                        };

                        std::vector<T> input = random_input(
                            total, n, [&](int j) { return fields_[j % mod_count].prime; });
                        std::vector<T> expected = reference(input, logn, dir, cpu_of);

                        const std::string name =
                            case_name("rns", dir, logn, poly, batch);
                        try
                        {
                            std::vector<Modulus<T>> moduli;
                            std::vector<Ninverse<T>> n_inverse;
                            for (auto& p : params)
                            {
                                moduli.push_back(p.modulus);
                                n_inverse.push_back(p.n_inv);
                            }
                            DeviceBuffer<Modulus<T>> d_moduli(moduli);
                            DeviceBuffer<Ninverse<T>> d_n_inverse(n_inverse);
                            DeviceBuffer<Root<T>> roots(
                                concat_root_tables(params, dir == INVERSE, n));

                            ntt_rns_configuration<T> cfg = {
                                .n_power = logn,
                                .ntt_type = dir,
                                .ntt_layout = PerPolynomial,
                                .reduction_poly = poly,
                                .zero_padding = false,
                                .mod_inverse = d_n_inverse.get(),
                                .stream = 0};

                            DeviceBuffer<T> in(input);
                            DeviceBuffer<T> out(total);
                            if (dir == FORWARD)
                                GPU_NTT(in.get(), out.get(), roots.get(),
                                        d_moduli.get(), cfg, batch, mod_count);
                            else
                                GPU_INTT(in.get(), out.get(), roots.get(),
                                         d_moduli.get(), cfg, batch, mod_count);
                            sync();
                            report.check(name + " out-of-place", out.download(),
                                         expected);

                            DeviceBuffer<T> inout(input);
                            if (dir == FORWARD)
                                GPU_NTT_Inplace(inout.get(), roots.get(),
                                                d_moduli.get(), cfg, batch,
                                                mod_count);
                            else
                                GPU_INTT_Inplace(inout.get(), roots.get(),
                                                 d_moduli.get(), cfg, batch,
                                                 mod_count);
                            sync();
                            report.check(name + " in-place", inout.download(),
                                         expected);
                        }
                        catch (const std::exception& e)
                        {
                            report.error(name, e.what());
                        }
                    }
            return report.finish();
        }

        // Signed input for the forward NTT, signed (centered) output for the
        // inverse NTT; single modulus and RNS.
        int signed_io()
        {
            TestReport report(suite_name("signed"));
            const int mod_count = 3;
            for (int logn = 1; logn <= max_logn_; logn++)
                for (ReductionPolynomial poly : kPolys)
                    for (type dir : kDirections)
                        for (bool use_rns : {false, true})
                        {
                            const int count = use_rns ? mod_count : 1;
                            const int batch = batch_for(logn, count);
                            const size_t n = size_t(1) << logn;
                            const size_t total = batch * n;
                            auto params = parameters(count, logn, poly);
                            auto cpus = cpu_references(params);
                            auto prime_of = [&](int j) { return fields_[j % count].prime; };

                            std::vector<T> input = random_input(total, n, prime_of);
                            std::vector<T> expected = reference(
                                input, logn, dir,
                                [&](int j) -> NTTCPU<T>& { return cpus[j % count]; });

                            const std::string name = case_name(
                                use_rns ? "signed rns" : "signed single", dir, logn,
                                poly, batch);
                            try
                            {
                                std::vector<Modulus<T>> moduli;
                                std::vector<Ninverse<T>> n_inverse;
                                for (auto& p : params)
                                {
                                    moduli.push_back(p.modulus);
                                    n_inverse.push_back(p.n_inv);
                                }
                                DeviceBuffer<Modulus<T>> d_moduli(moduli);
                                DeviceBuffer<Ninverse<T>> d_n_inverse(n_inverse);
                                DeviceBuffer<Root<T>> roots(
                                    concat_root_tables(params, dir == INVERSE, n));

                                ntt_configuration<T> cfg = {
                                    .n_power = logn,
                                    .ntt_type = dir,
                                    .ntt_layout = PerPolynomial,
                                    .reduction_poly = poly,
                                    .zero_padding = false,
                                    .mod_inverse = params[0].n_inv,
                                    .stream = 0};
                                ntt_rns_configuration<T> cfg_rns = {
                                    .n_power = logn,
                                    .ntt_type = dir,
                                    .ntt_layout = PerPolynomial,
                                    .reduction_poly = poly,
                                    .zero_padding = false,
                                    .mod_inverse = d_n_inverse.get(),
                                    .stream = 0};

                                if (dir == FORWARD)
                                {
                                    std::vector<S> signed_input(total);
                                    for (size_t i = 0; i < total; i++)
                                    {
                                        signed_input[i] =
                                            to_centered(input[i], prime_of(i / n));
                                    }
                                    DeviceBuffer<S> in(signed_input);
                                    DeviceBuffer<T> out(total);
                                    if (use_rns)
                                        GPU_NTT(in.get(), out.get(), roots.get(),
                                                d_moduli.get(), cfg_rns, batch,
                                                mod_count);
                                    else
                                        GPU_NTT(in.get(), out.get(), roots.get(),
                                                params[0].modulus, cfg, batch);
                                    sync();
                                    report.check(name, out.download(), expected);
                                }
                                else
                                {
                                    std::vector<S> signed_expected(total);
                                    for (size_t i = 0; i < total; i++)
                                    {
                                        signed_expected[i] = to_centered(
                                            expected[i], prime_of(i / n));
                                    }
                                    DeviceBuffer<T> in(input);
                                    DeviceBuffer<S> out(total);
                                    if (use_rns)
                                        GPU_INTT(in.get(), out.get(), roots.get(),
                                                 d_moduli.get(), cfg_rns, batch,
                                                 mod_count);
                                    else
                                        GPU_INTT(in.get(), out.get(), roots.get(),
                                                 params[0].modulus, cfg, batch);
                                    sync();
                                    report.check(name, out.download(),
                                                 signed_expected);
                                }
                            }
                            catch (const std::exception& e)
                            {
                                report.error(name, e.what());
                            }
                        }
            return report.finish();
        }

        // Column-wise NTT, supported for 2 <= N <= 2^9. With RNS, column j
        // is transformed with modulus j % mod_count.
        int per_coefficient(bool use_rns)
        {
            TestReport report(
                suite_name(use_rns ? "per_coefficient_rns" : "per_coefficient"));
            const int mod_count = 3;
            for (int logn = 1; logn <= std::min(9, max_logn_); logn++)
                for (ReductionPolynomial poly : kPolys)
                    for (type dir : kDirections)
                        {
                            const int count = use_rns ? mod_count : 1;
                            // A power of two with at least 2^11 coefficients.
                            const int batch = 1 << std::max(2, 11 - logn);
                            const size_t n = size_t(1) << logn;
                            const size_t total = batch * n;
                            auto params = parameters(count, logn, poly);
                            auto cpus = cpu_references(params);

                            std::vector<T> input = random_input(total);
                            for (size_t i = 0; i < total; i++)
                            {
                                input[i] %= fields_[(i % batch) % count].prime;
                            }
                            std::vector<T> expected = reference_per_coefficient(
                                input, logn, batch, dir,
                                [&](int j) -> NTTCPU<T>& { return cpus[j % count]; });

                            const std::string name = case_name(
                                use_rns ? "per_coefficient rns"
                                        : "per_coefficient single",
                                dir, logn, poly, batch);
                            try
                            {
                                std::vector<Modulus<T>> moduli;
                                std::vector<Ninverse<T>> n_inverse;
                                for (auto& p : params)
                                {
                                    moduli.push_back(p.modulus);
                                    n_inverse.push_back(p.n_inv);
                                }
                                DeviceBuffer<Modulus<T>> d_moduli(moduli);
                                DeviceBuffer<Ninverse<T>> d_n_inverse(n_inverse);
                                DeviceBuffer<Root<T>> roots(
                                    concat_root_tables(params, dir == INVERSE, n));

                                ntt_configuration<T> cfg = {
                                    .n_power = logn,
                                    .ntt_type = dir,
                                    .ntt_layout = PerCoefficient,
                                    .reduction_poly = poly,
                                    .zero_padding = false,
                                    .mod_inverse = params[0].n_inv,
                                    .stream = 0};
                                ntt_rns_configuration<T> cfg_rns = {
                                    .n_power = logn,
                                    .ntt_type = dir,
                                    .ntt_layout = PerCoefficient,
                                    .reduction_poly = poly,
                                    .zero_padding = false,
                                    .mod_inverse = d_n_inverse.get(),
                                    .stream = 0};

                                DeviceBuffer<T> in(input);
                                DeviceBuffer<T> out(total);
                                DeviceBuffer<T> inout(input);
                                if (use_rns && dir == FORWARD)
                                {
                                    GPU_NTT(in.get(), out.get(), roots.get(),
                                            d_moduli.get(), cfg_rns, batch,
                                            mod_count);
                                    GPU_NTT_Inplace(inout.get(), roots.get(),
                                                    d_moduli.get(), cfg_rns,
                                                    batch, mod_count);
                                }
                                else if (use_rns)
                                {
                                    GPU_INTT(in.get(), out.get(), roots.get(),
                                             d_moduli.get(), cfg_rns, batch,
                                             mod_count);
                                    GPU_INTT_Inplace(inout.get(), roots.get(),
                                                     d_moduli.get(), cfg_rns,
                                                     batch, mod_count);
                                }
                                else if (dir == FORWARD)
                                {
                                    GPU_NTT(in.get(), out.get(), roots.get(),
                                            params[0].modulus, cfg, batch);
                                    GPU_NTT_Inplace(inout.get(), roots.get(),
                                                    params[0].modulus, cfg,
                                                    batch);
                                }
                                else
                                {
                                    GPU_INTT(in.get(), out.get(), roots.get(),
                                             params[0].modulus, cfg, batch);
                                    GPU_INTT_Inplace(inout.get(), roots.get(),
                                                     params[0].modulus, cfg,
                                                     batch);
                                }
                                sync();
                                report.check(name + " out-of-place",
                                             out.download(), expected);
                                report.check(name + " in-place", inout.download(),
                                             expected);

                                // Signed input (NTT) / signed output (INTT).
                                auto prime_of = [&](size_t i) {
                                    return fields_[(i % batch) % count].prime;
                                };
                                if (dir == FORWARD)
                                {
                                    std::vector<S> signed_input(total);
                                    for (size_t i = 0; i < total; i++)
                                    {
                                        signed_input[i] =
                                            to_centered(input[i], prime_of(i));
                                    }
                                    DeviceBuffer<S> s_in(signed_input);
                                    DeviceBuffer<T> s_out(total);
                                    if (use_rns)
                                        GPU_NTT(s_in.get(), s_out.get(),
                                                roots.get(), d_moduli.get(),
                                                cfg_rns, batch, mod_count);
                                    else
                                        GPU_NTT(s_in.get(), s_out.get(),
                                                roots.get(), params[0].modulus,
                                                cfg, batch);
                                    sync();
                                    report.check(name + " signed input",
                                                 s_out.download(), expected);
                                }
                                else
                                {
                                    std::vector<S> signed_expected(total);
                                    for (size_t i = 0; i < total; i++)
                                    {
                                        signed_expected[i] =
                                            to_centered(expected[i], prime_of(i));
                                    }
                                    DeviceBuffer<T> s_in(input);
                                    DeviceBuffer<S> s_out(total);
                                    if (use_rns)
                                        GPU_INTT(s_in.get(), s_out.get(),
                                                 roots.get(), d_moduli.get(),
                                                 cfg_rns, batch, mod_count);
                                    else
                                        GPU_INTT(s_in.get(), s_out.get(),
                                                 roots.get(), params[0].modulus,
                                                 cfg, batch);
                                    sync();
                                    report.check(name + " signed output",
                                                 s_out.download(),
                                                 signed_expected);
                                }
                            }
                            catch (const std::exception& e)
                            {
                                report.error(name, e.what());
                            }
                        }
            return report.finish();
        }

        // Polynomial z is transformed with prime order[z % mod_count], taken
        // from the full modulus list. Supported for N >= 2^10.
        int modulus_ordered()
        {
            TestReport report(suite_name("modulus_ordered"));
            const std::vector<int> order = {2, 0, 3};
            const int mod_count = static_cast<int>(order.size());
            const int prime_count = static_cast<int>(fields_.size());
            for (int logn = 10; logn <= max_logn_; logn++)
                for (ReductionPolynomial poly : kPolys)
                    for (type dir : kDirections)
                    {
                        const int batch = batch_for(logn, 2 * mod_count);
                        const size_t n = size_t(1) << logn;
                        const size_t total = batch * n;
                        auto params = parameters(prime_count, logn, poly);
                        auto cpus = cpu_references(params);
                        auto prime_index = [&](int z) { return order[z % mod_count]; };

                        std::vector<T> input = random_input(
                            total, n, [&](int z) { return fields_[prime_index(z)].prime; });
                        std::vector<T> expected = reference(
                            input, logn, dir,
                            [&](int z) -> NTTCPU<T>& { return cpus[prime_index(z)]; });

                        const std::string name =
                            case_name("modulus_ordered", dir, logn, poly, batch);
                        try
                        {
                            std::vector<Modulus<T>> moduli;
                            std::vector<Ninverse<T>> n_inverse;
                            for (auto& p : params)
                            {
                                moduli.push_back(p.modulus);
                                n_inverse.push_back(p.n_inv);
                            }
                            DeviceBuffer<Modulus<T>> d_moduli(moduli);
                            DeviceBuffer<Ninverse<T>> d_n_inverse(n_inverse);
                            DeviceBuffer<int> d_order(order);
                            DeviceBuffer<Root<T>> roots(
                                concat_root_tables(params, dir == INVERSE, n));

                            ntt_rns_configuration<T> cfg = {
                                .n_power = logn,
                                .ntt_type = dir,
                                .ntt_layout = PerPolynomial,
                                .reduction_poly = poly,
                                .zero_padding = false,
                                .mod_inverse = d_n_inverse.get(),
                                .stream = 0};

                            DeviceBuffer<T> in(input);
                            DeviceBuffer<T> out(total);
                            GPU_NTT_Modulus_Ordered(in.get(), out.get(),
                                                    roots.get(), d_moduli.get(),
                                                    cfg, batch, mod_count,
                                                    d_order.get());
                            sync();
                            report.check(name + " out-of-place", out.download(),
                                         expected);

                            DeviceBuffer<T> inout(input);
                            GPU_NTT_Modulus_Ordered_Inplace(
                                inout.get(), roots.get(), d_moduli.get(), cfg,
                                batch, mod_count, d_order.get());
                            sync();
                            report.check(name + " in-place", inout.download(),
                                         expected);
                        }
                        catch (const std::exception& e)
                        {
                            report.error(name, e.what());
                        }
                    }
            return report.finish();
        }

        // The polynomial at position order[z] is transformed with modulus
        // z % mod_count. Supported for N >= 2^10.
        int poly_ordered()
        {
            TestReport report(suite_name("poly_ordered"));
            const int mod_count = 3;
            for (int logn = 10; logn <= max_logn_; logn++)
                for (ReductionPolynomial poly : kPolys)
                    for (type dir : kDirections)
                    {
                        const int batch = batch_for(logn, 2 * mod_count);
                        const size_t n = size_t(1) << logn;
                        const size_t total = batch * n;
                        auto params = parameters(mod_count, logn, poly);
                        auto cpus = cpu_references(params);

                        std::vector<int> order(batch);
                        std::iota(order.begin(), order.end(), 0);
                        std::shuffle(order.begin(), order.end(), rng_);
                        std::vector<int> modulus_of(batch);
                        for (int z = 0; z < batch; z++)
                        {
                            modulus_of[order[z]] = z % mod_count;
                        }

                        std::vector<T> input = random_input(
                            total, n, [&](int p) { return fields_[modulus_of[p]].prime; });
                        std::vector<T> expected = reference(
                            input, logn, dir,
                            [&](int p) -> NTTCPU<T>& { return cpus[modulus_of[p]]; });

                        const std::string name =
                            case_name("poly_ordered", dir, logn, poly, batch);
                        try
                        {
                            std::vector<Modulus<T>> moduli;
                            std::vector<Ninverse<T>> n_inverse;
                            for (auto& p : params)
                            {
                                moduli.push_back(p.modulus);
                                n_inverse.push_back(p.n_inv);
                            }
                            DeviceBuffer<Modulus<T>> d_moduli(moduli);
                            DeviceBuffer<Ninverse<T>> d_n_inverse(n_inverse);
                            DeviceBuffer<int> d_order(order);
                            DeviceBuffer<Root<T>> roots(
                                concat_root_tables(params, dir == INVERSE, n));

                            ntt_rns_configuration<T> cfg = {
                                .n_power = logn,
                                .ntt_type = dir,
                                .ntt_layout = PerPolynomial,
                                .reduction_poly = poly,
                                .zero_padding = false,
                                .mod_inverse = d_n_inverse.get(),
                                .stream = 0};

                            DeviceBuffer<T> in(input);
                            DeviceBuffer<T> out(total);
                            GPU_NTT_Poly_Ordered(in.get(), out.get(), roots.get(),
                                                 d_moduli.get(), cfg, batch,
                                                 mod_count, d_order.get());
                            sync();
                            report.check(name + " out-of-place", out.download(),
                                         expected);

                            DeviceBuffer<T> inout(input);
                            GPU_NTT_Poly_Ordered_Inplace(
                                inout.get(), roots.get(), d_moduli.get(), cfg,
                                batch, mod_count, d_order.get());
                            sync();
                            report.check(name + " in-place", inout.download(),
                                         expected);
                        }
                        catch (const std::exception& e)
                        {
                            report.error(name, e.what());
                        }
                    }
            return report.finish();
        }

        std::vector<PrimeField<T>> fields_;
        int max_logn_;
        std::mt19937_64 rng_;
    };

} // namespace

int main(int argc, char* argv[])
{
    int bits = 64;
    std::string suite = "all";
    int max_logn = 22;

    for (int i = 1; i + 1 < argc; i += 2)
    {
        if (std::strcmp(argv[i], "--bits") == 0)
            bits = std::atoi(argv[i + 1]);
        else if (std::strcmp(argv[i], "--suite") == 0)
            suite = argv[i + 1];
        else if (std::strcmp(argv[i], "--max-logn") == 0)
            max_logn = std::atoi(argv[i + 1]);
        else
        {
            std::cerr << "Unknown argument: " << argv[i] << std::endl;
            return 2;
        }
    }

    try
    {
        if (bits == 32)
            return MergeNTTTests<Data32>(max_logn).run(suite);
        if (bits == 64)
            return MergeNTTTests<Data64>(max_logn).run(suite);
        std::cerr << "--bits must be 32 or 64" << std::endl;
        return 2;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 2;
    }
}
