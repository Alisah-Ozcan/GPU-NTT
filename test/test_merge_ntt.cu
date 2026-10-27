// Copyright 2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

// Correctness tests of the merge NTT host API against the CPU reference.
//
// Usage: merge_ntt_tests --bits <32|64> --suite <name|all> --max-logn <N>
// Suites: single, pool_parameters, modulus, validation, rns, signed,
//         per_coefficient, per_coefficient_rns, modulus_ordered, poly_ordered,
//         large

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

    void sync()
    {
        GPUNTT_CUDA_CHECK(cudaDeviceSynchronize());
    }

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
            if (suite == "modulus")
                return modulus();
            if (suite == "validation")
                return validation();
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
            if (suite == "large")
                return large();
            if (suite == "all")
            {
                int result = 0;
                for (const char* s :
                     {"single", "pool_parameters", "modulus", "validation",
                      "rns", "signed", "per_coefficient", "per_coefficient_rns",
                      "modulus_ordered", "poly_ordered", "large"})
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
        static std::vector<T>
        reference_per_coefficient(const std::vector<T>& input, int logn,
                                  int batch, type dir, const CpuOf& cpu_of)
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
                        NTTParameters<T> params =
                            fields_[0].parameters(logn, poly);
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

            for (int logn = 1; logn <= std::min(max_logn_, pool_max_logn);
                 logn++)
                for (ReductionPolynomial poly : kPolys)
                    for (type dir : kDirections)
                    {
                        NTTParameters<T> params(logn, poly);
                        single_case(report, params, "pool", dir);
                    }
            return report.finish();
        }

        // Modulus<T> precomputes the bit length and the Barrett constant
        // mu = floor(2^(2 * bit + 1) / value). They are checked for values next
        // to powers of two, where floating point rounding goes wrong first.
        // Moduli above the supported size (30 bits for Data32, 61 bits for
        // Data64) must be rejected, and transforms are run with the largest
        // supported NTT-friendly prime.
        int modulus()
        {
            TestReport report(suite_name("modulus"));
            constexpr bool is32 = std::is_same<T, Data32>::value;
            constexpr int max_bits = Modulus<T>::max_bit;
            constexpr int type_bits = 8 * sizeof(T);

            std::vector<T> values;
            for (int k = 2; k <= type_bits; k++)
            {
                const T power = (k == type_bits) ? T(0) : T(1) << k;
                for (T distance : {T(1), T(3), T(57), T(255)})
                {
                    if (k == type_bits || distance < (power >> 1))
                    {
                        values.push_back(power - distance);
                    }
                }
                values.push_back((T(1) << (k - 1)) + 1);
            }

            for (T value : values)
            {
                T bits = 0;
                for (T v = value; v != 0; v >>= 1)
                {
                    bits++;
                }
                const std::string name =
                    "Modulus(" + std::to_string(value) + ")";
                if (bits > static_cast<T>(max_bits))
                {
                    bool rejected = false;
                    try
                    {
                        Modulus<T> m(value);
                    }
                    catch (const std::invalid_argument&)
                    {
                        rejected = true;
                    }
                    report.expect(rejected, name,
                                  std::to_string(bits) +
                                      "-bit modulus was not rejected");
                    continue;
                }

                const Modulus<T> m(value);
                const T mu =
                    static_cast<T>((Wide<T>(1) << (2 * bits + 1)) / value);
                report.expect(m.bit == bits && m.mu == mu, name,
                              "got bit=" + std::to_string(m.bit) +
                                  " mu=" + std::to_string(m.mu) +
                                  ", expected bit=" + std::to_string(bits) +
                                  " mu=" + std::to_string(mu));
            }

            auto largest = make_fields<T>(1, max_bits, is32 ? 23 : 20);
            for (int logn = 1;
                 logn <= std::min(max_logn_, largest[0].max_logn()); logn++)
                for (ReductionPolynomial poly : kPolys)
                    for (type dir : kDirections)
                    {
                        NTTParameters<T> params =
                            largest[0].parameters(logn, poly);
                        single_case(report, params, "largest prime", dir);
                    }
            return report.finish();
        }

        // Invalid batch sizes are rejected before any kernel is launched (the
        // calls use null device pointers).
        int validation()
        {
            TestReport report(suite_name("validation"));
            const Modulus<T> modulus(fields_[0].prime);
            auto expect_rejected =
                [&](const std::string& name, const std::function<void()>& call)
            {
                bool rejected = false;
                try
                {
                    call();
                }
                catch (const std::invalid_argument&)
                {
                    rejected = true;
                }
                catch (const std::exception& e)
                {
                    report.error(name, std::string("unexpected: ") + e.what());
                    return;
                }
                report.expect(rejected, name, "call was not rejected");
            };

            struct Case
            {
                std::string name;
                int logn;
                int batch;
                NTTLayout layout;
            };
            const std::vector<Case> cases = {
                {"batch=0", 12, 0, PerPolynomial},
                {"batch=-1", 12, -1, PerPolynomial},
                {"2^31 + 2^20 coefficients", 20, 2049, PerPolynomial},
                {"2^33 coefficients", 28, 32, PerPolynomial},
                {"per_coefficient batch=3", 8, 3, PerCoefficient},
                {"per_coefficient batch=24", 6, 24, PerCoefficient},
            };

            T* null_data = nullptr;
            Root<T>* null_roots = nullptr;
            Modulus<T>* null_moduli = nullptr;
            int* null_order = nullptr;
            for (const Case& c : cases)
                for (type dir : kDirections)
                {
                    ntt_configuration<T> cfg = {
                        .n_power = c.logn,
                        .ntt_type = dir,
                        .ntt_layout = c.layout,
                        .reduction_poly = ReductionPolynomial::X_N_plus,
                        .mod_inverse = 1,
                        .stream = 0};
                    ntt_rns_configuration<T> cfg_rns = {
                        .n_power = c.logn,
                        .ntt_type = dir,
                        .ntt_layout = c.layout,
                        .reduction_poly = ReductionPolynomial::X_N_plus,
                        .mod_inverse = nullptr,
                        .stream = 0};
                    const std::string suffix =
                        (dir == FORWARD ? " ntt " : " intt ") + c.name;

                    expect_rejected(
                        "single" + suffix,
                        [&]
                        {
                            if (dir == FORWARD)
                                GPU_NTT(null_data, null_data, null_roots,
                                        modulus, cfg, c.batch);
                            else
                                GPU_INTT(null_data, null_data, null_roots,
                                         modulus, cfg, c.batch);
                        });
                    expect_rejected(
                        "rns" + suffix,
                        [&]
                        {
                            if (dir == FORWARD)
                                GPU_NTT(null_data, null_data, null_roots,
                                        null_moduli, cfg_rns, c.batch, 1);
                            else
                                GPU_INTT(null_data, null_data, null_roots,
                                         null_moduli, cfg_rns, c.batch, 1);
                        });
                    if (c.layout == PerPolynomial)
                    {
                        expect_rejected("modulus_ordered" + suffix,
                                        [&]
                                        {
                                            GPU_NTT_Modulus_Ordered(
                                                null_data, null_data,
                                                null_roots, null_moduli,
                                                cfg_rns, c.batch, 1,
                                                null_order);
                                        });
                        expect_rejected("poly_ordered" + suffix,
                                        [&]
                                        {
                                            GPU_NTT_Poly_Ordered(
                                                null_data, null_data,
                                                null_roots, null_moduli,
                                                cfg_rns, c.batch, 1,
                                                null_order);
                                        });
                    }
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
                        auto cpu_of = [&](int j) -> NTTCPU<T>&
                        { return cpus[j % mod_count]; };

                        std::vector<T> input = random_input(
                            total, n,
                            [&](int j)
                            { return fields_[j % mod_count].prime; });
                        std::vector<T> expected =
                            reference(input, logn, dir, cpu_of);

                        const std::string name =
                            case_name("rns", dir, logn, poly, batch);
                        try
                        {
                            std::vector<Modulus<T>> moduli;
                            std::vector<T> n_inverse;
                            for (auto& p : params)
                            {
                                moduli.push_back(p.modulus);
                                n_inverse.push_back(p.n_inv);
                            }
                            DeviceBuffer<Modulus<T>> d_moduli(moduli);
                            DeviceBuffer<T> d_n_inverse(n_inverse);
                            DeviceBuffer<Root<T>> roots(
                                concat_root_tables(params, dir == INVERSE, n));

                            ntt_rns_configuration<T> cfg = {
                                .n_power = logn,
                                .ntt_type = dir,
                                .ntt_layout = PerPolynomial,
                                .reduction_poly = poly,
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
                            auto prime_of = [&](int j)
                            { return fields_[j % count].prime; };

                            std::vector<T> input =
                                random_input(total, n, prime_of);
                            std::vector<T> expected =
                                reference(input, logn, dir,
                                          [&](int j) -> NTTCPU<T>&
                                          { return cpus[j % count]; });

                            const std::string name = case_name(
                                use_rns ? "signed rns" : "signed single", dir,
                                logn, poly, batch);
                            try
                            {
                                std::vector<Modulus<T>> moduli;
                                std::vector<T> n_inverse;
                                for (auto& p : params)
                                {
                                    moduli.push_back(p.modulus);
                                    n_inverse.push_back(p.n_inv);
                                }
                                DeviceBuffer<Modulus<T>> d_moduli(moduli);
                                DeviceBuffer<T> d_n_inverse(n_inverse);
                                DeviceBuffer<Root<T>> roots(concat_root_tables(
                                    params, dir == INVERSE, n));

                                ntt_configuration<T> cfg = {
                                    .n_power = logn,
                                    .ntt_type = dir,
                                    .ntt_layout = PerPolynomial,
                                    .reduction_poly = poly,
                                    .mod_inverse = params[0].n_inv,
                                    .stream = 0};
                                ntt_rns_configuration<T> cfg_rns = {
                                    .n_power = logn,
                                    .ntt_type = dir,
                                    .ntt_layout = PerPolynomial,
                                    .reduction_poly = poly,
                                    .mod_inverse = d_n_inverse.get(),
                                    .stream = 0};

                                if (dir == FORWARD)
                                {
                                    std::vector<S> signed_input(total);
                                    for (size_t i = 0; i < total; i++)
                                    {
                                        signed_input[i] = to_centered(
                                            input[i], prime_of(i / n));
                                    }
                                    DeviceBuffer<S> in(signed_input);
                                    DeviceBuffer<T> out(total);
                                    if (use_rns)
                                        GPU_NTT(in.get(), out.get(),
                                                roots.get(), d_moduli.get(),
                                                cfg_rns, batch, mod_count);
                                    else
                                        GPU_NTT(in.get(), out.get(),
                                                roots.get(), params[0].modulus,
                                                cfg, batch);
                                    sync();
                                    report.check(name, out.download(),
                                                 expected);
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
                                        GPU_INTT(in.get(), out.get(),
                                                 roots.get(), d_moduli.get(),
                                                 cfg_rns, batch, mod_count);
                                    else
                                        GPU_INTT(in.get(), out.get(),
                                                 roots.get(), params[0].modulus,
                                                 cfg, batch);
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
            TestReport report(suite_name(use_rns ? "per_coefficient_rns"
                                                 : "per_coefficient"));
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
                            [&](int j) -> NTTCPU<T>&
                            { return cpus[j % count]; });

                        const std::string name =
                            case_name(use_rns ? "per_coefficient rns"
                                              : "per_coefficient single",
                                      dir, logn, poly, batch);
                        try
                        {
                            std::vector<Modulus<T>> moduli;
                            std::vector<T> n_inverse;
                            for (auto& p : params)
                            {
                                moduli.push_back(p.modulus);
                                n_inverse.push_back(p.n_inv);
                            }
                            DeviceBuffer<Modulus<T>> d_moduli(moduli);
                            DeviceBuffer<T> d_n_inverse(n_inverse);
                            DeviceBuffer<Root<T>> roots(
                                concat_root_tables(params, dir == INVERSE, n));

                            ntt_configuration<T> cfg = {
                                .n_power = logn,
                                .ntt_type = dir,
                                .ntt_layout = PerCoefficient,
                                .reduction_poly = poly,
                                .mod_inverse = params[0].n_inv,
                                .stream = 0};
                            ntt_rns_configuration<T> cfg_rns = {
                                .n_power = logn,
                                .ntt_type = dir,
                                .ntt_layout = PerCoefficient,
                                .reduction_poly = poly,
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
                                                d_moduli.get(), cfg_rns, batch,
                                                mod_count);
                            }
                            else if (use_rns)
                            {
                                GPU_INTT(in.get(), out.get(), roots.get(),
                                         d_moduli.get(), cfg_rns, batch,
                                         mod_count);
                                GPU_INTT_Inplace(inout.get(), roots.get(),
                                                 d_moduli.get(), cfg_rns, batch,
                                                 mod_count);
                            }
                            else if (dir == FORWARD)
                            {
                                GPU_NTT(in.get(), out.get(), roots.get(),
                                        params[0].modulus, cfg, batch);
                                GPU_NTT_Inplace(inout.get(), roots.get(),
                                                params[0].modulus, cfg, batch);
                            }
                            else
                            {
                                GPU_INTT(in.get(), out.get(), roots.get(),
                                         params[0].modulus, cfg, batch);
                                GPU_INTT_Inplace(inout.get(), roots.get(),
                                                 params[0].modulus, cfg, batch);
                            }
                            sync();
                            report.check(name + " out-of-place", out.download(),
                                         expected);
                            report.check(name + " in-place", inout.download(),
                                         expected);

                            // Signed input (NTT) / signed output (INTT).
                            auto prime_of = [&](size_t i)
                            { return fields_[(i % batch) % count].prime; };
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
                                            roots.get(), params[0].modulus, cfg,
                                            batch);
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
                                             s_out.download(), signed_expected);
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
                        auto prime_index = [&](int z)
                        { return order[z % mod_count]; };

                        std::vector<T> input = random_input(
                            total, n,
                            [&](int z)
                            { return fields_[prime_index(z)].prime; });
                        std::vector<T> expected =
                            reference(input, logn, dir,
                                      [&](int z) -> NTTCPU<T>&
                                      { return cpus[prime_index(z)]; });

                        const std::string name = case_name(
                            "modulus_ordered", dir, logn, poly, batch);
                        try
                        {
                            std::vector<Modulus<T>> moduli;
                            std::vector<T> n_inverse;
                            for (auto& p : params)
                            {
                                moduli.push_back(p.modulus);
                                n_inverse.push_back(p.n_inv);
                            }
                            DeviceBuffer<Modulus<T>> d_moduli(moduli);
                            DeviceBuffer<T> d_n_inverse(n_inverse);
                            DeviceBuffer<int> d_order(order);
                            DeviceBuffer<Root<T>> roots(
                                concat_root_tables(params, dir == INVERSE, n));

                            ntt_rns_configuration<T> cfg = {
                                .n_power = logn,
                                .ntt_type = dir,
                                .ntt_layout = PerPolynomial,
                                .reduction_poly = poly,
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
                            total, n,
                            [&](int p)
                            { return fields_[modulus_of[p]].prime; });
                        std::vector<T> expected =
                            reference(input, logn, dir,
                                      [&](int p) -> NTTCPU<T>&
                                      { return cpus[modulus_of[p]]; });

                        const std::string name =
                            case_name("poly_ordered", dir, logn, poly, batch);
                        try
                        {
                            std::vector<Modulus<T>> moduli;
                            std::vector<T> n_inverse;
                            for (auto& p : params)
                            {
                                moduli.push_back(p.modulus);
                                n_inverse.push_back(p.n_inv);
                            }
                            DeviceBuffer<Modulus<T>> d_moduli(moduli);
                            DeviceBuffer<T> d_n_inverse(n_inverse);
                            DeviceBuffer<int> d_order(order);
                            DeviceBuffer<Root<T>> roots(
                                concat_root_tables(params, dir == INVERSE, n));

                            ntt_rns_configuration<T> cfg = {
                                .n_power = logn,
                                .ntt_type = dir,
                                .ntt_layout = PerPolynomial,
                                .reduction_poly = poly,
                                .mod_inverse = d_n_inverse.get(),
                                .stream = 0};

                            DeviceBuffer<T> in(input);
                            DeviceBuffer<T> out(total);
                            GPU_NTT_Poly_Ordered(in.get(), out.get(),
                                                 roots.get(), d_moduli.get(),
                                                 cfg, batch, mod_count,
                                                 d_order.get());
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

        // N = 2^25, the smallest ring size that runs the _ kernel variants
        // (swapped grid axes). The CPU reference is too slow at this size, so
        // every variant is checked with a forward/inverse round trip, and the
        // in-place results (same buffer as input and output) are compared
        // with the out-of-place ones.
        int large()
        {
            TestReport report(suite_name("large"));
            const int logn = 25;
            const int mod_count = 2;
            const int batch = 2;
            const size_t n = size_t(1) << logn;
            const size_t total = batch * n;
            const std::vector<int> modulus_order = {1, 0};
            const std::vector<int> poly_order = {1, 0};

            // Data32 has a single prime below 2^30 with a 2^26-th root of
            // unity, so its RNS cases use that prime twice.
            std::vector<PrimeField<T>> fields =
                make_fields<T>(std::is_same<T, Data32>::value ? 1 : mod_count,
                               Modulus<T>::max_bit, logn + 1);
            while (static_cast<int>(fields.size()) < mod_count)
            {
                fields.push_back(fields[0]);
            }
            T min_prime = fields[0].prime;
            for (const auto& f : fields)
            {
                min_prime = std::min(min_prime, f.prime);
            }

            for (ReductionPolynomial poly : kPolys)
            {
                const std::string suffix =
                    std::string(" logn=") + std::to_string(logn) +
                    (poly == ReductionPolynomial::X_N_minus ? " X^N-1"
                                                            : " X^N+1") +
                    " batch=" + std::to_string(batch);
                try
                {
                    std::vector<NTTParameters<T>> params;
                    std::vector<Modulus<T>> moduli;
                    std::vector<T> n_inverse;
                    for (const auto& f : fields)
                    {
                        params.push_back(f.parameters(logn, poly));
                        moduli.push_back(params.back().modulus);
                        n_inverse.push_back(params.back().n_inv);
                    }
                    DeviceBuffer<Modulus<T>> d_moduli(moduli);
                    DeviceBuffer<T> d_n_inverse(n_inverse);
                    DeviceBuffer<int> d_modulus_order(modulus_order);
                    DeviceBuffer<int> d_poly_order(poly_order);
                    // The single modulus cases use the first table.
                    DeviceBuffer<Root<T>> forward_rns(
                        concat_root_tables(params, false, n));
                    DeviceBuffer<Root<T>> inverse_rns(
                        concat_root_tables(params, true, n));
                    Root<T>* forward_single = forward_rns.get();
                    Root<T>* inverse_single = inverse_rns.get();

                    auto cfg = [&](type dir)
                    {
                        return ntt_configuration<T>{.n_power = logn,
                                                    .ntt_type = dir,
                                                    .ntt_layout = PerPolynomial,
                                                    .reduction_poly = poly,
                                                    .mod_inverse =
                                                        params[0].n_inv,
                                                    .stream = 0};
                    };
                    auto cfg_rns = [&](type dir)
                    {
                        return ntt_rns_configuration<T>{
                            .n_power = logn,
                            .ntt_type = dir,
                            .ntt_layout = PerPolynomial,
                            .reduction_poly = poly,
                            .mod_inverse = d_n_inverse.get(),
                            .stream = 0};
                    };

                    const std::vector<T> input =
                        random_values<T>(rng_, total, min_prime);

                    using Op = std::function<void(T*, T*)>;
                    auto check = [&](const std::string& variant,
                                     const Op& forward, const Op& inverse)
                    {
                        std::vector<T> transformed;
                        {
                            DeviceBuffer<T> in(input);
                            DeviceBuffer<T> out(total);
                            DeviceBuffer<T> back(total);
                            forward(in.get(), out.get());
                            inverse(out.get(), back.get());
                            sync();
                            transformed = out.download();
                            report.check(variant + " round trip" + suffix,
                                         back.download(), input);
                        }
                        DeviceBuffer<T> inout(input);
                        forward(inout.get(), inout.get());
                        sync();
                        report.check(variant + " ntt in-place" + suffix,
                                     inout.download(), transformed);
                        inverse(inout.get(), inout.get());
                        sync();
                        report.check(variant + " intt in-place" + suffix,
                                     inout.download(), input);
                    };

                    check(
                        "single",
                        [&](T* in, T* out) {
                            GPU_NTT(in, out, forward_single, params[0].modulus,
                                    cfg(FORWARD), batch);
                        },
                        [&](T* in, T* out) {
                            GPU_INTT(in, out, inverse_single, params[0].modulus,
                                     cfg(INVERSE), batch);
                        });
                    check(
                        "rns",
                        [&](T* in, T* out)
                        {
                            GPU_NTT(in, out, forward_rns.get(), d_moduli.get(),
                                    cfg_rns(FORWARD), batch, mod_count);
                        },
                        [&](T* in, T* out)
                        {
                            GPU_INTT(in, out, inverse_rns.get(), d_moduli.get(),
                                     cfg_rns(INVERSE), batch, mod_count);
                        });
                    check(
                        "modulus_ordered",
                        [&](T* in, T* out)
                        {
                            GPU_NTT_Modulus_Ordered(
                                in, out, forward_rns.get(), d_moduli.get(),
                                cfg_rns(FORWARD), batch, mod_count,
                                d_modulus_order.get());
                        },
                        [&](T* in, T* out)
                        {
                            GPU_NTT_Modulus_Ordered(
                                in, out, inverse_rns.get(), d_moduli.get(),
                                cfg_rns(INVERSE), batch, mod_count,
                                d_modulus_order.get());
                        });
                    check(
                        "poly_ordered",
                        [&](T* in, T* out)
                        {
                            GPU_NTT_Poly_Ordered(in, out, forward_rns.get(),
                                                 d_moduli.get(),
                                                 cfg_rns(FORWARD), batch,
                                                 mod_count, d_poly_order.get());
                        },
                        [&](T* in, T* out)
                        {
                            GPU_NTT_Poly_Ordered(in, out, inverse_rns.get(),
                                                 d_moduli.get(),
                                                 cfg_rns(INVERSE), batch,
                                                 mod_count, d_poly_order.get());
                        });

                    // Signed input to the NTT and signed output of the INTT.
                    for (bool use_rns : {false, true})
                    {
                        const int count = use_rns ? mod_count : 1;
                        std::vector<S> signed_input(total);
                        for (size_t i = 0; i < total; i++)
                        {
                            signed_input[i] = to_centered(
                                input[i], fields[(i / n) % count].prime);
                        }
                        DeviceBuffer<S> in(signed_input);
                        DeviceBuffer<T> out(total);
                        DeviceBuffer<S> back(total);
                        if (use_rns)
                        {
                            GPU_NTT(in.get(), out.get(), forward_rns.get(),
                                    d_moduli.get(), cfg_rns(FORWARD), batch,
                                    mod_count);
                            GPU_INTT(out.get(), back.get(), inverse_rns.get(),
                                     d_moduli.get(), cfg_rns(INVERSE), batch,
                                     mod_count);
                        }
                        else
                        {
                            GPU_NTT(in.get(), out.get(), forward_single,
                                    params[0].modulus, cfg(FORWARD), batch);
                            GPU_INTT(out.get(), back.get(), inverse_single,
                                     params[0].modulus, cfg(INVERSE), batch);
                        }
                        sync();
                        report.check(std::string(use_rns ? "signed rns"
                                                         : "signed single") +
                                         " round trip" + suffix,
                                     back.download(), signed_input);
                    }
                }
                catch (const std::exception& e)
                {
                    report.error("large" + suffix, e.what());
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
