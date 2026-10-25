// Copyright 2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef GPUNTT_TEST_UTIL_H
#define GPUNTT_TEST_UTIL_H

#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "gpuntt/ntt_merge/ntt.cuh"
#include "gpuntt/ntt_merge/ntt_cpu.cuh"

namespace gpuntt_test
{
    using namespace gpuntt;

    template <typename T>
    using Wide = typename std::conditional<std::is_same<T, Data32>::value,
                                           Data64, __uint128_t>::type;

    template <typename T> T mulmod(T a, T b, T p)
    {
        return static_cast<T>((static_cast<Wide<T>>(a) * b) % p);
    }

    template <typename T> T powmod(T base, std::uint64_t exponent, T p)
    {
        T result = 1;
        base %= p;
        while (exponent > 0)
        {
            if (exponent & 1)
            {
                result = mulmod(result, base, p);
            }
            base = mulmod(base, base, p);
            exponent >>= 1;
        }
        return result;
    }

    // Deterministic Miller-Rabin for 32-bit and 64-bit inputs.
    template <typename T> bool is_prime(T n)
    {
        if (n < 2)
        {
            return false;
        }
        for (T small : {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37})
        {
            if (n % small == 0)
            {
                return n == small;
            }
        }

        T d = n - 1;
        int s = 0;
        while ((d & 1) == 0)
        {
            d >>= 1;
            s++;
        }

        for (T a : {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37})
        {
            T x = powmod(a, d, n);
            if (x == 1 || x == n - 1)
            {
                continue;
            }
            bool composite = true;
            for (int r = 1; r < s; r++)
            {
                x = mulmod(x, x, n);
                if (x == n - 1)
                {
                    composite = false;
                    break;
                }
            }
            if (composite)
            {
                return false;
            }
        }
        return true;
    }

    // A prime field with a root of unity of order 2^two_adicity, from which
    // the parameters for every supported ring size are derived.
    template <typename T> struct PrimeField
    {
        T prime;
        T root;
        int two_adicity;

        // Largest log2(N) for which both X^N - 1 and X^N + 1 are supported.
        int max_logn() const { return two_adicity - 1; }

        NTTParameters<T> parameters(int logn, ReductionPolynomial poly) const
        {
            // psi: primitive 2N-th root of unity, omega: primitive N-th root.
            T psi = powmod(root, std::uint64_t(1) << (two_adicity - logn - 1),
                           prime);
            T omega = mulmod(psi, psi, prime);
            return NTTParameters<T>(
                logn, NTTFactors<T>(Modulus<T>(prime), omega, psi), poly);
        }
    };

    // Returns `count` distinct prime fields p < 2^max_bits with
    // 2^two_adicity | p - 1, largest prime first. max_bits must not exceed
    // Modulus<T>::max_bit (30 for Data32, 61 for Data64).
    template <typename T>
    std::vector<PrimeField<T>> make_fields(int count, int max_bits,
                                           int two_adicity)
    {

        std::vector<PrimeField<T>> fields;
        const T step = T(1) << two_adicity;
        for (T k = ((T(1) << max_bits) - 1) / step;
             k > 0 && static_cast<int>(fields.size()) < count; k--)
        {
            T p = k * step + 1;
            if (!is_prime(p))
            {
                continue;
            }
            for (T x = 2;; x++)
            {
                T c = powmod(x, (p - 1) >> two_adicity, p);
                if (powmod(c, std::uint64_t(1) << (two_adicity - 1), p) ==
                    p - 1)
                {
                    fields.push_back({p, c, two_adicity});
                    break;
                }
            }
        }
        if (static_cast<int>(fields.size()) < count)
        {
            throw std::runtime_error("Not enough NTT-friendly primes.");
        }
        return fields;
    }

    template <typename T> std::vector<PrimeField<T>> make_fields(int count)
    {
        constexpr bool is32 = std::is_same<T, Data32>::value;
        return make_fields<T>(count, Modulus<T>::max_bit, is32 ? 23 : 29);
    }

    template <typename T> class DeviceBuffer
    {
      public:
        explicit DeviceBuffer(size_t count) : count_(count)
        {
            GPUNTT_CUDA_CHECK(
                cudaMalloc(&ptr_, (count > 0 ? count : 1) * sizeof(T)));
        }

        explicit DeviceBuffer(const std::vector<T>& host)
            : DeviceBuffer(host.size())
        {
            upload(host);
        }

        DeviceBuffer(const DeviceBuffer&) = delete;
        DeviceBuffer& operator=(const DeviceBuffer&) = delete;

        ~DeviceBuffer() { cudaFree(ptr_); }

        T* get() const { return ptr_; }

        void upload(const std::vector<T>& host)
        {
            GPUNTT_CUDA_CHECK(cudaMemcpy(ptr_, host.data(),
                                         count_ * sizeof(T),
                                         cudaMemcpyHostToDevice));
        }

        std::vector<T> download() const
        {
            std::vector<T> host(count_);
            GPUNTT_CUDA_CHECK(cudaMemcpy(host.data(), ptr_, count_ * sizeof(T),
                                         cudaMemcpyDeviceToHost));
            return host;
        }

      private:
        T* ptr_ = nullptr;
        size_t count_;
    };

    template <typename T>
    std::vector<T> random_values(std::mt19937_64& rng, size_t count, T prime)
    {
        std::uniform_int_distribution<std::uint64_t> dist(0, prime - 1);
        std::vector<T> values(count);
        for (auto& v : values)
        {
            v = static_cast<T>(dist(rng));
        }
        return values;
    }

    // Maps [0, p) to the centered range used by GPU-NTT's signed kernels.
    template <typename T>
    typename std::make_signed<T>::type to_centered(T value, T prime)
    {
        using S = typename std::make_signed<T>::type;
        return (value > (prime >> 1))
                   ? static_cast<S>(value) - static_cast<S>(prime)
                   : static_cast<S>(value);
    }

    // Root of unity tables of several moduli, one table per `stride` entries.
    template <typename T>
    std::vector<Root<T>>
    concat_root_tables(std::vector<NTTParameters<T>>& parameters, bool inverse,
                       size_t stride)
    {
        std::vector<Root<T>> table(stride * parameters.size(), 0);
        for (size_t m = 0; m < parameters.size(); m++)
        {
            std::vector<Root<T>> single =
                parameters[m].gpu_root_of_unity_table_generator(
                    inverse ? parameters[m].inverse_root_of_unity_table
                            : parameters[m].forward_root_of_unity_table);
            std::copy(single.begin(), single.end(),
                      table.begin() + m * stride);
        }
        return table;
    }

    class TestReport
    {
      public:
        explicit TestReport(std::string suite) : suite_(std::move(suite)) {}

        template <typename V>
        void check(const std::string& name, const std::vector<V>& got,
                   const std::vector<V>& expected)
        {
            cases_++;
            if (got.size() != expected.size())
            {
                fail(name, "size mismatch");
                return;
            }
            size_t mismatches = 0;
            size_t first = 0;
            for (size_t i = 0; i < got.size(); i++)
            {
                if (got[i] != expected[i])
                {
                    if (mismatches == 0)
                    {
                        first = i;
                    }
                    mismatches++;
                }
            }
            if (mismatches > 0)
            {
                fail(name, std::to_string(mismatches) + " of " +
                               std::to_string(got.size()) +
                               " values differ, first at index " +
                               std::to_string(first) + ": got " +
                               to_string(got[first]) + ", expected " +
                               to_string(expected[first]));
            }
        }

        void expect(bool condition, const std::string& name,
                    const std::string& message)
        {
            cases_++;
            if (!condition)
            {
                fail(name, message);
            }
        }

        void error(const std::string& name, const std::string& what)
        {
            cases_++;
            fail(name, what);
        }

        int finish() const
        {
            std::cout << suite_ << ": " << cases_ << " cases, " << failures_
                      << " failed" << std::endl;
            return failures_ == 0 ? 0 : 1;
        }

      private:
        template <typename V> static std::string to_string(V value)
        {
            if constexpr (std::is_signed<V>::value)
            {
                return std::to_string(static_cast<long long>(value));
            }
            else
            {
                return std::to_string(static_cast<unsigned long long>(value));
            }
        }

        void fail(const std::string& name, const std::string& message)
        {
            failures_++;
            std::cout << "FAILED " << suite_ << " " << name << ": " << message
                      << std::endl;
        }

        std::string suite_;
        int cases_ = 0;
        int failures_ = 0;
    };

} // namespace gpuntt_test
#endif // GPUNTT_TEST_UTIL_H
