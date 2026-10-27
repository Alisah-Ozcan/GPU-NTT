// Copyright 2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

// Device API with RNS: polynomial multiplication modulo X^N + 1 and
// Q = q_0 * q_1 * q_2 (N = 2^10), one kernel for NTT, product and INTT.
//
// An RNS polynomial is stored as one polynomial per prime: residue i of RNS
// polynomial j is polynomial j * kModCount + i, so polynomial z uses prime
// z % kModCount, as in the host RNS API. The root tables of all primes are
// built on the GPU from primes of fields.cuh, N entries apart. The result is
// checked against schoolbook multiplication for every prime.
//
// Usage: gpu_device_api_rns_examples [RNS_POLY_COUNT]

#include <cstdlib>
#include <random>

#include "gpuntt/ntt_merge/descriptor.cuh"
#include "gpuntt/ntt_merge/ntt.cuh"
#include "gpuntt/ntt_merge/ntt_cpu.cuh"
#include "example_util.cuh"

using namespace std;
using namespace gpuntt;
using namespace gpuntt_example;

typedef Data64 TestDataType;

constexpr int kLogN = 10;
constexpr int kModCount = 3;

using Forward =
    decltype(Size<(1 << kLogN)>() + Direction<FORWARD>() +
             Precision<TestDataType>() + ReductionPoly<X_N_plus>() + Block());
using Inverse =
    decltype(Size<(1 << kLogN)>() + Direction<INVERSE>() +
             Precision<TestDataType>() + ReductionPoly<X_N_plus>() + Block());

// All polynomials of a block share one modulus, so a block must hold one
// polynomial. That is the default for N = 2^10; smaller N need
// BatchesPerBlock<1>().
static_assert(Forward::batches_per_block == 1);

// a <- INTT(NTT(a) * b_hat) for polynomial blockIdx.x, with the modulus,
// roots and N^-1 of its prime. b_hat is already in the NTT domain.
__global__ void rns_multiply_kernel(TestDataType* a, const TestDataType* b_hat,
                                    const TestDataType* forward_roots,
                                    const TestDataType* inverse_roots,
                                    const Modulus<TestDataType>* moduli,
                                    const TestDataType* n_inverses)
{
    extern __shared__ char shared_memory_raw[];
    TestDataType* shared_memory =
        reinterpret_cast<TestDataType*>(shared_memory_raw);

    const int prime = blockIdx.x % kModCount;
    const Modulus<TestDataType> modulus = moduli[prime];
    const size_t root_offset = size_t(prime) << kLogN;
    const size_t offset = size_t(blockIdx.x) * Forward::size;

    Forward().load(shared_memory, a + offset, modulus);
    Forward().execute(shared_memory, forward_roots + root_offset, modulus);

    for (int i = threadIdx.x + threadIdx.y * blockDim.x; i < Forward::size;
         i += blockDim.x * blockDim.y)
    {
        shared_memory[i] = OPERATOR_GPU<TestDataType>::mult(
            shared_memory[i], b_hat[offset + i], modulus);
    }
    __syncthreads();

    Inverse().execute(shared_memory, inverse_roots + root_offset, modulus);
    Inverse().store(shared_memory, a + offset, modulus, n_inverses[prime]);
}

template <typename T> T* to_device(const vector<T>& values)
{
    T* data;
    GPUNTT_CUDA_CHECK(cudaMalloc(&data, values.size() * sizeof(T)));
    GPUNTT_CUDA_CHECK(cudaMemcpy(data, values.data(), values.size() * sizeof(T),
                                 cudaMemcpyHostToDevice));
    return data;
}

int main(int argc, char* argv[])
{
    CudaDevice();

    const int rns_count = (argc > 1) ? atoi(argv[1]) : 2;
    const int poly_count = rns_count * kModCount;
    const int n = Forward::size;
    const size_t total = size_t(poly_count) * n;

    // Three 61-bit primes and their moduli and N^-1.
    vector<NTTPrime<TestDataType>> primes(ntt_primes64 + 1,
                                          ntt_primes64 + 1 + kModCount);
    vector<Modulus<TestDataType>> moduli;
    vector<TestDataType> n_inverses;
    for (const NTTPrime<TestDataType>& prime : primes)
    {
        moduli.push_back(Modulus<TestDataType>(prime.value));
        n_inverses.push_back(Inverse::n_inv(moduli.back()));
    }

    // Root tables of all primes, prime i at offset i * N.
    TestDataType *forward_roots, *inverse_roots;
    GPUNTT_CUDA_CHECK(cudaMalloc(&forward_roots,
                                 size_t(kModCount) * n * sizeof(TestDataType)));
    GPUNTT_CUDA_CHECK(cudaMalloc(&inverse_roots,
                                 size_t(kModCount) * n * sizeof(TestDataType)));
    for (int i = 0; i < kModCount; i++)
    {
        Forward::make_twiddles(forward_roots + size_t(i) * n, primes[i]);
        Inverse::make_twiddles(inverse_roots + size_t(i) * n, primes[i]);
    }

    // Random a and b; coefficient k of polynomial z is below its prime.
    std::mt19937_64 gen(0);
    vector<TestDataType> a(total), b(total);
    for (size_t k = 0; k < total; k++)
    {
        const TestDataType q = primes[(k / n) % kModCount].value;
        a[k] = gen() % q;
        b[k] = gen() % q;
    }

    TestDataType* a_device = to_device(a);
    TestDataType* b_hat_device = to_device(b);
    Modulus<TestDataType>* moduli_device = to_device(moduli);
    TestDataType* n_inverses_device = to_device(n_inverses);

    // b_hat = NTT(b) with the host RNS API, on the same root tables.
    ntt_rns_configuration<TestDataType> cfg = {.n_power = kLogN,
                                               .ntt_type = FORWARD,
                                               .ntt_layout = PerPolynomial,
                                               .reduction_poly = X_N_plus,
                                               .mod_inverse = n_inverses_device,
                                               .stream = 0};
    GPU_NTT_Inplace(b_hat_device, forward_roots, moduli_device, cfg, poly_count,
                    kModCount);

    rns_multiply_kernel<<<Forward::grid(poly_count), Forward::block_dim,
                          Forward::shared_memory_size>>>(
        a_device, b_hat_device, forward_roots, inverse_roots, moduli_device,
        n_inverses_device);
    GPUNTT_CUDA_CHECK(cudaGetLastError());

    vector<TestDataType> product(total);
    GPUNTT_CUDA_CHECK(cudaMemcpy(product.data(), a_device,
                                 total * sizeof(TestDataType),
                                 cudaMemcpyDeviceToHost));

    bool correct = true;
    for (int z = 0; z < poly_count && correct; z++)
    {
        vector<TestDataType> a_z(a.begin() + size_t(z) * n,
                                 a.begin() + size_t(z + 1) * n);
        vector<TestDataType> b_z(b.begin() + size_t(z) * n,
                                 b.begin() + size_t(z + 1) * n);
        vector<TestDataType> expected = schoolbook_poly_multiplication(
            a_z, b_z, moduli[z % kModCount], ReductionPolynomial::X_N_plus);
        correct =
            check_result(product.data() + size_t(z) * n, expected.data(), n);
        if (!correct)
        {
            cout << "(in RNS polynomial " << z / kModCount << ", prime "
                 << z % kModCount << ")" << endl;
        }
    }
    if (correct)
    {
        cout << "All Correct for RNS polynomial multiplication, N = 2^" << kLogN
             << ", " << kModCount << " primes, " << rns_count
             << " RNS polynomials." << endl;
    }

    GPUNTT_CUDA_CHECK(cudaFree(a_device));
    GPUNTT_CUDA_CHECK(cudaFree(b_hat_device));
    GPUNTT_CUDA_CHECK(cudaFree(moduli_device));
    GPUNTT_CUDA_CHECK(cudaFree(n_inverses_device));
    GPUNTT_CUDA_CHECK(cudaFree(forward_roots));
    GPUNTT_CUDA_CHECK(cudaFree(inverse_roots));

    return correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
