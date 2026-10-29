// Copyright 2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

// Device API examples: NTT inside user kernels.
//
// 1. Polynomial multiplication modulo X^N + 1 in one kernel (N = 2^10):
//    forward NTT of a, pointwise product with NTT(b), inverse NTT. Checked
//    against schoolbook multiplication on the CPU.
// 2. Forward NTT of N = 2^16 polynomials, one kernel per stage. Checked
//    against the CPU NTT, and NTT::launch, also with 128 threads per kernel
//    (BlockDim<128>), against the stage kernels.
// 3. Polynomial multiplication modulo X^N + 1 with stage kernels (N = 2^12,
//    two stages per direction) in three kernels: the last forward stage,
//    the pointwise product and the first inverse stage share one kernel.
//    Checked against schoolbook multiplication on the CPU.
// 4. Multiplication by a ternary polynomial stored as int8_t (N = 2^10): the
//    kernel writes it to shared memory itself, into a second buffer after
//    the description's shared memory, and transforms both buffers. Checked
//    against schoolbook multiplication on the CPU.
// 5. The forward NTT of 2 with T = 128, 256, 512 and 1024 threads in every
//    kernel (BlockDim<T>), all with the same stage kernel. Checked against
//    the CPU NTT.
//
// Usage: gpu_device_api_examples [BATCH_SIZE]

#include <algorithm>
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

// Ring sizes up to 2^10 run in a single kernel.
constexpr int kSmallLogN = 10;
using SmallForward =
    decltype(Size<(1 << kSmallLogN)>() + Direction<FORWARD>() +
             Precision<TestDataType>() + ReductionPoly<X_N_plus>() + Block());
using SmallInverse =
    decltype(Size<(1 << kSmallLogN)>() + Direction<INVERSE>() +
             Precision<TestDataType>() + ReductionPoly<X_N_plus>() + Block());

// Larger ring sizes run as several stage kernels.
constexpr int kLargeLogN = 16;
using LargeForward =
    decltype(Size<(1 << kLargeLogN)>() + Direction<FORWARD>() +
             Precision<TestDataType>() + ReductionPoly<X_N_plus>() + Block());

// The same transform with 128 threads in every kernel, which takes more
// stages.
using LargeForward128 = decltype(LargeForward() + BlockDim<128>());

// a <- INTT(NTT(a) * b_hat) for every polynomial; b_hat is already in the NTT
// domain. Block x handles polynomials x * B, ..., x * B + B - 1.
__global__ void multiply_kernel(TestDataType* a, const TestDataType* b_hat,
                                const TestDataType* forward_roots,
                                const TestDataType* inverse_roots,
                                Modulus<TestDataType> modulus,
                                TestDataType n_inverse, int batch)
{
    TestDataType* shared_memory = SmallForward::shared_memory();
    TestDataType* polys = SmallForward::block_data(a);
    const TestDataType* polys_b_hat = SmallForward::block_data(b_hat);
    const int num_polys = SmallForward::num_polys(batch);

    SmallForward().load(shared_memory, polys, modulus, num_polys);
    SmallForward().execute(shared_memory, forward_roots, modulus);

    // Any element-wise work can go here; this kernel multiplies by b_hat.
    SmallForward().for_each(
        shared_memory,
        [&](TestDataType& value, size_t index)
        {
            value = OPERATOR_GPU<TestDataType>::mult(value, polys_b_hat[index],
                                                     modulus);
        },
        num_polys);

    SmallInverse().execute(shared_memory, inverse_roots, modulus);
    SmallInverse().store(shared_memory, polys, modulus, n_inverse, num_polys);
}

// Stage I of a forward NTT with stage kernels (LargeForward, or it with
// BlockDim<T>). Stage 0 reads in; every stage writes out, which the later
// stages read and write in place.
template <typename NTT, int I>
__global__ void large_stage_kernel(TestDataType* in, TestDataType* out,
                                   const TestDataType* roots,
                                   Modulus<TestDataType> modulus)
{
    using Stage = typename NTT::template stage<I>;
    TestDataType* shared_memory = Stage::shared_memory();
    Stage().load(shared_memory, in, modulus);
    Stage().execute(shared_memory, roots, modulus);
    Stage().store(shared_memory, out);
}

// N = 2^12 runs as two stage kernels per direction. The last forward stage
// and the first inverse stage keep the same coefficients in shared memory, so
// they share a kernel with the pointwise product in between.
constexpr int kStagedLogN = 12;
using StagedForward =
    decltype(Size<(1 << kStagedLogN)>() + Direction<FORWARD>() +
             Precision<TestDataType>() + ReductionPoly<X_N_plus>() + Block());
using StagedInverse =
    decltype(Size<(1 << kStagedLogN)>() + Direction<INVERSE>() +
             Precision<TestDataType>() + ReductionPoly<X_N_plus>() + Block());
static_assert(StagedForward::num_stages == 2 && StagedInverse::num_stages == 2);

// Kernel 1: first forward stage, a -> a.
__global__ void staged_forward_kernel(TestDataType* a,
                                      const TestDataType* forward_roots,
                                      Modulus<TestDataType> modulus)
{
    using Stage = StagedForward::stage<0>;
    TestDataType* shared_memory = Stage::shared_memory();
    Stage().load(shared_memory, a, modulus);
    Stage().execute(shared_memory, forward_roots, modulus);
    Stage().store(shared_memory, a);
}

// Kernel 2: last forward stage, product with b_hat, first inverse stage. After
// the forward execute, for_each gives every coefficient in shared memory with
// its index in a, which is also its index in b_hat.
__global__ void staged_multiply_kernel(TestDataType* a,
                                       const TestDataType* b_hat,
                                       const TestDataType* forward_roots,
                                       const TestDataType* inverse_roots,
                                       Modulus<TestDataType> modulus)
{
    using Last = StagedForward::stage<1>;
    using First = StagedInverse::stage<0>;
    TestDataType* shared_memory = Last::shared_memory();
    Last().load(shared_memory, a, modulus);
    Last().execute(shared_memory, forward_roots, modulus);
    Last().for_each(shared_memory,
                    [&](TestDataType& value, size_t index) {
                        value = OPERATOR_GPU<TestDataType>::mult(
                            value, b_hat[index], modulus);
                    });
    First().execute(shared_memory, inverse_roots, modulus);
    First().store(shared_memory, a);
}

// Kernel 3: last inverse stage, which multiplies by N^-1.
__global__ void staged_inverse_kernel(TestDataType* a,
                                      const TestDataType* inverse_roots,
                                      Modulus<TestDataType> modulus,
                                      TestDataType n_inverse)
{
    using Stage = StagedInverse::stage<1>;
    TestDataType* shared_memory = Stage::shared_memory();
    Stage().load(shared_memory, a, modulus);
    Stage().execute(shared_memory, inverse_roots, modulus);
    Stage().store(shared_memory, a, modulus, n_inverse);
}

// a <- a * s for every polynomial, with s ternary (-1, 0, 1) and stored as
// int8_t. load reads only the Precision or its signed type, so the kernel
// writes s to shared memory itself: a second buffer after the first one, so
// the launch adds SmallForward::shared_memory_size bytes. In the single
// kernel shared memory holds a polynomial in the order of data.
__global__ void ternary_multiply_kernel(TestDataType* a, const int8_t* s,
                                        const TestDataType* forward_roots,
                                        const TestDataType* inverse_roots,
                                        Modulus<TestDataType> modulus,
                                        TestDataType n_inverse)
{
    static_assert(SmallForward::batches_per_block == 1);
    TestDataType* a_shared = SmallForward::shared_memory();
    TestDataType* s_shared =
        a_shared + SmallForward::shared_memory_size / sizeof(TestDataType);
    TestDataType* poly = SmallForward::block_data(a);
    const int8_t* poly_s = SmallForward::block_data(s);

    for (int i = threadIdx.x + threadIdx.y * blockDim.x; i < SmallForward::size;
         i += blockDim.x * blockDim.y)
    {
        s_shared[i] =
            (poly_s[i] < 0) ? modulus.value - 1 : TestDataType(poly_s[i]);
    }
    // The threads wrote each other's entries, so execute needs a barrier.
    __syncthreads();

    SmallForward().load(a_shared, poly, modulus);
    SmallForward().execute(a_shared, forward_roots, modulus);
    SmallForward().execute(s_shared, forward_roots, modulus);
    SmallForward().for_each(a_shared,
                            [&](TestDataType& value, size_t index) {
                                value = OPERATOR_GPU<TestDataType>::mult(
                                    value, s_shared[index], modulus);
                            });
    SmallInverse().execute(a_shared, inverse_roots, modulus);
    SmallInverse().store(a_shared, poly, modulus, n_inverse);
}

// Launches the stages of NTT in order with large_stage_kernel. Stage 0 reads
// in; every stage writes out, which the later stages read and write in place.
template <typename NTT>
void run_large_stages(TestDataType* in, TestDataType* out,
                      const TestDataType* roots,
                      Modulus<TestDataType> modulus, int batch)
{
    NTT::for_each_stage(
        [&](auto I)
        {
            using Stage = typename NTT::template stage<I>;
            large_stage_kernel<NTT, I><<<Stage::grid(batch), Stage::block_dim,
                                         Stage::shared_memory_size>>>(
                I == 0 ? in : out, out, roots, modulus);
            GPUNTT_CUDA_CHECK(cudaGetLastError());
        });
}

template <typename T> T* device_root_table(const vector<T>& table)
{
    T* device_table;
    GPUNTT_CUDA_CHECK(cudaMalloc(&device_table, table.size() * sizeof(T)));
    GPUNTT_CUDA_CHECK(cudaMemcpy(device_table, table.data(),
                                 table.size() * sizeof(T),
                                 cudaMemcpyHostToDevice));
    return device_table;
}

int main(int argc, char* argv[])
{
    CudaDevice();

    const int batch = (argc > 1) ? atoi(argv[1]) : 3;
    std::mt19937 gen(0);

    // 1. Fused polynomial multiplication, N = 2^10, with a 61-bit prime of
    //    fields.cuh.
    {
        const NTTPrime<TestDataType>& prime = ntt_primes64[1];
        const Modulus<TestDataType> modulus(prime.value);
        const TestDataType n_inverse = SmallInverse::n_inv(modulus);
        const int n = SmallForward::size;
        std::uniform_int_distribution<std::uint64_t> dis(0, modulus.value - 1);

        vector<TestDataType> a(size_t(batch) * n), b(size_t(batch) * n);
        for (auto& x : a)
            x = dis(gen);
        for (auto& x : b)
            x = dis(gen);

        TestDataType* forward_roots =
            device_root_table(SmallForward::make_twiddles_host(prime));
        TestDataType* inverse_roots =
            device_root_table(SmallInverse::make_twiddles_host(prime));

        TestDataType *a_device, *b_hat_device;
        GPUNTT_CUDA_CHECK(
            cudaMalloc(&a_device, a.size() * sizeof(TestDataType)));
        GPUNTT_CUDA_CHECK(
            cudaMalloc(&b_hat_device, b.size() * sizeof(TestDataType)));
        GPUNTT_CUDA_CHECK(cudaMemcpy(a_device, a.data(),
                                     a.size() * sizeof(TestDataType),
                                     cudaMemcpyHostToDevice));
        GPUNTT_CUDA_CHECK(cudaMemcpy(b_hat_device, b.data(),
                                     b.size() * sizeof(TestDataType),
                                     cudaMemcpyHostToDevice));

        // b_hat = NTT(b) with the host API.
        ntt_configuration<TestDataType> cfg = {
            .n_power = kSmallLogN,
            .ntt_type = FORWARD,
            .ntt_layout = PerPolynomial,
            .reduction_poly = ReductionPolynomial::X_N_plus,
            .mod_inverse = n_inverse,
            .stream = 0};
        GPU_NTT_Inplace(b_hat_device, forward_roots, modulus, cfg, batch);

        multiply_kernel<<<SmallForward::grid(batch), SmallForward::block_dim,
                          SmallForward::shared_memory_size>>>(
            a_device, b_hat_device, forward_roots, inverse_roots, modulus,
            n_inverse, batch);
        GPUNTT_CUDA_CHECK(cudaGetLastError());

        vector<TestDataType> product(a.size());
        GPUNTT_CUDA_CHECK(cudaMemcpy(product.data(), a_device,
                                     product.size() * sizeof(TestDataType),
                                     cudaMemcpyDeviceToHost));

        bool correct = true;
        for (int i = 0; i < batch && correct; i++)
        {
            vector<TestDataType> a_i(a.begin() + size_t(i) * n,
                                     a.begin() + size_t(i + 1) * n);
            vector<TestDataType> b_i(b.begin() + size_t(i) * n,
                                     b.begin() + size_t(i + 1) * n);
            vector<TestDataType> expected = schoolbook_poly_multiplication(
                a_i, b_i, modulus, ReductionPolynomial::X_N_plus);
            correct = check_result(product.data() + size_t(i) * n,
                                   expected.data(), n);
            if (!correct)
            {
                cout << "(in " << i << ". Poly.)" << endl;
            }
        }
        if (correct)
        {
            cout << "All Correct for fused polynomial multiplication, N = 2^"
                 << kSmallLogN << "." << endl;
        }

        GPUNTT_CUDA_CHECK(cudaFree(a_device));
        GPUNTT_CUDA_CHECK(cudaFree(b_hat_device));
        GPUNTT_CUDA_CHECK(cudaFree(forward_roots));
        GPUNTT_CUDA_CHECK(cudaFree(inverse_roots));
    }

    // 2. Forward NTT with stage kernels, N = 2^16.
    {
        NTTParameters<TestDataType> parameters(kLargeLogN,
                                               ReductionPolynomial::X_N_plus);
        NTTCPU<TestDataType> cpu(parameters);
        const int n = parameters.n;
        std::uniform_int_distribution<std::uint64_t> dis(
            0, parameters.modulus.value - 1);

        vector<TestDataType> input(size_t(batch) * n);
        for (auto& x : input)
            x = dis(gen);

        // The root table is built on the GPU, which is much faster than on
        // the host for large N.
        TestDataType* roots;
        GPUNTT_CUDA_CHECK(cudaMalloc(&roots, LargeForward::twiddle_table_size *
                                                 sizeof(TestDataType)));
        LargeForward::make_twiddles(roots, parameters.modulus,
                                    parameters.root_of_unity);

        TestDataType *in_device, *out_device;
        GPUNTT_CUDA_CHECK(
            cudaMalloc(&in_device, input.size() * sizeof(TestDataType)));
        GPUNTT_CUDA_CHECK(
            cudaMalloc(&out_device, input.size() * sizeof(TestDataType)));
        GPUNTT_CUDA_CHECK(cudaMemcpy(in_device, input.data(),
                                     input.size() * sizeof(TestDataType),
                                     cudaMemcpyHostToDevice));

        run_large_stages<LargeForward>(in_device, out_device, roots,
                                       parameters.modulus, batch);

        vector<TestDataType> output(input.size());
        GPUNTT_CUDA_CHECK(cudaMemcpy(output.data(), out_device,
                                     output.size() * sizeof(TestDataType),
                                     cudaMemcpyDeviceToHost));

        bool correct = true;
        for (int i = 0; i < batch && correct; i++)
        {
            vector<TestDataType> poly(input.begin() + size_t(i) * n,
                                      input.begin() + size_t(i + 1) * n);
            vector<TestDataType> expected = cpu.ntt(poly);
            correct =
                check_result(output.data() + size_t(i) * n, expected.data(), n);
            if (!correct)
            {
                cout << "(in " << i << ". Poly.)" << endl;
            }
        }
        if (correct)
        {
            cout << "All Correct for staged NTT, N = 2^" << kLargeLogN << " ("
                 << LargeForward::num_stages << " kernels)." << endl;
        }

        // Without own work in the kernels, NTT::launch runs the same stages.
        LargeForward::launch(in_device, in_device, roots, parameters.modulus,
                             batch);
        vector<TestDataType> launched(input.size());
        GPUNTT_CUDA_CHECK(cudaMemcpy(launched.data(), in_device,
                                     launched.size() * sizeof(TestDataType),
                                     cudaMemcpyDeviceToHost));
        if (check_result(launched.data(), output.data(), int(output.size())))
        {
            cout << "All Correct for LargeForward::launch." << endl;
        }

        GPUNTT_CUDA_CHECK(cudaMemcpy(in_device, input.data(),
                                     input.size() * sizeof(TestDataType),
                                     cudaMemcpyHostToDevice));
        LargeForward128::launch(in_device, in_device, roots,
                                parameters.modulus, batch);
        GPUNTT_CUDA_CHECK(cudaMemcpy(launched.data(), in_device,
                                     launched.size() * sizeof(TestDataType),
                                     cudaMemcpyDeviceToHost));
        if (check_result(launched.data(), output.data(), int(output.size())))
        {
            cout << "All Correct for LargeForward128::launch ("
                 << LargeForward128::num_stages << " kernels of 128 threads)."
                 << endl;
        }

        GPUNTT_CUDA_CHECK(cudaFree(in_device));
        GPUNTT_CUDA_CHECK(cudaFree(out_device));
        GPUNTT_CUDA_CHECK(cudaFree(roots));
    }

    // 3. Polynomial multiplication with stage kernels, N = 2^12.
    {
        const NTTPrime<TestDataType>& prime = ntt_primes64[1];
        const Modulus<TestDataType> modulus(prime.value);
        const TestDataType n_inverse = StagedInverse::n_inv(modulus);
        const int n = StagedForward::size;
        std::uniform_int_distribution<std::uint64_t> dis(0, modulus.value - 1);

        vector<TestDataType> a(size_t(batch) * n), b(size_t(batch) * n);
        for (auto& x : a)
            x = dis(gen);
        for (auto& x : b)
            x = dis(gen);

        TestDataType *forward_roots, *inverse_roots;
        GPUNTT_CUDA_CHECK(
            cudaMalloc(&forward_roots, StagedForward::twiddle_table_size *
                                           sizeof(TestDataType)));
        GPUNTT_CUDA_CHECK(
            cudaMalloc(&inverse_roots, StagedInverse::twiddle_table_size *
                                           sizeof(TestDataType)));
        StagedForward::make_twiddles(forward_roots, prime);
        StagedInverse::make_twiddles(inverse_roots, prime);

        TestDataType *a_device, *b_hat_device;
        GPUNTT_CUDA_CHECK(
            cudaMalloc(&a_device, a.size() * sizeof(TestDataType)));
        GPUNTT_CUDA_CHECK(
            cudaMalloc(&b_hat_device, b.size() * sizeof(TestDataType)));
        GPUNTT_CUDA_CHECK(cudaMemcpy(a_device, a.data(),
                                     a.size() * sizeof(TestDataType),
                                     cudaMemcpyHostToDevice));
        GPUNTT_CUDA_CHECK(cudaMemcpy(b_hat_device, b.data(),
                                     b.size() * sizeof(TestDataType),
                                     cudaMemcpyHostToDevice));

        // b_hat = NTT(b) with the host API.
        ntt_configuration<TestDataType> cfg = {
            .n_power = kStagedLogN,
            .ntt_type = FORWARD,
            .ntt_layout = PerPolynomial,
            .reduction_poly = ReductionPolynomial::X_N_plus,
            .mod_inverse = n_inverse,
            .stream = 0};
        GPU_NTT_Inplace(b_hat_device, forward_roots, modulus, cfg, batch);

        using Stage0 = StagedForward::stage<0>;
        using Stage1 = StagedForward::stage<1>;
        using InverseStage1 = StagedInverse::stage<1>;
        staged_forward_kernel<<<Stage0::grid(batch), Stage0::block_dim,
                                Stage0::shared_memory_size>>>(
            a_device, forward_roots, modulus);
        GPUNTT_CUDA_CHECK(cudaGetLastError());
        staged_multiply_kernel<<<Stage1::grid(batch), Stage1::block_dim,
                                 Stage1::shared_memory_size>>>(
            a_device, b_hat_device, forward_roots, inverse_roots, modulus);
        GPUNTT_CUDA_CHECK(cudaGetLastError());
        staged_inverse_kernel<<<InverseStage1::grid(batch),
                                InverseStage1::block_dim,
                                InverseStage1::shared_memory_size>>>(
            a_device, inverse_roots, modulus, n_inverse);
        GPUNTT_CUDA_CHECK(cudaGetLastError());

        vector<TestDataType> product(a.size());
        GPUNTT_CUDA_CHECK(cudaMemcpy(product.data(), a_device,
                                     product.size() * sizeof(TestDataType),
                                     cudaMemcpyDeviceToHost));

        bool correct = true;
        for (int i = 0; i < batch && correct; i++)
        {
            vector<TestDataType> a_i(a.begin() + size_t(i) * n,
                                     a.begin() + size_t(i + 1) * n);
            vector<TestDataType> b_i(b.begin() + size_t(i) * n,
                                     b.begin() + size_t(i + 1) * n);
            vector<TestDataType> expected = schoolbook_poly_multiplication(
                a_i, b_i, modulus, ReductionPolynomial::X_N_plus);
            correct = check_result(product.data() + size_t(i) * n,
                                   expected.data(), n);
            if (!correct)
            {
                cout << "(in " << i << ". Poly.)" << endl;
            }
        }
        if (correct)
        {
            cout << "All Correct for staged polynomial multiplication, N = 2^"
                 << kStagedLogN << " (3 kernels)." << endl;
        }

        GPUNTT_CUDA_CHECK(cudaFree(a_device));
        GPUNTT_CUDA_CHECK(cudaFree(b_hat_device));
        GPUNTT_CUDA_CHECK(cudaFree(forward_roots));
        GPUNTT_CUDA_CHECK(cudaFree(inverse_roots));
    }

    // 4. Multiplication by a ternary polynomial stored as int8_t, N = 2^10.
    {
        const NTTPrime<TestDataType>& prime = ntt_primes64[1];
        const Modulus<TestDataType> modulus(prime.value);
        const TestDataType n_inverse = SmallInverse::n_inv(modulus);
        const int n = SmallForward::size;
        std::uniform_int_distribution<std::uint64_t> dis(0, modulus.value - 1);
        std::uniform_int_distribution<int> ternary(-1, 1);

        vector<TestDataType> a(size_t(batch) * n);
        vector<int8_t> s(size_t(batch) * n);
        for (auto& x : a)
            x = dis(gen);
        for (auto& x : s)
            x = int8_t(ternary(gen));

        TestDataType* forward_roots =
            device_root_table(SmallForward::make_twiddles_host(prime));
        TestDataType* inverse_roots =
            device_root_table(SmallInverse::make_twiddles_host(prime));

        TestDataType* a_device;
        int8_t* s_device;
        GPUNTT_CUDA_CHECK(
            cudaMalloc(&a_device, a.size() * sizeof(TestDataType)));
        GPUNTT_CUDA_CHECK(cudaMalloc(&s_device, s.size() * sizeof(int8_t)));
        GPUNTT_CUDA_CHECK(cudaMemcpy(a_device, a.data(),
                                     a.size() * sizeof(TestDataType),
                                     cudaMemcpyHostToDevice));
        GPUNTT_CUDA_CHECK(cudaMemcpy(s_device, s.data(),
                                     s.size() * sizeof(int8_t),
                                     cudaMemcpyHostToDevice));

        ternary_multiply_kernel<<<SmallForward::grid(batch),
                                  SmallForward::block_dim,
                                  2 * SmallForward::shared_memory_size>>>(
            a_device, s_device, forward_roots, inverse_roots, modulus,
            n_inverse);
        GPUNTT_CUDA_CHECK(cudaGetLastError());

        vector<TestDataType> product(a.size());
        GPUNTT_CUDA_CHECK(cudaMemcpy(product.data(), a_device,
                                     product.size() * sizeof(TestDataType),
                                     cudaMemcpyDeviceToHost));

        bool correct = true;
        for (int i = 0; i < batch && correct; i++)
        {
            vector<TestDataType> a_i(a.begin() + size_t(i) * n,
                                     a.begin() + size_t(i + 1) * n);
            vector<TestDataType> s_i(n);
            for (int j = 0; j < n; j++)
            {
                const int8_t c = s[size_t(i) * n + j];
                s_i[j] = (c < 0) ? modulus.value - 1 : TestDataType(c);
            }
            vector<TestDataType> expected = schoolbook_poly_multiplication(
                a_i, s_i, modulus, ReductionPolynomial::X_N_plus);
            correct = check_result(product.data() + size_t(i) * n,
                                   expected.data(), n);
            if (!correct)
            {
                cout << "(in " << i << ". Poly.)" << endl;
            }
        }
        if (correct)
        {
            cout << "All Correct for multiplication by a ternary int8_t "
                    "polynomial, N = 2^"
                 << kSmallLogN << "." << endl;
        }

        GPUNTT_CUDA_CHECK(cudaFree(a_device));
        GPUNTT_CUDA_CHECK(cudaFree(s_device));
        GPUNTT_CUDA_CHECK(cudaFree(forward_roots));
        GPUNTT_CUDA_CHECK(cudaFree(inverse_roots));
    }

    // 5. The forward NTT of 2 with T threads in every kernel. Only the
    //    description changes; a smaller T takes more stages.
    {
        NTTParameters<TestDataType> parameters(kLargeLogN,
                                               ReductionPolynomial::X_N_plus);
        NTTCPU<TestDataType> cpu(parameters);
        const int n = parameters.n;
        std::uniform_int_distribution<std::uint64_t> dis(
            0, parameters.modulus.value - 1);

        vector<TestDataType> input(size_t(batch) * n);
        for (auto& x : input)
            x = dis(gen);
        vector<TestDataType> expected(input.size());
        for (int i = 0; i < batch; i++)
        {
            vector<TestDataType> poly(input.begin() + size_t(i) * n,
                                      input.begin() + size_t(i + 1) * n);
            vector<TestDataType> poly_ntt = cpu.ntt(poly);
            std::copy(poly_ntt.begin(), poly_ntt.end(),
                      expected.begin() + size_t(i) * n);
        }

        // The root table does not depend on BlockDim.
        TestDataType* roots;
        GPUNTT_CUDA_CHECK(cudaMalloc(&roots, LargeForward::twiddle_table_size *
                                                 sizeof(TestDataType)));
        LargeForward::make_twiddles(roots, parameters.modulus,
                                    parameters.root_of_unity);

        TestDataType *in_device, *out_device;
        GPUNTT_CUDA_CHECK(
            cudaMalloc(&in_device, input.size() * sizeof(TestDataType)));
        GPUNTT_CUDA_CHECK(
            cudaMalloc(&out_device, input.size() * sizeof(TestDataType)));
        GPUNTT_CUDA_CHECK(cudaMemcpy(in_device, input.data(),
                                     input.size() * sizeof(TestDataType),
                                     cudaMemcpyHostToDevice));

        auto run_with_threads = [&](auto threads)
        {
            constexpr int T = decltype(threads)::value;
            using NTT = decltype(LargeForward() + BlockDim<T>());

            GPUNTT_CUDA_CHECK(cudaMemset(out_device, 0,
                                         input.size() * sizeof(TestDataType)));
            run_large_stages<NTT>(in_device, out_device, roots,
                                  parameters.modulus, batch);
            vector<TestDataType> output(input.size());
            GPUNTT_CUDA_CHECK(cudaMemcpy(output.data(), out_device,
                                         output.size() * sizeof(TestDataType),
                                         cudaMemcpyDeviceToHost));

            if (check_result(output.data(), expected.data(),
                             int(output.size())))
            {
                cout << "All Correct for staged NTT, N = 2^" << kLargeLogN
                     << ", BlockDim<" << T << ">: "
                     << NTT::num_stages << " kernels, blocks";
                NTT::for_each_stage(
                    [&](auto I)
                    {
                        const dim3 block = NTT::template stage<I>::block_dim;
                        cout << " (" << block.x << ", " << block.y << ")";
                    });
                cout << "." << endl;
            }
        };
        run_with_threads(std::integral_constant<int, 128>());
        run_with_threads(std::integral_constant<int, 256>());
        run_with_threads(std::integral_constant<int, 512>());
        run_with_threads(std::integral_constant<int, 1024>());

        GPUNTT_CUDA_CHECK(cudaFree(in_device));
        GPUNTT_CUDA_CHECK(cudaFree(out_device));
        GPUNTT_CUDA_CHECK(cudaFree(roots));
    }

    return EXIT_SUCCESS;
}
