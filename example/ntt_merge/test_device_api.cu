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
//    against the CPU NTT.
//
// Usage: gpu_device_api_examples [BATCH_SIZE]

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

// a <- INTT(NTT(a) * b_hat) for every polynomial; b_hat is already in the NTT
// domain. Block x handles polynomials x * B, ..., x * B + B - 1.
__global__ void multiply_kernel(TestDataType* a, const TestDataType* b_hat,
                                const Root<TestDataType>* forward_roots,
                                const Root<TestDataType>* inverse_roots,
                                Modulus<TestDataType> modulus,
                                Ninverse<TestDataType> n_inverse, int batch)
{
    extern __shared__ char shared_memory_raw[];
    TestDataType* shared_memory =
        reinterpret_cast<TestDataType*>(shared_memory_raw);

    const int first = blockIdx.x * SmallForward::batches_per_block;
    const int num_polys = min(SmallForward::batches_per_block, batch - first);
    const size_t offset = size_t(first) * SmallForward::size;

    SmallForward().load(shared_memory, a + offset, modulus, num_polys);
    SmallForward().execute(shared_memory, forward_roots, modulus);

    // Any element-wise work can go here; this kernel multiplies by b_hat.
    for (int i = threadIdx.x + threadIdx.y * blockDim.x;
         i < num_polys * SmallForward::size; i += blockDim.x * blockDim.y)
    {
        shared_memory[i] = OPERATOR_GPU<TestDataType>::mult(
            shared_memory[i], b_hat[offset + i], modulus);
    }
    __syncthreads();

    SmallInverse().execute(shared_memory, inverse_roots, modulus);
    SmallInverse().store(shared_memory, a + offset, modulus, n_inverse,
                         num_polys);
}

// Stage I of the large forward NTT. Stage 0 reads in; every stage writes out,
// which the later stages read and write in place.
template <int I>
__global__ void large_stage_kernel(TestDataType* in, TestDataType* out,
                                   const Root<TestDataType>* roots,
                                   Modulus<TestDataType> modulus)
{
    extern __shared__ char shared_memory_raw[];
    TestDataType* shared_memory =
        reinterpret_cast<TestDataType*>(shared_memory_raw);

    using Stage = LargeForward::stage<I>;
    Stage().load(shared_memory, in, modulus);
    Stage().execute(shared_memory, roots, modulus);
    Stage().store(shared_memory, out);
}

template <int I>
void launch_large_stage(TestDataType* in, TestDataType* out,
                        const Root<TestDataType>* roots,
                        Modulus<TestDataType> modulus, int batch)
{
    using Stage = LargeForward::stage<I>;
    large_stage_kernel<I>
        <<<Stage::grid(batch), Stage::block_dim, Stage::shared_memory_size>>>(
            I == 0 ? in : out, out, roots, modulus);
    GPUNTT_CUDA_CHECK(cudaGetLastError());
}

template <int... I>
void launch_large_stages(TestDataType* in, TestDataType* out,
                         const Root<TestDataType>* roots,
                         Modulus<TestDataType> modulus, int batch,
                         std::integer_sequence<int, I...>)
{
    (launch_large_stage<I>(in, out, roots, modulus, batch), ...);
}

template <typename T>
Root<T>* device_root_table(NTTParameters<T>& parameters, const vector<T>& table)
{
    vector<Root<T>> gpu_table =
        parameters.gpu_root_of_unity_table_generator(table);
    Root<T>* device_table;
    GPUNTT_CUDA_CHECK(
        cudaMalloc(&device_table, gpu_table.size() * sizeof(Root<T>)));
    GPUNTT_CUDA_CHECK(cudaMemcpy(device_table, gpu_table.data(),
                                 gpu_table.size() * sizeof(Root<T>),
                                 cudaMemcpyHostToDevice));
    return device_table;
}

int main(int argc, char* argv[])
{
    CudaDevice();

    const int batch = (argc > 1) ? atoi(argv[1]) : 3;
    std::mt19937 gen(0);

    // 1. Fused polynomial multiplication, N = 2^10.
    {
        NTTParameters<TestDataType> parameters(kSmallLogN,
                                               ReductionPolynomial::X_N_plus);
        const int n = parameters.n;
        std::uniform_int_distribution<std::uint64_t> dis(
            0, parameters.modulus.value - 1);

        vector<TestDataType> a(size_t(batch) * n), b(size_t(batch) * n);
        for (auto& x : a)
            x = dis(gen);
        for (auto& x : b)
            x = dis(gen);

        Root<TestDataType>* forward_roots = device_root_table(
            parameters, parameters.forward_root_of_unity_table);
        Root<TestDataType>* inverse_roots = device_root_table(
            parameters, parameters.inverse_root_of_unity_table);

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
            .mod_inverse = parameters.n_inv,
            .stream = 0};
        GPU_NTT_Inplace(b_hat_device, forward_roots, parameters.modulus, cfg,
                        batch);

        multiply_kernel<<<SmallForward::grid(batch), SmallForward::block_dim,
                          SmallForward::shared_memory_size>>>(
            a_device, b_hat_device, forward_roots, inverse_roots,
            parameters.modulus, parameters.n_inv, batch);
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
                a_i, b_i, parameters.modulus, ReductionPolynomial::X_N_plus);
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

        Root<TestDataType>* roots = device_root_table(
            parameters, parameters.forward_root_of_unity_table);

        TestDataType *in_device, *out_device;
        GPUNTT_CUDA_CHECK(
            cudaMalloc(&in_device, input.size() * sizeof(TestDataType)));
        GPUNTT_CUDA_CHECK(
            cudaMalloc(&out_device, input.size() * sizeof(TestDataType)));
        GPUNTT_CUDA_CHECK(cudaMemcpy(in_device, input.data(),
                                     input.size() * sizeof(TestDataType),
                                     cudaMemcpyHostToDevice));

        launch_large_stages(
            in_device, out_device, roots, parameters.modulus, batch,
            std::make_integer_sequence<int, LargeForward::num_stages>());

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

        GPUNTT_CUDA_CHECK(cudaFree(in_device));
        GPUNTT_CUDA_CHECK(cudaFree(out_device));
        GPUNTT_CUDA_CHECK(cudaFree(roots));
    }

    return EXIT_SUCCESS;
}
