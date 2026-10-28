// Copyright 2023-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <cstdlib>
#include <random>

#include "gpuntt/ntt_merge/ntt.cuh"
#include "gpuntt/ntt_merge/ntt_cpu.cuh"
#include "example_util.cuh"

using namespace std;
using namespace gpuntt;
using namespace gpuntt_example;

template <typename T>
void cpu_transpose(const std::vector<T>& in, std::vector<T>& out, int width,
                   int height)
{
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            out[x * height + y] = in[y * width + x];
}

// PerPolynomial INTT, checked against the CPU INTT.
template <typename T> bool per_polynomial_intt(int logn, int batch)
{
    NTTParameters<T> parameters(logn, ReductionPolynomial::X_N_minus);

    // NTT generator with certain modulus and root of unity
    NTTCPU<T> generator(parameters);

    std::random_device rd;
    // std::mt19937 gen(rd());
    std::mt19937 gen(0);
    std::uint64_t minNumber = 0;
    std::uint64_t maxNumber = parameters.modulus.value - 1;
    std::uniform_int_distribution<std::uint64_t> dis(minNumber, maxNumber);

    // Random data generation for polynomials
    vector<vector<T>> input1(batch);
    for (int j = 0; j < batch; j++)
    {
        for (int i = 0; i < parameters.n; i++)
        {
            input1[j].push_back(dis(gen));
        }
    }

    // Performing CPU NTT
    vector<vector<T>> ntt_result(batch);
    for (int i = 0; i < batch; i++)
    {
        ntt_result[i] = generator.intt(input1[i]);
    }
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////

    T* InOut_Datas;

    GPUNTT_CUDA_CHECK(
        cudaMalloc(&InOut_Datas, batch * parameters.n * sizeof(T)));

    for (int j = 0; j < batch; j++)
    {
        GPUNTT_CUDA_CHECK(cudaMemcpy(InOut_Datas + (parameters.n * j),
                                     input1[j].data(), parameters.n * sizeof(T),
                                     cudaMemcpyHostToDevice));
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////

    T* Inverse_Omega_Table_Device;

    GPUNTT_CUDA_CHECK(cudaMalloc(&Inverse_Omega_Table_Device,
                                 parameters.root_of_unity_size * sizeof(T)));

    vector<T> inverse_omega_table =
        parameters.gpu_root_of_unity_table_generator(
            parameters.inverse_root_of_unity_table);
    GPUNTT_CUDA_CHECK(cudaMemcpy(
        Inverse_Omega_Table_Device, inverse_omega_table.data(),
        parameters.root_of_unity_size * sizeof(T), cudaMemcpyHostToDevice));

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////

    Modulus<T>* test_modulus;
    GPUNTT_CUDA_CHECK(cudaMalloc(&test_modulus, sizeof(Modulus<T>)));

    Modulus<T> test_modulus_[1] = {parameters.modulus};

    GPUNTT_CUDA_CHECK(cudaMemcpy(test_modulus, test_modulus_,
                                 sizeof(Modulus<T>), cudaMemcpyHostToDevice));

    T* test_ninverse;
    GPUNTT_CUDA_CHECK(cudaMalloc(&test_ninverse, sizeof(T)));

    T test_ninverse_[1] = {parameters.n_inv};

    GPUNTT_CUDA_CHECK(cudaMemcpy(test_ninverse, test_ninverse_, sizeof(T),
                                 cudaMemcpyHostToDevice));

    ntt_rns_configuration<T> cfg_intt = {.n_power = logn,
                                         .ntt_type = INVERSE,
                                         .ntt_layout = PerPolynomial,
                                         .reduction_poly =
                                             ReductionPolynomial::X_N_minus,
                                         .mod_inverse = test_ninverse,
                                         .stream = 0};

    ////////////////
    T* Out_Datas;

    GPUNTT_CUDA_CHECK(cudaMalloc(&Out_Datas, batch * parameters.n * sizeof(T)));

    for (int j = 0; j < batch; j++)
    {
        GPUNTT_CUDA_CHECK(cudaMemcpy(Out_Datas + (parameters.n * j),
                                     input1[j].data(), parameters.n * sizeof(T),
                                     cudaMemcpyHostToDevice));
    }
    GPU_INTT(InOut_Datas, Out_Datas, Inverse_Omega_Table_Device, test_modulus,
             cfg_intt, batch, 1);

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////

    T* Output_Host;

    Output_Host = (T*) malloc(batch * parameters.n * sizeof(T));
    GPUNTT_CUDA_CHECK(cudaMemcpy(Output_Host, Out_Datas,
                                 batch * parameters.n * sizeof(T),
                                 cudaMemcpyDeviceToHost));

    // Comparing GPU NTT results and CPU NTT results
    bool check = true;
    for (int i = 0; i < batch; i++)
    {
        check = check_result(Output_Host + (i * parameters.n),
                             ntt_result[i].data(), parameters.n);

        if (!check)
        {
            cout << "(in " << i << ". Poly.)" << endl;
            break;
        }

        if ((i == (batch - 1)) && check)
        {
            cout << "All Correct PerPolynomial INTT." << endl;
        }
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////

    GPUNTT_CUDA_CHECK(cudaFree(InOut_Datas));
    GPUNTT_CUDA_CHECK(cudaFree(Inverse_Omega_Table_Device));
    GPUNTT_CUDA_CHECK(cudaFree(test_modulus));
    free(Output_Host);

    return check;
}

// Signed INTT Test Case, it takes unsigned input and returns signed output.
template <typename T, typename TSigned> bool signed_intt(int logn, int batch)
{
    NTTParameters<T> parameters(logn, ReductionPolynomial::X_N_minus);

    // NTT generator with certain modulus and root of unity
    NTTCPU<T> generator(parameters);

    std::random_device rd;
    // std::mt19937 gen(rd());
    std::mt19937 gen(0);
    std::uint64_t minNumber = 0;
    std::uint64_t maxNumber = parameters.modulus.value - 1;
    std::uniform_int_distribution<std::uint64_t> dis(minNumber, maxNumber);

    // Random data generation for polynomials
    vector<vector<T>> input1(batch);
    for (int j = 0; j < batch; j++)
    {
        for (int i = 0; i < parameters.n; i++)
        {
            input1[j].push_back(dis(gen));
        }
    }

    // Performing CPU NTT
    vector<vector<T>> ntt_result(batch);
    for (int i = 0; i < batch; i++)
    {
        ntt_result[i] = generator.intt(input1[i]);
    }

    std::uint64_t mod_half = parameters.modulus.value >> 1;
    vector<vector<TSigned>> ntt_result_signed(batch);
    for (int j = 0; j < batch; j++)
    {
        for (int i = 0; i < parameters.n; i++)
        {
            std::uint64_t r_modq = ntt_result[j][i];

            if (r_modq > mod_half)
            {
                ntt_result_signed[j].push_back(
                    static_cast<std::int64_t>(r_modq) -
                    static_cast<std::int64_t>(parameters.modulus.value));
            }
            else
            {
                ntt_result_signed[j].push_back(
                    static_cast<std::int64_t>(r_modq));
            }
        }
    }
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////

    T* In_Datas;
    TSigned* Out_Datas;

    GPUNTT_CUDA_CHECK(cudaMalloc(&In_Datas, batch * parameters.n * sizeof(T)));

    GPUNTT_CUDA_CHECK(
        cudaMalloc(&Out_Datas, batch * parameters.n * sizeof(TSigned)));

    for (int j = 0; j < batch; j++)
    {
        GPUNTT_CUDA_CHECK(cudaMemcpy(In_Datas + (parameters.n * j),
                                     input1[j].data(), parameters.n * sizeof(T),
                                     cudaMemcpyHostToDevice));
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////

    T* Inverse_Omega_Table_Device;

    GPUNTT_CUDA_CHECK(cudaMalloc(&Inverse_Omega_Table_Device,
                                 parameters.root_of_unity_size * sizeof(T)));

    vector<T> inverse_omega_table =
        parameters.gpu_root_of_unity_table_generator(
            parameters.inverse_root_of_unity_table);
    GPUNTT_CUDA_CHECK(cudaMemcpy(
        Inverse_Omega_Table_Device, inverse_omega_table.data(),
        parameters.root_of_unity_size * sizeof(T), cudaMemcpyHostToDevice));

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////

    Modulus<T>* test_modulus;
    GPUNTT_CUDA_CHECK(cudaMalloc(&test_modulus, sizeof(Modulus<T>)));

    Modulus<T> test_modulus_[1] = {parameters.modulus};

    GPUNTT_CUDA_CHECK(cudaMemcpy(test_modulus, test_modulus_,
                                 sizeof(Modulus<T>), cudaMemcpyHostToDevice));

    T* test_ninverse;
    GPUNTT_CUDA_CHECK(cudaMalloc(&test_ninverse, sizeof(T)));

    T test_ninverse_[1] = {parameters.n_inv};

    GPUNTT_CUDA_CHECK(cudaMemcpy(test_ninverse, test_ninverse_, sizeof(T),
                                 cudaMemcpyHostToDevice));

    ntt_rns_configuration<T> cfg_intt = {.n_power = logn,
                                         .ntt_type = INVERSE,
                                         .ntt_layout = PerPolynomial,
                                         .reduction_poly =
                                             ReductionPolynomial::X_N_minus,
                                         .mod_inverse = test_ninverse,
                                         .stream = 0};

    GPU_INTT(In_Datas, Out_Datas, Inverse_Omega_Table_Device, test_modulus,
             cfg_intt, batch, 1);

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////

    TSigned* Output_Host;

    Output_Host = (TSigned*) malloc(batch * parameters.n * sizeof(TSigned));
    GPUNTT_CUDA_CHECK(cudaMemcpy(Output_Host, Out_Datas,
                                 batch * parameters.n * sizeof(TSigned),
                                 cudaMemcpyDeviceToHost));

    // Comparing GPU NTT results and CPU NTT results
    bool check = true;
    for (int i = 0; i < batch; i++)
    {
        check = check_result(Output_Host + (i * parameters.n),
                             ntt_result_signed[i].data(), parameters.n);

        if (!check)
        {
            cout << "(in " << i << ". Poly.)" << endl;
            break;
        }

        if ((i == (batch - 1)) && check)
        {
            cout << "All Correct PerPolynomial INTT." << endl;
        }
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////

    GPUNTT_CUDA_CHECK(cudaFree(In_Datas));
    GPUNTT_CUDA_CHECK(cudaFree(Out_Datas));
    GPUNTT_CUDA_CHECK(cudaFree(Inverse_Omega_Table_Device));
    GPUNTT_CUDA_CHECK(cudaFree(test_modulus));
    free(Output_Host);

    return check;
}

// PerCoefficient INTT of a W x H matrix, checked against the PerPolynomial
// INTT of its transpose.
template <typename T> bool per_coefficient_intt()
{
    // W & H should be power of 2!
    int W = 1024, H = 128;
    int log_W = log2(W), log_H = log2(H);
    int total_size = W * H;

    NTTParameters<T> parameters(log_H, ReductionPolynomial::X_N_plus);

    // NTT generator with certain modulus and root of unity
    NTTCPU<T> generator(parameters);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uint64_t minNumber = 0;
    std::uint64_t maxNumber = parameters.modulus.value - 1;
    std::uniform_int_distribution<std::uint64_t> dis(minNumber, maxNumber);

    // Random data generation for polynomials
    vector<T> input_normal;
    for (int j = 0; j < H; j++)
    {
        for (int i = 0; i < W; i++)
        {
            input_normal.push_back(dis(gen));
        }
    }

    vector<T> input_transpose(total_size);
    cpu_transpose(input_normal, input_transpose, W, H);

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////

    T* InOut_Normal_Datas;

    GPUNTT_CUDA_CHECK(cudaMalloc(&InOut_Normal_Datas, total_size * sizeof(T)));

    GPUNTT_CUDA_CHECK(cudaMemcpy(InOut_Normal_Datas, input_normal.data(),
                                 total_size * sizeof(T),
                                 cudaMemcpyHostToDevice));

    T* InOut_Transpose_Datas;

    GPUNTT_CUDA_CHECK(
        cudaMalloc(&InOut_Transpose_Datas, total_size * sizeof(T)));

    GPUNTT_CUDA_CHECK(cudaMemcpy(InOut_Transpose_Datas, input_transpose.data(),
                                 total_size * sizeof(T),
                                 cudaMemcpyHostToDevice));

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////

    T* Inverse_Omega_Table_Device;

    GPUNTT_CUDA_CHECK(cudaMalloc(&Inverse_Omega_Table_Device,
                                 parameters.root_of_unity_size * sizeof(T)));

    vector<T> inverse_omega_table =
        parameters.gpu_root_of_unity_table_generator(
            parameters.inverse_root_of_unity_table);
    GPUNTT_CUDA_CHECK(cudaMemcpy(
        Inverse_Omega_Table_Device, inverse_omega_table.data(),
        parameters.root_of_unity_size * sizeof(T), cudaMemcpyHostToDevice));

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////

    ntt_configuration<T> cfg_intt = {.n_power = log_H,
                                     .ntt_type = INVERSE,
                                     .ntt_layout = PerPolynomial,
                                     .reduction_poly =
                                         ReductionPolynomial::X_N_plus,
                                     .mod_inverse = parameters.n_inv,
                                     .stream = 0};

    GPU_INTT_Inplace(InOut_Transpose_Datas, Inverse_Omega_Table_Device,
                     parameters.modulus, cfg_intt, W);

    vector<T> out_ntt(total_size);
    GPUNTT_CUDA_CHECK(cudaMemcpy(out_ntt.data(), InOut_Transpose_Datas,
                                 total_size * sizeof(T),
                                 cudaMemcpyDeviceToHost));

    vector<T> out_transpose(total_size);
    cpu_transpose(out_ntt, out_transpose, H, W);

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////

    ntt_configuration<T> cfg_intt_c = {.n_power = log_H,
                                       .ntt_type = INVERSE,
                                       .ntt_layout = PerCoefficient,
                                       .reduction_poly =
                                           ReductionPolynomial::X_N_plus,
                                       .mod_inverse = parameters.n_inv,
                                       .stream = 0};

    GPU_INTT_Inplace(InOut_Normal_Datas, Inverse_Omega_Table_Device,
                     parameters.modulus, cfg_intt_c, W);

    vector<T> out_ntt_transpose(total_size);
    GPUNTT_CUDA_CHECK(cudaMemcpy(out_ntt_transpose.data(), InOut_Normal_Datas,
                                 total_size * sizeof(T),
                                 cudaMemcpyDeviceToHost));

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // Comparing GPU NTT results and CPU NTT results
    bool check = true;
    for (int i = 0; i < W; i++)
    {
        check = check_result(out_ntt_transpose.data() + (i * H),
                             out_transpose.data() + (i * H), H);

        if (!check)
        {
            cout << "(in " << i << ". Poly.)" << endl;
            break;
        }

        if ((i == (W - 1)) && check)
        {
            cout << "All Correct PerCoefficient INTT." << endl;
        }
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////

    GPUNTT_CUDA_CHECK(cudaFree(InOut_Normal_Datas));
    GPUNTT_CUDA_CHECK(cudaFree(InOut_Transpose_Datas));
    GPUNTT_CUDA_CHECK(cudaFree(Inverse_Omega_Table_Device));

    return check;
}

template <typename T, typename TSigned> bool run(int logn, int batch)
{
    bool correct = true;
    correct &= per_polynomial_intt<T>(logn, batch);
    correct &= signed_intt<T, TSigned>(logn, batch);
    correct &= per_coefficient_intt<T>();
    return correct;
}

int main(int argc, char* argv[])
{
    CudaDevice();

    int device = 0; // Assuming you are using device 0
    cudaSetDevice(device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    std::cout << "Maximum Grid Size: " << prop.maxGridSize[0] << " x "
              << prop.maxGridSize[1] << " x " << prop.maxGridSize[2]
              << std::endl;

    const ExampleArgs args = parse_example_args(argc, argv);
    const bool correct = (args.bits == 32)
                             ? run<Data32, Data32s>(args.logn, args.batch)
                             : run<Data64, Data64s>(args.logn, args.batch);

    return correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
