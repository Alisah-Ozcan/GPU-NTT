// Copyright 2024 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <cstdlib>
#include <random>

#include "ntt.cuh"

#define DEFAULT_MODULUS

using namespace std;
using namespace gpuntt;

int LOGN;
int BATCH;

// typedef Data32 TestDataType; // Use for 32-bit Test
typedef Data64 TestDataType; // Use for 64-bit Test

template <typename T>
void cpu_transpose(const std::vector<T>& in, std::vector<T>& out, int width,
                   int height)
{
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            out[x * height + y] = in[y * width + x];
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
    {
        if (argc < 3)
        {
            LOGN = 5;
            BATCH = 1;
        }
        else
        {
            LOGN = atoi(argv[1]);
            BATCH = atoi(argv[2]);
        }

#ifdef DEFAULT_MODULUS
        NTTParameters<TestDataType> parameters(LOGN,
                                               ReductionPolynomial::X_N_minus);
#else
        NTTFactors factor((Modulus) 576460752303415297, 288482366111684746,
                          238394956950829);
        NTTParameters parameters(LOGN, factor, ReductionPolynomial::X_N_minus);
#endif

        // NTT generator with certain modulus and root of unity
        NTTCPU<TestDataType> generator(parameters);

        std::random_device rd;
        // std::mt19937 gen(rd());
        std::mt19937 gen(0);
        unsigned long long minNumber = 0;
        unsigned long long maxNumber = parameters.modulus.value - 1;
        std::uniform_int_distribution<unsigned long long> dis(minNumber,
                                                              maxNumber);

        // Random data generation for polynomials
        vector<vector<TestDataType>> input1(BATCH);
        for (int j = 0; j < BATCH; j++)
        {
            for (int i = 0; i < parameters.n; i++)
            {
                input1[j].push_back(dis(gen));
            }
        }

        // Performing CPU NTT
        vector<vector<TestDataType>> ntt_result(BATCH);
        for (int i = 0; i < BATCH; i++)
        {
            ntt_result[i] = generator.ntt(input1[i]);
        }
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////

        TestDataType* InOut_Datas;

        GPUNTT_CUDA_CHECK(cudaMalloc(&InOut_Datas, BATCH * parameters.n *
                                                       sizeof(TestDataType)));

        for (int j = 0; j < BATCH; j++)
        {
            GPUNTT_CUDA_CHECK(cudaMemcpy(
                InOut_Datas + (parameters.n * j), input1[j].data(),
                parameters.n * sizeof(TestDataType), cudaMemcpyHostToDevice));
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////

        Root<TestDataType>* Forward_Omega_Table_Device;

        GPUNTT_CUDA_CHECK(cudaMalloc(&Forward_Omega_Table_Device,
                                     parameters.root_of_unity_size *
                                         sizeof(Root<TestDataType>)));

        vector<Root<TestDataType>> forward_omega_table =
            parameters.gpu_root_of_unity_table_generator(
                parameters.forward_root_of_unity_table);

        GPUNTT_CUDA_CHECK(cudaMemcpy(
            Forward_Omega_Table_Device, forward_omega_table.data(),
            parameters.root_of_unity_size * sizeof(Root<TestDataType>),
            cudaMemcpyHostToDevice));

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////

        Modulus<TestDataType>* test_modulus;
        GPUNTT_CUDA_CHECK(
            cudaMalloc(&test_modulus, sizeof(Modulus<TestDataType>)));

        Modulus<TestDataType> test_modulus_[1] = {parameters.modulus};

        GPUNTT_CUDA_CHECK(cudaMemcpy(test_modulus, test_modulus_,
                                     sizeof(Modulus<TestDataType>),
                                     cudaMemcpyHostToDevice));

        ntt_rns_configuration<TestDataType> cfg_ntt = {
            .n_power = LOGN,
            .ntt_type = FORWARD,
            .ntt_layout = PerPolynomial,
            .reduction_poly = ReductionPolynomial::X_N_minus,
            .zero_padding = false,
            .stream = 0};
        GPU_NTT_Inplace(InOut_Datas, Forward_Omega_Table_Device, test_modulus,
                        cfg_ntt, BATCH, 1);

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////

        TestDataType* Output_Host;

        Output_Host =
            (TestDataType*) malloc(BATCH * parameters.n * sizeof(TestDataType));

        GPUNTT_CUDA_CHECK(
            cudaMemcpy(Output_Host, InOut_Datas,
                       BATCH * parameters.n * sizeof(TestDataType),
                       cudaMemcpyDeviceToHost));

        // Comparing GPU NTT results and CPU NTT results
        bool check = true;
        for (int i = 0; i < BATCH; i++)
        {
            check = check_result(Output_Host + (i * parameters.n),
                                 ntt_result[i].data(), parameters.n);

            if (!check)
            {
                cout << "(in " << i << ". Poly.)" << endl;
                break;
            }

            if ((i == (BATCH - 1)) && check)
            {
                cout << "All Correct for PerPolynomial NTT." << endl;
            }
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////

        GPUNTT_CUDA_CHECK(cudaFree(InOut_Datas));
        GPUNTT_CUDA_CHECK(cudaFree(Forward_Omega_Table_Device));
        free(Output_Host);
    }

    {
        // W & H should be power of 2!
        int W = 1024, H = 512;
        int log_W = log2(W), log_H = log2(H);
        int total_size = W * H;

        NTTParameters<TestDataType> parameters(log_H,
                                               ReductionPolynomial::X_N_plus);

        // NTT generator with certain modulus and root of unity
        NTTCPU<TestDataType> generator(parameters);

        std::random_device rd;
        std::mt19937 gen(rd());
        unsigned long long minNumber = 0;
        unsigned long long maxNumber = parameters.modulus.value - 1;
        std::uniform_int_distribution<unsigned long long> dis(minNumber,
                                                              maxNumber);

        // Random data generation for polynomials
        vector<TestDataType> input_normal;
        for (int j = 0; j < H; j++)
        {
            for (int i = 0; i < W; i++)
            {
                input_normal.push_back(dis(gen));
            }
        }

        vector<TestDataType> input_transpose(total_size);
        cpu_transpose(input_normal, input_transpose, W, H);

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////

        TestDataType* InOut_Normal_Datas;

        GPUNTT_CUDA_CHECK(
            cudaMalloc(&InOut_Normal_Datas, total_size * sizeof(TestDataType)));

        GPUNTT_CUDA_CHECK(cudaMemcpy(InOut_Normal_Datas, input_normal.data(),
                                     total_size * sizeof(TestDataType),
                                     cudaMemcpyHostToDevice));

        TestDataType* InOut_Transpose_Datas;

        GPUNTT_CUDA_CHECK(cudaMalloc(&InOut_Transpose_Datas,
                                     total_size * sizeof(TestDataType)));

        GPUNTT_CUDA_CHECK(cudaMemcpy(
            InOut_Transpose_Datas, input_transpose.data(),
            total_size * sizeof(TestDataType), cudaMemcpyHostToDevice));

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////

        Root<TestDataType>* Forward_Omega_Table_Device;

        GPUNTT_CUDA_CHECK(cudaMalloc(&Forward_Omega_Table_Device,
                                     parameters.root_of_unity_size *
                                         sizeof(Root<TestDataType>)));

        vector<Root<TestDataType>> forward_omega_table =
            parameters.gpu_root_of_unity_table_generator(
                parameters.forward_root_of_unity_table);

        GPUNTT_CUDA_CHECK(cudaMemcpy(
            Forward_Omega_Table_Device, forward_omega_table.data(),
            parameters.root_of_unity_size * sizeof(Root<TestDataType>),
            cudaMemcpyHostToDevice));

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////

        ntt_configuration<TestDataType> cfg_ntt = {
            .n_power = log_H,
            .ntt_type = FORWARD,
            .ntt_layout = PerPolynomial,
            .reduction_poly = ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = 0};
        GPU_NTT_Inplace(InOut_Transpose_Datas, Forward_Omega_Table_Device,
                        parameters.modulus, cfg_ntt, W);

        vector<TestDataType> out_ntt(total_size);
        GPUNTT_CUDA_CHECK(cudaMemcpy(out_ntt.data(), InOut_Transpose_Datas,
                                     total_size * sizeof(TestDataType),
                                     cudaMemcpyDeviceToHost));

        vector<TestDataType> out_transpose(total_size);
        cpu_transpose(out_ntt, out_transpose, H, W);

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////

        ntt_configuration<TestDataType> cfg_ntt_c = {
            .n_power = log_H,
            .ntt_type = FORWARD,
            .ntt_layout = PerCoefficient,
            .reduction_poly = ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = 0};
        GPU_NTT_Inplace(InOut_Normal_Datas, Forward_Omega_Table_Device,
                        parameters.modulus, cfg_ntt_c, W);

        vector<TestDataType> out_ntt_transpose(total_size);
        GPUNTT_CUDA_CHECK(cudaMemcpy(
            out_ntt_transpose.data(), InOut_Normal_Datas,
            total_size * sizeof(TestDataType), cudaMemcpyDeviceToHost));

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
                cout << "All Correct for PerCoefficient NTT." << endl;
            }
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////

        GPUNTT_CUDA_CHECK(cudaFree(InOut_Normal_Datas));
        GPUNTT_CUDA_CHECK(cudaFree(InOut_Transpose_Datas));
        GPUNTT_CUDA_CHECK(cudaFree(Forward_Omega_Table_Device));
    }

    return EXIT_SUCCESS;
}