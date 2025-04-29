// Copyright 2024 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <cstdlib>
#include <random>

#include "ntt.cuh"
#include "ntt_4step.cuh"
#include "ntt_4step_cpu.cuh"

#define DEFAULT_MODULUS

using namespace std;
using namespace gpuntt;

int LOGN;
int BATCH;
int N;

// typedef Data32 TestDataType; // Use for 32-bit Test
typedef Data64 TestDataType; // Use for 64-bit Test

int main(int argc, char* argv[])
{
    CudaDevice();

    if (argc < 3)
    {
        LOGN = 12;
        BATCH = 1;
    }
    else
    {
        LOGN = atoi(argv[1]);
        BATCH = atoi(argv[2]);
    }

    // Current 4step NTT implementation only works for
    // ReductionPolynomial::X_N_minus!
    NTTParameters4Step<TestDataType> parameters(LOGN,
                                                ReductionPolynomial::X_N_minus);

    // NTT generator with certain modulus and root of unity
    NTT_4STEP_CPU<TestDataType> generator(parameters);

    std::random_device rd;
    std::mt19937 gen(rd());
    unsigned long long minNumber = 0;
    unsigned long long maxNumber = parameters.modulus.value - 1;
    std::uniform_int_distribution<unsigned long long> dis(minNumber, maxNumber);

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

    TestDataType* Input_Datas;

    GPUNTT_CUDA_CHECK(
        cudaMalloc(&Input_Datas, BATCH * parameters.n * sizeof(TestDataType)));

    TestDataType* Output_Datas;
    GPUNTT_CUDA_CHECK(
        cudaMalloc(&Output_Datas, BATCH * parameters.n * sizeof(TestDataType)));

    for (int j = 0; j < BATCH; j++)
    {
        GPUNTT_CUDA_CHECK(cudaMemcpy(
            Input_Datas + (parameters.n * j), input1[j].data(),
            parameters.n * sizeof(TestDataType), cudaMemcpyHostToDevice));
    }

    //////////////////////////////////////////////////////////////////////////

    vector<Root<TestDataType>> psitable1 =
        parameters.gpu_root_of_unity_table_generator(
            parameters.n1_based_root_of_unity_table);
    Root<TestDataType>* psitable_device1;
    GPUNTT_CUDA_CHECK(cudaMalloc(
        &psitable_device1, (parameters.n1 >> 1) * sizeof(Root<TestDataType>)));
    GPUNTT_CUDA_CHECK(
        cudaMemcpy(psitable_device1, psitable1.data(),
                   (parameters.n1 >> 1) * sizeof(Root<TestDataType>),
                   cudaMemcpyHostToDevice));

    vector<Root<TestDataType>> psitable2 =
        parameters.gpu_root_of_unity_table_generator(
            parameters.n2_based_root_of_unity_table);
    Root<TestDataType>* psitable_device2;
    GPUNTT_CUDA_CHECK(cudaMalloc(
        &psitable_device2, (parameters.n2 >> 1) * sizeof(Root<TestDataType>)));
    GPUNTT_CUDA_CHECK(
        cudaMemcpy(psitable_device2, psitable2.data(),
                   (parameters.n2 >> 1) * sizeof(Root<TestDataType>),
                   cudaMemcpyHostToDevice));

    Root<TestDataType>* W_Table_device;
    GPUNTT_CUDA_CHECK(
        cudaMalloc(&W_Table_device, parameters.n * sizeof(Root<TestDataType>)));
    GPUNTT_CUDA_CHECK(cudaMemcpy(
        W_Table_device, parameters.W_root_of_unity_table.data(),
        parameters.n * sizeof(Root<TestDataType>), cudaMemcpyHostToDevice));

    //////////////////////////////////////////////////////////////////////////

    Modulus<TestDataType>* test_modulus;
    GPUNTT_CUDA_CHECK(cudaMalloc(&test_modulus, sizeof(Modulus<TestDataType>)));

    Modulus<TestDataType> test_modulus_[1] = {parameters.modulus};

    GPUNTT_CUDA_CHECK(cudaMemcpy(test_modulus, test_modulus_,
                                 sizeof(Modulus<TestDataType>),
                                 cudaMemcpyHostToDevice));

    Ninverse<TestDataType>* test_ninverse;
    GPUNTT_CUDA_CHECK(
        cudaMalloc(&test_ninverse, sizeof(Ninverse<TestDataType>)));

    Ninverse<TestDataType> test_ninverse_[1] = {parameters.n_inv};

    GPUNTT_CUDA_CHECK(cudaMemcpy(test_ninverse, test_ninverse_,
                                 sizeof(Ninverse<TestDataType>),
                                 cudaMemcpyHostToDevice));

    ntt4step_rns_configuration<TestDataType> cfg_intt = {.n_power = LOGN,
                                                         .ntt_type = FORWARD,
                                                         .mod_inverse =
                                                             test_ninverse,
                                                         .stream = 0};

    //////////////////////////////////////////////////////////////////////////
    GPU_Transpose(Input_Datas, Output_Datas, parameters.n1, parameters.n2,
                  parameters.logn, BATCH);

    GPU_4STEP_NTT(Output_Datas, Input_Datas, psitable_device1, psitable_device2,
                  W_Table_device, test_modulus, cfg_intt, BATCH, 1);

    GPU_Transpose(Input_Datas, Output_Datas, parameters.n1, parameters.n2,
                  parameters.logn, BATCH);

    vector<TestDataType> Output_Host(parameters.n * BATCH);
    cudaMemcpy(Output_Host.data(), Output_Datas,
               parameters.n * BATCH * sizeof(TestDataType),
               cudaMemcpyDeviceToHost);

    // Comparing GPU NTT results and CPU NTT results
    bool check = true;
    for (int i = 0; i < BATCH; i++)
    {
        check = check_result(Output_Host.data() + (i * parameters.n),
                             ntt_result[i].data(), parameters.n);

        if (!check)
        {
            cout << "(in " << i << ". Poly.)" << endl;
            break;
        }

        if ((i == (BATCH - 1)) && check)
        {
            cout << "All Correct." << endl;
        }
    }

    return EXIT_SUCCESS;
}