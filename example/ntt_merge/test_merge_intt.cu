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

//typedef Data32 TestDataType; // Use for 32-bit Test
typedef Data64 TestDataType; // Use for 64-bit Test

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

#ifdef DEFAULT_MODULUS
    NTTParameters<TestDataType> parameters(LOGN, ReductionPolynomial::X_N_minus);
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
        ntt_result[i] = generator.intt(input1[i]);
    }
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////

    TestDataType* InOut_Datas;

    GPUNTT_CUDA_CHECK(
        cudaMalloc(&InOut_Datas, BATCH * parameters.n * sizeof(TestDataType)));

    for (int j = 0; j < BATCH; j++)
    {
        GPUNTT_CUDA_CHECK(
            cudaMemcpy(InOut_Datas + (parameters.n * j), input1[j].data(),
                       parameters.n * sizeof(TestDataType), cudaMemcpyHostToDevice));
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////

    Root<TestDataType>* Inverse_Omega_Table_Device;

    GPUNTT_CUDA_CHECK(
        cudaMalloc(&Inverse_Omega_Table_Device,
                   parameters.root_of_unity_size * sizeof(Root<TestDataType>)));

    vector<Root<TestDataType>> inverse_omega_table =
        parameters.gpu_root_of_unity_table_generator(
            parameters.inverse_root_of_unity_table);
    GPUNTT_CUDA_CHECK(cudaMemcpy(Inverse_Omega_Table_Device,
                                 inverse_omega_table.data(),
                                 parameters.root_of_unity_size * sizeof(Root<TestDataType>),
                                 cudaMemcpyHostToDevice));

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////

    Modulus<TestDataType>* test_modulus;
    GPUNTT_CUDA_CHECK(cudaMalloc(&test_modulus, sizeof(Modulus<TestDataType>)));

    Modulus<TestDataType> test_modulus_[1] = {parameters.modulus};

    GPUNTT_CUDA_CHECK(cudaMemcpy(test_modulus, test_modulus_, sizeof(Modulus<TestDataType>),
                                 cudaMemcpyHostToDevice));

    Ninverse<TestDataType>* test_ninverse;
    GPUNTT_CUDA_CHECK(cudaMalloc(&test_ninverse, sizeof(Ninverse<TestDataType>)));

    Ninverse<TestDataType> test_ninverse_[1] = {parameters.n_inv};

    GPUNTT_CUDA_CHECK(cudaMemcpy(test_ninverse, test_ninverse_,
                                 sizeof(Ninverse<TestDataType>), cudaMemcpyHostToDevice));

    ntt_rns_configuration<TestDataType> cfg_intt = {
        .n_power = LOGN,
        .ntt_type = INVERSE,
        .reduction_poly = ReductionPolynomial::X_N_minus,
        .zero_padding = false,
        .mod_inverse = test_ninverse,
        .stream = 0};

    ////////////////
    TestDataType* Out_Datas;

    GPUNTT_CUDA_CHECK(
        cudaMalloc(&Out_Datas, BATCH * parameters.n * sizeof(TestDataType)));

    for (int j = 0; j < BATCH; j++)
    {
        GPUNTT_CUDA_CHECK(
            cudaMemcpy(Out_Datas + (parameters.n * j), input1[j].data(),
                       parameters.n * sizeof(TestDataType), cudaMemcpyHostToDevice));
    }
    GPU_NTT(InOut_Datas, Out_Datas, Inverse_Omega_Table_Device, test_modulus,
            cfg_intt, BATCH, 1);

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////

    TestDataType* Output_Host;

    Output_Host = (TestDataType*) malloc(BATCH * parameters.n * sizeof(TestDataType));
    GPUNTT_CUDA_CHECK(cudaMemcpy(Output_Host, Out_Datas,
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
            cout << "All Correct." << endl;
        }
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////

    GPUNTT_CUDA_CHECK(cudaFree(InOut_Datas));
    GPUNTT_CUDA_CHECK(cudaFree(Inverse_Omega_Table_Device));
    free(Output_Host);

    return EXIT_SUCCESS;
}