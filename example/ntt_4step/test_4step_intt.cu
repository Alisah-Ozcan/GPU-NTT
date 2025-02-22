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

    ModularReductionType modular_reduction_type = ModularReductionType::BARRET;

    // Current 4step NTT implementation only works for
    // ReductionPolynomial::X_N_minus!
    NTTParameters4Step<Data64> parameters(LOGN, modular_reduction_type,
                                          ReductionPolynomial::X_N_minus);

    // NTT generator with certain modulus and root of unity
    NTT_4STEP_CPU<Data64> generator(parameters);

    std::random_device rd;
    std::mt19937 gen(rd());
    unsigned long long minNumber = 0;
    unsigned long long maxNumber = parameters.modulus.value - 1;
    std::uniform_int_distribution<unsigned long long> dis(minNumber, maxNumber);

    // Random data generation for polynomials
    vector<vector<Data64>> input1(BATCH);
    for (int j = 0; j < BATCH; j++)
    {
        for (int i = 0; i < parameters.n; i++)
        {
            input1[j].push_back(dis(gen));
        }
    }

    // Performing CPU NTT
    vector<vector<Data64>> ntt_result(BATCH);
    for (int i = 0; i < BATCH; i++)
    {
        ntt_result[i] = generator.intt(input1[i]);
    }
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////

    Data64* Input_Datas;

    GPUNTT_CUDA_CHECK(
        cudaMalloc(&Input_Datas, BATCH * parameters.n * sizeof(Data64)));

    Data64* Output_Datas;
    GPUNTT_CUDA_CHECK(
        cudaMalloc(&Output_Datas, BATCH * parameters.n * sizeof(Data64)));

    for (int j = 0; j < BATCH; j++)
    {
        vector<Data64> cpu_intt_transposed_input =
            generator.intt_first_transpose(input1[j]); // INTT TRANSPOSE IN CPU

        GPUNTT_CUDA_CHECK(cudaMemcpy(
            Input_Datas + (parameters.n * j), cpu_intt_transposed_input.data(),
            parameters.n * sizeof(Data64), cudaMemcpyHostToDevice));
    }

    //////////////////////////////////////////////////////////////////////////

    vector<Root64> psitable1 = parameters.gpu_root_of_unity_table_generator(
        parameters.n1_based_inverse_root_of_unity_table);
    Root64* psitable_device1;
    GPUNTT_CUDA_CHECK(
        cudaMalloc(&psitable_device1, (parameters.n1 >> 1) * sizeof(Root64)));
    GPUNTT_CUDA_CHECK(cudaMemcpy(psitable_device1, psitable1.data(),
                                 (parameters.n1 >> 1) * sizeof(Root64),
                                 cudaMemcpyHostToDevice));

    vector<Root64> psitable2 = parameters.gpu_root_of_unity_table_generator(
        parameters.n2_based_inverse_root_of_unity_table);
    Root64* psitable_device2;
    GPUNTT_CUDA_CHECK(
        cudaMalloc(&psitable_device2, (parameters.n2 >> 1) * sizeof(Root64)));
    GPUNTT_CUDA_CHECK(cudaMemcpy(psitable_device2, psitable2.data(),
                                 (parameters.n2 >> 1) * sizeof(Root64),
                                 cudaMemcpyHostToDevice));

    Root64* W_Table_device;
    GPUNTT_CUDA_CHECK(
        cudaMalloc(&W_Table_device, parameters.n * sizeof(Root64)));
    GPUNTT_CUDA_CHECK(cudaMemcpy(
        W_Table_device, parameters.W_inverse_root_of_unity_table.data(),
        parameters.n * sizeof(Root64), cudaMemcpyHostToDevice));

    //////////////////////////////////////////////////////////////////////////

    Modulus64* test_modulus;
    GPUNTT_CUDA_CHECK(cudaMalloc(&test_modulus, sizeof(Modulus64)));

    Modulus64 test_modulus_[1] = {parameters.modulus};

    GPUNTT_CUDA_CHECK(cudaMemcpy(test_modulus, test_modulus_, sizeof(Modulus64),
                                 cudaMemcpyHostToDevice));

    Ninverse64* test_ninverse;
    GPUNTT_CUDA_CHECK(cudaMalloc(&test_ninverse, sizeof(Ninverse64)));

    Ninverse64 test_ninverse_[1] = {parameters.n_inv};

    GPUNTT_CUDA_CHECK(cudaMemcpy(test_ninverse, test_ninverse_,
                                 sizeof(Ninverse64), cudaMemcpyHostToDevice));

    ntt4step_rns_configuration<Data64> cfg_intt = {.n_power = LOGN,
                                                   .ntt_type = INVERSE,
                                                   .mod_inverse = test_ninverse,
                                                   .stream = 0};

    //////////////////////////////////////////////////////////////////////////

    GPU_4STEP_NTT(Input_Datas, Output_Datas, psitable_device1, psitable_device2,
                  W_Table_device, test_modulus, cfg_intt, BATCH, 1);

    GPU_Transpose(Output_Datas, Input_Datas, parameters.n1, parameters.n2,
                  parameters.logn, BATCH);

    vector<Data64> Output_Host(parameters.n * BATCH);
    cudaMemcpy(Output_Host.data(), Input_Datas,
               parameters.n * BATCH * sizeof(Data64), cudaMemcpyDeviceToHost);

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

/*

cmake . -B./cmake-build
cmake . -D MODULAR_REDUCTION_TYPE=0 -B./cmake-build
cmake . -D MODULAR_REDUCTION_TYPE=1 -B./cmake-build
cmake . -D MODULAR_REDUCTION_TYPE=2 -B./cmake-build

cmake --build ./cmake-build/ --parallel

./cmake-build/bin/gpu_ntt_examples

*/