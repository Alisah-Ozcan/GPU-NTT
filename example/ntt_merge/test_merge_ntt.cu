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

    ModularReductionType modular_reduction_type = ModularReductionType::BARRET;

#ifdef DEFAULT_MODULUS
    NTTParameters<Data64> parameters(LOGN, modular_reduction_type,
                                     ReductionPolynomial::X_N_minus);
#else
    NTTFactors factor((Modulus) 576460752303415297, 288482366111684746,
                      238394956950829);
    NTTParameters parameters(LOGN, factor, ReductionPolynomial::X_N_minus);
#endif

    // NTT generator with certain modulus and root of unity
    NTTCPU<Data64> generator(parameters);

    std::random_device rd;
    // std::mt19937 gen(rd());
    std::mt19937 gen(0);
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
        ntt_result[i] = generator.ntt(input1[i]);
    }
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////

    Data64* InOut_Datas;

    GPUNTT_CUDA_CHECK(
        cudaMalloc(&InOut_Datas, BATCH * parameters.n * sizeof(Data64)));

    for (int j = 0; j < BATCH; j++)
    {
        GPUNTT_CUDA_CHECK(
            cudaMemcpy(InOut_Datas + (parameters.n * j), input1[j].data(),
                       parameters.n * sizeof(Data64), cudaMemcpyHostToDevice));
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////

    Root64* Forward_Omega_Table_Device;

    GPUNTT_CUDA_CHECK(
        cudaMalloc(&Forward_Omega_Table_Device,
                   parameters.root_of_unity_size * sizeof(Root64)));

    vector<Root64> forward_omega_table =
        parameters.gpu_root_of_unity_table_generator(
            parameters.forward_root_of_unity_table);

    GPUNTT_CUDA_CHECK(cudaMemcpy(Forward_Omega_Table_Device,
                                 forward_omega_table.data(),
                                 parameters.root_of_unity_size * sizeof(Root64),
                                 cudaMemcpyHostToDevice));

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////

    Modulus64* test_modulus;
    GPUNTT_CUDA_CHECK(cudaMalloc(&test_modulus, sizeof(Modulus64)));

    Modulus64 test_modulus_[1] = {parameters.modulus};

    GPUNTT_CUDA_CHECK(cudaMemcpy(test_modulus, test_modulus_, sizeof(Modulus64),
                                 cudaMemcpyHostToDevice));

    ntt_rns_configuration<Data64> cfg_ntt = {.n_power = LOGN,
                                             .ntt_type = FORWARD,
                                             .reduction_poly =
                                                 ReductionPolynomial::X_N_minus,
                                             .zero_padding = false,
                                             .stream = 0};
    GPU_NTT_Inplace(InOut_Datas, Forward_Omega_Table_Device, test_modulus,
                    cfg_ntt, BATCH, 1);

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////

    Data64* Output_Host;

    Output_Host = (Data64*) malloc(BATCH * parameters.n * sizeof(Data64));

    GPUNTT_CUDA_CHECK(cudaMemcpy(Output_Host, InOut_Datas,
                                 BATCH * parameters.n * sizeof(Data64),
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
    GPUNTT_CUDA_CHECK(cudaFree(Forward_Omega_Table_Device));
    free(Output_Host);

    return EXIT_SUCCESS;
}