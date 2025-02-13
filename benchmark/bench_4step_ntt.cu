// Copyright 2024 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <cstdlib>
#include <random>

#include "ntt_4step.cuh"
#include "ntt_4step_cpu.cuh"

#define DEFAULT_MODULUS

using namespace std;

int LOGN;
int BATCH;

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

        if ((LOGN < 12) || (24 < LOGN))
        {
            throw std::runtime_error("LOGN should be in range 12 to 24.");
        }
    }

    ModularReductionType modular_reduction_type = ModularReductionType::BARRET;

    // Current 4step NTT implementation only works for
    // ReductionPolynomial::X_N_minus!
    NTTParameters4Step<Data64> parameters(LOGN, modular_reduction_type,
                                          ReductionPolynomial::X_N_minus);

    int N = parameters.n;

    const int test_count = 100;
    const int bestof = 25;
    float time_measurements[test_count];
    for (int loop = 0; loop < test_count; loop++)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        unsigned long long minNumber = (unsigned long long) 1 << 40;
        unsigned long long maxNumber = ((unsigned long long) 1 << 40) - 1;
        std::uniform_int_distribution<unsigned long long> dis(minNumber,
                                                              maxNumber);
        unsigned long long number = dis(gen);

        std::uniform_int_distribution<unsigned long long> dis2(0, number);

        Modulus64 modulus(number);

        // Random data generation for polynomials
        vector<vector<Data64>> input1(BATCH);
        for (int j = 0; j < BATCH; j++)
        {
            for (int i = 0; i < N; i++)
            {
                input1[j].push_back(dis2(gen));
            }
        }

        vector<Root64> forward_root_table1;

        for (int i = 0; i < (parameters.n1 >> 1); i++)
        {
            forward_root_table1.push_back(dis2(gen));
        }
        Ninverse64 n_inv = dis2(gen);

        vector<Root64> forward_root_table2;
        for (int i = 0; i < (parameters.n2 >> 1); i++)
        {
            forward_root_table2.push_back(dis2(gen));
        }

        vector<Root64> W_root_table;
        for (int i = 0; i < parameters.n; i++)
        {
            W_root_table.push_back(dis2(gen));
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////

        Data64* Input_Datas;

        THROW_IF_CUDA_ERROR(
            cudaMalloc(&Input_Datas, BATCH * N * sizeof(Data64)));

        for (int j = 0; j < BATCH; j++)
        {
            THROW_IF_CUDA_ERROR(cudaMemcpy(Input_Datas + (N * j),
                                           input1[j].data(), N * sizeof(Data64),
                                           cudaMemcpyHostToDevice));
        }

        Data64* Output_Datas;

        THROW_IF_CUDA_ERROR(
            cudaMalloc(&Output_Datas, BATCH * N * sizeof(Data64)));

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////

        Root64* Forward_Omega_Table1_Device;
        THROW_IF_CUDA_ERROR(cudaMalloc(&Forward_Omega_Table1_Device,
                                       (parameters.n1 >> 1) * sizeof(Root64)));
        THROW_IF_CUDA_ERROR(cudaMemcpy(
            Forward_Omega_Table1_Device, forward_root_table1.data(),
            (parameters.n1 >> 1) * sizeof(Root64), cudaMemcpyHostToDevice));

        Root64* Forward_Omega_Table2_Device;
        THROW_IF_CUDA_ERROR(cudaMalloc(&Forward_Omega_Table2_Device,
                                       (parameters.n2 >> 1) * sizeof(Root64)));
        THROW_IF_CUDA_ERROR(cudaMemcpy(
            Forward_Omega_Table2_Device, forward_root_table2.data(),
            (parameters.n2 >> 1) * sizeof(Root64), cudaMemcpyHostToDevice));

        Root64* W_Table_Device;
        THROW_IF_CUDA_ERROR(
            cudaMalloc(&W_Table_Device, parameters.n * sizeof(Root64)));
        THROW_IF_CUDA_ERROR(cudaMemcpy(W_Table_Device, W_root_table.data(),
                                       (parameters.n >> 1) * sizeof(Root64),
                                       cudaMemcpyHostToDevice));

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////

        unsigned long long* activity_output;
        THROW_IF_CUDA_ERROR(cudaMalloc(&activity_output,
                                       64 * 512 * sizeof(unsigned long long)));
        GPU_ACTIVITY_HOST(activity_output, 111111);
        THROW_IF_CUDA_ERROR(cudaFree(activity_output));

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////

        Modulus64* modulus_device;
        THROW_IF_CUDA_ERROR(cudaMalloc(&modulus_device, sizeof(Modulus64)));

        Modulus64 test_modulus_[1] = {modulus};

        THROW_IF_CUDA_ERROR(cudaMemcpy(modulus_device, test_modulus_,
                                       sizeof(Modulus64),
                                       cudaMemcpyHostToDevice));

        Ninverse64* ninverse_device;
        THROW_IF_CUDA_ERROR(cudaMalloc(&ninverse_device, sizeof(Ninverse64)));

        Ninverse64 test_ninverse_[1] = {n_inv};

        THROW_IF_CUDA_ERROR(cudaMemcpy(ninverse_device, test_ninverse_,
                                       sizeof(Ninverse64),
                                       cudaMemcpyHostToDevice));

        ntt4step_configuration<Data64> cfg_ntt = {
            .n_power = LOGN, .ntt_type = FORWARD, .stream = 0};

        float time = 0;
        cudaEvent_t startx, stopx;
        cudaEventCreate(&startx);
        cudaEventCreate(&stopx);

        cudaEventRecord(startx);

        // GPU_4STEP_NTT(Input_Datas, Output_Datas, Forward_Omega_Table1_Device,
        //               Forward_Omega_Table2_Device, W_Table_Device,
        //               modulus_device, cfg_ntt, BATCH, 1);

        GPU_4STEP_NTT(Input_Datas, Output_Datas, Forward_Omega_Table1_Device,
                      Forward_Omega_Table2_Device, W_Table_Device, modulus,
                      cfg_ntt, BATCH);

        cudaEventRecord(stopx);
        cudaEventSynchronize(stopx);
        cudaEventElapsedTime(&time, startx, stopx);
        time_measurements[loop] = time;

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////

        THROW_IF_CUDA_ERROR(cudaFree(Input_Datas));
        THROW_IF_CUDA_ERROR(cudaFree(Output_Datas));
        THROW_IF_CUDA_ERROR(cudaFree(Forward_Omega_Table1_Device));
        THROW_IF_CUDA_ERROR(cudaFree(Forward_Omega_Table2_Device));
        THROW_IF_CUDA_ERROR(cudaFree(W_Table_Device));
    }

    cout << endl
         << endl
         << "Average: " << calculate_mean(time_measurements, test_count)
         << endl;
    cout << "Best Average: "
         << find_min_average(time_measurements, test_count, bestof) << endl;

    cout << "Standart Deviation: "
         << calculate_standard_deviation(time_measurements, test_count) << endl;

    return EXIT_SUCCESS;
}