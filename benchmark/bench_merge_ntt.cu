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

    if (argc < 3)
    {
        LOGN = 12;
        BATCH = 1;
    }
    else
    {
        LOGN = atoi(argv[1]);
        BATCH = atoi(argv[2]);

        if ((LOGN < 12) || (28 < LOGN))
        {
            throw std::runtime_error("LOGN should be in range 12 to 28.");
        }
    }

    // NTT generator with certain modulus and root of unity

    int N = 1 << LOGN;
    int ROOT_SIZE = N >> 1;

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

        vector<Root64> forward_root_table;
        vector<Root64> inverse_root_table;

        for (int i = 0; i < ROOT_SIZE; i++)
        {
            forward_root_table.push_back(dis2(gen));
            inverse_root_table.push_back(dis2(gen));
        }
        Ninverse64 n_inv = dis2(gen);

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////

        Data64* InOut_Datas;

        GPUNTT_CUDA_CHECK(cudaMalloc(&InOut_Datas, BATCH * N * sizeof(Data64)));

        for (int j = 0; j < BATCH; j++)
        {
            GPUNTT_CUDA_CHECK(cudaMemcpy(InOut_Datas + (N * j),
                                         input1[j].data(), N * sizeof(Data64),
                                         cudaMemcpyHostToDevice));
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////

        Root64* Forward_Omega_Table_Device;

        GPUNTT_CUDA_CHECK(cudaMalloc(&Forward_Omega_Table_Device,
                                     ROOT_SIZE * sizeof(Root64)));

        GPUNTT_CUDA_CHECK(
            cudaMemcpy(Forward_Omega_Table_Device, forward_root_table.data(),
                       ROOT_SIZE * sizeof(Root64), cudaMemcpyHostToDevice));

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////

        unsigned long long* activity_output;
        GPUNTT_CUDA_CHECK(cudaMalloc(&activity_output,
                                     64 * 512 * sizeof(unsigned long long)));
        GPU_ACTIVITY_HOST(activity_output, 111111);
        GPUNTT_CUDA_CHECK(cudaFree(activity_output));

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////

        Modulus64* modulus_device;
        GPUNTT_CUDA_CHECK(cudaMalloc(&modulus_device, sizeof(Modulus64)));

        Modulus64 test_modulus_[1] = {modulus};

        GPUNTT_CUDA_CHECK(cudaMemcpy(modulus_device, test_modulus_,
                                     sizeof(Modulus64),
                                     cudaMemcpyHostToDevice));

        Ninverse64* ninverse_device;
        GPUNTT_CUDA_CHECK(cudaMalloc(&ninverse_device, sizeof(Ninverse64)));

        Ninverse64 test_ninverse_[1] = {n_inv};

        GPUNTT_CUDA_CHECK(cudaMemcpy(ninverse_device, test_ninverse_,
                                     sizeof(Ninverse64),
                                     cudaMemcpyHostToDevice));

        ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = LOGN,
            .ntt_type = FORWARD,
            .reduction_poly = ReductionPolynomial::X_N_minus,
            .zero_padding = false,
            .stream = 0};

        float time = 0;
        cudaEvent_t startx, stopx;
        cudaEventCreate(&startx);
        cudaEventCreate(&stopx);

        cudaEventRecord(startx);
        GPU_NTT_Inplace(InOut_Datas, Forward_Omega_Table_Device, modulus_device,
                        cfg_ntt, BATCH, 1);

        cudaEventRecord(stopx);
        cudaEventSynchronize(stopx);
        cudaEventElapsedTime(&time, startx, stopx);
        time_measurements[loop] = time;

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////

        GPUNTT_CUDA_CHECK(cudaFree(InOut_Datas));
        GPUNTT_CUDA_CHECK(cudaFree(Forward_Omega_Table_Device));
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