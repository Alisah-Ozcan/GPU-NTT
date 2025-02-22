// Copyright 2024 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <cstdlib> // For atoi or atof functions
#include <random>

#include "ntt.cuh"
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
    std::mt19937 gen(rd());
    Data64 minNumber = 0;
    Data64 maxNumber = parameters.modulus.value - 1;
    std::uniform_int_distribution<Data64> dis(minNumber, maxNumber);

    // Random data generation for polynomials
    vector<vector<Data64>> input1(BATCH);
    vector<vector<Data64>> input2(BATCH);
    for (int j = 0; j < BATCH; j++)
    {
        for (int i = 0; i < parameters.n; i++)
        {
            input1[j].push_back(dis(gen));
            input2[j].push_back(dis(gen));
        }
    }

    // Performing CPU NTT
    vector<vector<Data64>> ntt_mult_result(BATCH);
    for (int i = 0; i < BATCH; i++)
    {
        vector<Data64> ntt_input1 = generator.ntt(input1[i]);
        vector<Data64> ntt_input2 = generator.ntt(input2[i]);
        vector<Data64> output = generator.mult(ntt_input1, ntt_input2);
        ntt_mult_result[i] = generator.intt(output);
    }

    // Comparing CPU NTT multiplication results and schoolbook multiplication
    // results
    bool check = true;
    for (int i = 0; i < BATCH; i++)
    {
        std::vector<Data64> schoolbook_result =
            schoolbook_poly_multiplication<Data64>(
                input1[i], input2[i], parameters.modulus,
                ReductionPolynomial::X_N_minus);

        check = check_result(ntt_mult_result[i].data(),
                             schoolbook_result.data(), parameters.n);
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

cmake --build ./cmake-build/ --parallel

./cmake-build/bin/cpu_ntt_examples

*/

//