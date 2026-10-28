// Copyright 2023-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

// Polynomial multiplication with the CPU NTT, checked against schoolbook
// multiplication.
//
// Usage: cpu_merge_ntt_examples [RING_SIZE_IN_LOG2] [BATCH_SIZE] [32|64]

#include <cstdlib>
#include <random>

#include "gpuntt/ntt_merge/ntt_cpu.cuh"
#include "example_util.cuh"

using namespace std;
using namespace gpuntt;
using namespace gpuntt_example;

template <typename TestDataType> bool run(int logn, int batch)
{
    NTTParameters<TestDataType> parameters(logn,
                                           ReductionPolynomial::X_N_minus);

    // NTT generator with certain modulus and root of unity
    NTTCPU<TestDataType> generator(parameters);

    std::random_device rd;
    std::mt19937 gen(rd());
    TestDataType minNumber = 0;
    TestDataType maxNumber = parameters.modulus.value - 1;
    std::uniform_int_distribution<TestDataType> dis(minNumber, maxNumber);

    // Random data generation for polynomials
    vector<vector<TestDataType>> input1(batch);
    vector<vector<TestDataType>> input2(batch);
    for (int j = 0; j < batch; j++)
    {
        for (int i = 0; i < parameters.n; i++)
        {
            input1[j].push_back(dis(gen));
            input2[j].push_back(dis(gen));
        }
    }

    // Performing CPU NTT
    vector<vector<TestDataType>> ntt_mult_result(batch);
    for (int i = 0; i < batch; i++)
    {
        vector<TestDataType> ntt_input1 = generator.ntt(input1[i]);
        vector<TestDataType> ntt_input2 = generator.ntt(input2[i]);
        vector<TestDataType> output = generator.mult(ntt_input1, ntt_input2);
        ntt_mult_result[i] = generator.intt(output);
    }

    // Comparing CPU NTT multiplication results and schoolbook multiplication
    // results
    bool check = true;
    for (int i = 0; i < batch; i++)
    {
        std::vector<TestDataType> schoolbook_result =
            schoolbook_poly_multiplication<TestDataType>(
                input1[i], input2[i], parameters.modulus,
                ReductionPolynomial::X_N_minus);

        check = check_result(ntt_mult_result[i].data(),
                             schoolbook_result.data(), parameters.n);
        if (!check)
        {
            cout << "(in " << i << ". Poly.)" << endl;
            break;
        }

        if ((i == (batch - 1)) && check)
        {
            cout << "All Correct." << endl;
        }
    }

    return check;
}

int main(int argc, char* argv[])
{
    CudaDevice();

    const ExampleArgs args = parse_example_args(argc, argv);
    const bool correct = (args.bits == 32) ? run<Data32>(args.logn, args.batch)
                                           : run<Data64>(args.logn, args.batch);

    return correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
