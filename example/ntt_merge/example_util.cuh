// Copyright 2023-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef GPUNTT_EXAMPLE_UTIL_H
#define GPUNTT_EXAMPLE_UTIL_H

#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "gpuntt/common/common.cuh"

namespace gpuntt_example
{
    inline void CudaDevice()
    {
        cudaDeviceProp deviceProp;
        int deviceID = 0;

        GPUNTT_CUDA_CHECK(cudaSetDevice(deviceID));
        GPUNTT_CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, deviceID));
        printf("GPU Device %d: %s (compute capability %d.%d)\n\n", deviceID,
               deviceProp.name, deviceProp.major, deviceProp.minor);
    }

    // Command line of the merge examples:
    //   <program> [RING_SIZE_IN_LOG2] [BATCH_SIZE] [32|64]
    // with defaults 12, 1 and 64 (the coefficient type Data32 or Data64).
    struct ExampleArgs
    {
        int logn = 12;
        int batch = 1;
        int bits = 64;
    };

    inline ExampleArgs parse_example_args(int argc, char* argv[])
    {
        ExampleArgs args;
        if (argc > 1)
            args.logn = atoi(argv[1]);
        if (argc > 2)
            args.batch = atoi(argv[2]);
        if (argc > 3)
            args.bits = atoi(argv[3]);

        if (argc > 4 || args.logn < 1 || args.batch < 1 ||
            (args.bits != 32 && args.bits != 64))
        {
            std::cerr << "Usage: " << argv[0]
                      << " [RING_SIZE_IN_LOG2] [BATCH_SIZE] [32|64]"
                      << std::endl;
            std::exit(EXIT_FAILURE);
        }
        return args;
    }

    template <typename T> bool check_result(T* input1, T* input2, int size)
    {
        bool chk = true;
        for (int i = 0; i < size; i++)
        {
            if (input1[i] != input2[i])
            {
                std::cout << "Error in index: " << i << " -> " << input1[i]
                          << " - " << input2[i] << " " << std::endl;
                chk = false;
                break;
            }
        }

        return chk;
    }

} // namespace gpuntt_example
#endif // GPUNTT_EXAMPLE_UTIL_H
