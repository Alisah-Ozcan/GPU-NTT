// Copyright 2023-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef GPUNTT_EXAMPLE_UTIL_H
#define GPUNTT_EXAMPLE_UTIL_H

#include <cstdio>
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
