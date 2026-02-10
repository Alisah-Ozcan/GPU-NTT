// Copyright 2024 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef COMMON_NTT_H
#define COMMON_NTT_H

#include <cuda_runtime.h>

#include <cassert>
#include <cstdint>
#include <exception>
#include <iostream>
#include <string>

namespace gpuntt
{

    class CudaException : public std::exception
    {
      public:
        CudaException(const std::string& file, int line, cudaError_t error)
            : file_(file), line_(line), error_(error)
        {
        }

        const char* what() const noexcept override
        {
            return m_error_string.c_str();
        }

      private:
        std::string file_;
        int line_;
        cudaError_t error_;
        std::string m_error_string = "CUDA Error in " + file_ + " at line " +
                                     std::to_string(line_) + ": " +
                                     cudaGetErrorString(error_);
    };

#define GPUNTT_CUDA_CHECK(err)                                                 \
    do                                                                         \
    {                                                                          \
        cudaError_t error = err;                                               \
        if (error != cudaSuccess)                                              \
        {                                                                      \
            throw CudaException(__FILE__, __LINE__, error);                    \
        }                                                                      \
    } while (0)

    void customAssert(bool condition, const std::string& errorMessage);

    void CudaDevice();

    template <typename T> bool check_result(T* input1, T* input2, int size);

} // namespace gpuntt
#endif // COMMON_NTT_H
