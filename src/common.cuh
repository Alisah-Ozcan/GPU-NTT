#pragma once
#include <cuda_runtime.h>

#include <cassert>
#include <exception>
#include <iostream>
#include <string>

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

#define THROW_IF_CUDA_ERROR(err)                            \
    do                                                      \
    {                                                       \
        cudaError_t error = err;                            \
        if (error != cudaSuccess)                           \
        {                                                   \
            throw CudaException(__FILE__, __LINE__, error); \
        }                                                   \
    } while (0)

void customAssert(bool condition, const std::string& errorMessage);

void CudaDevice();

float calculate_mean(const float array[], int size);

float calculate_standard_deviation(const float array[], int size);

float find_best_average(const float array[], int array_size, int num_elements);

float find_min_average(const float array[], int array_size, int num_elements);

template <typename T>
bool check_result(T* input1, T* input2, int size);
