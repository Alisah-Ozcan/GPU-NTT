#include "common.cuh"

namespace gpuntt
{
    void customAssert(bool condition, const std::string& errorMessage)
    {
        if (!condition)
        {
            throw std::invalid_argument(errorMessage);
        }
    }

    void CudaDevice()
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

        // if (chk)
        //     std::cout << "All correct." << std::endl;

        return chk;
    }

    template bool check_result<unsigned long long>(unsigned long long* input1,
                                                   unsigned long long* input2,
                                                   int size);

    template bool check_result<unsigned>(unsigned* input1, unsigned* input2,
                                         int size);

} // namespace gpuntt