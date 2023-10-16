#include <cstdlib>  // For atoi or atof functions
#include <random>

#include "../src/ntt.cuh"

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
        unsigned long long minNumber = (unsigned long long)1 << 40;
        unsigned long long maxNumber = ((unsigned long long)1 << 40) - 1;
        std::uniform_int_distribution<unsigned long long> dis(minNumber,
                                                              maxNumber);
        unsigned long long number = dis(gen);

        std::uniform_int_distribution<unsigned long long> dis2(0, number);

        Modulus modulus(number);

        // Random data generation for polynomials
        vector<vector<Data>> input1(BATCH);
        for (int j = 0; j < BATCH; j++)
        {
            for (int i = 0; i < N; i++)
            {
                input1[j].push_back(dis2(gen));
            }
        }

        vector<Root_> forward_root_table;
        vector<Root_> inverse_root_table;
#ifdef PLANTARD_64
        for (int i = 0; i < ROOT_SIZE; i++)
        {
            __uint128_t forward = ((__uint128_t)(dis(gen)) << (__uint128_t)64) +
                                  ((__uint128_t)(dis(gen)));
            __uint128_t inverse = ((__uint128_t)(dis(gen)) << (__uint128_t)64) +
                                  ((__uint128_t)(dis(gen)));
            forward_root_table.push_back(forward);
            inverse_root_table.push_back(inverse);
        }

        Ninverse n_inv = {.x = dis(gen), .y = dis(gen)};
#else
        for (int i = 0; i < ROOT_SIZE; i++)
        {
            forward_root_table.push_back(dis2(gen));
            inverse_root_table.push_back(dis2(gen));
        }
        Ninverse n_inv = dis2(gen);
#endif

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////

        Data* InOut_Datas;

        THROW_IF_CUDA_ERROR(cudaMalloc(&InOut_Datas, BATCH * N * sizeof(Data)));

        for (int j = 0; j < BATCH; j++)
        {
            THROW_IF_CUDA_ERROR(cudaMemcpy(InOut_Datas + (N * j),
                                           input1[j].data(), N * sizeof(Data),
                                           cudaMemcpyHostToDevice));
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////

        Root* Forward_Omega_Table_Device;

        THROW_IF_CUDA_ERROR(
            cudaMalloc(&Forward_Omega_Table_Device, ROOT_SIZE * sizeof(Root)));

        THROW_IF_CUDA_ERROR(
            cudaMemcpy(Forward_Omega_Table_Device, forward_root_table.data(),
                       ROOT_SIZE * sizeof(Root), cudaMemcpyHostToDevice));

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////

        Root* Inverse_Omega_Table_Device;

        THROW_IF_CUDA_ERROR(
            cudaMalloc(&Inverse_Omega_Table_Device, ROOT_SIZE * sizeof(Root)));

        THROW_IF_CUDA_ERROR(
            cudaMemcpy(Inverse_Omega_Table_Device, inverse_root_table.data(),
                       ROOT_SIZE * sizeof(Root), cudaMemcpyHostToDevice));

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////

        unsigned long long* activity_output;
        THROW_IF_CUDA_ERROR(cudaMalloc(&activity_output,
                                       64 * 512 * sizeof(unsigned long long)));
        GPU_ACTIVITY_HOST(activity_output, 111111);
        THROW_IF_CUDA_ERROR(cudaFree(activity_output));

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////

        ntt_configuration cfg_ntt = {
            .n_power = LOGN,
            .ntt_type = FORWARD,
            .reduction_poly = ReductionPolynomial::X_N_minus,
            .zero_padding = false,
            .stream = 0};

        ntt_configuration cfg_intt = {
            .n_power = LOGN,
            .ntt_type = INVERSE,
            .reduction_poly = ReductionPolynomial::X_N_minus,
            .zero_padding = false,
            .mod_inverse = n_inv,
            .stream = 0};

        float time = 0;
        cudaEvent_t startx, stopx;
        cudaEventCreate(&startx);
        cudaEventCreate(&stopx);

        cudaEventRecord(startx);
        GPU_NTT(InOut_Datas, Forward_Omega_Table_Device, modulus, cfg_ntt,
                BATCH);

        cudaEventRecord(stopx);
        cudaEventSynchronize(stopx);
        cudaEventElapsedTime(&time, startx, stopx);
        time_measurements[loop] = time;

        // GPU_NTT(InOut_Datas, Inverse_Omega_Table_Device, modulus,
        //         cfg_intt, BATCH);

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////

        THROW_IF_CUDA_ERROR(cudaFree(InOut_Datas));
        THROW_IF_CUDA_ERROR(cudaFree(Forward_Omega_Table_Device));
        THROW_IF_CUDA_ERROR(cudaFree(Inverse_Omega_Table_Device));
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

