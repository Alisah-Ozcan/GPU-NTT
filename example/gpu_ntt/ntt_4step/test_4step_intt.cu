#include <cstdlib>
#include <random>

#include "ntt.cuh"
#include "ntt_4step.cuh"
#include "ntt_4step_cpu.cuh"

#define DEFAULT_MODULUS

using namespace std;

int LOGN;
int BATCH;
int N;

int main(int argc, char* argv[])
{
    CudaDevice();

    if(argc < 3)
    {
        LOGN = 12;
        BATCH = 1;
    }
    else
    {
        LOGN = atoi(argv[1]);
        BATCH = atoi(argv[2]);
    }

#ifdef BARRETT_64
    ModularReductionType modular_reduction_type = ModularReductionType::BARRET;
#elif defined(GOLDILOCKS_64)
    ModularReductionType modular_reduction_type = ModularReductionType::GOLDILOCK;
#elif defined(PLANTARD_64)
    ModularReductionType modular_reduction_type = ModularReductionType::PLANTARD;
#else
#error "Please define reduction type."
#endif

    // Current 4step NTT implementation only works for ReductionPolynomial::X_N_minus!
    NTTParameters4Step parameters(LOGN, modular_reduction_type, ReductionPolynomial::X_N_minus);

    // NTT generator with certain modulus and root of unity
    NTT_4STEP_CPU generator(parameters);

    std::random_device rd;
    std::mt19937 gen(rd());
    unsigned long long minNumber = 0;
    unsigned long long maxNumber = parameters.modulus.value - 1;
    std::uniform_int_distribution<unsigned long long> dis(minNumber, maxNumber);

    // Random data generation for polynomials
    vector<vector<Data>> input1(BATCH);
    for(int j = 0; j < BATCH; j++)
    {
        for(int i = 0; i < parameters.n; i++)
        {
            input1[j].push_back(dis(gen));
        }
    }

    // Performing CPU NTT
    vector<vector<Data>> ntt_result(BATCH);
    for(int i = 0; i < BATCH; i++)
    {
        ntt_result[i] = generator.intt(input1[i]);
    }
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////

    Data* Input_Datas;

    THROW_IF_CUDA_ERROR(cudaMalloc(&Input_Datas, BATCH * parameters.n * sizeof(Data)));

    Data* Output_Datas;
    THROW_IF_CUDA_ERROR(cudaMalloc(&Output_Datas, BATCH * parameters.n * sizeof(Data)));

    for(int j = 0; j < BATCH; j++)
    {
        vector<Data> cpu_intt_transposed_input =
            generator.intt_first_transpose(input1[j]);  // INTT TRANSPOSE IN CPU

        THROW_IF_CUDA_ERROR(cudaMemcpy(Input_Datas + (parameters.n * j),
                                       cpu_intt_transposed_input.data(),
                                       parameters.n * sizeof(Data), cudaMemcpyHostToDevice));
    }

    //////////////////////////////////////////////////////////////////////////

    vector<Root_> psitable1 = parameters.gpu_root_of_unity_table_generator(
        parameters.n1_based_inverse_root_of_unity_table);
    Root* psitable_device1;
    THROW_IF_CUDA_ERROR(cudaMalloc(&psitable_device1, (parameters.n1 >> 1) * sizeof(Root)));
    THROW_IF_CUDA_ERROR(cudaMemcpy(psitable_device1, psitable1.data(),
                                   (parameters.n1 >> 1) * sizeof(Root), cudaMemcpyHostToDevice));

    vector<Root_> psitable2 = parameters.gpu_root_of_unity_table_generator(
        parameters.n2_based_inverse_root_of_unity_table);
    Root* psitable_device2;
    THROW_IF_CUDA_ERROR(cudaMalloc(&psitable_device2, (parameters.n2 >> 1) * sizeof(Root)));
    THROW_IF_CUDA_ERROR(cudaMemcpy(psitable_device2, psitable2.data(),
                                   (parameters.n2 >> 1) * sizeof(Root), cudaMemcpyHostToDevice));

    Root* W_Table_device;
    THROW_IF_CUDA_ERROR(cudaMalloc(&W_Table_device, parameters.n * sizeof(Root)));
    THROW_IF_CUDA_ERROR(cudaMemcpy(W_Table_device, parameters.W_inverse_root_of_unity_table.data(),
                                   parameters.n * sizeof(Root), cudaMemcpyHostToDevice));

    //////////////////////////////////////////////////////////////////////////

    Modulus* test_modulus;
    THROW_IF_CUDA_ERROR(cudaMalloc(&test_modulus, sizeof(Modulus)));

    Modulus test_modulus_[1] = {parameters.modulus};

    THROW_IF_CUDA_ERROR(
        cudaMemcpy(test_modulus, test_modulus_, sizeof(Modulus), cudaMemcpyHostToDevice));

    Ninverse* test_ninverse;
    THROW_IF_CUDA_ERROR(cudaMalloc(&test_ninverse, sizeof(Ninverse)));

    Ninverse test_ninverse_[1] = {parameters.n_inv};

    THROW_IF_CUDA_ERROR(
        cudaMemcpy(test_ninverse, test_ninverse_, sizeof(Ninverse), cudaMemcpyHostToDevice));

    ntt4step_rns_configuration cfg_intt = {.n_power = LOGN,
                                      .ntt_type = INVERSE,
                                      .mod_inverse = test_ninverse,
                                      .stream = 0};

    //////////////////////////////////////////////////////////////////////////

    GPU_4STEP_NTT(Input_Datas, Output_Datas, psitable_device1, psitable_device2, W_Table_device,
                  test_modulus, cfg_intt, BATCH, 1);

    GPU_Transpose(Output_Datas, Input_Datas, parameters.n1, parameters.n2, parameters.logn, BATCH);

    vector<Data> Output_Host(parameters.n * BATCH);
    cudaMemcpy(Output_Host.data(), Input_Datas, parameters.n * BATCH * sizeof(Data),
               cudaMemcpyDeviceToHost);

    // Comparing GPU NTT results and CPU NTT results
    bool check = true;
    for(int i = 0; i < BATCH; i++)
    {
        check = check_result(Output_Host.data() + (i * parameters.n), ntt_result[i].data(),
                             parameters.n);

        if(!check)
        {
            cout << "(in " << i << ". Poly.)" << endl;
            break;
        }

        if((i == (BATCH - 1)) && check)
        {
            cout << "All Correct." << endl;
        }
    }

    return EXIT_SUCCESS;
}

/*

cmake . -B./cmake-build
cmake . -D MODULAR_REDUCTION_TYPE=0 -B./cmake-build
cmake . -D MODULAR_REDUCTION_TYPE=1 -B./cmake-build
cmake . -D MODULAR_REDUCTION_TYPE=2 -B./cmake-build

cmake --build ./cmake-build/ --parallel

./cmake-build/bin/gpu_ntt_examples

*/