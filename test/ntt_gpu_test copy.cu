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

        int device = 0; // Assuming you are using device 0
    cudaSetDevice(device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    std::cout << "Maximum Grid Size: " << prop.maxGridSize[0] << " x " << prop.maxGridSize[1] << " x " << prop.maxGridSize[2] << std::endl;


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

#ifdef BARRETT_64
    ModularReductionType modular_reduction_type = ModularReductionType::BARRET;
#elif defined(GOLDILOCKS_64)
    ModularReductionType modular_reduction_type =
        ModularReductionType::GOLDILOCK;
#elif defined(PLANTARD_64)
    ModularReductionType modular_reduction_type =
        ModularReductionType::PLANTARD;
#else
#error "Please define reduction type."
#endif

#ifdef DEFAULT_MODULUS
    NTTParameters parameters(LOGN, modular_reduction_type,
                             ReductionPolynomial::X_N_minus);
#else
    NTTFactors factor((Modulus)576460752303415297, 288482366111684746,
                      238394956950829);
    NTTParameters parameters(LOGN, factor, ReductionPolynomial::X_N_minus);
#endif

    // NTT generator with certain modulus and root of unity
    NTT_CPU generator(parameters);

    std::random_device rd;
    //std::mt19937 gen(rd());
    std::mt19937 gen(0);
    unsigned long long minNumber = 0;
    unsigned long long maxNumber = parameters.modulus.value - 1;
    std::uniform_int_distribution<unsigned long long> dis(minNumber, maxNumber);

    // Random data generation for polynomials
    vector<vector<Data>> input1(BATCH);
    for (int j = 0; j < BATCH; j++)
    {
        for (int i = 0; i < parameters.n; i++)
        {
            input1[j].push_back(dis(gen));
        }
    }

    // Performing CPU NTT
    vector<vector<Data>> ntt_result(BATCH);
    for (int i = 0; i < BATCH; i++)
    {
        ntt_result[i] = generator.ntt(input1[i]);
    }
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////

    Data* InOut_Datas;

    THROW_IF_CUDA_ERROR(
        cudaMalloc(&InOut_Datas, BATCH * parameters.n * sizeof(Data)));

    for (int j = 0; j < BATCH; j++)
    {
        THROW_IF_CUDA_ERROR(
            cudaMemcpy(InOut_Datas + (parameters.n * j), input1[j].data(),
                       parameters.n * sizeof(Data), cudaMemcpyHostToDevice));
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////

    Root* Forward_Omega_Table_Device;

    THROW_IF_CUDA_ERROR(
        cudaMalloc(&Forward_Omega_Table_Device,
                   parameters.root_of_unity_size * sizeof(Root)));

    vector<Root_> forward_omega_table =
        parameters.gpu_root_of_unity_table_generator(
            parameters.forward_root_of_unity_table);

    THROW_IF_CUDA_ERROR(cudaMemcpy(
        Forward_Omega_Table_Device, forward_omega_table.data(),
        parameters.root_of_unity_size * sizeof(Root), cudaMemcpyHostToDevice));

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////

    Root* Inverse_Omega_Table_Device;

    THROW_IF_CUDA_ERROR(
        cudaMalloc(&Inverse_Omega_Table_Device,
                   parameters.root_of_unity_size * sizeof(Root)));

    vector<Root_> inverse_omega_table =
        parameters.gpu_root_of_unity_table_generator(
            parameters.inverse_root_of_unity_table);
    THROW_IF_CUDA_ERROR(cudaMemcpy(
        Inverse_Omega_Table_Device, inverse_omega_table.data(),
        parameters.root_of_unity_size * sizeof(Root), cudaMemcpyHostToDevice));

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////

    ntt_configuration cfg_ntt = {
        .n_power = LOGN,
        .ntt_type = FORWARD,
        .reduction_poly = ReductionPolynomial::X_N_minus,
        .zero_padding = false,
        .stream = 0};
    GPU_NTT_Inplace(InOut_Datas, Forward_Omega_Table_Device, parameters.modulus,
            cfg_ntt, BATCH);

    ntt_configuration cfg_intt = {
        .n_power = LOGN,
        .ntt_type = INVERSE,
        .reduction_poly = ReductionPolynomial::X_N_minus,
        .zero_padding = false,
        .mod_inverse = parameters.n_inv,
        .stream = 0};
    GPU_NTT_Inplace(InOut_Datas, Inverse_Omega_Table_Device, parameters.modulus,
            cfg_intt, BATCH);

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////

    Data* Output_Host;

    Output_Host = (Data*)malloc(BATCH * parameters.n * sizeof(Data));

    THROW_IF_CUDA_ERROR(cudaMemcpy(Output_Host, InOut_Datas,
                                   BATCH * parameters.n * sizeof(Data),
                                   cudaMemcpyDeviceToHost));

    // Comparing GPU NTT results and CPU NTT results
    bool check = true;
    for (int i = 0; i < BATCH; i++)
    {
        check = check_result(Output_Host + (i * parameters.n), input1[i].data(),
                             parameters.n);

        //check = check_result(Output_Host + (i * parameters.n), ntt_result[i].data(),
        //                     parameters.n);
        
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

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////

    THROW_IF_CUDA_ERROR(cudaFree(InOut_Datas));
    THROW_IF_CUDA_ERROR(cudaFree(Forward_Omega_Table_Device));
    THROW_IF_CUDA_ERROR(cudaFree(Inverse_Omega_Table_Device));
    free(Output_Host);

    return EXIT_SUCCESS;
}
