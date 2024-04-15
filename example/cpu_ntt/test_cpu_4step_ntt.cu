#include <cstdlib>  // For atoi or atof functions
#include <fstream>
#include <random>

#include "ntt.cuh"
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

    NTTParameters4Step parameters(LOGN, modular_reduction_type, ReductionPolynomial::X_N_minus);

    // NTT generator with certain modulus and root of unity
    NTT_4STEP_CPU generator(parameters);

    std::random_device rd;
    std::mt19937 gen(rd());
    unsigned long long minNumber = 0;
    unsigned long long maxNumber = parameters.modulus.value - 1;
    std::uniform_int_distribution<unsigned long long> dis(minNumber, maxNumber);

    // Random data generation for polynomials
    vector<Data> input1;
    vector<Data> input2;
    for(int j = 0; j < BATCH; j++)
    {
        for(int i = 0; i < parameters.n; i++)
        {
            input1.push_back(dis(gen));
            input2.push_back(dis(gen));
        }
    }

    // Performing CPU NTT
    vector<Data> ntt_input1 = generator.ntt(input1);
    vector<Data> ntt_input2 = generator.ntt(input2);
    vector<Data> output = generator.mult(ntt_input1, ntt_input2);
    vector<Data> ntt_mult_result = generator.intt(output);

    // Comparing CPU NTT multiplication results and schoolbook multiplication
    // results
    bool check = true;
    std::vector<Data> schoolbook_result = schoolbook_poly_multiplication(
        input1, input2, parameters.modulus, ReductionPolynomial::X_N_minus);

    check = check_result(ntt_mult_result.data(), schoolbook_result.data(), parameters.n);

    if(check)
    {
        cout << "All Correct." << endl;
    }

    return EXIT_SUCCESS;
}

/*

cmake . -B./cmake-build
cmake . -D MODULAR_REDUCTION_TYPE=0 -B./cmake-build

cmake --build ./cmake-build/ --parallel

./cmake-build/bin/cpu_ntt_examples

*/

//