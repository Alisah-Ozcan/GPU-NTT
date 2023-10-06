#include <cstdlib>  // For atoi or atof functions
#include <random>

#include "../src/ntt.cuh"

#define DEFAULT_MODULUS

using namespace std;

int LOGN;
int BATCH;
int N;

int main(int argc, char* argv[])
{
    CudaDevice();

    if (argc == 0)
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
    std::mt19937 gen(rd());
    unsigned long long minNumber = 0;
    unsigned long long maxNumber = parameters.modulus.value - 1;
    std::uniform_int_distribution<unsigned long long> dis(minNumber, maxNumber);

    // Random data generation for polynomials
    vector<vector<Data>> input1(BATCH);
    vector<vector<Data>> input2(BATCH);
    for (int j = 0; j < BATCH; j++)
    {
        for (int i = 0; i < parameters.n; i++)
        {
            input1[j].push_back(dis(gen));
            input2[j].push_back(dis(gen));
        }
    }

    // Performing CPU NTT
    vector<vector<Data>> ntt_mult_result(BATCH);
    for (int i = 0; i < BATCH; i++)
    {
        vector<Data> ntt_input1 = generator.ntt(input1[i]);
        vector<Data> ntt_input2 = generator.ntt(input2[i]);
        vector<Data> output = generator.mult(ntt_input1, ntt_input2);
        ntt_mult_result[i] = generator.intt(output);
    }

    // Comparing CPU NTT multiplication results and schoolbook multiplication
    // results
    bool check = true;
    for (int i = 0; i < BATCH; i++)
    {
        std::vector<Data> schoolbook_result = schoolbook_poly_multiplication(
            input1[i], input2[i], parameters.modulus,
            ReductionPolynomial::X_N_minus);

        check = check_result(ntt_mult_result[i].data(),
                             schoolbook_result.data(), parameters.n);
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

    return EXIT_SUCCESS;
}


