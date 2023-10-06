// --------------------- //
// Authors: Alisah Ozcan
// --------------------- //

#ifndef NTT_PARAMETERS_H
#define NTT_PARAMETERS_H

#include <vector>

#include "common.cuh"
#include "modular_arith.cuh"

int bitreverse(int index, int n_power);

enum ReductionPolynomial
{
    X_N_plus,
    X_N_minus
};  // X_N_minus: X^n - 1, X_N_plus: X^n + 1

enum ModularReductionType
{
    BARRET,
    PLANTARD,
    GOLDILOCK
};

// For any prime(64-bit)
struct NTTFactors
{
    Modulus modulus;
    Data omega;
    Data psi;

    // Constructor to initialize the NTTFactors
    __host__ NTTFactors(Modulus q_, Data omega_, Data psi_)
    {
        modulus = q_;
        omega = omega_;
        psi = psi_;
    }
    __host__ NTTFactors() {}

    // TODO: add check mechanism here
};

class NTTParameters
{
   public:
    int logn;
    Data n;

    ReductionPolynomial poly_reduction;
    ModularReductionType modular_reduction;

    Modulus modulus;

    Data omega;
    Data psi;

    Ninverse n_inv;

    Data root_of_unity;
    Data inverse_root_of_unity;

    Data root_of_unity_size;

    std::vector<Data> forward_root_of_unity_table;
    std::vector<Data> inverse_root_of_unity_table;

#ifdef PLANTARD_64
    __uint128_t R;  // For plantard Reduction
#endif

    // For testing all cases with barretti goldilock and plantard reduction
    NTTParameters(int LOGN, ModularReductionType modular_reduction_type,
                  ReductionPolynomial poly_reduce_type);

    // For any prime(64-bit)
    NTTParameters(int LOGN, NTTFactors ntt_factors,
                  ReductionPolynomial poly_reduce_type);

    NTTParameters();

   private:
    Modulus modulus_pool();

    Data omega_pool();

    Data psi_pool();

    void forward_root_of_unity_table_generator();

    void inverse_root_of_unity_table_generator();

    void n_inverse_generator();

#ifdef PLANTARD_64
    __uint128_t R_pool();
#endif

   public:
    std::vector<Root_> gpu_root_of_unity_table_generator(
        std::vector<Data> table);
};

#endif  // NTT_PARAMETERS_H
