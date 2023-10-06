// --------------------- //
// Authors: Alisah Ozcan
// --------------------- //

#ifndef NTT_CPU_H
#define NTT_CPU_H

#include "nttparameters.cuh"

std::vector<Data> schoolbook_poly_multiplication(
    std::vector<Data> a, std::vector<Data> b, Modulus modulus,
    ReductionPolynomial reduction_poly);

class NTT_CPU
{
   public:
    NTTParameters parameters;

    NTT_CPU(NTTParameters parameters_);

   public:
    std::vector<Data> mult(std::vector<Data> &input1,
                           std::vector<Data> &input2);

    std::vector<Data> ntt(std::vector<Data> &input);

    std::vector<Data> intt(std::vector<Data> &input);
};

#endif  // NTT_CPU_H

/*


std::vector<Data> schoolbook_poly_multiplication(
    std::vector<Data> a, std::vector<Data> b, Modulus modulus,
    ReductionPolynomial reduction_poly);

class NTT_CPU
{
   public:
    bool ntt_domain;           // static
    NTTParameters parameters;  // static

    std::vector<Data> poly;  // static

    NTT_CPU(NTTParameters parameters_, std::vector<Data> poly_);

   public:
    void mult(NTT_CPU& b);

    void ntt();

    void intt();
};

*/