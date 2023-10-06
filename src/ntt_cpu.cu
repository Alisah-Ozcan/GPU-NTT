// --------------------- //
// Authors: Alisah Ozcan
// --------------------- //

#include "ntt_cpu.cuh"

std::vector<Data> schoolbook_poly_multiplication(
    std::vector<Data> a, std::vector<Data> b, Modulus modulus,
    ReductionPolynomial reduction_poly)
{
    int length = a.size();
    std::vector<Data> mult_vector(length * 2, 0);

    for (int i = 0; i < length; i++)
    {
        for (int j = 0; j < length; j++)
        {
            Data mult_result = VALUE::mult(a[i], b[j], modulus);
            mult_vector[i + j] =
                VALUE::add(mult_vector[i + j], mult_result, modulus);
        }
    }

    std::vector<Data> result(length, 0);
    if (reduction_poly == ReductionPolynomial::X_N_minus)
    {
        for (int i = 0; i < length; i++)
        {
            result[i] =
                VALUE::add(mult_vector[i], mult_vector[i + length], modulus);
        }
    }
    else if (reduction_poly == ReductionPolynomial::X_N_plus)
    {
        for (int i = 0; i < length; i++)
        {
            result[i] =
                VALUE::sub(mult_vector[i], mult_vector[i + length], modulus);
        }
    }
    else
    {
        throw std::runtime_error("Poly reduction type is not supported!");
    }

    return result;
}

NTT_CPU::NTT_CPU(NTTParameters parameters_) { parameters = parameters_; }

std::vector<Data> NTT_CPU::mult(std::vector<Data> &input1,
                                std::vector<Data> &input2)
{
    std::vector<Data> output;
    for (int i = 0; i < parameters.n; i++)
    {
        output.push_back(VALUE::mult(input1[i], input2[i], parameters.modulus));
    }

    return output;
}

std::vector<Data> NTT_CPU::ntt(std::vector<Data> &input)
{
    // Merged NTT with pre-processing (optimized) (iterative)
    // This is not NTT, this is pre-processing + NTT
    // (see: https://eprint.iacr.org/2016/504.pdf)

    std::vector<Data> output = input;

    int t = parameters.n;
    int m = 1;

    while (m < parameters.n)
    {
        t = t >> 1;

        for (int i = 0; i < m; i++)
        {
            int j1 = 2 * i * t;
            int j2 = j1 + t - 1;

            int index;
            if (parameters.poly_reduction == ReductionPolynomial::X_N_minus)
            {
                index = bitreverse(i, parameters.logn - 1);
            }
            else
            {  // poly_reduce_type = ReductionPolynomial::X_N_plus
                index = bitreverse(m + i, parameters.logn);
            }

            Data S = parameters.forward_root_of_unity_table[index];

            for (int j = j1; j < (j2 + 1); j++)
            {
                Data U = output[j];
                Data V = VALUE::mult(output[j + t], S, parameters.modulus);

                output[j] = VALUE::add(U, V, parameters.modulus);
                output[j + t] = VALUE::sub(U, V, parameters.modulus);
            }
        }

        m = m << 1;
    }

    return output;
}

std::vector<Data> NTT_CPU::intt(std::vector<Data> &input)
{
    // Merged INTT with post-processing (optimized) (iterative)
    // This is not NTT, this is pre-processing + NTT
    // (see: https://eprint.iacr.org/2016/504.pdf)

    std::vector<Data> output = input;

    int t = 1;
    int m = parameters.n;
    while (m > 1)
    {
        int j1 = 0;
        int h = m >> 1;
        for (int i = 0; i < h; i++)
        {
            int j2 = j1 + t - 1;
            int index;
            if (parameters.poly_reduction == ReductionPolynomial::X_N_minus)
            {
                index = bitreverse(i, parameters.logn - 1);
            }
            else
            {  // poly_reduce_type = ReductionPolynomial::X_N_plus
                index = bitreverse(h + i, parameters.logn);
            }

            Data S = parameters.inverse_root_of_unity_table[index];

            for (int j = j1; j < (j2 + 1); j++)
            {
                Data U = output[j];
                Data V = output[j + t];

                output[j] = VALUE::add(U, V, parameters.modulus);
                output[j + t] = VALUE::sub(U, V, parameters.modulus);
                output[j + t] =
                    VALUE::mult(output[j + t], S, parameters.modulus);
            }

            j1 = j1 + (t << 1);
        }

        t = t << 1;
        m = m >> 1;
    }

    Data n_inv = VALUE::modinv(parameters.n, parameters.modulus);

    for (int i = 0; i < parameters.n; i++)
    {
        output[i] = VALUE::mult(output[i], n_inv, parameters.modulus);
    }

    return output;
}

/*

std::vector<Data> schoolbook_poly_multiplication(
    std::vector<Data> a, std::vector<Data> b, Modulus modulus,
    ReductionPolynomial reduction_poly)
{
    int length = a.size();
    std::vector<Data> mult_vector(length * 2, 0);

    for (int i = 0; i < length; i++)
    {
        for (int j = 0; j < length; j++)
        {
            Data mult_result = VALUE::mult(a[i], b[j], modulus);
            mult_vector[i + j] =
                VALUE::add(mult_vector[i + j], mult_result, modulus);
        }
    }

    std::vector<Data> result(length, 0);
    if (reduction_poly == ReductionPolynomial::X_N_minus)
    {
        for (int i = 0; i < length; i++)
        {
            result[i] =
                VALUE::add(mult_vector[i], mult_vector[i + length], modulus);
        }
    }
    else if (reduction_poly == ReductionPolynomial::X_N_plus)
    {
        for (int i = 0; i < length; i++)
        {
            result[i] =
                VALUE::sub(mult_vector[i], mult_vector[i + length], modulus);
        }
    }
    else
    {
        throw std::runtime_error("Poly reduction type is not supported!");
    }

    return result;
}

NTT_CPU::NTT_CPU(NTTParameters parameters_, std::vector<Data> poly_)
{
    ntt_domain = false;
    parameters = parameters_;
    poly = poly_;
}

void NTT_CPU::mult(NTT_CPU& b)
{
    if (ntt_domain && (b.ntt_domain))
    {
        for (int i = 0; i < parameters.n; i++)
        {
            poly[i] = VALUE::mult(poly[i], b.poly[i], parameters.modulus);
        }
    }
    else
    {
        throw std::runtime_error("Both polynomial should be in NTT domain!");
    }
}
void NTT_CPU::ntt()
{
    // Merged NTT with pre-processing (optimized) (iterative)
    // This is not NTT, this is pre-processing + NTT
    // (see: https://eprint.iacr.org/2016/504.pdf)

    customAssert(ntt_domain == false, "Poly should be in polynomial domain!");
    ntt_domain = true;

    int t = parameters.n;
    int m = 1;

    while (m < parameters.n)
    {
        t = t >> 1;

        for (int i = 0; i < m; i++)
        {
            int j1 = 2 * i * t;
            int j2 = j1 + t - 1;

            int index;
            if (parameters.poly_reduction == ReductionPolynomial::X_N_minus)
            {
                index = bitreverse(i, parameters.logn - 1);
            }
            else
            {  // poly_reduce_type = ReductionPolynomial::X_N_plus
                index = bitreverse(m + i, parameters.logn);
            }

            Data S = parameters.forward_root_of_unity_table[index];

            for (int j = j1; j < (j2 + 1); j++)
            {
                Data U = poly[j];
                Data V = VALUE::mult(poly[j + t], S, parameters.modulus);

                poly[j] = VALUE::add(U, V, parameters.modulus);
                poly[j + t] = VALUE::sub(U, V, parameters.modulus);
            }
        }

        m = m << 1;
    }
}

void NTT_CPU::intt()
{
    // Merged INTT with post-processing (optimized) (iterative)
    // This is not NTT, this is pre-processing + NTT
    // (see: https://eprint.iacr.org/2016/504.pdf)

    customAssert(ntt_domain == true, "Poly should be in NTT domain!");
    ntt_domain = false;

    int t = 1;
    int m = parameters.n;
    while (m > 1)
    {
        int j1 = 0;
        int h = m >> 1;
        for (int i = 0; i < h; i++)
        {
            int j2 = j1 + t - 1;
            int index;
            if (parameters.poly_reduction == ReductionPolynomial::X_N_minus)
            {
                index = bitreverse(i, parameters.logn - 1);
            }
            else
            {  // poly_reduce_type = ReductionPolynomial::X_N_plus
                index = bitreverse(h + i, parameters.logn);
            }

            Data S = parameters.inverse_root_of_unity_table[index];

            for (int j = j1; j < (j2 + 1); j++)
            {
                Data U = poly[j];
                Data V = poly[j + t];

                poly[j] = VALUE::add(U, V, parameters.modulus);
                poly[j + t] = VALUE::sub(U, V, parameters.modulus);
                poly[j + t] = VALUE::mult(poly[j + t], S, parameters.modulus);
            }

            j1 = j1 + (t << 1);
        }

        t = t << 1;
        m = m >> 1;
    }

    Data n_inv = VALUE::modinv(parameters.n, parameters.modulus);

    for (int i = 0; i < parameters.n; i++)
    {
        poly[i] = VALUE::mult(poly[i], n_inv, parameters.modulus);
    }
}

*/