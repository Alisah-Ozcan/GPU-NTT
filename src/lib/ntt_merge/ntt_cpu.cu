// --------------------- //
// Authors: Alisah Ozcan
// --------------------- //

#include "ntt_cpu.cuh"

namespace gpuntt
{

    template <typename T>
    std::vector<T>
    schoolbook_poly_multiplication(std::vector<T> a, std::vector<T> b,
                                   Modulus<T> modulus,
                                   ReductionPolynomial reduction_poly)
    {
        int length = a.size();
        std::vector<T> mult_vector(length * 2, 0);

        for (int i = 0; i < length; i++)
        {
            for (int j = 0; j < length; j++)
            {
                T mult_result = OPERATOR<T>::mult(a[i], b[j], modulus);
                mult_vector[i + j] =
                    OPERATOR<T>::add(mult_vector[i + j], mult_result, modulus);
            }
        }

        std::vector<T> result(length, 0);
        if (reduction_poly == ReductionPolynomial::X_N_minus)
        {
            for (int i = 0; i < length; i++)
            {
                result[i] = OPERATOR<T>::add(mult_vector[i],
                                             mult_vector[i + length], modulus);
            }
        }
        else if (reduction_poly == ReductionPolynomial::X_N_plus)
        {
            for (int i = 0; i < length; i++)
            {
                result[i] = OPERATOR<T>::sub(mult_vector[i],
                                             mult_vector[i + length], modulus);
            }
        }
        else
        {
            throw std::runtime_error("Poly reduction type is not supported!");
        }

        return result;
    }

    template std::vector<Data32> schoolbook_poly_multiplication<Data32>(
        std::vector<Data32> a, std::vector<Data32> b, Modulus<Data32> modulus,
        ReductionPolynomial reduction_poly);

    template std::vector<Data64> schoolbook_poly_multiplication<Data64>(
        std::vector<Data64> a, std::vector<Data64> b, Modulus<Data64> modulus,
        ReductionPolynomial reduction_poly);

    template <typename T> NTTCPU<T>::NTTCPU(NTTParameters<T> parameters_)
    {
        parameters = parameters_;
    }

    template <typename T>
    std::vector<T> NTTCPU<T>::mult(std::vector<T>& input1,
                                   std::vector<T>& input2)
    {
        std::vector<T> output;
        for (int i = 0; i < parameters.n; i++)
        {
            output.push_back(
                OPERATOR<T>::mult(input1[i], input2[i], parameters.modulus));
        }

        return output;
    }

    template <typename T> std::vector<T> NTTCPU<T>::ntt(std::vector<T>& input)
    {
        // Merged NTT with pre-processing (optimized) (iterative)
        // This is not NTT, this is pre-processing + NTT
        // (see: https://eprint.iacr.org/2016/504.pdf)

        std::vector<T> output = input;

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
                { // poly_reduce_type = ReductionPolynomial::X_N_plus
                    index = bitreverse(m + i, parameters.logn);
                }

                T S = parameters.forward_root_of_unity_table[index];

                for (int j = j1; j < (j2 + 1); j++)
                {
                    T U = output[j];
                    T V =
                        OPERATOR<T>::mult(output[j + t], S, parameters.modulus);

                    output[j] = OPERATOR<T>::add(U, V, parameters.modulus);
                    output[j + t] = OPERATOR<T>::sub(U, V, parameters.modulus);
                }
            }

            m = m << 1;
        }

        return output;
    }

    template <typename T> std::vector<T> NTTCPU<T>::intt(std::vector<T>& input)
    {
        // Merged INTT with post-processing (optimized) (iterative)
        // This is not NTT, this is pre-processing + NTT
        // (see: https://eprint.iacr.org/2016/504.pdf)

        std::vector<T> output = input;

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
                { // poly_reduce_type = ReductionPolynomial::X_N_plus
                    index = bitreverse(h + i, parameters.logn);
                }

                T S = parameters.inverse_root_of_unity_table[index];

                for (int j = j1; j < (j2 + 1); j++)
                {
                    T U = output[j];
                    T V = output[j + t];

                    output[j] = OPERATOR<T>::add(U, V, parameters.modulus);
                    output[j + t] = OPERATOR<T>::sub(U, V, parameters.modulus);
                    output[j + t] =
                        OPERATOR<T>::mult(output[j + t], S, parameters.modulus);
                }

                j1 = j1 + (t << 1);
            }

            t = t << 1;
            m = m >> 1;
        }

        T n_inv = OPERATOR<T>::modinv(parameters.n, parameters.modulus);

        for (int i = 0; i < parameters.n; i++)
        {
            output[i] = OPERATOR<T>::mult(output[i], n_inv, parameters.modulus);
        }

        return output;
    }

    template class NTTCPU<Data32>;
    template class NTTCPU<Data64>;

} // namespace gpuntt