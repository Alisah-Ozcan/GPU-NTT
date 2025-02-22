// --------------------- //
// Authors: Alisah Ozcan
// --------------------- //

#include "ntt_4step_cpu.cuh"

namespace gpuntt
{

    template <typename T>
    NTT_4STEP_CPU<T>::NTT_4STEP_CPU(NTTParameters4Step<T> parameters_)
    {
        parameters = parameters_;
    }

    template <typename T>
    std::vector<T> NTT_4STEP_CPU<T>::mult(std::vector<T>& input1,
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

    /////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////

    template <typename T>
    std::vector<T> NTT_4STEP_CPU<T>::ntt(std::vector<T>& input)
    {
        std::vector<std::vector<T>> matrix =
            vector_to_matrix(input, parameters.n1, parameters.n2);
        std::vector<std::vector<T>> transposed_matrix =
            transpose_matrix(matrix);

        for (int i = 0; i < parameters.n2; i++)
        {
            core_ntt(transposed_matrix[i],
                     parameters.n1_based_root_of_unity_table,
                     int(log2(parameters.n1)));
        }

        std::vector<std::vector<T>> transposed_matrix2 =
            transpose_matrix(transposed_matrix);
        std::vector<T> vector_ = matrix_to_vector(transposed_matrix2);

        product(vector_, parameters.W_root_of_unity_table, parameters.logn);

        std::vector<std::vector<T>> transposed_matrix3 =
            vector_to_matrix(vector_, parameters.n1, parameters.n2);

        for (int i = 0; i < parameters.n1; i++)
        {
            core_ntt(transposed_matrix3[i],
                     parameters.n2_based_root_of_unity_table,
                     int(log2(parameters.n2)));
        }

        transposed_matrix2 = transpose_matrix(transposed_matrix3);
        std::vector<T> result = matrix_to_vector(transposed_matrix2);

        return result;
    }

    template <typename T>
    std::vector<T> NTT_4STEP_CPU<T>::intt(std::vector<T>& input)
    {
        std::vector<std::vector<T>> transposed_matrix =
            vector_to_matrix_intt(input, parameters.n1, parameters.n2);

        for (int i = 0; i < parameters.n2; i++)
        {
            core_intt(transposed_matrix[i],
                      parameters.n1_based_inverse_root_of_unity_table,
                      int(log2(parameters.n1)));
        }

        std::vector<std::vector<T>> transposed_matrix2 =
            transpose_matrix(transposed_matrix);
        std::vector<T> vector_ = matrix_to_vector(transposed_matrix2);

        product(vector_, parameters.W_inverse_root_of_unity_table,
                parameters.logn);

        std::vector<std::vector<T>> transposed_matrix3 =
            vector_to_matrix(vector_, parameters.n1, parameters.n2);

        for (int i = 0; i < parameters.n1; i++)
        {
            core_intt(transposed_matrix3[i],
                      parameters.n2_based_inverse_root_of_unity_table,
                      int(log2(parameters.n2)));
        }

        transposed_matrix2 = transpose_matrix(transposed_matrix3);

        std::vector<T> result = matrix_to_vector(transposed_matrix2);

        for (int i = 0; i < parameters.n; i++)
        {
            result[i] = OPERATOR<T>::mult(result[i], parameters.n_inv,
                                          parameters.modulus);
        }

        return result;
    }

    /////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////

    template <typename T>
    void NTT_4STEP_CPU<T>::core_ntt(std::vector<T>& input,
                                    std::vector<T> root_table, int log_size)
    {
        // Merged NTT with pre-processing (optimized) (iterative)
        // This is not NTT, this is pre-processing + NTT
        // (see: https://eprint.iacr.org/2016/504.pdf)

        int n_ = 1 << log_size;
        int t = n_;
        int m = 1;

        while (m < n_)
        {
            t = t >> 1;

            for (int i = 0; i < m; i++)
            {
                int j1 = 2 * i * t;
                int j2 = j1 + t - 1;

                int index = bitreverse(i, log_size - 1);

                T S = root_table[index];

                for (int j = j1; j < (j2 + 1); j++)
                {
                    T U = input[j];
                    T V =
                        OPERATOR<T>::mult(input[j + t], S, parameters.modulus);

                    input[j] = OPERATOR<T>::add(U, V, parameters.modulus);
                    input[j + t] = OPERATOR<T>::sub(U, V, parameters.modulus);
                }
            }

            m = m << 1;
        }
    }
    template <typename T>
    void NTT_4STEP_CPU<T>::core_intt(std::vector<T>& input,
                                     std::vector<T> root_table, int log_size)
    {
        // Merged INTT with post-processing (optimized) (iterative)
        // This is not NTT, this is pre-processing + NTT
        // (see: https://eprint.iacr.org/2016/504.pdf)

        int n_ = 1 << log_size;
        int t = 1;
        int m = n_;

        while (m > 1)
        {
            int j1 = 0;
            int h = m >> 1;
            for (int i = 0; i < h; i++)
            {
                int j2 = j1 + t - 1;

                int index = bitreverse(i, log_size - 1);

                T S = root_table[index];

                for (int j = j1; j < (j2 + 1); j++)
                {
                    T U = input[j];
                    T V = input[j + t];

                    input[j] = OPERATOR<T>::add(U, V, parameters.modulus);
                    input[j + t] = OPERATOR<T>::sub(U, V, parameters.modulus);
                    input[j + t] =
                        OPERATOR<T>::mult(input[j + t], S, parameters.modulus);
                }

                j1 = j1 + (t << 1);
            }

            t = t << 1;
            m = m >> 1;
        }
    }

    /////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////
    template <typename T>
    void NTT_4STEP_CPU<T>::product(std::vector<T>& input,
                                   std::vector<T> root_table, int log_size)
    {
        int n_ = 1 << log_size;
        for (int i = 0; i < n_; i++)
        {
            input[i] =
                OPERATOR<T>::mult(input[i], root_table[i], parameters.modulus);
        }
    }

    template <typename T>
    std::vector<std::vector<T>>
    NTT_4STEP_CPU<T>::vector_to_matrix(const std::vector<T>& array, int rows,
                                       int cols)
    {
        std::vector<std::vector<T>> matrix(rows, std::vector<T>(cols));

        for (int i = 0; i < rows; ++i)
        {
            for (int j = 0; j < cols; ++j)
            {
                matrix[i][j] = array[(i * cols) + j];
            }
        }

        return matrix;
    }

    template <typename T>
    std::vector<std::vector<T>>
    NTT_4STEP_CPU<T>::vector_to_matrix_intt(const std::vector<T>& array,
                                            int rows, int cols)
    {
        std::vector<std::vector<T>> matrix(cols);

        for (int i = 0; i < rows; ++i)
        {
            for (int j = 0; j < cols; ++j)
            {
                matrix[int(((i * cols) + j) / rows)].push_back(
                    array[i + (j * rows)]);
            }
        }

        return matrix;
    }

    template <typename T>
    std::vector<T> NTT_4STEP_CPU<T>::matrix_to_vector(
        const std::vector<std::vector<T>>& originalMatrix)
    {
        int rows = originalMatrix.size();
        int cols = originalMatrix[0].size();

        std::vector<T> result;

        for (int i = 0; i < rows; ++i)
        {
            for (int j = 0; j < cols; ++j)
            {
                result.push_back(originalMatrix[i][j]);
            }
        }

        return result;
    }

    template <typename T>
    std::vector<std::vector<T>> NTT_4STEP_CPU<T>::transpose_matrix(
        const std::vector<std::vector<T>>& originalMatrix)
    {
        int rows = originalMatrix.size();
        int cols = originalMatrix[0].size();

        std::vector<std::vector<T>> transpose(cols, std::vector<T>(rows));

        for (int i = 0; i < rows; ++i)
        {
            for (int j = 0; j < cols; ++j)
            {
                transpose[j][i] = originalMatrix[i][j];
            }
        }

        return transpose;
    }

    template <typename T>
    std::vector<T>
    NTT_4STEP_CPU<T>::intt_first_transpose(const std::vector<T>& input)
    {
        std::vector<std::vector<T>> transposed_matrix =
            vector_to_matrix_intt(input, parameters.n1, parameters.n2);

        std::vector<T> result = matrix_to_vector(transposed_matrix);

        return result;
    }

    template class NTT_4STEP_CPU<Data32>;
    template class NTT_4STEP_CPU<Data64>;

} // namespace gpuntt