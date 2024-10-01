// --------------------- //
// Authors: Alisah Ozcan
// --------------------- //

#include "ntt_4step_cpu.cuh"

NTT_4STEP_CPU::NTT_4STEP_CPU(NTTParameters4Step parameters_)
{
    parameters = parameters_;
}

std::vector<Data> NTT_4STEP_CPU::mult(std::vector<Data>& input1,
                                      std::vector<Data>& input2)
{
    std::vector<Data> output;
    for (int i = 0; i < parameters.n; i++)
    {
        output.push_back(VALUE::mult(input1[i], input2[i], parameters.modulus));
    }

    return output;
}

/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////

std::vector<Data> NTT_4STEP_CPU::ntt(std::vector<Data>& input)
{
    std::vector<std::vector<Data>> matrix =
        vector_to_matrix(input, parameters.n1, parameters.n2);
    std::vector<std::vector<Data>> transposed_matrix = transpose_matrix(matrix);

    for (int i = 0; i < parameters.n2; i++)
    {
        core_ntt(transposed_matrix[i], parameters.n1_based_root_of_unity_table,
                 int(log2(parameters.n1)));
    }

    std::vector<std::vector<Data>> transposed_matrix2 =
        transpose_matrix(transposed_matrix);
    std::vector<Data> vector_ = matrix_to_vector(transposed_matrix2);

    product(vector_, parameters.W_root_of_unity_table, parameters.logn);

    std::vector<std::vector<Data>> transposed_matrix3 =
        vector_to_matrix(vector_, parameters.n1, parameters.n2);

    for (int i = 0; i < parameters.n1; i++)
    {
        core_ntt(transposed_matrix3[i], parameters.n2_based_root_of_unity_table,
                 int(log2(parameters.n2)));
    }

    transposed_matrix2 = transpose_matrix(transposed_matrix3);
    std::vector<Data> result = matrix_to_vector(transposed_matrix2);

    return result;
}

std::vector<Data> NTT_4STEP_CPU::intt(std::vector<Data>& input)
{
    std::vector<std::vector<Data>> transposed_matrix =
        vector_to_matrix_intt(input, parameters.n1, parameters.n2);

    for (int i = 0; i < parameters.n2; i++)
    {
        core_intt(transposed_matrix[i],
                  parameters.n1_based_inverse_root_of_unity_table,
                  int(log2(parameters.n1)));
    }

    std::vector<std::vector<Data>> transposed_matrix2 =
        transpose_matrix(transposed_matrix);
    std::vector<Data> vector_ = matrix_to_vector(transposed_matrix2);

    product(vector_, parameters.W_inverse_root_of_unity_table, parameters.logn);

    std::vector<std::vector<Data>> transposed_matrix3 =
        vector_to_matrix(vector_, parameters.n1, parameters.n2);

    for (int i = 0; i < parameters.n1; i++)
    {
        core_intt(transposed_matrix3[i],
                  parameters.n2_based_inverse_root_of_unity_table,
                  int(log2(parameters.n2)));
    }

    transposed_matrix2 = transpose_matrix(transposed_matrix3);

    std::vector<Data> result = matrix_to_vector(transposed_matrix2);

    for (int i = 0; i < parameters.n; i++)
    {
        result[i] =
            VALUE::mult(result[i], parameters.n_inv, parameters.modulus);
    }

    return result;
}

/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////

void NTT_4STEP_CPU::core_ntt(std::vector<Data>& input,
                             std::vector<Data> root_table, int log_size)
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

            Data S = root_table[index];

            for (int j = j1; j < (j2 + 1); j++)
            {
                Data U = input[j];
                Data V = VALUE::mult(input[j + t], S, parameters.modulus);

                input[j] = VALUE::add(U, V, parameters.modulus);
                input[j + t] = VALUE::sub(U, V, parameters.modulus);
            }
        }

        m = m << 1;
    }
}

void NTT_4STEP_CPU::core_intt(std::vector<Data>& input,
                              std::vector<Data> root_table, int log_size)
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

            Data S = root_table[index];

            for (int j = j1; j < (j2 + 1); j++)
            {
                Data U = input[j];
                Data V = input[j + t];

                input[j] = VALUE::add(U, V, parameters.modulus);
                input[j + t] = VALUE::sub(U, V, parameters.modulus);
                input[j + t] = VALUE::mult(input[j + t], S, parameters.modulus);
            }

            j1 = j1 + (t << 1);
        }

        t = t << 1;
        m = m >> 1;
    }
}

/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////

void NTT_4STEP_CPU::product(std::vector<Data>& input,
                            std::vector<Data> root_table, int log_size)
{
    int n_ = 1 << log_size;
    for (int i = 0; i < n_; i++)
    {
        input[i] = VALUE::mult(input[i], root_table[i], parameters.modulus);
    }
}

std::vector<std::vector<Data>>
NTT_4STEP_CPU::vector_to_matrix(const std::vector<Data>& array, int rows,
                                int cols)
{
    std::vector<std::vector<Data>> matrix(rows, std::vector<Data>(cols));

    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            matrix[i][j] = array[(i * cols) + j];
        }
    }

    return matrix;
}

std::vector<std::vector<Data>>
NTT_4STEP_CPU::vector_to_matrix_intt(const std::vector<Data>& array, int rows,
                                     int cols)
{
    std::vector<std::vector<Data>> matrix(cols);

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

std::vector<Data> NTT_4STEP_CPU::matrix_to_vector(
    const std::vector<std::vector<Data>>& originalMatrix)
{
    int rows = originalMatrix.size();
    int cols = originalMatrix[0].size();

    std::vector<Data> result;

    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            result.push_back(originalMatrix[i][j]);
        }
    }

    return result;
}

std::vector<std::vector<Data>> NTT_4STEP_CPU::transpose_matrix(
    const std::vector<std::vector<Data>>& originalMatrix)
{
    int rows = originalMatrix.size();
    int cols = originalMatrix[0].size();

    std::vector<std::vector<Data>> transpose(cols, std::vector<Data>(rows));

    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            transpose[j][i] = originalMatrix[i][j];
        }
    }

    return transpose;
}

std::vector<Data>
NTT_4STEP_CPU::intt_first_transpose(const std::vector<Data>& input)
{
    std::vector<std::vector<Data>> transposed_matrix =
        vector_to_matrix_intt(input, parameters.n1, parameters.n2);

    std::vector<Data> result = matrix_to_vector(transposed_matrix);

    return result;
}