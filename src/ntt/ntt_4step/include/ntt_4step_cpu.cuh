// --------------------- //
// Authors: Alisah Ozcan
// --------------------- //

#ifndef NTT_4STEP_CPU_H
#define NTT_4STEP_CPU_H

#include "nttparameters.cuh"

class NTT_4STEP_CPU
{
   public:
    NTTParameters4Step parameters;

    NTT_4STEP_CPU(NTTParameters4Step parameters_);

   public:
    std::vector<Data> mult(std::vector<Data>& input1, std::vector<Data>& input2);

    std::vector<Data> ntt(std::vector<Data>& input);

    std::vector<Data> intt(std::vector<Data>& input);

   private:
    void core_ntt(std::vector<Data>& input, std::vector<Data> root_table, int log_size);

    void core_intt(std::vector<Data>& input, std::vector<Data> root_table, int log_size);

    void product(std::vector<Data>& input, std::vector<Data> root_table, int log_size);

    std::vector<std::vector<Data>> vector_to_matrix(const std::vector<Data>& array, int rows,
                                                    int cols);

    std::vector<std::vector<Data>> vector_to_matrix_intt(const std::vector<Data>& array, int rows,
                                                         int cols);

    std::vector<Data> matrix_to_vector(const std::vector<std::vector<Data>>& originalMatrix);

    std::vector<std::vector<Data>> transpose_matrix(
        const std::vector<std::vector<Data>>& originalMatrix);

   public:
    std::vector<Data> intt_first_transpose(const std::vector<Data>& input);
};

#endif  // NTT_4STEP_CPU_H
