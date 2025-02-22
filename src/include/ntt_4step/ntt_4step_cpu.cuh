// Copyright 2024 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef NTT_4STEP_CPU_H
#define NTT_4STEP_CPU_H

#include "nttparameters.cuh"

namespace gpuntt
{
    template <typename T> class NTT_4STEP_CPU
    {
      public:
        NTTParameters4Step<T> parameters;

        NTT_4STEP_CPU(NTTParameters4Step<T> parameters_);

      public:
        std::vector<T> mult(std::vector<T>& input1, std::vector<T>& input2);

        std::vector<T> ntt(std::vector<T>& input);

        std::vector<T> intt(std::vector<T>& input);

      private:
        void core_ntt(std::vector<T>& input, std::vector<T> root_table,
                      int log_size);

        void core_intt(std::vector<T>& input, std::vector<T> root_table,
                       int log_size);

        void product(std::vector<T>& input, std::vector<T> root_table,
                     int log_size);

        std::vector<std::vector<T>>
        vector_to_matrix(const std::vector<T>& array, int rows, int cols);

        std::vector<std::vector<T>>
        vector_to_matrix_intt(const std::vector<T>& array, int rows, int cols);

        std::vector<T>
        matrix_to_vector(const std::vector<std::vector<T>>& originalMatrix);

        std::vector<std::vector<T>>
        transpose_matrix(const std::vector<std::vector<T>>& originalMatrix);

      public:
        std::vector<T> intt_first_transpose(const std::vector<T>& input);
    };

} // namespace gpuntt
#endif // NTT_4STEP_CPU_H
