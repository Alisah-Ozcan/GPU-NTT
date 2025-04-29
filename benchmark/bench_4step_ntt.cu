// Copyright 2024-2025 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include "ntt_4step.cuh"
#include "ntt_4step_cpu.cuh"
#include "bench_util.cuh"

using namespace std;
using namespace gpuntt;

// typedef Data32 BenchmarkDataType; // Use for 32-bit benchmark
typedef Data64 BenchmarkDataType; // Use for 64-bit benchmark

void GPU_4STEP_NTT_Forward_Benchmark(nvbench::state& state)
{
    const auto ring_size_logN = state.get_int64("Ring Size LogN");
    const auto batch_count = state.get_int64("Batch Count");
    const auto ring_size = 1 << ring_size_logN;
    std::vector<int> matrix_shape = bench_matrix_dimention(ring_size_logN);

    thrust::device_vector<BenchmarkDataType> input_data(ring_size *
                                                        batch_count);
    thrust::transform(thrust::counting_iterator<int>(0),
                      thrust::counting_iterator<int>(ring_size * batch_count),
                      input_data.begin(),
                      random_functor<BenchmarkDataType>(1234));

    thrust::device_vector<BenchmarkDataType> output_data(ring_size *
                                                         batch_count);
    thrust::transform(thrust::counting_iterator<int>(0),
                      thrust::counting_iterator<int>(ring_size * batch_count),
                      output_data.begin(),
                      random_functor<BenchmarkDataType>(1234));

    thrust::device_vector<Root<BenchmarkDataType>> W_root_of_unity_table(
        ring_size);
    thrust::transform(thrust::counting_iterator<int>(0),
                      thrust::counting_iterator<int>((ring_size)),
                      W_root_of_unity_table.begin(),
                      random_functor<Root<BenchmarkDataType>>(1234));

    thrust::device_vector<Root<BenchmarkDataType>> n1_root_of_unity_table(
        matrix_shape[0] >> 1);
    thrust::transform(thrust::counting_iterator<int>(0),
                      thrust::counting_iterator<int>((matrix_shape[0] >> 1)),
                      n1_root_of_unity_table.begin(),
                      random_functor<Root<BenchmarkDataType>>(1234));

    thrust::device_vector<Root<BenchmarkDataType>> n2_root_of_unity_table(
        matrix_shape[1] >> 1);
    thrust::transform(thrust::counting_iterator<int>(0),
                      thrust::counting_iterator<int>((matrix_shape[1] >> 1)),
                      n2_root_of_unity_table.begin(),
                      random_functor<Root<BenchmarkDataType>>(1234));

    state.add_global_memory_reads<BenchmarkDataType>(
        (ring_size * batch_count) + ((ring_size >> 1) * batch_count),
        "Read Memory Size");
    state.add_global_memory_writes<BenchmarkDataType>(ring_size * batch_count,
                                                      "Write Memory Size");
    state.collect_l1_hit_rates();
    state.collect_l2_hit_rates();
    // state.collect_loads_efficiency();
    // state.collect_stores_efficiency();
    // state.collect_dram_throughput();

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    state.set_cuda_stream(nvbench::make_cuda_stream_view(stream));

    ntt4step_configuration<BenchmarkDataType> cfg_ntt = {
        .n_power = static_cast<int>(ring_size_logN),
        .ntt_type = FORWARD,
        .stream = stream};

    Modulus<BenchmarkDataType> mod_data(10000ULL);

    state.exec(
        [&](nvbench::launch& launch)
        {
            GPU_4STEP_NTT(
                thrust::raw_pointer_cast(input_data.data()),
                thrust::raw_pointer_cast(output_data.data()),
                thrust::raw_pointer_cast(n2_root_of_unity_table.data()),
                thrust::raw_pointer_cast(n1_root_of_unity_table.data()),
                thrust::raw_pointer_cast(W_root_of_unity_table.data()),
                mod_data, cfg_ntt, static_cast<int>(batch_count));
        });

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
}

NVBENCH_BENCH(GPU_4STEP_NTT_Forward_Benchmark)
    .add_int64_axis("Ring Size LogN",
                    {12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24})
    .add_int64_axis("Batch Count", {1})
    .set_timeout(1);

void GPU_4STEP_NTT_Inverse_Benchmark(nvbench::state& state)
{
    const auto ring_size_logN = state.get_int64("Ring Size LogN");
    const auto batch_count = state.get_int64("Batch Count");
    const auto ring_size = 1 << ring_size_logN;
    std::vector<int> matrix_shape = bench_matrix_dimention(ring_size_logN);

    thrust::device_vector<BenchmarkDataType> input_data(ring_size *
                                                        batch_count);
    thrust::transform(thrust::counting_iterator<int>(0),
                      thrust::counting_iterator<int>(ring_size * batch_count),
                      input_data.begin(),
                      random_functor<BenchmarkDataType>(1234));

    thrust::device_vector<BenchmarkDataType> output_data(ring_size *
                                                         batch_count);
    thrust::transform(thrust::counting_iterator<int>(0),
                      thrust::counting_iterator<int>(ring_size * batch_count),
                      output_data.begin(),
                      random_functor<BenchmarkDataType>(1234));

    thrust::device_vector<Root<BenchmarkDataType>> W_root_of_unity_table(
        ring_size);
    thrust::transform(thrust::counting_iterator<int>(0),
                      thrust::counting_iterator<int>((ring_size)),
                      W_root_of_unity_table.begin(),
                      random_functor<Root<BenchmarkDataType>>(1234));

    thrust::device_vector<Root<BenchmarkDataType>> n1_root_of_unity_table(
        matrix_shape[0] >> 1);
    thrust::transform(thrust::counting_iterator<int>(0),
                      thrust::counting_iterator<int>((matrix_shape[0] >> 1)),
                      n1_root_of_unity_table.begin(),
                      random_functor<Root<BenchmarkDataType>>(1234));

    thrust::device_vector<Root<BenchmarkDataType>> n2_root_of_unity_table(
        matrix_shape[1] >> 1);
    thrust::transform(thrust::counting_iterator<int>(0),
                      thrust::counting_iterator<int>((matrix_shape[1] >> 1)),
                      n2_root_of_unity_table.begin(),
                      random_functor<Root<BenchmarkDataType>>(1234));

    state.add_global_memory_reads<BenchmarkDataType>(
        (ring_size * batch_count) + ((ring_size >> 1) * batch_count),
        "Read Memory Size");
    state.add_global_memory_writes<BenchmarkDataType>(ring_size * batch_count,
                                                      "Write Memory Size");
    state.collect_l1_hit_rates();
    state.collect_l2_hit_rates();
    // state.collect_loads_efficiency();
    // state.collect_stores_efficiency();
    // state.collect_dram_throughput();

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    state.set_cuda_stream(nvbench::make_cuda_stream_view(stream));

    Ninverse<BenchmarkDataType> inv_mod_data(20000ULL);

    ntt4step_configuration<BenchmarkDataType> cfg_ntt = {
        .n_power = static_cast<int>(ring_size_logN),
        .ntt_type = INVERSE,
        .mod_inverse = inv_mod_data,
        .stream = stream};

    Modulus<BenchmarkDataType> mod_data(10000ULL);

    state.exec(
        [&](nvbench::launch& launch)
        {
            GPU_4STEP_NTT(
                thrust::raw_pointer_cast(input_data.data()),
                thrust::raw_pointer_cast(output_data.data()),
                thrust::raw_pointer_cast(n2_root_of_unity_table.data()),
                thrust::raw_pointer_cast(n1_root_of_unity_table.data()),
                thrust::raw_pointer_cast(W_root_of_unity_table.data()),
                mod_data, cfg_ntt, static_cast<int>(batch_count));
        });

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
}

NVBENCH_BENCH(GPU_4STEP_NTT_Inverse_Benchmark)
    .add_int64_axis("Ring Size LogN",
                    {12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24})
    .add_int64_axis("Batch Count", {1})
    .set_timeout(1);