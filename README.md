# GPU-NTT

Welcome to the GPU-NTT-Optimization repository! We present cutting-edge algorithms and implementations for optimizing the Number Theoretic Transform (NTT) on Graphics Processing Units (GPUs).

The associated research paper: https://eprint.iacr.org/2023/1410

## Development

### Requirements

- [CMake](https://cmake.org/download/) >=3.2
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)

### Testing & Benchmarking

Three different 64 modular reduction supported.(High bit sizes will added with using third-party codes.) They represented as numbers:

- MODULAR_REDUCTION_TYPE=0 -> Barrett Reduction(64 bit)
- MODULAR_REDUCTION_TYPE=1 -> Goldilocks Reduction(64 bit)
- MODULAR_REDUCTION_TYPE=2 -> Plantard Reduction(64 bit)

#### Testing CPU NTT vs Schoolbook Polynomial Multiplication

To build tests:

```bash
$ cmake . -D CMAKE_CUDA_ARCHITECTURES=86 -D MODULAR_REDUCTION_TYPE=0 -B./cmake-build 
$ cmake --build ./cmake-build/ --target ntt_cpu_test --parallel
```

To run tests:

```bash
$ ./cmake-build/ntt_cpu_test <RING_SIZE_IN_LOG2> <BATCH_SIZE>
$ Example: ./cmake-build/ntt_cpu_test 12 1
```

#### Testing GPU NTT vs CPU NTT

To build tests:

```bash
$ cmake . -D CMAKE_CUDA_ARCHITECTURES=86 -D MODULAR_REDUCTION_TYPE=0 -B./cmake-build 
$ cmake --build ./cmake-build/ --target test_ntt_gpu --parallel
```

To run tests:

```bash
$ ./cmake-build/test_ntt_gpu <RING_SIZE_IN_LOG2> <BATCH_SIZE>
```

#### Benchmarking GPU NTT

To build tests:

```bash
$ cmake . -D CMAKE_CUDA_ARCHITECTURES=86 -D MODULAR_REDUCTION_TYPE=0 -B./cmake-build 
$ cmake --build ./cmake-build/ --target ntt_gpu_bench --parallel
```

To run tests:

```bash
$ ./cmake-build/ntt_gpu_bench <RING_SIZE_IN_LOG2> <BATCH_SIZE>
```