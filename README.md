# GPU-NTT

Welcome to the GPU-NTT-Optimization repository! We present cutting-edge algorithms and implementations for optimizing the 2 diferrent Number Theoretic Transform (NTT) model (`Merge` & `4-Step`) on Graphics Processing Units (GPUs).

The associated research paper: https://eprint.iacr.org/2023/1410

FFT variant of GPU-NTT is available: https://github.com/Alisah-Ozcan/GPU-FFT

You no longer need to manually select a reduction method, as this version is now available in the [paper_version](https://github.com/Alisah-Ozcan/GPU-NTT/tree/paper_version)
 branch. GPU-NTT now **automatically supports both 32-bit and 64-bit operations using Barrett Reduction**, and various optimizations will be introduced over time.

## Development

### Requirements

- [CMake](https://cmake.org/download/) >=3.2
- [GCC](https://gcc.gnu.org/)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)

### Build & Install

To build:

```bash
$ cmake . -D CMAKE_CUDA_ARCHITECTURES=86 -B./build
$ cmake --build ./build/ --parallel
```

To install:

```bash
$ cmake . -D CMAKE_CUDA_ARCHITECTURES=86 -B./build
$ cmake --build ./build/ --parallel
$ sudo cmake --install build
```

### Testing & Benchmarking

#### Testing CPU Merge & 4-Step NTT vs Schoolbook Polynomial Multiplication

Choose one of data type which is upper line of the example files:
- typedef Data32 TestDataType;
- typedef Data64 TestDataType;

To run examples:

```bash
$ cmake -D CMAKE_CUDA_ARCHITECTURES=86 -D GPUNTT_BUILD_EXAMPLES=ON -B./build
$ cmake --build ./build/ --parallel

$ ./build/bin/example/cpu_4step_ntt_examples <RING_SIZE_IN_LOG2> <BATCH_SIZE>
$ ./build/bin/example/cpu_merge_ntt_examples <RING_SIZE_IN_LOG2> <BATCH_SIZE>
$ Example: ./build/bin/example/cpu_merge_ntt_examples 15 1
```

#### Testing GPU NTTs vs CPU NTTs

Choose one of data type which is upper line of the example files:
- typedef Data32 TestDataType;
- typedef Data64 TestDataType;

To run examples:

```bash
$ cmake -D CMAKE_CUDA_ARCHITECTURES=86 -D GPUNTT_BUILD_EXAMPLES=ON -B./build
$ cmake --build ./build/ --parallel

$ ./build/bin/example/gpu_4step_ntt_examples <RING_SIZE_IN_LOG2> <BATCH_SIZE>
$ ./build/bin/example/gpu_4step_intt_examples <RING_SIZE_IN_LOG2> <BATCH_SIZE>
$ ./build/bin/example/gpu_merge_ntt_examples <RING_SIZE_IN_LOG2> <BATCH_SIZE>
$ ./build/bin/example/gpu_merge_intt_examples <RING_SIZE_IN_LOG2> <BATCH_SIZE>
$ Example: ./build/bin/example/gpu_merge_ntt_examples 12 1
```

#### Benchmarking GPU NTT

Choose one of data type which is upper line of the benchmark files:
- typedef Data32 BenchmarkDataType;
- typedef Data64 BenchmarkDataType;

To run benchmarks:

```bash
$ cmake -D CMAKE_CUDA_ARCHITECTURES=86 -D GPUNTT_BUILD_BENCHMARKS=ON -B./build
$ cmake --build ./build/ --parallel

$ ./build/bin/benchmark/benchmark_4step_ntt --disable-blocking-kernel
$ ./build/bin/benchmark/benchmark_merge_ntt --disable-blocking-kernel
```

## Using GPU-NTT in a downstream CMake project

Make sure GPU-NTT is installed before integrating it into your project. The installed GPU-NTT library provides a set of config files that make it easy to integrate GPU-NTT into your own CMake project. In your CMakeLists.txt, simply add:

```cmake
project(<your-project> LANGUAGES CXX CUDA)
find_package(CUDAToolkit REQUIRED)
# ...
find_package(GPUNTT)
# ...
target_link_libraries(<your-target> (PRIVATE|PUBLIC|INTERFACE) GPUNTT::ntt CUDA::cudart)
# ...
set_target_properties(<your-target> PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# ...
```

## How to Cite GPU-NTT

Please use the below BibTeX, to cite GPU-NTT in academic papers.

```
@misc{cryptoeprint:2023/1410,
      author = {Ali Şah Özcan and Erkay Savaş},
      title = {Two Algorithms for Fast GPU Implementation of NTT},
      howpublished = {Cryptology ePrint Archive, Paper 2023/1410},
      year = {2023},
      note = {\url{https://eprint.iacr.org/2023/1410}},
      url = {https://eprint.iacr.org/2023/1410}
}
```

## License
This project is licensed under the [Apache License](LICENSE). For more details, please refer to the License file.

## Contact
If you have any questions or feedback, feel free to contact me: 
- Email: alisah@sabanciuniv.edu
- LinkedIn: [Profile](https://www.linkedin.com/in/ali%C5%9Fah-%C3%B6zcan-472382305/)