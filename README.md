# GPU-NTT

Welcome to the GPU-NTT-Optimization repository! We present cutting-edge algorithms and implementations for optimizing the 2 diferrent Number Theoretic Transform (NTT) model (`Merge` & `4-Step`) on Graphics Processing Units (GPUs).

The associated research paper: https://eprint.iacr.org/2023/1410

FFT variant of GPU-NTT is available: https://github.com/Alisah-Ozcan/GPU-FFT

## Development

### Requirements

- [CMake](https://cmake.org/download/) >=3.2
- [GCC](https://gcc.gnu.org/)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)

### Build & Install

Three different 64 modular reduction supported.(High bit sizes will added with using third-party codes.) They represented as numbers:

- MODULAR_REDUCTION_TYPE=0 -> Barrett Reduction(64 bit)
- MODULAR_REDUCTION_TYPE=1 -> Goldilocks Reduction(64 bit)
- MODULAR_REDUCTION_TYPE=2 -> Plantard Reduction(64 bit)

To build:

```bash
$ cmake . -D CMAKE_CUDA_ARCHITECTURES=86 -D MODULAR_REDUCTION_TYPE=0 -B./build
$ cmake --build ./build/ --parallel
```

To install:

```bash
$ cmake . -D CMAKE_CUDA_ARCHITECTURES=86 -D MODULAR_REDUCTION_TYPE=0 -B./build
$ cmake --build ./build/ --parallel
$ sudo cmake --install build
```

### Testing & Benchmarking

#### Testing CPU Merge & 4-Step NTT vs Schoolbook Polynomial Multiplication

To run examples:

```bash
$ cmake -D CMAKE_CUDA_ARCHITECTURES=86 -D MODULAR_REDUCTION_TYPE=0 -D GPUNTT_BUILD_EXAMPLES=ON -B./build
$ cmake --build ./build/ --parallel

$ ./build/bin/cpu_4step_ntt <RING_SIZE_IN_LOG2> <BATCH_SIZE>
$ ./build/bin/cpu_merge_ntt <RING_SIZE_IN_LOG2> <BATCH_SIZE>
$ Example: ./build/bin/cpu_merge_ntt 15 1
```

#### Testing GPU NTTs vs CPU NTTs

To run examples:

```bash
$ cmake -D CMAKE_CUDA_ARCHITECTURES=86 -D MODULAR_REDUCTION_TYPE=0 -D GPUNTT_BUILD_EXAMPLES=ON -B./build
$ cmake --build ./build/ --parallel

$ ./build/bin/gpu_4step_ntt_examples <RING_SIZE_IN_LOG2> <BATCH_SIZE>
$ ./build/bin/gpu_4step_intt_examples <RING_SIZE_IN_LOG2> <BATCH_SIZE>
$ ./build/bin/gpu_merge_ntt_examples <RING_SIZE_IN_LOG2> <BATCH_SIZE>
$ ./build/bin/gpu_merge_intt_examples <RING_SIZE_IN_LOG2> <BATCH_SIZE>
$ Example: ./build/bin/gpu_merge_ntt_examples 12 1
```

#### Benchmarking GPU NTT

To run benchmarks:

```bash
$ cmake -D CMAKE_CUDA_ARCHITECTURES=86 -D MODULAR_REDUCTION_TYPE=0 -D GPUNTT_BUILD_BENCHMARKS=ON -B./build
$ cmake --build ./build/ --parallel

$ ./build/bin/benchmark_4step_ntt <RING_SIZE_IN_LOG2> <BATCH_SIZE>
$ ./build/bin/benchmark_merge_ntt <RING_SIZE_IN_LOG2> <BATCH_SIZE>
$ Example: ./build/bin/benchmark_merge_ntt 12 1
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
add_compile_definitions(BARRETT_64) # Builded reduction method 
target_compile_definitions(<your-target> PRIVATE BARRETT_64)
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