# GPU-NTT

Welcome to the GPU-NTT-Optimization repository! We present cutting-edge algorithms and implementations for optimizing the 2 diferrent Number Theoretic Transform (NTT) model (Merge & 4-Step) on Graphics Processing Units (GPUs).

The associated research paper: https://eprint.iacr.org/2023/1410

FFT variant of GPU-NTT is available: https://github.com/Alisah-Ozcan/GPU-FFT

## Development

### Requirements

- [CMake](https://cmake.org/download/) >=3.2
- [GCC](https://gcc.gnu.org/)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)

### Testing & Benchmarking

Three different 64 modular reduction supported.(High bit sizes will added with using third-party codes.) They represented as numbers:

- MODULAR_REDUCTION_TYPE=0 -> Barrett Reduction(64 bit)
- MODULAR_REDUCTION_TYPE=1 -> Goldilocks Reduction(64 bit)
- MODULAR_REDUCTION_TYPE=2 -> Plantard Reduction(64 bit)

#### Testing CPU Merge & 4-Step NTT vs Schoolbook Polynomial Multiplication

To build tests:

```bash
$ cmake . -D MODULAR_REDUCTION_TYPE=0 -B./build
$ cmake --build ./build/ --parallel
```

To run tests:

```bash
$ ./build/bin/cpu_4step_ntt <RING_SIZE_IN_LOG2> <BATCH_SIZE>
$ Example: ./build/bin/cpu_4step_ntt 15 1

$ ./build/bin/cpu_merge_ntt <RING_SIZE_IN_LOG2> <BATCH_SIZE>
$ Example: ./build/bin/cpu_merge_ntt 15 1
```

#### Testing GPU NTTs vs CPU NTTs

To build tests:

```bash
$ cmake . -D MODULAR_REDUCTION_TYPE=0 -B./build
$ cmake --build ./build/ --parallel
```

To run tests:

```bash
$ ./build/bin/gpu_4step_ntt_examples <RING_SIZE_IN_LOG2> <BATCH_SIZE>
$ ./build/bin/gpu_4step_intt_examples <RING_SIZE_IN_LOG2> <BATCH_SIZE>
$ ./build/bin/gpu_merge_ntt_examples <RING_SIZE_IN_LOG2> <BATCH_SIZE>
$ ./build/bin/gpu_merge_intt_examples <RING_SIZE_IN_LOG2> <BATCH_SIZE>
```

#### Benchmarking GPU NTT

To build tests:

```bash
$ cmake . -D MODULAR_REDUCTION_TYPE=0 -B./build
$ cmake --build ./build/ --parallel
```

To run tests:

```bash
$ ./build/bin/benchmark_4step_ntt <RING_SIZE_IN_LOG2> <BATCH_SIZE>
$ ./build/bin/benchmark_merge_ntt <RING_SIZE_IN_LOG2> <BATCH_SIZE>
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