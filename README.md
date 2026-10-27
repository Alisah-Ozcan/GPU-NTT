# GPU-NTT
Welcome to the GPU-NTT-Optimization repository! We present cutting-edge algorithms and implementations for optimizing the Merge and [4-Step](https://github.com/Alisah-Ozcan/GPU-NTT/tree/v.0.0.0) Number Theoretic Transforms (NTT) on GPUs. GPU-NTT automatically supports both 32-bit and 64-bit arithmetic via Barrett reduction (moduli up to 30 bits for 32-bit data and up to 61 bits for 64-bit data; larger moduli are rejected with `std::invalid_argument`); the older manual reduction selection is kept in the [paper_version](https://github.com/Alisah-Ozcan/GPU-NTT/tree/paper_version) branch.

The associated research paper: https://eprint.iacr.org/2023/1410

FFT variant of GPU-NTT is available: https://github.com/Alisah-Ozcan/GPU-FFT

## Development

### Requirements

- [CMake](https://cmake.org/download/) >=3.26
- [GCC](https://gcc.gnu.org/)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)

### Build & Install

Configure + build:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build --parallel
```

Install:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build --parallel
cmake --install build
```

Notes:
- If you install to a system location (default: `/usr/local`), you may need `sudo` or set `-DCMAKE_INSTALL_PREFIX=/your/prefix`.
- If you omit `-DCMAKE_CUDA_ARCHITECTURES=...`, GPU-NTT defaults to `80;86;89;90`.
- If CMake cannot find `nvcc`, set `CUDACXX=/path/to/nvcc` or pass `-DCMAKE_CUDA_COMPILER=/path/to/nvcc`.
- If you change compilers/toolchains, prefer a clean configure: `cmake --fresh -S . -B build`.

### Compile Options

GPU-NTT uses C++17/CUDA17 and applies per-configuration compile flags.

- Build type (single-config generators like Makefiles/Ninja): `-DCMAKE_BUILD_TYPE=Release|Debug|RelWithDebInfo|MinSizeRel`
- Optimization/debug defaults:
  - `Release`: `-O3 -DNDEBUG`
  - `RelWithDebInfo`: `-O3 -g -DNDEBUG`
  - `Debug`: `-g` (and CUDA adds line info)
- Extra warnings: `-DGPUNTT_ENABLE_WARNINGS=ON`

Example:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DGPUNTT_ENABLE_WARNINGS=ON
cmake --build build --parallel
```

### Testing & Benchmarking

#### Examples (CPU & GPU Merge)

Choose one of the data types at the top of the example files:
- typedef Data32 TestDataType;
- typedef Data64 TestDataType;

Configure + build:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86 -DGPUNTT_BUILD_EXAMPLES=ON
cmake --build build --parallel
```

Run CPU examples:

```bash
./build/bin/example/cpu_merge_ntt_examples <RING_SIZE_IN_LOG2> <BATCH_SIZE>
# Example: ./build/bin/example/cpu_merge_ntt_examples 15 1
```

Run GPU examples:

```bash
./build/bin/example/gpu_merge_ntt_examples <RING_SIZE_IN_LOG2> <BATCH_SIZE>
./build/bin/example/gpu_merge_intt_examples <RING_SIZE_IN_LOG2> <BATCH_SIZE>
# Example: ./build/bin/example/gpu_merge_ntt_examples 12 1
```

Run the device API example (see [Device API](#device-api)):

```bash
./build/bin/example/gpu_device_api_examples <BATCH_SIZE>
```

#### Tests

The tests compare every merge NTT host API variant (single modulus, RNS, signed input/output, column-wise, modulus/poly ordered; in-place and out-of-place) against the CPU reference for 32-bit and 64-bit data.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86 -DGPUNTT_BUILD_TESTS=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

- `-DGPUNTT_TEST_MAX_LOGN=<N>` sets the largest ring size covered (default `22`, up to `28` for 64-bit and `22` for 32-bit).
- The `large` suite runs every variant at N = 2^25, where the kernels switch to a different grid mapping, independently of `GPUNTT_TEST_MAX_LOGN`. The CPU reference is too slow at this size, so it checks forward/inverse round trips and compares in-place with out-of-place results. It needs about 3 GB of GPU memory.
- A single suite can be run directly: `./build/bin/test/merge_ntt_tests --bits 64 --suite rns --max-logn 16`

#### Benchmarking GPU NTT

Benchmarks use [nvbench](https://github.com/NVIDIA/nvbench), which requires CMake >= 3.30.4. Every benchmark runs for both 32-bit and 64-bit data:
- `merge_ntt`: single modulus NTT/INTT for `LogN` 1–28, one polynomial (`single`) or about 2^22 coefficients (`bulk`; one polynomial from `LogN` 22), X^N-1 and X^N+1. Needs about 4 GB of GPU memory at `LogN` 28.
- `merge_ntt_variants`: RNS, signed input/output and modulus/poly ordered NTT, `LogN` up to 26.
- `merge_ntt_per_coefficient`: column-wise NTT/INTT with single modulus and RNS.

Configure + build:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86 -DGPUNTT_BUILD_BENCHMARKS=ON
cmake --build build --parallel
```

Run benchmarks:

```bash
./build/bin/benchmark/benchmark_merge_ntt --disable-blocking-kernel
# Only one benchmark / configuration:
./build/bin/benchmark/benchmark_merge_ntt --disable-blocking-kernel -b merge_ntt -a Data=U64 -a LogN=16
```

Compare two runs (e.g. before and after a change):

```bash
./build/bin/benchmark/benchmark_merge_ntt --disable-blocking-kernel --json before.json
# ... change and rebuild ...
./build/bin/benchmark/benchmark_merge_ntt --disable-blocking-kernel --json after.json
python3 benchmark/compare_results.py before.json after.json --threshold 3
```

`compare_results.py` has no dependencies; it lists the configurations whose GPU time changed by more than the threshold and the measured noise, and exits with status 1 if any configuration got slower. The `single` workload (one polynomial) takes only a few microseconds and is dominated by launch latency, so its run-to-run noise is high; use `--skip Workload=single` to compare only the stable workloads. nvbench's own `build/_deps/nvbench-src/scripts/nvbench_compare.py` can be used as well (requires `pip install tabulate colorama`).

## Using GPU-NTT in a downstream CMake project

Make sure GPU-NTT is installed before integrating it into your project. The installed GPU-NTT library provides a set of config files that make it easy to integrate GPU-NTT into your own CMake project. In your CMakeLists.txt, simply add:

```cmake
project(<your-project> LANGUAGES CXX CUDA)
find_package(CUDAToolkit REQUIRED)
# ...
find_package(GPUNTT CONFIG REQUIRED)
# ...
target_link_libraries(<your-target> (PRIVATE|PUBLIC|INTERFACE) GPUNTT::ntt CUDA::cudart)
# ...
set_target_properties(<your-target> PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# ...
```

All GPU-NTT types and functions are in the `gpuntt` namespace. For compatibility, the arithmetic types (`Data32`, `Data64`, `Modulus`, `OPERATOR`, `OPERATOR_GPU`, `Root`, `Ninverse`, ...) are also available in the global namespace; define `GPUNTT_NO_GLOBAL_ALIASES` to disable them.

## Device API

Besides the host functions (`GPU_NTT`, `GPU_INTT`, ...), GPU-NTT can run the NTT inside your own kernels. Other work on the data, such as a pointwise product, then runs in the same kernel without an extra round trip through global memory. An NTT is described at compile time as a sum of operators:

```cpp
#include "gpuntt/ntt_merge/descriptor.cuh"
using namespace gpuntt;

using NTT = decltype(Size<1024>() + Direction<FORWARD>() + Precision<Data64>() +
                     ReductionPoly<X_N_plus>() + Block());
```

| Operator | Required | Meaning |
|---|---|---|
| `Size<N>` | yes | Ring size, a power of two in [2, 2^28]. |
| `Direction<FORWARD>`, `Direction<INVERSE>` | yes | |
| `Precision<Data32>`, `Precision<Data64>` | yes | Coefficient type. `load` also accepts signed input (`Data32s`, `Data64s`) and the inverse `store` can write signed output. |
| `ReductionPoly<X_N_minus>`, `ReductionPoly<X_N_plus>` | for `execute` | X^N - 1 or X^N + 1. |
| `Block` | no | The NTT is computed by a thread block. |
| `BatchesPerBlock<B>` | no | Polynomials per block, only for N up to 2^9. The default comes from the tuned kernel table. |
| `BlockDim<N / 2, B>` | no | Same as `BatchesPerBlock<B>`. |
| `Layout<PerPolynomial>` | no | Only `PerPolynomial` is supported. |
| `SM<890>` | no | Accepted; all architectures currently use the same tuned kernel tables. |

Each operator may appear once; invalid descriptions fail to compile with a `static_assert` message. The device functions live in headers and are inlined into your kernel, so the device API needs neither relocatable device code (`-rdc`) nor link-time optimization. The root tables, the modulus and N^-1 are the ones the host API takes.

### Root tables and N^-1

A description with `ReductionPoly` builds its root table on the host from a prime modulus and a root of unity: a primitive 2N-th root for X^N + 1 or a primitive N-th root for X^N - 1 (`psi` and `omega` of `NTTParameters`). The inverse description builds the inverse table from the same root.

```cpp
Modulus<Data64> modulus(prime);
std::vector<Root<Data64>> roots = NTT::make_twiddles_host(modulus, psi);  // NTT::twiddle_table_size entries
Data64 n_inverse = InverseNTT::n_inv(modulus);  // N^-1 mod prime
```

- `NTT::twiddle_table_size` is N for X^N + 1 and N / 2 for X^N - 1. `make_twiddles_host` throws `std::invalid_argument` if the root does not have the right order.
- The tables are the ones `GPU_NTT` / `GPU_INTT` take, so they can be shared with the host API. RNS tables for the host API hold the roots of prime `i` at offset `i << log2(N)` for both reduction polynomials.
- `Modulus<T>` can also be built inside a kernel. There, a modulus larger than the supported size (30 bits for `Data32`, 61 bits for `Data64`) is not rejected.

### Ring sizes up to 2^10: one kernel

Block `b` transforms the `B = NTT::batches_per_block` polynomials that start at `data + b * B * N`:

```cpp
__global__ void my_ntt(Data64* data, const Root<Data64>* roots,
                       Modulus<Data64> modulus, int batch)
{
    extern __shared__ char shared_memory_raw[];
    Data64* shared_memory = reinterpret_cast<Data64*>(shared_memory_raw);

    const int first = blockIdx.x * NTT::batches_per_block;
    const int num_polys = min(NTT::batches_per_block, batch - first);
    Data64* polys = data + size_t(first) * NTT::size;

    NTT().load(shared_memory, polys, modulus, num_polys);
    NTT().execute(shared_memory, roots, modulus);
    // ... your own work on shared_memory, then __syncthreads() ...
    NTT().store(shared_memory, polys, num_polys);
}

my_ntt<<<NTT::grid(batch), NTT::block_dim, NTT::shared_memory_size>>>(
    data, roots, modulus, batch);
```

- Shared memory holds the block's polynomials one after another (polynomial `j` at `shared_memory + j * N`), with coefficients in the same order as in global memory. After the forward `execute` they are in the order of the `GPU_NTT` output, so element-wise work can use `OPERATOR_GPU<T>::add`, `sub` and `mult` directly.
- `execute` ends with a block barrier. Code that writes shared memory between `execute` and `store` (or another `execute`) must end with `__syncthreads()`.
- The inverse `execute` leaves N · INTT(x) in shared memory; the inverse `store(shared_memory, data, modulus, n_inverse, num_polys)` multiplies by `n_inverse` = N^-1 while writing, which saves a pass over shared memory.
- `num_polys` (at most B) is the number of polynomials of a last block that is not full.

### Larger ring sizes: one kernel per stage

Larger transforms run as `NTT::num_stages` kernels. Stage kernels transform polynomial `blockIdx.z` of the batch:

```cpp
using BigNTT = decltype(Size<(1 << 16)>() + Direction<FORWARD>() +
                        Precision<Data64>() + ReductionPoly<X_N_plus>() + Block());

template <int I>
__global__ void my_stage(Data64* in, Data64* out, const Root<Data64>* roots,
                         Modulus<Data64> modulus)
{
    extern __shared__ char shared_memory_raw[];
    Data64* shared_memory = reinterpret_cast<Data64*>(shared_memory_raw);

    BigNTT::stage<I>().load(shared_memory, in, modulus);
    BigNTT::stage<I>().execute(shared_memory, roots, modulus);
    BigNTT::stage<I>().store(shared_memory, out);
}

// For I = 0, 1, ..., BigNTT::num_stages - 1, in order. Stage 0 reads `in`; every
// stage writes `out`, which the later stages read and write in place.
using Stage = BigNTT::stage<I>;
my_stage<I><<<Stage::grid(batch), Stage::block_dim, Stage::shared_memory_size>>>(
    I == 0 ? in : out, out, roots, modulus);
```

- The last inverse stage (`BigNTT::stage<I>::last`) stores with `store(shared_memory, out, modulus, n_inverse)`.
- A stage holds a strided part of each polynomial in shared memory, so element-wise work with other polynomials fits best in the single kernel sizes or in a separate kernel.

### RNS

There is no separate RNS interface: pick the modulus of each polynomial yourself and pass the matching part of the root table, which holds the roots of prime `i` at offset `i << log2(N)`. For example, in a stage kernel `modulus = moduli[blockIdx.z % mod_count]` and `roots + (size_t(blockIdx.z % mod_count) << log2(N))`. In the single kernel API this needs `BatchesPerBlock<1>`, since all polynomials of a block share one modulus.

A complete example (fused polynomial multiplication modulo X^N + 1, checked against schoolbook multiplication, and a staged NTT) is in [`example/ntt_merge/test_device_api.cu`](example/ntt_merge/test_device_api.cu); the device API tests are in [`test/test_ntt_device.cu`](test/test_ntt_device.cu).

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
or
```
@ARTICLE{11003946,
      author={Ozcan, Alisah and Javeed, Arsalan and Savas, Erkay},
      journal={IEEE Access}, 
      title={High-Performance Number Theoretic Transform on GPU Through radix2-CT and 4-Step Algorithms}, 
      year={2025},
      volume={13},
      number={},
      pages={87862-87883},
      keywords={Graphics processing units;Polynomials;Instruction sets;Parallel processing;Kernel;Optimization;Memory management;Transforms;Computational efficiency;Tensors;Graphical processing unit;homomorphic cryptography;hardware acceleration;number theoretic transform;polynomial arithmetic},
      doi={10.1109/ACCESS.2025.3570024}
}
```
## License
This project is licensed under the [Apache License](LICENSE). For more details, please refer to the License file.

## Contact
If you have any questions or feedback, feel free to contact me: 
- Email: alisah@sabanciuniv.edu
- LinkedIn: [Profile](https://www.linkedin.com/in/ali%C5%9Fah-%C3%B6zcan-472382305/)
