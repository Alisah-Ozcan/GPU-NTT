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

The examples take the ring size, the batch size and the coefficient type (`32` for `Data32`, `64` for `Data64`) from the command line; the defaults are `12 1 64`.

Configure + build:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86 -DGPUNTT_BUILD_EXAMPLES=ON
cmake --build build --parallel
```

Run CPU examples:

```bash
./build/bin/example/cpu_merge_ntt_examples [RING_SIZE_IN_LOG2] [BATCH_SIZE] [32|64]
# Example: ./build/bin/example/cpu_merge_ntt_examples 15 1 64
```

Run GPU examples:

```bash
./build/bin/example/gpu_merge_ntt_examples [RING_SIZE_IN_LOG2] [BATCH_SIZE] [32|64]
./build/bin/example/gpu_merge_intt_examples [RING_SIZE_IN_LOG2] [BATCH_SIZE] [32|64]
# Example: ./build/bin/example/gpu_merge_ntt_examples 12 1 32
```

Run the device API examples (see [Device API](#device-api)):

```bash
./build/bin/example/gpu_device_api_examples <BATCH_SIZE>
./build/bin/example/gpu_device_api_rns_examples <RNS_POLY_COUNT>
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

`benchmark_device_api` holds `poly_mul`: polynomial multiplication modulo X^N + 1 with one operand in the NTT domain, a ← INTT(NTT(a) · b̂), for about 2^22 coefficients and `LogN` 1–10. `Method=fused` runs it in one kernel with the device API, `Method=separate` runs `GPU_NTT`, a pointwise product kernel and `GPU_INTT`.

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
```

- Neither relocatable device code (`CUDA_SEPARABLE_COMPILATION`, `-rdc`) nor device link-time optimization is needed, also for the device API.
- `GPUNTT_CUDA_ARCHITECTURES` holds the GPU architectures the installed library was compiled for.
- The installed `gpuntt/config.cuh` records the build settings (`GPUNTT_CC_89_KERNEL_CONFIG`, version, architectures), so code compiled without CMake, e.g. `nvcc -I<prefix>/include/GPUNTT-1.0 ... -L<prefix>/lib -lntt-1.0`, uses the same kernel tables as the library.

All GPU-NTT types and functions are in the `gpuntt` namespace. For compatibility, the arithmetic types (`Data32`, `Data64`, `Modulus`, `OPERATOR`, `OPERATOR_GPU`, ...) are also available in the global namespace; define `GPUNTT_NO_GLOBAL_ALIASES` to disable them.

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
std::vector<Data64> roots = NTT::make_twiddles_host(modulus, psi);  // NTT::twiddle_table_size entries
Data64 n_inverse = InverseNTT::n_inv(modulus);  // N^-1 mod prime
```

- `NTT::make_twiddles(table, modulus, root, stream = 0)` writes the same table with a kernel to `table` in device memory (`NTT::twiddle_table_size` entries). For large N it is much faster: about 9 ms instead of 38 s on the host for N = 2^28 on an RTX 4090.
- `NTT::twiddle_table_size` is N for X^N + 1 and N / 2 for X^N - 1. `make_twiddles_host` throws `std::invalid_argument` if the root does not have the right order.
- The tables are the ones `GPU_NTT` / `GPU_INTT` take, so they can be shared with the host API. RNS tables for the host API hold the roots of prime `i` at offset `i << log2(N)` for both reduction polynomials.
- [`gpuntt/common/fields.cuh`](src/include/gpuntt/common/fields.cuh) lists ready primes with their roots of unity: `ntt_primes32` (29 and 30 bits, ring sizes up to 2^22, the first one up to 2^25) and `ntt_primes64` (60 and 61 bits, every ring size up to 2^28). `NTT::make_twiddles_host(ntt_primes64[i])` and `NTT::make_twiddles(table, ntt_primes64[i])` pick the root themselves; `root_of_unity(prime, log_order)` gives a root of order 2^log_order.
- `Modulus<T>` can also be built inside a kernel. There, a modulus larger than the supported size (30 bits for `Data32`, 61 bits for `Data64`) is not rejected.

### Ring sizes up to 2^10: one kernel

Block `b` transforms the `B = NTT::batches_per_block` polynomials that start at `data + b * B * N`:

```cpp
__global__ void my_ntt(Data64* data, const Data64* other, const Data64* roots,
                       Modulus<Data64> modulus, int batch)
{
    Data64* shared_memory = NTT::shared_memory();
    Data64* polys = NTT::block_data(data);          // data + blockIdx.x * B * N
    const Data64* other_polys = NTT::block_data(other);
    const int num_polys = NTT::num_polys(batch);    // B, fewer in a last block

    NTT().load(shared_memory, polys, modulus, num_polys);
    NTT().execute(shared_memory, roots, modulus);
    // Your own element-wise work, e.g. a product with another array:
    NTT().for_each(shared_memory, [&](Data64& value, size_t index) {
        value = OPERATOR_GPU<Data64>::mult(value, other_polys[index], modulus);
    }, num_polys);
    NTT().store(shared_memory, polys, num_polys);
}

my_ntt<<<NTT::grid(batch), NTT::block_dim, NTT::shared_memory_size>>>(
    data, other, roots, modulus, batch);
```

- `NTT::shared_memory()` is the kernel's dynamic shared memory; launch with `NTT::shared_memory_size` bytes.
- Shared memory holds the block's polynomials one after another (polynomial `j` at `shared_memory + j * N`), with coefficients in the same order as in global memory. After the forward `execute` they are in the order of the `GPU_NTT` output.
- `NTT().for_each(shared_memory, f, num_polys)` calls `f(value, index)` for every entry, spread over the threads; `index` is the entry's position in the array `load` reads. It ends with a block barrier. `OPERATOR_GPU<T>::add`, `sub` and `mult` work on the values.
- `execute` ends with a block barrier. Code that writes shared memory between `execute` and `store` (or another `execute`) without `for_each` must end with `__syncthreads()`.
- The inverse `execute` leaves N · INTT(x) in shared memory; the inverse `store(shared_memory, data, modulus, n_inverse, num_polys)` multiplies by `n_inverse` = N^-1 while writing, which saves a pass over shared memory.
- `num_polys` (at most B) is the number of polynomials of a last block that is not full.

### Larger ring sizes: one kernel per stage

Larger transforms run as `NTT::num_stages` kernels. Stage kernels transform polynomial `blockIdx.z` of the batch:

```cpp
using BigNTT = decltype(Size<(1 << 16)>() + Direction<FORWARD>() +
                        Precision<Data64>() + ReductionPoly<X_N_plus>() + Block());

template <int I>
__global__ void my_stage(Data64* in, Data64* out, const Data64* roots,
                         Modulus<Data64> modulus)
{
    using Stage = BigNTT::stage<I>;
    Data64* shared_memory = Stage::shared_memory();

    Stage().load(shared_memory, in, modulus);
    Stage().execute(shared_memory, roots, modulus);
    Stage().store(shared_memory, out);
}

// Launches the stages in order, I = 0, 1, ..., BigNTT::num_stages - 1. Stage 0
// reads `in`; every stage writes `out`, which the later stages read and write
// in place.
BigNTT::for_each_stage([&](auto I) {
    using Stage = BigNTT::stage<I>;
    my_stage<I><<<Stage::grid(batch), Stage::block_dim, Stage::shared_memory_size>>>(
        I == 0 ? in : out, out, roots, modulus);
});
```

- The last inverse stage (`BigNTT::stage<I>::last`) stores with `store(shared_memory, out, modulus, n_inverse)`.
- A stage holds a strided part of each polynomial in shared memory. `Stage().for_each(shared_memory, f)` calls `f(value, index)` for each entry of the thread, where `index` is the entry's position in `data` (the array `load` reads and `store` writes), so element-wise work with other arrays can run inside a stage kernel. It ends with a block barrier.
- The last forward stage and the first inverse stage have the same launch parameters and keep the same coefficients in shared memory, so they can share a kernel with element-wise work in between, e.g. for a polynomial product a · b with b̂ = NTT(b):

```cpp
using BigInverseNTT = decltype(Size<(1 << 16)>() + Direction<INVERSE>() +
                               Precision<Data64>() + ReductionPoly<X_N_plus>() + Block());
using Last = BigNTT::stage<BigNTT::num_stages - 1>;
using First = BigInverseNTT::stage<0>;

Data64* shared_memory = Last::shared_memory();
Last().load(shared_memory, a, modulus);
Last().execute(shared_memory, roots, modulus);
Last().for_each(shared_memory, [&](Data64& value, size_t index) {
    value = OPERATOR_GPU<Data64>::mult(value, b_hat[index], modulus);
});
First().execute(shared_memory, inverse_roots, modulus);
First().store(shared_memory, a);  // with (modulus, n_inverse) if First::last
// launched with Last::grid(batch), Last::block_dim, Last::shared_memory_size
```

  The other stages run in their own kernels as above. [`example/ntt_merge/test_device_api.cu`](example/ntt_merge/test_device_api.cu) multiplies polynomials with N = 2^12 this way.

### Entries of a thread

Every thread of a single kernel or a stage kernel owns two entries of shared memory: `load` fills them, `store` writes them back, and they keep their positions in between. `NTT().entries(num_polys)` (single kernel) and `Stage().entries()` return them as a `ThreadEntries`: entry `k` (`k < ThreadEntries::count`, which is 2) is `shared_memory[position[k]]` and belongs at `data[index[k]]`, the index that `for_each` passes. In a last single kernel block that is not full, the threads of the missing polynomials have `valid == false`. `for_each` is built on `entries`.

With the entries a thread works on its own values directly, e.g. reads the values of another array into registers before `execute`:

```cpp
const ThreadEntries entries = NTT().entries(num_polys);
NTT().load(shared_memory, polys, modulus, num_polys);
Data64 other_values[ThreadEntries::count] = {};
if (entries.valid)
    for (int k = 0; k < ThreadEntries::count; k++)
        other_values[k] = other_polys[entries.index[k]];
NTT().execute(shared_memory, roots, modulus);
if (entries.valid)
    for (int k = 0; k < ThreadEntries::count; k++) {
        Data64& value = shared_memory[entries.position[k]];
        value = OPERATOR_GPU<Data64>::mult(value, other_values[k], modulus);
    }
NTT().store(shared_memory, polys, num_polys);
```

- `store` writes the thread's own entries, so changing them needs no barrier before `store`; before another `execute`, end with `__syncthreads()`.
- `position` also addresses the same entry in a second shared memory buffer of the same layout, e.g. one filled by a second `load`.
- A thread can keep its two entries of one transform in registers while the same shared memory transforms another polynomial, e.g. NTT(a) + NTT(b) in one kernel without extra shared memory ([`test/test_ntt_device.cu`](test/test_ntt_device.cu), `two_ntt_sum_kernel`).

### RNS

There is no separate RNS interface: pick the modulus of each polynomial yourself and pass the matching part of the root table, which holds the roots of prime `i` at offset `i << log2(N)`. For example, in a stage kernel `modulus = moduli[blockIdx.z % mod_count]` and `roots + (size_t(blockIdx.z % mod_count) << log2(N))`. In the single kernel API this needs `BatchesPerBlock<1>` (the default for N = 2^10), since all polynomials of a block share one modulus.

[`example/ntt_merge/test_device_api_rns.cu`](example/ntt_merge/test_device_api_rns.cu) multiplies RNS polynomials modulo X^N + 1 and three primes of `fields.cuh` in one kernel: residue `i` of RNS polynomial `j` is polynomial `j * mod_count + i`, the layout of the host RNS API, and the root tables of all primes are built on the GPU with `make_twiddles`.

### Performance of fusion

The `poly_mul` benchmark (see [Benchmarking](#benchmarking-gpu-ntt)) compares the fused kernel with three host API calls for about 2^22 coefficients. On an RTX 4090 the fused kernel is 1.3–1.4x faster for `Data32` (e.g. 74.7 µs instead of 100.9 µs at N = 2^10) and 1.3–1.5x faster for `Data64` (149.9 µs instead of 194.2 µs at N = 2^10, 119.5 µs instead of 179.7 µs at N = 2^8), since the data passes through global memory once instead of three times.

A complete example (polynomial multiplication modulo X^N + 1 in one kernel and with stage kernels, checked against schoolbook multiplication, and a staged NTT) is in [`example/ntt_merge/test_device_api.cu`](example/ntt_merge/test_device_api.cu); the device API tests are in [`test/test_ntt_device.cu`](test/test_ntt_device.cu).

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
