# Changelog

## v1.0.0

The merge NTT now runs compile-time kernels and has a device API for running the NTT inside your own kernels. Code written for v0.0.0 needs the changes listed under [Migrating from v0.0.0](#migrating-from-v000).

### Added

- Device API (`gpuntt/ntt_merge/descriptor.cuh`): an NTT is described at compile time as a sum of operators (`Size`, `Direction`, `Precision`, `ReductionPoly`, `Block`, `BatchesPerBlock`, `BlockDim`, ...) and runs inside your own kernels with `load`, `execute` and `store`, per stage for ring sizes above 2^10. See the README section "Device API".
  - `for_each` and `entries()` give each thread's coefficients in shared memory for element-wise work, e.g. a pointwise product between the forward and the inverse NTT in one kernel.
  - The last forward stage and the first inverse stage share their layout, so they can run in one kernel.
  - `NTT::launch` runs the whole transform from the host.
  - `BlockDim<T>` sets the threads per kernel (128, 256, 512 or 1024) for ring sizes from 2^10; the stages are planned at compile time.
- Root tables and N^-1 from a description: `make_twiddles_host`, `make_twiddles` (built on the GPU), `twiddle_table_size`, `n_inv`.
- `gpuntt/common/fields.cuh`: ready primes with their roots of unity (`ntt_primes32`, `ntt_primes64`, `root_of_unity`).
- `Modulus<T>` can be built in device code.
- Installed `gpuntt/config.cuh` with the version (`GPUNTT_VERSION_MAJOR`, `GPUNTT_VERSION_MINOR`, `GPUNTT_VERSION_PATCH`, `GPUNTT_VERSION`), the CUDA architectures and `GPUNTT_CC_89_KERNEL_CONFIG`.
- CMake package: `find_package(GPUNTT 1.0)` checks the version (same major version, at least the requested one); `GPUNTT_CUDA_ARCHITECTURES` holds the architectures the library was compiled for.
- Input checks that throw `std::invalid_argument`: ring sizes out of range, invalid batch sizes, and moduli larger than the Barrett reduction supports (30 bits for `Data32`, 61 bits for `Data64`).
- CTest correctness tests (`GPUNTT_BUILD_TESTS`), benchmarks up to ring size 2^28, and examples that take `[logn] [batch] [32|64]` from the command line.

### Changed

- All types and functions are in the `gpuntt` namespace. The arithmetic types (`Data32`, `Data64`, `Modulus`, `OPERATOR`, `OPERATOR_GPU`, ...) are also available in the global namespace unless `GPUNTT_NO_GLOBAL_ALIASES` is defined.
- The host API (`GPU_NTT`, `GPU_INTT`, ...) runs kernels whose launch parameters are fixed at compile time for each ring size. The results are the same as before; the kernel tables are no longer part of the public header.
- `gpuntt/ntt_merge/ntt.cuh` no longer includes the CPU reference; include `gpuntt/ntt_merge/ntt_cpu.cuh` for `NTTCPU`.
- The version is `1.0.0` and follows the release tags. Installed paths and the library name carry `<major>.<minor>`: `include/GPUNTT-1.0`, `lib/cmake/GPUNTT-1.0`, `libntt-1.0.a`.

### Removed

- The 4-step NTT (`GPU_4STEP_NTT`, `GPU_Transpose`, `NTTParameters4Step`, the `ntt_4step` headers, examples and benchmarks). It stays available in v0.0.0.
- The unused `zero_padding` field of `ntt_configuration` and `ntt_rns_configuration`.
- The type aliases `Root`, `Root32`, `Root64`, `Ninverse`, `Ninverse32` and `Ninverse64`.
- The kernels declared in `ntt.cuh` (`ForwardCore`, `InverseCore`, `ForwardCoreLowRing`, `InverseCoreLowRing`, ...). Use `GPU_NTT` / `GPU_INTT` or the device API.
- `CudaDevice()` and `check_result()` from `common.cuh`; they were example helpers.

### Fixed

- Single modulus `GPU_INTT` with several kernels read its input from the wrong pointer.
- Root table and modulus indexing in the RNS `PerCoefficient` NTT.
- The ordered `GPU_INTT` for large ring sizes read its input from the wrong pointer.
- A warp level race in the last forward NTT stages.
- `NTTParameters` accepted ring sizes outside the supported range.

### Migrating from v0.0.0

- Qualify names with `gpuntt::` or add `using namespace gpuntt;`. The arithmetic types keep working without it unless `GPUNTT_NO_GLOBAL_ALIASES` is defined.
- Remove `.zero_padding = ...` from `ntt_configuration` and `ntt_rns_configuration`.
- Replace `Root<T>` and `Ninverse<T>` with `T`, `Root64` and `Ninverse64` with `Data64`, `Root32` and `Ninverse32` with `Data32`.
- Include `gpuntt/ntt_merge/ntt_cpu.cuh` where `NTTCPU` is used.
- Moduli above 30 bits (`Data32`) or 61 bits (`Data64`) are now rejected with `std::invalid_argument`; the Barrett reduction does not support them.
- For the 4-step NTT, stay on v0.0.0.

## v0.0.0

The state before the rewrite: merge and 4-step NTT for 32-bit and 64-bit data with Barrett reduction, as on the `main` branch up to commit `d03c5ea`, with updated copyright headers.
