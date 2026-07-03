// Standalone validated native Goldilocks/G64 benchmark for the paper_version
// 4-step NTT path.
//
// This benchmark uses:
//   - Goldilocks modulus p = 2^64 - 2^32 + 1
//   - generator 7
//   - primitive root/twiddle tables
//   - the expected GPU 4-step root-table bit-reversal order
//   - the transpose -> GPU_4STEP_NTT -> transpose device wrapper
//   - an independent natural-order CPU NTT reference for correctness
//
// The CPU reference is used only for validation and is not part of the
// CUDA event timing region.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "ntt.cuh"
#include "ntt_4step.cuh"

static constexpr uint64_t GOLDILOCKS_P = 0xffffffff00000001ULL;
static constexpr uint64_t GOLDILOCKS_GENERATOR = 7ULL;

#define CUDA_CHECK(x)                                                        \
    do {                                                                     \
        cudaError_t err__ = (x);                                             \
        if (err__ != cudaSuccess) {                                          \
            std::cerr << "CUDA error: " << cudaGetErrorString(err__)         \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)

struct Options {
    int logn = 12;
    int runs = 5;
    int warmup = 1;
    uint64_t seed = 1234;
    uint64_t delta_index = 0;
    std::string input_mode = "random";   // random | zero | ones | delta
    std::string w_mode = "bitrev_i";  // bitrev_i | identity | linear
    std::string validate = "full";       // none | full
    std::string timed_region = "device_wrapper"; // device_wrapper | core
    int dump_mismatch = 8;
};

static char* need_arg(int& i, int argc, char** argv, const char* k) {
    if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + k);
    return argv[++i];
}

static Options parse_args(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; i++) {
        if (!std::strcmp(argv[i], "--logn")) opt.logn = std::atoi(need_arg(i, argc, argv, argv[i]));
        else if (!std::strcmp(argv[i], "--runs")) opt.runs = std::atoi(need_arg(i, argc, argv, argv[i]));
        else if (!std::strcmp(argv[i], "--warmup")) opt.warmup = std::atoi(need_arg(i, argc, argv, argv[i]));
        else if (!std::strcmp(argv[i], "--seed")) opt.seed = std::strtoull(need_arg(i, argc, argv, argv[i]), nullptr, 10);
        else if (!std::strcmp(argv[i], "--delta-index")) opt.delta_index = std::strtoull(need_arg(i, argc, argv, argv[i]), nullptr, 10);
        else if (!std::strcmp(argv[i], "--input")) opt.input_mode = need_arg(i, argc, argv, argv[i]);
        else if (!std::strcmp(argv[i], "--validate")) opt.validate = need_arg(i, argc, argv, argv[i]);
        else if (!std::strcmp(argv[i], "--timed-region")) opt.timed_region = need_arg(i, argc, argv, argv[i]);
        else if (!std::strcmp(argv[i], "--w-mode")) opt.w_mode = need_arg(i, argc, argv, argv[i]);
        else if (!std::strcmp(argv[i], "--dump-mismatch")) opt.dump_mismatch = std::atoi(need_arg(i, argc, argv, argv[i]));
        else {
            throw std::runtime_error(std::string("unknown arg: ") + argv[i]);
        }
    }

    if (opt.logn < 12 || opt.logn > 24) throw std::runtime_error("logn must be 12..24");
    if (opt.input_mode != "random" && opt.input_mode != "zero" &&
        opt.input_mode != "ones" && opt.input_mode != "delta") {
        throw std::runtime_error("--input must be random, zero, ones, or delta");
    }
    if (opt.w_mode != "bitrev_i" && opt.w_mode != "identity" && opt.w_mode != "linear") {
        throw std::runtime_error("--w-mode must be bitrev_i, identity, or linear");
    }
    if (opt.validate != "none" && opt.validate != "full") {
        throw std::runtime_error("--validate must be none or full");
    }
    if (opt.timed_region != "device_wrapper" && opt.timed_region != "core") {
        throw std::runtime_error("--timed-region must be device_wrapper or core");
    }
    return opt;
}

static inline uint64_t g_add(uint64_t a, uint64_t b) {
    uint64_t r = a + b;
    if (r < a || r >= GOLDILOCKS_P) r -= GOLDILOCKS_P;
    return r;
}

static inline uint64_t g_sub(uint64_t a, uint64_t b) {
    return (a >= b) ? (a - b) : (GOLDILOCKS_P - (b - a));
}

static inline uint64_t g_mul(uint64_t a, uint64_t b) {
    __uint128_t z = static_cast<__uint128_t>(a) * b;
    uint64_t lo = static_cast<uint64_t>(z);
    uint64_t hi = static_cast<uint64_t>(z >> 64);

    __uint128_t t = static_cast<__uint128_t>(lo)
                  + (static_cast<__uint128_t>(hi) << 32)
                  - hi;

    uint64_t r = static_cast<uint64_t>(t);
    uint64_t carry = static_cast<uint64_t>(t >> 64);

    __uint128_t t2 = static_cast<__uint128_t>(r)
                   + (static_cast<__uint128_t>(carry) << 32)
                   - carry;

    r = static_cast<uint64_t>(t2);
    carry = static_cast<uint64_t>(t2 >> 64);

    if (carry) {
        __uint128_t t3 = static_cast<__uint128_t>(r)
                       + (static_cast<__uint128_t>(carry) << 32)
                       - carry;
        r = static_cast<uint64_t>(t3);
    }

    while (r >= GOLDILOCKS_P) r -= GOLDILOCKS_P;
    return r;
}

static uint64_t g_pow(uint64_t base, uint64_t exp) {
    uint64_t r = 1;
    while (exp) {
        if (exp & 1) r = g_mul(r, base);
        base = g_mul(base, base);
        exp >>= 1;
    }
    return r;
}

static int bitreverse_int(int x, int bits) {
    int r = 0;
    for (int i = 0; i < bits; i++) {
        r = (r << 1) | (x & 1);
        x >>= 1;
    }
    return r;
}

static std::vector<int> matrix_shape(int logn) {
    switch (logn) {
        case 12: return {32, 128};
        case 13: return {32, 256};
        case 14: return {32, 512};
        case 15: return {64, 512};
        case 16: return {128, 512};
        case 17: return {32, 4096};
        case 18: return {32, 8192};
        case 19: return {32, 16384};
        case 20: return {32, 32768};
        case 21: return {64, 32768};
        case 22: return {128, 32768};
        case 23: return {128, 65536};
        case 24: return {256, 65536};
        default: throw std::runtime_error("unsupported logn");
    }
}

static void generate_g64_4step_tables(
    int logn,
    const std::string& w_mode,
    std::vector<Root>& n1_roots,
    std::vector<Root>& n2_roots,
    std::vector<Root>& w_table
) {
    const uint64_t n = 1ULL << logn;
    const auto shape = matrix_shape(logn);
    const int n1 = shape[0];
    const int n2 = shape[1];

    const uint64_t omega = g_pow(GOLDILOCKS_GENERATOR, (GOLDILOCKS_P - 1) / n);
    const uint64_t omega_n1 = g_pow(omega, n / n1);
    const uint64_t omega_n2 = g_pow(omega, n / n2);

    n1_roots.assign(n1 >> 1, 1);
    n2_roots.assign(n2 >> 1, 1);

    for (int i = 1; i < (n1 >> 1); i++) n1_roots[i] = static_cast<Root>(g_mul(n1_roots[i - 1], omega_n1));
    for (int i = 1; i < (n2 >> 1); i++) n2_roots[i] = static_cast<Root>(g_mul(n2_roots[i - 1], omega_n2));

    // Match NTTParameters4Step::gpu_root_of_unity_table_generator:
    // GPU root tables are stored in bit-reversed order.
    {
        std::vector<Root> a = n1_roots;
        int lg = 0;
        while ((1 << lg) < static_cast<int>(a.size())) lg++;
        for (int i = 0; i < static_cast<int>(a.size()); i++) {
            n1_roots[i] = a[bitreverse_int(i, lg)];
        }
    }
    {
        std::vector<Root> a = n2_roots;
        int lg = 0;
        while ((1 << lg) < static_cast<int>(a.size())) lg++;
        for (int i = 0; i < static_cast<int>(a.size()); i++) {
            n2_roots[i] = a[bitreverse_int(i, lg)];
        }
    }

    const int lg_n1 = static_cast<int>(std::log2(n1));
    w_table.clear();
    w_table.reserve(static_cast<size_t>(n1) * n2);

    for (size_t k = 0; k < static_cast<size_t>(n1) * n2; k++) {
        if (w_mode == "identity") {
            w_table.push_back(static_cast<Root>(1));
        } else if (w_mode == "linear") {
            w_table.push_back(static_cast<Root>(g_pow(omega, k)));
        } else {
            size_t i = k / n2;
            size_t j = k % n2;
            uint64_t ii = static_cast<uint64_t>(bitreverse_int(static_cast<int>(i), lg_n1));
            w_table.push_back(static_cast<Root>(g_pow(omega, ii * static_cast<uint64_t>(j))));
        }
    }
}

static void fill_input(std::vector<Data>& a, const Options& opt) {
    if (opt.input_mode == "zero") {
        std::fill(a.begin(), a.end(), static_cast<Data>(0));
    } else if (opt.input_mode == "ones") {
        std::fill(a.begin(), a.end(), static_cast<Data>(1));
    } else if (opt.input_mode == "delta") {
        std::fill(a.begin(), a.end(), static_cast<Data>(0));
        if (opt.delta_index >= a.size()) throw std::runtime_error("delta-index out of range");
        a[opt.delta_index] = static_cast<Data>(1);
    } else {
        std::mt19937_64 rng(opt.seed);
        for (auto& x : a) x = static_cast<Data>(rng() % GOLDILOCKS_P);
    }
}

static void cpu_ntt_forward_natural(std::vector<uint64_t>& a) {
    const size_t n = a.size();
    const int logn = static_cast<int>(std::log2(n));
    const uint64_t omega = g_pow(GOLDILOCKS_GENERATOR, (GOLDILOCKS_P - 1) / n);

    for (size_t i = 0; i < n; i++) {
        size_t j = 0;
        size_t x = i;
        for (int b = 0; b < logn; b++) {
            j = (j << 1) | (x & 1);
            x >>= 1;
        }
        if (i < j) std::swap(a[i], a[j]);
    }

    for (size_t len = 2; len <= n; len <<= 1) {
        uint64_t wlen = g_pow(omega, n / len);
        for (size_t i = 0; i < n; i += len) {
            uint64_t w = 1;
            for (size_t j = 0; j < len / 2; j++) {
                uint64_t u = a[i + j];
                uint64_t v = g_mul(a[i + j + len / 2], w);
                a[i + j] = g_add(u, v);
                a[i + j + len / 2] = g_sub(u, v);
                w = g_mul(w, wlen);
            }
        }
    }
}

static float percentile(std::vector<float> v, double p) {
    std::sort(v.begin(), v.end());
    double idx = p * (v.size() - 1);
    size_t lo = static_cast<size_t>(std::floor(idx));
    size_t hi = static_cast<size_t>(std::ceil(idx));
    if (lo == hi) return v[lo];
    double t = idx - lo;
    return static_cast<float>(v[lo] * (1.0 - t) + v[hi] * t);
}

static void print_stats(const std::vector<float>& times) {
    auto sorted = times;
    std::sort(sorted.begin(), sorted.end());

    double sum = std::accumulate(times.begin(), times.end(), 0.0);
    double mean = sum / times.size();
    double var = 0.0;
    for (float x : times) var += (x - mean) * (x - mean);
    var /= times.size();

    std::cout << "min_ms = " << sorted.front() << "\n";
    std::cout << "median_ms = " << percentile(times, 0.50) << "\n";
    std::cout << "p95_ms = " << percentile(times, 0.95) << "\n";
    std::cout << "mean_ms = " << mean << "\n";
    std::cout << "max_ms = " << sorted.back() << "\n";
    std::cout << "stddev_ms = " << std::sqrt(var) << "\n";
}

static std::vector<float> run_gpu(
    const Options& opt,
    const std::vector<Data>& input,
    const std::vector<Root>& n1_roots,
    const std::vector<Root>& n2_roots,
    const std::vector<Root>& w_table,
    std::vector<Data>& output
) {
    const size_t n = input.size();
    auto shape = matrix_shape(opt.logn);

    Data *d_in = nullptr, *d_tmp = nullptr, *d_out = nullptr;
    Root *d_n1 = nullptr, *d_n2 = nullptr, *d_w = nullptr;

    CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(Data)));
    CUDA_CHECK(cudaMalloc(&d_tmp, n * sizeof(Data)));
    CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(Data)));
    CUDA_CHECK(cudaMalloc(&d_n1, n1_roots.size() * sizeof(Root)));
    CUDA_CHECK(cudaMalloc(&d_n2, n2_roots.size() * sizeof(Root)));
    CUDA_CHECK(cudaMalloc(&d_w, w_table.size() * sizeof(Root)));

    CUDA_CHECK(cudaMemcpy(d_in, input.data(), n * sizeof(Data), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_n1, n1_roots.data(), n1_roots.size() * sizeof(Root), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_n2, n2_roots.data(), n2_roots.size() * sizeof(Root), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w, w_table.data(), w_table.size() * sizeof(Root), cudaMemcpyHostToDevice));

    ntt4step_configuration cfg{};
    cfg.n_power = opt.logn;
    cfg.ntt_type = FORWARD;
    cfg.stream = 0;

    Modulus modulus(static_cast<Data>(GOLDILOCKS_P));

    std::vector<float> times;
    times.reserve(opt.runs);

    for (int i = 0; i < opt.warmup + opt.runs; i++) {
        CUDA_CHECK(cudaMemset(d_tmp, 0, n * sizeof(Data)));
        CUDA_CHECK(cudaMemset(d_out, 0, n * sizeof(Data)));

        cudaEvent_t start_ev, stop_ev;
        CUDA_CHECK(cudaEventCreate(&start_ev));
        CUDA_CHECK(cudaEventCreate(&stop_ev));

        if (opt.timed_region == "core") {
            // Prepare the internal 4-step input layout outside the timed region.
            GPU_Transpose(d_in, d_tmp, shape[0], shape[1], opt.logn, 1);
            CUDA_CHECK(cudaDeviceSynchronize());

            CUDA_CHECK(cudaEventRecord(start_ev));

            GPU_4STEP_NTT(
                d_tmp,
                d_out,
                d_n1,
                d_n2,
                d_w,
                modulus,
                cfg,
                1
            );

            CUDA_CHECK(cudaEventRecord(stop_ev));
            CUDA_CHECK(cudaEventSynchronize(stop_ev));

            // Reconstruct the wrapper output outside the timed region for validation.
            GPU_Transpose(d_out, d_tmp, shape[0], shape[1], opt.logn, 1);
            CUDA_CHECK(cudaDeviceSynchronize());
        } else {
            CUDA_CHECK(cudaEventRecord(start_ev));

            // Timed device wrapper:
            // natural input -> transposed layout -> 4step NTT -> natural output
            GPU_Transpose(d_in, d_tmp, shape[0], shape[1], opt.logn, 1);

            GPU_4STEP_NTT(
                d_tmp,
                d_out,
                d_n1,
                d_n2,
                d_w,
                modulus,
                cfg,
                1
            );

            GPU_Transpose(d_out, d_tmp, shape[0], shape[1], opt.logn, 1);

            CUDA_CHECK(cudaEventRecord(stop_ev));
            CUDA_CHECK(cudaEventSynchronize(stop_ev));
        }
        CUDA_CHECK(cudaEventSynchronize(stop_ev));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_ev, stop_ev));

        CUDA_CHECK(cudaEventDestroy(start_ev));
        CUDA_CHECK(cudaEventDestroy(stop_ev));

        if (i >= opt.warmup) times.push_back(ms);
    }

    output.resize(n);
    CUDA_CHECK(cudaMemcpy(output.data(), d_tmp, n * sizeof(Data), cudaMemcpyDeviceToHost));

    // GPU-NTT 4step leaves output in 2D bit-reversed layout.
    // Convert to natural order for independent CPU-reference validation.
    {
        std::vector<Data> raw = output;
        const int n1 = shape[0];
        const int n2 = shape[1];

        int lg_n1 = 0;
        while ((1 << lg_n1) < n1) lg_n1++;

        int lg_n2 = 0;
        while ((1 << lg_n2) < n2) lg_n2++;

        for (int row = 0; row < n2; row++) {
            for (int col = 0; col < n1; col++) {
                const size_t src = static_cast<size_t>(row) * n1 + col;
                const size_t dst =
                    static_cast<size_t>(bitreverse_int(row, lg_n2)) * n1 +
                    static_cast<size_t>(bitreverse_int(col, lg_n1));
                output[dst] = raw[src];
            }
        }
    }

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_tmp));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_n1));
    CUDA_CHECK(cudaFree(d_n2));
    CUDA_CHECK(cudaFree(d_w));

    return times;
}

static bool validate_full(
    const Options& opt,
    const std::vector<Data>& input,
    const std::vector<Data>& gpu_output
) {
    std::vector<uint64_t> cpu(input.begin(), input.end());

    auto t0 = std::chrono::high_resolution_clock::now();
    cpu_ntt_forward_natural(cpu);
    auto t1 = std::chrono::high_resolution_clock::now();

    double cpu_s = std::chrono::duration<double>(t1 - t0).count();

    size_t mismatches = 0;
    for (size_t i = 0; i < cpu.size(); i++) {
        if (cpu[i] != static_cast<uint64_t>(gpu_output[i])) {
            if (static_cast<int>(mismatches) < opt.dump_mismatch) {
                std::cout << "mismatch[" << i << "] cpu=" << cpu[i]
                          << " gpu=" << static_cast<uint64_t>(gpu_output[i]) << "\n";
            }
            mismatches++;
        }
    }

    std::cout << "cpu_reference_seconds = " << cpu_s << "\n";
    std::cout << "mismatch_count = " << mismatches << "\n";
    return mismatches == 0;
}

int main(int argc, char** argv) {
    try {
        Options opt = parse_args(argc, argv);

        const size_t n = 1ULL << opt.logn;

        std::cout << "benchmark = gpu_ntt_paper_version_goldilocks_single_cu\n";
        std::cout << "field = Goldilocks\n";
        std::cout << "modulus_hex = 0xffffffff00000001\n";
        std::cout << "generator = 7\n";
        std::cout << "logn = " << opt.logn << "\n";
        std::cout << "n = " << n << "\n";
        std::cout << "lde = 8\n";
        std::cout << "logical_size = " << (n / 8) << "\n";
        std::cout << "batch = 1\n";
        std::cout << "operation = forward_ntt\n";
        std::cout << "timed_region = "
                  << (opt.timed_region == "core"
                      ? "core_only_gpu_4step_ntt"
                      : "device_wrapper_transpose_core_transpose")
                  << "\n";
        std::cout << "validation_mode = " << opt.validate << "\n";
        std::cout << "input_mode = " << opt.input_mode << "\n";
        std::cout << "w_mode = " << opt.w_mode << "\n";
        if (opt.input_mode == "delta") std::cout << "delta_index = " << opt.delta_index << "\n";

        std::vector<Data> input(n);
        fill_input(input, opt);

        std::vector<Root> n1_roots, n2_roots, w_table;
        generate_g64_4step_tables(opt.logn, opt.w_mode, n1_roots, n2_roots, w_table);

        std::cout << "n1_roots = " << n1_roots.size() << "\n";
        std::cout << "n2_roots = " << n2_roots.size() << "\n";
        std::cout << "w_table = " << w_table.size() << "\n";

        std::vector<Data> gpu_output;
        auto times = run_gpu(opt, input, n1_roots, n2_roots, w_table, gpu_output);

        std::cout << "runs = " << opt.runs << "\n";
        std::cout << "warmup = " << opt.warmup << "\n";
        print_stats(times);

        if (opt.validate == "full") {
            bool ok = validate_full(opt, input, gpu_output);
            std::cout << "validation_status = " << (ok ? "pass" : "fail") << "\n";
            return ok ? 0 : 2;
        }

        std::cout << "validation_status = not_run\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error = " << e.what() << "\n";
        return 1;
    }
}

