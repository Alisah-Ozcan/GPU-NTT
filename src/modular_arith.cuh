// --------------------- //
// Authors: Alisah Ozcan
// --------------------- //

#include <math.h>

#include <cinttypes>
#include <string>
#include <vector>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//#define BARRETT_64
//#define GOLDILOCKS_64
//#define PLANTARD_64

#ifndef MODULAR_ARITHMATIC_H
#define MODULAR_ARITHMATIC_H

typedef unsigned long long Data;

#if defined(BARRETT_64) || defined(GOLDILOCKS_64)
typedef unsigned long long Root;
typedef unsigned long long Root_;
typedef unsigned long long Ninverse;
#elif defined(PLANTARD_64)
typedef ulonglong2 Root;
typedef __uint128_t Root_;
typedef ulonglong2 Ninverse;
#else
#error "Please define reduction type."
#endif

struct Modulus
{
    Data value;
#if defined(BARRETT_64) || defined(PLANTARD_64)
    Data bit;
    Data mu;
#elif defined(GOLDILOCKS_64)

#else
#error "Please define reduction type."
#endif

    // Constructor to initialize the Modulus
    __host__ Modulus(Data q_)
    {
        value = q_;
#if defined(BARRETT_64) || defined(PLANTARD_64)
        bit = bit_generator();
        mu = mu_generator();
#elif defined(GOLDILOCKS_64)

#else
#error "Please define reduction type."
#endif
    }
    __host__ Modulus()
    {
        value = 0;
#if defined(BARRETT_64) || defined(PLANTARD_64)
        bit = 0;
        mu = 0;
#elif defined(GOLDILOCKS_64)

#else
#error "Please define reduction type."
#endif
    }

   private:
#if defined(BARRETT_64) || defined(PLANTARD_64)
    Data bit_generator()
    {
        Data q_bit = log2(value) + 1;

        return q_bit;
    }

    Data mu_generator()
    {
        __uint128_t mu_ = (__uint128_t)(1) << ((2 * bit) + 1);
        mu_ = mu_ / value;

        return mu_;
    }
#elif defined(GOLDILOCKS_64)

#else
#error "Please define reduction type."
#endif
};

#if defined(BARRETT_64) || defined(PLANTARD_64)

namespace barrett64_cpu
{  // It does not work  modulus higher than 62 bit.

class BarrettOperations
{
   public:
    // Modular Addition for 64 bit
    // result = (input1 + input2) % modulus
    static __host__ Data add(Data& input1, Data& input2, Modulus& modulus)
    {
        Data sum = input1 + input2;
        sum = (sum >= modulus.value) ? (sum - modulus.value) : sum;

        return sum;
    }

    // Modular Substraction for 64 bit
    // result = (input1 - input2) % modulus
    static __host__ Data sub(Data& input1, Data& input2, Modulus& modulus)
    {
        Data dif = input1 + modulus.value;
        dif = dif - input2;
        dif = (dif >= modulus.value) ? (dif - modulus.value) : dif;

        return dif;
    }

    // Modular Multiplication for 64 bit
    // result = (input1 * input2) % modulus
    static __host__ Data mult(Data& input1, Data& input2, Modulus& modulus)
    {
        __uint128_t mult = (__uint128_t)input1 * (__uint128_t)input2;

        __uint128_t r = mult >> (modulus.bit - 2);
        r = r * (__uint128_t)modulus.mu;
        r = r >> (modulus.bit + 3);
        r = r * (__uint128_t)modulus.value;
        mult = mult - r;

        Data result = uint64_t(mult & UINT64_MAX);

        if (result >= modulus.value)
        {
            result -= modulus.value;
        }

        return result;
    }

    // Modular Exponentiation for 64 bit
    // result = (base ^ exponent) % modulus
    static __host__ Data exp(Data& base, Data& exponent, Modulus& modulus)
    {
        // with window method
        unsigned long long result = 1;

        int modulus_bit = log2(modulus.value) + 1;
        for (int i = modulus_bit - 1; i > -1; i--)
        {
            result =
                barrett64_cpu::BarrettOperations::mult(result, result, modulus);
            if (((exponent >> i) & 1u))
            {
                result = barrett64_cpu::BarrettOperations::mult(result, base,
                                                                modulus);
            }
        }

        return result;
    }

    // Modular Modular Inverse for 64 bit
    // result = (input ^ (-1)) % modulus
    static __host__ Data modinv(Data& input, Modulus& modulus)
    {
        Data index = modulus.value - 2;
        return barrett64_cpu::BarrettOperations::exp(input, index, modulus);
    }
};

}  // namespace barrett64_cpu

#elif defined(GOLDILOCKS_64)

namespace goldilocks64_cpu
{  // It does not work  modulus higher than 62 bit.

class GoldilocksOperations
{
   public:
    // Modular Addition for 64 bit
    // result = (input1 + input2) % modulus
    static __host__ Data add(Data& input1, Data& input2, Modulus& modulus)
    {
        __uint128_t sum = (__uint128_t)input1 + (__uint128_t)input2;
        sum = (sum >= modulus.value) ? (sum - modulus.value) : sum;

        return uint64_t(sum & UINT64_MAX);
    }

    // Modular Substraction for 64 bit
    // result = (input1 - input2) % modulus
    static __host__ Data sub(Data& input1, Data& input2, Modulus& modulus)
    {
        __uint128_t dif = (__uint128_t)input1 + (__uint128_t)modulus.value;
        dif = dif - input2;
        dif = (dif >= modulus.value) ? (dif - modulus.value) : dif;

        return uint64_t(dif & UINT64_MAX);
    }

    // Modular Multiplication for 64 bit
    // result = (input1 * input2) % modulus
    static __host__ Data mult(Data& input1, Data& input2, Modulus& modulus)
    {
        __uint128_t mult = (__uint128_t)input1 * (__uint128_t)input2;

        unsigned long long lo = uint64_t(mult & UINT64_MAX);
        unsigned long long hi = uint64_t(mult >> 64u);
        unsigned long long hiL = uint64_t(hi & UINT32_MAX);
        unsigned long long hiH = uint64_t(hi >> 32u);

        __uint128_t pre_result =
            (__uint128_t)(hiL * 4294967295) + (__uint128_t)lo;

        if (pre_result < hiH)
        {
            pre_result = pre_result + modulus.value;
            pre_result = pre_result - hiH;
            return uint64_t(pre_result & UINT64_MAX);
        }
        else
        {
            pre_result = pre_result - hiH;
            if (pre_result >= modulus.value)
            {
                pre_result = pre_result - modulus.value;
            }
            return uint64_t(pre_result & UINT64_MAX);
        }
    }

    // Modular Exponentiation for 64 bit
    // result = (base ^ exponent) % modulus
    static __host__ Data exp(Data& base, Data& exponent, Modulus& modulus)
    {
        // with window method
        unsigned long long result = 1;

        int modulus_bit = log2(modulus.value) + 1;
        for (int i = modulus_bit - 1; i > -1; i--)
        {
            result = goldilocks64_cpu::GoldilocksOperations::mult(result, result,
                                                                modulus);
            if (((exponent >> i) & 1u))
            {
                result = goldilocks64_cpu::GoldilocksOperations::mult(
                    result, base, modulus);
            }
        }

        return result;
    }

    // Modular Modular Inverse for 64 bit
    // result = (input ^ (-1)) % modulus
    static __host__ Data modinv(Data& input, Modulus& modulus)
    {
        Data index = modulus.value - 2;
        return goldilocks64_cpu::GoldilocksOperations::exp(input, index, modulus);
    }
};

}  // namespace goldilocks64_cpu

#else
#error "Please define reduction type."
#endif

#if defined(BARRETT_64) || defined(PLANTARD_64)
typedef barrett64_cpu::BarrettOperations VALUE;
#elif defined(GOLDILOCKS_64)
typedef goldilocks64_cpu::GoldilocksOperations VALUE;
#else
#error "Please define reduction type."
#endif

#ifdef BARRETT_64
namespace barrett64_gpu
{
class BarrettOperations
{
   private:
    class uint128_t
    {
       public:
        // x -> LSB side
        // y -> MSB side
        ulonglong2 value;

        __device__ __forceinline__ uint128_t()
        {
            value.x = 0;
            value.y = 0;
        }

        __device__ __forceinline__ uint128_t(const uint64_t& input)
        {
            value.x = input;
            value.y = 0;
        }

        __device__ __forceinline__ void operator=(const uint128_t& input)
        {
            value.x = input.value.x;
            value.y = input.value.y;
        }

        __device__ __forceinline__ void operator=(const uint64_t& input)
        {
            value.x = input;
            value.y = 0;
        }

        __device__ __forceinline__ uint128_t operator<<(const unsigned& shift)
        {
            uint128_t result;

            result.value.y = value.y << shift;
            result.value.y = (value.x >> (64 - shift)) | result.value.y;
            result.value.x = value.x << shift;

            return result;
        }

        __device__ __forceinline__ uint128_t operator>>(const unsigned& shift)
        {
            uint128_t result;

            result.value.x = value.x >> shift;
            result.value.x = (value.y << (64 - shift)) | result.value.x;
            result.value.y = value.y >> shift;

            return result;
        }

        __device__ __forceinline__ uint128_t operator-(uint128_t& other)
        {
            uint128_t result;

            asm("{\n\t"
                "sub.cc.u64      %1, %3, %5;    \n\t"
                "subc.u64        %0, %2, %4;    \n\t"
                "}"
                : "=l"(result.value.y), "=l"(result.value.x)
                : "l"(value.y), "l"(value.x), "l"(other.value.y),
                  "l"(other.value.x));

            return result;
        }

        __device__ __forceinline__ uint128_t operator-=(const uint128_t& other)
        {
            uint128_t result;
            asm("{\n\t"
                "sub.cc.u64      %1, %3, %5;    \n\t"
                "subc.u64        %0, %2, %4;    \n\t"
                "}"
                : "=l"(result.value.y), "=l"(result.value.x)
                : "l"(value.y), "l"(value.x), "l"(other.value.y),
                  "l"(other.value.x));

            return result;
        }
    };

   public:
    // Modular Addition for 64 bit
    // result = (input1 + input2) % modulus
    static __device__ Data add(Data& input1, Data& input2, Modulus& modulus)
    {
        Data sum = input1 + input2;
        sum = (sum >= modulus.value) ? (sum - modulus.value) : sum;

        return sum;
    }

    // Modular Substraction for 64 bit
    // result = (input1 - input2) % modulus
    static __device__ Data sub(Data& input1, Data& input2, Modulus& modulus)
    {
        Data dif = input1 + modulus.value;
        dif = dif - input2;
        dif = (dif >= modulus.value) ? (dif - modulus.value) : dif;

        return dif;
    }

    static __device__ uint128_t mult128(const unsigned long long& a,
                                        const unsigned long long& b)
    {
        uint128_t result;

        asm("{\n\t"
            "mul.lo.u64      %1, %2, %3;    \n\t"
            "mul.hi.u64      %0, %2, %3;    \n\t"

            "}"
            : "=l"(result.value.y), "=l"(result.value.x)
            : "l"(a), "l"(b));

        return result;
    }

    // Modular Multiplication for 64 bit
    // result = (input1 * input2) % modulus
    static __device__ Data mult(Data& input1, Data& input2, Modulus& modulus)
    {
        uint128_t z = mult128(input1, input2);

        uint128_t w = z >> (modulus.bit - 2);

        w = mult128(w.value.x, modulus.mu);

        w = w >> (modulus.bit + 3);

        w = mult128(w.value.x, modulus.value);

        z = z - w;

        return (z.value.x >= modulus.value) ? (z.value.x -= modulus.value)
                                            : (z.value.x);
    }
};

}  // namespace barrett64_gpu

#elif defined(GOLDILOCKS_64)

namespace goldilocks64_gpu
{
struct uint128_t
{
   public:
    // x -> LSB side
    // y -> MSB side
    // ulonglong2 value;
    unsigned long long x, y;

    __device__ __forceinline__ uint128_t()
    {
        x = 0;
        y = 0;
    }

    __device__ __forceinline__ uint128_t(const uint64_t& input)
    {
        x = input;
        y = 0;
    }

    __device__ __forceinline__ void operator=(const uint128_t& input)
    {
        x = input.x;
        y = input.y;
    }

    __device__ __forceinline__ void operator=(const uint64_t& input)
    {
        x = input;
        y = 0;
    }

    __device__ __forceinline__ uint128_t operator<<(const unsigned& shift)
    {
        uint128_t result;

        result.y = y << shift;
        result.y = (x >> (64 - shift)) | result.y;
        result.x = x << shift;

        return result;
    }

    __device__ __forceinline__ uint128_t operator>>(const unsigned& shift)
    {
        uint128_t result;

        result.x = x >> shift;
        result.x = (y << (64 - shift)) | result.x;
        result.y = y >> shift;

        return result;
    }
};

__device__ __forceinline__ uint128_t operator-(uint128_t& input1,
                                               const uint128_t& input2)
{
    uint128_t result;

    asm("{\n\t"
        "sub.cc.u64      %1, %3, %5;    \n\t"
        "subc.u64        %0, %2, %4;    \n\t"
        "}"
        : "=l"(result.y), "=l"(result.x)
        : "l"(input1.y), "l"(input1.x), "l"(input2.y), "l"(input2.x));

    return result;
}

__device__ __forceinline__ uint128_t operator-(uint128_t& input1,
                                               const Data& input2)
{
    uint128_t result;

    asm("{\n\t"
        "sub.cc.u64      %1, %3, %5;    \n\t"
        "subc.u64        %0, %2, %4;    \n\t"
        "}"
        : "=l"(result.y), "=l"(result.x)
        : "l"(input1.y), "l"(input1.x), "l"((Data)0), "l"(input2));

    return result;
}

__device__ __forceinline__ void operator-=(uint128_t& input1,
                                           const uint128_t& input2)
{
    asm("{\n\t"
        "sub.cc.u64      %1, %3, %5;    \n\t"
        "subc.u64        %0, %2, %4;    \n\t"
        "}"
        : "=l"(input1.y), "=l"(input1.x)
        : "l"(input1.y), "l"(input1.x), "l"(input2.y), "l"(input2.x));
}

__device__ __forceinline__ void operator-=(uint128_t& input1,
                                           const Data& input2)
{
    asm("{\n\t"
        "sub.cc.u64      %1, %3, %5;    \n\t"
        "subc.u64        %0, %2, %4;    \n\t"
        "}"
        : "=l"(input1.y), "=l"(input1.x)
        : "l"(input1.y), "l"(input1.x), "l"((Data)0), "l"(input2));
}

__device__ __forceinline__ uint128_t operator+(uint128_t& input1,
                                               const uint128_t& input2)
{
    uint128_t result;

    result.x = input1.x + input2.x;
    result.y = input1.y + input2.y + (result.x < input1.x);

    return result;
}

__device__ __forceinline__ uint128_t operator+(uint128_t& input1,
                                               const Data& input2)
{
    uint128_t result;

    result.x = input1.x + input2;
    result.y = input1.y + (result.x < input1.x);

    return result;
}

__device__ __forceinline__ bool operator>=(uint128_t& input1,
                                           const uint128_t& input2)
{
    if (input1.y > input2.y)
        return true;
    else if (input1.y < input2.y)
        return false;
    else if (input1.x >= input2.x)
        return true;
    else
        return false;
}

__device__ __forceinline__ bool operator>=(uint128_t& input1,
                                           const Data& input2)
{
    if (input1.y > 0)
        return true;
    else if (input1.x >= input2)
        return true;
    else
        return false;
}

struct GoldilocksOperations
{
   public:
    // Modular Addition for 64 bit
    // result = (input1 + input2) % modulus
    __device__ __forceinline__ static Data add(Data& input1, Data& input2,
                                               Modulus& modulus)
    {
        // uint128_t sum = (uint128_t)input1 + input2;
        // sum = (sum >= modulus.value) ? (sum - modulus.value) : sum;

        uint128_t sum = input1;
        sum = sum + input2;
        // sum = (sum >= modulus.value) ? (sum - modulus.value) : sum;
        sum = sum - ((sum >= modulus.value) * modulus.value);

        return sum.x;
    }

    // Modular Substraction for 64 bit
    // result = (input1 - input2) % modulus
    __device__ __forceinline__ static Data sub(Data& input1, Data& input2,
                                               Modulus& modulus)
    {
        uint128_t dif = input1;
        dif = dif + modulus.value;

        dif = dif - input2;
        // dif = (dif >= modulus.value) ? (dif - modulus.value) : dif;
        dif = dif - ((dif >= modulus.value) * modulus.value);

        return dif.x;
    }

    // Modular Multiplication for 64 bit
    // result = (input1 * input2) % modulus
    __device__ __forceinline__ static Data mult(Data& input1, Data& input2,
                                                Modulus& modulus)
    {
        // --------------------- //
        // Authors: Alisah Ozcan
        // --------------------- //
        uint128_t result;
        unsigned long long q = 18446744069414584321;

        asm("{\n\t"
            ".reg .u64      r0, r1;         \n\t"
            ".reg .u32      r2, r3;         \n\t"
            ".reg .u64      r5;         \n\t"

            ".reg .pred p;                   \n\t"

            "mul.lo.u64      r1, %2, %3;    \n\t"
            "mul.hi.u64      r0, %2, %3;    \n\t"

            "mov.b64        {r3,r2}, r0;    \n\t"
            "mov.b64        r5, {r2, 0};    \n\t"

            "not.b64        r5, r5;    \n\t"

            "mul.wide.u32      r0, %4, r3;    \n\t"

            "add.cc.u64     %1, r1, r0;     \n\t"
            "addc.u64       %0, 0, 0;      \n\t"

            "add.u64     r5, r5, 1;     \n\t"
            "add.u64     %1, %1, r5;     \n\t"

            "and.b64       %0, %0, 0x00000001;      \n\t"

            // if case
            "setp.ne.u64     p, 1, %0;     \n\t"
            "@p bra L1;                    \n\t"
            "sub.cc.u64      %1, %1, %5;   \n\t"
            "L1:                           \n\t"

            "}"
            : "=l"(result.y), "=l"(result.x)
            : "l"(input1), "l"(input2), "r"(unsigned(4294967295)), "l"(q));

        return result.x;
    }
};

}  // namespace goldilocks64_gpu

#elif defined(PLANTARD_64)

namespace plantard64_gpu
{
class PlantardOperations
{
   private:
    class uint128_t
    {
       public:
        // x -> LSB side
        // y -> MSB side
        ulonglong2 value;

        __device__ __forceinline__ uint128_t()
        {
            value.x = 0;
            value.y = 0;
        }

        __device__ __forceinline__ uint128_t(const uint64_t& input)
        {
            value.x = input;
            value.y = 0;
        }

        __device__ __forceinline__ void operator=(const uint128_t& input)
        {
            value.x = input.value.x;
            value.y = input.value.y;
        }

        __device__ __forceinline__ void operator=(const uint64_t& input)
        {
            value.x = input;
            value.y = 0;
        }

        __device__ __forceinline__ uint128_t operator<<(const unsigned& shift)
        {
            uint128_t result;

            result.value.y = value.y << shift;
            result.value.y = (value.x >> (64 - shift)) | result.value.y;
            result.value.x = value.x << shift;

            return result;
        }

        __device__ __forceinline__ uint128_t operator>>(const unsigned& shift)
        {
            uint128_t result;

            result.value.x = value.x >> shift;
            result.value.x = (value.y << (64 - shift)) | result.value.x;
            result.value.y = value.y >> shift;

            return result;
        }

        __device__ __forceinline__ uint128_t operator-(uint128_t& other)
        {
            uint128_t result;

            asm("{\n\t"
                "sub.cc.u64      %1, %3, %5;    \n\t"
                "subc.u64        %0, %2, %4;    \n\t"
                "}"
                : "=l"(result.value.y), "=l"(result.value.x)
                : "l"(value.y), "l"(value.x), "l"(other.value.y),
                  "l"(other.value.x));

            return result;
        }

        __device__ __forceinline__ uint128_t operator-=(const uint128_t& other)
        {
            uint128_t result;
            asm("{\n\t"
                "sub.cc.u64      %1, %3, %5;    \n\t"
                "subc.u64        %0, %2, %4;    \n\t"
                "}"
                : "=l"(result.value.y), "=l"(result.value.x)
                : "l"(value.y), "l"(value.x), "l"(other.value.y),
                  "l"(other.value.x));

            return result;
        }
    };

   public:
    // Modular Addition for 64 bit
    // result = (input1 + input2) % modulus
    static __device__ Data add(Data& input1, Data& input2, Modulus& modulus)
    {
        Data sum = input1 + input2;
        sum = (sum >= modulus.value) ? (sum - modulus.value) : sum;

        return sum;
    }

    // Modular Substraction for 64 bit
    // result = (input1 - input2) % modulus
    static __device__ Data sub(Data& input1, Data& input2, Modulus& modulus)
    {
        Data dif = input1 + modulus.value;
        dif = dif - input2;
        dif = (dif >= modulus.value) ? (dif - modulus.value) : dif;

        return dif;
    }

    // Modular Multiplication for 64 bit
    // result = (input1 * input2) % modulus
    static __device__ Data mult(Data& input1, ulonglong2& input2,
                                Modulus& modulus)
    {
        // --------------------- //
        // Authors: Alisah Ozcan
        // --------------------- //

        Data result;
        asm("{\n\t"
            //"mul.hi.u64      %0, %1, %2;    \n\t"
            "mad.lo.u64      %0, %1, %3, 1;    \n\t"
            "mad.hi.u64      %0, %1, %2, %0;  \n\t"

            "mul.hi.u64      %0, %0, %4;   \n\t"

            "}"
            : "=l"(result)
            : "l"(input1), "l"(input2.x), "l"(input2.y), "l"(modulus.value));

        return result;
    }
};

}  // namespace plantard64_gpu

#else
#error "Please define reduction type."
#endif

#ifdef BARRETT_64
typedef barrett64_gpu::BarrettOperations VALUE_GPU;
#elif defined(GOLDILOCKS_64)
typedef goldilocks64_gpu::GoldilocksOperations VALUE_GPU;
#elif defined(PLANTARD_64)
typedef plantard64_gpu::PlantardOperations VALUE_GPU;
#else
#error "Please define reduction type."
#endif

#endif  // MODULAR_ARITHMATIC_H
