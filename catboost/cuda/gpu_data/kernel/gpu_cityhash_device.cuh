#pragma once

#include <util/system/types.h>

// Device-side CityHash64 (v1) implementation, compatible with util/digest/city.cpp for lengths <= 32.
// Used to compute CalcCatFeatureHash(ToString(value)) for integer categorical values on GPU.
// Shared between gpu_input_factorize.cu and gpu_input_utils.cu.

namespace NKernel {
    namespace NCityHashDevice {

        __device__ __forceinline__ ui64 CityRotate(ui64 val, int shift) {
            return shift == 0 ? val : ((val >> shift) | (val << (64 - shift)));
        }

        __device__ __forceinline__ ui64 CityRotateByAtLeast1(ui64 val, int shift) {
            return (val >> shift) | (val << (64 - shift));
        }

        __device__ __forceinline__ ui64 CityShiftMix(ui64 val) {
            return val ^ (val >> 47);
        }

        __device__ __forceinline__ ui64 CityHash128to64(ui64 low, ui64 high) {
            // Murmur-inspired hashing (matches Hash128to64 in util/digest/city.h).
            const ui64 kMul = 0x9ddfea08eb382d69ULL;
            ui64 a = (low ^ high) * kMul;
            a ^= (a >> 47);
            ui64 b = (high ^ a) * kMul;
            b ^= (b >> 47);
            b *= kMul;
            return b;
        }

        __device__ __forceinline__ ui64 CityHashLen16(ui64 u, ui64 v) {
            return CityHash128to64(u, v);
        }

        __device__ __forceinline__ ui64 CityFetch64(const char* p) {
            // Little-endian unaligned load.
            ui64 result = 0;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                result |= (static_cast<ui64>(static_cast<unsigned char>(p[i])) << (8 * i));
            }
            return result;
        }

        __device__ __forceinline__ ui32 CityFetch32(const char* p) {
            ui32 result = 0;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                result |= (static_cast<ui32>(static_cast<unsigned char>(p[i])) << (8 * i));
            }
            return result;
        }

        __device__ __forceinline__ ui64 CityHashLen0to16(const char* s, ui32 len) {
            constexpr ui64 k2 = 0x9ae16a3b2f90404fULL;
            constexpr ui64 k3 = 0xc949d7c7509e6557ULL;

            if (len > 8) {
                const ui64 a = CityFetch64(s);
                const ui64 b = CityFetch64(s + len - 8);
                return CityHashLen16(a, CityRotateByAtLeast1(b + len, static_cast<int>(len))) ^ b;
            }
            if (len >= 4) {
                const ui64 a = CityFetch32(s);
                return CityHashLen16(static_cast<ui64>(len) + (a << 3), CityFetch32(s + len - 4));
            }
            if (len > 0) {
                const ui8 a = static_cast<ui8>(s[0]);
                const ui8 b = static_cast<ui8>(s[len >> 1]);
                const ui8 c = static_cast<ui8>(s[len - 1]);
                const ui32 y = static_cast<ui32>(a) + (static_cast<ui32>(b) << 8);
                const ui32 z = static_cast<ui32>(len) + (static_cast<ui32>(c) << 2);
                return CityShiftMix(static_cast<ui64>(y) * k2 ^ static_cast<ui64>(z) * k3) * k2;
            }
            return k2;
        }

        __device__ __forceinline__ ui64 CityHashLen17to32(const char* s, ui32 len) {
            constexpr ui64 k0 = 0xc3a5c85c97cb3127ULL;
            constexpr ui64 k1 = 0xb492b66fbe98f273ULL;
            constexpr ui64 k2 = 0x9ae16a3b2f90404fULL;
            constexpr ui64 k3 = 0xc949d7c7509e6557ULL;

            const ui64 a = CityFetch64(s) * k1;
            const ui64 b = CityFetch64(s + 8);
            const ui64 c = CityFetch64(s + len - 8) * k2;
            const ui64 d = CityFetch64(s + len - 16) * k0;
            return CityHashLen16(
                CityRotate(a - b, 43) + CityRotate(c, 30) + d,
                a + CityRotate(b ^ k3, 20) - c + len
            );
        }

        __device__ __forceinline__ ui64 CityHash64Len0to32(const char* s, ui32 len) {
            if (len <= 16) {
                return CityHashLen0to16(s, len);
            }
            return CityHashLen17to32(s, len);
        }

        __device__ __forceinline__ int CityU64ToDecString(ui64 x, char* out) {
            char tmp[32];
            int len = 0;
            do {
                const ui64 q = x / 10;
                const ui32 digit = static_cast<ui32>(x - q * 10);
                tmp[len++] = static_cast<char>('0' + digit);
                x = q;
            } while (x != 0);

            for (int i = 0; i < len; ++i) {
                out[i] = tmp[len - 1 - i];
            }
            return len;
        }

        __device__ __forceinline__ int CityI64ToDecString(i64 v, char* out) {
            ui64 x = 0;
            int pos = 0;
            if (v < 0) {
                out[pos++] = '-';
                // Avoid overflow for INT64_MIN.
                x = static_cast<ui64>(-(v + 1)) + 1;
            } else {
                x = static_cast<ui64>(v);
            }
            pos += CityU64ToDecString(x, out + pos);
            return pos;
        }

    }  // namespace NCityHashDevice
}  // namespace NKernel
