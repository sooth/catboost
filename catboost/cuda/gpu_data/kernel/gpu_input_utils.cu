#include "gpu_input_utils.cuh"
#include "gpu_cityhash_device.cuh"

#include <catboost/cuda/cuda_lib/memcpy_tracker.h>

#include <cub/device/device_reduce.cuh>

namespace NKernel {
    namespace {
        using namespace NCityHashDevice;

        template <typename T>
        __global__ void CopyStridedCastToFloatImpl(
            const char* __restrict src,
            ui64 strideBytes,
            ui32 size,
            float* __restrict dst
        ) {
            const ui32 i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < size) {
                const auto* ptr = reinterpret_cast<const T*>(src + static_cast<ui64>(i) * strideBytes);
                dst[i] = static_cast<float>(*ptr);
            }
        }

        template <typename T>
        __global__ void HashStridedSignedToCatHashImpl(
            const char* __restrict src,
            ui64 strideBytes,
            ui32 size,
            ui32* __restrict dst
        ) {
            const ui32 i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < size) {
                const auto* ptr = reinterpret_cast<const T*>(src + static_cast<ui64>(i) * strideBytes);
                char buf[32];
                const int len = CityI64ToDecString(static_cast<i64>(*ptr), buf);
                const ui64 h = CityHash64Len0to32(buf, static_cast<ui32>(len));
                dst[i] = static_cast<ui32>(h & 0xffffffffULL);
            }
        }

        template <typename T>
        __global__ void HashStridedUnsignedToCatHashImpl(
            const char* __restrict src,
            ui64 strideBytes,
            ui32 size,
            ui32* __restrict dst
        ) {
            const ui32 i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < size) {
                const auto* ptr = reinterpret_cast<const T*>(src + static_cast<ui64>(i) * strideBytes);
                char buf[32];
                const int len = CityU64ToDecString(static_cast<ui64>(*ptr), buf);
                const ui64 h = CityHash64Len0to32(buf, static_cast<ui32>(len));
                dst[i] = static_cast<ui32>(h & 0xffffffffULL);
            }
        }

        template <typename T>
        __global__ void MapStridedSignedCodesToCatHashImpl(
            const char* __restrict src,
            ui64 strideBytes,
            ui32 size,
            const ui32* __restrict dict,
            ui32 dictSize,
            ui32 nullValue,
            ui32* __restrict dst
        ) {
            const ui32 i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < size) {
                const auto* ptr = reinterpret_cast<const T*>(src + static_cast<ui64>(i) * strideBytes);
                const i64 code = static_cast<i64>(*ptr);
                if ((code < 0) || (static_cast<ui64>(code) >= static_cast<ui64>(dictSize))) {
                    dst[i] = nullValue;
                } else {
                    dst[i] = dict[static_cast<ui32>(code)];
                }
            }
        }

        template <typename T>
        __global__ void MapStridedUnsignedCodesToCatHashImpl(
            const char* __restrict src,
            ui64 strideBytes,
            ui32 size,
            const ui32* __restrict dict,
            ui32 dictSize,
            ui32 nullValue,
            ui32* __restrict dst
        ) {
            const ui32 i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < size) {
                const auto* ptr = reinterpret_cast<const T*>(src + static_cast<ui64>(i) * strideBytes);
                const ui64 code = static_cast<ui64>(*ptr);
                if (code >= static_cast<ui64>(dictSize)) {
                    dst[i] = nullValue;
                } else {
                    dst[i] = dict[static_cast<ui32>(code)];
                }
            }
        }

        template <typename T>
        inline void LaunchCopyStridedCastToFloat(
            const void* src,
            ui64 srcStrideBytes,
            ui32 size,
            float* dst,
            TCudaStream stream
        ) {
            const ui32 blockSize = 256;
            const ui32 numBlocks = (size + blockSize - 1) / blockSize;
            CopyStridedCastToFloatImpl<T><<<numBlocks, blockSize, 0, stream>>>(
                reinterpret_cast<const char*>(src),
                srcStrideBytes,
                size,
                dst
            );
        }

        template <typename T, bool IsSigned>
        inline void LaunchHashStridedToCatHash(
            const void* src,
            ui64 srcStrideBytes,
            ui32 size,
            ui32* dst,
            TCudaStream stream
        ) {
            const ui32 blockSize = 256;
            const ui32 numBlocks = (size + blockSize - 1) / blockSize;
            if (IsSigned) {
                HashStridedSignedToCatHashImpl<T><<<numBlocks, blockSize, 0, stream>>>(
                    reinterpret_cast<const char*>(src),
                    srcStrideBytes,
                    size,
                    dst
                );
            } else {
                HashStridedUnsignedToCatHashImpl<T><<<numBlocks, blockSize, 0, stream>>>(
                    reinterpret_cast<const char*>(src),
                    srcStrideBytes,
                    size,
                    dst
                );
            }
        }

        template <typename T, bool IsSigned>
        inline void LaunchMapStridedCatCodesToCatHash(
            const void* src,
            ui64 srcStrideBytes,
            ui32 size,
            const ui32* dict,
            ui32 dictSize,
            ui32 nullValue,
            ui32* dst,
            TCudaStream stream
        ) {
            const ui32 blockSize = 256;
            const ui32 numBlocks = (size + blockSize - 1) / blockSize;
            if (IsSigned) {
                MapStridedSignedCodesToCatHashImpl<T><<<numBlocks, blockSize, 0, stream>>>(
                    reinterpret_cast<const char*>(src),
                    srcStrideBytes,
                    size,
                    dict,
                    dictSize,
                    nullValue,
                    dst
                );
            } else {
                MapStridedUnsignedCodesToCatHashImpl<T><<<numBlocks, blockSize, 0, stream>>>(
                    reinterpret_cast<const char*>(src),
                    srcStrideBytes,
                    size,
                    dict,
                    dictSize,
                    nullValue,
                    dst
                );
            }
        }
    }

    void CopyStridedGpuInputToFloat(
        const void* src,
        ui64 srcStrideBytes,
        ui32 size,
        EGpuInputDType dtype,
        float* dst,
        TCudaStream stream
    ) {
        if (size == 0) {
            return;
        }
        switch (dtype) {
            case EGpuInputDType::Float32: {
                CUDA_SAFE_CALL(cudaMemcpy2DAsync(
                    /*dst*/ dst,
                    /*dpitch*/ sizeof(float),
                    /*src*/ src,
                    /*spitch*/ srcStrideBytes,
                    /*width*/ sizeof(float),
                    /*height*/ size,
                    cudaMemcpyDeviceToDevice,
                    stream
                ));
                break;
            }
            case EGpuInputDType::Float64:
                LaunchCopyStridedCastToFloat<double>(src, srcStrideBytes, size, dst, stream);
                break;
            case EGpuInputDType::Int8:
                LaunchCopyStridedCastToFloat<i8>(src, srcStrideBytes, size, dst, stream);
                break;
            case EGpuInputDType::Int16:
                LaunchCopyStridedCastToFloat<i16>(src, srcStrideBytes, size, dst, stream);
                break;
            case EGpuInputDType::Int32:
                LaunchCopyStridedCastToFloat<i32>(src, srcStrideBytes, size, dst, stream);
                break;
            case EGpuInputDType::Int64:
                LaunchCopyStridedCastToFloat<i64>(src, srcStrideBytes, size, dst, stream);
                break;
            case EGpuInputDType::UInt8:
                LaunchCopyStridedCastToFloat<ui8>(src, srcStrideBytes, size, dst, stream);
                break;
            case EGpuInputDType::UInt16:
                LaunchCopyStridedCastToFloat<ui16>(src, srcStrideBytes, size, dst, stream);
                break;
            case EGpuInputDType::UInt32:
                LaunchCopyStridedCastToFloat<ui32>(src, srcStrideBytes, size, dst, stream);
                break;
            case EGpuInputDType::UInt64:
                LaunchCopyStridedCastToFloat<ui64>(src, srcStrideBytes, size, dst, stream);
                break;
            case EGpuInputDType::Bool:
                LaunchCopyStridedCastToFloat<ui8>(src, srcStrideBytes, size, dst, stream);
                break;
        }
    }

    void HashStridedGpuInputToCatHash(
        const void* src,
        ui64 srcStrideBytes,
        ui32 size,
        EGpuInputDType dtype,
        ui32* dst,
        TCudaStream stream
    ) {
        if (size == 0) {
            return;
        }
        switch (dtype) {
            case EGpuInputDType::Int8:
                LaunchHashStridedToCatHash<i8, true>(src, srcStrideBytes, size, dst, stream);
                break;
            case EGpuInputDType::Int16:
                LaunchHashStridedToCatHash<i16, true>(src, srcStrideBytes, size, dst, stream);
                break;
            case EGpuInputDType::Int32:
                LaunchHashStridedToCatHash<i32, true>(src, srcStrideBytes, size, dst, stream);
                break;
            case EGpuInputDType::Int64:
                LaunchHashStridedToCatHash<i64, true>(src, srcStrideBytes, size, dst, stream);
                break;
            case EGpuInputDType::UInt8:
                LaunchHashStridedToCatHash<ui8, false>(src, srcStrideBytes, size, dst, stream);
                break;
            case EGpuInputDType::UInt16:
                LaunchHashStridedToCatHash<ui16, false>(src, srcStrideBytes, size, dst, stream);
                break;
            case EGpuInputDType::UInt32:
                LaunchHashStridedToCatHash<ui32, false>(src, srcStrideBytes, size, dst, stream);
                break;
            case EGpuInputDType::UInt64:
                LaunchHashStridedToCatHash<ui64, false>(src, srcStrideBytes, size, dst, stream);
                break;
            default:
                // Unsupported dtype for categorical hashing is a hard error.
                CUDA_SAFE_CALL(cudaErrorInvalidValue);
                break;
        }
    }

    void MapStridedCatCodesToCatHash(
        const void* src,
        ui64 srcStrideBytes,
        ui32 size,
        EGpuInputDType dtype,
        const ui32* dict,
        ui32 dictSize,
        ui32 nullValue,
        ui32* dst,
        TCudaStream stream
    ) {
        if (size == 0) {
            return;
        }
        switch (dtype) {
            case EGpuInputDType::Int8:
                LaunchMapStridedCatCodesToCatHash<i8, true>(src, srcStrideBytes, size, dict, dictSize, nullValue, dst, stream);
                break;
            case EGpuInputDType::Int16:
                LaunchMapStridedCatCodesToCatHash<i16, true>(src, srcStrideBytes, size, dict, dictSize, nullValue, dst, stream);
                break;
            case EGpuInputDType::Int32:
                LaunchMapStridedCatCodesToCatHash<i32, true>(src, srcStrideBytes, size, dict, dictSize, nullValue, dst, stream);
                break;
            case EGpuInputDType::Int64:
                LaunchMapStridedCatCodesToCatHash<i64, true>(src, srcStrideBytes, size, dict, dictSize, nullValue, dst, stream);
                break;
            case EGpuInputDType::UInt8:
                LaunchMapStridedCatCodesToCatHash<ui8, false>(src, srcStrideBytes, size, dict, dictSize, nullValue, dst, stream);
                break;
            case EGpuInputDType::UInt16:
                LaunchMapStridedCatCodesToCatHash<ui16, false>(src, srcStrideBytes, size, dict, dictSize, nullValue, dst, stream);
                break;
            case EGpuInputDType::UInt32:
                LaunchMapStridedCatCodesToCatHash<ui32, false>(src, srcStrideBytes, size, dict, dictSize, nullValue, dst, stream);
                break;
            case EGpuInputDType::UInt64:
                LaunchMapStridedCatCodesToCatHash<ui64, false>(src, srcStrideBytes, size, dict, dictSize, nullValue, dst, stream);
                break;
            default:
                // Caller is expected to validate categorical input dtype.
                break;
        }
    }

    void ComputeMinMaxToHost(
        const float* values,
        ui32 size,
        float* minValue,
        float* maxValue,
        TCudaStream stream
    ) {
        if (size == 0) {
            if (minValue) {
                *minValue = 0.0f;
            }
            if (maxValue) {
                *maxValue = 0.0f;
            }
            return;
        }

        float* dMin = nullptr;
        float* dMax = nullptr;
        CUDA_SAFE_CALL(cudaMalloc(&dMin, sizeof(float)));
        CUDA_SAFE_CALL(cudaMalloc(&dMax, sizeof(float)));

        size_t tmpBytesMin = 0;
        size_t tmpBytesMax = 0;
        CUDA_SAFE_CALL(cub::DeviceReduce::Min(nullptr, tmpBytesMin, values, dMin, static_cast<int>(size), stream));
        CUDA_SAFE_CALL(cub::DeviceReduce::Max(nullptr, tmpBytesMax, values, dMax, static_cast<int>(size), stream));

        size_t tmpBytes = (tmpBytesMin > tmpBytesMax) ? tmpBytesMin : tmpBytesMax;
        void* tmp = nullptr;
        CUDA_SAFE_CALL(cudaMalloc(&tmp, tmpBytes));

        CUDA_SAFE_CALL(cub::DeviceReduce::Min(tmp, tmpBytes, values, dMin, static_cast<int>(size), stream));
        CUDA_SAFE_CALL(cub::DeviceReduce::Max(tmp, tmpBytes, values, dMax, static_cast<int>(size), stream));

        if (minValue) {
            NCudaLib::TMemcpyTracker::Instance().RecordMemcpyAsync(minValue, dMin, sizeof(float), cudaMemcpyDeviceToHost);
            CUDA_SAFE_CALL(cudaMemcpyAsync(minValue, dMin, sizeof(float), cudaMemcpyDeviceToHost, stream));
        }
        if (maxValue) {
            NCudaLib::TMemcpyTracker::Instance().RecordMemcpyAsync(maxValue, dMax, sizeof(float), cudaMemcpyDeviceToHost);
            CUDA_SAFE_CALL(cudaMemcpyAsync(maxValue, dMax, sizeof(float), cudaMemcpyDeviceToHost, stream));
        }
        CUDA_SAFE_CALL(cudaStreamSynchronize(stream));

        CUDA_SAFE_CALL(cudaFree(tmp));
        CUDA_SAFE_CALL(cudaFree(dMin));
        CUDA_SAFE_CALL(cudaFree(dMax));
    }

}
