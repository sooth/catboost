#include "gpu_input_factorize.cuh"
#include "gpu_cityhash_device.cuh"

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_run_length_encode.cuh>
#include <cub/device/device_scan.cuh>

#include <type_traits>

namespace NKernel {
    namespace {
        using namespace NCityHashDevice;

        // RAII wrapper for CUDA device memory to prevent leaks on allocation failure.
        // Similar to TCudaDeviceBuffer in gpu_input_quantization.cpp but local to this TU.
        template <typename T>
        class TCudaDeviceBuffer {
        public:
            TCudaDeviceBuffer() = default;

            explicit TCudaDeviceBuffer(size_t count) {
                if (count > 0) {
                    CUDA_SAFE_CALL(cudaMalloc(&Ptr_, count * sizeof(T)));
                }
            }

            ~TCudaDeviceBuffer() {
                if (Ptr_) {
                    CUDA_SAFE_CALL(cudaFree(Ptr_));
                }
            }

            // Non-copyable, non-movable (simple RAII for local scope).
            TCudaDeviceBuffer(const TCudaDeviceBuffer&) = delete;
            TCudaDeviceBuffer& operator=(const TCudaDeviceBuffer&) = delete;
            TCudaDeviceBuffer(TCudaDeviceBuffer&&) = delete;
            TCudaDeviceBuffer& operator=(TCudaDeviceBuffer&&) = delete;

            T* Get() { return Ptr_; }
            const T* Get() const { return Ptr_; }
            T** PtrAddr() { return &Ptr_; }

        private:
            T* Ptr_ = nullptr;
        };

        // Wrapper for untyped (void*) CUDA allocations (CUB workspace buffers).
        class TCudaWorkspace {
        public:
            TCudaWorkspace() = default;

            explicit TCudaWorkspace(size_t bytes) {
                if (bytes > 0) {
                    CUDA_SAFE_CALL(cudaMalloc(&Ptr_, bytes));
                }
            }

            ~TCudaWorkspace() {
                if (Ptr_) {
                    CUDA_SAFE_CALL(cudaFree(Ptr_));
                }
            }

            TCudaWorkspace(const TCudaWorkspace&) = delete;
            TCudaWorkspace& operator=(const TCudaWorkspace&) = delete;
            TCudaWorkspace(TCudaWorkspace&&) = delete;
            TCudaWorkspace& operator=(TCudaWorkspace&&) = delete;

            void* Get() { return Ptr_; }

        private:
            void* Ptr_ = nullptr;
        };

        // Kernel to XOR the sign bit of signed integers so that CUB radix sort
        // produces a correct numerical ordering. CUB treats keys as raw bitstrings,
        // so for signed types the sign bit (MSB) makes negatives sort after positives.
        // XORing the sign bit flips this, giving correct ascending order.
        template <typename T>
        __global__ void FlipSignBitImpl(T* __restrict data, ui32 size) {
            const ui32 i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < size) {
                using UnsignedT = typename std::make_unsigned<T>::type;
                constexpr UnsignedT signBit = UnsignedT(1) << (sizeof(T) * 8 - 1);
                data[i] = static_cast<T>(static_cast<UnsignedT>(data[i]) ^ signBit);
            }
        }

        template <typename T>
        inline void LaunchFlipSignBit(T* data, ui32 size, TCudaStream stream) {
            const ui32 blockSize = 256;
            const ui32 numBlocks = (size + blockSize - 1) / blockSize;
            FlipSignBitImpl<T><<<numBlocks, blockSize, 0, stream>>>(data, size);
        }

        template <typename T>
        __global__ void HashUniqueSignedImpl(const T* __restrict uniqueValues, ui32 uniqueCount, ui32* __restrict hashesOut) {
            const ui32 i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < uniqueCount) {
                char buf[32];
                const int len = CityI64ToDecString(static_cast<i64>(uniqueValues[i]), buf);
                const ui64 h = CityHash64Len0to32(buf, static_cast<ui32>(len));
                hashesOut[i] = static_cast<ui32>(h & 0xffffffffULL);
            }
        }

        template <typename T>
        __global__ void HashUniqueUnsignedImpl(const T* __restrict uniqueValues, ui32 uniqueCount, ui32* __restrict hashesOut) {
            const ui32 i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < uniqueCount) {
                char buf[32];
                const int len = CityU64ToDecString(static_cast<ui64>(uniqueValues[i]), buf);
                const ui64 h = CityHash64Len0to32(buf, static_cast<ui32>(len));
                hashesOut[i] = static_cast<ui32>(h & 0xffffffffULL);
            }
        }

        template <typename T>
        __global__ void CopyStridedImpl(
            const char* __restrict src,
            ui64 strideBytes,
            ui32 size,
            T* __restrict dst
        ) {
            const ui32 i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < size) {
                const auto* ptr = reinterpret_cast<const T*>(src + static_cast<ui64>(i) * strideBytes);
                dst[i] = *ptr;
            }
        }

        __global__ void FillSequenceImpl(ui32 size, ui32* __restrict dst) {
            const ui32 i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < size) {
                dst[i] = i;
            }
        }

        template <typename T>
        __global__ void ComputeChangeFlagsImpl(
            const T* __restrict sorted,
            ui32 size,
            ui32* __restrict flags
        ) {
            const ui32 i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < size) {
                if (i == 0) {
                    flags[i] = 0;
                } else {
                    flags[i] = (sorted[i] != sorted[i - 1]) ? 1u : 0u;
                }
            }
        }

        __global__ void ScatterRanksImpl(
            const ui32* __restrict sortedIdx,
            const ui32* __restrict runIdsSorted,
            ui32 size,
            ui32* __restrict ranksOut
        ) {
            const ui32 i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < size) {
                ranksOut[sortedIdx[i]] = runIdsSorted[i];
            }
        }

        template <typename T>
        inline void LaunchCopyStrided(
            const void* src,
            ui64 strideBytes,
            ui32 size,
            T* dst,
            TCudaStream stream
        ) {
            const ui32 blockSize = 256;
            const ui32 numBlocks = (size + blockSize - 1) / blockSize;
            CopyStridedImpl<T><<<numBlocks, blockSize, 0, stream>>>(
                reinterpret_cast<const char*>(src),
                strideBytes,
                size,
                dst
            );
        }

        inline void LaunchFillSequence(ui32 size, ui32* dst, TCudaStream stream) {
            const ui32 blockSize = 256;
            const ui32 numBlocks = (size + blockSize - 1) / blockSize;
            FillSequenceImpl<<<numBlocks, blockSize, 0, stream>>>(size, dst);
        }

        template <typename T>
        inline void LaunchComputeChangeFlags(const T* sorted, ui32 size, ui32* flags, TCudaStream stream) {
            const ui32 blockSize = 256;
            const ui32 numBlocks = (size + blockSize - 1) / blockSize;
            ComputeChangeFlagsImpl<T><<<numBlocks, blockSize, 0, stream>>>(sorted, size, flags);
        }

        inline void LaunchScatterRanks(const ui32* sortedIdx, const ui32* runIdsSorted, ui32 size, ui32* ranksOut, TCudaStream stream) {
            const ui32 blockSize = 256;
            const ui32 numBlocks = (size + blockSize - 1) / blockSize;
            ScatterRanksImpl<<<numBlocks, blockSize, 0, stream>>>(sortedIdx, runIdsSorted, size, ranksOut);
        }

        template <typename T>
        void FactorizeImpl(
            const void* src,
            ui64 strideBytes,
            ui32 size,
            ui32* ranksOut,
            T* uniqueValuesOut,
            ui32* countsOut,
            ui32* uniqueCountOut,
            TCudaStream stream
        ) {
            if (size == 0) {
                if (uniqueCountOut) {
                    CUDA_SAFE_CALL(cudaMemsetAsync(uniqueCountOut, 0, sizeof(ui32), stream));
                }
                return;
            }

            // All allocations use RAII to prevent leaks if any subsequent allocation fails.
            TCudaDeviceBuffer<T> values(size);
            TCudaDeviceBuffer<ui32> indices(size);
            TCudaDeviceBuffer<T> valuesSorted(size);
            TCudaDeviceBuffer<ui32> indicesSorted(size);
            TCudaDeviceBuffer<ui32> flags(size);
            TCudaDeviceBuffer<ui32> runIdsSorted(size);

            LaunchCopyStrided<T>(src, strideBytes, size, values.Get(), stream);
            LaunchFillSequence(size, indices.Get(), stream);

            // For signed types, XOR the sign bit before sorting so that CUB radix sort
            // (which treats keys as unsigned bitstrings) produces correct numerical order.
            // Negatives would otherwise sort after positives.
            constexpr bool isSigned = std::is_signed<T>::value;
            if (isSigned) {
                LaunchFlipSignBit<T>(values.Get(), size, stream);
            }

            size_t sortTmpBytes = 0;
            CUDA_SAFE_CALL(cub::DeviceRadixSort::SortPairs(
                nullptr,
                sortTmpBytes,
                values.Get(),
                valuesSorted.Get(),
                indices.Get(),
                indicesSorted.Get(),
                static_cast<int>(size),
                /*begin_bit*/ 0,
                /*end_bit*/ static_cast<int>(sizeof(T) * 8),
                stream
            ));
            TCudaWorkspace sortTmp(sortTmpBytes);
            CUDA_SAFE_CALL(cub::DeviceRadixSort::SortPairs(
                sortTmp.Get(),
                sortTmpBytes,
                values.Get(),
                valuesSorted.Get(),
                indices.Get(),
                indicesSorted.Get(),
                static_cast<int>(size),
                /*begin_bit*/ 0,
                /*end_bit*/ static_cast<int>(sizeof(T) * 8),
                stream
            ));

            // Flip sign bits back in sorted output so that unique values and change flags
            // operate on the original value domain.
            if (isSigned) {
                LaunchFlipSignBit<T>(valuesSorted.Get(), size, stream);
            }

            LaunchComputeChangeFlags<T>(valuesSorted.Get(), size, flags.Get(), stream);

            size_t scanTmpBytes = 0;
            CUDA_SAFE_CALL(cub::DeviceScan::InclusiveSum(
                nullptr,
                scanTmpBytes,
                flags.Get(),
                runIdsSorted.Get(),
                static_cast<int>(size),
                stream
            ));
            TCudaWorkspace scanTmp(scanTmpBytes);
            CUDA_SAFE_CALL(cub::DeviceScan::InclusiveSum(
                scanTmp.Get(),
                scanTmpBytes,
                flags.Get(),
                runIdsSorted.Get(),
                static_cast<int>(size),
                stream
            ));

            LaunchScatterRanks(indicesSorted.Get(), runIdsSorted.Get(), size, ranksOut, stream);

            size_t rleTmpBytes = 0;
            CUDA_SAFE_CALL(cub::DeviceRunLengthEncode::Encode(
                nullptr,
                rleTmpBytes,
                valuesSorted.Get(),
                uniqueValuesOut,
                countsOut,
                uniqueCountOut,
                static_cast<int>(size),
                stream
            ));
            TCudaWorkspace rleTmp(rleTmpBytes);
            CUDA_SAFE_CALL(cub::DeviceRunLengthEncode::Encode(
                rleTmp.Get(),
                rleTmpBytes,
                valuesSorted.Get(),
                uniqueValuesOut,
                countsOut,
                uniqueCountOut,
                static_cast<int>(size),
                stream
            ));
            // All RAII buffers freed automatically here.
        }

        __global__ void MapRanksToBinsImpl(
            const ui32* __restrict ranks,
            ui32 size,
            const ui32* __restrict binsForRank,
            ui32* __restrict dstBins
        ) {
            const ui32 i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < size) {
                dstBins[i] = binsForRank[ranks[i]];
            }
        }

        __global__ void GatherUi32BinsToUi8Impl(
            const ui32* __restrict srcBins,
            ui32 size,
            const ui32* __restrict gatherIndices,
            ui8* __restrict dstBins
        ) {
            const ui32 i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < size) {
                const ui32 srcIdx = gatherIndices ? gatherIndices[i] : i;
                dstBins[i] = static_cast<ui8>(srcBins[srcIdx]);
            }
        }
    }

    void FactorizeStridedGpuInputToUnique(
        const void* src,
        ui64 srcStrideBytes,
        ui32 size,
        EGpuInputDType dtype,
        ui32* ranksOut,
        void* uniqueValuesOut,
        ui32* countsOut,
        ui32* uniqueCountOut,
        TCudaStream stream
    ) {
        if (size == 0) {
            if (uniqueCountOut) {
                CUDA_SAFE_CALL(cudaMemsetAsync(uniqueCountOut, 0, sizeof(ui32), stream));
            }
            return;
        }

        switch (dtype) {
            case EGpuInputDType::Int8:
                FactorizeImpl<i8>(src, srcStrideBytes, size, ranksOut, reinterpret_cast<i8*>(uniqueValuesOut), countsOut, uniqueCountOut, stream);
                break;
            case EGpuInputDType::Int16:
                FactorizeImpl<i16>(src, srcStrideBytes, size, ranksOut, reinterpret_cast<i16*>(uniqueValuesOut), countsOut, uniqueCountOut, stream);
                break;
            case EGpuInputDType::Int32:
                FactorizeImpl<i32>(src, srcStrideBytes, size, ranksOut, reinterpret_cast<i32*>(uniqueValuesOut), countsOut, uniqueCountOut, stream);
                break;
            case EGpuInputDType::Int64:
                FactorizeImpl<i64>(src, srcStrideBytes, size, ranksOut, reinterpret_cast<i64*>(uniqueValuesOut), countsOut, uniqueCountOut, stream);
                break;
            case EGpuInputDType::UInt8:
                FactorizeImpl<ui8>(src, srcStrideBytes, size, ranksOut, reinterpret_cast<ui8*>(uniqueValuesOut), countsOut, uniqueCountOut, stream);
                break;
            case EGpuInputDType::UInt16:
                FactorizeImpl<ui16>(src, srcStrideBytes, size, ranksOut, reinterpret_cast<ui16*>(uniqueValuesOut), countsOut, uniqueCountOut, stream);
                break;
            case EGpuInputDType::UInt32:
                FactorizeImpl<ui32>(src, srcStrideBytes, size, ranksOut, reinterpret_cast<ui32*>(uniqueValuesOut), countsOut, uniqueCountOut, stream);
                break;
            case EGpuInputDType::UInt64:
                FactorizeImpl<ui64>(src, srcStrideBytes, size, ranksOut, reinterpret_cast<ui64*>(uniqueValuesOut), countsOut, uniqueCountOut, stream);
                break;
            default:
                // Not supported here; caller should validate dtype.
                CUDA_SAFE_CALL(cudaMemsetAsync(uniqueCountOut, 0, sizeof(ui32), stream));
                break;
        }
    }

    void MapRanksToBins(
        const ui32* ranks,
        ui32 size,
        const ui32* binsForRank,
        ui32* dstBins,
        TCudaStream stream
    ) {
        if (size == 0) {
            return;
        }
        const ui32 blockSize = 256;
        const ui32 numBlocks = (size + blockSize - 1) / blockSize;
        MapRanksToBinsImpl<<<numBlocks, blockSize, 0, stream>>>(ranks, size, binsForRank, dstBins);
    }

    void GatherUi32BinsToUi8(
        const ui32* srcBins,
        ui32 size,
        const ui32* gatherIndices,
        ui8* dstBins,
        TCudaStream stream
    ) {
        if (size == 0) {
            return;
        }
        const ui32 blockSize = 256;
        const ui32 numBlocks = (size + blockSize - 1) / blockSize;
        GatherUi32BinsToUi8Impl<<<numBlocks, blockSize, 0, stream>>>(srcBins, size, gatherIndices, dstBins);
    }

    void HashUniqueNumericToCatHash(
        const void* uniqueValues,
        ui32 uniqueCount,
        EGpuInputDType dtype,
        ui32* hashesOut,
        TCudaStream stream
    ) {
        if (uniqueCount == 0) {
            return;
        }
        const ui32 blockSize = 256;
        const ui32 numBlocks = (uniqueCount + blockSize - 1) / blockSize;

        switch (dtype) {
            case EGpuInputDType::Int8:
                HashUniqueSignedImpl<i8><<<numBlocks, blockSize, 0, stream>>>(reinterpret_cast<const i8*>(uniqueValues), uniqueCount, hashesOut);
                break;
            case EGpuInputDType::Int16:
                HashUniqueSignedImpl<i16><<<numBlocks, blockSize, 0, stream>>>(reinterpret_cast<const i16*>(uniqueValues), uniqueCount, hashesOut);
                break;
            case EGpuInputDType::Int32:
                HashUniqueSignedImpl<i32><<<numBlocks, blockSize, 0, stream>>>(reinterpret_cast<const i32*>(uniqueValues), uniqueCount, hashesOut);
                break;
            case EGpuInputDType::Int64:
                HashUniqueSignedImpl<i64><<<numBlocks, blockSize, 0, stream>>>(reinterpret_cast<const i64*>(uniqueValues), uniqueCount, hashesOut);
                break;
            case EGpuInputDType::UInt8:
                HashUniqueUnsignedImpl<ui8><<<numBlocks, blockSize, 0, stream>>>(reinterpret_cast<const ui8*>(uniqueValues), uniqueCount, hashesOut);
                break;
            case EGpuInputDType::UInt16:
                HashUniqueUnsignedImpl<ui16><<<numBlocks, blockSize, 0, stream>>>(reinterpret_cast<const ui16*>(uniqueValues), uniqueCount, hashesOut);
                break;
            case EGpuInputDType::UInt32:
                HashUniqueUnsignedImpl<ui32><<<numBlocks, blockSize, 0, stream>>>(reinterpret_cast<const ui32*>(uniqueValues), uniqueCount, hashesOut);
                break;
            case EGpuInputDType::UInt64:
                HashUniqueUnsignedImpl<ui64><<<numBlocks, blockSize, 0, stream>>>(reinterpret_cast<const ui64*>(uniqueValues), uniqueCount, hashesOut);
                break;
            default:
                // Not supported here; caller should validate dtype.
                break;
        }
    }

}
