#include <catboost/cuda/gpu_data/kernel/gpu_input_utils.cuh>
#include <catboost/cuda/gpu_data/kernel/gpu_input_targets.cuh>

#include <catboost/libs/cat_feature/cat_feature.h>

#include <library/cpp/testing/unittest/registar.h>

#include <util/generic/vector.h>
#include <util/string/cast.h>

#include <cuda_runtime.h>

#include <cmath>
#include <limits>

namespace {

    template <class TValue, class TStringValue>
    static ui32 CalcCpuCatHash(const TValue v) {
        return CalcCatFeatureHash(ToString(static_cast<TStringValue>(v)));
    }

}

Y_UNIT_TEST_SUITE(TGpuInputKernelsTest) {

    Y_UNIT_TEST(TestCopyStridedGpuInputToFloatContiguous) {
        CUDA_SAFE_CALL(cudaSetDevice(0));

        const TVector<float> input = {1.0f, 2.0f, 3.0f, 4.0f};
        const ui32 size = static_cast<ui32>(input.size());

        float* dSrc = nullptr;
        float* dDst = nullptr;
        CUDA_SAFE_CALL(cudaMalloc(&dSrc, size * sizeof(float)));
        CUDA_SAFE_CALL(cudaMalloc(&dDst, size * sizeof(float)));
        Y_DEFER {
            if (dSrc) { cudaFree(dSrc); }
            if (dDst) { cudaFree(dDst); }
        };

        CUDA_SAFE_CALL(cudaMemcpy(dSrc, input.data(), size * sizeof(float), cudaMemcpyHostToDevice));

        NKernel::CopyStridedGpuInputToFloat(
            dSrc,
            sizeof(float),
            size,
            NKernel::EGpuInputDType::Float32,
            dDst,
            /*stream*/ 0
        );
        CUDA_SAFE_CALL(cudaDeviceSynchronize());

        TVector<float> result;
        result.yresize(size);
        CUDA_SAFE_CALL(cudaMemcpy(result.data(), dDst, size * sizeof(float), cudaMemcpyDeviceToHost));

        for (size_t i = 0; i < size; ++i) {
            UNIT_ASSERT_DOUBLES_EQUAL_C(result[i], input[i], 1e-6f, "index=" << i);
        }
    }

    Y_UNIT_TEST(TestCopyStridedGpuInputToFloatStrided) {
        CUDA_SAFE_CALL(cudaSetDevice(0));

        // Interleaved: [1.0, X, 2.0, X, 3.0, X, 4.0, X] where X is padding
        const TVector<float> interleaved = {1.0f, 0.0f, 2.0f, 0.0f, 3.0f, 0.0f, 4.0f, 0.0f};
        const TVector<float> expected = {1.0f, 2.0f, 3.0f, 4.0f};
        const ui32 size = 4;

        float* dSrc = nullptr;
        float* dDst = nullptr;
        CUDA_SAFE_CALL(cudaMalloc(&dSrc, interleaved.size() * sizeof(float)));
        CUDA_SAFE_CALL(cudaMalloc(&dDst, size * sizeof(float)));
        Y_DEFER {
            if (dSrc) { cudaFree(dSrc); }
            if (dDst) { cudaFree(dDst); }
        };

        CUDA_SAFE_CALL(cudaMemcpy(dSrc, interleaved.data(), interleaved.size() * sizeof(float), cudaMemcpyHostToDevice));

        NKernel::CopyStridedGpuInputToFloat(
            dSrc,
            2 * sizeof(float),  // stride = every other element
            size,
            NKernel::EGpuInputDType::Float32,
            dDst,
            /*stream*/ 0
        );
        CUDA_SAFE_CALL(cudaDeviceSynchronize());

        TVector<float> result;
        result.yresize(size);
        CUDA_SAFE_CALL(cudaMemcpy(result.data(), dDst, size * sizeof(float), cudaMemcpyDeviceToHost));

        for (size_t i = 0; i < size; ++i) {
            UNIT_ASSERT_DOUBLES_EQUAL_C(result[i], expected[i], 1e-6f, "index=" << i);
        }
    }

    Y_UNIT_TEST(TestCopyStridedGpuInputToFloatFromFloat64) {
        CUDA_SAFE_CALL(cudaSetDevice(0));

        const TVector<double> input = {1.5, 2.5, 3.5, 4.5};
        const TVector<float> expected = {1.5f, 2.5f, 3.5f, 4.5f};
        const ui32 size = static_cast<ui32>(input.size());

        void* dSrc = nullptr;
        float* dDst = nullptr;
        CUDA_SAFE_CALL(cudaMalloc(&dSrc, size * sizeof(double)));
        CUDA_SAFE_CALL(cudaMalloc(&dDst, size * sizeof(float)));
        Y_DEFER {
            if (dSrc) { cudaFree(dSrc); }
            if (dDst) { cudaFree(dDst); }
        };

        CUDA_SAFE_CALL(cudaMemcpy(dSrc, input.data(), size * sizeof(double), cudaMemcpyHostToDevice));

        NKernel::CopyStridedGpuInputToFloat(
            dSrc,
            sizeof(double),
            size,
            NKernel::EGpuInputDType::Float64,
            dDst,
            /*stream*/ 0
        );
        CUDA_SAFE_CALL(cudaDeviceSynchronize());

        TVector<float> result;
        result.yresize(size);
        CUDA_SAFE_CALL(cudaMemcpy(result.data(), dDst, size * sizeof(float), cudaMemcpyDeviceToHost));

        for (size_t i = 0; i < size; ++i) {
            UNIT_ASSERT_DOUBLES_EQUAL_C(result[i], expected[i], 1e-6f, "index=" << i);
        }
    }

    Y_UNIT_TEST(TestHashStridedGpuInputToCatHashInt32) {
        CUDA_SAFE_CALL(cudaSetDevice(0));

        const TVector<i32> values = {0, 1, 42, -1, std::numeric_limits<i32>::max()};
        const ui32 size = static_cast<ui32>(values.size());

        void* dSrc = nullptr;
        ui32* dHashes = nullptr;
        CUDA_SAFE_CALL(cudaMalloc(&dSrc, size * sizeof(i32)));
        CUDA_SAFE_CALL(cudaMalloc(&dHashes, size * sizeof(ui32)));
        Y_DEFER {
            if (dSrc) { cudaFree(dSrc); }
            if (dHashes) { cudaFree(dHashes); }
        };

        CUDA_SAFE_CALL(cudaMemcpy(dSrc, values.data(), size * sizeof(i32), cudaMemcpyHostToDevice));

        NKernel::HashStridedGpuInputToCatHash(
            dSrc,
            sizeof(i32),
            size,
            NKernel::EGpuInputDType::Int32,
            dHashes,
            /*stream*/ 0
        );
        CUDA_SAFE_CALL(cudaDeviceSynchronize());

        TVector<ui32> hashes;
        hashes.yresize(size);
        CUDA_SAFE_CALL(cudaMemcpy(hashes.data(), dHashes, size * sizeof(ui32), cudaMemcpyDeviceToHost));

        for (size_t i = 0; i < size; ++i) {
            const ui32 expected = CalcCpuCatHash<i32, i64>(values[i]);
            UNIT_ASSERT_VALUES_EQUAL_C(hashes[i], expected, "index=" << i);
        }
    }

    Y_UNIT_TEST(TestHashStridedGpuInputToCatHashInt64) {
        CUDA_SAFE_CALL(cudaSetDevice(0));

        const TVector<i64> values = {
            0LL,
            1LL,
            42LL,
            -1LL,
            1234567890123456789LL,
            std::numeric_limits<i64>::max()
        };
        const ui32 size = static_cast<ui32>(values.size());

        void* dSrc = nullptr;
        ui32* dHashes = nullptr;
        CUDA_SAFE_CALL(cudaMalloc(&dSrc, size * sizeof(i64)));
        CUDA_SAFE_CALL(cudaMalloc(&dHashes, size * sizeof(ui32)));
        Y_DEFER {
            if (dSrc) { cudaFree(dSrc); }
            if (dHashes) { cudaFree(dHashes); }
        };

        CUDA_SAFE_CALL(cudaMemcpy(dSrc, values.data(), size * sizeof(i64), cudaMemcpyHostToDevice));

        NKernel::HashStridedGpuInputToCatHash(
            dSrc,
            sizeof(i64),
            size,
            NKernel::EGpuInputDType::Int64,
            dHashes,
            /*stream*/ 0
        );
        CUDA_SAFE_CALL(cudaDeviceSynchronize());

        TVector<ui32> hashes;
        hashes.yresize(size);
        CUDA_SAFE_CALL(cudaMemcpy(hashes.data(), dHashes, size * sizeof(ui32), cudaMemcpyDeviceToHost));

        for (size_t i = 0; i < size; ++i) {
            const ui32 expected = CalcCpuCatHash<i64, i64>(values[i]);
            UNIT_ASSERT_VALUES_EQUAL_C(hashes[i], expected, "index=" << i);
        }
    }

    Y_UNIT_TEST(TestBinarizeToUi8) {
        CUDA_SAFE_CALL(cudaSetDevice(0));

        const TVector<float> values = {0.1f, 0.5f, 0.9f, 1.5f, 2.5f};
        const TVector<float> borders = {0.3f, 1.0f, 2.0f};
        const TVector<ui8> expected = {0, 1, 1, 2, 3};
        const ui32 size = static_cast<ui32>(values.size());
        const ui32 borderCount = static_cast<ui32>(borders.size());

        float* dValues = nullptr;
        float* dBorders = nullptr;
        ui8* dDst = nullptr;
        CUDA_SAFE_CALL(cudaMalloc(&dValues, size * sizeof(float)));
        CUDA_SAFE_CALL(cudaMalloc(&dBorders, borderCount * sizeof(float)));
        CUDA_SAFE_CALL(cudaMalloc(&dDst, size * sizeof(ui8)));
        Y_DEFER {
            if (dValues) { cudaFree(dValues); }
            if (dBorders) { cudaFree(dBorders); }
            if (dDst) { cudaFree(dDst); }
        };

        CUDA_SAFE_CALL(cudaMemcpy(dValues, values.data(), size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(dBorders, borders.data(), borderCount * sizeof(float), cudaMemcpyHostToDevice));

        NKernel::BinarizeToUi8(
            dValues,
            size,
            dBorders,
            borderCount,
            dDst,
            /*stream*/ 0
        );
        CUDA_SAFE_CALL(cudaDeviceSynchronize());

        TVector<ui8> result;
        result.yresize(size);
        CUDA_SAFE_CALL(cudaMemcpy(result.data(), dDst, size * sizeof(ui8), cudaMemcpyDeviceToHost));

        for (size_t i = 0; i < size; ++i) {
            UNIT_ASSERT_VALUES_EQUAL_C(static_cast<ui32>(result[i]), static_cast<ui32>(expected[i]), "index=" << i);
        }
    }

    Y_UNIT_TEST(TestBinarizeToUi8ZeroBorders) {
        CUDA_SAFE_CALL(cudaSetDevice(0));

        const TVector<float> values = {0.1f, 0.5f, 0.9f, 1.5f, 2.5f};
        const ui32 size = static_cast<ui32>(values.size());

        float* dValues = nullptr;
        ui8* dDst = nullptr;
        CUDA_SAFE_CALL(cudaMalloc(&dValues, size * sizeof(float)));
        CUDA_SAFE_CALL(cudaMalloc(&dDst, size * sizeof(ui8)));
        Y_DEFER {
            if (dValues) { cudaFree(dValues); }
            if (dDst) { cudaFree(dDst); }
        };

        CUDA_SAFE_CALL(cudaMemcpy(dValues, values.data(), size * sizeof(float), cudaMemcpyHostToDevice));

        NKernel::BinarizeToUi8(
            dValues,
            size,
            /*borders*/ nullptr,
            /*borderCount*/ 0,
            dDst,
            /*stream*/ 0
        );
        CUDA_SAFE_CALL(cudaDeviceSynchronize());

        TVector<ui8> result;
        result.yresize(size);
        CUDA_SAFE_CALL(cudaMemcpy(result.data(), dDst, size * sizeof(ui8), cudaMemcpyDeviceToHost));

        for (size_t i = 0; i < size; ++i) {
            UNIT_ASSERT_VALUES_EQUAL_C(static_cast<ui32>(result[i]), 0u, "index=" << i);
        }
    }

    Y_UNIT_TEST(TestBinarizeToUi32) {
        CUDA_SAFE_CALL(cudaSetDevice(0));

        const TVector<float> values = {0.1f, 0.5f, 0.9f, 1.5f, 2.5f};
        const TVector<float> borders = {0.3f, 1.0f, 2.0f};
        const TVector<ui32> expected = {0, 1, 1, 2, 3};
        const ui32 size = static_cast<ui32>(values.size());
        const ui32 borderCount = static_cast<ui32>(borders.size());

        float* dValues = nullptr;
        float* dBorders = nullptr;
        ui32* dDst = nullptr;
        CUDA_SAFE_CALL(cudaMalloc(&dValues, size * sizeof(float)));
        CUDA_SAFE_CALL(cudaMalloc(&dBorders, borderCount * sizeof(float)));
        CUDA_SAFE_CALL(cudaMalloc(&dDst, size * sizeof(ui32)));
        Y_DEFER {
            if (dValues) { cudaFree(dValues); }
            if (dBorders) { cudaFree(dBorders); }
            if (dDst) { cudaFree(dDst); }
        };

        CUDA_SAFE_CALL(cudaMemcpy(dValues, values.data(), size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(dBorders, borders.data(), borderCount * sizeof(float), cudaMemcpyHostToDevice));

        NKernel::BinarizeToUi32(
            dValues,
            size,
            dBorders,
            borderCount,
            dDst,
            /*stream*/ 0
        );
        CUDA_SAFE_CALL(cudaDeviceSynchronize());

        TVector<ui32> result;
        result.yresize(size);
        CUDA_SAFE_CALL(cudaMemcpy(result.data(), dDst, size * sizeof(ui32), cudaMemcpyDeviceToHost));

        for (size_t i = 0; i < size; ++i) {
            UNIT_ASSERT_VALUES_EQUAL_C(result[i], expected[i], "index=" << i);
        }
    }

    Y_UNIT_TEST(TestBinarizeToUi32ZeroBorders) {
        CUDA_SAFE_CALL(cudaSetDevice(0));

        const TVector<float> values = {0.1f, 0.5f, 0.9f, 1.5f, 2.5f};
        const ui32 size = static_cast<ui32>(values.size());

        float* dValues = nullptr;
        ui32* dDst = nullptr;
        CUDA_SAFE_CALL(cudaMalloc(&dValues, size * sizeof(float)));
        CUDA_SAFE_CALL(cudaMalloc(&dDst, size * sizeof(ui32)));
        Y_DEFER {
            if (dValues) { cudaFree(dValues); }
            if (dDst) { cudaFree(dDst); }
        };

        CUDA_SAFE_CALL(cudaMemcpy(dValues, values.data(), size * sizeof(float), cudaMemcpyHostToDevice));

        NKernel::BinarizeToUi32(
            dValues,
            size,
            /*borders*/ nullptr,
            /*borderCount*/ 0,
            dDst,
            /*stream*/ 0
        );
        CUDA_SAFE_CALL(cudaDeviceSynchronize());

        TVector<ui32> result;
        result.yresize(size);
        CUDA_SAFE_CALL(cudaMemcpy(result.data(), dDst, size * sizeof(ui32), cudaMemcpyDeviceToHost));

        for (size_t i = 0; i < size; ++i) {
            UNIT_ASSERT_VALUES_EQUAL_C(result[i], 0u, "index=" << i);
        }
    }
}
