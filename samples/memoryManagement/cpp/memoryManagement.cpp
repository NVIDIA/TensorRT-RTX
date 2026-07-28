/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "NvInfer.h"
#include "NvInferRuntime.h"

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <vector>

//! Applications implement nvinfer1::ILogger to receive messages from TensorRT-RTX.
class Logger final : public nvinfer1::ILogger
{
    void log(Severity severity, char const* message) noexcept override
    {
        if (severity <= Severity::kWARNING)
        {
            std::cerr << message << std::endl;
        }
    }
};

#define CUDA_CHECK(call)                                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        cudaError_t const status = (call);                                                                             \
        if (status != cudaSuccess)                                                                                     \
        {                                                                                                              \
            std::cerr << "CUDA error " << cudaGetErrorString(status) << " at " << __FILE__ << ":" << __LINE__          \
                      << std::endl;                                                                                    \
            return false;                                                                                              \
        }                                                                                                              \
    } while (0)

//! A synchronous GPU allocator backed by cudaMalloc / cudaFree. It counts the allocations it
//! services so the sample can show TensorRT-RTX routing its internal allocations through it.
//!
//! NOTE: TensorRT uses ⁠Asynchronous allocation (cudaMallocAsync⁠) as its GPU memory allocator as
//! default if you don't register a custom allocator. This is the recommended approach for most users.
//! However, in specific environments (such as CiG mode or RTX Spark unified memory), users may
//! encounter memory allocation failures even when physical GPU memory is still available.
//! Upgrading to the latest CUDA driver can help mitigate this issue. Alternatively, users can
//! implement a custom ⁠IGPUAllocator⁠ backed by standard synchronous allocations via cudaMalloc⁠,
//! as shown here, to completely prevent allocation failures. Note that using ⁠cudaMalloc⁠ may result in a
//! slight performance trade-off and disables support for CUDA Graphs.
class SyncGpuAllocator final : public nvinfer1::IGpuAsyncAllocator
{
public:
    void* allocateAsync(uint64_t size, uint64_t /*alignment*/, nvinfer1::AllocatorFlags /*flags*/,
        cudaStream_t /*stream*/) noexcept override
    {
        // A size-0 request must return nullptr.
        if (size == 0)
        {
            return nullptr;
        }

        void* memory{nullptr};
        if (cudaMalloc(&memory, size) != cudaSuccess)
        {
            return nullptr;
        }
        mNumAllocations.fetch_add(1, std::memory_order_relaxed);
        mBytesAllocated.fetch_add(size, std::memory_order_relaxed);
        return memory;
    }

    bool deallocateAsync(void* memory, cudaStream_t /*stream*/) noexcept override
    {
        // TensorRT-RTX may pass nullptr; treat it as success.
        return memory == nullptr || cudaFree(memory) == cudaSuccess;
    }

    uint64_t numAllocations() const noexcept
    {
        return mNumAllocations.load(std::memory_order_relaxed);
    }

    uint64_t bytesAllocated() const noexcept
    {
        return mBytesAllocated.load(std::memory_order_relaxed);
    }

private:
    // Must be thread-safe: TensorRT-RTX may call the allocator from multiple threads.
    std::atomic<uint64_t> mNumAllocations{0};
    std::atomic<uint64_t> mBytesAllocated{0};
};

// A small MLP (input -> matmul -> ReLU -> matmul -> output) with a dynamic batch dimension.
constexpr int32_t kInputFeatures = 512;
constexpr int32_t kHiddenFeatures = 1024;
constexpr int32_t kOutputFeatures = 512;

// Min / opt / max for the dynamic batch dimension.
constexpr int32_t kMinBatch = 1;
constexpr int32_t kOptBatch = 256;
constexpr int32_t kMaxBatch = 512;

constexpr char const* kInputName = "input";
constexpr char const* kOutputName = "output";

// cudaMalloc satisfies the 256-byte alignment TensorRT-RTX requires for device memory.
constexpr uint64_t kGpuAllocationAlignment = 256;

// Weight backing store; must outlive engine construction. FP16 weights keep the GEMMs runnable on
// Turing GPUs, which do not support FP32 matrix multiplies.
struct WeightsData
{
    WeightsData()
        : fc1(static_cast<size_t>(kInputFeatures) * kHiddenFeatures, __float2half(0.01F))
        , fc2(static_cast<size_t>(kHiddenFeatures) * kOutputFeatures, __float2half(0.01F))
    {
    }

    std::vector<__half> fc1;
    std::vector<__half> fc2;
};

//! Build the MLP (input -> matmul -> ReLU -> matmul -> output) with a dynamic batch dimension,
//! so the activation memory depends on the runtime shape.
std::unique_ptr<nvinfer1::IHostMemory> buildSerializedEngine(
    Logger& logger, SyncGpuAllocator& allocator, WeightsData const& weights)
{
    std::unique_ptr<nvinfer1::IBuilder> builder{nvinfer1::createInferBuilder(logger)};
    if (!builder)
    {
        return nullptr;
    }

    // Register the allocator before building so build-time device allocations go through it.
    builder->setGpuAllocator(&allocator);

    std::unique_ptr<nvinfer1::IBuilderConfig> config{builder->createBuilderConfig()};
    if (!config)
    {
        return nullptr;
    }

    // TensorRT-RTX networks are strongly typed, so no creation flags are required.
    std::unique_ptr<nvinfer1::INetworkDefinition> network{builder->createNetworkV2(0U)};
    if (!network)
    {
        return nullptr;
    }

    // Dynamic batch dimension (-1) with a fixed feature dimension.
    nvinfer1::ITensor* input
        = network->addInput(kInputName, nvinfer1::DataType::kHALF, nvinfer1::Dims2{-1, kInputFeatures});

    nvinfer1::Weights const fc1Weights{
        nvinfer1::DataType::kHALF, weights.fc1.data(), static_cast<int64_t>(weights.fc1.size())};
    nvinfer1::Weights const fc2Weights{
        nvinfer1::DataType::kHALF, weights.fc2.data(), static_cast<int64_t>(weights.fc2.size())};

    auto* fc1Constant = network->addConstant(nvinfer1::Dims2{kInputFeatures, kHiddenFeatures}, fc1Weights);
    auto* fc2Constant = network->addConstant(nvinfer1::Dims2{kHiddenFeatures, kOutputFeatures}, fc2Weights);
    if (input == nullptr || fc1Constant == nullptr || fc2Constant == nullptr)
    {
        return nullptr;
    }

    // matmul -> ReLU -> matmul; the ReLU keeps the batch-scaling intermediate from folding away.
    auto* fc1 = network->addMatrixMultiply(
        *input, nvinfer1::MatrixOperation::kNONE, *fc1Constant->getOutput(0), nvinfer1::MatrixOperation::kNONE);
    auto* relu = fc1 == nullptr ? nullptr : network->addActivation(*fc1->getOutput(0), nvinfer1::ActivationType::kRELU);
    auto* fc2 = relu == nullptr ? nullptr
                                : network->addMatrixMultiply(*relu->getOutput(0), nvinfer1::MatrixOperation::kNONE,
                                    *fc2Constant->getOutput(0), nvinfer1::MatrixOperation::kNONE);
    if (fc2 == nullptr)
    {
        return nullptr;
    }

    fc2->getOutput(0)->setName(kOutputName);
    network->markOutput(*fc2->getOutput(0));

    // Optimization profile covering the dynamic batch range.
    nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
    profile->setDimensions(kInputName, nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims2{kMinBatch, kInputFeatures});
    profile->setDimensions(kInputName, nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims2{kOptBatch, kInputFeatures});
    profile->setDimensions(kInputName, nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims2{kMaxBatch, kInputFeatures});
    config->addOptimizationProfile(profile);

    return std::unique_ptr<nvinfer1::IHostMemory>{builder->buildSerializedNetwork(*network, *config)};
}

//! Frees the device buffers it is given (and synchronizes the stream) when it goes out of scope,
//! so the inference helper can return early on error without leaking.
class ScopedDeviceBuffers
{
public:
    explicit ScopedDeviceBuffers(cudaStream_t stream)
        : mStream(stream)
    {
    }

    ~ScopedDeviceBuffers()
    {
        // Wait for any in-flight inference before freeing the synchronous allocations.
        cudaStreamSynchronize(mStream);
        for (void* buffer : mBuffers)
        {
            cudaFree(buffer);
        }
    }

    void add(void* buffer)
    {
        mBuffers.push_back(buffer);
    }

private:
    cudaStream_t mStream;
    std::vector<void*> mBuffers;
};

//! Run one inference for the shape currently set on the context (profile, input shape, and for
//! kUSER_MANAGED the device memory, must already be set). The I/O buffers allocated here are
//! user-owned and separate from the execution-context activation memory.
bool runInferenceForCurrentShape(
    nvinfer1::ICudaEngine& engine, nvinfer1::IExecutionContext& context, cudaStream_t stream)
{
    // inferShapes() is the readiness check for the name-based shape API (0 == all shapes set).
    if (context.inferShapes(0, nullptr) != 0)
    {
        std::cerr << "Not all input dimensions are specified." << std::endl;
        return false;
    }

    // ScopedDeviceBuffers frees these on every return path below.
    ScopedDeviceBuffers buffers(stream);
    for (int32_t i = 0; i < engine.getNbIOTensors(); ++i)
    {
        char const* name = engine.getIOTensorName(i);
        nvinfer1::Dims const shape = context.getTensorShape(name);
        int64_t const volume = std::accumulate(shape.d, shape.d + shape.nbDims, int64_t{1}, std::multiplies<int64_t>());
        size_t const bytes = static_cast<size_t>(volume) * sizeof(__half);

        void* buffer{nullptr};
        CUDA_CHECK(cudaMalloc(&buffer, bytes));
        buffers.add(buffer);
        // Zero-initialize inputs for deterministic output; outputs are overwritten by inference.
        CUDA_CHECK(cudaMemset(buffer, 0, bytes));
        if (!context.setTensorAddress(name, buffer))
        {
            std::cerr << "Failed to set the address for tensor " << name << "." << std::endl;
            return false;
        }
    }

    if (!context.enqueueV3(stream))
    {
        std::cerr << "Failed to run inference." << std::endl;
        return false;
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return true;
}

//! Strategy 1: TensorRT-RTX owns the activation memory (kSTATIC), serviced by the allocator.
bool demonstrateTrtManagedMemory(nvinfer1::ICudaEngine& engine, SyncGpuAllocator& allocator, cudaStream_t stream)
{
    std::cout << "\n=== Strategy 1: TensorRT-RTX-managed memory (kSTATIC) ===" << std::endl;

    uint64_t const allocationsBefore = allocator.numAllocations();

    // kSTATIC pre-allocates a block for the worst case across profiles, via our allocator.
    std::unique_ptr<nvinfer1::IExecutionContext> context{engine.createExecutionContext()};
    if (!context)
    {
        std::cerr << "Failed to create execution context." << std::endl;
        return false;
    }

    std::cout << "TensorRT-RTX reserved " << engine.getDeviceMemorySizeV2()
              << " bytes (worst case across all profiles)." << std::endl;
    std::cout << "The custom allocator serviced " << (allocator.numAllocations() - allocationsBefore)
              << " allocation(s) while creating the context." << std::endl;

    // Run inference at the largest shape to exercise the reserved memory.
    if (!context->setInputShape(kInputName, nvinfer1::Dims2{kMaxBatch, kInputFeatures}))
    {
        std::cerr << "Failed to set input shape." << std::endl;
        return false;
    }
    return runInferenceForCurrentShape(engine, *context, stream);
}

//! Strategy 2: the application owns the activation memory (kUSER_MANAGED), sizing it per input
//! shape with updateDeviceMemorySizeForShapes() and allocating through the same allocator.
bool demonstrateUserManagedMemory(nvinfer1::ICudaEngine& engine, SyncGpuAllocator& allocator, cudaStream_t stream)
{
    std::cout << "\n=== Strategy 2: user-managed memory (kUSER_MANAGED) ===" << std::endl;

    std::unique_ptr<nvinfer1::IRuntimeConfig> runtimeConfig{engine.createRuntimeConfig()};
    if (!runtimeConfig)
    {
        std::cerr << "Failed to create runtime config." << std::endl;
        return false;
    }
    runtimeConfig->setExecutionContextAllocationStrategy(nvinfer1::ExecutionContextAllocationStrategy::kUSER_MANAGED);

    std::unique_ptr<nvinfer1::IExecutionContext> context{engine.createExecutionContext(runtimeConfig.get())};
    if (!context)
    {
        std::cerr << "Failed to create execution context." << std::endl;
        return false;
    }

    int64_t const worstCaseBytes = engine.getDeviceMemorySizeV2();
    std::cout << "Worst-case device memory across all profiles: " << worstCaseBytes << " bytes." << std::endl;

    for (int32_t batch : {kMinBatch, kOptBatch, kMaxBatch})
    {
        // Select the optimization profile and set all input shapes before querying the size.
        if (!context->setOptimizationProfileAsync(0, stream)
            || !context->setInputShape(kInputName, nvinfer1::Dims2{batch, kInputFeatures}))
        {
            std::cerr << "Failed to set profile/input shape." << std::endl;
            return false;
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Size the memory for exactly these shapes (0 if the profile/shapes are unset or none needed).
        size_t const bytesForShape = context->updateDeviceMemorySizeForShapes();

        // Allocate it ourselves through the same allocator; a size of 0 -> null pointer is valid.
        void* deviceMemory = bytesForShape == 0
            ? nullptr
            : allocator.allocateAsync(bytesForShape, kGpuAllocationAlignment, nvinfer1::AllocatorFlags{}, stream);
        if (bytesForShape != 0 && deviceMemory == nullptr)
        {
            std::cerr << "Failed to allocate device memory." << std::endl;
            return false;
        }
        context->setDeviceMemoryV2(deviceMemory, static_cast<int64_t>(bytesForShape));

        std::cout << "batch " << batch << ": needs " << bytesForShape << " bytes ("
                  << (worstCaseBytes == 0 ? 0 : 100 * static_cast<int64_t>(bytesForShape) / worstCaseBytes)
                  << "% of the worst case)." << std::endl;

        bool const ok = runInferenceForCurrentShape(engine, *context, stream);

        // Release the memory only after inference has completed.
        allocator.deallocateAsync(deviceMemory, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        if (!ok)
        {
            return false;
        }
    }
    return true;
}

int main()
{
    Logger logger;

    // The allocator and weights must outlive every TensorRT-RTX object that uses them.
    SyncGpuAllocator allocator;
    WeightsData weights;

    std::unique_ptr<nvinfer1::IHostMemory> serializedEngine = buildSerializedEngine(logger, allocator, weights);
    if (!serializedEngine)
    {
        std::cerr << "Failed to build the engine." << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "Built the engine (" << serializedEngine->size() << " bytes)." << std::endl;

    std::unique_ptr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(logger)};
    if (!runtime)
    {
        std::cerr << "Failed to create the runtime." << std::endl;
        return EXIT_FAILURE;
    }

    // Register the custom allocator so runtime and execution-context allocations use it.
    runtime->setGpuAllocator(&allocator);

    std::unique_ptr<nvinfer1::ICudaEngine> engine{
        runtime->deserializeCudaEngine(serializedEngine->data(), serializedEngine->size())};
    if (!engine)
    {
        std::cerr << "Failed to deserialize the engine." << std::endl;
        return EXIT_FAILURE;
    }

    cudaStream_t stream{};
    if (cudaStreamCreate(&stream) != cudaSuccess)
    {
        std::cerr << "Failed to create CUDA stream." << std::endl;
        return EXIT_FAILURE;
    }

    bool const success = demonstrateTrtManagedMemory(*engine, allocator, stream)
        && demonstrateUserManagedMemory(*engine, allocator, stream);

    cudaStreamDestroy(stream);

    if (!success)
    {
        return EXIT_FAILURE;
    }

    std::cout << "\nDone. The custom allocator serviced " << allocator.numAllocations() << " allocation(s), "
              << allocator.bytesAllocated() << " bytes in total." << std::endl;
    return EXIT_SUCCESS;
}
