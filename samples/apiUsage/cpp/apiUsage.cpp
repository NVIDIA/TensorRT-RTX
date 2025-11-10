/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cassert>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <memory>
#include <numeric> // For std::accumulate
#include <optional>
#include <unordered_map>
#include <vector>

//! TensorRT-RTX applications are responsible for implementing the
//! nvinfer1::ILogger interface. This is used to log messages from the
//! TensorRT-RTX library.
class Logger : public nvinfer1::ILogger
{
public:
    Logger() = default;
    ~Logger() override = default;

private:
    std::string severityToString(nvinfer1::ILogger::Severity severity)
    {
        switch (severity)
        {
        case nvinfer1::ILogger::Severity::kVERBOSE: return "VERBOSE";
        case nvinfer1::ILogger::Severity::kINFO: return "INFO";
        case nvinfer1::ILogger::Severity::kWARNING: return "WARNING";
        case nvinfer1::ILogger::Severity::kERROR: return "ERROR";
        case nvinfer1::ILogger::Severity::kINTERNAL_ERROR: return "INTERNAL_ERROR";
        default: return "UNKNOWN";
        }
    }

    void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override
    {
        std::cout << severityToString(severity) << ": " << msg << std::endl;
    }
};

// These sizes are arbitrary.
constexpr int32_t kInputSize = 3;
constexpr int32_t kHiddenSize = 10;
constexpr int32_t kOutputSize = 2;

// Define min/max/opt shapes for dynamic dimensions.
constexpr int32_t kMinBatchSize = 1;
constexpr int32_t kOptBatchSize = 4;
constexpr int32_t kMaxBatchSize = 32;

//! Create a builder configuration. This is used to configure options for
//! how you want your network to be optimized.
std::unique_ptr<nvinfer1::IBuilderConfig> createBuilderConfig(nvinfer1::IBuilder* builder, Logger& logger)
{
    // Create a builder configuration to specify optional settings.
    auto builderConfig = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());

    // Set optimization profiles for dynamic shapes
    // Create an optimization profile.
    nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
    profile->setDimensions("input", nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims2{kMinBatchSize, kInputSize});
    profile->setDimensions("input", nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims2{kOptBatchSize, kInputSize});
    profile->setDimensions("input", nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims2{kMaxBatchSize, kInputSize});

    // Add optimization profile to the builder config.
    builderConfig->addOptimizationProfile(profile);

    // Set compute capabilities.
    {
        constexpr bool useExplicitComputeCapabilities = false;
        constexpr bool useExplicitComputeCapabilitiesTuring = false;

        // In this example, we intend to run ahead-of-time (AOT) compilation on
        // the end-user's machine, so we set the compute capability to kCURRENT.
        // This provides the fastest ahead-of-time compilation, but produces an
        // engine that is only compatible with the current GPU.
        builderConfig->setNbComputeCapabilities(1);
        builderConfig->setComputeCapability(nvinfer1::ComputeCapability::kCURRENT, /* index */ 0);

        // For engines that are deployed with the application to a diverse set of
        // GPUs, one can either
        // a) leave the compute capability unset. The default
        // behavior is to support all RTX compute capabilities, Ampere and later.
        // or
        // b) provide a list of compute capabilities of the end-users' machine explicitly.
        // For example, to build an engine that is runnable on Ada and Blackwell RTX GPUs,
        // SM89 and SM120, you can do:
        if constexpr (useExplicitComputeCapabilities)
        {
            builderConfig->setNbComputeCapabilities(2);
            builderConfig->setComputeCapability(nvinfer1::ComputeCapability::kSM89, /* index */ 0);
            builderConfig->setComputeCapability(nvinfer1::ComputeCapability::kSM120, /* index */ 1);
        }
        // Turing GPUs are not supported by default when leaving the compute capability unset. In
        // this case, you can explicitly set the compute capability to SM75 to support Turing GPUs
        // as shown below.
        if constexpr (useExplicitComputeCapabilitiesTuring)
        {
            builderConfig->setNbComputeCapabilities(1);
            builderConfig->setComputeCapability(nvinfer1::ComputeCapability::kSM75, /* index */ 0);
        }
    }

    // Set refit flags.
    {
        // Build an engine with weights stripped.
        builderConfig->setFlag(nvinfer1::BuilderFlag::kSTRIP_PLAN);

        // Build an engine whose weights can be refit.
        builderConfig->setFlag(nvinfer1::BuilderFlag::kREFIT);
    }

    return builderConfig;
}

struct WeightsData
{
    // The weights in this example are initialized to 1.0f, but typically would
    // be loaded from a file or other source.
    WeightsData()
        : fc1WeightsData(kInputSize * kHiddenSize, 1.0f)
        , fc2WeightsData(kHiddenSize * kOutputSize, 1.0f)
    {
    }

    std::vector<float> fc1WeightsData;
    std::vector<float> fc2WeightsData;
};

//! Create a simple fully connected network with one input, one hidden layer, and one output.
std::unique_ptr<nvinfer1::INetworkDefinition> createNetwork(
    nvinfer1::IBuilder* builder, const nvinfer1::Weights& fc1Weights, const nvinfer1::Weights& fc2Weights)
{
    // Specify network creation options.
    // Note: TensorRT-RTX only supports strongly typed networks, explicitly specify this to avoid warning.
    nvinfer1::NetworkDefinitionCreationFlags flags = 1U
        << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kSTRONGLY_TYPED);

    // Create an empty network graph.
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(flags));

    // Add network input tensor with dynamic batch dimension.
    // -1 indicates dynamic batch size.
    auto input = network->addInput(
        "input", nvinfer1::DataType::kFLOAT, nvinfer1::Dims2{-1, kInputSize}); // -1 indicates dynamic batch size

    // Create constant layers containing weights for fc1/fc2.
    auto fc1WeightsLayer = network->addConstant(nvinfer1::Dims2{kInputSize, kHiddenSize}, fc1Weights);
    fc1WeightsLayer->setName("fully connected layer 1 weights");

    auto fc2WeightsLayer = network->addConstant(nvinfer1::Dims2{kHiddenSize, kOutputSize}, fc2Weights);
    fc2WeightsLayer->setName("fully connected layer 2 weights");

    // Name the fc1 and fc2 weights in the network.
    network->setWeightsName(fc1Weights, "fc1 weights");
    network->setWeightsName(fc2Weights, "fc2 weights");

    // Add a fully connected layer, fc1.
    auto fc1 = network->addMatrixMultiply(
        *input, nvinfer1::MatrixOperation::kNONE, *fc1WeightsLayer->getOutput(0), nvinfer1::MatrixOperation::kNONE);
    fc1->setName("fully connected layer 1");

    // Add a relu layer.
    auto relu = network->addActivation(*fc1->getOutput(0), nvinfer1::ActivationType::kRELU);
    relu->setName("relu activation");

    // Add a fully connected layer, fc2.
    auto fc2 = network->addMatrixMultiply(*relu->getOutput(0), nvinfer1::MatrixOperation::kNONE,
        *fc2WeightsLayer->getOutput(0), nvinfer1::MatrixOperation::kNONE);
    fc2->setName("fully connected layer 2");

    // Mark the network output tensor.
    fc2->getOutput(0)->setName("output");
    network->markOutput(*fc2->getOutput(0));

    return network;
}

//! Build the serialized engine.
//! In TensorRT-RTX, we often refer to this stage as "Ahead-of-Time" (AOT)
//! compilation. This stage tends to be slower than the "Just-in-Time" (JIT)
//! compilation stage. For this reason, you should perform this operation at
//! installation time or first run, and then save the resulting engine.
//!
//! You may choose to build the engine once and then deploy it to end-users;
//! it is OS-independent and by default supports Ampere and later GPUs. But
//! be aware that the engine does not guarantee forward compatibility, so
//! you must build a new engine for each new TensorRT-RTX version.
std::unique_ptr<nvinfer1::IHostMemory> createSerializedEngine(
    Logger& logger, const nvinfer1::Weights& fc1Weights, const nvinfer1::Weights& fc2Weights)
{
    // Create a builder object.
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
    if (!builder)
    {
        std::cerr << "Failed to create builder!" << std::endl;
        return nullptr;
    }

    // Create a builder configuration to specify optional/advanced settings.
    std::unique_ptr<nvinfer1::IBuilderConfig> builderConfig = createBuilderConfig(builder.get(), logger);
    if (!builderConfig)
    {
        std::cerr << "Failed to create builder config!" << std::endl;
        return nullptr;
    }

    // Create a simple fully connected network.
    std::unique_ptr<nvinfer1::INetworkDefinition> network = createNetwork(builder.get(), fc1Weights, fc2Weights);
    if (!network)
    {
        std::cerr << "Failed to create network definition!" << std::endl;
        return nullptr;
    }

    // Perform AOT optimizations on the network graph and generate an engine.
    std::unique_ptr<nvinfer1::IHostMemory> serializedEngine(builder->buildSerializedNetwork(*network, *builderConfig));

    return serializedEngine;
}

template <typename T>
void printBuffer(std::ostream& os, const std::string& name, const T& buffer)
{
    os << name << ": ";
    for (const auto& value : buffer)
    {
        os << value << " ";
    }
    os << std::endl;
}

template <typename T>
using NonOwningPtr = T*;

// Thin wrapper to perform inference.
struct InferenceContext
{
    InferenceContext(
        NonOwningPtr<nvinfer1::ICudaEngine> inferenceEngine, NonOwningPtr<nvinfer1::IExecutionContext> context);
    ~InferenceContext() = default;

    std::optional<std::vector<float>> runInference(
        const std::vector<float>& input, int32_t batchSize, cudaStream_t stream) const;

    Logger logger;
    NonOwningPtr<nvinfer1::ICudaEngine> inferenceEngine;
    NonOwningPtr<nvinfer1::IExecutionContext> context;
};

#define CUDA_ASSERT(cudaCall)                                                                                          \
    do                                                                                                                 \
    {                                                                                                                  \
        cudaError_t __cudaError = (cudaCall);                                                                          \
        if (__cudaError != cudaSuccess)                                                                                \
        {                                                                                                              \
            std::cerr << "CUDA error: " << cudaGetErrorString(__cudaError) << " at " << __FILE__ << ":" << __LINE__    \
                      << std::endl;                                                                                    \
            assert(false);                                                                                             \
        }                                                                                                              \
    } while (0)

InferenceContext::InferenceContext(
    NonOwningPtr<nvinfer1::ICudaEngine> argInferenceEngine, NonOwningPtr<nvinfer1::IExecutionContext> argContext)
    : inferenceEngine(argInferenceEngine)
    , context(argContext)
{
}

std::optional<std::vector<float>> InferenceContext::runInference(
    const std::vector<float>& input, int32_t batchSize, cudaStream_t stream) const
{
    // Select the optimization profile.
    if (!context->setOptimizationProfileAsync(0, stream))
    {
        std::cerr << "Failed to set optimization profile!" << std::endl;
        return std::nullopt;
    }

    // Set the runtime input shape.
    if (!context->setInputShape("input", nvinfer1::Dims2{batchSize, kInputSize}))
    {
        std::cerr << "Failed to set input shape, invalid binding dimensions!" << std::endl;
        return std::nullopt;
    }

    if (!context->allInputDimensionsSpecified())
    {
        std::cerr << "Failed to specify all binding shapes!" << std::endl;
        return std::nullopt;
    }

    if (context->inferShapes(0, nullptr) != 0)
    {
        std::cerr << "Failed to infer tensor shapes!" << std::endl;
        return std::nullopt;
    }

    int32_t const nbIOTensors = inferenceEngine->getNbIOTensors();

    if (nbIOTensors != 2)
    {
        std::cerr << "Expected 2 I/O tensors, got " << nbIOTensors << std::endl;
        return std::nullopt;
    }

    std::vector<int64_t> tensorSizes(nbIOTensors, 0);
    std::vector<std::string> tensorNames(nbIOTensors, "");
    for (int32_t i = 0; i < nbIOTensors; ++i)
    {
        char const* tensorName = inferenceEngine->getIOTensorName(i);
        nvinfer1::Dims tensorShape = context->getTensorShape(tensorName);
        auto const tensorSize = std::accumulate(
            tensorShape.d, tensorShape.d + tensorShape.nbDims, int64_t{1}, std::multiplies<int64_t>());
        tensorSizes[i] = tensorSize;
        tensorNames[i] = std::string(tensorName);
    }

    int64_t const inputTensorSize = tensorSizes[0];
    int64_t const outputTensorSize = tensorSizes[1];

    // Check if the input tensor size is correct given user-fed binding.
    if (inputTensorSize != static_cast<int64_t>(input.size()))
    {
        std::cerr << "Invalid size of calculated and expected host buffers for input" << std::endl;
        return std::nullopt;
    }

    // Check if the output tensor size is correct given apriori knowledge of the output tensor shape.
    if (outputTensorSize != static_cast<int64_t>(batchSize) * kOutputSize)
    {
        std::cerr << "Invalid size of calculated and expected device buffers for output" << std::endl;
        return std::nullopt;
    }

    std::vector<void*> bindings(nbIOTensors, nullptr);
    std::vector<float> output(outputTensorSize);

    for (int32_t i = 0; i < nbIOTensors; ++i)
    {
        // Allocate GPU memory for input/output bindings.
        CUDA_ASSERT(cudaMallocAsync(&bindings[i], tensorSizes[i] * sizeof(float), stream));
    }

    bool setTensorAddressesSuccess = true;
    for (int32_t i = 0; i < nbIOTensors; ++i)
    {
        // Specify the tensor addresses.
        bool const status = context->setTensorAddress(tensorNames[i].c_str(), bindings[i]);
        setTensorAddressesSuccess = setTensorAddressesSuccess && status;
    }
    if (!setTensorAddressesSuccess)
    {
        std::cerr << "Failed to set tensor addresses!" << std::endl;
    }

    // Copy the input data to the device.
    CUDA_ASSERT(
        cudaMemcpyAsync(bindings[0], input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice, stream));

    // Enqueue the inference.
    bool enqueueSuccess = context->enqueueV3(stream);
    if (!enqueueSuccess)
    {
        std::cerr << "Failed to enqueue inference!" << std::endl;
    }

    // Copy the output data from the device to host.
    CUDA_ASSERT(
        cudaMemcpyAsync(output.data(), bindings[1], output.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));

    // Free the GPU memory.
    for (int32_t i = 0; i < nbIOTensors; ++i)
    {
        CUDA_ASSERT(cudaFreeAsync(bindings[i], stream));
    }

    CUDA_ASSERT(cudaStreamSynchronize(stream));

    return (setTensorAddressesSuccess && enqueueSuccess) ? std::make_optional(std::move(output)) : std::nullopt;
}

void useOptionalAdvancedDynamicShapesAPI(
    nvinfer1::IRuntimeConfig* runtimeConfig, nvinfer1::ICudaEngine* inferenceEngine)
{
    // Optionally, print the profile dimensions for the input tensor.
    {
        // Query the profile dimensions for the input tensor and report.
        auto const tensorName = inferenceEngine->getIOTensorName(0);
        // There is only one profile in this example, so use profileIndex 0.
        nvinfer1::Dims minShape
            = inferenceEngine->getProfileShape(tensorName, /*profileIndex*/ 0, nvinfer1::OptProfileSelector::kMIN);
        nvinfer1::Dims optShape
            = inferenceEngine->getProfileShape(tensorName, /*profileIndex*/ 0, nvinfer1::OptProfileSelector::kOPT);
        nvinfer1::Dims maxShape
            = inferenceEngine->getProfileShape(tensorName, /*profileIndex*/ 0, nvinfer1::OptProfileSelector::kMAX);

        std::cout << "Profile dimensions in engine:" << std::endl;
        printBuffer(
            std::cout, "- Minimum", std::vector(std::begin(minShape.d), std::begin(minShape.d) + minShape.nbDims));
        printBuffer(
            std::cout, "- Optimum", std::vector(std::begin(optShape.d), std::begin(optShape.d) + optShape.nbDims));
        printBuffer(
            std::cout, "- Maximum", std::vector(std::begin(maxShape.d), std::begin(maxShape.d) + maxShape.nbDims));
    }

    // Optionally, set the kernel specialization strategy.
    {
        // TensorRT-RTX supports multiple kernel specialization strategies for dynamic shapes, where
        // input shapes are specified at runtime.
        // The strategy configures runtime behavior such that it performs inference for a given input
        // shape with a fallback kernel, while asynchronously compiling a shape-specialized kernel in
        // the background. When the shape-specialized kernel is ready, it will be used for the next inference.
        // This can be used to balance inference performance and kernel compilation time.
        // The default strategy is kLAZY, which showcases above behavior.
        // kEAGER always compiles a shape-specialized kernel for the input shape.
        // kNONE never compiles a shape-specialized kernel, and always uses the fallback kernel.
        runtimeConfig->setDynamicShapesKernelSpecializationStrategy(
            nvinfer1::DynamicShapesKernelSpecializationStrategy::kLAZY);
        // Get API to check the strategy.
        (void) runtimeConfig->getDynamicShapesKernelSpecializationStrategy();
    }
}

int main()
{
    Logger logger;
    // The data backing IConstantLayers must remain valid until the engine has
    // been built and then refit. Therefore we first create weights data, and then
    // individual weights, which are kept alive until after the engine is built and
    // then passed to the refitter.
    WeightsData weightsData;
    auto fc1Weights = nvinfer1::Weights{nvinfer1::DataType::kFLOAT, weightsData.fc1WeightsData.data(),
        static_cast<int64_t>(weightsData.fc1WeightsData.size())};
    auto fc2Weights = nvinfer1::Weights{nvinfer1::DataType::kFLOAT, weightsData.fc2WeightsData.data(),
        static_cast<int64_t>(weightsData.fc2WeightsData.size())};

    std::unique_ptr<nvinfer1::IHostMemory> serializedEngine = createSerializedEngine(logger, fc1Weights, fc2Weights);
    if (!serializedEngine)
    {
        std::cerr << "Failed to build serialized engine!" << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "Successfully built the network. Engine size: " << serializedEngine->size() << " bytes." << std::endl;

    auto runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
    if (!runtime)
    {
        std::cerr << "Failed to create runtime!" << std::endl;
        return EXIT_FAILURE;
    }

    // Check if the engine can be successfully deserialized.
    {
        int64_t const nbHeaderBytes = runtime->getEngineHeaderSize();
        auto dataPtr = static_cast<void const*>(serializedEngine->data());
        if (serializedEngine->size() < static_cast<size_t>(nbHeaderBytes))
        {
            std::cerr << "Serialized engine data is smaller than expected header size!" << std::endl;
            return EXIT_FAILURE;
        }

        // Diagnostics is an invalidity bitmask and is useful for debugging.
        uint64_t diagnostics;
        auto const validity = runtime->getEngineValidity(dataPtr, nbHeaderBytes, &diagnostics);
        if (validity == nvinfer1::EngineValidity::kINVALID)
        {
            using Diag = nvinfer1::EngineInvalidityDiagnostics;
            std::unordered_map<Diag, std::string> const diagMessages{
                {Diag::kVERSION_MISMATCH, "TensorRT version mismatch"},
                {Diag::kUNSUPPORTED_CC, "Unsupported compute capability"},
                {Diag::kOLD_CUDA_DRIVER, "CUDA driver version too old"},
                {Diag::kOLD_CUDA_RUNTIME, "CUDA runtime version too old"},
                {Diag::kINSUFFICIENT_GPU_MEMORY, "Insufficient GPU memory"},
                {Diag::kMALFORMED_ENGINE, "Malformed engine data"}, {Diag::kCUDA_ERROR, "CUDA error occurred"}};

            for (const auto& [diag, message] : diagMessages)
            {
                if (diagnostics & static_cast<uint64_t>(diag))
                {
                    std::cerr << "Engine is invalid due to: " << message << std::endl;
                }
            }
            return EXIT_FAILURE;
        }
        // validity can also be kSUBOPTIMAL or kINVALID, consult the documentation for more details.
    }

    // Deserialize the engine.
    auto inferenceEngine = std::unique_ptr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(serializedEngine->data(), serializedEngine->size()));
    if (!inferenceEngine)
    {
        std::cerr << "Failed to deserialize engine!" << std::endl;
        return EXIT_FAILURE;
    }

    // Create a refitter object for the engine.
    auto refitter = std::unique_ptr<nvinfer1::IRefitter>(nvinfer1::createInferRefitter(*inferenceEngine, logger));
    if (!refitter)
    {
        std::cerr << "Failed to create refitter!" << std::endl;
        return EXIT_FAILURE;
    }

    // Refit fc1 and fc2 weights in the engine.
    bool const fc1RefitSuccess = refitter->setNamedWeights("fc1 weights", fc1Weights);
    bool const fc2RefitSuccess = refitter->setNamedWeights("fc2 weights", fc2Weights);
    if (!(fc1RefitSuccess && fc2RefitSuccess))
    {
        std::cerr << "Failed to set named weights!" << std::endl;
        return EXIT_FAILURE;
    }

    bool const refitSuccess = refitter->refitCudaEngine();
    if (!refitSuccess)
    {
        std::cerr << "Failed to refit engine!" << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "Successfully refit the engine." << std::endl;

    // Optional settings to configure the behavior of the inference runtime.
    auto runtimeConfig = std::unique_ptr<nvinfer1::IRuntimeConfig>(inferenceEngine->createRuntimeConfig());
    if (!runtimeConfig)
    {
        std::cerr << "Failed to create runtime config!" << std::endl;
        return EXIT_FAILURE;
    }

    // Create a runtime cache.
    auto runtimeCache = std::unique_ptr<nvinfer1::IRuntimeCache>(runtimeConfig->createRuntimeCache());
    if (!runtimeCache)
    {
        std::cerr << "Failed to create runtime cache!" << std::endl;
        return EXIT_FAILURE;
    }

    // Set the runtime cache in runtime configuration.
    bool const setRuntimeCacheSuccess = runtimeConfig->setRuntimeCache(*runtimeCache);
    if (!setRuntimeCacheSuccess)
    {
        std::cerr << "Failed to set runtime cache!" << std::endl;
        return EXIT_FAILURE;
    }
    useOptionalAdvancedDynamicShapesAPI(runtimeConfig.get(), inferenceEngine.get());

    // Enable Cudagraphs Whole Graph Capture for accelerated inference
    {
        // TensorRT-RTX can record CUDA graphs to reduce kernel launch overhead during JIT inference.
        // kDISABLED skips graph capture and runs kernels directly on the stream
        // kWHOLE_GRAPH_CAPTURE captures the complete computational graph of the model
        //    and executes it atomically on the GPU stream. It automatically handles dynamic shape
        //    cases, capturing the CUDA graph after shape-specialized kernels are compiled for a given shape.
        bool const setCudaGraphStrategySuccess
            = runtimeConfig->setCudaGraphStrategy(nvinfer1::CudaGraphStrategy::kWHOLE_GRAPH_CAPTURE);
        if (!setCudaGraphStrategySuccess)
        {
            std::cerr << "Failed to set cuda graph strategy!" << std::endl;
            return EXIT_FAILURE;
        }
        // Query API to illustrate retrieval.
        (void) runtimeConfig->getCudaGraphStrategy();
    }

    // Create an engine execution context out of the deserialized engine.
    // TRT-RTX performs "Just-in-Time" (JIT) optimization here, targeting the current GPU.
    // JIT phase is faster than AOT phase, and typically completes in under 15 seconds.
    auto context
        = std::unique_ptr<nvinfer1::IExecutionContext>(inferenceEngine->createExecutionContext(runtimeConfig.get()));
    if (!context)
    {
        std::cerr << "Failed to create execution context!" << std::endl;
        return EXIT_FAILURE;
    }

    // Helper to perform inference for different batch sizes.
    InferenceContext inferenceContext(inferenceEngine.get(), context.get());

    // Run inference with different batch sizes
    std::vector<int32_t> batchSizes = {kMinBatchSize, (kMinBatchSize + kOptBatchSize) / 2, kOptBatchSize,
        (kOptBatchSize + kMaxBatchSize) / 2, kMaxBatchSize};

    // Create a stream for asynchronous execution.
    cudaStream_t stream;
    CUDA_ASSERT(cudaStreamCreate(&stream));

    for (int32_t batchSize : batchSizes)
    {
        std::cout << "Running inference with batch size: " << batchSize << std::endl;

        // Create input data for this batch size
        std::vector<float> input(batchSize * kInputSize);
        for (int32_t i = 0; i < batchSize * kInputSize; i++)
        {
            input[i] = static_cast<float>(batchSize + batchSize * (i % kInputSize));
        }

        // Set shapes, allocate data on GPU, run inference and finally copy output back to host.
        std::optional<std::vector<float>> output = inferenceContext.runInference(input, batchSize, stream);
        CUDA_ASSERT(cudaStreamSynchronize(stream));

        if (!output)
        {
            std::cerr << "Failed to run inference for batch size: " << batchSize << std::endl;
            CUDA_ASSERT(cudaStreamDestroy(stream));
            return EXIT_FAILURE;
        }

        printBuffer(std::cout, "Input", input);
        printBuffer(std::cout, "Output", *output);
    }

    CUDA_ASSERT(cudaStreamDestroy(stream));
    std::cout << "Successfully ran the network with dynamic shapes." << std::endl;

    // Now that we have finished running inference and we want to shut down our
    // application, we can serialize the runtime cache. Normally here we would
    // save the serialized cache to persistent storage using the
    // IHostMemory interface consisting of data(), size() and type().

    // Serialize the runtime cache.
    auto serializedRuntimeCache = std::unique_ptr<nvinfer1::IHostMemory>(runtimeCache->serialize());
    if (!serializedRuntimeCache)
    {
        std::cerr << "Failed to serialize runtime cache!" << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Successfully serialized the runtime cache. " << std::endl
              << "Cache size in bytes: " << serializedRuntimeCache->size() << std::endl;

    return EXIT_SUCCESS;
}
