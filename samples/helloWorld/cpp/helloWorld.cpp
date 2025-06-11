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
#include "NvOnnxParser.h"

#include <cassert>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <memory>
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
// Set --onnx=/path/to/helloWorld.onnx to parse the provided ONNX model.
std::string kOnnxModelPath = "";

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
std::unique_ptr<nvinfer1::INetworkDefinition> createNetwork(nvinfer1::IBuilder& builder, const WeightsData& weightsData)
{
    // Specify network creation options.
    // Note: TensorRT-RTX only supports strongly typed networks, explicitly specify this to avoid warning.
    nvinfer1::NetworkDefinitionCreationFlags flags = 1U
        << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kSTRONGLY_TYPED);

    // Create an empty network graph.
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder.createNetworkV2(flags));

    // Add network input tensor.
    auto input = network->addInput("input", nvinfer1::DataType::kFLOAT, nvinfer1::Dims2{1, kInputSize});

    // Create constant layers containing weights for fc1/fc2.
    nvinfer1::Weights fc1Weights = nvinfer1::Weights{nvinfer1::DataType::kFLOAT, weightsData.fc1WeightsData.data(),
        static_cast<int64_t>(weightsData.fc1WeightsData.size())};
    auto fc1WeightsLayer = network->addConstant(nvinfer1::Dims2{kInputSize, kHiddenSize}, fc1Weights);
    fc1WeightsLayer->setName("fully connected layer 1 weights");

    nvinfer1::Weights fc2Weights = nvinfer1::Weights{nvinfer1::DataType::kFLOAT, weightsData.fc2WeightsData.data(),
        static_cast<int64_t>(weightsData.fc2WeightsData.size())};
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

//! Create a network by parsing the included "helloWorld.onnx" model.
//! The ONNX model contains the same layers and weights as the custom network.
std::unique_ptr<nvinfer1::INetworkDefinition> createNetworkFromOnnx(nvinfer1::IBuilder& builder, Logger& logger)
{
    // Specify network creation options.
    // Note: TensorRT-RTX only supports strongly typed networks, explicitly specify this to avoid warning.
    nvinfer1::NetworkDefinitionCreationFlags flags = 1U
        << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kSTRONGLY_TYPED);

    // Create an empty network graph.
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder.createNetworkV2(flags));
    if (!network)
    {
        std::cerr << "Failed to create network!" << std::endl;
        return nullptr;
    }

    // Parse the network from the ONNX model.
    auto parser = nvonnxparser::createParser(*network, logger);
    if (!parser)
    {
        std::cerr << "Failed to create parser!" << std::endl;
        return nullptr;
    }
    if (!parser->parseFromFile(kOnnxModelPath.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kVERBOSE)))
    {
        std::cerr << "Failed to parse ONNX file!" << std::endl;
        return nullptr;
    }

    // Check input and output dimensions to ensure that the selected model is what we expect.
    auto input = network->getInput(0);
    if (!input || input->getDimensions().d[0] != 1 || input->getDimensions().d[1] != kInputSize)
    {
        std::cerr << "Invalid ONNX input dimension, expected [1, " << kInputSize << "]!" << std::endl;
        return nullptr;
    }
    auto output = network->getOutput(0);
    if (!output || output->getDimensions().d[0] != 1 || output->getDimensions().d[1] != kOutputSize)
    {
        std::cerr << "Invalid ONNX output dimension, expected [1, " << kOutputSize << "]!" << std::endl;
        return nullptr;
    }

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
std::unique_ptr<nvinfer1::IHostMemory> createSerializedEngine(Logger& logger)
{
    // Create a builder object.
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
    if (!builder)
    {
        std::cerr << "Failed to create builder!" << std::endl;
        return nullptr;
    }

    // Create a builder configuration to specify optional settings.
    auto builderConfig = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!builderConfig)
    {
        std::cerr << "Failed to create builder config!" << std::endl;
        return nullptr;
    }

    // The data backing IConstantLayers must remain valid until the engine has
    // been built; therefore we create weightsData here.
    WeightsData weightsData;

    // Create a simple fully connected network.
    std::unique_ptr<nvinfer1::INetworkDefinition> network;
    if (!kOnnxModelPath.empty())
    {
        network = createNetworkFromOnnx(*builder, logger);
    }
    else
    {
        network = createNetwork(*builder, weightsData);
    }

    if (!network)
    {
        std::cerr << "Failed to create network definition!" << std::endl;
        return nullptr;
    }

    // Perform AOT optimizations on the network graph and generate an engine.
    auto serializedEngine
        = std::unique_ptr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *builderConfig));

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

int main(int argc, char** argv)
{
    Logger logger;

    // Set --onnx=/path/to/helloWorld.onnx to parse the provided ONNX model.
    if (argc > 1 && std::string(argv[1]).find("--onnx=") != std::string::npos)
    {
        kOnnxModelPath = std::string(argv[1]).substr(std::string("--onnx=").length());
        std::cout << "Enabled parsing for ONNX model: " << kOnnxModelPath << std::endl;
    }

    // Build the serialized engine. This is what TRT-RTX calls "Ahead-of-Time" (AOT) phase.
    std::unique_ptr<nvinfer1::IHostMemory> serializedEngine = createSerializedEngine(logger);
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

    // Deserialize the engine.
    auto inferenceEngine = std::unique_ptr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(serializedEngine->data(), serializedEngine->size()));
    if (!inferenceEngine)
    {
        std::cerr << "Failed to create inference engine!" << std::endl;
        return EXIT_FAILURE;
    }

    // Optional settings to configure the behavior of the inference runtime.
    auto runtimeConfig = std::unique_ptr<nvinfer1::IRuntimeConfig>(inferenceEngine->createRuntimeConfig());
    if (!runtimeConfig)
    {
        std::cerr << "Failed to create runtime config!" << std::endl;
        return EXIT_FAILURE;
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

    // Create a stream for asynchronous execution.
    cudaStream_t stream;
    CUDA_ASSERT(cudaStreamCreate(&stream));

    // Allocate GPU memory for input and output bindings.
    std::vector<void*> bindings(2);
    CUDA_ASSERT(cudaMallocAsync(&bindings[0], kInputSize * sizeof(float), stream));
    CUDA_ASSERT(cudaMallocAsync(&bindings[1], kOutputSize * sizeof(float), stream));

    // Specify the tensor addresses.
    context->setTensorAddress("input", bindings[0]);
    context->setTensorAddress("output", bindings[1]);

    std::vector<float> inputBuffer(kInputSize);
    std::vector<float> outputBuffer(kOutputSize);

    for (int32_t i = 0; i < 5; i++)
    {
        // Copy input data to the device.
        std::fill(inputBuffer.begin(), inputBuffer.end(), static_cast<float>(i));
        CUDA_ASSERT(cudaMemcpyAsync(
            bindings[0], inputBuffer.data(), inputBuffer.size() * sizeof(float), cudaMemcpyHostToDevice, stream));

        // Enqueue the inference.
        bool status = context->enqueueV3(stream);
        if (!status)
        {
            std::cerr << "Failed to execute inference!" << std::endl;
            CUDA_ASSERT(cudaFreeAsync(bindings[0], stream));
            CUDA_ASSERT(cudaFreeAsync(bindings[1], stream));
            CUDA_ASSERT(cudaStreamSynchronize(stream));
            CUDA_ASSERT(cudaStreamDestroy(stream));
            return EXIT_FAILURE;
        }

        // Copy the output data from the device to host.
        CUDA_ASSERT(cudaMemcpyAsync(
            outputBuffer.data(), bindings[1], outputBuffer.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));

        // Synchronize the stream.
        CUDA_ASSERT(cudaStreamSynchronize(stream));

        printBuffer(std::cout, "Input", inputBuffer);
        printBuffer(std::cout, "Output", outputBuffer);
    }

    std::cout << "Successfully ran the network." << std::endl;

    CUDA_ASSERT(cudaFreeAsync(bindings[0], stream));
    CUDA_ASSERT(cudaFreeAsync(bindings[1], stream));
    CUDA_ASSERT(cudaStreamSynchronize(stream));
    CUDA_ASSERT(cudaStreamDestroy(stream));

    return EXIT_SUCCESS;
}
