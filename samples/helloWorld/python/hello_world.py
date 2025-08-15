#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import argparse

import numpy as np
import tensorrt_rtx as trt
from cuda.bindings import runtime as cudart


def cuda_assert(call: tuple) -> object:
    res = None
    err = call[0]
    if len(call) > 1:
        res = call[1]
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA error: {err}")
    return res


# These sizes are arbitrary.
k_input_size = 3
k_hidden_size = 10
k_output_size = 2
k_bytes_per_float = 4

# Set --onnx=/path/to/helloWorld.onnx to parse the provided ONNX model.
k_onnx_model_path = ""

logger = trt.Logger(trt.Logger.VERBOSE)


# Create a simple fully connected network with one input, one hidden layer, and one output.
def create_network(builder: trt.Builder, fc1_weights: trt.Weights, fc2_weights: trt.Weights) -> trt.INetworkDefinition:
    # Specify network creation options.
    # Note: TensorRT-RTX only supports strongly typed networks, explicitly specify this to avoid warning.
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)

    # Create an empty network graph.
    network = builder.create_network(flags)

    # Add the network input.
    input = network.add_input(name="input", dtype=trt.float32, shape=(1, k_input_size))

    # Create constant layers containing weights for fc1/fc2.
    fc1_weights_layer = network.add_constant(trt.Dims2(k_input_size, k_hidden_size), fc1_weights)
    fc1_weights_layer.name = "fully connected layer 1 weights"

    fc2_weights_layer = network.add_constant(trt.Dims2(k_hidden_size, k_output_size), fc2_weights)
    fc2_weights_layer.name = "fully connected layer 2 weights"

    # Add a fully connected layer, fc1.
    fc1 = network.add_matrix_multiply(
        input, trt.MatrixOperation.NONE, fc1_weights_layer.get_output(0), trt.MatrixOperation.NONE
    )
    fc1.name = "fully connected layer 1"

    # Add a relu layer.
    relu = network.add_activation(fc1.get_output(0), type=trt.ActivationType.RELU)
    relu.name = "relu activation"

    # Add a fully connected layer, fc2.
    fc2 = network.add_matrix_multiply(
        relu.get_output(0), trt.MatrixOperation.NONE, fc2_weights_layer.get_output(0), trt.MatrixOperation.NONE
    )
    fc2.name = "fully connected layer 2"

    # Mark the network output tensor.
    fc2.get_output(0).name = "output"
    network.mark_output(fc2.get_output(0))

    return network


# Create a network by parsing the included "helloWorld.onnx" model.
# The ONNX model contains the same layers and weights as the custom network.
def create_network_from_onnx(builder: trt.Builder, onnx_file_path: str) -> trt.INetworkDefinition:
    print("Parsing ONNX file: ", onnx_file_path)

    # Specify network creation options.
    # Note: TensorRT-RTX only supports strongly typed networks, explicitly specify this to avoid warning.
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)

    # Create an empty network graph.
    network = builder.create_network(flags)

    # Parse the network from the ONNX model.
    parser = trt.OnnxParser(network, logger)
    if not parser:
        raise RuntimeError("Failed to create parser!")
    if not parser.parse_from_file(onnx_file_path):
        raise RuntimeError("Failed to parse ONNX file!")

    # Check input and output dimensions to ensure that the selected model is what we expect.
    input = network.get_input(0)
    if input.shape != (1, k_input_size):
        raise ValueError(f"Invalid ONNX input dimension, expected [1, {k_input_size}]!")
    output = network.get_output(0)
    if output.shape != (1, k_output_size):
        raise ValueError(f"Invalid ONNX output dimension, expected [1, {k_output_size}]!")

    return network


# Build the serialized engine.
# In TensorRT-RTX, we often refer to this stage as "Ahead-of-Time" (AOT)
# compilation. This stage tends to be slower than the "Just-in-Time" (JIT)
# compilation stage. For this reason, you should perform this operation at
# installation time or first run, and then save the resulting engine.
#
# You may choose to build the engine once and then deploy it to end-users;
# it is OS-independent and by default supports Ampere and later GPUs. But
# be aware that the engine does not guarantee forward compatibility, so
# you must build a new engine for each new TensorRT-RTX version.
def create_serialized_engine() -> trt.IHostMemory:
    # The weights in this example are initialized to 1.0f, but typically would
    # be loaded from a file or other source.
    # The data backing IConstantLayers must remain valid until the engine has
    # been built; therefore we create weights_data here.
    fc1_weights_data = np.ones(k_input_size * k_hidden_size, dtype=np.float32)
    fc2_weights_data = np.ones(k_hidden_size * k_output_size, dtype=np.float32)

    # Create a builder object.
    builder = trt.Builder(logger)
    if not builder:
        raise RuntimeError("Failed to create builder!")

    # Create a builder configuration to specify optional settings.
    builder_config = builder.create_builder_config()
    if not builder_config:
        raise RuntimeError("Failed to create builder configuration!")

    # Create a simple fully connected network.
    if k_onnx_model_path:
        network = create_network_from_onnx(builder, k_onnx_model_path)
    else:
        fc1_weights = trt.Weights(fc1_weights_data)
        fc2_weights = trt.Weights(fc2_weights_data)
        network = create_network(builder, fc1_weights, fc2_weights)

    # Perform AOT optimizations on the network graph and generate an engine.
    serialized_engine = builder.build_serialized_network(network, builder_config)

    return serialized_engine


# Create an engine execution context out of the serialized engine, then perform inference.
def run_inference(serialized_engine: trt.IHostMemory) -> None:
    runtime = trt.Runtime(logger)
    if not runtime:
        raise RuntimeError("Failed to create runtime!")

    # Deserialize the engine.
    inference_engine = runtime.deserialize_cuda_engine(serialized_engine)
    if not inference_engine:
        raise RuntimeError("Failed to deserialize engine!")

    # Optional settings to configure the behavior of the inference runtime.
    runtime_config = inference_engine.create_runtime_config()
    if not runtime_config:
        raise RuntimeError("Failed to create runtime config!")

    # Create an engine execution context out of the deserialized engine.
    # TRT-RTX performs "Just-in-Time" (JIT) optimization here, targeting the current GPU.
    # JIT phase is faster than AOT phase, and typically completes in under 15 seconds.
    context = inference_engine.create_execution_context(runtime_config)
    if not context:
        raise RuntimeError("Failed to create execution context!")

    # Create a stream for asynchronous execution.
    stream = cuda_assert(cudart.cudaStreamCreate())

    # Allocate GPU memory for input and output bindings.
    input_binding = cuda_assert(cudart.cudaMallocAsync(k_input_size * k_bytes_per_float, stream))
    output_binding = cuda_assert(cudart.cudaMallocAsync(k_output_size * k_bytes_per_float, stream))

    input_buffer = np.zeros(k_input_size, dtype=np.float32)
    output_buffer = np.zeros(k_output_size, dtype=np.float32)

    # Specify the tensor addresses.
    context.set_tensor_address("input", input_binding)
    context.set_tensor_address("output", output_binding)

    try:
        for i in range(5):
            input_buffer.fill(i)

            # Copy input data into the GPU input buffer and execute inference.
            cuda_assert(
                cudart.cudaMemcpyAsync(
                    input_binding,
                    input_buffer.ctypes.data,
                    len(input_buffer) * k_bytes_per_float,
                    cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                    stream,
                )
            )

            status = context.execute_async_v3(stream_handle=stream)
            if not status:
                raise RuntimeError("Failed to execute inference!")

            # Read the results back from GPU output buffer.
            cuda_assert(
                cudart.cudaMemcpyAsync(
                    output_buffer.ctypes.data,
                    output_binding,
                    len(output_buffer) * k_bytes_per_float,
                    cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                    stream,
                )
            )
            cuda_assert(cudart.cudaStreamSynchronize(stream))
            print("Input: ", input_buffer)
            print("Output: ", output_buffer)

    finally:
        cuda_assert(cudart.cudaFreeAsync(input_binding, stream))
        cuda_assert(cudart.cudaFreeAsync(output_binding, stream))
        cuda_assert(cudart.cudaStreamSynchronize(stream))
        cuda_assert(cudart.cudaStreamDestroy(stream))

    print("Successfully ran the network.")


def main() -> None:
    serialized_engine = create_serialized_engine()
    if not serialized_engine:
        raise RuntimeError("Failed to build serialized engine!")
    print(f"Successfully built the network. Engine size: {serialized_engine.nbytes} bytes.")

    run_inference(serialized_engine)


if __name__ == "__main__":
    # Set --onnx=/path/to/helloWorld.onnx to parse the provided ONNX model.
    parser = argparse.ArgumentParser(description="TensorRT-RTX hello world sample")
    parser.add_argument("--onnx", type=str, help="Path to ONNX model file")
    args = parser.parse_args()
    k_onnx_model_path = args.onnx
    main()
