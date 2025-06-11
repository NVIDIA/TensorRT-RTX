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

import numpy as np
import tensorrt_rtx as trt
from cuda.bindings import runtime as cudart


def cuda_assert(call):
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

# Define min/max/opt shapes for dynamic dimensions
k_min_batch_size = 1
k_opt_batch_size = 4
k_max_batch_size = 32

logger = trt.Logger(trt.Logger.VERBOSE)


def create_builder_config(builder):
    # Create a builder configuration to specify optional settings.
    builder_config = builder.create_builder_config()

    # Set optimization profiles for dynamic shapes
    # Create an optimization profile.
    profile = builder.create_optimization_profile()
    min_profile = trt.Dims2(k_min_batch_size, k_input_size)
    opt_profile = trt.Dims2(k_opt_batch_size, k_input_size)
    max_profile = trt.Dims2(k_max_batch_size, k_input_size)
    profile.set_shape("input", min_profile, opt_profile, max_profile)

    # Add optimization profile to the builder config.
    builder_config.add_optimization_profile(profile)

    # Set compute capabilities.
    use_explicit_compute_capabilities = False
    use_explicit_compute_capabilities_turing = False
    assert not (use_explicit_compute_capabilities and use_explicit_compute_capabilities_turing)

    # In this example, we intend to run ahead-of-time (AOT) compilation on
    # the end-user's machine, so we set the compute capability to CURRENT.
    # This provides the fastest ahead-of-time compilation, but produces an
    # engine that is only compatible with the current GPU.
    builder_config.num_compute_capabilities = 1
    builder_config.set_compute_capability(trt.ComputeCapability.CURRENT, 0)

    # For engines that are deployed with the application to a diverse set of
    # GPUs, one can either
    # a) leave the compute capability unset. The default
    # behavior is to support all RTX compute capabilities, Ampere and later.
    # or
    # b) provide a list of compute capabilities of the end-users' machine explicitly.
    if use_explicit_compute_capabilities:
        builder_config.num_compute_capabilities = 2
        builder_config.set_compute_capability(trt.ComputeCapability.SM89, 0)
        builder_config.set_compute_capability(trt.ComputeCapability.SM120, 1)

    # Turing GPUs are not supported by default when leaving the compute capability unset.
    if use_explicit_compute_capabilities_turing:
        builder_config.num_compute_capabilities = 1
        builder_config.set_compute_capability(trt.ComputeCapability.SM75, 0)

    # Set refit flags.
    # Build an engine with weights stripped.
    builder_config.set_flag(trt.BuilderFlag.STRIP_PLAN)

    # Build an engine whose weights can be refit.
    builder_config.set_flag(trt.BuilderFlag.REFIT)

    return builder_config


# Create a simple fully connected network with one input, one hidden layer, and one output.
def create_network(builder, fc1_weights, fc2_weights):
    # Specify network creation options.
    # Note: TensorRT-RTX only supports strongly typed networks, explicitly specify this to avoid warning.
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)

    # Create an empty network graph.
    network = builder.create_network(flags)

    # Add network input tensor with dynamic batch dimension.
    # -1 indicates dynamic batch size.
    input = network.add_input("input", trt.float32, trt.Dims2(-1, k_input_size))

    # Create constant layers containing weights for fc1/fc2.
    fc1_weights_layer = network.add_constant(trt.Dims2(k_input_size, k_hidden_size), fc1_weights)
    fc1_weights_layer.name = "fully connected layer 1 weights"

    fc2_weights_layer = network.add_constant(trt.Dims2(k_hidden_size, k_output_size), fc2_weights)
    fc2_weights_layer.name = "fully connected layer 2 weights"

    # Name the fc1 and fc2 weights in the network.
    network.set_weights_name(fc1_weights, "fc1 weights")
    network.set_weights_name(fc2_weights, "fc2 weights")

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
def create_serialized_engine(fc1_weights, fc2_weights):
    # Create a builder object.
    builder = trt.Builder(logger)
    if not builder:
        raise RuntimeError("Failed to create builder!")

    # Create a builder configuration to specify optional/advanced settings.
    builder_config = create_builder_config(builder)
    if not builder_config:
        raise RuntimeError("Failed to create builder config!")

    # Create a simple fully connected network.
    network = create_network(builder, fc1_weights, fc2_weights)
    if not network:
        raise RuntimeError("Failed to create network definition!")

    # Perform AOT optimizations on the network graph and generate an engine.
    serialized_engine = builder.build_serialized_network(network, builder_config)

    return serialized_engine


def use_optional_advanced_dynamic_shapes_api(runtime_config, inference_engine):
    # Optionally, print the profile dimensions for the input tensor.
    # Query the profile dimensions for the input tensor and report.
    tensor_name = inference_engine.get_tensor_name(0)
    # There is only one profile in this example, so use profileIndex 0.
    min_shape, opt_shape, max_shape = inference_engine.get_tensor_profile_shape(tensor_name, 0)

    print("Profile dimensions in engine:")
    print("- Minimum:", min_shape)
    print("- Optimum:", opt_shape)
    print("- Maximum:", max_shape)

    # Optionally, set the kernel specialization strategy.
    # TensorRT-RTX supports multiple kernel specialization strategies for dynamic shapes, where
    # input shapes are specified at runtime.
    # The strategy configures runtime behavior such that it performs inference for a given input
    # shape with a fallback kernel, while asynchronously compiling a shape-specialized kernel in
    # the background. When the shape-specialized kernel is ready, it will be used for the next inference.
    # This can be used to balance inference performance and kernel compilation time.
    # The default strategy is LAZY, which showcases above behavior.
    # EAGER always compiles a shape-specialized kernel for the input shape.
    # NONE never compiles a shape-specialized kernel, and always uses the fallback kernel.
    runtime_config.dynamic_shapes_kernel_specialization_strategy = trt.DynamicShapesKernelSpecializationStrategy.LAZY


# Helper to perform inference for changing shapes.
class InferenceContext:
    def __init__(self, inference_engine, context):
        self.inference_engine = inference_engine
        self.context = context

    def run_inference(self, input_buffer, batch_size, stream):
        # Select the optimization profile.
        if not self.context.set_optimization_profile_async(0, stream):
            raise RuntimeError("Failed to set optimization profile!")

        # Set the runtime input shape.
        if not self.context.set_input_shape("input", trt.Dims2(batch_size, k_input_size)):
            raise RuntimeError("Failed to set input shape, invalid binding dimensions!")

        if not self.context.all_binding_shapes_specified:
            raise RuntimeError("Failed to specify all binding shapes!")

        if self.context.infer_shapes():  # If non-empty list of names
            raise RuntimeError("Failed to infer tensor shapes!")

        num_io_tensors = self.inference_engine.num_io_tensors

        if num_io_tensors != 2:
            raise RuntimeError(f"Expected 2 I/O tensors, got {num_io_tensors}")

        tensor_names = [self.inference_engine.get_tensor_name(i) for i in range(num_io_tensors)]

        def get_tensor_size(tensor_name):
            tensor_shape = self.context.get_tensor_shape(tensor_name)
            return np.prod(tensor_shape)

        tensor_sizes = list(map(get_tensor_size, tensor_names))

        input_tensor_size = tensor_sizes[0]
        output_tensor_size = tensor_sizes[1]

        # Check if the input tensor size is correct given user-fed binding.
        if input_tensor_size != len(input_buffer):
            raise RuntimeError("Invalid size of calculated and expected host buffers for input")

        # Check if the output tensor size is correct given apriori knowledge of the output tensor shape.
        if output_tensor_size != batch_size * k_output_size:
            raise RuntimeError("Invalid size of calculated and expected device buffers for output")

        def allocate_binding(tensor_size):
            # Allocate GPU memory for input and output bindings.
            return cuda_assert(cudart.cudaMallocAsync(tensor_size * k_bytes_per_float, stream))

        bindings = list(map(allocate_binding, tensor_sizes))
        output_buffer = np.zeros(output_tensor_size, dtype=np.float32)

        try:
            # Set the tensor addresses.
            def set_tensor_address(tensor_name, binding):
                # Specify the tensor addresses.
                return self.context.set_tensor_address(tensor_name, binding)

            # Set the tensor addresses.
            set_tensor_addresses_success = all(map(lambda x: set_tensor_address(*x), zip(tensor_names, bindings)))

            if not set_tensor_addresses_success:
                raise RuntimeError("Failed to set tensor addresses!")

            # Copy the input data to the device.
            cuda_assert(
                cudart.cudaMemcpyAsync(
                    bindings[0],
                    input_buffer.ctypes.data,
                    len(input_buffer) * k_bytes_per_float,
                    cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                    stream,
                )
            )

            # Enqueue the inference.
            execute_success = self.context.execute_async_v3(stream_handle=stream)
            if not execute_success:
                raise RuntimeError("Failed to execute inference!")

            # Copy the output data from the device to host.
            cuda_assert(
                cudart.cudaMemcpyAsync(
                    output_buffer.ctypes.data,
                    bindings[1],
                    len(output_buffer) * k_bytes_per_float,
                    cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                    stream,
                )
            )

            cuda_assert(cudart.cudaStreamSynchronize(stream))
        finally:
            # Free the GPU memory.
            for binding in bindings:
                cuda_assert(cudart.cudaFreeAsync(binding, stream))

        return output_buffer


def run_inference(serialized_engine, fc1_weights, fc2_weights):
    runtime = trt.Runtime(logger)
    if not runtime:
        raise RuntimeError("Failed to create runtime!")

    # Deserialize the engine.
    inference_engine = runtime.deserialize_cuda_engine(serialized_engine)
    if not inference_engine:
        raise RuntimeError("Failed to deserialize engine!")

    # Create a refitter object for the engine.
    refitter = trt.Refitter(inference_engine, logger)
    if not refitter:
        raise RuntimeError("Failed to create refitter!")

    # Refit fc1 and fc2 weights in the engine.
    fc1_refit_success = refitter.set_named_weights("fc1 weights", fc1_weights)
    fc2_refit_success = refitter.set_named_weights("fc2 weights", fc2_weights)
    if not (fc1_refit_success and fc2_refit_success):
        raise RuntimeError("Failed to set named weights!")

    refit_success = refitter.refit_cuda_engine()
    if not refit_success:
        raise RuntimeError("Failed to refit engine!")

    print("Successfully refit the engine.")

    # Optional settings to configure the behavior of the inference runtime.
    runtime_config = inference_engine.create_runtime_config()
    if not runtime_config:
        raise RuntimeError("Failed to create runtime config!")

    # Create a runtime cache.
    runtime_cache = runtime_config.create_runtime_cache()
    if not runtime_cache:
        raise RuntimeError("Failed to create runtime cache!")

    # Set the runtime cache in runtime configuration.
    set_runtime_cache_success = runtime_config.set_runtime_cache(runtime_cache)
    if not set_runtime_cache_success:
        raise RuntimeError("Failed to set runtime cache!")

    use_optional_advanced_dynamic_shapes_api(runtime_config, inference_engine)

    # Create an engine execution context out of the deserialized engine.
    # TRT-RTX performs "Just-in-Time" (JIT) optimization here, targeting the current GPU.
    # JIT phase is faster than AOT phase, and typically completes in under 15 seconds.
    context = inference_engine.create_execution_context(runtime_config)
    if not context:
        raise RuntimeError("Failed to create execution context!")

    # Helper to perform inference for different batch sizes.
    inference_context = InferenceContext(inference_engine, context)

    # Run inference with different batch sizes
    batch_sizes = [
        k_min_batch_size,
        (k_min_batch_size + k_opt_batch_size) // 2,
        k_opt_batch_size,
        (k_opt_batch_size + k_max_batch_size) // 2,
        k_max_batch_size,
    ]

    # Create a stream for asynchronous execution.
    stream = cuda_assert(cudart.cudaStreamCreate())

    try:
        for batch_size in batch_sizes:
            print(f"Running inference with batch size: {batch_size}")

            # Create input data for this batch size
            input_buffer = np.zeros(batch_size * k_input_size, dtype=np.float32)
            indices = np.arange(batch_size * k_input_size) % k_input_size
            input_buffer[:] = batch_size + batch_size * indices

            # Set shapes, allocate data on GPU, run inference and finally copy output back to host.
            output_buffer = inference_context.run_inference(input_buffer, batch_size, stream)
            cuda_assert(cudart.cudaStreamSynchronize(stream))

            if output_buffer is None:
                raise RuntimeError(f"Failed to run inference for batch size: {batch_size}")

            print("Input:", input_buffer)
            print("Output:", output_buffer)
    finally:
        cuda_assert(cudart.cudaStreamDestroy(stream))

    print("Successfully ran the network with dynamic shapes.")

    # Now that we have finished running inference and we want to shut down our
    # application, we can serialize the runtime cache. Normally here we would
    # save the serialized cache to persistent storage using the
    # IHostMemory interface consisting of data(), size() and type().

    # Serialize the runtime cache.
    serialized_runtime_cache = runtime_cache.serialize()
    if not serialized_runtime_cache:
        raise RuntimeError("Failed to serialize runtime cache!")

    print("Successfully serialized the runtime cache.")
    print(f"Cache size in bytes: {serialized_runtime_cache.nbytes}")


def main():
    # The data backing IConstantLayers must remain valid until the engine has
    # been built and then refit. Therefore we first create weights data, and then
    # individual weights, which are kept alive until after the engine is built and
    # then passed to the refitter.

    # The weights in this example are initialized to 1.0f, but typically would
    # be loaded from a file or other source.
    fc1_weights_data = np.ones(k_input_size * k_hidden_size, dtype=np.float32)
    fc2_weights_data = np.ones(k_hidden_size * k_output_size, dtype=np.float32)

    fc1_weights = trt.Weights(fc1_weights_data)
    fc2_weights = trt.Weights(fc2_weights_data)

    serialized_engine = create_serialized_engine(fc1_weights, fc2_weights)
    if not serialized_engine:
        raise RuntimeError("Failed to build serialized engine!")
    print(f"Successfully built the network. Engine size: {serialized_engine.nbytes} bytes.")

    run_inference(serialized_engine, fc1_weights, fc2_weights)


if __name__ == "__main__":
    main()
