# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# A small MLP (input -> matmul -> ReLU -> matmul -> output) with a dynamic batch dimension.
K_INPUT_FEATURES = 512
K_HIDDEN_FEATURES = 1024
K_OUTPUT_FEATURES = 512

# Min / opt / max for the dynamic batch dimension.
K_MIN_BATCH = 1
K_OPT_BATCH = 256
K_MAX_BATCH = 512

# FP16 keeps the GEMMs runnable on Turing GPUs, which do not support FP32 matrix multiplies.
K_BYTES_PER_HALF = 2

# cudaMalloc satisfies the 256-byte alignment TensorRT-RTX requires for device memory.
K_GPU_ALLOCATION_ALIGNMENT = 256

K_INPUT_NAME = "input"
K_OUTPUT_NAME = "output"

logger = trt.Logger(trt.Logger.WARNING)


# Unpack a cuda-python call, raising on a non-success status.
def cuda_assert(call: tuple) -> object:
    err = call[0]
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA error: {err}")
    return call[1] if len(call) > 1 else None


# A synchronous GPU allocator backed by cudaMalloc / cudaFree. All entry points are overridden so
# pybind can dispatch to this subclass; it counts allocations so the sample can show TRT-RTX using it.
class SyncGpuAllocator(trt.IGpuAsyncAllocator):
    def __init__(self) -> None:
        super().__init__()
        self.num_allocations = 0
        self.bytes_allocated = 0

    def allocate_async(self, size: int, alignment: int, flags: int, stream: int) -> int:
        # An allocation request of size 0 must return a null (0) pointer.
        if size == 0:
            return 0
        err, ptr = cudart.cudaMalloc(size)
        if err != cudart.cudaError_t.cudaSuccess:
            return 0
        self.num_allocations += 1
        self.bytes_allocated += size
        return int(ptr)

    def deallocate_async(self, memory: int, stream: int) -> bool:
        # TensorRT-RTX may pass a null (0) pointer; that is always a success.
        if not memory:
            return True
        (err,) = cudart.cudaFree(memory)
        return err == cudart.cudaError_t.cudaSuccess

    # The deprecated synchronous entry points forward to the asynchronous ones.
    def allocate(self, size: int, alignment: int, flags: int) -> int:
        return self.allocate_async(size, alignment, flags, 0)

    def deallocate(self, memory: int) -> bool:
        return self.deallocate_async(memory, 0)

    # Resizing is not supported; returning 0 tells TensorRT-RTX to keep the original allocation.
    def reallocate(self, address: int, alignment: int, new_size: int) -> int:
        return 0

    def reallocate_async(self, address: int, alignment: int, new_size: int, stream: int) -> int:
        return 0


# Build the MLP (input -> matmul -> ReLU -> matmul -> output) with a dynamic batch dimension.
def build_serialized_engine(allocator: trt.IGpuAsyncAllocator) -> trt.IHostMemory:
    builder = trt.Builder(logger)

    # Register the allocator before building so build-time allocations go through it.
    builder.gpu_allocator = allocator

    config = builder.create_builder_config()

    # TensorRT-RTX networks are strongly typed, so no creation flags are required.
    network = builder.create_network(0)

    # Dynamic batch dimension (-1) with a fixed feature dimension.
    input_tensor = network.add_input(K_INPUT_NAME, trt.float16, trt.Dims2(-1, K_INPUT_FEATURES))

    fc1_weights = trt.Weights(np.full(K_INPUT_FEATURES * K_HIDDEN_FEATURES, 0.01, dtype=np.float16))
    fc2_weights = trt.Weights(np.full(K_HIDDEN_FEATURES * K_OUTPUT_FEATURES, 0.01, dtype=np.float16))

    fc1_constant = network.add_constant(trt.Dims2(K_INPUT_FEATURES, K_HIDDEN_FEATURES), fc1_weights)
    fc2_constant = network.add_constant(trt.Dims2(K_HIDDEN_FEATURES, K_OUTPUT_FEATURES), fc2_weights)

    # matmul -> ReLU -> matmul; the ReLU keeps the batch-scaling intermediate from folding away.
    fc1 = network.add_matrix_multiply(
        input_tensor, trt.MatrixOperation.NONE, fc1_constant.get_output(0), trt.MatrixOperation.NONE
    )
    relu = network.add_activation(fc1.get_output(0), type=trt.ActivationType.RELU)
    fc2 = network.add_matrix_multiply(
        relu.get_output(0), trt.MatrixOperation.NONE, fc2_constant.get_output(0), trt.MatrixOperation.NONE
    )

    fc2.get_output(0).name = K_OUTPUT_NAME
    network.mark_output(fc2.get_output(0))

    # Optimization profile covering the dynamic batch range.
    profile = builder.create_optimization_profile()
    profile.set_shape(
        K_INPUT_NAME,
        trt.Dims2(K_MIN_BATCH, K_INPUT_FEATURES),
        trt.Dims2(K_OPT_BATCH, K_INPUT_FEATURES),
        trt.Dims2(K_MAX_BATCH, K_INPUT_FEATURES),
    )
    config.add_optimization_profile(profile)

    return builder.build_serialized_network(network, config)


# Run one inference for the shape currently set on the context. The I/O buffers allocated here are
# user-owned and separate from the execution-context activation memory.
def run_inference_for_current_shape(engine: trt.ICudaEngine, context: trt.IExecutionContext, stream: int) -> None:
    # infer_shapes() is the readiness check for the name-based shape API (empty == all shapes set).
    if context.infer_shapes():
        raise RuntimeError("Not all input dimensions are specified.")

    buffers = []
    try:
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            volume = int(np.prod(context.get_tensor_shape(name)))
            ptr = cuda_assert(cudart.cudaMalloc(volume * K_BYTES_PER_HALF))
            # Track the buffer immediately so it is freed even if a later call raises.
            buffers.append(ptr)
            # Zero-initialize inputs for deterministic output; outputs are overwritten by inference.
            cuda_assert(cudart.cudaMemset(ptr, 0, volume * K_BYTES_PER_HALF))
            if not context.set_tensor_address(name, ptr):
                raise RuntimeError("Failed to set tensor address.")

        if not context.execute_async_v3(stream_handle=stream):
            raise RuntimeError("Failed to run inference.")
        cuda_assert(cudart.cudaStreamSynchronize(stream))
    finally:
        # Wait for any in-flight inference before freeing the synchronous allocations.
        cuda_assert(cudart.cudaStreamSynchronize(stream))
        for ptr in buffers:
            cuda_assert(cudart.cudaFree(ptr))


# Strategy 1: TensorRT-RTX owns the activation memory (STATIC, the default).
def demonstrate_trt_managed_memory(engine: trt.ICudaEngine, allocator: SyncGpuAllocator, stream: int) -> None:
    print("\n=== Strategy 1: TensorRT-RTX-managed memory (STATIC) ===")

    allocations_before = allocator.num_allocations

    # STATIC pre-allocates a block for the worst case across profiles, via our allocator.
    context = engine.create_execution_context()

    print(f"TensorRT-RTX reserved {engine.device_memory_size_v2} bytes (worst case across all profiles).")
    print(
        f"The custom allocator serviced {allocator.num_allocations - allocations_before} allocation(s) while "
        "creating the context."
    )

    # Run inference at the largest shape to exercise the reserved memory.
    if not context.set_input_shape(K_INPUT_NAME, trt.Dims2(K_MAX_BATCH, K_INPUT_FEATURES)):
        raise RuntimeError("Failed to set input shape.")
    run_inference_for_current_shape(engine, context, stream)


# Strategy 2: the application owns the activation memory (USER_MANAGED).
def demonstrate_user_managed_memory(engine: trt.ICudaEngine, allocator: SyncGpuAllocator, stream: int) -> None:
    print("\n=== Strategy 2: user-managed memory (USER_MANAGED) ===")

    runtime_config = engine.create_runtime_config()
    runtime_config.set_execution_context_allocation_strategy(trt.ExecutionContextAllocationStrategy.USER_MANAGED)

    context = engine.create_execution_context(runtime_config)

    worst_case_bytes = engine.device_memory_size_v2
    print(f"Worst-case device memory across all profiles: {worst_case_bytes} bytes.")

    for batch in (K_MIN_BATCH, K_OPT_BATCH, K_MAX_BATCH):
        # Select the optimization profile and set all input shapes before querying the size.
        if not context.set_optimization_profile_async(0, stream) or not context.set_input_shape(
            K_INPUT_NAME, trt.Dims2(batch, K_INPUT_FEATURES)
        ):
            raise RuntimeError("Failed to set profile/input shape.")
        cuda_assert(cudart.cudaStreamSynchronize(stream))

        # Size the memory for exactly these shapes (0 if profile/shapes were not set, or none needed).
        bytes_for_shape = context.update_device_memory_size_for_shapes()

        # Allocate it ourselves through the same allocator; a size of 0 -> null (0) is valid.
        device_memory = (
            allocator.allocate_async(bytes_for_shape, K_GPU_ALLOCATION_ALIGNMENT, 0, stream) if bytes_for_shape else 0
        )
        if bytes_for_shape and not device_memory:
            raise RuntimeError("Failed to allocate device memory.")
        context.set_device_memory(device_memory, bytes_for_shape)

        percent = 0 if worst_case_bytes == 0 else 100 * bytes_for_shape // worst_case_bytes
        print(f"batch {batch}: needs {bytes_for_shape} bytes ({percent}% of the worst case).")

        try:
            run_inference_for_current_shape(engine, context, stream)
        finally:
            # Release the memory only after inference has completed.
            allocator.deallocate_async(device_memory, stream)
            cuda_assert(cudart.cudaStreamSynchronize(stream))


def main() -> None:
    # The allocator must outlive every TensorRT-RTX object that uses it.
    allocator = SyncGpuAllocator()

    serialized_engine = build_serialized_engine(allocator)
    if not serialized_engine:
        raise RuntimeError("Failed to build the engine.")
    print(f"Built the engine ({serialized_engine.nbytes} bytes).")

    runtime = trt.Runtime(logger)

    # Register the custom allocator so runtime and execution-context allocations use it.
    runtime.gpu_allocator = allocator

    engine = runtime.deserialize_cuda_engine(serialized_engine)
    if not engine:
        raise RuntimeError("Failed to deserialize the engine.")

    stream = cuda_assert(cudart.cudaStreamCreate())
    try:
        demonstrate_trt_managed_memory(engine, allocator, stream)
        demonstrate_user_managed_memory(engine, allocator, stream)
    finally:
        cuda_assert(cudart.cudaStreamDestroy(stream))

    print(
        f"\nDone. The custom allocator serviced {allocator.num_allocations} allocation(s), "
        f"{allocator.bytes_allocated} bytes in total."
    )


if __name__ == "__main__":
    main()
