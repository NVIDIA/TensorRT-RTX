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

# Disable ruff import order check for this file
# because we need to set POLYGRAPHY_USE_TENSORRT_RTX before importing polygraphy
# ruff: noqa: E402
import gc
import logging
import os
import subprocess
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional

import cuda.bindings.runtime as cudart
import tensorrt_rtx as trt
import torch

from .engine_metadata import metadata_manager

# Initialize logger for this module
logger = logging.getLogger("rtx_demo.utils.engine")

# Ensure POLYGRAPHY_USE_TENSORRT_RTX environment variable is set before importing polygraphy
if os.environ.get("POLYGRAPHY_USE_TENSORRT_RTX") != "1":
    logger.info("Setting POLYGRAPHY_USE_TENSORRT_RTX=1 to enable TensorRT-RTX features in polygraphy.")
    logger.info("You can set this environment variable manually to avoid this message.")
    os.environ["POLYGRAPHY_USE_TENSORRT_RTX"] = "1"

# Check polygraphy version requirement
import polygraphy
from packaging import version

if version.parse(polygraphy.__version__) < version.parse("0.49.24"):
    raise ImportError(f"polygraphy version {polygraphy.__version__} is too old. Please upgrade to >= 0.49.24")

from polygraphy.backend.common import bytes_from_path
from polygraphy.backend.trt import engine_from_bytes
from polygraphy.util import LockFile, load_file, save_file

# Map of TensorRT dtype -> torch dtype
trt_to_torch_dtype_dict = {
    trt.DataType.BOOL: torch.bool,
    trt.DataType.UINT8: torch.uint8,
    trt.DataType.INT8: torch.int8,
    trt.DataType.INT32: torch.int32,
    trt.DataType.INT64: torch.int64,
    trt.DataType.HALF: torch.float16,
    trt.DataType.FLOAT: torch.float32,
    trt.DataType.BF16: torch.bfloat16,
}


def _CUASSERT(cuda_ret):
    err = cuda_ret[0]
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(
            f"CUDA ERROR: {err}, error code reference: https://nvidia.github.io/cuda-python/module/cudart.html#cuda.cudart.cudaError_t"
        )
    if len(cuda_ret) > 1:
        return cuda_ret[1]
    return None


class Engine:
    def __init__(
        self,
        engine_path: Path,
        precision: str,
        model_name: str,
        runtime_cache_path: Optional[str] = None,
        cuda_graph_strategy: str = "disabled",
    ):
        self.engine_path = engine_path
        self.engine = None
        self.context = None
        self.tensors = OrderedDict()
        self.precision = precision
        self.model_name = model_name
        self.runtime_config = None
        self.runtime_cache = None
        self.runtime_cache_path = runtime_cache_path
        self.cuda_graph_strategy = cuda_graph_strategy

    def __del__(self):
        del self.tensors
        del self.context
        del self.engine
        del self.runtime_cache
        del self.runtime_config

    def _validate_input_profile(self, input_profile: dict[str, Any]):
        """Validate input profile formatting"""
        if not isinstance(input_profile, dict) or not input_profile:
            raise ValueError("input_profile must be a non-empty dictionary")

        is_static = None  # Will be determined from first entry

        for name, dims in input_profile.items():
            if not isinstance(name, str):
                raise ValueError(f"Key '{name}' must be a string")
            if not isinstance(dims, (list, tuple)):
                raise ValueError(f"Value for '{name}' must be a list or tuple")

            # Determine pattern from first entry
            if is_static is None:
                is_static = all(isinstance(d, int) for d in dims)
                if not is_static:  # noqa: SIM102
                    # Must be dynamic: exactly 3 lists/tuples of integers
                    if len(dims) != 3 or not all(
                        isinstance(d, (list, tuple)) and all(isinstance(x, int) for x in d) for d in dims
                    ):
                        raise ValueError(f"'{name}' must be integers (static) or 3 lists of integers (dynamic)")

            # Validate subsequent entries match the pattern
            elif is_static:
                if not all(isinstance(d, int) for d in dims):
                    raise ValueError(f"Mixed schemes: '{name}' should be integers like first entry")
            else:  # dynamic
                if len(dims) != 3 or not all(
                    isinstance(d, (list, tuple)) and all(isinstance(x, int) for x in d) for d in dims
                ):
                    raise ValueError(f"Mixed schemes: '{name}' should be 3 lists of integers like first entry")

        return is_static

    def build(
        self,
        onnx_path: str,
        input_profile: dict[str, Any],
        static_shape: bool = True,
        verbose: bool = False,
        extra_args: Optional[set[str]] = None,
    ):
        """Build TensorRT engine from ONNX model"""
        logger.info(f"Building TensorRT engine for {onnx_path}: {self.engine_path}")

        # Validate input profile
        is_static = self._validate_input_profile(input_profile)
        assert is_static == static_shape, (
            f"Input profile and static_shape mismatch: Input profile is {'static' if is_static else 'dynamic'} while expected shape is {'static' if static_shape else 'dynamic'}"
        )

        # Build command with arguments
        build_command = [f"polygraphy convert {onnx_path} --convert-to trt --use-gpu --output {self.engine_path}"]

        build_args = []
        verbosity = "extra_verbose" if verbose else "error"
        build_args.append(f"--verbosity {verbosity}")

        # Build shape arguments
        if is_static:
            input_shapes = []
            for name, dims in input_profile.items():
                input_shapes.append(f"{name}:{str(list(dims)).replace(' ', '')}")
            build_args.append(f"--input-shapes {' '.join(input_shapes)}")
        else:  # dynamic
            min_shapes, opt_shapes, max_shapes = [], [], []
            for name, dims in input_profile.items():
                min_shapes.append(f"{name}:{str(list(dims[0])).replace(' ', '')}")
                opt_shapes.append(f"{name}:{str(list(dims[1])).replace(' ', '')}")
                max_shapes.append(f"{name}:{str(list(dims[2])).replace(' ', '')}")
            build_args.extend(
                [
                    f"--trt-min-shapes {' '.join(min_shapes)}",
                    f"--trt-opt-shapes {' '.join(opt_shapes)}",
                    f"--trt-max-shapes {' '.join(max_shapes)}",
                ]
            )

        # Add extra arguments
        if extra_args:
            if isinstance(extra_args, set):
                build_args.extend(extra_args)
            else:
                raise ValueError("extra_args must be a set of strings")

        # Execute build command
        build_args = [arg for arg in build_args if arg]
        final_command = " ".join(build_command + build_args)

        try:
            logger.debug(f"Engine build command: {final_command}")

            subprocess.run(final_command, check=True, shell=True)

            # Save metadata after successful build
            metadata_manager.save_metadata(
                engine_path=self.engine_path,
                model_name=self.model_name,
                precision=self.precision,
                onnx_path=onnx_path,
                input_shapes=input_profile,
                extra_args=extra_args,
            )

        except subprocess.CalledProcessError as exc:
            error_msg = f"Failed to build TensorRT engine. Error details:\nCommand: {exc.cmd}\n"
            raise RuntimeError(error_msg) from exc

    def load(self):
        """Load TensorRT engine from file"""
        if self.engine is not None:
            logger.warning(f"Engine {self.engine_path} already loaded, skip reloading")
            return

        engine_bytes = bytes_from_path(self.engine_path)

        # Load engine and calculate memory requirements
        serialized_size_gb = len(engine_bytes) / (1024**3)
        logger.debug(f"Serialized weights: {serialized_size_gb:.2f} GB")

        self.engine = engine_from_bytes(engine_bytes)

        device_memory_gb = self.engine.device_memory_size_v2 / (1024**3)
        logger.debug(f"Device memory: {device_memory_gb:.2f} GB")
        logger.debug(f"Total VRAM estimate: {(serialized_size_gb + device_memory_gb):.2f} GB")

        # Print tensor summary
        logger.debug(f"I/O tensors: {self.engine.num_io_tensors}")
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            dtype = self.engine.get_tensor_dtype(name)
            mode = self.engine.get_tensor_mode(name)
            logger.debug(f"  {name}: {shape} {dtype} ({mode})")

    def unload(self):
        """Unload TensorRT engine"""
        if self.engine is not None:
            logger.info(f"Unloading TensorRT engine: {self.engine_path}")
            del self.engine
            self.engine = None
            del self.runtime_cache
            self.runtime_cache = None
            gc.collect()
            torch.cuda.empty_cache()
        else:
            logger.warning(f"Unload an unloaded engine {self.engine_path}, skip unloading")

    def activate(self, device_memory: Optional[int] = None, defer_memory_allocation: bool = False):
        """Create execution context"""

        self.runtime_config = self.engine.create_runtime_config()

        if self.cuda_graph_strategy == "whole_graph_capture":
            self.runtime_config.cuda_graph_strategy = trt.CudaGraphStrategy.WHOLE_GRAPH_CAPTURE

        if self.runtime_cache_path:
            if self.runtime_cache is None:
                logger.debug("Creating runtime cache")
                self.runtime_cache = self.runtime_config.create_runtime_cache()
                try:
                    if os.path.exists(self.runtime_cache_path):
                        logger.info(f"Loading runtime cache from {self.runtime_cache_path}")
                        with LockFile(self.runtime_cache_path):
                            cache_bytes = load_file(self.runtime_cache_path)
                            if cache_bytes:
                                logger.info("Deserializing runtime cache")
                                self.runtime_cache.deserialize(cache_bytes)
                            else:
                                logger.info("No runtime cache loaded.")
                except Exception as e:
                    logger.warning(f"Error loading runtime cache from {self.runtime_cache_path}: {e}")
            else:
                logger.info("Using existing runtime cache")

            self.runtime_config.set_runtime_cache(self.runtime_cache)

        time_start = time.time()
        if device_memory is not None:
            self.runtime_config.set_execution_context_allocation_strategy(trt.ExecutionContextAllocationStrategy(2))
            self.context = self.engine.create_execution_context(self.runtime_config)
            self.context.device_memory = device_memory
        elif defer_memory_allocation:
            self.runtime_config.set_execution_context_allocation_strategy(trt.ExecutionContextAllocationStrategy(2))
            self.context = self.engine.create_execution_context(self.runtime_config)
            logger.debug(f"[MEMORY] {self.model_name}: Deferred memory allocation")
        else:
            self.context = self.engine.create_execution_context(self.runtime_config)
            logger.debug(
                f"[MEMORY] {self.model_name}: Using individual workspace {self.engine.device_memory_size_v2 / (1024**3):.3f} GB"
            )

        time_end = time.time()

        model_name, precision = self.engine_path.parts[-3:-1]
        jit_time = round(time_end - time_start, 3)
        logger.debug(
            f"[I] JIT Compilation + Execution Context Creation Time for {model_name} in {precision}: {jit_time} seconds"
        )
        return jit_time

    def reactivate(self, device_memory: int):
        """Reactivate context with new device memory"""
        assert self.context
        self.context.device_memory = device_memory

    def deactivate(self):
        """Deactivate execution context"""
        if self.runtime_cache_path and self.runtime_cache is not None:
            logger.debug(f"Saving runtime cache to {self.runtime_cache_path}")
            with LockFile(self.runtime_cache_path), self.runtime_cache.serialize() as buffer:
                save_file(buffer, self.runtime_cache_path, description="Runtime cache")
        else:
            logger.debug(f"No runtime cache to save: {self.runtime_cache_path}")

        del self.runtime_cache
        self.runtime_cache = None

        del self.runtime_config
        self.runtime_config = None

        del self.context
        self.context = None

    def allocate_buffers(self, shape_dict: Optional[dict[str, Any]] = None, device: str = "cuda"):
        """Allocate input/output buffers for inference"""
        total_buffer_memory = 0

        logger.debug(f"[MEMORY] Allocating buffers for {self.model_name}:")

        for binding in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(binding)

            # Determine tensor shape
            if shape_dict and name in shape_dict:
                shape = shape_dict[name]
            else:
                shape = self.engine.get_tensor_shape(name)
                logger.warning(f"Could not find '{name}' in shape dict. Using shape {shape} from engine.")

            # Set input shape for dynamic inputs
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.context.set_input_shape(name, shape)

            # Create tensor
            dtype = trt_to_torch_dtype_dict[self.engine.get_tensor_dtype(name)]

            # Handle shape format (list containing tuple vs direct tuple)
            actual_shape = shape[0] if isinstance(shape, list) and len(shape) == 1 else shape
            tensor = torch.empty(tuple(actual_shape), dtype=dtype, device=device)

            # Track memory usage
            tensor_memory = tensor.numel() * tensor.element_size()
            total_buffer_memory += tensor_memory

            mode_str = "INPUT" if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT else "OUTPUT"
            logger.debug(f"[MEMORY]   {name} ({mode_str}): {actual_shape} {dtype} = {tensor_memory / (1024**2):.2f} MB")

            self.tensors[name] = tensor

        total_gb = total_buffer_memory / (1024**3)
        combined_gb = (total_buffer_memory + self.engine.device_memory_size_v2) / (1024**3)
        logger.debug(f"[MEMORY] Total buffers: {total_gb:.3f} GB")
        logger.debug(f"[MEMORY] Combined (buffers + engine): {combined_gb:.3f} GB")

    def deallocate_buffers(self):
        """Deallocate all buffers and free memory"""
        logger.debug(f"[MEMORY] Deallocating buffers for {self.model_name}")

        if self.engine is None:
            return

        for idx in range(self.engine.num_io_tensors):
            binding = self.engine[idx]
            if binding in self.tensors:
                tensor = self.tensors[binding]
                del tensor
                del self.tensors[binding]

        if self.tensors:
            logger.warning(f"I/O buffers not empty after deallocation: {self.tensors.keys()}")

        gc.collect()
        torch.cuda.empty_cache()

    def infer(self, feed_dict: dict[str, Any], stream: torch.cuda.Stream):
        """Run inference with the engine"""
        # Copy input data to tensors
        for name, buf in feed_dict.items():
            self.tensors[name].copy_(buf)

        # Set tensor addresses in context
        for name, tensor in self.tensors.items():
            self.context.set_tensor_address(name, tensor.data_ptr())

        # Execute inference
        noerror = self.context.execute_async_v3(stream)
        if not noerror:
            raise ValueError(f"ERROR: Inference with {self.engine_path} failed.")

        return self.tensors
