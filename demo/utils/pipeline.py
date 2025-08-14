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


"""
Simple Pipeline for ONNX -> TRT-RTX flow
"""

import gc
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Optional

import cuda.bindings.runtime as cudart
import torch

from .model_registry import registry
from .path_manager import PathManager
from .timing_data import InferenceTimingData

# Initialize logger for this module
logger = logging.getLogger("rtx_demo.utils.pipeline")


class Pipeline(ABC):
    """
    Simple pipeline for ONNX -> TRT-RTX flow.

    Usage:
    1. Create Pipeline with precision config
    2. Load Engines (builds if needed)
    3. Activate Engines (load into memory)
    4. Load Resources (allocate buffers)
    5. Run Inference
    """

    def __init__(
        self,
        pipeline_name: str,
        cache_dir: str = "./demo_cache",
        device: str = "cuda",
        verbose: bool = True,
        cache_mode: str = "full",
        enable_timing: bool = True,
        hf_token: Optional[str] = None,
        low_vram: bool = False,
        log_level: str = "INFO",
        enable_runtime_cache: bool = False,
    ):
        """
        Initialize pipeline.

        Args:
            pipeline_name: Name of the pipeline
            cache_dir: Directory for caching models
            device: Device to run on
            verbose: Enable verbose logging
            cache_mode: Cache management mode ("full" or "lean")
            enable_timing: Enable verbose timing
            hf_token: Hugging Face token
            low_vram: Enable low VRAM mode
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            enable_runtime_cache: Enable use of serialized runtime cache to improve JIT compilation times
        """
        # Configure logging FIRST, before any other operations
        self.configure_logging(verbose, log_level)

        self.pipeline_name = pipeline_name

        # Initialize path manager
        self.path_manager = PathManager(cache_dir=cache_dir, cache_mode=cache_mode)

        # Core attributes
        self.device = device
        self.verbose = verbose
        self.hf_token = hf_token
        self.low_vram = low_vram

        if enable_runtime_cache:
            self.runtime_cache_path = os.path.join(cache_dir, "runtime.cache")
        else:
            self.runtime_cache_path = None

        # Pipeline state
        self.engines = {}
        self.model_instances = {}
        self.shape_dicts = {}
        self.shape_config = {}
        self.current_shapes = {}
        self.stream = None
        self.shared_device_memory = None

        # Get default precision config from registry
        self.precision_config = registry.get_default_precisions(pipeline_name)

        # Timing infrastructure
        self.enable_timing = enable_timing
        self.timing_data = InferenceTimingData()
        self._cuda_events = {}

        logger.debug(f"Pipeline: {pipeline_name}")
        logger.debug(f"Initial Precision Config: {self.precision_config}")

    def configure_logging(self, verbose: bool = True, log_level: str = "INFO", log_format: Optional[str] = None):
        """Configure centralized logging for the pipeline and all child modules.

        Args:
            verbose: Whether to enable verbose logging
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_format: Custom log format string. If None, uses a default format.
        """
        # Set up the default log format if none provided
        if log_format is None:
            log_format = "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"

        # Determine the effective logging level
        effective_level = getattr(logging, log_level.upper(), logging.INFO)

        # Create formatter
        formatter = logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")

        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(effective_level)
        console_handler.setFormatter(formatter)

        # Configure the main rtx_demo logger namespace
        rtx_logger = logging.getLogger("rtx_demo")
        rtx_logger.handlers.clear()  # Clear any existing handlers
        rtx_logger.addHandler(console_handler)
        rtx_logger.setLevel(effective_level)
        rtx_logger.propagate = False  # Don't propagate to avoid duplicate messages

        # Directly set level on ALL rtx_demo.* loggers (more reliable than inheritance)
        all_logger_names = list(logging.getLogger().manager.loggerDict.keys())
        for logger_name in all_logger_names:
            if logger_name.startswith("rtx_demo."):
                child_logger = logging.getLogger(logger_name)
                child_logger.handlers.clear()  # Remove any child handlers
                child_logger.setLevel(effective_level)  # Set level directly
                child_logger.propagate = True  # Allow propagation to parent

        logger.debug(f"Logging configured - Level: {log_level.upper()}, Verbose: {verbose}")

    def _create_cuda_events(self, name: str) -> tuple[torch.cuda.Event, torch.cuda.Event]:
        """Create and cache CUDA events for timing."""
        if name not in self._cuda_events:
            start_event, end_event = cudart.cudaEventCreate()[1], cudart.cudaEventCreate()[1]
            self._cuda_events[name] = (start_event, end_event)
        return self._cuda_events[name]

    def _record_cuda_timing(self, name: str, func: Callable, *args, **kwargs) -> tuple[Any, Optional[float]]:
        """
        Record CUDA timing for a function call using events.

        Args:
            name: Name for the timing event
            func: Function to time
            *args, **kwargs: Arguments for the function

        Returns:
            Function result and elapsed time in milliseconds
        """
        if not self.enable_timing:
            return func(*args, **kwargs), None

        start_event, end_event = self._create_cuda_events(name)

        # Ensure CUDA is synchronized before starting
        cudart.cudaEventRecord(start_event, self.stream)

        result = func(*args, **kwargs)

        cudart.cudaEventRecord(end_event, self.stream)
        cudart.cudaStreamSynchronize(self.stream)

        elapsed_time = cudart.cudaEventElapsedTime(start_event, end_event)[1]

        # Append timing to pipeline_times dictionary
        self.timing_data.pipeline_times[name].append(elapsed_time)

        return result, elapsed_time

    def reset_timing_data(self):
        """Reset timing data for a new inference run."""
        self.timing_data = InferenceTimingData()

    def get_model_names(self) -> list:
        """Return list of model names used by this pipeline"""
        raise NotImplementedError("Subclasses must implement get_model_names")

    def initialize_models(self):
        """Initialize model objects - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement initialize_models")

    @abstractmethod
    def load_engines(self) -> None:
        """Load TensorRT engines, building if necessary."""
        raise NotImplementedError("Subclasses must implement load_engines")

    def activate_engines(self, shared_device_memory: Optional[int] = None) -> dict[str, float]:
        """Activate all engines (load into memory)"""
        jit_times = {}

        if shared_device_memory is None:
            max_device_memory = self.calculate_max_device_memory()
            _, shared_device_memory = cudart.cudaMalloc(max_device_memory)

        self.shared_device_memory = shared_device_memory

        logger.debug("Activating engines...")

        # Load and activate TensorRT engines
        runtime_cache = None
        for model_name, engine in self.engines.items():
            engine.runtime_cache = runtime_cache
            logger.debug(f"  Activating {model_name}")

            jit_time = engine.activate(device_memory=self.shared_device_memory)
            jit_times[model_name] = jit_time
            if runtime_cache is None:
                runtime_cache = engine.runtime_cache

        total_jit_time = sum(jit_times.values())
        logger.debug(f"All engines activated (total JIT time: {total_jit_time:.3f}s)")

        return jit_times

    @abstractmethod
    def load_resources(self, **shape_params) -> None:
        """
        Allocate buffers for inference.

        Args:
            **shape_params: Shape parameters (e.g., batch_size=1, height=512, width=512)
        """
        raise NotImplementedError("Subclasses must implement load_resources")

    def calculate_max_device_memory(self) -> int:
        """Calculate maximum device memory needed across all engines"""
        if not self.engines:
            return 0

        max_device_memory = 0
        total_engine_memory = 0

        logger.debug("[MEMORY] Calculating shared workspace requirements:")

        for model_name, engine in self.engines.items():
            engine_memory = engine.engine.device_memory_size_v2
            total_engine_memory += engine_memory
            max_device_memory = max(max_device_memory, engine_memory)

            logger.debug(f"[MEMORY]   {model_name}: {engine_memory / (1024**3):.3f} GB")

        workspace_savings = (total_engine_memory - max_device_memory) / (1024**3)
        logger.debug(f"[MEMORY] Shared workspace (max): {max_device_memory / (1024**3):.3f} GB")
        logger.debug(f"[MEMORY] Memory savings: {workspace_savings:.3f} GB")

        return max_device_memory

    def run_engine(self, model_name: str, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Run inference on a specific engine"""
        engine = self.engines[model_name]
        return engine.infer(inputs, self.stream, use_cuda_graph=False)

    def infer(self, *args, **kwargs):
        """Run the full pipeline inference - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement infer")

    def cleanup(self) -> None:
        """Clean up all resources"""
        logger.debug("Cleaning up pipeline...")

        # Clear CUDA Events
        for start_event, end_event in self._cuda_events.values():
            cudart.cudaEventDestroy(start_event)
            cudart.cudaEventDestroy(end_event)
            del start_event
            del end_event

        # Deactivate engines
        for model_name, engine in self.engines.items():
            logger.debug(f"  Deactivating {model_name}")
            engine.deallocate_buffers()
            engine.deactivate()
            engine.unload()
            del engine

        if self.shared_device_memory is not None:
            logger.info("[MEMORY] Freeing shared workspace memory")
            cudart.cudaFree(self.shared_device_memory)
            self.shared_device_memory = None

        if self.stream is not None:
            cudart.cudaStreamDestroy(self.stream)
            del self.stream
            self.stream = None

        for model in self.model_instances.values():
            del model

        gc.collect()
        torch.cuda.empty_cache()

        # Clear state
        self.engines.clear()
        self.model_instances.clear()
        self._cuda_events.clear()
        self.current_shapes.clear()

        # Clear GPU memory
        logger.info("[MEMORY] Clearing GPU cache")
        gc.collect()
        torch.cuda.empty_cache()

    def __del__(self):
        self.cleanup()

    def calculate_total_gpu_vram_usage(self) -> dict:
        """Calculate total GPU VRAM usage estimate using formula: weights + workspace + buffers"""
        # Model weights (serialized engine sizes)
        total_weights = 0
        engine_weights = {}

        for model_name, engine in self.engines.items():
            engine_path = Path(engine.engine_path)
            if engine_path.exists():
                weight_size = engine_path.stat().st_size
                engine_weights[model_name] = weight_size
                total_weights += weight_size

        # Shared workspace memory
        max_workspace = self.calculate_max_device_memory()

        # Buffer allocations
        total_buffers = 0
        buffer_breakdown = {}

        for model_name, engine in self.engines.items():
            model_buffer_size = 0
            for tensor in engine.tensors.values():
                model_buffer_size += tensor.numel() * tensor.element_size()
            buffer_breakdown[model_name] = model_buffer_size
            total_buffers += model_buffer_size

        # Total VRAM estimate
        total_vram = total_weights + max_workspace + total_buffers

        return {
            "total_weights_gb": total_weights / (1024**3),
            "max_workspace_gb": max_workspace / (1024**3),
            "total_buffers_gb": total_buffers / (1024**3),
            "total_vram_gb": total_vram / (1024**3),
            "breakdown": {
                "weights": {name: size / (1024**3) for name, size in engine_weights.items()},
                "buffers": {name: size / (1024**3) for name, size in buffer_breakdown.items()},
            },
        }

    def print_gpu_vram_summary(self):
        """Print GPU VRAM usage summary"""
        vram_info = self.calculate_total_gpu_vram_usage()

        logger.info(f"[VRAM] Total estimated usage: {vram_info['total_vram_gb']:.2f} GB")
        logger.info(f"[VRAM]   Model weights: {vram_info['total_weights_gb']:.2f} GB")
        logger.info(f"[VRAM]   Shared workspace: {vram_info['max_workspace_gb']:.2f} GB")
        logger.info(f"[VRAM]   Buffer allocations: {vram_info['total_buffers_gb']:.2f} GB")

        logger.info("[VRAM] Per-model breakdown:")
        for model_name in self.engines:
            weights = vram_info["breakdown"]["weights"].get(model_name, 0)
            buffers = vram_info["breakdown"]["buffers"].get(model_name, 0)
            logger.info(f"[VRAM]   {model_name}: {weights:.2f}GB weights + {buffers:.2f}GB buffers")

        return vram_info
