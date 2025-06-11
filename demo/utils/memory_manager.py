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


import logging
import time

import cuda.bindings.runtime as cudart

# Initialize logger for this module
logger = logging.getLogger("rtx_demo.utils.memory_manager")


class ModelMemoryManager:
    """
    Context manager for efficiently loading and unloading models to optimize VRAM usage.

    This class provides the following memory optimization mode:

    **Low VRAM Mode**: Just-in-time model loading
    - Load/allocate models only when needed
    - Deallocate immediately after use

    Args:
        pipeline: The pipeline instance containing engines and model instances
        model_names (list): List of model names to manage
        low_vram (bool): Whether to enable low VRAM mode
    """

    def __init__(self, pipeline, model_name, low_vram=False):
        self.pipeline = pipeline
        self.model_name = model_name
        self.low_vram = low_vram
        self.timing = {}

        assert self.model_name in self.pipeline.engines, f"Model {self.model_name} not found in pipeline.engines"
        assert isinstance(self.model_name, str), "model_name must be a string"

    def __enter__(self):
        if not self.low_vram:
            return self
        else:
            return self._enter_low_vram()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.low_vram:
            return
        else:
            return self._exit_low_vram()

    def _enter_low_vram(self):
        """Low VRAM mode entry - load and allocate specified models"""
        logger.debug(f"[MEMORY] Low VRAM: Loading models {self.model_name}...")

        if self.model_name not in self.pipeline.shape_dicts:
            raise RuntimeError(f"Model {self.model_name} not found in pipeline.shape_dicts")

        engine = self.pipeline.engines[self.model_name]
        shape_dict = self.pipeline.shape_dicts[self.model_name]

        start_time = time.time()

        # Load engine
        engine.load()

        # Allocate device memory for this model
        device_memory_size = engine.engine.device_memory_size_v2
        _, device_memory = cudart.cudaMalloc(device_memory_size)
        self.pipeline.shared_device_memory = device_memory

        # Activate engine with allocated memory
        engine.activate(device_memory=device_memory)

        # Allocate buffers
        engine.allocate_buffers(shape_dict, device=self.pipeline.device)

        setup_time = time.time() - start_time
        self.timing[f"{self.model_name}_setup"] = setup_time

        return

    def _exit_low_vram(self):
        """Handle low VRAM mode exit - deallocate specified models"""
        logger.debug(f"[MEMORY] Low VRAM: Deallocating models {self.model_name}...")

        engine = self.pipeline.engines[self.model_name]

        start_time = time.time()

        # Deallocate buffers
        engine.deallocate_buffers()

        # Deactivate engine
        engine.deactivate()

        # Unload engine
        engine.unload()

        # Free workspace memory
        cudart.cudaFree(self.pipeline.shared_device_memory)
        self.pipeline.shared_device_memory = None

        logger.debug("[MEMORY] Low VRAM: Freed workspace memory")

        cleanup_time = time.time() - start_time
        self.timing[f"{self.model_name}_cleanup"] = cleanup_time

        logger.debug(f"[MEMORY] Low VRAM: {self.model_name} deallocated in {cleanup_time:.3f}s")

    def get_timing_summary(self):
        """Get timing summary for memory operations"""
        return self.timing.copy()
