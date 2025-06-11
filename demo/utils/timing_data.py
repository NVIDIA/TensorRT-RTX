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
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

# Initialize logger for this module
logger = logging.getLogger("rtx_demo.utils.timing_data")


@dataclass
class InferenceTimingData:
    """Container for detailed inference timing data using CUDA events."""

    # Pipeline timings (in milliseconds) - maps model type to list of execution times
    pipeline_times: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))

    # Total times
    total_inference_time: float = 0.0

    # Metadata
    num_inference_steps: int = 0
    height: int = 0
    width: int = 0
    batch_size: int = 0
    guidance_scale: float = 0.0

    def to_dict(self) -> Dict:
        """Return timing data as a parseable dictionary."""
        # Calculate total end-to-end runtime
        total_e2e = self.total_inference_time
        if total_e2e == 0 and self.pipeline_times:
            # Calculate total from pipeline_times if total_inference_time is not set
            total_e2e = sum(np.sum(times) for times in self.pipeline_times.values() if times)

        # Extract transformer runtimes
        transformer_times = []
        if "transformer" in self.pipeline_times:
            transformer_raw = self.pipeline_times["transformer"]
            for time_entry in transformer_raw:
                transformer_times.append(float(time_entry))

        # Calculate throughput
        throughput = 0.0
        if total_e2e > 0:
            throughput = 1000.0 / total_e2e  # images per second

        return {
            "total_e2e_runtime_ms": total_e2e,
            "transformer_runtimes_ms": transformer_times,
            "throughput_images_per_sec": throughput,
            "metadata": {
                "height": self.height,
                "width": self.width,
                "batch_size": self.batch_size,
                "num_inference_steps": self.num_inference_steps,
                "guidance_scale": self.guidance_scale,
            },
        }

    def print_summary(self):
        """Print a summary of timing data."""
        logger.info("\n" + "=" * 60)
        logger.info("INFERENCE TIMING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Configuration: {self.height}x{self.width}")
        logger.info(f"Batch size: {self.batch_size}, Guidance scale: {self.guidance_scale}")
        logger.info(f"Inference steps: {self.num_inference_steps}")
        logger.info("-" * 60)

        # Pipeline timing data
        if self.pipeline_times:
            for model_type, times in self.pipeline_times.items():
                if times:  # Only show models with timing data
                    times_array = np.array(times)
                    mean_time = float(np.mean(times_array))
                    std_time = float(np.std(times_array))
                    total_time = float(np.sum(times_array))
                    count = len(times)

                    logger.info(f"{model_type.replace('_', ' ').title()}:")
                    if count > 1:
                        logger.info(f"  - Total:   {total_time:8.2f} ms ({count} executions)")
                        logger.info(f"  - Average: {mean_time:8.2f} ± {std_time:.2f} ms")
                        logger.info(
                            f"  - Range:   {float(np.min(times_array)):.2f} - {float(np.max(times_array)):.2f} ms"
                        )
                        logger.info(f"  - Entries: {times_array}")
                    else:
                        logger.info(f"  - Time:    {total_time:8.2f} ms")

        logger.info("-" * 60)
        if self.total_inference_time > 0:
            logger.info(f"Total Inference:  {self.total_inference_time:8.2f} ms")
        elif self.pipeline_times:
            # Calculate total from pipeline_times
            total_pipeline_time = sum(np.sum(times) for times in self.pipeline_times.values() if times)
            if total_pipeline_time > 0:
                logger.info(f"Total Pipeline:   {total_pipeline_time:8.2f} ms")

        # Calculate throughput
        effective_total = self.total_inference_time
        if effective_total == 0 and self.pipeline_times:
            effective_total = sum(np.sum(times) for times in self.pipeline_times.values() if times)

        if effective_total > 0:
            img_per_sec = 1000.0 / effective_total
            logger.info(f"Throughput:       {img_per_sec:8.2f} images/second")

        logger.info("=" * 60)
