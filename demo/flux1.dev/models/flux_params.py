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


from dataclasses import dataclass

from utils.base_params import BaseModelParams


@dataclass(frozen=True)
class FluxParams(BaseModelParams):
    """Parameters for Flux models"""

    # Batch dimension
    MIN_BATCH_SIZE: int = 1
    MAX_BATCH_SIZE: int = 4

    # Image dimensions
    MIN_HEIGHT: int = 256
    MAX_HEIGHT: int = 1024
    MIN_WIDTH: int = 256
    MAX_WIDTH: int = 1024

    # Text sequence length
    CLIP_SEQUENCE_LENGTH: int = 77
    T5_SEQUENCE_LENGTH: int = 512

    # Inference steps
    MIN_NUM_INFERENCE_STEPS: int = 1
    MAX_NUM_INFERENCE_STEPS: int = 50

    # VAE spatial compression ratio
    VAE_SPATIAL_COMPRESSION_RATIO: int = 8
