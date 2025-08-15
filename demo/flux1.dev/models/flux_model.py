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
from typing import Any, Optional

from diffusers import AutoencoderKL, FluxTransformer2DModel
from diffusers.configuration_utils import FrozenDict
from models.flux_params import FluxParams
from transformers import AutoConfig
from utils.base_model import BaseModel

# Initialize logger for this module
logger = logging.getLogger("rtx_demo.flux1.dev.models.flux_model")


class FluxTransformerModel(BaseModel):
    """Flux Transformer model for text-to-image generation"""

    def __init__(
        self,
        name: str,
        device: str = "cuda",
        model_params: Optional[FluxParams] = None,
        model_id: str = "black-forest-labs/FLUX.1-dev",
        hf_token: Optional[str] = None,
    ):
        super().__init__(name, device, model_params, hf_token)
        self.model_id = model_id

        # Load configuration from HuggingFace
        logger.debug(f"Loading Flux Transformer config from {model_id}/transformer")
        self.config = FrozenDict(
            FluxTransformer2DModel.load_config(
                model_id,
                subfolder="transformer",
                token=self.hf_token,
            )
        )

    def get_input_profile(
        self, use_static_shape: bool, batch_size: int = 1, height: int = 512, width: int = 512
    ) -> dict[str, Any]:
        """Return TensorRT input profile for dynamic shapes"""
        latent_height = height // self.model_params.VAE_SPATIAL_COMPRESSION_RATIO
        latent_width = width // self.model_params.VAE_SPATIAL_COMPRESSION_RATIO
        opt_latent_dim = (latent_height // 2) * (latent_width // 2)

        # Build static input profile
        if use_static_shape:
            input_profile = {
                "hidden_states": (batch_size, opt_latent_dim, self.config.in_channels),
                "encoder_hidden_states": (
                    batch_size,
                    self.model_params.T5_SEQUENCE_LENGTH,
                    self.config.joint_attention_dim,
                ),
                "pooled_projections": (batch_size, self.config.pooled_projection_dim),
                "timestep": (batch_size,),
                "img_ids": (opt_latent_dim, 3),
                "txt_ids": (self.model_params.T5_SEQUENCE_LENGTH, 3),
                "guidance": (batch_size,),
            }
        else:
            min_latent_height = self.model_params.MIN_HEIGHT // self.model_params.VAE_SPATIAL_COMPRESSION_RATIO
            min_latent_width = self.model_params.MIN_WIDTH // self.model_params.VAE_SPATIAL_COMPRESSION_RATIO
            max_latent_height = self.model_params.MAX_HEIGHT // self.model_params.VAE_SPATIAL_COMPRESSION_RATIO
            max_latent_width = self.model_params.MAX_WIDTH // self.model_params.VAE_SPATIAL_COMPRESSION_RATIO
            min_latent_dim = (min_latent_height // 2) * (min_latent_width // 2)
            max_latent_dim = (max_latent_height // 2) * (max_latent_width // 2)

            input_profile = {
                "hidden_states": [
                    (
                        self.model_params.MIN_BATCH_SIZE,
                        min_latent_dim,
                        self.config.in_channels,
                    ),  # min
                    (batch_size, opt_latent_dim, self.config.in_channels),  # opt
                    (
                        self.model_params.MAX_BATCH_SIZE,
                        max_latent_dim,
                        self.config.in_channels,
                    ),  # max
                ],
                "encoder_hidden_states": [
                    (
                        self.model_params.MIN_BATCH_SIZE,
                        self.model_params.T5_SEQUENCE_LENGTH,
                        self.config.joint_attention_dim,
                    ),
                    (
                        batch_size,
                        self.model_params.T5_SEQUENCE_LENGTH,
                        self.config.joint_attention_dim,
                    ),
                    (
                        self.model_params.MAX_BATCH_SIZE,
                        self.model_params.T5_SEQUENCE_LENGTH,
                        self.config.joint_attention_dim,
                    ),
                ],
                "pooled_projections": [
                    (self.model_params.MIN_BATCH_SIZE, self.config.pooled_projection_dim),
                    (batch_size, self.config.pooled_projection_dim),
                    (self.model_params.MAX_BATCH_SIZE, self.config.pooled_projection_dim),
                ],
                "timestep": [
                    (self.model_params.MIN_BATCH_SIZE,),
                    (batch_size,),
                    (self.model_params.MAX_BATCH_SIZE,),
                ],
                "img_ids": [
                    (min_latent_dim, 3),
                    (opt_latent_dim, 3),
                    (max_latent_dim, 3),
                ],
                "txt_ids": [
                    (self.model_params.T5_SEQUENCE_LENGTH, 3),
                    (self.model_params.T5_SEQUENCE_LENGTH, 3),
                    (self.model_params.T5_SEQUENCE_LENGTH, 3),
                ],
                "guidance": [
                    (self.model_params.MIN_BATCH_SIZE,),
                    (batch_size,),
                    (self.model_params.MAX_BATCH_SIZE,),
                ],
            }

        return input_profile

    def get_shape_dict(self, batch_size: int = 1, height: int = 512, width: int = 512) -> dict[str, Any]:
        """Return shape dictionary for tensor allocation"""
        latent_height = height // self.model_params.VAE_SPATIAL_COMPRESSION_RATIO
        latent_width = width // self.model_params.VAE_SPATIAL_COMPRESSION_RATIO
        latent_dim = (latent_height // 2) * (latent_width // 2)

        return {
            "hidden_states": (batch_size, latent_dim, self.config.in_channels),
            "encoder_hidden_states": (
                batch_size,
                self.model_params.T5_SEQUENCE_LENGTH,
                self.config.joint_attention_dim,
            ),
            "pooled_projections": (batch_size, self.config.pooled_projection_dim),
            "timestep": (batch_size,),
            "img_ids": (latent_dim, 3),
            "txt_ids": (self.model_params.T5_SEQUENCE_LENGTH, 3),
            "guidance": (batch_size,),
            "latent": (
                batch_size,
                latent_dim,
                self.config.in_channels,
            ),  # Use in_channels for output too
        }


class FluxTextEncoderModel(BaseModel):
    """Flux CLIP Text Encoder model"""

    def __init__(
        self,
        name: str,
        device: str = "cuda",
        model_params: Optional[FluxParams] = None,
        model_id: str = "black-forest-labs/FLUX.1-dev",
        hf_token: Optional[str] = None,
    ):
        super().__init__(name, device, model_params, hf_token)
        self.model_id = model_id

        # Load configuration from HuggingFace
        logger.debug(f"Loading CLIP Text Encoder config from {model_id}")
        self.config = AutoConfig.from_pretrained(model_id, subfolder="text_encoder", token=self.hf_token)

    def get_input_profile(self, use_static_shape: bool, batch_size: int = 1, **kwargs) -> dict[str, Any]:
        """Return TensorRT input profile for dynamic shapes"""
        if use_static_shape:
            return {
                "input_ids": (batch_size, self.model_params.CLIP_SEQUENCE_LENGTH),
            }
        return {
            "input_ids": [
                (self.model_params.MIN_BATCH_SIZE, self.model_params.CLIP_SEQUENCE_LENGTH),
                (batch_size, self.model_params.CLIP_SEQUENCE_LENGTH),
                (self.model_params.MAX_BATCH_SIZE, self.model_params.CLIP_SEQUENCE_LENGTH),
            ]
        }

    def get_shape_dict(self, batch_size: int = 1, **kwargs) -> dict[str, Any]:
        """Return shape dictionary for tensor allocation"""
        # Handle both direct config and nested text_config structures
        hidden_size = getattr(self.config, "hidden_size", None)
        if hidden_size is None and hasattr(self.config, "text_config"):
            hidden_size = self.config.text_config.hidden_size
        projection_dim = getattr(self.config, "projection_dim", hidden_size)

        return {
            "input_ids": (batch_size, self.model_params.CLIP_SEQUENCE_LENGTH),
            "text_embeddings": (batch_size, self.model_params.CLIP_SEQUENCE_LENGTH, hidden_size),
            "pooled_embeddings": (batch_size, projection_dim),
        }


class FluxT5EncoderModel(BaseModel):
    """Flux T5 Text Encoder model"""

    def __init__(
        self,
        name: str,
        device: str = "cuda",
        model_params: Optional[FluxParams] = None,
        model_id: str = "black-forest-labs/FLUX.1-dev",
        hf_token: Optional[str] = None,
    ):
        super().__init__(name, device, model_params, hf_token)
        self.model_id = model_id

        # Load configuration from HuggingFace
        logger.debug(f"Loading T5 Text Encoder config from {model_id}")
        self.config = AutoConfig.from_pretrained(model_id, subfolder="text_encoder_2", token=self.hf_token)

    def get_input_profile(self, use_static_shape: bool, batch_size: int = 1) -> dict[str, Any]:
        """Return TensorRT input profile for dynamic shapes"""
        if use_static_shape:
            return {
                "input_ids": (batch_size, self.model_params.T5_SEQUENCE_LENGTH),
            }
        return {
            "input_ids": [
                (self.model_params.MIN_BATCH_SIZE, self.model_params.T5_SEQUENCE_LENGTH),
                (batch_size, self.model_params.T5_SEQUENCE_LENGTH),
                (self.model_params.MAX_BATCH_SIZE, self.model_params.T5_SEQUENCE_LENGTH),
            ]
        }

    def get_shape_dict(self, batch_size: int = 1) -> dict[str, Any]:
        """Return shape dictionary for tensor allocation"""
        # T5 uses 'd_model' for hidden size
        hidden_size = getattr(self.config, "d_model", 4096)

        return {
            "input_ids": (batch_size, self.model_params.T5_SEQUENCE_LENGTH),
            "text_embeddings": (batch_size, self.model_params.T5_SEQUENCE_LENGTH, hidden_size),
        }


class FluxVAEModel(BaseModel):
    """Flux VAE Decoder model"""

    def __init__(
        self,
        name: str,
        device: str = "cuda",
        model_params: Optional[FluxParams] = None,
        model_id: str = "black-forest-labs/FLUX.1-dev",
        hf_token: Optional[str] = None,
    ):
        super().__init__(name, device, model_params, hf_token)
        self.model_id = model_id

        # Load configuration from HuggingFace
        logger.debug(f"Loading VAE config from {model_id}/vae")
        self.config = FrozenDict(AutoencoderKL.load_config(model_id, subfolder="vae", token=self.hf_token))

    def get_input_profile(
        self, use_static_shape: bool, batch_size: int = 1, height: int = 512, width: int = 512
    ) -> dict[str, Any]:
        """Return TensorRT input profile for dynamic shapes"""
        latent_height = height // self.model_params.VAE_SPATIAL_COMPRESSION_RATIO
        latent_width = width // self.model_params.VAE_SPATIAL_COMPRESSION_RATIO

        min_latent_height = self.model_params.MIN_HEIGHT // self.model_params.VAE_SPATIAL_COMPRESSION_RATIO
        min_latent_width = self.model_params.MIN_WIDTH // self.model_params.VAE_SPATIAL_COMPRESSION_RATIO
        max_latent_height = self.model_params.MAX_HEIGHT // self.model_params.VAE_SPATIAL_COMPRESSION_RATIO
        max_latent_width = self.model_params.MAX_WIDTH // self.model_params.VAE_SPATIAL_COMPRESSION_RATIO

        if use_static_shape:
            return {
                "latent": (batch_size, self.config.latent_channels, latent_height, latent_width),
            }

        return {
            "latent": [
                (
                    self.model_params.MIN_BATCH_SIZE,
                    self.config.latent_channels,
                    min_latent_height,
                    min_latent_width,
                ),
                (batch_size, self.config.latent_channels, latent_height, latent_width),
                (
                    self.model_params.MAX_BATCH_SIZE,
                    self.config.latent_channels,
                    max_latent_height,
                    max_latent_width,
                ),
            ]
        }

    def get_shape_dict(self, batch_size: int = 1, height: int = 512, width: int = 512) -> dict[str, Any]:
        """Return shape dictionary for tensor allocation"""
        latent_height = height // self.model_params.VAE_SPATIAL_COMPRESSION_RATIO
        latent_width = width // self.model_params.VAE_SPATIAL_COMPRESSION_RATIO

        return {
            "latent": (batch_size, self.config.latent_channels, latent_height, latent_width),
            "images": (batch_size, self.config.out_channels, height, width),
        }
