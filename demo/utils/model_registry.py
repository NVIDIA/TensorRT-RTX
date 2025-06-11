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
Model Registry with shared model definitions.

Contains ONLY model definitions, configurations, and metadata.
Path management is handled by PathManager.
"""

import logging
from typing import Any, Dict, List, Optional

import tensorrt_rtx as trt
import torch

# Initialize logger for this module
logger = logging.getLogger("rtx_demo.utils.model_registry")

# Shared model definitions - each model defined once with all its variants
MODELS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "flux_clip_text_encoder": {
        "bf16": {
            "onnx_repository": "black-forest-labs/FLUX.1-dev-onnx",
            "onnx_subfolder": "clip.opt",
            "input_shapes": {"input_ids": ("B", 77)},
            "input_dtypes": {"input_ids": trt.DataType.INT32},
            "output_shapes": {
                "text_embeddings": ("B", 77, 768),
                "pooled_embeddings": ("B", 768),
            },
        },
    },
    "flux_t5_text_encoder": {
        "bf16": {
            "onnx_repository": "black-forest-labs/FLUX.1-dev-onnx",
            "onnx_subfolder": "t5.opt",
            "input_shapes": {"input_ids": ("B", 512)},
            "input_dtypes": {"input_ids": trt.DataType.INT32},
            "output_shapes": {
                "text_embeddings": ("B", 512, 4096),
            },
        },
        "fp8": {
            "onnx_repository": "black-forest-labs/FLUX.1-dev-onnx",
            "onnx_subfolder": "t5-fp8.opt",
            "input_shapes": {"input_ids": ("B", 512)},
            "input_dtypes": {"input_ids": trt.DataType.INT32},
            "output_shapes": {
                "text_embeddings": ("B", 512, 4096),
            },
        },
    },
    "flux_transformer": {
        "bf16": {
            "onnx_repository": "black-forest-labs/FLUX.1-dev-onnx",
            "onnx_subfolder": "transformer.opt/bf16",
            "input_shapes": {
                "hidden_states": ("B", "latent_dim", 64),
                "encoder_hidden_states": ("B", 512, 4096),
                "pooled_projections": ("B", 768),
                "timestep": ("B",),
                "img_ids": ("latent_dim", 3),
                "txt_ids": (512, 3),
                "guidance": ("B",),
            },
            "input_dtypes": {
                "hidden_states": trt.DataType.BF16,
                "encoder_hidden_states": trt.DataType.BF16,
                "pooled_projections": trt.DataType.BF16,
                "timestep": trt.DataType.BF16,
                "img_ids": trt.DataType.FLOAT,
                "txt_ids": trt.DataType.FLOAT,
                "guidance": trt.DataType.FLOAT,
            },
            "output_shapes": {
                "latent": ("B", "latent_dim", 64),
            },
        },
        "fp8": {
            "onnx_repository": "black-forest-labs/FLUX.1-dev-onnx",
            "onnx_subfolder": "transformer.opt/fp8",
            "input_shapes": {
                "hidden_states": ("B", "latent_dim", 64),
                "encoder_hidden_states": ("B", 512, 4096),
                "pooled_projections": ("B", 768),
                "timestep": ("B",),
                "img_ids": ("latent_dim", 3),
                "txt_ids": (512, 3),
                "guidance": ("B",),
            },
            "input_dtypes": {
                "hidden_states": trt.DataType.BF16,
                "encoder_hidden_states": trt.DataType.BF16,
                "pooled_projections": trt.DataType.BF16,
                "timestep": trt.DataType.BF16,
                "img_ids": trt.DataType.FLOAT,
                "txt_ids": trt.DataType.FLOAT,
                "guidance": trt.DataType.FLOAT,
            },
            "output_shapes": {
                "latent": ("B", "latent_dim", 64),
            },
        },
        "fp4": {
            "onnx_repository": "black-forest-labs/FLUX.1-dev-onnx",
            "onnx_subfolder": "transformer.opt/fp4",
            "input_shapes": {
                "hidden_states": ("B", "latent_dim", 64),
                "encoder_hidden_states": ("B", 512, 4096),
                "pooled_projections": ("B", 768),
                "timestep": ("B",),
                "img_ids": ("latent_dim", 3),
                "txt_ids": (512, 3),
                "guidance": ("B",),
            },
            "input_dtypes": {
                "hidden_states": trt.DataType.BF16,
                "encoder_hidden_states": trt.DataType.BF16,
                "pooled_projections": trt.DataType.BF16,
                "timestep": trt.DataType.BF16,
                "img_ids": trt.DataType.FLOAT,
                "txt_ids": trt.DataType.FLOAT,
                "guidance": trt.DataType.FLOAT,
            },
            "output_shapes": {
                "latent": ("B", "latent_dim", 64),
            },
        },
    },
    "flux_vae_decoder": {
        "bf16": {
            "onnx_repository": "black-forest-labs/FLUX.1-dev-onnx",
            "onnx_subfolder": "vae.opt",
            "input_shapes": {"latent": ("B", 16, "latent_height", "latent_width")},
            "input_dtypes": {"latent": trt.DataType.BF16},
            "output_shapes": {
                "images": ("B", 3, "height", "width"),
            },
        },
    },
}

# Pipeline compositions - map pipeline roles to actual models
PIPELINES: Dict[str, Dict[str, str]] = {
    "flux_1_dev": {
        "clip_text_encoder": "flux_clip_text_encoder",
        "t5_text_encoder": "flux_t5_text_encoder",
        "transformer": "flux_transformer",
        "vae_decoder": "flux_vae_decoder",
    },
}

# Default precisions per pipeline
DEFAULT_PRECISIONS: Dict[str, Dict[str, str]] = {
    # Flux default precisions
    "flux_1_dev": {
        "clip_text_encoder": "bf16",
        "t5_text_encoder": "bf16",
        "transformer": "fp8",
        "vae_decoder": "bf16",
    },
}


# Short form precision mapping
SHORT_FORM_PRECISIONS: Dict[str, torch.dtype] = {
    "fp16": torch.float16,
    "fp8": torch.float8_e4m3fn,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
    "int32": torch.int32,
    "int64": torch.int64,
    "bool": torch.bool,
}


class ModelRegistry:
    """Registry with model definitions and configurations"""

    def __init__(self):
        self.models = MODELS
        self.pipelines = PIPELINES
        self.defaults = DEFAULT_PRECISIONS
        self.short_form_precisions = SHORT_FORM_PRECISIONS
        self._validate()

    def _validate(self):
        """Validate registry consistency"""
        # Validate input shapes/dtypes keys match
        for model_id, precisions in self.models.items():
            for precision, config in precisions.items():
                shape_keys = set(config.get("input_shapes", {}).keys())
                dtype_keys = set(config.get("input_dtypes", {}).keys())
                if shape_keys != dtype_keys:
                    raise ValueError(f"{model_id}[{precision}]: shape/dtype key mismatch")

        # Validate pipeline references and precision consistency
        for pipeline, roles in self.pipelines.items():
            if pipeline not in self.defaults:
                raise ValueError(f"Pipeline {pipeline} missing in DEFAULT_PRECISIONS")
            if set(roles.keys()) != set(self.defaults[pipeline].keys()):
                raise ValueError(f"Pipeline {pipeline}: role mismatch with defaults")
            for role, model_id in roles.items():
                if model_id not in self.models:
                    raise ValueError(f"Model {model_id} not found in MODELS")
                precision = self.defaults[pipeline][role]
                if precision not in self.short_form_precisions:
                    raise ValueError(f"Invalid precision {precision}")
                if precision not in self.models[model_id]:
                    raise ValueError(f"Precision {precision} not available for {model_id}")

    def get_torch_dtype(self, precision: str) -> torch.dtype:
        """Get the torch.dtype for a precision"""
        return self.short_form_precisions.get(precision)

    def get_model_id(self, pipeline_name: str, role: str) -> Optional[str]:
        """Get the actual model ID for a pipeline role"""
        return self.pipelines.get(pipeline_name, {}).get(role)

    def is_model_id(self, model_id: str) -> bool:
        """Check if a model ID is valid"""
        return model_id in self.models

    def get_model_config(self, pipeline_name: str, role: str, precision: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a model in a pipeline role"""
        model_id = self.get_model_id(pipeline_name, role)
        if not model_id:
            return None

        return self.models.get(model_id, {}).get(precision)

    def get_available_precisions(self, pipeline_name: str, role: str) -> List:
        """Get available precisions for a model role"""
        model_id = self.get_model_id(pipeline_name, role)
        if not model_id:
            return []

        return list(self.models.get(model_id, {}).keys())

    def get_pipeline_roles(self, pipeline_name: str) -> List:
        """Get all roles in a pipeline"""
        return list(self.pipelines.get(pipeline_name, {}).keys())

    def get_pipeline_config(self, pipeline_name: str) -> Dict[str, str]:
        """Get all configs in a pipeline"""
        return self.pipelines.get(pipeline_name, {})

    def get_default_precision(self, pipeline_name: str, role: str) -> str:
        """Get default precision for a role"""
        return self.defaults.get(pipeline_name, {}).get(role, "fp16")

    def get_default_precisions(self, pipeline_name: str) -> Dict[str, str]:
        """Get default precisions for a pipeline"""
        return self.defaults.get(pipeline_name, {})

    def get_onnx_path(self, pipeline_name: str, role: str, precision: str) -> Optional[str]:
        """
        Get ONNX source (local path) for a model role, if present.

        Note: When copying from local paths, ALL files in the source directory are copied.
        """
        config = self.get_model_config(pipeline_name, role, precision)

        if not config:
            return None

        return config.get("onnx_path")

    def get_onnx_repository(self, pipeline_name: str, role: str, precision: str) -> Optional[str]:
        """Get ONNX repository for a model role."""
        config = self.get_model_config(pipeline_name, role, precision)
        if not config:
            return None
        return config.get("onnx_repository")

    def get_onnx_subfolder(self, pipeline_name: str, role: str, precision: str) -> Optional[str]:
        """Get ONNX subfolder for a model role."""
        config = self.get_model_config(pipeline_name, role, precision)
        if not config:
            return None
        return config.get("onnx_subfolder")

    def validate_precision_config(self, pipeline_name: str, precision_config: Dict[str, str]) -> Dict[str, str]:
        """Validate and fill in missing precisions with defaults"""
        validated_config = {}
        pipeline_roles = self.get_pipeline_roles(pipeline_name)

        for role in pipeline_roles:
            if role in precision_config:
                available_precisions = self.get_available_precisions(pipeline_name, role)
                if precision_config[role] in available_precisions:
                    validated_config[role] = precision_config[role]
                else:
                    logger.warning(f"Precision '{precision_config[role]}' not available for {role}, using default")
                    validated_config[role] = self.get_default_precision(pipeline_name, role)
            else:
                validated_config[role] = self.get_default_precision(pipeline_name, role)

        return validated_config

    def print_available_models(self, pipeline_name: str):
        """Print available models and precisions for a pipeline"""
        logger.info(f"\nAvailable models for {pipeline_name}:")
        roles = self.get_pipeline_roles(pipeline_name)

        for role in roles:
            model_id = self.get_model_id(pipeline_name, role)
            precisions = self.get_available_precisions(pipeline_name, role)
            default = self.get_default_precision(pipeline_name, role)
            logger.info(f"  {role} ({model_id}): {precisions} (default: {default})")

    def print_sharing_info(self):
        """Print model sharing information"""
        logger.info("\nModel Sharing Analysis:")

        # Find which models are shared across pipelines
        model_usage = {}
        for pipeline_name, roles in self.pipelines.items():
            for role, model_id in roles.items():
                if model_id not in model_usage:
                    model_usage[model_id] = []
                model_usage[model_id].append(f"{pipeline_name}:{role}")

        shared_models = {k: v for k, v in model_usage.items() if len(v) > 1}
        unique_models = {k: v for k, v in model_usage.items() if len(v) == 1}

        if shared_models:
            logger.info("Shared models (cached once, linked across pipelines):")
            for model_id, usage in shared_models.items():
                logger.info(f"  {model_id}: {', '.join(usage)}")

        if unique_models:
            logger.info("Pipeline-specific models:")
            for model_id, usage in unique_models.items():
                logger.info(f"  {model_id}: {usage[0]}")

    def get_io_names(self, model_id: str, precision: str) -> tuple[list, list]:
        """Get input and output names for a model and precision"""
        if model_id not in self.models or precision not in self.models[model_id]:
            return [], []

        config = self.models[model_id][precision]
        input_names = list(config.get("input_shapes", {}).keys())
        output_names = list(config.get("output_shapes", {}).keys())
        return input_names, output_names


# Global registry instance
registry = ModelRegistry()
