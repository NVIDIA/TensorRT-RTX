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
Test script to verify precision changes are handled correctly across runs.
"""

import json
from pathlib import Path

import pytest
from utils.path_manager import ModelConfig, PathManager


@pytest.mark.integration
class TestPrecisionChanges:
    """Test precision changes across multiple runs."""

    def test_precision_changes_lean_mode(self, temp_cache_dir: Path):
        """Test precision changes across multiple runs in lean mode."""
        # Test 1: Initial run with specific precisions
        path_manager = PathManager(cache_dir=str(temp_cache_dir), cache_mode="lean")

        initial_config = {
            "text_encoder": ModelConfig("flux_t5_text_encoder", "fp16", "dynamic"),
            "transformer": ModelConfig("flux_transformer", "fp16", "dynamic"),  # fp16 initially
            "vae": ModelConfig("flux_vae_decoder", "fp16", "dynamic"),
        }

        path_manager.set_pipeline_models("flux_1_dev", initial_config)

        # Create dummy files for initial models
        for _, (model_id, precision, shape_mode) in initial_config.items():
            onnx_path = path_manager.get_onnx_path(model_id, precision)
            engine_path = path_manager.get_engine_path(model_id, precision, shape_mode)
            metadata_path = path_manager.get_metadata_path(model_id, precision, shape_mode)

            onnx_path.touch()
            engine_path.touch()
            metadata_path.touch()

        # Verify state file was created
        state_file = temp_cache_dir / ".cache_state.json"
        assert state_file.exists(), "State file should be created"

        with open(state_file) as f:
            state = json.load(f)

        expected_state = {
            "flux_1_dev": {
                "text_encoder": {"model_id": "flux_t5_text_encoder", "precision": "fp16", "shape_mode": "dynamic"},
                "transformer": {"model_id": "flux_transformer", "precision": "fp16", "shape_mode": "dynamic"},
                "vae": {"model_id": "flux_vae_decoder", "precision": "fp16", "shape_mode": "dynamic"},
            }
        }
        assert state == expected_state, f"State mismatch: {state} != {expected_state}"

    def test_precision_change_cleanup(self, temp_cache_dir: Path):
        """Test that changing precision cleans up old files in lean mode."""
        path_manager = PathManager(cache_dir=str(temp_cache_dir), cache_mode="lean")

        # Initial configuration
        initial_config = {
            "text_encoder": ModelConfig("flux_t5_text_encoder", "fp16", "dynamic"),
            "transformer": ModelConfig("flux_transformer", "fp16", "dynamic"),
            "vae": ModelConfig("flux_vae_decoder", "fp16", "dynamic"),
        }

        path_manager.set_pipeline_models("flux_1_dev", initial_config)

        # Create initial files
        for _, (model_id, precision, _) in initial_config.items():
            onnx_path = path_manager.get_onnx_path(model_id, precision)
            onnx_path.touch()

        # Change transformer precision
        changed_config = {
            "text_encoder": ModelConfig("flux_t5_text_encoder", "fp16", "dynamic"),  # Same
            "transformer": ModelConfig("flux_transformer", "fp8", "dynamic"),  # Changed fp16 -> fp8
            "vae": ModelConfig("flux_vae_decoder", "fp16", "dynamic"),  # Same
        }

        # Check that old fp16 transformer exists before change
        old_transformer_onnx = path_manager.get_onnx_path("flux_transformer", "fp16")
        assert old_transformer_onnx.exists(), "Old fp16 transformer should exist before change"

        path_manager.set_pipeline_models("flux_1_dev", changed_config)

        # Create new fp8 transformer file
        new_transformer_onnx = path_manager.get_onnx_path("flux_transformer", "fp8")
        new_transformer_onnx.touch()

        # Verify cleanup happened
        assert not old_transformer_onnx.exists(), "Old fp16 transformer should be deleted"
        assert new_transformer_onnx.exists(), "New fp8 transformer should exist"

        # Verify shared models are preserved
        shared_text_encoder = path_manager.get_onnx_path("flux_t5_text_encoder", "fp16")
        shared_vae = path_manager.get_onnx_path("flux_vae_decoder", "fp16")
        assert shared_text_encoder.exists(), "Shared text encoder should be preserved"
        assert shared_vae.exists(), "Shared VAE should be preserved"

    def test_lean_mode_single_active_pipeline(self, temp_cache_dir: Path):
        """Test that lean mode only keeps the currently active pipeline, cleaning up models from previous pipelines."""
        path_manager = PathManager(cache_dir=str(temp_cache_dir), cache_mode="lean")

        # First pipeline configuration
        pipeline1_config = {
            "text_encoder": ModelConfig("flux_t5_text_encoder", "fp16", "dynamic"),
            "transformer": ModelConfig("flux_transformer", "fp16", "dynamic"),
            "vae": ModelConfig("flux_vae_decoder", "fp16", "dynamic"),
        }

        path_manager.set_pipeline_models("flux_1_dev", pipeline1_config)

        # Create files
        for _, (model_id, precision, _) in pipeline1_config.items():
            onnx_path = path_manager.get_onnx_path(model_id, precision)
            onnx_path.touch()

        # Switch to another pipeline that uses some shared models
        pipeline2_config = {
            "text_encoder": ModelConfig("flux_t5_text_encoder", "fp16", "dynamic"),  # Shared with pipeline1
            "transformer": ModelConfig("flux_transformer", "fp16", "dynamic"),  # Shared with pipeline1
            "vae": ModelConfig("sdxl_vae", "fp16", "dynamic"),  # Different VAE
        }

        path_manager.set_pipeline_models("another_pipeline", pipeline2_config)

        # Create new VAE file
        sdxl_vae_path = path_manager.get_onnx_path("sdxl_vae", "fp16")
        sdxl_vae_path.touch()

        # Verify pipeline1 models that are shared are preserved, but flux_vae_decoder is cleaned up
        text_encoder_path = path_manager.get_onnx_path("flux_t5_text_encoder", "fp16")
        transformer_path = path_manager.get_onnx_path("flux_transformer", "fp16")
        old_vae_path = path_manager.get_onnx_path("flux_vae_decoder", "fp16")
        new_vae_path = path_manager.get_onnx_path("sdxl_vae", "fp16")

        assert text_encoder_path.exists(), "Shared text encoder should be preserved"
        assert transformer_path.exists(), "Shared transformer should be preserved"
        assert not old_vae_path.exists(), "Old VAE not used by current pipeline should be cleaned up"
        assert new_vae_path.exists(), "New VAE should exist"

        # Verify only the current pipeline is tracked
        assert "flux_1_dev" not in path_manager.pipeline_states, "Previous pipeline should not be tracked"
        assert "another_pipeline" in path_manager.pipeline_states, "Current pipeline should be tracked"

        # Now change the current pipeline to use completely different models
        completely_different_config = {
            "text_encoder": ModelConfig("different_text_encoder", "fp8", "static"),
            "transformer": ModelConfig("different_transformer", "fp8", "static"),
            "vae": ModelConfig("different_vae", "fp8", "static"),
        }

        # Create new model files
        for _, (model_id, precision, _) in completely_different_config.items():
            onnx_path = path_manager.get_onnx_path(model_id, precision)
            onnx_path.touch()

        path_manager.set_pipeline_models("another_pipeline", completely_different_config)

        # All previous models should be cleaned up since none are shared with the new config
        assert not text_encoder_path.exists(), "Previous text encoder should be cleaned up"
        assert not transformer_path.exists(), "Previous transformer should be cleaned up"
        assert not new_vae_path.exists(), "Previous VAE should be cleaned up"

        # New models should exist
        new_text_encoder = path_manager.get_onnx_path("different_text_encoder", "fp8")
        new_transformer = path_manager.get_onnx_path("different_transformer", "fp8")
        new_vae = path_manager.get_onnx_path("different_vae", "fp8")

        assert new_text_encoder.exists(), "New text encoder should exist"
        assert new_transformer.exists(), "New transformer should exist"
        assert new_vae.exists(), "New VAE should exist"

    def test_full_mode_keeps_all_models(self, temp_cache_dir: Path):
        """Test that full mode never deletes models regardless of precision changes."""
        path_manager = PathManager(cache_dir=str(temp_cache_dir), cache_mode="full")

        # Initial configuration
        initial_config = {
            "transformer": ("test_transformer", "fp16"),
        }

        path_manager.set_pipeline_models("test_pipeline", initial_config)

        # Create initial file
        fp16_path = path_manager.get_onnx_path("test_transformer", "fp16")
        fp16_path.touch()

        # Change precision
        changed_config = {
            "transformer": ("test_transformer", "fp8"),
        }

        path_manager.set_pipeline_models("test_pipeline", changed_config)

        # Create new precision file
        fp8_path = path_manager.get_onnx_path("test_transformer", "fp8")
        fp8_path.touch()

        # In full mode, both should exist
        assert fp16_path.exists(), "Full mode should keep fp16 model"
        assert fp8_path.exists(), "Full mode should have fp8 model"
