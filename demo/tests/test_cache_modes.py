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
Test cache modes (lean vs full) using pytest.
"""

import json
from pathlib import Path

import pytest
from utils.path_manager import ModelConfig, PathManager


@pytest.mark.cache
@pytest.mark.unit
class TestCacheModes:
    """Test lean vs full cache modes."""

    def test_full_cache_mode(self, temp_cache_dir: Path):
        """Test full cache mode keeps all models."""
        path_manager = PathManager(cache_dir=str(temp_cache_dir), cache_mode="full")
        shape_mode = "dynamic"
        # Simulate pipeline 1 models
        pipeline1_models = {
            "text_encoder": ModelConfig("flux_t5_text_encoder", "fp16", shape_mode),
            "transformer": ModelConfig("flux_transformer", "fp8", shape_mode),
            "vae": ModelConfig("flux_vae_decoder", "fp16", shape_mode),
        }

        path_manager.set_pipeline_models("flux1.dev", pipeline1_models)

        # Create dummy files
        for _, (model_id, precision, shape_mode) in pipeline1_models.items():
            onnx_path = path_manager.get_onnx_path(model_id, precision)
            engine_path = path_manager.get_engine_path(model_id, precision, shape_mode)
            metadata_path = path_manager.get_metadata_path(model_id, precision, shape_mode)

            onnx_path.touch()
            engine_path.touch()
            metadata_path.touch()

        # Switch to different models for the same pipeline
        pipeline2_models = {
            "text_encoder": ModelConfig("flux_t5_text_encoder", "fp16", shape_mode),  # Same
            "transformer": ModelConfig("flux_transformer", "fp16", shape_mode),  # Different
            "vae": ModelConfig("sdxl_vae", "fp16", shape_mode),  # Different
        }

        # Create dummy files for new models
        for _, (model_id, precision, shape_mode) in pipeline2_models.items():
            if model_id != "flux_t5_text_encoder":  # Don't recreate shared model
                onnx_path = path_manager.get_onnx_path(model_id, precision)
                engine_path = path_manager.get_engine_path(model_id, precision, shape_mode)
                metadata_path = path_manager.get_metadata_path(model_id, precision, shape_mode)

                onnx_path.touch()
                engine_path.touch()
                metadata_path.touch()

        # Update pipeline
        path_manager.set_pipeline_models("flux_1_dev", pipeline2_models)

        # In full mode, old models should still exist
        old_transformer_path = path_manager.get_onnx_path("flux_transformer", "fp8")
        old_vae_path = path_manager.get_onnx_path("flux_vae_decoder", "fp16")

        assert old_transformer_path.exists(), "Full mode should keep old models"
        assert old_vae_path.exists(), "Full mode should keep old models"

    def test_lean_cache_mode(self, temp_cache_dir: Path):
        """Test lean cache mode cleans up unused models."""
        path_manager = PathManager(cache_dir=str(temp_cache_dir), cache_mode="lean")

        # Same test as full mode but with lean mode
        pipeline1_models = {
            "text_encoder": ModelConfig("flux_t5_text_encoder", "fp16", "dynamic"),
            "transformer": ModelConfig("flux_transformer", "fp8", "dynamic"),
            "vae": ModelConfig("flux_vae_decoder", "fp16", "dynamic"),
        }

        path_manager.set_pipeline_models("flux_1_dev", pipeline1_models)

        # Create dummy files
        for _, (model_id, precision, shape_mode) in pipeline1_models.items():
            onnx_path = path_manager.get_onnx_path(model_id, precision)
            engine_path = path_manager.get_engine_path(model_id, precision, shape_mode)
            metadata_path = path_manager.get_metadata_path(model_id, precision, shape_mode)

            onnx_path.touch()
            engine_path.touch()
            metadata_path.touch()

        # Switch to different models
        pipeline2_models = {
            "text_encoder": ModelConfig("flux_t5_text_encoder", "fp16", "dynamic"),  # Shared
            "transformer": ModelConfig("flux_transformer", "fp16", "dynamic"),  # Different precision
            "vae": ModelConfig("sdxl_vae", "fp16", "dynamic"),  # Different model
        }

        # Create dummy files for new models
        for _, (model_id, precision, shape_mode) in pipeline2_models.items():
            if model_id != "flux_t5_text_encoder":
                onnx_path = path_manager.get_onnx_path(model_id, precision)
                engine_path = path_manager.get_engine_path(model_id, precision, shape_mode)
                metadata_path = path_manager.get_metadata_path(model_id, precision, shape_mode)

                onnx_path.touch()
                engine_path.touch()
                metadata_path.touch()

        # Update pipeline - should trigger cleanup
        path_manager.set_pipeline_models("flux_1_dev", pipeline2_models)

        # In lean mode, old models should be deleted
        old_transformer_path = path_manager.get_onnx_path("flux_transformer", "fp8")
        old_vae_path = path_manager.get_onnx_path("flux_vae_decoder", "fp16")
        shared_text_encoder_path = path_manager.get_onnx_path("flux_t5_text_encoder", "fp16")

        assert not old_transformer_path.exists(), "Lean mode should delete unused models"
        assert not old_vae_path.exists(), "Lean mode should delete unused models"
        assert shared_text_encoder_path.exists(), "Lean mode should keep shared models"

    def test_cache_mode_validation(self, temp_cache_dir: Path):
        """Test cache mode validation."""
        with pytest.raises(ValueError, match="cache_mode must be 'lean' or 'full'"):
            PathManager(cache_dir=str(temp_cache_dir), cache_mode="invalid")

    def test_lean_mode_only_current_pipeline(self, temp_cache_dir: Path):
        """Test that lean mode only keeps the current pipeline, not multiple pipelines."""
        path_manager = PathManager(cache_dir=str(temp_cache_dir), cache_mode="lean")

        # Create pipeline A
        pipeline_a_models = {
            "text_encoder": ("t5_text_encoder", "fp16"),
            "transformer": ("model_a_transformer", "fp8"),
            "vae": ("model_a_vae", "fp16"),
        }

        path_manager.set_pipeline_models("pipeline_a", pipeline_a_models)
        for _, (model_id, precision) in pipeline_a_models.items():
            onnx_path = path_manager.get_onnx_path(model_id, precision)
            onnx_path.touch()

        # Verify pipeline A is active and its models exist
        assert "pipeline_a" in path_manager.pipeline_states
        pipeline_a_transformer = path_manager.get_onnx_path("model_a_transformer", "fp8")
        assert pipeline_a_transformer.exists(), "Pipeline A models should exist"

        # Switch to pipeline B - should replace pipeline A entirely
        pipeline_b_models = {
            "text_encoder": ("t5_text_encoder", "fp16"),  # Same model as A
            "transformer": ("model_b_transformer", "fp8"),  # Different model
            "vae": ("model_b_vae", "fp16"),  # Different model
        }

        path_manager.set_pipeline_models("pipeline_b", pipeline_b_models)
        for _, (model_id, precision) in pipeline_b_models.items():
            if model_id != "t5_text_encoder":  # Don't recreate shared model
                onnx_path = path_manager.get_onnx_path(model_id, precision)
                onnx_path.touch()

        # Verify only pipeline B is active
        assert "pipeline_a" not in path_manager.pipeline_states
        assert "pipeline_b" in path_manager.pipeline_states

        # Verify pipeline A unique models are deleted, shared models preserved
        pipeline_a_transformer = path_manager.get_onnx_path("model_a_transformer", "fp8")
        pipeline_a_vae = path_manager.get_onnx_path("model_a_vae", "fp16")
        pipeline_b_transformer = path_manager.get_onnx_path("model_b_transformer", "fp8")
        pipeline_b_vae = path_manager.get_onnx_path("model_b_vae", "fp16")
        shared_text_encoder = path_manager.get_onnx_path("t5_text_encoder", "fp16")

        assert not pipeline_a_transformer.exists(), "Pipeline A unique models should be deleted"
        assert not pipeline_a_vae.exists(), "Pipeline A unique models should be deleted"
        assert pipeline_b_transformer.exists(), "Pipeline B models should exist"
        assert pipeline_b_vae.exists(), "Pipeline B models should exist"
        assert shared_text_encoder.exists(), "Shared model should be preserved"

    def test_tuple_to_model_config_conversion(self, temp_cache_dir: Path):
        """Test conversion of tuple format to ModelConfig objects."""
        path_manager = PathManager(cache_dir=str(temp_cache_dir), cache_mode="lean")

        # Test tuple format input
        tuple_models = {
            "text_encoder": ("flux_t5_text_encoder", "fp16"),  # 2-element tuple
            "transformer": ("flux_transformer", "fp8", "dynamic"),  # 3-element tuple
            "vae": ModelConfig("flux_vae_decoder", "fp16", "static"),  # Already ModelConfig
        }

        # This should not raise an error and should convert tuples properly
        path_manager.set_pipeline_models("test_pipeline", tuple_models)

        # Verify the pipeline state contains proper ModelConfig objects
        pipeline_state = path_manager.pipeline_states["test_pipeline"]

        assert isinstance(pipeline_state["text_encoder"], ModelConfig)
        assert pipeline_state["text_encoder"].model_id == "flux_t5_text_encoder"
        assert pipeline_state["text_encoder"].precision == "fp16"
        assert pipeline_state["text_encoder"].shape_mode == "static"  # Default

        assert isinstance(pipeline_state["transformer"], ModelConfig)
        assert pipeline_state["transformer"].model_id == "flux_transformer"
        assert pipeline_state["transformer"].precision == "fp8"
        assert pipeline_state["transformer"].shape_mode == "dynamic"

        assert isinstance(pipeline_state["vae"], ModelConfig)
        assert pipeline_state["vae"].model_id == "flux_vae_decoder"
        assert pipeline_state["vae"].precision == "fp16"
        assert pipeline_state["vae"].shape_mode == "static"

    def test_json_serialization_and_loading(self, temp_cache_dir: Path):
        """Test that pipeline states are properly saved and loaded from JSON."""
        cache_dir = str(temp_cache_dir)

        # Create path manager and set some models
        path_manager1 = PathManager(cache_dir=cache_dir, cache_mode="lean")
        models = {
            "text_encoder": ("model1", "fp16"),
            "transformer": ("model2", "fp8", "dynamic"),
        }
        path_manager1.set_pipeline_models("test_pipeline", models)

        # Create a new path manager instance to test loading
        path_manager2 = PathManager(cache_dir=cache_dir, cache_mode="lean")

        # Verify the state was loaded correctly
        assert "test_pipeline" in path_manager2.pipeline_states
        pipeline_state = path_manager2.pipeline_states["test_pipeline"]

        assert isinstance(pipeline_state["text_encoder"], ModelConfig)
        assert pipeline_state["text_encoder"].model_id == "model1"
        assert pipeline_state["text_encoder"].precision == "fp16"
        assert pipeline_state["text_encoder"].shape_mode == "static"

        assert isinstance(pipeline_state["transformer"], ModelConfig)
        assert pipeline_state["transformer"].model_id == "model2"
        assert pipeline_state["transformer"].precision == "fp8"
        assert pipeline_state["transformer"].shape_mode == "dynamic"

    def test_lean_mode_single_pipeline_only(self, temp_cache_dir: Path):
        """Test that lean mode only keeps one pipeline active and cleans up everything else."""
        path_manager = PathManager(cache_dir=str(temp_cache_dir), cache_mode="lean")

        # Set up pipeline A
        pipeline_a_models = {
            "text_encoder": ("model_a_text", "fp16"),
            "transformer": ("model_a_transformer", "fp8"),
        }
        path_manager.set_pipeline_models("pipeline_a", pipeline_a_models)

        # Create dummy files for pipeline A
        for _, config in pipeline_a_models.items():
            model_config = ModelConfig(config[0], config[1], config[2] if len(config) > 2 else "static")
            onnx_path = path_manager.get_onnx_path(model_config.model_id, model_config.precision)
            engine_path = path_manager.get_engine_path(
                model_config.model_id, model_config.precision, model_config.shape_mode
            )
            onnx_path.touch()
            engine_path.touch()

        # Switch to pipeline B - should clean up ALL pipeline A models
        pipeline_b_models = {
            "text_encoder": ("model_b_text", "fp16"),
            "transformer": ("model_b_transformer", "fp8"),
        }
        path_manager.set_pipeline_models("pipeline_b", pipeline_b_models)

        # Create dummy files for pipeline B
        for _, config in pipeline_b_models.items():
            model_config = ModelConfig(config[0], config[1], config[2] if len(config) > 2 else "static")
            onnx_path = path_manager.get_onnx_path(model_config.model_id, model_config.precision)
            engine_path = path_manager.get_engine_path(
                model_config.model_id, model_config.precision, model_config.shape_mode
            )
            onnx_path.touch()
            engine_path.touch()

        # Verify pipeline A models are deleted (lean mode only keeps current pipeline)
        for _, config in pipeline_a_models.items():
            model_config = ModelConfig(config[0], config[1], config[2] if len(config) > 2 else "static")
            onnx_path = path_manager.get_onnx_path(model_config.model_id, model_config.precision)
            engine_path = path_manager.get_engine_path(
                model_config.model_id, model_config.precision, model_config.shape_mode
            )
            assert not onnx_path.exists(), f"Pipeline A model {model_config.model_id} should be cleaned up"
            assert not engine_path.exists(), f"Pipeline A engine {model_config.model_id} should be cleaned up"

        # Verify pipeline B models still exist
        for _, config in pipeline_b_models.items():
            model_config = ModelConfig(config[0], config[1], config[2] if len(config) > 2 else "static")
            onnx_path = path_manager.get_onnx_path(model_config.model_id, model_config.precision)
            engine_path = path_manager.get_engine_path(
                model_config.model_id, model_config.precision, model_config.shape_mode
            )
            assert onnx_path.exists(), f"Pipeline B model {model_config.model_id} should exist"
            assert engine_path.exists(), f"Pipeline B engine {model_config.model_id} should exist"

        # Verify only pipeline B is in the state
        assert "pipeline_a" not in path_manager.pipeline_states
        assert "pipeline_b" in path_manager.pipeline_states

    def test_invalid_config_format_handling(self, temp_cache_dir: Path):
        """Test handling of invalid config formats."""
        path_manager = PathManager(cache_dir=str(temp_cache_dir), cache_mode="lean")

        # Test with invalid config format
        invalid_models = {
            "valid": ("model1", "fp16"),
            "invalid_short_tuple": ("model2",),  # Too short
            "invalid_type": "invalid_string",  # Wrong type
            "valid_modelconfig": ModelConfig("model3", "fp8", "dynamic"),
        }

        # This should not crash and should only process valid configs
        path_manager.set_pipeline_models("test_pipeline", invalid_models)

        # Verify only valid configs were processed
        pipeline_state = path_manager.pipeline_states["test_pipeline"]

        assert "valid" in pipeline_state
        assert "valid_modelconfig" in pipeline_state
        assert "invalid_short_tuple" not in pipeline_state
        assert "invalid_type" not in pipeline_state

    def test_full_mode_state_accumulation(self, temp_cache_dir: Path):
        """Test that full mode accumulates all pipeline states in JSON for complete tracking."""
        path_manager = PathManager(cache_dir=str(temp_cache_dir), cache_mode="full")

        # Set pipeline A
        pipeline_a = {
            "transformer": ("model_a_transformer", "fp8"),
            "text_encoder": ("shared_text_encoder", "fp16"),
        }
        path_manager.set_pipeline_models("pipeline_a", pipeline_a)

        # Set pipeline B
        pipeline_b = {
            "transformer": ("model_b_transformer", "fp8"),
            "text_encoder": ("shared_text_encoder", "fp16"),  # Same as A
            "vae": ("model_b_vae", "fp16"),
        }
        path_manager.set_pipeline_models("pipeline_b", pipeline_b)

        # Set pipeline C
        pipeline_c = {
            "transformer": ("model_c_transformer", "fp8"),
            "scheduler": ("model_c_scheduler", "fp32"),
        }
        path_manager.set_pipeline_models("pipeline_c", pipeline_c)

        # Check in-memory state accumulates all pipelines
        assert len(path_manager.pipeline_states) == 3, "Should have all 3 pipelines in memory"
        assert all(p in path_manager.pipeline_states for p in ["pipeline_a", "pipeline_b", "pipeline_c"])

        # Check JSON file contains all pipelines
        state_file = temp_cache_dir / ".cache_state.json"
        with open(state_file) as f:
            saved_state = json.load(f)

        assert len(saved_state) == 3, "Should have all 3 pipelines in JSON"
        assert all(p in saved_state for p in ["pipeline_a", "pipeline_b", "pipeline_c"])

        # Count total unique models
        all_models = path_manager._get_all_models_from_states(path_manager.pipeline_states)
        expected_models = {
            ("model_a_transformer", "fp8", "static"),
            ("model_b_transformer", "fp8", "static"),
            ("model_c_transformer", "fp8", "static"),
            ("shared_text_encoder", "fp16", "static"),
            ("model_b_vae", "fp16", "static"),
            ("model_c_scheduler", "fp32", "static"),
        }
        assert all_models == expected_models, f"Expected {expected_models}, got {all_models}"

    def test_lean_mode_uses_accumulated_state_for_cleanup(self, temp_cache_dir: Path):
        """Test that lean mode uses the complete accumulated state for thorough cleanup."""
        cache_dir = str(temp_cache_dir)

        # First, run in full mode to accumulate multiple pipelines
        pm_full = PathManager(cache_dir=cache_dir, cache_mode="full")

        # Set multiple pipelines in full mode
        pipeline_a = {
            "transformer": ("model_a_transformer", "fp8"),
            "text_encoder": ("shared_text_encoder", "fp16"),
        }
        pm_full.set_pipeline_models("pipeline_a", pipeline_a)

        pipeline_b = {
            "transformer": ("model_b_transformer", "fp8"),
            "text_encoder": ("shared_text_encoder", "fp16"),  # Shared
            "vae": ("model_b_vae", "fp16"),
        }
        pm_full.set_pipeline_models("pipeline_b", pipeline_b)

        pipeline_c = {
            "transformer": ("model_c_transformer", "fp8"),
            "scheduler": ("model_c_scheduler", "fp32"),
        }
        pm_full.set_pipeline_models("pipeline_c", pipeline_c)

        # Create dummy files for all models
        all_models = pm_full._get_all_models_from_states(pm_full.pipeline_states)
        for model_id, precision, shape_mode in all_models:
            onnx_path = pm_full.get_onnx_path(model_id, precision)
            engine_path = pm_full.get_engine_path(model_id, precision, shape_mode)
            onnx_path.touch()
            engine_path.touch()

        # Verify all files exist before switching to lean mode
        for model_id, precision, shape_mode in all_models:
            onnx_path = pm_full.get_onnx_path(model_id, precision)
            engine_path = pm_full.get_engine_path(model_id, precision, shape_mode)
            assert onnx_path.exists(), f"ONNX should exist before lean switch: {model_id}"
            assert engine_path.exists(), f"Engine should exist before lean switch: {model_id}"

        # Switch to lean mode - should load accumulated state for cleanup purposes
        pm_lean = PathManager(cache_dir=cache_dir, cache_mode="lean")

        # In lean mode, initial state is loaded but no specific pipeline is active yet
        assert len(pm_lean.pipeline_states) == 3, "Should load all accumulated pipelines initially"

        # Set a new pipeline that only reuses some models
        new_pipeline = {
            "transformer": ("model_a_transformer", "fp8"),  # Keep this one
            "text_encoder": ("shared_text_encoder", "fp16"),  # Keep this one
            "new_model": ("completely_new_model", "bf16"),  # New model
        }

        pm_lean.set_pipeline_models("new_pipeline", new_pipeline)

        # Check that state now only has the new pipeline
        assert len(pm_lean.pipeline_states) == 1, "Lean mode should only keep current pipeline"
        assert "new_pipeline" in pm_lean.pipeline_states
        assert "pipeline_a" not in pm_lean.pipeline_states
        assert "pipeline_b" not in pm_lean.pipeline_states
        assert "pipeline_c" not in pm_lean.pipeline_states

        # Check which models should be cleaned up vs kept
        remaining_models = pm_lean._get_all_models_from_states(pm_lean.pipeline_states)
        expected_remaining = {
            ("model_a_transformer", "fp8", "static"),  # Reused
            ("shared_text_encoder", "fp16", "static"),  # Reused
            ("completely_new_model", "bf16", "static"),  # New
        }
        assert remaining_models == expected_remaining

        # Check files were actually deleted from disk
        deleted_models = {
            ("model_b_transformer", "fp8", "static"),
            ("model_c_transformer", "fp8", "static"),
            ("model_b_vae", "fp16", "static"),
            ("model_c_scheduler", "fp32", "static"),
        }

        for model_id, precision, shape_mode in deleted_models:
            onnx_path = pm_lean.get_onnx_path(model_id, precision)
            engine_path = pm_lean.get_engine_path(model_id, precision, shape_mode)
            assert not onnx_path.exists(), f"ONNX file should be deleted: {onnx_path}"
            assert not engine_path.exists(), f"Engine file should be deleted: {engine_path}"

        # Check kept files still exist on disk
        for model_id, precision, shape_mode in expected_remaining:
            if model_id != "completely_new_model":  # Don't check new model we didn't create
                onnx_path = pm_lean.get_onnx_path(model_id, precision)
                engine_path = pm_lean.get_engine_path(model_id, precision, shape_mode)
                assert onnx_path.exists(), f"ONNX file should be kept: {onnx_path}"
                assert engine_path.exists(), f"Engine file should be kept: {engine_path}"

    def test_lean_mode_preserves_onnx_across_shape_modes(self, temp_cache_dir: Path):
        """Test that lean mode preserves ONNX files when switching between static/dynamic versions."""
        path_manager = PathManager(cache_dir=str(temp_cache_dir), cache_mode="lean")

        # Pipeline with static version of a model
        static_pipeline = {
            "transformer": ("flux_transformer", "fp8", "static"),
        }
        path_manager.set_pipeline_models("static_pipeline", static_pipeline)

        # Create dummy files for static version
        model_config = ModelConfig("flux_transformer", "fp8", "static")
        onnx_path = path_manager.get_onnx_path(model_config.model_id, model_config.precision)
        static_engine_path = path_manager.get_engine_path(model_config.model_id, model_config.precision, "static")
        static_metadata_path = path_manager.get_metadata_path(model_config.model_id, model_config.precision, "static")

        onnx_path.touch()
        static_engine_path.touch()
        static_metadata_path.touch()

        # Switch to dynamic version of the same model
        dynamic_pipeline = {
            "transformer": ("flux_transformer", "fp8", "dynamic"),
        }
        path_manager.set_pipeline_models("dynamic_pipeline", dynamic_pipeline)

        # Create dummy files for dynamic version
        dynamic_engine_path = path_manager.get_engine_path("flux_transformer", "fp8", "dynamic")
        dynamic_metadata_path = path_manager.get_metadata_path("flux_transformer", "fp8", "dynamic")

        dynamic_engine_path.touch()
        dynamic_metadata_path.touch()

        # CRITICAL TEST: ONNX file should be preserved (shared between static/dynamic)
        assert onnx_path.exists(), "ONNX file should be preserved when switching shape modes"

        # Static engine files should be deleted
        assert not static_engine_path.exists(), "Static engine should be deleted"
        assert not static_metadata_path.exists(), "Static metadata should be deleted"

        # Dynamic engine files should exist
        assert dynamic_engine_path.exists(), "Dynamic engine should exist"
        assert dynamic_metadata_path.exists(), "Dynamic metadata should exist"

        # Now switch to a completely different model - ONNX should be deleted
        different_pipeline = {
            "transformer": ("different_model", "fp16", "static"),
        }
        path_manager.set_pipeline_models("different_pipeline", different_pipeline)

        # Create dummy files for different model
        different_onnx = path_manager.get_onnx_path("different_model", "fp16")
        different_engine = path_manager.get_engine_path("different_model", "fp16", "static")
        different_onnx.touch()
        different_engine.touch()

        # Now the original ONNX should be deleted (no active shape modes)
        assert not onnx_path.exists(), "ONNX file should be deleted when no shape modes are active"
        assert not dynamic_engine_path.exists(), "Dynamic engine should be deleted"
        assert not dynamic_metadata_path.exists(), "Dynamic metadata should be deleted"
