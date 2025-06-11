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
Test cache consistency fix using pytest.

This test simulates the problem where canonical files exist but pipeline links don't,
and shows how the new PathManager methods fix this inconsistent state.
"""

from pathlib import Path

import pytest
from utils.path_manager import PathManager


@pytest.mark.cache
@pytest.mark.unit
class TestCacheConsistency:
    """Test cache consistency functionality."""

    def test_cache_consistency_fix(self, path_manager: PathManager, temp_source_dir: Path):
        """Test cache consistency issue detection and fix."""
        # Create a source ONNX file
        source_onnx = temp_source_dir / "transformer.onnx"
        pytest.create_dummy_onnx_file(source_onnx)

        # Test parameters
        model_id = "test_transformer"
        precision = "fp16"

        # Step 1: Setup cache (before files exist)
        canonical_onnx = path_manager.get_onnx_path(model_id, precision)
        assert not canonical_onnx.exists()

        # Step 2: Acquire ONNX file
        success = path_manager.acquire_onnx_file(model_id, precision, str(source_onnx))

        assert success, "ONNX acquisition should succeed"
        assert canonical_onnx.exists(), "Canonical ONNX should exist after acquisition"

        # Step 3: Check cache state (fix the key name)
        cache_status = path_manager.check_cached_files(model_id, precision, "dynamic")
        assert cache_status["onnx"], "ONNX should exist in cache"

    def test_deletion_methods(self, path_manager: PathManager, temp_source_dir: Path):
        """Test the deletion methods work correctly."""
        # Create test models
        model_configs = [("test_transformer", "fp16"), ("test_vae", "fp16")]

        for model_id, precision in model_configs:
            source_onnx = temp_source_dir / f"{model_id}.onnx"
            pytest.create_dummy_onnx_file(source_onnx)

            success = path_manager.acquire_onnx_file(model_id, precision, str(source_onnx))
            assert success, f"Should acquire {model_id}_{precision}"

            canonical_onnx = path_manager.get_onnx_path(model_id, precision)
            assert canonical_onnx.exists(), f"Canonical {model_id}_{precision} should exist"

        # Test deleting only ONNX files for one model
        path_manager.delete_cached_onnx_files("test_vae", "fp16")

        vae_onnx = path_manager.get_onnx_path("test_vae", "fp16")
        transformer_onnx = path_manager.get_onnx_path("test_transformer", "fp16")

        assert not vae_onnx.exists(), "VAE ONNX should be deleted"
        assert transformer_onnx.exists(), "Transformer ONNX should remain"

        # Test deleting all files for remaining model
        path_manager.delete_cached_files("test_transformer", "fp16", "dynamic")
        assert not transformer_onnx.exists(), "Transformer ONNX should be deleted"
