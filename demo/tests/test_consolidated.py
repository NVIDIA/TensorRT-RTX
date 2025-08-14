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
Test the consolidated structure with proper separation of concerns.

This test verifies:
- PathManager handles all path operations
- ModelRegistry contains only model definitions
- Pipeline uses both correctly
"""

from pathlib import Path, PurePosixPath

import pytest
from utils.model_registry import registry
from utils.path_manager import PathManager


@pytest.mark.integration
class TestConsolidated:
    """Test separation of concerns between PathManager and ModelRegistry."""

    def test_separation_of_concerns(self, temp_cache_dir: Path):
        """Test that PathManager and ModelRegistry work together properly."""
        # Test PathManager
        path_manager = PathManager(str(temp_cache_dir))

        # Test model registry
        available_pipelines = list(registry.pipelines.keys())
        assert len(available_pipelines) > 0, "Should have available pipelines"
        assert "flux_1_dev" in available_pipelines, "Flux.1 [dev] pipeline should exist"

        # Test path operations - use a valid pipeline name
        model_id = registry.get_model_id("flux_1_dev", "transformer")
        assert model_id is not None, "Should get model ID for transformer"

        # Test path generation
        canonical_path = path_manager.get_engine_path(model_id, "fp16", "static")
        assert canonical_path is not None, "Should generate canonical path"
        assert canonical_path.parent.exists(), "Parent directories should be created"

        # Test that path contains expected structure
        assert "shared/engines" in str(PurePosixPath(canonical_path)), "Should use shared engines directory"
        assert model_id in str(canonical_path), "Should contain model ID"
        assert "fp16" in str(canonical_path), "Should contain precision"

    def test_model_registry_functionality(self):
        """Test core model registry functionality."""
        # Test pipeline existence
        pipelines = list(registry.pipelines.keys())
        assert "flux_1_dev" in pipelines, "Flux.1 [dev] pipeline should exist"

        # Test model ID retrieval
        model_id = registry.get_model_id("flux_1_dev", "transformer")
        assert isinstance(model_id, str), "Model ID should be a string"

        # Test precision options
        precisions = registry.get_available_precisions("flux_1_dev", "transformer")
        assert isinstance(precisions, list), "Precisions should be a list"
        assert len(precisions) > 0, "Should have available precisions"

        # Test default precision
        default_precision = registry.get_default_precision("flux_1_dev", "transformer")
        assert default_precision in precisions, "Default precision should be in available precisions"

    def test_pathmanager_functionality(self, temp_cache_dir: Path):
        """Test core PathManager functionality."""
        path_manager = PathManager(str(temp_cache_dir))

        # Test path generation for different file types
        model_id = "test_model"
        precision = "fp16"

        onnx_path = path_manager.get_onnx_path(model_id, precision)
        engine_path = path_manager.get_engine_path(model_id, precision, "dynamic")
        metadata_path = path_manager.get_metadata_path(model_id, precision, "dynamic")

        # Verify proper file extensions
        assert onnx_path.suffix == ".onnx", "ONNX path should have .onnx extension"
        assert engine_path.suffix == ".engine", "Engine path should have .engine extension"
        assert metadata_path.suffix == ".json", "Metadata path should have .json extension"

        # Verify hierarchical structure
        assert "shared/onnx" in str(PurePosixPath(onnx_path)), "ONNX should be in shared/onnx"
        assert "shared/engines" in str(PurePosixPath(engine_path)), "Engine should be in shared/engines"
        assert "shared/engines" in str(PurePosixPath(metadata_path)), "Metadata should be in shared/engines"
