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
Test script to verify the new hierarchical folder structure.
"""

from pathlib import PurePosixPath

import pytest
from utils.path_manager import PathManager


@pytest.mark.paths
@pytest.mark.unit
class TestHierarchicalStructure:
    """Test hierarchical folder structure functionality."""

    @pytest.mark.parametrize("shape_mode", ["dynamic", "static"])
    def test_shared_paths_structure(self, path_manager: PathManager, shape_mode: str):
        """Test that shared paths follow the correct hierarchical structure."""
        shared_onnx = path_manager.get_onnx_path("t5_text_encoder", "fp16")
        shared_engine = path_manager.get_engine_path("t5_text_encoder", "fp16", shape_mode)
        shared_metadata = path_manager.get_metadata_path("t5_text_encoder", "fp16", shape_mode)

        # Verify structure by checking path components
        assert "shared/onnx/t5_text_encoder/fp16/t5_text_encoder.onnx" in str(PurePosixPath(shared_onnx))
        assert f"shared/engines/t5_text_encoder/fp16/t5_text_encoder_{shape_mode}.engine" in str(
            PurePosixPath(shared_engine)
        )
        assert f"shared/engines/t5_text_encoder/fp16/t5_text_encoder_{shape_mode}.metadata.json" in str(
            PurePosixPath(shared_metadata)
        )

    def test_different_precisions_separated(self, path_manager: PathManager):
        """Test that different precisions for same model are properly separated."""
        fp16_onnx = path_manager.get_onnx_path("flux_transformer", "fp16")
        fp8_onnx = path_manager.get_onnx_path("flux_transformer", "fp8")

        # Verify they're in different directories
        assert fp16_onnx.parent != fp8_onnx.parent, "Different precisions should be in different directories"
        assert "fp16" in str(fp16_onnx), "FP16 path should contain 'fp16'"
        assert "fp8" in str(fp8_onnx), "FP8 path should contain 'fp8'"

    @pytest.mark.parametrize("shape_mode", ["dynamic", "static"])
    def test_directory_creation(self, path_manager: PathManager, shape_mode: str):
        """Test that directories are created correctly."""
        shared_onnx = path_manager.get_onnx_path("t5_text_encoder", "fp16")
        shared_engine = path_manager.get_engine_path("t5_text_encoder", "fp16", shape_mode)

        assert shared_onnx.parent.exists(), "Shared ONNX directory should be created"
        assert shared_engine.parent.exists(), "Shared engine directory should be created"

    @pytest.mark.parametrize("shape_mode", ["dynamic", "static"])
    def test_file_naming_consistency(self, path_manager: PathManager, shape_mode: str):
        """Test that file naming is consistent and clean."""
        shared_onnx = path_manager.get_onnx_path("t5_text_encoder", "fp16")
        shared_engine = path_manager.get_engine_path("t5_text_encoder", "fp16", shape_mode)

        # All files should have clean names without precision suffixes
        assert shared_onnx.name == "t5_text_encoder.onnx", f"ONNX should have clean name, got: {shared_onnx.name}"
        assert shared_engine.name == f"t5_text_encoder_{shape_mode}.engine", (
            f"Engine should have clean name, got: {shared_engine.name}"
        )
