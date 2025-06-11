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
Test cache persistence and safety of local dev files.

This test verifies:
1. Cache persistence - running twice doesn't re-copy files
2. Safety - delete_cached_files never deletes original source files
"""

from pathlib import Path

import pytest
from utils.path_manager import PathManager


@pytest.mark.cache
@pytest.mark.integration
class TestCacheSafety:
    """Test cache persistence and safety of local dev files."""

    def test_cache_persistence_and_safety(self, temp_cache_dir: Path, temp_source_dir: Path):
        """Test cache persistence and safety of local dev files."""
        # Create original dev files
        original_onnx = temp_source_dir / "my_model_fp16.onnx"
        original_data = temp_source_dir / "my_model_fp16.onnx.data"
        original_config = temp_source_dir / "config.json"

        original_onnx.write_text("original onnx content")
        original_data.write_text("original data content")
        original_config.write_text("original config content")

        # Create PathManager with separate cache directory
        path_manager = PathManager(str(temp_cache_dir))

        # Test 1: First run - should copy files
        success1 = path_manager.acquire_onnx_file("my_model", "fp16", str(original_onnx))

        assert success1, "First acquisition should succeed"

        cached_onnx = path_manager.get_onnx_path("my_model", "fp16")
        assert cached_onnx.exists(), "Cached ONNX should exist after first acquisition"

        # Test 2: Second run - should skip copying (files already exist)
        success2 = path_manager.acquire_onnx_file("my_model", "fp16", str(original_onnx))

        assert success2, "Second run should be successful (skip copy)"

        # Test 3: Cache deletion safety
        path_manager.delete_cached_files("my_model", "fp16", "dynamic")

        # CRITICAL TEST: Verify originals are STILL safe
        originals_still_safe = all(f.exists() for f in [original_onnx, original_data, original_config])
        assert originals_still_safe, "SAFETY FAILURE: Original dev files were deleted!"

        # Verify cache is cleaned up
        remaining_files = list(path_manager.shared_onnx_dir.rglob("*"))
        remaining_model_files = [f for f in remaining_files if f.is_file() and "my_model" in str(f)]
        assert len(remaining_model_files) == 0, "Cache should be cleaned up"

        # Test 4: Verify file paths are different
        canonical_path = path_manager.get_onnx_path("my_model", "fp16")
        assert original_onnx.parent != canonical_path.parent, "Original and cache directories should be different"
