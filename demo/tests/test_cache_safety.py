# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


@pytest.mark.unit
@pytest.mark.parametrize("variant", ["full", "weightless", "native"])
@pytest.mark.parametrize("shape_mode", ["static", "dynamic"])
def test_cache_helpers_cover_engine_variants(path_manager, variant, shape_mode):
    """Find and delete one shape's engine artifacts without touching neighboring files."""
    plan = path_manager.get_engine_path("transformer", "fp8", shape_mode, weightless=variant != "full")
    if variant == "native":
        plan = plan.with_name(plan.stem + ".safetensors" + plan.suffix)
    artifacts = [plan]
    if variant == "full":
        artifacts.append(path_manager.get_metadata_path("transformer", "fp8", shape_mode))
    else:
        artifacts.extend(plan.with_suffix(suffix) for suffix in (".checkpoint.json", ".weights.json", ".weights.bin"))
    other_shape = "dynamic" if shape_mode == "static" else "static"
    other_plan = path_manager.get_engine_path("transformer", "fp8", other_shape, weightless=True)
    other_plan = other_plan.with_name(other_plan.stem + ".safetensors" + other_plan.suffix)
    protected = [
        other_plan,
        other_plan.with_suffix(".weights.bin"),
        path_manager.get_engine_path("transformer", "bf16", shape_mode, weightless=True),
        path_manager.get_engine_path("clip", "fp8", shape_mode, weightless=True),
        plan.parent / "unrelated.bin",
        plan.parent / "unrelated.json",
    ]
    for path in artifacts + protected:
        path.write_bytes(b"test")

    status = path_manager.check_cached_files("transformer", "fp8", shape_mode)
    assert status["engine"] and status["metadata"]
    path_manager.delete_cached_engine_files("transformer", "fp8", shape_mode)
    assert not any(path.exists() for path in artifacts)
    assert all(path.exists() for path in protected)
    status = path_manager.check_cached_files("transformer", "fp8", shape_mode)
    assert not status["engine"] and not status["metadata"]


@pytest.mark.unit
@pytest.mark.parametrize("suffix", [".checkpoint.json", ".weights.json", ".weights.bin"])
def test_lean_cleanup_removes_orphaned_weightless_artifacts(path_manager, suffix):
    """Missing engine plans must not prevent cleanup of their remaining sidecars."""
    base = path_manager.get_engine_path("transformer", "fp8", "static", weightless=True)
    plan = base.with_name(base.stem + ".safetensors" + base.suffix)
    artifact = plan.with_suffix(suffix)
    artifact.touch()
    path_manager._cleanup_unused_model("transformer", "fp8", "static")
    assert not artifact.exists()


@pytest.mark.unit
def test_cache_deletion_rejects_unrelated_binary_files(path_manager, tmp_path):
    """Allowing weight backups must not allow arbitrary binary files to be deleted."""
    unrelated = tmp_path / "user.bin"
    unrelated.touch()
    assert not path_manager._safe_delete_file(unrelated)
    assert unrelated.exists()
