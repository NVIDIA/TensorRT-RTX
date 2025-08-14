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
Test ONNX acquisition functionality using pytest.

This test demonstrates how the PathManager can acquire ONNX files from:
1. Local file paths (for development)
2. Remote URLs (for production) with proper temporary directory handling
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from utils.path_manager import PathManager


@pytest.mark.integration
@pytest.mark.paths
class TestONNXAcquisition:
    """Test ONNX file acquisition functionality."""

    def test_local_file_acquisition_copies_all_files(self, path_manager: PathManager, temp_source_dir: Path):
        """Test that local file acquisition copies ALL files from the directory."""
        # Create main ONNX file
        test_onnx = temp_source_dir / "test_model_fp16.onnx"
        test_onnx.write_text("fake onnx content")

        # Create multiple additional files to test that ALL files are copied
        additional_files = [
            "test_model_fp16.onnx.data",  # Standard external data
            "another_model.onnx",  # Different model
            "config.json",  # Configuration file
            "weights.bin",  # Weights file
            "metadata.txt",  # Metadata file
            "readme.md",  # Documentation
        ]

        for additional_file in additional_files:
            file_path = temp_source_dir / additional_file
            file_path.write_text(f"fake content for {additional_file}")

        # Test local file acquisition
        success = path_manager.acquire_onnx_file("test_model", "fp16", str(test_onnx), "", "", None)

        assert success, "Local file acquisition should succeed"

        canonical_onnx = path_manager.get_onnx_path("test_model", "fp16")
        assert canonical_onnx.exists(), "Main ONNX file should exist"

        # Check ALL files from the directory were copied
        shared_files = list(canonical_onnx.parent.iterdir())
        shared_file_names = [f.name for f in shared_files]

        # Should have the main ONNX file with canonical name plus all additional files
        expected_files = ["test_model.onnx"] + additional_files  # Canonical name, not original
        for expected_file in expected_files:
            assert expected_file in shared_file_names, f"Expected file {expected_file} should be copied"

        assert len(shared_files) == len(expected_files), (
            f"Expected {len(expected_files)} files, got {len(shared_files)}"
        )

    @patch("utils.path_manager.snapshot_download")
    def test_remote_download_with_nested_structure(self, mock_snapshot_download, path_manager: PathManager):
        """Test remote download handles nested directory structure correctly."""

        def mock_download_side_effect(repo_id, allow_patterns, local_dir, token=None):
            """Mock snapshot_download to create nested structure like HuggingFace does."""
            temp_dir = Path(local_dir)

            # Create the nested structure that HuggingFace would create
            nested_dir = temp_dir / "models" / "onnx" / "test_model" / "fp16"
            nested_dir.mkdir(parents=True, exist_ok=True)

            # Create mock ONNX files in the nested directory
            (nested_dir / "test_model.onnx").write_text("fake onnx content from remote")
            (nested_dir / "test_model.onnx.data").write_text("fake onnx data from remote")
            (nested_dir / "config.json").write_text('{"model": "test"}')

        mock_snapshot_download.side_effect = mock_download_side_effect

        # Test remote acquisition with non-existent local path (triggers remote download)
        fake_remote_path = "nonexistent/path/to/model.onnx"

        success = path_manager.acquire_onnx_file(
            "test_model",
            "fp16",
            fake_remote_path,
            "test_pipeline",
            "models/onnx/test_model/fp16",
            None,
        )

        assert success, "Remote download should succeed"

        # Verify snapshot_download was called with correct parameters
        mock_snapshot_download.assert_called_once_with(
            repo_id="test_pipeline",
            allow_patterns=os.path.join("models/onnx/test_model/fp16", "*"),
            local_dir=mock_snapshot_download.call_args[1]["local_dir"],  # temp directory
            token=None,
        )

        # Check that files were moved to the correct final location
        canonical_onnx = path_manager.get_onnx_path("test_model", "fp16")
        assert canonical_onnx.exists(), "Main ONNX file should exist after download"

        # Check all files were moved correctly
        target_dir = canonical_onnx.parent
        expected_files = ["test_model.onnx", "test_model.onnx.data", "config.json"]

        for expected_file in expected_files:
            file_path = target_dir / expected_file
            assert file_path.exists(), f"File {expected_file} should exist after download"

        # Verify content was preserved
        assert canonical_onnx.read_text() == "fake onnx content from remote"

    @patch("utils.path_manager.snapshot_download")
    def test_remote_download_with_token(self, mock_snapshot_download, path_manager: PathManager):
        """Test remote download passes HuggingFace token correctly."""

        def mock_download_side_effect(repo_id, allow_patterns, local_dir, token=None):
            temp_dir = Path(local_dir)
            nested_dir = temp_dir / "models" / "onnx" / "private_model" / "fp16"
            nested_dir.mkdir(parents=True, exist_ok=True)
            (nested_dir / "private_model.onnx").write_text("private model content")

        mock_snapshot_download.side_effect = mock_download_side_effect

        # Test with HF token
        test_token = "hf_test_token_12345"
        success = path_manager.acquire_onnx_file(
            "private_model",
            "fp16",
            "nonexistent/path",
            "private_pipeline",
            "models/onnx/private_model/fp16",
            test_token,
        )

        assert success, "Remote download with token should succeed"

        # Verify token was passed correctly
        mock_snapshot_download.assert_called_once()
        call_args = mock_snapshot_download.call_args
        assert call_args[1]["token"] == test_token, "HF token should be passed to snapshot_download"

    @patch("utils.path_manager.snapshot_download")
    def test_remote_download_error_handling(self, mock_snapshot_download, path_manager: PathManager):
        """Test remote download handles errors gracefully."""

        # Mock snapshot_download to raise an exception
        mock_snapshot_download.side_effect = Exception("Network error")

        success = path_manager.acquire_onnx_file(
            "error_model",
            "fp16",
            "nonexistent/path",
            "error_pipeline",
            "models/onnx/error_model/fp16",
            None,
        )

        assert not success, "Remote download should fail gracefully on error"

        # Verify the target ONNX file doesn't exist after error
        canonical_onnx = path_manager.get_onnx_path("error_model", "fp16")
        assert not canonical_onnx.exists(), "ONNX file should not exist after failed download"

    @patch("utils.path_manager.snapshot_download")
    def test_remote_download_temporary_directory_cleanup(self, mock_snapshot_download, path_manager: PathManager):
        """Test that temporary directories are properly cleaned up."""
        created_temp_dirs = []
        original_tempdir = tempfile.TemporaryDirectory

        def track_temp_dirs(*args, **kwargs):
            temp_dir_obj = original_tempdir(*args, **kwargs)
            created_temp_dirs.append(Path(temp_dir_obj.name))
            return temp_dir_obj

        with patch("tempfile.TemporaryDirectory", side_effect=track_temp_dirs):
            # Mock download that succeeds
            def mock_download_side_effect(repo_id, allow_patterns, local_dir, token=None):
                temp_dir = Path(local_dir)
                nested_dir = temp_dir / "models" / "onnx" / "cleanup_model" / "fp16"
                nested_dir.mkdir(parents=True, exist_ok=True)
                (nested_dir / "cleanup_model.onnx").write_text("test content")

            mock_snapshot_download.side_effect = mock_download_side_effect

            # Test remote download with mocked response
            success = path_manager.acquire_onnx_file(
                "cleanup_model",
                "fp16",
                "nonexistent/path",
                "cleanup_pipeline",
                "models/onnx/cleanup_model/fp16",
                None,
            )

            assert success, "Download should succeed"

            # After successful completion, temp directories should be cleaned up
            for temp_dir in created_temp_dirs:
                assert not temp_dir.exists(), f"Temporary directory {temp_dir} should be cleaned up"

    def test_local_path_with_nonexistent_source_fails_gracefully(self, path_manager: PathManager):
        """Test that local path acquisition fails gracefully when source doesn't exist."""
        nonexistent_path = "/nonexistent/path/to/model.onnx"

        success = path_manager.acquire_onnx_file("nonexistent_model", "fp16", nonexistent_path, "", "", None)

        assert not success, "Acquisition should fail when source doesn't exist"

        # Verify no files were created
        canonical_onnx = path_manager.get_onnx_path("nonexistent_model", "fp16")
        assert not canonical_onnx.exists(), "ONNX file should not exist after failed acquisition"

    def test_local_path_with_subdirectories_copies_all_files(self, path_manager: PathManager, temp_source_dir: Path):
        """Test that local path acquisition copies files even with complex directory structure."""
        # Create a complex source directory structure
        model_dir = temp_source_dir / "complex_model"
        model_dir.mkdir()

        # Create subdirectory with additional files
        sub_dir = model_dir / "weights"
        sub_dir.mkdir()

        # Create main ONNX file
        main_onnx = model_dir / "complex_model_fp16.onnx"
        main_onnx.write_text("main onnx content")

        # Create files in main directory
        (model_dir / "complex_model_fp16.onnx.data").write_text("onnx data")
        (model_dir / "config.json").write_text('{"model": "complex"}')
        (model_dir / "tokenizer.json").write_text('{"vocab_size": 1000}')

        # Create files in subdirectory (should NOT be copied since we only copy from parent dir)
        (sub_dir / "weights.bin").write_text("weights content")
        (sub_dir / "optimizer.bin").write_text("optimizer content")

        # Test acquisition
        success = path_manager.acquire_onnx_file("complex_model", "fp16", str(main_onnx), "", "", None)

        assert success, "Complex directory acquisition should succeed"

        canonical_onnx = path_manager.get_onnx_path("complex_model", "fp16")
        assert canonical_onnx.exists(), "Main ONNX file should exist"

        # Check that files from the main directory were copied
        target_dir = canonical_onnx.parent
        copied_files = [f.name for f in target_dir.iterdir()]

        # Should have main ONNX file (renamed) and other files from same directory
        expected_files = [
            "complex_model.onnx",  # Renamed from complex_model_fp16.onnx
            "complex_model_fp16.onnx.data",
            "config.json",
            "tokenizer.json",
        ]

        for expected_file in expected_files:
            assert expected_file in copied_files, f"Expected file {expected_file} should be copied"

        # Should NOT have subdirectory files (only copies from parent directory)
        assert "weights.bin" not in copied_files, "Subdirectory files should not be copied"
        assert "optimizer.bin" not in copied_files, "Subdirectory files should not be copied"

        # Verify content preservation
        assert canonical_onnx.read_text() == "main onnx content"
