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
Pytest configuration and common fixtures.
"""

import sys
import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest

# Add the parent directory to the Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.path_manager import PathManager


@pytest.fixture
def temp_cache_dir() -> Generator[Path, None, None]:
    """Create a temporary cache directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_dir = Path(temp_dir) / "test_cache"
        yield cache_dir
        # Cleanup is automatic with tempfile.TemporaryDirectory


@pytest.fixture
def path_manager(temp_cache_dir: Path) -> PathManager:
    """Create a PathManager instance with temporary cache directory."""
    return PathManager(cache_dir=str(temp_cache_dir))


@pytest.fixture
def temp_source_dir() -> Generator[Path, None, None]:
    """Create a temporary source directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        source_dir = Path(temp_dir) / "source_models"
        source_dir.mkdir()
        yield source_dir
        # Cleanup is automatic with tempfile.TemporaryDirectory


def create_dummy_onnx_file(file_path: Path, content: str = "# Dummy ONNX file for testing\n") -> None:
    """Create a dummy ONNX file for testing."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w") as f:
        f.write(content)

    # Also create a related file (like .onnx.data)
    data_file = file_path.with_suffix(".onnx.data")
    with open(data_file, "w") as f:
        f.write("# Dummy ONNX data file\n")


# Make the helper function available to all tests
pytest.create_dummy_onnx_file = create_dummy_onnx_file
