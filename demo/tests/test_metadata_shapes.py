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
Test script to verify the fixed _shapes_fit_profile method.

This script tests the different input shape profile formats:
1. Static shapes (single tuple)
2. Dynamic shapes (list of 3 tuples)
3. Effectively static shapes (all 3 tuples are the same)
"""

import pytest
from utils.engine_metadata import EngineMetadata


@pytest.mark.unit
class TestMetadataShapes:
    """Test _shapes_fit_profile method with different profile formats."""

    def test_static_shapes(self):
        """Test static shapes (single tuple)."""
        metadata_static = EngineMetadata(
            model_name="test_model",
            precision="fp16",
            onnx_path="/path/to/model.onnx",
            onnx_hash="abcd1234",
            input_shapes={
                "input_ids": (1, 77),  # Static shape: single tuple
                "attention_mask": (1, 77),
            },
            extra_args=set(),
            build_timestamp=1234567890,
        )

        # Test exact match (should pass)
        new_shapes_exact = {"input_ids": (1, 77), "attention_mask": (1, 77)}
        result = metadata_static._shapes_fit_profile(new_shapes_exact)
        assert result is True, "Static exact match should pass"

        # Test different shape (should fail)
        new_shapes_different = {"input_ids": (1, 128), "attention_mask": (1, 77)}
        result = metadata_static._shapes_fit_profile(new_shapes_different)
        assert result is False, "Static different shape should fail"

    def test_dynamic_shapes(self):
        """Test dynamic shapes (list of 3 tuples)."""
        metadata_dynamic = EngineMetadata(
            model_name="test_model",
            precision="fp16",
            onnx_path="/path/to/model.onnx",
            onnx_hash="abcd1234",
            input_shapes={
                "latents": [(1, 4, 32, 32), (1, 4, 64, 64), (1, 4, 128, 128)],  # min, opt, max
                "timestep": [(1,), (1,), (1,)],  # Effectively static
            },
            extra_args=set(),
            build_timestamp=1234567890,
        )

        # Test shape within range (should pass)
        new_shapes_within = {"latents": (1, 4, 48, 48), "timestep": (1,)}
        result = metadata_dynamic._shapes_fit_profile(new_shapes_within)
        assert result is True, "Dynamic within range should pass"

        # Test shape at min boundary (should pass)
        new_shapes_min = {"latents": (1, 4, 32, 32), "timestep": (1,)}
        result = metadata_dynamic._shapes_fit_profile(new_shapes_min)
        assert result is True, "Dynamic at min boundary should pass"

        # Test shape at max boundary (should pass)
        new_shapes_max = {"latents": (1, 4, 128, 128), "timestep": (1,)}
        result = metadata_dynamic._shapes_fit_profile(new_shapes_max)
        assert result is True, "Dynamic at max boundary should pass"

        # Test shape below min (should fail)
        new_shapes_below = {"latents": (1, 4, 16, 16), "timestep": (1,)}
        result = metadata_dynamic._shapes_fit_profile(new_shapes_below)
        assert result is False, "Dynamic below min should fail"

        # Test shape above max (should fail)
        new_shapes_above = {"latents": (1, 4, 256, 256), "timestep": (1,)}
        result = metadata_dynamic._shapes_fit_profile(new_shapes_above)
        assert result is False, "Dynamic above max should fail"

    def test_effectively_static_shapes(self):
        """Test effectively static shapes (all 3 tuples are the same)."""
        metadata_effectively_static = EngineMetadata(
            model_name="test_model",
            precision="fp16",
            onnx_path="/path/to/model.onnx",
            onnx_hash="abcd1234",
            input_shapes={
                "embeddings": [(1, 77, 768), (1, 77, 768), (1, 77, 768)],  # All the same = static
                "mask": [(1, 77), (1, 77), (1, 77)],
            },
            extra_args=set(),
            build_timestamp=1234567890,
        )

        # Test exact match (should pass)
        new_shapes_exact = {"embeddings": (1, 77, 768), "mask": (1, 77)}
        result = metadata_effectively_static._shapes_fit_profile(new_shapes_exact)
        assert result is True, "Effectively static exact match should pass"

        # Test different shape (should fail)
        new_shapes_different = {"embeddings": (1, 77, 512), "mask": (1, 77)}
        result = metadata_effectively_static._shapes_fit_profile(new_shapes_different)
        assert result is False, "Effectively static different shape should fail"

    def test_mixed_static_and_dynamic(self):
        """Test mixed static and dynamic shapes."""
        metadata_mixed = EngineMetadata(
            model_name="test_model",
            precision="fp16",
            onnx_path="/path/to/model.onnx",
            onnx_hash="abcd1234",
            input_shapes={
                "static_input": (1, 768),  # Static
                "dynamic_input": [(1, 1, 512), (1, 4, 512), (1, 8, 512)],  # Dynamic
            },
            extra_args=set(),
            build_timestamp=1234567890,
        )

        # Test valid combination (should pass)
        new_shapes_valid = {"static_input": (1, 768), "dynamic_input": (1, 6, 512)}
        result = metadata_mixed._shapes_fit_profile(new_shapes_valid)
        assert result is True, "Mixed valid should pass"

        # Test invalid static (should fail)
        new_shapes_invalid_static = {"static_input": (1, 512), "dynamic_input": (1, 6, 512)}
        result = metadata_mixed._shapes_fit_profile(new_shapes_invalid_static)
        assert result is False, "Mixed invalid static should fail"

        # Test invalid dynamic (should fail)
        new_shapes_invalid_dynamic = {"static_input": (1, 768), "dynamic_input": (1, 16, 512)}
        result = metadata_mixed._shapes_fit_profile(new_shapes_invalid_dynamic)
        assert result is False, "Mixed invalid dynamic should fail"
