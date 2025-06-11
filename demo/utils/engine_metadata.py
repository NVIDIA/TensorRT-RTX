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
Engine Metadata Management

Tracks engine compilation parameters to determine when recompilation is needed.
"""

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Set

# Initialize logger for this module
logger = logging.getLogger("rtx_demo.utils.engine_metadata")


@dataclass
class EngineMetadata:
    """Metadata for a compiled TensorRT engine"""

    model_name: str
    precision: str
    onnx_path: str
    onnx_hash: str
    input_shapes: Dict[str, Any]
    extra_args: Set[str]
    build_timestamp: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EngineMetadata":
        """Create from dictionary"""
        return cls(**data)

    def is_compatible_with(self, new_shapes: Dict[str, Any], new_extra_args: Optional[Set[str]] = None) -> bool:
        """Check if engine is compatible with new shapes and extra args"""
        new_extra_args = set() if new_extra_args is not None else set()
        if self.extra_args != new_extra_args:
            return False
        return self._shapes_fit_profile(new_shapes)

    def _shapes_fit_profile(self, new_shapes: Dict[str, Any]) -> bool:
        """Check if new shapes fit within dynamic profile"""
        for input_name, new_shape in new_shapes.items():
            if input_name not in self.input_shapes:
                return False

            profile = self.input_shapes[input_name]

            # Check if this is a dynamic profile (list/tuple of 3 tuples)
            is_dynamic_profile = (
                isinstance(profile, (list, tuple))
                and len(profile) == 3
                and all(isinstance(shape, (list, tuple)) for shape in profile)
            )

            if is_dynamic_profile:
                min_shape, opt_shape, max_shape = profile

                # Check if it's effectively static (all shapes are the same)
                if min_shape == opt_shape == max_shape:
                    if tuple(new_shape) != tuple(min_shape):
                        return False
                else:
                    # True dynamic profile - check if new shape fits within range
                    if len(new_shape) != len(min_shape) or len(new_shape) != len(max_shape):
                        return False

                    for new_dim, min_dim, max_dim in zip(new_shape, min_shape, max_shape):
                        if new_dim < min_dim or new_dim > max_dim:
                            return False
            else:
                # Static profile: profile is a single tuple
                if tuple(new_shape) != tuple(profile):
                    return False

        return True


class EngineMetadataManager:
    """Manages engine metadata files"""

    def _get_metadata_path(self, engine_path: Path) -> Path:
        """Get metadata file path for an engine"""
        return engine_path.with_suffix(".metadata.json")

    def _compute_onnx_hash(self, onnx_path: str) -> str:
        """Compute hash of ONNX file"""
        try:
            with open(onnx_path, "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()[:16]
        except Exception:
            return "unknown"

    def save_metadata(
        self,
        engine_path: Path,
        model_name: str,
        precision: str,
        onnx_path: str,
        input_shapes: Dict,
        extra_args: Optional[Set[str]] = None,
    ) -> None:
        """Save engine metadata to file."""
        metadata = {
            "model_name": model_name,
            "precision": precision,
            "onnx_path": str(onnx_path),
            "input_shapes": input_shapes,
            "extra_args": list(extra_args) if extra_args else [],
            "build_timestamp": time.time(),
            "tensorrt_version": self._get_tensorrt_version(),
        }

        metadata_path = engine_path.with_suffix(".metadata.json")

        try:
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            logger.debug(f"Saved engine metadata: {metadata_path}")

        except Exception as e:
            logger.warning(f"Failed to save metadata for {engine_path}: {e}, unnecessary recompilations may occur.")

    def load_metadata(self, metadata_path: Path) -> Optional[dict]:
        """Load engine metadata from file."""
        try:
            with open(metadata_path) as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load metadata from {metadata_path}: {e}")
            return None

    def check_engine_compatibility(
        self,
        engine_path: Path,
        target_shapes: Dict,
        static_shape: bool,
        extra_args: Optional[Set[str]] = None,
    ) -> tuple[bool, str]:
        """Check if an existing engine is compatible with current requirements."""
        if not engine_path.exists():
            return False, "No cached engine found"

        metadata_path = engine_path.with_suffix(".metadata.json")
        if not metadata_path.exists():
            return False, "No cached engine metadata found"

        try:
            metadata = self.load_metadata(metadata_path)
            if not metadata:
                return False, "Engine metadata corrupted or unreadable"

            # Check TensorRT version compatibility
            saved_trt_version = metadata.get("tensorrt_version", "unknown")
            current_trt_version = self._get_tensorrt_version()
            if saved_trt_version != current_trt_version:
                return (
                    False,
                    f"TensorRT version changed: engine built with {saved_trt_version}, current system has {current_trt_version}",
                )

            # Determine shape mode from engine filename
            is_static_engine = "_static." in str(engine_path)
            if is_static_engine != static_shape:
                return (
                    False,
                    f"Shape mode mismatch: cached engine is static:{is_static_engine}, user requested static:{static_shape}",
                )

            # Check shape compatibility based on engine type
            saved_shapes = metadata.get("input_shapes", {})

            if static_shape:
                # For static shapes, shapes must match exactly
                # Normalize both saved and target shapes to handle list/tuple format differences
                normalized_saved = self._normalize_shapes_for_comparison(saved_shapes)
                normalized_target = self._normalize_shapes_for_comparison(target_shapes)

                if normalized_saved != normalized_target:
                    return (
                        False,
                        f"Shape profile incompatible: cached engine expects {saved_shapes}, user requested {target_shapes}",
                    )
            else:
                # For dynamic shapes, check that target shapes fall within saved ranges
                for input_name, target_shape in target_shapes.items():
                    if input_name not in saved_shapes:
                        return False, f"Missing input '{input_name}' in saved shapes"

                    saved_shape = saved_shapes[input_name]

                    # Saved shape should be (min_shape, opt_shape, max_shape) for dynamic
                    if not isinstance(saved_shape, (list, tuple)) or len(saved_shape) != 3:
                        return (
                            False,
                            f"Invalid saved dynamic shape format for '{input_name}': {saved_shape}",
                        )

                    min_shape, opt_shape, max_shape = saved_shape

                    # Target shape should be a single shape that fits within min/max bounds
                    if (
                        isinstance(target_shape, (list, tuple))
                        and isinstance(target_shape[0], (list, tuple))
                        and len(target_shape) == 3
                    ):
                        # Target is also min/opt/max format - check compatibility
                        target_min, target_opt, target_max = target_shape
                        if target_min != min_shape or target_opt != opt_shape or target_max != max_shape:
                            return False, f"Dynamic shape range mismatch for '{input_name}'"
                    else:
                        # Target is single shape - check if it fits within bounds
                        target_shape = tuple(target_shape) if isinstance(target_shape, list) else target_shape
                        min_shape = tuple(min_shape) if isinstance(min_shape, list) else min_shape
                        max_shape = tuple(max_shape) if isinstance(max_shape, list) else max_shape

                        if len(target_shape) != len(min_shape):
                            return False, f"Shape dimension mismatch for '{input_name}'"

                        for i, (target_dim, min_dim, max_dim) in enumerate(zip(target_shape, min_shape, max_shape)):
                            if target_dim < min_dim or target_dim > max_dim:
                                return (
                                    False,
                                    f"Target shape dimension {i} ({target_dim}) outside bounds [{min_dim}, {max_dim}] for '{input_name}'",
                                )

            # Check extra args compatibility
            saved_args = set(metadata.get("extra_args", []))
            target_args = extra_args or set()
            if saved_args != target_args:
                return (
                    False,
                    f"Build configuration changed: cached engine used {saved_args or 'default settings'}, user requested {target_args or 'default settings'}",
                )

            return True, "Compatible"

        except Exception as e:
            return False, f"Error checking compatibility: {e}"

    def cleanup_metadata(self, engine_path: Path) -> None:
        """Remove metadata file"""
        metadata_path = self._get_metadata_path(engine_path)
        if metadata_path.exists():
            metadata_path.unlink()
            logger.debug(f"Removed metadata: {metadata_path}")

    def _get_tensorrt_version(self) -> str:
        """Get TensorRT version string"""
        try:
            import tensorrt_rtx as trt

            return trt.__version__
        except Exception:
            return "unknown"

    def _normalize_shapes_for_comparison(self, shapes: Dict) -> Dict:
        """Normalize shapes for comparison by converting lists to tuples recursively"""
        normalized = {}
        for input_name, shape in shapes.items():
            if isinstance(shape, list):
                # Convert list to tuple, handling nested lists as well
                normalized[input_name] = (
                    tuple(tuple(s) if isinstance(s, list) else s for s in shape)
                    if isinstance(shape[0], (list, tuple))
                    else tuple(shape)
                )
            elif isinstance(shape, tuple):
                # Ensure nested tuples are also normalized
                normalized[input_name] = (
                    tuple(tuple(s) if isinstance(s, list) else s for s in shape)
                    if len(shape) > 0 and isinstance(shape[0], (list, tuple))
                    else shape
                )
            else:
                normalized[input_name] = shape
        return normalized


# Global metadata manager instance
metadata_manager = EngineMetadataManager()
