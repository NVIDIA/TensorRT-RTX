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
Path Manager for cache directory structure and file operations.

Handles:
- Shared cache directory structure for ONNX and engines
- Direct file path access without symlinks
- File operations and cleanup
- Smart caching and usage tracking
"""

import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import NamedTuple, Optional, Union

from huggingface_hub import snapshot_download

# Initialize logger for this module
logger = logging.getLogger("rtx_demo.utils.path_manager")


class ModelConfig(NamedTuple):
    """Configuration for a model in a pipeline."""

    model_id: str
    precision: str
    shape_mode: str  # "static" or "dynamic"


class PathManager:
    """
    Path Manager for shared cache directories.

    Directory structure:
    cache_dir/
    ├── shared/                           # Canonical storage for all models
    │   ├── onnx/
    │   │   ├── model_id/
    │   │   │   └── precision/
    │   │   │       ├── model_id.onnx
    │   │   │       └── model_id.onnx.data (external data)
    │   │   │
    │   └── engines/
    │       ├── model_id/
    │       │   └── precision/
    │       │       ├── model_id_static.engine
    │       │       ├── model_id_dynamic.engine
    │       │       ├── model_id_static.metadata.json
    │       │       └── model_id_dynamic.metadata.json
    └── .cache_state.json                 # Pipeline usage tracking for cleanup
    """

    def __init__(self, cache_dir: str = "./demo_cache", cache_mode: str = "full"):
        """
        Initialize PathManager.

        Args:
            cache_dir: Base cache directory
            cache_mode: "lean" (delete unused models) or "full" (keep all models)
        """
        if cache_mode not in ["lean", "full"]:
            raise ValueError(f"cache_mode must be 'lean' or 'full', got: {cache_mode}")

        self.cache_dir = Path(cache_dir).resolve()
        self.cache_mode = cache_mode

        # Create base directories
        self.shared_dir = self.cache_dir / "shared"
        self.shared_onnx_dir = self.shared_dir / "onnx"
        self.shared_engines_dir = self.shared_dir / "engines"

        self._ensure_directories_exist([self.shared_onnx_dir, self.shared_engines_dir])

        # State file for persistent tracking
        self.state_file = self.cache_dir / ".cache_state.json"
        self.pipeline_states: dict[str, dict[str, ModelConfig]] = self._load_pipeline_states()

        logger.info(f"PathManager initialized - Cache: {self.cache_dir}, Mode: {self.cache_mode}")
        logger.debug(f"Loaded {len(self.pipeline_states)} pipeline states from cache")

    def _ensure_directories_exist(self, directories: list[Path]) -> None:
        """Ensure all directories exist."""
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def _ensure_model_directory_exists(self, model_id: str, precision: str, file_type: str) -> Path:
        """Ensure model directory exists and return the path."""
        if file_type == "onnx":
            model_dir = self.shared_onnx_dir / model_id / precision
        elif file_type == "engine":
            model_dir = self.shared_engines_dir / model_id / precision
        else:
            raise ValueError(f"Invalid file_type: {file_type}. Must be 'onnx' or 'engine'")

        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir

    def _load_pipeline_states(self) -> dict[str, dict[str, ModelConfig]]:
        """Load all accumulated pipeline states from persistent storage."""
        if not self.state_file.exists():
            logger.debug("No state file found, starting with empty pipeline states")
            return {}

        try:
            with open(self.state_file, encoding="utf-8") as f:
                data = json.load(f)

            # Convert back to proper format
            result = {}
            for pipeline_name, roles in data.items():
                result[pipeline_name] = {}
                for role, model_info in roles.items():
                    result[pipeline_name][role] = ModelConfig(
                        model_id=model_info["model_id"],
                        precision=model_info["precision"],
                        shape_mode=model_info["shape_mode"],
                    )

            logger.debug(f"Loaded {len(result)} pipeline states from {self.state_file}")
            all_tracked_models = self._get_all_models_from_states(result)
            logger.debug(f"Total models tracked: {len(all_tracked_models)}")

            return result
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to load pipeline states, starting fresh: {e}")
            return {}

    def _save_pipeline_states(self) -> None:
        """Save all accumulated pipeline states to persistent storage."""
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

            # Convert ModelConfig objects to serializable format
            serializable_states = {
                pipeline_name: {
                    role: {
                        "model_id": model_config.model_id,
                        "precision": model_config.precision,
                        "shape_mode": model_config.shape_mode,
                    }
                    for role, model_config in roles.items()
                }
                for pipeline_name, roles in self.pipeline_states.items()
            }

            # Save complete state (all pipelines) to JSON
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(serializable_states, f, indent=2)

            logger.debug(f"Saved {len(serializable_states)} pipeline states to {self.state_file}")
        except (OSError, json.JSONEncodeError) as e:
            logger.error(f"Failed to save pipeline states: {e}")

    def get_onnx_path(self, model_id: str, precision: str) -> Path:
        """Get ONNX file path in shared directory."""
        model_dir = self._ensure_model_directory_exists(model_id, precision, "onnx")
        return model_dir / f"{model_id}.onnx"

    def get_engine_path(self, model_id: str, precision: str, shape_mode: str) -> Path:
        """Get engine file path in shared directory."""
        model_dir = self._ensure_model_directory_exists(model_id, precision, "engine")
        return model_dir / f"{model_id}_{shape_mode}.engine"

    def get_metadata_path(self, model_id: str, precision: str, shape_mode: str) -> Path:
        """Get metadata file path in shared directory."""
        model_dir = self._ensure_model_directory_exists(model_id, precision, "engine")
        return model_dir / f"{model_id}_{shape_mode}.metadata.json"

    def get_model_directory(self, model_id: str, precision: str, file_type: str) -> Path:
        """Get the directory containing model files."""
        return self._ensure_model_directory_exists(model_id, precision, file_type)

    def check_cached_files(self, model_id: str, precision: str, shape_mode: str) -> dict:
        """Check which files exist in cache."""
        return {
            "onnx": self.get_onnx_path(model_id, precision).exists(),
            "engine": self.get_engine_path(model_id, precision, shape_mode).exists(),
            "metadata": self.get_metadata_path(model_id, precision, shape_mode).exists(),
        }

    def _safe_delete_file(self, file_path: Path) -> bool:
        """Safely delete a file with type checking and logging. Returns True if deleted."""
        if not file_path.exists():
            return False

        # Check if this is a known safe cache file type
        file_name = file_path.name.lower()
        safe_endings = {".onnx", ".engine", ".json", ".data", ".onnx_data", ".onnx.data"}
        is_safe = any(file_name.endswith(ending) for ending in safe_endings)

        if is_safe:
            file_path.unlink()
            logger.debug(f"Deleted: {file_path}")
            return True
        else:
            logger.warning(f"Skipping unknown file type in cache directory: {file_path}")
            logger.warning("This file will NOT be deleted for safety. Please review manually if needed.")
            return False

    def delete_cached_onnx_files(self, model_id: str, precision: str) -> None:
        """Delete cached ONNX files for a model."""
        logger.debug(f"Deleting cached ONNX files for {model_id}_{precision}")

        model_dir = self.get_model_directory(model_id, precision, "onnx")
        if not model_dir.exists():
            return

        # Delete all files in the ONNX directory with safety checks
        try:
            for file_path in model_dir.iterdir():
                if file_path.is_file():
                    self._safe_delete_file(file_path)

            self._cleanup_empty_dirs(model_id, precision, "onnx")
        except OSError as e:
            logger.error(f"Failed to delete ONNX files for {model_id}_{precision}: {e}")

    def delete_cached_engine_files(self, model_id: str, precision: str, shape_mode: str) -> None:
        """Delete cached engine files for a model."""
        logger.debug(f"Deleting cached {shape_mode} engine files for {model_id}_{precision}")

        files_to_delete = [
            self.get_engine_path(model_id, precision, shape_mode),
            self.get_metadata_path(model_id, precision, shape_mode),
        ]

        try:
            for file_path in files_to_delete:
                self._safe_delete_file(file_path)

            self._cleanup_empty_dirs(model_id, precision, "engine")
        except OSError as e:
            logger.error(f"Failed to delete engine files for {model_id}_{precision}_{shape_mode}: {e}")

    def delete_cached_files(self, model_id: str, precision: str, shape_mode: str) -> None:
        """Delete all cached files (both ONNX and engine files) for a model."""
        self.delete_cached_onnx_files(model_id, precision)
        self.delete_cached_engine_files(model_id, precision, shape_mode)

    def list_cached_models(self) -> dict:
        """List all cached models."""
        cached = {"onnx": [], "engines": []}

        # Scan for ONNX models
        if self.shared_onnx_dir.exists():
            for model_dir in self.shared_onnx_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                for precision_dir in model_dir.iterdir():
                    if precision_dir.is_dir() and any(precision_dir.glob("*.onnx")):
                        cached["onnx"].append(f"{model_dir.name}_{precision_dir.name}")

        # Scan for engine models
        if self.shared_engines_dir.exists():
            for model_dir in self.shared_engines_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                for precision_dir in model_dir.iterdir():
                    if precision_dir.is_dir() and any(precision_dir.glob("*.engine")):
                        cached["engines"].append(f"{model_dir.name}_{precision_dir.name}")

        return cached

    def print_cache_summary(self) -> None:
        """Print cache summary."""
        cached = self.list_cached_models()

        logger.info(f"\nCache Summary ({self.cache_dir}):")
        logger.info(f"ONNX models: {len(cached.get('onnx', []))}")
        for model in cached.get("onnx", []):
            logger.info(f"  {model}")

        logger.info(f"Engine models: {len(cached.get('engines', []))}")
        for model in cached.get("engines", []):
            logger.info(f"  {model}")

    def _copy_local_files(self, source_path: Path, target_dir: Path) -> bool:
        """Copy files from local source to target directory."""
        try:
            source_dir = source_path.parent
            logger.debug(f"Copying ONNX from: {source_dir}")

            # Copy the main ONNX file with the correct name
            # Extract model_id from target directory: target_dir is .../model_id/precision/
            model_id = target_dir.parent.name
            if source_path.suffix == ".onnx":
                target_path = target_dir / f"{model_id}.onnx"
            else:
                target_path = target_dir / source_path.name
            shutil.copy2(source_path, target_path)

            # Copy ALL files from the source directory except the main file
            for source_file in source_dir.iterdir():
                if source_file.is_file() and source_file != source_path:
                    target_file = target_dir / source_file.name
                    shutil.copy2(source_file, target_file)

            logger.debug(f"ONNX files copied to: {target_dir}")
            return True
        except OSError as e:
            logger.error(f"Failed to copy local files: {e}")
            return False

    def _download_from_repository(
        self, onnx_repository: str, onnx_subfolder: str, target_dir: Path, hf_token: Optional[str]
    ) -> bool:
        """Download files from HuggingFace repository."""
        try:
            logger.debug(f"Downloading ONNX from: {onnx_repository}/{onnx_subfolder}")

            with tempfile.TemporaryDirectory() as temp_dir:
                temp_download_path = Path(temp_dir)

                snapshot_download(
                    repo_id=onnx_repository,
                    allow_patterns=os.path.join(onnx_subfolder, "*"),
                    local_dir=temp_download_path,
                    token=hf_token,
                )

                # Move downloaded files to target directory
                source_files_dir = temp_download_path / onnx_subfolder
                if not source_files_dir.exists():
                    logger.error(f"Downloaded files not found at {source_files_dir}")
                    return False

                for source_file in source_files_dir.iterdir():
                    if source_file.is_file():
                        # Rename the ONNX file to the expected name based on the model_id from target directory structure
                        if source_file.suffix == ".onnx":
                            # Extract model_id from target directory: target_dir is .../model_id/precision/
                            model_id = target_dir.parent.name
                            file_name = f"{model_id}.onnx"
                        else:
                            file_name = source_file.name
                        target_file = target_dir / file_name
                        shutil.move(str(source_file), str(target_file))

                return True
        except Exception as e:
            logger.error(f"Failed to download from repository: {e}")
            return False

    def acquire_onnx_file(
        self,
        model_id: str,
        precision: str,
        onnx_local_path: Optional[str] = None,
        onnx_repository: Optional[str] = None,
        onnx_subfolder: Optional[str] = None,
        hf_token: Optional[str] = None,
    ) -> bool:
        """
        Acquire ONNX file from source (URL or local path) and store in shared cache.

        For local paths: Copies ALL files from the source directory.
        For URLs: Downloads the specified files and moves them to the target location.
        """
        # Validate arguments
        if not ((onnx_subfolder and onnx_repository) or onnx_local_path):
            raise ValueError("Either onnx_local_path or both onnx_repository and onnx_subfolder must be provided")

        logger.debug(f"Acquiring ONNX file: {onnx_repository}/{onnx_subfolder} {model_id}_{precision}")
        onnx_path = self.get_onnx_path(model_id, precision)

        if onnx_path.exists():
            logger.info(f"ONNX already exists: {onnx_path}")
            return True

        target_dir = onnx_path.parent

        # Try local path first, then repository
        if onnx_local_path and Path(onnx_local_path).exists():
            return self._copy_local_files(Path(onnx_local_path), target_dir)
        elif onnx_repository and onnx_subfolder:
            return self._download_from_repository(onnx_repository, onnx_subfolder, target_dir, hf_token)
        else:
            logger.error("No valid source provided or local file does not exist")
            return False

    def _convert_config_input(self, model_configs: dict[str, Union[ModelConfig, tuple]]) -> dict[str, ModelConfig]:
        """Convert input configurations to ModelConfig objects."""
        converted_configs = {}
        for role, config in model_configs.items():
            if isinstance(config, tuple) and len(config) >= 2:
                # Handle tuple format (model_id, precision) or (model_id, precision, shape_mode)
                shape_mode = config[2] if len(config) > 2 else "static"
                converted_configs[role] = ModelConfig(model_id=config[0], precision=config[1], shape_mode=shape_mode)
            elif isinstance(config, ModelConfig):
                converted_configs[role] = config
            else:
                logger.error(f"Invalid config format for {role}: {config}")
                continue

        return converted_configs

    def set_pipeline_models(self, pipeline_name: str, model_configs: dict[str, Union[ModelConfig, tuple]]) -> None:
        """
        Set the models needed for current pipeline and handle cleanup in lean mode.

        In full mode: Accumulates all pipeline states in JSON for complete cache tracking.
        In lean mode: Uses accumulated state to clean up all unused models from previous pipelines.

        Args:
            pipeline_name: Name of the pipeline
            model_configs: Dict of {role: ModelConfig or tuple} for this pipeline
        """
        converted_configs = self._convert_config_input(model_configs)

        if self.cache_mode == "lean":
            self._handle_lean_mode_cleanup(pipeline_name, converted_configs)
        else:
            # Full mode - accumulate all pipeline states for complete cache tracking
            logger.debug(f"Full mode: Adding/updating pipeline '{pipeline_name}' to accumulated state")
            self.pipeline_states[pipeline_name] = converted_configs.copy()
            logger.debug(f"Full mode: Saving state with {len(self.pipeline_states)} total pipelines")
            self._save_pipeline_states()

    def _handle_lean_mode_cleanup(self, pipeline_name: str, model_configs: dict[str, ModelConfig]) -> None:
        """Handle cleanup logic for lean mode - only keep the current pipeline active."""
        logger.debug(f"Lean mode: Setting pipeline '{pipeline_name}' as the only active pipeline")

        # Get all models from previously tracked pipelines (accumulated state)
        all_old_models = self._get_all_models_from_states(self.pipeline_states)
        old_pipeline_names = list(self.pipeline_states.keys())

        logger.debug(
            f"Lean mode: Found {len(all_old_models)} models from {len(old_pipeline_names)} previous pipelines: {old_pipeline_names}"
        )

        # In lean mode: only keep the current pipeline, clear all others
        self.pipeline_states.clear()
        self.pipeline_states[pipeline_name] = model_configs.copy()

        # Get models needed by the current pipeline
        current_models = self._get_all_models_from_states(self.pipeline_states)

        # Delete all models not needed by current pipeline
        models_to_cleanup = all_old_models - current_models

        if models_to_cleanup:
            logger.info(f"Lean mode: Cleaning up {len(models_to_cleanup)} unused models from previous pipelines")
            for model_id, precision, shape_mode in models_to_cleanup:
                logger.debug(f"  Cleaning up: {model_id}_{precision}_{shape_mode}")
                self._cleanup_unused_model(model_id, precision, shape_mode)
        else:
            logger.debug("Lean mode: No models to cleanup - current pipeline uses all previously cached models")

        logger.debug(
            f"Lean mode: Now tracking only current pipeline '{pipeline_name}' with {len(current_models)} models"
        )
        self._save_pipeline_states()

    def _get_all_models_from_states(self, pipeline_states: dict[str, dict[str, ModelConfig]]) -> set[tuple]:
        """Extract all models from pipeline states."""
        all_models = set()
        for pipeline_config in pipeline_states.values():
            for model_config in pipeline_config.values():
                if not isinstance(model_config, ModelConfig):
                    logger.error(f"Invalid model config type: {type(model_config)} - {model_config}")
                    continue
                all_models.add((model_config.model_id, model_config.precision, model_config.shape_mode))
        return all_models

    def _has_other_active_shape_modes(self, model_id: str, precision: str, current_shape_mode: str) -> bool:
        """Check if other shape modes of the same model + precision are still active."""
        for pipeline_config in self.pipeline_states.values():
            for model_config in pipeline_config.values():
                if (
                    model_config.model_id == model_id
                    and model_config.precision == precision
                    and model_config.shape_mode != current_shape_mode
                ):
                    logger.debug(f"Found other active shape mode: {model_config.shape_mode} for {model_id}_{precision}")
                    return True
        return False

    def _cleanup_unused_model(self, model_id: str, precision: str, shape_mode: str) -> None:
        """Clean up an unused model in lean mode. Only deletes ONNX if no other shape_mode is active."""
        logger.debug(f"Cleaning up unused model: {model_id}_{precision}_{shape_mode}")

        files_exist = self.check_cached_files(model_id, precision, shape_mode)

        if not any(files_exist.values()):
            return

        # Always delete engine files for this specific shape_mode
        if files_exist["engine"] or files_exist["metadata"]:
            self.delete_cached_engine_files(model_id, precision, shape_mode)

        # Only delete ONNX files if no other shape_mode of this model is still active
        if files_exist["onnx"]:
            should_delete_onnx = not self._has_other_active_shape_modes(model_id, precision, shape_mode)
            if should_delete_onnx:
                logger.debug(f"No other shape modes active for {model_id}_{precision}, deleting ONNX")
                self.delete_cached_onnx_files(model_id, precision)
            else:
                logger.debug(f"Other shape modes still active for {model_id}_{precision}, keeping ONNX")

    def _cleanup_empty_dirs(self, model_id: str, precision: str, file_type: str) -> None:
        """Clean up empty directories after model deletion."""
        if file_type == "onnx":
            precision_dir = self.shared_onnx_dir / model_id / precision
            model_dir = self.shared_onnx_dir / model_id
        elif file_type == "engine":
            precision_dir = self.shared_engines_dir / model_id / precision
            model_dir = self.shared_engines_dir / model_id
        else:
            return

        # Clean up precision directory if empty
        self._remove_if_empty(precision_dir)
        # Clean up model directory if empty
        self._remove_if_empty(model_dir)

    def _remove_if_empty(self, directory: Path) -> None:
        """Remove directory if it exists and is empty."""
        if directory.exists():
            try:
                if not any(directory.iterdir()):
                    directory.rmdir()
                    logger.debug(f"Removed empty directory: {directory}")
            except OSError:
                pass  # Directory not empty or other issue
