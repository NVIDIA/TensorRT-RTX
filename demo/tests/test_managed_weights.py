# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import ctypes
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import utils.managed_weights as managed_weights_module
from utils.managed_weights import (
    ManagedWeightsScheduler,
    _BufferWeightLoader,
)
from utils.path_manager import PathManager


@pytest.fixture(autouse=True)
def forbid_redundant_cuda_calls(monkeypatch):
    for api, name in (
        (managed_weights_module.cuda, "cuInit"),
        (managed_weights_module.cudart, "cudaSetDevice"),
        (managed_weights_module.cudart, "cudaMemGetInfo"),
    ):
        monkeypatch.setattr(api, name, Mock(side_effect=AssertionError(f"Unexpected {name} in weight management")))


class _FakeManager:
    """Minimal weights-manager implementation for scheduler tests."""

    def __init__(self, size):
        self.size = size
        self.unload_count = 0

    def get_size(self):
        return self.size

    def unload(self, stream):
        del stream
        self.unload_count += 1
        return True


class _FakeEngine:
    """Return one fixed weights manager."""

    def __init__(self, manager):
        self.manager = manager

    def create_weights_manager(self):
        return self.manager


class _FakeLoader:
    """Record scheduler operations without allocating GPU memory."""

    def __init__(self):
        self.operations = []

    def restore(self, path, size, manager, stream, name):
        self.operations.append(("restore", path.stem, size))
        return [path.stem]

    def release_blocks(self, blocks):
        self.operations.append(("release", tuple(blocks)))

    def close(self):
        self.operations.append(("close",))


def make_scheduler(tmp_path):
    loader = _FakeLoader()
    scheduler = ManagedWeightsScheduler("vmm", execution_stream=7, device=0, loader=loader)
    sizes = {
        "t5_text_encoder": 100,
        "clip_text_encoder": 200,
        "transformer": 500,
        "vae_decoder": 50,
    }
    for name, size in sizes.items():
        backup = tmp_path / f"{name}.weights"
        plan = tmp_path / f"{name}.engine"
        backup.touch()
        plan.touch()
        scheduler.register_engine(name, _FakeEngine(_FakeManager(size)), tmp_path / f"{name}.onnx", plan, backup)
    return scheduler, loader


@pytest.mark.unit
def test_scheduler_restores_stages_on_demand(tmp_path):
    scheduler, loader = make_scheduler(tmp_path)
    scheduler.configure_memory_pool()
    assert scheduler.weight_budget == 500

    with scheduler.use_weights("t5_text_encoder"):
        assert scheduler.entries["t5_text_encoder"].state == "running"
        assert scheduler.entries["clip_text_encoder"].state == "unloaded"

    assert scheduler.entries["t5_text_encoder"].state == "unloaded"
    assert not any(operation[:2] == ("restore", "clip_text_encoder") for operation in loader.operations)
    with scheduler.use_weights("clip_text_encoder"):
        assert scheduler.entries["clip_text_encoder"].state == "running"

    assert scheduler.entries["clip_text_encoder"].state == "unloaded"
    restore_clip = loader.operations.index(("restore", "clip_text_encoder", 200))
    release_t5 = loader.operations.index(("release", ("t5_text_encoder",)))
    assert release_t5 < restore_clip
    scheduler.close()


@pytest.mark.unit
def test_initialization_requires_configured_pool(tmp_path, monkeypatch):
    scheduler, loader = make_scheduler(tmp_path)
    entry = scheduler.entries["t5_text_encoder"]
    monkeypatch.setattr(scheduler, "_backup_is_valid", lambda entry: True)

    with pytest.raises(RuntimeError, match="Managed-weight memory plan has not been configured"):
        scheduler.initialize_weights(entry.name)
    assert loader.operations == []

    scheduler.configure_memory_pool()
    scheduler.initialize_weights(entry.name)
    assert entry.state == "loaded"
    assert entry.weights_owned
    scheduler.close()


@pytest.mark.unit
def test_pool_size_matches_largest_engine(tmp_path):
    loader = _FakeLoader()
    scheduler = ManagedWeightsScheduler("vmm", execution_stream=7, device=0, loader=loader)
    plan = tmp_path / "large.engine"
    plan.touch()
    scheduler.register_engine(
        "large",
        _FakeEngine(_FakeManager(2 * 1024 * 1024 + 1)),
        tmp_path / "large.onnx",
        plan,
        tmp_path / "large.weights",
    )

    scheduler.configure_memory_pool()
    assert scheduler.weight_budget == 2 * 1024 * 1024 + 1
    scheduler.close()


@pytest.mark.unit
def test_managed_weight_manifest(tmp_path):
    scheduler, loader = make_scheduler(tmp_path)
    scheduler.configure_memory_pool()
    entry = scheduler.entries["t5_text_encoder"]
    entry.backup_path.write_bytes(b"x" * entry.size)
    manifest = scheduler._make_manifest(entry)
    assert manifest["format_version"] == managed_weights_module.MANIFEST_VERSION
    assert manifest["tensorrt_version"] == managed_weights_module.trt.__version__
    entry.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert scheduler._backup_is_valid(entry)

    manifest["cuda_allocation_granularity"] = 4096
    manifest["informational_field"] = "ignored"
    entry.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert scheduler._backup_is_valid(entry)

    manifest["source_identity"] = "different-checkpoint"
    entry.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert not scheduler._backup_is_valid(entry)
    manifest = scheduler._make_manifest(entry)
    manifest["plan_sha256"] = "stale"
    entry.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert not scheduler._backup_is_valid(entry)

    scheduler.close()


@pytest.mark.unit
@pytest.mark.parametrize("field", ["format_version", "tensorrt_version"])
@pytest.mark.parametrize("missing", [False, True], ids=["mismatch", "missing"])
def test_managed_weight_manifest_rejects_invalid_version(tmp_path, field, missing):
    scheduler, _ = make_scheduler(tmp_path)
    entry = scheduler.entries["t5_text_encoder"]
    entry.backup_path.write_bytes(b"x" * entry.size)
    manifest = scheduler._make_manifest(entry)
    if missing:
        del manifest[field]
    else:
        manifest[field] = (
            managed_weights_module.MANIFEST_VERSION + 1
            if field == "format_version"
            else managed_weights_module.trt.__version__ + ".different-build"
        )
    entry.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert not scheduler._backup_is_valid(entry)
    scheduler.close()


@pytest.mark.unit
def test_buffer_loader_restores_file_backed_host_storage(tmp_path, monkeypatch):
    payload = bytes(range(32))
    weights_path = tmp_path / "stage.weights"
    weights_path.write_bytes(payload)
    loader = _BufferWeightLoader()

    class BufferManager:
        restored = None

        def restore_from_buffer(self, pointer, size, location, stream):
            del location, stream
            self.restored = ctypes.string_at(pointer, size)
            return True

    monkeypatch.setattr(managed_weights_module, "_cuda_assert", lambda call, operation: None)
    monkeypatch.setattr(
        managed_weights_module.cudart,
        "cudaStreamSynchronize",
        lambda stream: (managed_weights_module.cudart.cudaError_t.cudaSuccess,),
    )
    manager = BufferManager()
    assert loader.restore(weights_path, len(payload), manager, 7, "stage") == []
    assert loader._pending is None
    assert manager.restored == payload


@pytest.mark.unit
def test_buffer_loader_keeps_source_alive_until_failed_restore_is_unloaded(tmp_path, monkeypatch):
    weights_path = tmp_path / "stage.weights"
    weights_path.write_bytes(bytes(range(32)))
    loader = _BufferWeightLoader()

    class BufferManager:
        unload_count = 0

        @staticmethod
        def restore_from_buffer(pointer, size, location, stream):
            del pointer, size, location, stream
            return True

        def unload(self, stream):
            del stream
            self.unload_count += 1
            return True

    monkeypatch.setattr(
        managed_weights_module.cudart,
        "cudaStreamSynchronize",
        lambda stream: (managed_weights_module.cudart.cudaError_t.cudaErrorUnknown,),
    )
    manager = BufferManager()
    with pytest.raises(RuntimeError, match="buffer restore"):
        loader.restore(weights_path, weights_path.stat().st_size, manager, 7, "stage")
    pending = loader._pending
    storage = pending.storage
    assert pending.engine_mapping_started
    assert not pending.storage._mmap.closed

    loader.close()
    assert manager.unload_count == 1
    assert storage._mmap.closed
    assert pending.storage is None
    assert loader._pending is None


@pytest.mark.unit
@pytest.mark.parametrize("operation", ["refit", "restore"])
@pytest.mark.parametrize("fail", [False, True])
def test_scheduler_propagates_allocation_errors(tmp_path, monkeypatch, operation, fail):
    scheduler, loader = make_scheduler(tmp_path)
    scheduler.configure_memory_pool()
    entry = scheduler.entries["t5_text_encoder"]

    def allocate(*args):
        if fail:
            raise RuntimeError("device allocation failed")
        return 1 if operation == "refit" else []

    if operation == "refit":
        entry.refit = allocate
        monkeypatch.setattr(scheduler, "_save_backup", lambda entry: None)
    else:
        monkeypatch.setattr(loader, "restore", allocate)

    monkeypatch.setattr(scheduler, "_backup_is_valid", lambda entry: operation == "restore")

    def action():
        scheduler.initialize_weights(entry.name)

    if fail:
        with pytest.raises(RuntimeError, match="device allocation failed"):
            action()
        assert entry.state == "failed"
        assert not entry.weights_owned
    else:
        action()
        assert entry.state == "loaded"
        assert entry.weights_owned
    scheduler.close()
    assert ("close",) in loader.operations


@pytest.mark.unit
@pytest.mark.parametrize("raises", [False, True])
def test_buffer_allocation_failure_closes_source(tmp_path, monkeypatch, raises):
    path = tmp_path / "stage.weights"
    path.write_bytes(bytes(range(32)))
    loader = _BufferWeightLoader()
    sources = []
    memmap = managed_weights_module.np.memmap

    def open_source(*args, **kwargs):
        storage = memmap(*args, **kwargs)
        sources.append(storage)
        return storage

    monkeypatch.setattr(managed_weights_module.np, "memmap", open_source)
    manager = SimpleNamespace(
        restore_from_buffer=Mock(
            side_effect=RuntimeError("device allocation failed") if raises else None,
            return_value=False,
        )
    )
    with pytest.raises(RuntimeError, match="device allocation failed|Failed to restore managed weights"):
        loader.restore(path, 32, manager, 7, "stage")
    assert len(sources) == 1 and sources[0]._mmap.closed
    assert loader._pending is None


@pytest.mark.unit
def test_failed_loaded_entry_is_unloaded_during_close(tmp_path):
    scheduler, _ = make_scheduler(tmp_path)
    scheduler.configure_memory_pool()
    entry = scheduler.entries["transformer"]
    manager = entry.manager
    entry.state = "failed"
    entry.weights_owned = True

    scheduler.close()

    assert manager.unload_count == 1


@pytest.mark.unit
def test_scheduler_accepts_three_engine_registry(tmp_path):
    loader = _FakeLoader()
    scheduler = ManagedWeightsScheduler("vmm", execution_stream=7, device=0, loader=loader)
    engine_order = ["text_encoder", "transformer", "vae_decoder"]
    for index, name in enumerate(engine_order, start=1):
        plan = tmp_path / f"{name}.engine"
        backup = tmp_path / f"{name}.weights"
        plan.touch()
        backup.touch()
        scheduler.register_engine(
            name,
            _FakeEngine(_FakeManager(index * 64)),
            tmp_path / f"{name}.onnx",
            plan,
            backup,
        )
    scheduler.configure_memory_pool()

    for name in engine_order:
        with scheduler.use_weights(name):
            assert scheduler.entries[name].state == "running"

    assert len(scheduler.entries) == 3
    assert all(entry.state == "unloaded" for entry in scheduler.entries.values())
    scheduler.close()


@pytest.mark.unit
def test_twenty_restore_run_unload_cycles_leave_no_loaded_weights(tmp_path):
    scheduler, _ = make_scheduler(tmp_path)
    engine_order = list(scheduler.entries)
    scheduler.configure_memory_pool()

    for _ in range(20):
        for name in engine_order:
            with scheduler.use_weights(name):
                pass
        assert all(entry.state == "unloaded" for entry in scheduler.entries.values())

    assert all(entry.manager.unload_count == 20 for entry in scheduler.entries.values())
    scheduler.close()


@pytest.mark.unit
@pytest.mark.cache
def test_weightless_cache_files_are_isolated(path_manager: PathManager):
    model_id = "test_model"
    precision = "bf16"
    shape_mode = "static"
    normal_engine = path_manager.get_engine_path(model_id, precision, shape_mode)
    weightless_engine = path_manager.get_engine_path(model_id, precision, shape_mode, weightless=True)
    assert normal_engine != weightless_engine
    assert normal_engine.parent == weightless_engine.parent
    assert "_weightless_static.engine" in weightless_engine.name


@pytest.mark.unit
def test_scheduler_supports_pinned_buffer_weights():
    scheduler = ManagedWeightsScheduler("buffer", 7, 0, pinned_host=True)
    assert isinstance(scheduler.loader, _BufferWeightLoader)
    assert scheduler.loader.pinned_weight_cache.enabled
    scheduler.close()
