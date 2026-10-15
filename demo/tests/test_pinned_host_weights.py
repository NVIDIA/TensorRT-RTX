# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise persistent pinned sources without a CUDA device."""

import ctypes
from pathlib import Path
from unittest.mock import Mock

import pytest
from utils import managed_weights as weights
from weight_test_utils import BlockFactory, Manager, make_weight_file, restore_and_check, update_weight_file


@pytest.fixture(params=["buffer", "vmm"])
def loader(monkeypatch, request):
    success = weights.cudart.cudaError_t.cudaSuccess
    allocations = {}
    operations = []

    def allocate(size):
        storage = ctypes.create_string_buffer(size)
        address = ctypes.addressof(storage)
        allocations[address] = storage
        return success, address

    def free(address):
        operations.append("free")
        allocations.pop(int(address))
        return (success,)

    def copy(destination, source, size, kind, stream):
        operations.append("copy")
        ctypes.memmove(destination, source, size)
        return (success,)

    def synchronize(stream):
        operations.append("sync")
        return (success,)

    monkeypatch.setattr(weights.cudart, "cudaMallocHost", allocate)
    monkeypatch.setattr(weights.cudart, "cudaFreeHost", free)
    monkeypatch.setattr(weights.cudart, "cudaMemcpyAsync", copy)
    monkeypatch.setattr(weights.cudart, "cudaStreamSynchronize", synchronize)
    monkeypatch.setattr(weights.cudart, "cudaStreamDestroy", lambda *args: (success,))
    if request.param == "buffer":
        instance = weights._BufferWeightLoader(pinned_host=True)
    else:
        instance = weights._VmmWeightLoader.__new__(weights._VmmWeightLoader)
        instance.stream, instance.granularity = 7, 4
        instance.pool = weights._VmmBlockPool(0, 4, block_factory=BlockFactory())
        instance.pool.configure(list(range(64, 513, 64)))
        instance._pending = None
        instance.staging = []
        instance.pinned_weight_cache = weights._PinnedWeightCache(True)
        instance.copy_chunk_size = 16
    yield instance
    before_close = len(operations)
    instance.close()
    if request.param == "vmm":
        assert instance.pool._block_factory.allocated_bytes == 0
        assert operations[before_close] == "sync"
        before_close += 1
    assert not allocations
    assert all(operation == "free" for operation in operations[before_close:])


@pytest.mark.unit
def test_setup_eliminates_payload_reads_and_keeps_gpu_reuse(loader, tmp_path, monkeypatch):
    large = make_weight_file(tmp_path, "large", 8 * 64, 17)
    small = make_weight_file(tmp_path, "small", 2 * 64, 91)
    vmm = isinstance(loader, weights._VmmWeightLoader)
    if vmm:
        loader.pool.configure([2 * 64, 8 * 64])
    for path in (large, small):
        loader.prepare_pinned_weights(path, path.stat().st_size)
    buffers = loader.pinned_weight_cache.pinned_weight_buffers
    assert sum(buffer.size for _, buffer in buffers.values()) == 10 * 64
    first = buffers[large.resolve()]
    loader.prepare_pinned_weights(large, 8 * 64)
    assert buffers[large.resolve()] is first
    payloads = {path: path.read_bytes() for path in (large, small)}

    def forbidden(*args, **kwargs):
        raise AssertionError("Inference must not open weight payload files")

    monkeypatch.setattr(Path, "open", forbidden)
    copy = Mock(wraps=weights.cudart.cudaMemcpyAsync)
    monkeypatch.setattr(weights.cudart, "cudaMemcpyAsync", copy)
    paths = (large, small, large)
    for path in paths:
        manager = Manager(payloads[path])
        blocks = loader.restore(path, len(manager.expected), manager, 7, path.stem)
        manager.check()
        loader.release_blocks(blocks)
    if vmm:
        assert sum(call.args[2] for call in copy.call_args_list) == 12 * 64
        assert loader.pool._block_factory.peak_bytes == 8 * 64
        assert len(loader.pool._block_factory.blocks) == 2
    else:
        copy.assert_not_called()


@pytest.mark.unit
def test_inference_requires_setup(loader, tmp_path):
    path = make_weight_file(tmp_path, "weights", 128, 29)
    with pytest.raises(RuntimeError, match="prepared during setup"):
        restore_and_check(loader, path)


@pytest.mark.unit
def test_replaced_file_rejects_stale_pinned_weights(loader, tmp_path):
    path = make_weight_file(tmp_path, "weights", 128, 13)
    loader.prepare_pinned_weights(path, 128)
    make_weight_file(tmp_path, "replacement", 128, 27).replace(path)
    with pytest.raises(RuntimeError, match="cache changed"):
        restore_and_check(loader, path)


@pytest.mark.unit
@pytest.mark.parametrize("loader", ["vmm"], indirect=True)
def test_copy_failure_returns_blocks_and_retains_source(loader, tmp_path, monkeypatch):
    path = make_weight_file(tmp_path, "weights", 128, 71)
    loader.prepare_pinned_weights(path, 128)
    original = weights.cudart.cudaMemcpyAsync
    calls = 0

    def fail(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected copy failure")
        return original(*args)

    with monkeypatch.context() as patch:
        patch.setattr(weights.cudart, "cudaMemcpyAsync", fail)
        with pytest.raises(RuntimeError, match="injected copy failure"):
            restore_and_check(loader, path)
    assert loader._pending is None
    assert loader.pinned_weight_cache.pinned_weight_buffers[path.resolve()][1].size == 128
    assert all(block.content is None for block in loader.pool._free)
    loader.release_blocks(restore_and_check(loader, path))


@pytest.mark.unit
def test_preparation_failure_frees_pinned_allocation(loader, tmp_path, monkeypatch):
    path = make_weight_file(tmp_path, "weights", 128, 37)

    original = weights._PinnedBuffer.read

    def mutate(buffer, *args):
        original(buffer, *args)
        with update_weight_file(path), path.open("r+b") as output:
            output.write(b"changed!")

    with monkeypatch.context() as patch:
        patch.setattr(weights._PinnedBuffer, "read", mutate)
        with pytest.raises(RuntimeError, match="file changed during pinned preparation"):
            loader.prepare_pinned_weights(path, 128)
    assert not loader.pinned_weight_cache.pinned_weight_buffers
    loader.prepare_pinned_weights(path, 128)


@pytest.mark.unit
def test_host_allocation_failure_can_be_retried(loader, tmp_path, monkeypatch):
    path = make_weight_file(tmp_path, "weights", 128, 41)
    with monkeypatch.context() as patch:
        patch.setattr(
            weights.cudart, "cudaMallocHost", lambda size: (weights.cudart.cudaError_t.cudaErrorMemoryAllocation, 0)
        )
        with pytest.raises(RuntimeError, match="cudaMallocHost"):
            loader.prepare_pinned_weights(path, 128)
    assert not loader.pinned_weight_cache.pinned_weight_buffers
    loader.prepare_pinned_weights(path, 128)


@pytest.mark.unit
@pytest.mark.parametrize("loader", ["vmm"], indirect=True)
def test_restore_queues_copy_and_inference_without_host_wait(loader, tmp_path, monkeypatch):
    path = make_weight_file(tmp_path, "weights", 128, 19)
    loader.prepare_pinned_weights(path, 128)
    pending = []
    handles = {}
    success = weights.cudart.cudaError_t.cudaSuccess
    sync = Mock(wraps=weights.cudart.cudaStreamSynchronize)
    monkeypatch.setattr(weights.cudart, "cudaStreamSynchronize", sync)

    def copy(destination, source, size, kind, stream):
        assert int(stream) == loader.stream
        pending.append(lambda: ctypes.memmove(destination, source, size))
        return (success,)

    def restore(handle, offset, size, stream):
        assert int(stream) == loader.stream
        handles[offset] = (handle, size)
        return True

    manager = Mock(restore_from_vmm_allocation=restore)
    monkeypatch.setattr(weights.cudart, "cudaMemcpyAsync", copy)
    blocks = loader.restore(path, 128, manager, 7, "stage")
    sync.assert_not_called()
    assert pending
    expected = path.read_bytes()
    observed = []
    pending.append(
        lambda: observed.append(
            b"".join(ctypes.string_at(handle, size) for _, (handle, size) in sorted(handles.items()))
        )
    )
    for operation in pending:
        operation()
    assert observed == [expected]
    loader.release_blocks(blocks)
    sync.assert_not_called()


@pytest.mark.unit
@pytest.mark.parametrize("loader", ["buffer"], indirect=True)
def test_buffer_restore_waits_for_copy_and_keeps_pinned_source(loader, tmp_path, monkeypatch):
    path = make_weight_file(tmp_path, "weights", 128, 19)
    expected = path.read_bytes()
    loader.prepare_pinned_weights(path, 128)
    buffer = loader.pinned_weight_cache.get(path, 128)[1]
    pending, observed = [], []

    def restore(pointer, size, location, stream):
        assert pointer == buffer.pointer
        assert location == weights.trt.TensorLocation.HOST
        assert stream == 7
        pending.append(lambda: observed.append(ctypes.string_at(pointer, size)))
        return True

    def synchronize(stream):
        assert stream == 7 and buffer.pointer
        for operation in pending:
            operation()
        pending.clear()
        return (weights.cudart.cudaError_t.cudaSuccess,)

    monkeypatch.setattr(weights.cudart, "cudaStreamSynchronize", synchronize)
    manager = Mock(restore_from_buffer=restore)
    for _ in range(2):
        assert loader.restore(path, 128, manager, 7, "stage") == []
        assert not pending and loader._pending is None
        assert buffer.pointer
    assert observed == [expected, expected]


@pytest.mark.unit
@pytest.mark.parametrize("loader", ["buffer"], indirect=True)
def test_buffer_restore_failure_keeps_pinned_source_until_unload(loader, tmp_path, monkeypatch):
    path = make_weight_file(tmp_path, "weights", 128, 19)
    loader.prepare_pinned_weights(path, 128)
    buffer = loader.pinned_weight_cache.get(path, 128)[1]
    manager = Mock(restore_from_buffer=Mock(return_value=True), unload=Mock(return_value=False))
    with monkeypatch.context() as patch:
        patch.setattr(weights.cudart, "cudaStreamSynchronize", Mock(side_effect=RuntimeError("sync failed")))
        with pytest.raises(RuntimeError, match="sync failed"):
            loader.restore(path, 128, manager, 7, "stage")
        assert loader._pending.storage is buffer
        with pytest.raises(RuntimeError, match="previous failed restore"):
            loader.restore(path, 128, manager, 7, "stage")
        with pytest.raises(RuntimeError, match="Failed to unload"):
            loader.close()
        assert buffer.pointer
        assert loader.pinned_weight_cache.get(path, 128)[1] is buffer
    manager.unload.return_value = True
    loader.close()
    assert not buffer.pointer
    assert loader._pending is None
    assert not loader.pinned_weight_cache.pinned_weight_buffers


@pytest.mark.unit
@pytest.mark.parametrize("loader", ["buffer"], indirect=True)
@pytest.mark.parametrize("raises", [False, True])
def test_buffer_restore_error_keeps_cache_for_retry(loader, tmp_path, raises):
    path = make_weight_file(tmp_path, "weights", 128, 19)
    loader.prepare_pinned_weights(path, 128)
    buffer = loader.pinned_weight_cache.get(path, 128)[1]
    manager = Mock(
        restore_from_buffer=Mock(
            side_effect=RuntimeError("restore failed") if raises else None,
            return_value=False,
        )
    )
    with pytest.raises(RuntimeError, match="restore failed|Failed to restore"):
        loader.restore(path, 128, manager, 7, "stage")
    assert loader._pending is None
    assert buffer.pointer
    loader.release_blocks(restore_and_check(loader, path))


@pytest.mark.unit
def test_pinned_cache_close_failure_can_be_retried(loader, tmp_path, monkeypatch):
    paths = [make_weight_file(tmp_path, name, 128, value) for name, value in (("first", 1), ("second", 2))]
    cache = loader.pinned_weight_cache
    for path in paths:
        loader.prepare_pinned_weights(path, 128)
    second = cache.get(paths[1], 128)[1]
    with monkeypatch.context() as patch:
        patch.setattr(second, "close", Mock(side_effect=RuntimeError("free failed")))
        with pytest.raises(RuntimeError, match="free failed"):
            cache.close()
        assert list(cache.pinned_weight_buffers) == [paths[1].resolve()]
        assert second.pointer
    cache.close()
    assert not cache.pinned_weight_buffers


@pytest.mark.unit
@pytest.mark.parametrize("loader", ["vmm"], indirect=True)
def test_close_sync_failure_preserves_pinned_sources_and_blocks(loader, tmp_path, monkeypatch):
    path = make_weight_file(tmp_path, "weights", 128, 19)
    loader.prepare_pinned_weights(path, 128)
    loader.release_blocks(restore_and_check(loader, path))
    with monkeypatch.context() as patch:
        patch.setattr(weights.cudart, "cudaStreamSynchronize", Mock(side_effect=RuntimeError("sync failed")))
        with pytest.raises(RuntimeError, match="sync failed"):
            loader.close()
        assert loader.pinned_weight_cache.pinned_weight_buffers[path.resolve()][1].pointer
        assert loader.pool._block_factory.allocated_bytes == 128
