# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check VMM cache contents and block reuse without a CUDA device."""

import ctypes
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from utils import managed_weights as weights
from utils.managed_weights import _VmmBlockPool
from weight_test_utils import (
    BlockFactory,
    Manager,
    MemoryBlock,
    StagingBuffer,
    copied_bytes,
    make_weight_file,
    restore_and_check,
    update_weight_file,
)


@pytest.fixture
def loader(monkeypatch):
    success = (weights.cudart.cudaError_t.cudaSuccess,)
    for name in ("cudaStreamSynchronize", "cudaStreamDestroy"):
        monkeypatch.setattr(weights.cudart, name, lambda *args: success)
    instance = weights._VmmWeightLoader.__new__(weights._VmmWeightLoader)
    instance.granularity = 4
    instance.stream = 7
    instance.pool = weights._VmmBlockPool(0, 4, block_factory=BlockFactory())
    instance.pool.configure(list(range(64, 513, 64)))
    instance._pending = None
    instance.staging = [StagingBuffer(), StagingBuffer()]
    instance.pinned_weight_cache = weights._PinnedWeightCache(False)
    instance.copy_chunk_size = 16
    yield instance
    if not instance.pool._closed:
        instance.close()
    assert instance.pool._block_factory.allocated_bytes == 0


@pytest.mark.unit
def test_restore_copies_only_overwritten_ranges(loader, tmp_path):
    large = make_weight_file(tmp_path, "large", 8 * 64, 3)
    small = make_weight_file(tmp_path, "small", 2 * 64, 113)
    loader.release_blocks(restore_and_check(loader, large))
    loader.release_blocks(restore_and_check(loader, small))
    copied_before = copied_bytes(loader)
    loader.release_blocks(restore_and_check(loader, large))
    assert copied_bytes(loader) - copied_before == 2 * 64
    copied_before = copied_bytes(loader)
    loader.release_blocks(restore_and_check(loader, large))
    assert copied_bytes(loader) == copied_before


@pytest.mark.unit
@pytest.mark.parametrize("replace", [False, True])
def test_changed_weight_file_is_not_a_cache_hit(loader, tmp_path, replace):
    path = make_weight_file(tmp_path, "weights", 8 * 64, 7)
    loader.release_blocks(restore_and_check(loader, path))
    if replace:
        make_weight_file(tmp_path, "replacement", 8 * 64, 97).replace(path)
    else:
        with update_weight_file(path):
            make_weight_file(tmp_path, "weights", 8 * 64, 97)
    copied_before = copied_bytes(loader)
    loader.release_blocks(restore_and_check(loader, path))
    assert copied_bytes(loader) - copied_before == 8 * 64


@pytest.mark.unit
def test_close_releases_cached_blocks_after_restore_cleanup_failure(loader, tmp_path, monkeypatch):
    path = make_weight_file(tmp_path, "weights", 8 * 64, 19)
    loader.release_blocks(restore_and_check(loader, path))
    manager = Manager(path.read_bytes())
    manager.restore_from_vmm_allocation = Mock(return_value=False)
    manager.unload = Mock(return_value=True)
    with monkeypatch.context() as patch:
        patch.setattr(weights.cudart, "cudaStreamSynchronize", Mock(side_effect=RuntimeError("sync failed")))
        with pytest.raises(RuntimeError, match="sync failed"):
            loader.restore(path, path.stat().st_size, manager, 7, path.stem)
        assert loader._pending is not None and loader._pending.ranges
        assert not loader.pool._free
    loader.close()
    assert loader._pending is None
    assert loader.pool._block_factory.allocated_bytes == 0


@pytest.mark.unit
def test_partial_allocation_failure_returns_blocks(loader, tmp_path, monkeypatch):
    path = make_weight_file(tmp_path, "weights", 3 * 64, 21)
    factory = loader.pool._block_factory
    allocations = []

    def allocate(*args):
        if allocations:
            raise RuntimeError("cuMemCreate failed: CUDA_ERROR_OUT_OF_MEMORY")
        block = factory(*args)
        allocations.append(block)
        return block

    with monkeypatch.context() as patch:
        patch.setattr(loader.pool, "_block_factory", allocate)
        manager = Manager(path.read_bytes())
        with pytest.raises(RuntimeError, match="CUDA_ERROR_OUT_OF_MEMORY"):
            loader.restore(path, path.stat().st_size, manager, 7, path.stem)
    assert loader._pending is None
    assert not manager.ranges
    assert loader.pool._free == allocations
    assert all(block.content is None for block in allocations)
    assert factory.allocated_bytes == 64
    loader.release_blocks(restore_and_check(loader, path))


@pytest.mark.unit
def test_partial_copy_failure_does_not_cache_incomplete_block(loader, tmp_path, monkeypatch):
    path = make_weight_file(tmp_path, "weights", 8 * 64, 21)

    def fail(*args):
        raise RuntimeError("injected copy failure")

    with monkeypatch.context() as patch:
        patch.setattr(loader.staging[1], "copy_async", fail)
        with pytest.raises(RuntimeError, match="injected copy failure"):
            restore_and_check(loader, path)
    assert loader._pending is None
    copied_before = copied_bytes(loader)
    loader.release_blocks(restore_and_check(loader, path))
    assert copied_bytes(loader) - copied_before == 8 * 64


@pytest.mark.unit
def test_file_mutation_during_copy_invalidates_pending_contents(loader, tmp_path, monkeypatch):
    path = make_weight_file(tmp_path, "weights", 8 * 64, 7)
    original = loader.staging[0].copy_async

    def mutate(*args):
        original(*args)
        with update_weight_file(path), path.open("r+b") as output:
            output.write(b"changed!")

    with monkeypatch.context() as patch:
        patch.setattr(loader.staging[0], "copy_async", mutate)
        with pytest.raises(RuntimeError, match="file changed during restore"):
            restore_and_check(loader, path)
    assert loader._pending is None
    assert all(block.content is None for block in loader.pool._free)
    loader.release_blocks(restore_and_check(loader, path))


@pytest.mark.unit
def test_mapping_failure_unloads_before_returning_cached_blocks(loader, tmp_path):
    path = make_weight_file(tmp_path, "weights", 8 * 64, 15)
    loader.release_blocks(restore_and_check(loader, path))

    class FailingManager(Manager):
        def restore_from_vmm_allocation(self, handle, offset, size, stream):
            super().restore_from_vmm_allocation(handle, offset, size, stream)
            return offset < 64

        def unload(self, stream):
            assert not loader.pool._free
            self.ranges.clear()
            return True

    manager = FailingManager(path.read_bytes())
    with pytest.raises(RuntimeError, match="Failed to restore managed-weight range"):
        loader.restore(path, path.stat().st_size, manager, 7, path.stem)
    assert not manager.ranges
    assert loader._pending is None
    assert len(loader.pool._free) == 8
    loader.release_blocks(restore_and_check(loader, path))


@pytest.mark.unit
@pytest.mark.parametrize("sizes", [(208, 80, 480, 16), (480, 80, 208, 16), (128, 128, 512, 128)])
def test_engine_boundaries_reuse_prefix_without_reallocation(loader, tmp_path, sizes):
    paths = [make_weight_file(tmp_path, str(index), size, 31 * index) for index, size in enumerate(sizes)]
    loader.pool.configure(list(sizes))
    largest = max(range(len(sizes)), key=sizes.__getitem__)
    factory = loader.pool._block_factory
    handles = {}
    for cycle in range(3):
        for index, path in enumerate(paths):
            before = copied_bytes(loader)
            blocks = restore_and_check(loader, path)
            ranges = loader.pool.weight_ranges(sizes[index])
            assert len(blocks) == len(ranges)
            for (offset, length), block in zip(ranges, blocks):
                assert block.size == length
                assert block.handle == handles.setdefault(offset, block.handle)
            if cycle and index == largest:
                assert copied_bytes(loader) - before == max(size for i, size in enumerate(sizes) if i != largest)
            loader.release_blocks(blocks)
    assert len(factory.blocks) == len(set(sizes))
    assert factory.peak_bytes == max(sizes)
    assert not any(block.closed for block in factory.blocks)


@pytest.mark.unit
@pytest.mark.parametrize("sizes", [[], [0], [-4], [15], [16, 33]])
def test_engine_boundaries_reject_invalid_sizes(loader, sizes):
    with pytest.raises(ValueError, match="positive and VMM-aligned"):
        loader.pool.configure(sizes)


@pytest.mark.unit
def test_engine_boundaries_cannot_change_after_allocation(loader, tmp_path):
    path = make_weight_file(tmp_path, "weights", 80, 11)
    loader.pool.configure([16, 80])
    loader.release_blocks(restore_and_check(loader, path))
    with pytest.raises(RuntimeError, match="before allocating"):
        loader.pool.configure([16, 128])
    with pytest.raises(ValueError, match="configured engine boundary"):
        loader.pool.weight_ranges(64)


@pytest.mark.unit
@pytest.mark.parametrize("failure", ["allocation", "copy", "mapping"])
def test_engine_boundary_restore_failure_can_retry(loader, tmp_path, monkeypatch, failure):
    path = make_weight_file(tmp_path, "weights", 480, 11)
    loader.pool.configure([16, 80, 208, 480])
    manager = Manager(path.read_bytes())
    factory = loader.pool._block_factory
    allocate = factory.__call__
    copy = loader.staging[0].copy_async
    mapping = manager.restore_from_vmm_allocation
    calls = 0

    def fail_second(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected failure")
        return {"allocation": allocate, "copy": copy, "mapping": mapping}[failure](*args)

    def unload(stream):
        manager.ranges.clear()
        return True

    manager.unload = unload
    with monkeypatch.context() as patch:
        if failure == "allocation":
            patch.setattr(loader.pool, "_block_factory", fail_second)
        elif failure == "copy":
            patch.setattr(loader.staging[0], "copy_async", fail_second)
        else:
            patch.setattr(manager, "restore_from_vmm_allocation", fail_second)
        with pytest.raises(RuntimeError, match="injected failure"):
            loader.restore(path, path.stat().st_size, manager, 7, path.stem)
    assert loader._pending is None
    assert all(block.content is None for block in loader.pool._free)
    loader.release_blocks(restore_and_check(loader, path))
    assert len(factory.blocks) == 4
    assert factory.peak_bytes == 480


@pytest.mark.unit
def test_engine_boundaries_preserve_active_blocks(loader, tmp_path):
    small = make_weight_file(tmp_path, "small", 80, 13)
    large = make_weight_file(tmp_path, "large", 480, 51)
    loader.pool.configure([16, 80, 208, 480])
    loader.release_blocks(restore_and_check(loader, large))
    active = restore_and_check(loader, small)
    with pytest.raises(RuntimeError, match="pool is exhausted"):
        restore_and_check(loader, large)
    assert loader._pending is None
    assert len(loader.pool._free) + len(active) == len(loader.pool._block_factory.blocks)
    assert b"".join(ctypes.string_at(block.handle, block.size) for block in active) == small.read_bytes()
    loader.release_blocks(active)
    loader.release_blocks(restore_and_check(loader, large))


@pytest.mark.unit
@pytest.mark.parametrize("failure", ["sync", "unload"])
def test_scheduler_retains_failed_restore_until_cleanup(loader, tmp_path, monkeypatch, failure):
    path = make_weight_file(tmp_path, "weights", 128, 17)
    plan = tmp_path / "engine.plan"
    plan.touch()
    manager = Manager(path.read_bytes())
    manager.get_size = lambda: 128
    manager.unload = lambda stream: True
    scheduler = weights.ManagedWeightsScheduler("vmm", 7, 0, loader=loader)
    scheduler.register_engine("stage", SimpleNamespace(create_weights_manager=lambda: manager), tmp_path, plan, path)
    scheduler.configure_memory_pool()

    def fail(*args):
        raise RuntimeError("injected cleanup failure")

    with monkeypatch.context() as patch:
        if failure == "sync":
            patch.setattr(loader.staging[0], "copy_async", fail)
            patch.setattr(weights.cudart, "cudaStreamSynchronize", fail)
        else:
            patch.setattr(manager, "restore_from_vmm_allocation", lambda *args: False)
            patch.setattr(manager, "unload", lambda *args: False)
        with pytest.raises(RuntimeError), scheduler.use_weights("stage"):
            pytest.fail("Execution must not start after a failed restore")
        assert scheduler.entries["stage"].state == "failed"
        assert loader._pending is not None and loader._pending.ranges
        assert not loader.pool._free
        with pytest.raises(RuntimeError):
            scheduler.close()
        assert loader._pending is not None
        assert loader.pool._block_factory.allocated_bytes == 128

    scheduler.close()
    assert loader._pending is None
    assert loader.pool._block_factory.allocated_bytes == 0


@pytest.mark.unit
def test_invalid_pool_configuration_does_not_enable_initialization(loader, tmp_path):
    plan = tmp_path / "engine.plan"
    plan.touch()
    manager = SimpleNamespace(get_size=lambda: 130)
    scheduler = weights.ManagedWeightsScheduler("vmm", 7, 0, loader=loader)
    scheduler.register_engine(
        "stage", SimpleNamespace(create_weights_manager=lambda: manager), tmp_path, plan, tmp_path / "weights"
    )
    with pytest.raises(ValueError, match="VMM-aligned"):
        scheduler.configure_memory_pool()
    assert scheduler.weight_budget == 0
    with pytest.raises(RuntimeError, match="not been configured"):
        scheduler.initialize_weights("stage")


@pytest.mark.unit
def test_restore_rejects_a_different_execution_stream(loader, tmp_path):
    path = make_weight_file(tmp_path, "weights", 128, 19)
    with pytest.raises(ValueError, match="same stream"):
        loader.restore(path, 128, Manager(path.read_bytes()), 8, "stage")
    assert loader._pending is None
    assert not loader.pool._block_factory.blocks


@pytest.mark.unit
def test_setup_waits_for_weights_before_context_creation(loader, tmp_path, monkeypatch):
    path = make_weight_file(tmp_path, "weights", 128, 19)
    manager = Manager(path.read_bytes())
    manager.get_size = lambda: 128
    manager.unload = lambda stream: True
    plan = tmp_path / "engine.plan"
    plan.touch()
    scheduler = weights.ManagedWeightsScheduler("vmm", 7, 0, loader=loader)
    scheduler.register_engine("stage", SimpleNamespace(create_weights_manager=lambda: manager), tmp_path, plan, path)
    scheduler.configure_memory_pool()
    monkeypatch.setattr(scheduler, "_backup_is_valid", lambda entry: True)
    sync = Mock(return_value=(weights.cudart.cudaError_t.cudaSuccess,))
    monkeypatch.setattr(weights.cudart, "cudaStreamSynchronize", sync)
    scheduler.initialize_weights("stage")
    sync.assert_called_once_with(7)
    scheduler.close()


@pytest.mark.unit
def test_vmm_pool_rejects_exhaustion_and_reuses_released_block():
    factory = Mock(side_effect=MemoryBlock)
    pool = _VmmBlockPool(device=0, granularity=64, block_factory=factory)
    assert pool.configure([64]) == 64
    first = pool.acquire_block_for_write(64)

    with pytest.raises(RuntimeError, match="pool is exhausted"):
        pool.acquire_block_for_write(64)
    pool.release_blocks([first])
    assert pool.acquire_block_for_write(64) is first

    pool.release_blocks([first])
    factory.assert_called_once_with(64, 0, 64)
    pool.close()
    assert first.closed


@pytest.mark.unit
def test_vmm_pool_allocates_and_reuses_exact_tail_size():
    pool = _VmmBlockPool(device=0, granularity=4, block_factory=MemoryBlock)
    pool.configure([60, 64])
    tail = pool.acquire_block_for_write(size=60)
    assert tail.arguments == (60, 0, 4)

    pool.release_blocks([tail])
    assert pool.acquire_block_for_write(size=60) is tail
    pool.release_blocks([tail])

    remainder = pool.acquire_block_for_write(4, offset=60)
    assert remainder.arguments == (4, 0, 4)
    assert not tail.closed
    pool.release_blocks([remainder])
    pool.close()


@pytest.mark.unit
@pytest.mark.parametrize("fail_close", [False, True])
def test_vmm_pool_rejects_cached_acquisition_after_close(monkeypatch, fail_close):
    pool = _VmmBlockPool(device=0, granularity=64, block_factory=MemoryBlock)
    pool.configure([64, 128])
    blocks = [pool.acquire_block_for_write(64, offset=offset) for offset in (0, 64)]
    identity = ("weights",)
    for offset, block in zip((0, 64), blocks):
        block.content = (identity, offset, 64)
    pool.release_blocks(blocks)

    with monkeypatch.context() as patch:
        if fail_close:
            patch.setattr(blocks[1], "close", Mock(side_effect=RuntimeError("release failed")))
        with pytest.raises(RuntimeError, match="release failed") if fail_close else nullcontext():
            pool.close()
        assert blocks[0].closed
        assert blocks[1].closed == (not fail_close)
        with pytest.raises(RuntimeError, match="pool is closed"):
            pool.acquire_cached_blocks(identity)

    pool.close()
    assert all(block.closed for block in blocks)


@pytest.mark.unit
def test_vmm_pool_close_can_be_retried_after_checked_out_block_returns():
    pool = _VmmBlockPool(device=0, granularity=64, block_factory=MemoryBlock)
    pool.configure([64])
    block = pool.acquire_block_for_write(64)

    with pytest.raises(RuntimeError, match="blocks are in use"):
        pool.close()

    pool.release_blocks([block])
    pool.close()
    assert block.closed


@pytest.mark.unit
def test_loader_borrows_execution_stream(monkeypatch):
    forbidden = Mock(side_effect=AssertionError("The caller owns the execution stream"))
    monkeypatch.setattr(weights.cudart, "cudaStreamCreate", forbidden)
    monkeypatch.setattr(weights.cudart, "cudaStreamDestroy", forbidden)
    monkeypatch.setattr(
        weights.cudart, "cudaStreamSynchronize", Mock(return_value=(weights.cudart.cudaError_t.cudaSuccess,))
    )
    monkeypatch.setattr(
        weights.cuda, "cuMemGetAllocationGranularity", Mock(return_value=(weights.cuda.CUresult.CUDA_SUCCESS, 4))
    )
    instance = weights._VmmWeightLoader(0, 7, pinned_host=True)
    assert instance.stream == 7
    instance.close()
    forbidden.assert_not_called()


@pytest.mark.unit
@pytest.mark.parametrize("fail_close", [False, True])
def test_block_keeps_copy_mapping_until_close(monkeypatch, fail_close):
    success = weights.cuda.CUresult.CUDA_SUCCESS
    calls = {}
    for name, result in {
        "cuMemCreate": (success, 1),
        "cuMemAddressReserve": (success, 4096),
        "cuMemMap": (success,),
        "cuMemSetAccess": (success,),
        "cuMemUnmap": (success,),
        "cuMemAddressFree": (success,),
        "cuMemRelease": (success,),
    }.items():
        calls[name] = Mock(return_value=result)
        monkeypatch.setattr(weights.cuda, name, calls[name])
    block = weights._VmmBlock(128, 0, 4)
    assert block.map_for_write() == block.map_for_write() == 4096
    calls["cuMemMap"].assert_called_once()
    calls["cuMemUnmap"].assert_not_called()
    if fail_close:
        calls["cuMemUnmap"].side_effect = RuntimeError("unmap failed")
        with pytest.raises(RuntimeError, match="unmap failed"):
            block.close()
        assert block.address == 4096 and block.handle == 1
        calls["cuMemRelease"].assert_not_called()
        calls["cuMemUnmap"].side_effect = None
    block.close()
    assert block.address == 0 and block.handle == 0
    calls["cuMemAddressFree"].assert_called_once()
    calls["cuMemRelease"].assert_called_once()
