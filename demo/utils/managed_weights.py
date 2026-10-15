# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Weight storage and weights offloading between pipeline stages for TensorRT-RTX demos.

Host storage:
    _PinnedBuffer: Pinned host memory with asynchronous copies to device memory.
    _PinnedWeightCache: Maps engine weight files to full-file _PinnedBuffer objects,
        optionally populated during setup. Without this cache, loaders read files
        on demand through a memory map (buffer mode) or small pinned buffers (VMM).
    _FileStreamWriter: Writes engine weight backups to disk.

GPU storage and restore:
    _VmmBlock: One physical GPU allocation that can back engine weight memory.
    _VmmBlockPool: Shares blocks between engines, retaining cached weight contents.
    _BufferWeightLoader: Copies file-backed or pinned host weights into TRT-owned memory.
    _VmmWeightLoader: Maps pooled blocks into engine weight memory, copying only
        missing or overwritten weights from host memory.

State records:
    _ManagedEngineWeights: Per-engine weights, cache paths, and allocation state.
    _PendingBufferRestore / _PendingVmmRestore: Per-restore resources and cleanup state.

Pipeline scheduling:
    ManagedWeightsScheduler: Prepares, restores, and unloads weights between stages.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import logging
import os
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import tensorrt_rtx as trt
from cuda.bindings import driver as cuda
from cuda.bindings import runtime as cudart

logger = logging.getLogger("rtx_demo.utils.managed_weights")

MIB = 1024 * 1024
MANIFEST_VERSION = 1


def _cuda_assert(call, operation: str):
    status = call[0]
    if status != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"{operation} failed: {status}")
    return call[1] if len(call) > 1 else None


def _driver_assert(call, operation: str):
    status = call[0]
    if status != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"{operation} failed: {status}")
    return call[1] if len(call) > 1 else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(MIB), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: Path, stat) -> tuple:
    return (str(path.resolve()), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


class _FileStreamWriter(trt.IStreamWriter):
    """Write managed-weight chunks directly to a file."""

    def __init__(self, path: Path):
        trt.IStreamWriter.__init__(self)
        self._file = path.open("wb")

    def write(self, data: bytes) -> int:
        """Write one TensorRT-provided chunk."""
        return self._file.write(data)

    def close(self) -> None:
        """Close the destination file."""
        self._file.close()


class _PinnedBuffer:
    """Pinned host storage with optional copy tracking before overwrite or release."""

    def __init__(self, size: int, *, track_pending_copies: bool = False):
        self.size = size
        self.pointer = _cuda_assert(cudart.cudaMallocHost(size), "cudaMallocHost")
        self._event = None
        self._pending = False
        try:
            if track_pending_copies:
                self._event = _cuda_assert(cudart.cudaEventCreate(), "cudaEventCreate")
        except BaseException:
            self.close()
            raise

    def _check_range(self, size: int, offset: int) -> None:
        if not self.pointer or size < 0 or offset < 0 or offset + size > self.size:
            raise ValueError("Invalid pinned-buffer range")

    def _wait(self) -> None:
        if self._pending:
            _cuda_assert(cudart.cudaEventSynchronize(self._event), "cudaEventSynchronize")
            self._pending = False

    def read(self, source, size: int, offset: int = 0) -> None:
        self._check_range(size, offset)
        self._wait()
        storage = (ctypes.c_ubyte * size).from_address(int(self.pointer) + offset)
        view = memoryview(storage).cast("B")
        try:
            if source.readinto(view) != size:
                raise RuntimeError("Unexpected end of managed-weight file")
        finally:
            view.release()

    def copy_async(self, destination: int, size: int, stream: int, offset: int = 0) -> None:
        self._check_range(size, offset)
        _cuda_assert(
            cudart.cudaMemcpyAsync(
                destination,
                int(self.pointer) + offset,
                size,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                stream,
            ),
            "cudaMemcpyAsync(pinned weights)",
        )
        if self._event is not None:
            _cuda_assert(cudart.cudaEventRecord(self._event, stream), "cudaEventRecord")
            self._pending = True

    def close(self) -> None:
        self._wait()
        if self._event is not None:
            _cuda_assert(cudart.cudaEventDestroy(self._event), "cudaEventDestroy")
            self._event = None
        if self.pointer:
            # Without copy tracking, the loader must synchronize before release.
            _cuda_assert(cudart.cudaFreeHost(self.pointer), "cudaFreeHost")
            self.pointer = 0


class _PinnedWeightCache:
    """Retain complete weight files in pinned host memory for repeated restores."""

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.pinned_weight_buffers = {}

    def prepare(self, path: Path, size: int) -> None:
        if not self.enabled:
            return
        path = path.resolve()
        if path not in self.pinned_weight_buffers:
            with path.open("rb") as source:
                identity = _file_identity(path, os.fstat(source.fileno()))
                if identity[3] != size:
                    raise RuntimeError("Managed-weight file size changed before pinned preparation")
                buffer = _PinnedBuffer(size)
                try:
                    for offset in range(0, size, 64 * MIB):
                        buffer.read(source, min(64 * MIB, size - offset), offset)
                    if _file_identity(path, os.fstat(source.fileno())) != identity:
                        raise RuntimeError("Managed-weight file changed during pinned preparation")
                    if _file_identity(path, path.stat()) != identity:
                        raise RuntimeError("Pinned managed-weight cache changed; repeat pipeline setup")
                except BaseException:
                    buffer.close()
                    raise
            self.pinned_weight_buffers[path] = (identity, buffer)
        self.get(path, size)

    def get(self, path: Path, size: int):
        if not self.enabled:
            return None
        cached = self.pinned_weight_buffers.get(path.resolve())
        if cached is None:
            raise RuntimeError("Pinned managed weights must be prepared during setup")
        identity, buffer = cached
        if size != buffer.size or _file_identity(path, path.stat()) != identity:
            raise RuntimeError("Pinned managed-weight cache changed; repeat pipeline setup")
        return cached

    def close(self) -> None:
        """Free cached buffers after the loader has finished all transfers."""
        for path in list(self.pinned_weight_buffers):
            self.pinned_weight_buffers[path][1].close()
            del self.pinned_weight_buffers[path]


class _VmmBlock:
    """One reusable physical allocation used to back a range of engine weights."""

    def __init__(self, size: int, device: int, granularity: int):
        self.size = size
        self.device = device
        self.granularity = granularity
        self.address = 0
        self.handle = 0
        self._mapped = False
        # File identity, offset, size; copies and inference share the loader's stream.
        self.content = None

        properties = cuda.CUmemAllocationProp()
        properties.type = cuda.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        properties.location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        properties.location.id = device
        properties.requestedHandleTypes = cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_NONE
        self.handle = _driver_assert(cuda.cuMemCreate(size, properties, 0), "cuMemCreate")

    def map_for_write(self) -> int:
        if self.address:
            return int(self.address)
        self.address = _driver_assert(
            cuda.cuMemAddressReserve(self.size, self.granularity, cuda.CUdeviceptr(0), 0),
            "cuMemAddressReserve",
        )
        try:
            _driver_assert(cuda.cuMemMap(self.address, self.size, 0, self.handle, 0), "cuMemMap")
            self._mapped = True
            access = cuda.CUmemAccessDesc()
            access.location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
            access.location.id = self.device
            access.flags = cuda.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
            _driver_assert(cuda.cuMemSetAccess(self.address, self.size, [access], 1), "cuMemSetAccess")
        except BaseException:
            self._unmap_after_write()
            raise
        return int(self.address)

    def _unmap_after_write(self) -> None:
        if self.address:
            if self._mapped:
                _driver_assert(cuda.cuMemUnmap(self.address, self.size), "cuMemUnmap")
                self._mapped = False
            _driver_assert(cuda.cuMemAddressFree(self.address, self.size), "cuMemAddressFree")
            self.address = 0

    def close(self) -> None:
        self._unmap_after_write()
        if self.handle:
            _driver_assert(cuda.cuMemRelease(self.handle), "cuMemRelease")
            self.handle = 0


class _VmmBlockPool:
    """Share physical weight blocks between sequentially executed engines.

    Split at each distinct engine weight size. For sizes 1, 3, and 8 GiB
    (each x represents 1 GiB; brackets delimit physical blocks):

        Pool:          [x][xx][xxxxx]
        1 GiB engine:  [x]
        3 GiB engine:  [x][xx]
        8 GiB engine:  [x][xx][xxxxx]

    Allocate blocks lazily: up to three blocks of 1, 2, and 5 GiB here.
    Unload the active engine before returning its blocks, retaining their
    allocations and contents. Restore the next engine by copying only missing
    or overwritten ranges, then mapping its blocks into its weight address
    range. Mapping itself does not copy the contents.

    After switching from the 8 GiB engine to the 3 GiB engine and back,
    reload [x][xx] but reuse the unchanged [xxxxx] without a host copy.
    """

    def __init__(self, device: int, granularity: int, block_factory=_VmmBlock):
        self.device = device
        self.granularity = granularity
        self._block_factory = block_factory
        self._free = []
        self._closed = False
        self._block_sizes = {}
        self._allocated_blocks = {}

    def configure(self, sizes: list[int]) -> int:
        """Partition the largest engine's weight range at every engine boundary."""
        boundaries = sorted(set(sizes))
        if not boundaries or any(size <= 0 or size % self.granularity for size in boundaries):
            raise ValueError("Engine weight sizes must be positive and VMM-aligned")
        if self._closed or self._allocated_blocks:
            raise RuntimeError("Configure engine boundaries before allocating VMM blocks")
        self._block_sizes = {start: end - start for start, end in zip([0] + boundaries[:-1], boundaries)}
        return boundaries[-1]

    def weight_ranges(self, size: int) -> list[tuple[int, int]]:
        if size not in (offset + length for offset, length in self._block_sizes.items()):
            raise ValueError("Weight size does not match a configured engine boundary")
        return [(offset, length) for offset, length in self._block_sizes.items() if offset < size]

    def acquire_block_for_write(self, size: int, *, offset: int = 0) -> _VmmBlock:
        if self._closed:
            raise RuntimeError("The VMM block pool is closed")
        if self._block_sizes.get(offset) != size:
            raise ValueError("Weight range does not match the shared pool layout")
        block = self._allocated_blocks.get(offset)
        if block is None:
            block = self._block_factory(size, self.device, self.granularity)
            self._allocated_blocks[offset] = block
        elif block in self._free:
            self._free.remove(block)
        else:
            raise RuntimeError("The VMM block pool is exhausted; unload the active engine first")
        block.content = None
        return block

    def acquire_cached_blocks(self, identity: tuple) -> dict[int, _VmmBlock]:
        """Reserve matching blocks before loading missing ranges can overwrite them."""
        if self._closed:
            raise RuntimeError("The VMM block pool is closed")
        cached = {
            block.content[1]: block
            for block in self._free
            if block.content is not None and block.content[0] == identity
        }
        self._free = [block for block in self._free if block not in cached.values()]
        return cached

    def release_blocks(self, blocks: list[_VmmBlock]) -> None:
        """Return blocks without freeing memory or clearing cached contents."""
        if self._closed:
            raise RuntimeError("Cannot return a VMM block to a closed pool")
        self._free.extend(blocks)

    def close(self) -> None:
        if len(self._free) != len(self._allocated_blocks):
            raise RuntimeError("Cannot close the VMM block pool while blocks are in use")
        self._closed = True
        for block in self._free:
            block.close()
        self._free.clear()
        self._allocated_blocks.clear()


@dataclass
class _PendingVmmRestore:
    path: Path
    name: str
    size: int
    ranges: list[tuple[_VmmBlock, int, int]] = field(default_factory=list)
    manager: object | None = None
    execution_stream: int = 0
    engine_mapping_started: bool = False


@dataclass
class _PendingBufferRestore:
    path: Path
    name: str
    storage: np.memmap | _PinnedBuffer | None
    manager: object | None = None
    execution_stream: int = 0
    engine_mapping_started: bool = False


class _BufferWeightLoader:
    """Restore managed weights from file-backed or cached pinned host buffers."""

    def __init__(self, pinned_host: bool = False):
        self._pending = None
        self.pinned_weight_cache = _PinnedWeightCache(pinned_host)

    def prepare_pinned_weights(self, path: Path, size: int) -> None:
        self.pinned_weight_cache.prepare(path, size)

    def restore(self, path: Path, size: int, manager, execution_stream: int, name: str) -> list:
        """Restore weights from a host buffer and wait for the copy."""
        if self._pending is not None:
            raise RuntimeError("Close the loader to clean up the previous failed restore")
        pending = self._prepare_restore(path, size, name)
        self._pending = pending
        try:
            return self._restore(pending, manager, execution_stream)
        finally:
            if pending.storage is None:
                self._pending = None

    def _prepare_restore(self, path: Path, expected_size: int, name: str | None = None) -> _PendingBufferRestore:
        path = Path(path)
        if path.stat().st_size != expected_size:
            raise RuntimeError(f"Managed-weight file size does not match the engine: {path}")
        pinned = self.pinned_weight_cache.get(path, expected_size)
        storage = pinned[1] if pinned is not None else np.memmap(path, mode="r", dtype=np.uint8, shape=(expected_size,))
        return _PendingBufferRestore(path, name or path.stem, storage)

    def _restore(self, pending: _PendingBufferRestore, manager, execution_stream: int) -> list:
        storage = pending.storage
        pointer, size = (
            (storage.pointer, storage.size)
            if isinstance(storage, _PinnedBuffer)
            else (storage.ctypes.data, storage.nbytes)
        )
        try:
            restored = manager.restore_from_buffer(
                int(pointer),
                int(size),
                trt.TensorLocation.HOST,
                execution_stream,
            )
        except BaseException:
            self._close_host_mapping(pending)
            raise
        if not restored:
            self._close_host_mapping(pending)
            raise RuntimeError(f"Failed to restore managed weights for {pending.name} from a host buffer")

        pending.manager = manager
        pending.execution_stream = execution_stream
        pending.engine_mapping_started = True
        # Keep the source mapping alive if synchronization fails.
        _cuda_assert(cudart.cudaStreamSynchronize(execution_stream), "cudaStreamSynchronize(buffer restore)")
        pending.engine_mapping_started = False
        pending.manager = None
        self._close_host_mapping(pending)
        return []

    def _cancel_restore(self, pending: _PendingBufferRestore) -> None:
        if pending.engine_mapping_started:
            if not pending.manager.unload(pending.execution_stream):
                raise RuntimeError(f"Failed to unload {pending.name} while cancelling its buffer restore")
            pending.engine_mapping_started = False
            pending.manager = None
        self._close_host_mapping(pending)

    @staticmethod
    def _close_host_mapping(pending: _PendingBufferRestore) -> None:
        mapping = getattr(pending.storage, "_mmap", None)
        if mapping is not None and not mapping.closed:
            mapping.close()
        pending.storage = None

    @staticmethod
    def release_blocks(blocks: list) -> None:
        if blocks:
            raise RuntimeError("Host-buffer restore does not own VMM blocks")

    def close(self) -> None:
        if self._pending is not None:
            self._cancel_restore(self._pending)
            self._pending = None
        self.pinned_weight_cache.close()


class _VmmWeightLoader:
    """Copy pooled VMM weights on the caller's execution stream using persistent mappings."""

    def __init__(self, device: int, execution_stream: int, staging_mib: int = 8, pinned_host: bool = False):
        if staging_mib <= 0:
            raise ValueError("Managed-weight copy size must be positive")
        properties = cuda.CUmemAllocationProp()
        properties.type = cuda.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        properties.location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        properties.location.id = device
        properties.requestedHandleTypes = cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_NONE
        self.granularity = _driver_assert(
            cuda.cuMemGetAllocationGranularity(
                properties,
                cuda.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM,
            ),
            "cuMemGetAllocationGranularity",
        )
        self.stream = execution_stream
        self.pool = _VmmBlockPool(device, self.granularity)
        self._pending = None
        staging_size = max(self.granularity, staging_mib * MIB)
        self.pinned_weight_cache = _PinnedWeightCache(pinned_host)
        self.copy_chunk_size = staging_size
        self.staging = []
        try:
            if not pinned_host:
                for _ in range(2):
                    self.staging.append(_PinnedBuffer(staging_size, track_pending_copies=True))
        except BaseException:
            for stage in self.staging:
                stage.close()
            raise

    def prepare_pinned_weights(self, path: Path, size: int) -> None:
        """Populate the optional pinned source cache during pipeline setup."""
        self.pinned_weight_cache.prepare(path, size)

    def _prepare_restore(self, path: Path, expected_size: int, name: str | None = None) -> _PendingVmmRestore:
        if path.stat().st_size != expected_size:
            raise RuntimeError(f"Managed-weight file size does not match the engine: {path}")
        if expected_size % self.granularity != 0:
            raise RuntimeError("Managed-weight size is not aligned to CUDA VMM granularity")

        return _PendingVmmRestore(path, name or path.stem, expected_size)

    def restore(self, path: Path, size: int, manager, execution_stream: int, name: str) -> list[_VmmBlock]:
        """Restore engine mappings, copying only missing or overwritten weight ranges."""
        if int(execution_stream) != int(self.stream):
            raise ValueError("VMM weight copies and inference must use the same stream")
        if self._pending is not None:
            raise RuntimeError("Close the loader to clean up the previous failed restore")
        pending = self._prepare_restore(path, size, name)
        self._pending = pending
        try:
            return self._restore(pending, manager, execution_stream)
        finally:
            # A failed copy or unmap may still own blocks before engine mapping starts.
            if not pending.ranges and not pending.engine_mapping_started:
                self._pending = None

    def _load_blocks_async(self, pending: _PendingVmmRestore, expected_size: int) -> None:
        """Queue missing weight copies; host reads and staging-buffer reuse may block."""
        staging_index = 0
        try:
            pinned = self.pinned_weight_cache.get(pending.path, expected_size)
            identity, pinned_buffer = pinned if pinned is not None else (None, None)
            with nullcontext() if pinned_buffer is not None else pending.path.open("rb") as source:
                if pinned_buffer is None:
                    identity = _file_identity(pending.path, os.fstat(source.fileno()))
                if identity[3] != expected_size:
                    raise RuntimeError("Managed-weight file size changed before restore")
                ranges = self.pool.weight_ranges(expected_size)
                cached = self.pool.acquire_cached_blocks(identity)
                pending.ranges.extend((block, offset, block.content[2]) for offset, block in cached.items())
                for range_offset, range_size in ranges:
                    if range_offset in cached:
                        continue
                    block = self.pool.acquire_block_for_write(range_size, offset=range_offset)
                    try:
                        address = block.map_for_write()
                    except BaseException:
                        self.pool.release_blocks([block])
                        raise
                    pending.ranges.append((block, range_offset, range_size))

                    if source is not None:
                        source.seek(range_offset)
                    copied = 0
                    while copied < range_size:
                        chunk_size = min(self.copy_chunk_size, range_size - copied)
                        if pinned_buffer is not None:
                            pinned_buffer.copy_async(address + copied, chunk_size, self.stream, range_offset + copied)
                        else:
                            stage = self.staging[staging_index]
                            stage.read(source, chunk_size)
                            stage.copy_async(address + copied, chunk_size, self.stream)
                            staging_index = (staging_index + 1) % len(self.staging)
                        copied += chunk_size
                    block.content = (identity, range_offset, range_size)
                if pinned_buffer is not None:
                    self.pinned_weight_cache.get(pending.path, expected_size)
                elif _file_identity(pending.path, os.fstat(source.fileno())) != identity:
                    raise RuntimeError("Managed-weight file changed during restore")
                pending.ranges.sort(key=lambda item: item[1])
        except BaseException:
            self._release_pending_blocks(pending)
            raise

    def _restore(self, pending: _PendingVmmRestore, manager, execution_stream: int) -> list[_VmmBlock]:
        self._load_blocks_async(pending, pending.size)
        blocks = []
        pending.manager = manager
        pending.execution_stream = execution_stream
        try:
            # Keep copy mappings alive; later inference is ordered after copies on this stream.
            for block, offset, size in pending.ranges:
                pending.engine_mapping_started = True
                if not manager.restore_from_vmm_allocation(int(block.handle), offset, size, execution_stream):
                    raise RuntimeError(f"Failed to restore managed-weight range [{offset}, {offset + size})")
                blocks.append(block)
        except BaseException as error:
            if pending.engine_mapping_started and not manager.unload(execution_stream):
                raise RuntimeError(
                    f"Failed to unload {pending.name} after a ranged restore failure; VMM blocks remain owned"
                ) from error
            pending.engine_mapping_started = False
            self._release_pending_blocks(pending)
            raise
        pending.ranges.clear()
        pending.engine_mapping_started = False
        pending.manager = None
        return blocks

    def _cancel_restore(self, pending: _PendingVmmRestore) -> None:
        if pending.engine_mapping_started:
            if not pending.manager.unload(pending.execution_stream):
                raise RuntimeError(f"Failed to unload {pending.name} while cancelling its restore")
            pending.engine_mapping_started = False
        self._release_pending_blocks(pending)

    def _release_pending_blocks(self, pending: _PendingVmmRestore) -> None:
        if pending.engine_mapping_started:
            raise RuntimeError(f"Cannot release VMM blocks still mapped to {pending.name}")
        _cuda_assert(cudart.cudaStreamSynchronize(self.stream), "cudaStreamSynchronize(restore)")
        blocks = []
        for block, _, _ in pending.ranges:
            block.content = None
            blocks.append(block)
        self.pool.release_blocks(blocks)
        pending.ranges.clear()

    def release_blocks(self, blocks: list[_VmmBlock]) -> None:
        self.pool.release_blocks(blocks)

    def close(self) -> None:
        if self._pending is not None:
            self._cancel_restore(self._pending)
            self._pending = None
        _cuda_assert(cudart.cudaStreamSynchronize(self.stream), "cudaStreamSynchronize(restore close)")
        for stage in self.staging:
            stage.close()
        self.staging.clear()
        self.pinned_weight_cache.close()
        self.pool.close()


@dataclass
class _ManagedEngineWeights:
    name: str
    engine: object
    manager: object
    model_path: Path
    plan_path: Path
    backup_path: Path
    manifest_path: Path
    plan_sha256: str
    size: int
    refit: Callable | None = None
    source_identity: str | None = None
    state: str = "unloaded"
    blocks: list[_VmmBlock] = field(default_factory=list)
    weights_owned: bool = False


class ManagedWeightsScheduler:
    """Schedule engine weight mappings within a bounded, reusable GPU pool."""

    def __init__(
        self,
        mode: str,
        execution_stream: int,
        device: int,
        staging_mib: int = 8,
        loader=None,
        pinned_host: bool = False,
    ):
        if mode not in {"buffer", "vmm"}:
            raise ValueError(f"Unsupported managed-weight mode: {mode}")
        self.pinned_host = pinned_host
        self.execution_stream = execution_stream
        self.entries = {}
        self.loader = loader or (
            _VmmWeightLoader(device, execution_stream, staging_mib, pinned_host=pinned_host)
            if mode == "vmm"
            else _BufferWeightLoader(pinned_host=pinned_host)
        )
        self.weight_budget = 0

    def register_engine(
        self,
        name: str,
        engine,
        model_path: Path,
        plan_path: Path,
        backup_path: Path,
        manifest_path: Path | None = None,
        refit: Callable | None = None,
        source_identity: str | None = None,
    ) -> None:
        if name in self.entries:
            raise ValueError(f"Managed-weight engine is already registered: {name}")
        manager = engine.create_weights_manager()
        if manager is None:
            raise RuntimeError(f"Engine {name} does not support managed-weight offload")
        size = int(manager.get_size())
        if size <= 0:
            raise RuntimeError(f"Engine {name} reported an invalid managed-weight size: {size}")
        plan_path = Path(plan_path)
        if not plan_path.is_file():
            raise RuntimeError(f"Managed-weight engine plan does not exist: {plan_path}")
        backup_path = Path(backup_path)
        self.entries[name] = _ManagedEngineWeights(
            name=name,
            engine=engine,
            manager=manager,
            model_path=Path(model_path),
            plan_path=plan_path,
            backup_path=backup_path,
            manifest_path=manifest_path or backup_path.with_suffix(backup_path.suffix + ".json"),
            plan_sha256=_sha256(plan_path),
            size=size,
            refit=refit,
            source_identity=source_identity,
        )

    def configure_memory_pool(self) -> None:
        """Size the shared pool from the registered engines."""
        sizes = [entry.size for entry in self.entries.values()]
        if isinstance(self.loader, _VmmWeightLoader):
            self.weight_budget = self.loader.pool.configure(sizes)
        else:
            self.weight_budget = max(sizes, default=0)
        if self.pinned_host:
            logger.info(
                "[WEIGHTS] Setup will pin %.2f GiB of host RAM for the session",
                sum(entry.size for entry in self.entries.values()) / (1024 * MIB),
            )

    def _make_manifest(self, entry: _ManagedEngineWeights) -> dict:
        return {
            "format_version": MANIFEST_VERSION,
            "tensorrt_version": trt.__version__,
            "engine_name": entry.name,
            "plan_sha256": entry.plan_sha256,
            "managed_weights_size": entry.size,
            "cuda_allocation_granularity": getattr(self.loader, "granularity", 0),
            "source_model": str(entry.model_path.resolve()),
            "source_identity": entry.source_identity,
        }

    def _backup_is_valid(self, entry: _ManagedEngineWeights) -> bool:
        if not entry.backup_path.is_file() or not entry.manifest_path.is_file():
            return False
        try:
            manifest = json.loads(entry.manifest_path.read_text(encoding="utf-8"))
            expected = self._make_manifest(entry)
            identity_fields = (
                "format_version",
                "tensorrt_version",
                "engine_name",
                "plan_sha256",
                "managed_weights_size",
                "source_model",
                "source_identity",
            )
            valid = all(manifest.get(field) == expected[field] for field in identity_fields)
            valid = valid and entry.backup_path.stat().st_size == entry.size
        except (OSError, ValueError, TypeError):
            valid = False
        if not valid:
            logger.warning("Ignoring stale managed-weight cache for %s", entry.name)
        return valid

    def initialize_weights(self, name: str) -> None:
        """Prepare weights and backups before creating the execution context."""
        entry = self.entries[name]
        if self.weight_budget <= 0:
            raise RuntimeError("Managed-weight memory plan has not been configured")
        if self._backup_is_valid(entry):
            if self.pinned_host:
                self.loader.prepare_pinned_weights(entry.backup_path, entry.size)
            self._restore_weights(entry)
            if isinstance(self.loader, _VmmWeightLoader):
                # Context creation does not take the execution stream.
                _cuda_assert(cudart.cudaStreamSynchronize(self.execution_stream), "cudaStreamSynchronize(weight setup)")
            return

        try:
            if entry.refit is None:
                raise RuntimeError(f"No checkpoint refitter was registered for {name}")
            group_count = entry.refit(entry.engine, self.execution_stream)
            entry.state = "loaded"
            entry.weights_owned = True
            logger.info("[WEIGHTS] Refitted %s (%d groups)", name, group_count)
            self._save_backup(entry)
            if self.pinned_host:
                self.loader.prepare_pinned_weights(entry.backup_path, entry.size)
        except BaseException:
            entry.state = "failed"
            raise

    def _save_backup(self, entry: _ManagedEngineWeights) -> None:
        entry.backup_path.parent.mkdir(parents=True, exist_ok=True)
        entry.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = entry.backup_path.with_suffix(entry.backup_path.suffix + ".tmp")
        temporary_manifest = entry.manifest_path.with_suffix(entry.manifest_path.suffix + ".tmp")
        try:
            writer = _FileStreamWriter(temporary_path)
            try:
                if not entry.manager.save_to_stream(writer, self.execution_stream):
                    raise RuntimeError(f"Failed to save managed weights for {entry.name}")
            finally:
                writer.close()
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise
        if temporary_path.stat().st_size != entry.size:
            temporary_path.unlink(missing_ok=True)
            raise RuntimeError(f"Managed-weight backup for {entry.name} has an unexpected size")
        try:
            temporary_manifest.write_text(
                json.dumps(self._make_manifest(entry), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary_path, entry.backup_path)
            os.replace(temporary_manifest, entry.manifest_path)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            temporary_manifest.unlink(missing_ok=True)
            raise

    def unload_weights(self, name: str) -> None:
        entry = self.entries[name]
        if entry.state == "unloaded":
            return
        if entry.state not in {"loaded", "running"} and not (entry.state == "failed" and entry.weights_owned):
            raise RuntimeError(f"Cannot unload {name} in state {entry.state}")
        if not entry.manager.unload(self.execution_stream):
            entry.state = "failed"
            raise RuntimeError(f"Failed to unload managed weights for {name}")
        self.loader.release_blocks(entry.blocks)
        entry.blocks = []
        entry.weights_owned = False
        entry.state = "unloaded"

    def _restore_weights(self, entry: _ManagedEngineWeights) -> None:
        if entry.state == "loaded":
            return
        if entry.state != "unloaded":
            raise RuntimeError(f"Invalid restore state for {entry.name}: {entry.state}")
        try:
            entry.blocks = self.loader.restore(
                entry.backup_path, entry.size, entry.manager, self.execution_stream, entry.name
            )
        except BaseException:
            entry.state = "failed"
            raise
        entry.state = "loaded"
        entry.weights_owned = True

    @contextmanager
    def use_weights(self, name: str):
        entry = self.entries[name]
        self._restore_weights(entry)
        entry.state = "running"
        try:
            yield
        finally:
            if entry.state == "running":
                entry.state = "loaded"
            self.unload_weights(name)

    def close(self) -> None:
        for entry in self.entries.values():
            if entry.state in {"loaded", "running"} or entry.weights_owned:
                self.unload_weights(entry.name)
        self.loader.close()
        self.entries.clear()
