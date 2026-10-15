# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check shared pinned storage and staging synchronization without a CUDA device."""

import ctypes
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from utils import managed_weights as weights


@pytest.fixture
def cuda_calls(monkeypatch):
    success = weights.cudart.cudaError_t.cudaSuccess
    allocations = {}
    pending = []

    def allocate(size):
        storage = ctypes.create_string_buffer(size)
        pointer = ctypes.addressof(storage)
        allocations[pointer] = storage
        return success, pointer

    def free(pointer):
        assert not pending
        del allocations[int(pointer)]
        return (success,)

    def copy(destination, source, size, kind, stream):
        pending.append((destination, source, size))
        return (success,)

    def synchronize(*args):
        for destination, source, size in pending:
            ctypes.memmove(destination, source, size)
        pending.clear()
        return (success,)

    calls = SimpleNamespace()
    functions = {
        "cudaMallocHost": allocate,
        "cudaFreeHost": free,
        "cudaMemcpyAsync": copy,
        "cudaEventCreate": lambda: (success, 1),
        "cudaEventRecord": lambda *args: (success,),
        "cudaEventDestroy": lambda *args: (success,),
        "cudaEventSynchronize": synchronize,
        "cudaStreamSynchronize": synchronize,
    }
    for name, function in functions.items():
        mock = Mock(side_effect=function)
        monkeypatch.setattr(weights.cudart, name, mock)
        setattr(calls, name, mock)
    yield calls
    assert not allocations
    assert not pending


@pytest.mark.unit
def test_staging_waits_before_refill_and_close(cuda_calls):
    buffer = weights._PinnedBuffer(4, track_pending_copies=True)
    first, second = ctypes.create_string_buffer(4), ctypes.create_string_buffer(4)
    buffer.read(BytesIO(b"abcd"), 4)
    buffer.copy_async(ctypes.addressof(first), 4, 7)
    buffer.read(BytesIO(b"efgh"), 4)
    assert first.raw == b"abcd"
    cuda_calls.cudaEventSynchronize.assert_called_once_with(1)
    buffer.copy_async(ctypes.addressof(second), 4, 7)
    buffer.close()
    assert second.raw == b"efgh"
    assert cuda_calls.cudaEventSynchronize.call_count == 2
    buffer.close()
    cuda_calls.cudaEventDestroy.assert_called_once_with(1)
    cuda_calls.cudaFreeHost.assert_called_once()


@pytest.mark.unit
def test_retained_buffer_copies_ranges_without_events(cuda_calls):
    buffer = weights._PinnedBuffer(8)
    buffer.read(BytesIO(b"abcd"), 4)
    buffer.read(BytesIO(b"efgh"), 4, offset=4)
    first, second = ctypes.create_string_buffer(4), ctypes.create_string_buffer(4)
    buffer.copy_async(ctypes.addressof(first), 4, 7, offset=4)
    buffer.copy_async(ctypes.addressof(second), 4, 7)
    cuda_calls.cudaStreamSynchronize(7)
    assert first.raw == b"efgh"
    assert second.raw == b"abcd"
    buffer.close()
    buffer.close()
    cuda_calls.cudaEventCreate.assert_not_called()
    cuda_calls.cudaEventRecord.assert_not_called()
    cuda_calls.cudaFreeHost.assert_called_once()


@pytest.mark.unit
def test_event_creation_failure_frees_host_memory(cuda_calls):
    cuda_calls.cudaEventCreate.side_effect = None
    cuda_calls.cudaEventCreate.return_value = (weights.cudart.cudaError_t.cudaErrorMemoryAllocation,)
    with pytest.raises(RuntimeError, match="cudaEventCreate"):
        weights._PinnedBuffer(8, track_pending_copies=True)
    cuda_calls.cudaFreeHost.assert_called_once()


@pytest.mark.unit
def test_short_read_can_be_cleaned_up(cuda_calls):
    buffer = weights._PinnedBuffer(8)
    with pytest.raises(RuntimeError, match="Unexpected end"):
        buffer.read(BytesIO(b"abc"), 8)
    buffer.close()


@pytest.mark.unit
@pytest.mark.parametrize("size,offset", [(9, 0), (4, 5), (4, -1), (-1, 0)])
def test_invalid_ranges_are_rejected_before_copy(cuda_calls, size, offset):
    buffer = weights._PinnedBuffer(8)
    with pytest.raises(ValueError, match="Invalid pinned-buffer range"):
        buffer.read(BytesIO(b"abcdefgh"), size, offset)
    with pytest.raises(ValueError, match="Invalid pinned-buffer range"):
        buffer.copy_async(0, size, 7, offset)
    cuda_calls.cudaMemcpyAsync.assert_not_called()
    buffer.close()
