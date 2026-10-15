# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-backed weight storage and copy helpers for loader tests."""

import ctypes
import os
from contextlib import contextmanager


@contextmanager
def update_weight_file(path):
    """Make a test write visible even on filesystems with coarse timestamps."""
    previous = path.stat()
    yield
    os.utime(path, ns=(previous.st_atime_ns, previous.st_mtime_ns + 2_000_000_000))


class MemoryBlock:
    def __init__(self, size, device, granularity):
        self.size = size
        self.arguments = (size, device, granularity)
        self.storage = ctypes.create_string_buffer(size)
        self.handle = ctypes.addressof(self.storage)
        self.content = None
        self.closed = False

    def map_for_write(self):
        return self.handle

    def close(self):
        self.closed = True


class BlockFactory:
    """Track live test allocations independently of the pool's bookkeeping."""

    def __init__(self):
        self.blocks = []
        self.peak_bytes = 0

    @property
    def allocated_bytes(self):
        return sum(block.size for block in self.blocks if not block.closed)

    def __call__(self, *args):
        block = MemoryBlock(*args)
        self.blocks.append(block)
        self.peak_bytes = max(self.peak_bytes, self.allocated_bytes)
        return block


class StagingBuffer:
    size = 16

    def __init__(self):
        self.copied_bytes = 0

    def read(self, source, size):
        self.data = source.read(size)
        assert len(self.data) == size

    def copy_async(self, address, size, stream):
        ctypes.memmove(address, self.data, size)
        self.copied_bytes += size

    def close(self):
        pass


class Manager:
    def __init__(self, expected):
        self.expected = expected
        self.ranges = {}

    def restore_from_vmm_allocation(self, handle, offset, size, stream):
        assert offset not in self.ranges
        self.ranges[offset] = ctypes.string_at(handle, size)
        return True

    def restore_from_buffer(self, pointer, size, location, stream):
        self.ranges[0] = ctypes.string_at(pointer, size)
        return True

    def check(self):
        assert b"".join(self.ranges[offset] for offset in sorted(self.ranges)) == self.expected


def copied_bytes(loader):
    return sum(stage.copied_bytes for stage in loader.staging)


def restore_and_check(loader, path):
    manager = Manager(path.read_bytes())
    blocks = loader.restore(path, len(manager.expected), manager, 7, path.stem)
    manager.check()
    return blocks


def make_weight_file(tmp_path, name, size, value):
    path = tmp_path / name
    path.write_bytes(bytes((value + i) % 256 for i in range(size)))
    return path
