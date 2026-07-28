# TensorRT for RTX Memory Management Sample

This sample demonstrates two ways to control the device memory used by TensorRT for
RTX inference, both built around a single custom GPU allocator. It builds a small
dynamic-shape network and runs inference under each strategy using only TensorRT for
RTX and CUDA APIs.

## Custom GPU allocator

The sample implements a synchronous GPU allocator backed by `cudaMalloc` and
`cudaFree` by deriving from `IGpuAsyncAllocator` (C++) / `trt.IGpuAsyncAllocator`
(Python). The allocator is registered on the builder before building the engine and
on the runtime before deserializing the engine, so every allocation TensorRT for RTX
makes is serviced by it.

The custom allocator must outlive every TensorRT for RTX object that uses it.

## Two memory-ownership strategies

The execution context needs a block of device memory for internal activation tensors
during inference. The sample shows the same allocator serving both ownership models:

1. **TensorRT for RTX owns the memory (`kSTATIC`, the default).** TensorRT for RTX
   sizes and manages the activation memory for the worst case across all optimization
   profiles. `ICudaEngine::getDeviceMemorySizeV2()` reports that size. Because the
   custom allocator is registered on the runtime, the reservation is serviced by it.

2. **The application owns the memory (`kUSER_MANAGED`).** TensorRT for RTX does not
   allocate the activation memory. For each set of input shapes the application selects
   the optimization profile, sets the input shapes, and calls
   `IExecutionContext::updateDeviceMemorySizeForShapes()` to compute the exact size
   required. It then allocates that buffer itself — here, through the same custom
   allocator — and provides it with `IExecutionContext::setDeviceMemoryV2()`. For
   smaller input shapes this can reserve less memory than the profile maximum reported
   by `getDeviceMemorySizeV2()`, and the saving grows as the gap between the current
   and maximum shapes widens.

   The optimization profile and all input shapes must be set before calling
   `updateDeviceMemorySizeForShapes()`; otherwise it returns 0.

## Build and Run

### Prerequisites

- CMake 3.17 or later
- Python 3.9 or later
- CUDA Toolkit
- An installation of TensorRT for RTX

### Build and run the C++ sample

From this directory:

```powershell
# Windows
$Env:PATH_TO_TRT_RTX = "C:\path\to\TensorRT-RTX"
$Env:PATH = "$Env:PATH_TO_TRT_RTX\bin;$Env:PATH"
cmake -B build -S . -DTRTRTX_INSTALL_DIR="$Env:PATH_TO_TRT_RTX"
cmake --build build --config Release
.\build\cpp\Release\memoryManagement.exe
```

```bash
# Linux
export PATH_TO_TRT_RTX=/path/to/TensorRT-RTX
export LD_LIBRARY_PATH="${PATH_TO_TRT_RTX}/lib:${LD_LIBRARY_PATH}"
cmake -B build -S . -DTRTRTX_INSTALL_DIR="${PATH_TO_TRT_RTX}"
cmake --build build
./build/cpp/memoryManagement
```

### Run the Python sample

1. Install TensorRT for RTX:

   ```bash
   python -m pip install tensorrt-rtx
   ```

2. Install `numpy` and `cuda-python` from `python/requirements.txt`:

   ```bash
   python -m pip install -r python/requirements.txt
   ```

3. Run the sample:

   ```bash
   python python/memory_management.py
   ```

The sample will:

1. Build a small dynamic-shape network and register a custom GPU allocator.
2. Run inference with TensorRT for RTX managing the execution-context device memory (`kSTATIC`).
3. Run inference with the application managing the device memory (`kUSER_MANAGED`), sized per input shape.
4. Report the device memory required for each shape and the custom allocator's activity.

## Code Overview

The sample demonstrates several key concepts related to TensorRT for RTX memory management:

- Implementing and registering a custom GPU allocator (`IGpuAsyncAllocator`) on the builder and runtime.
- Letting TensorRT for RTX size and own the execution-context device memory (`kSTATIC`).
- Application-managed device memory, sized per input shape with `updateDeviceMemorySizeForShapes()` and provided via `setDeviceMemoryV2()`.
- Inference execution with changing dynamic shapes.

For detailed comments explaining each step, please refer to the [memoryManagement.cpp](cpp/memoryManagement.cpp) and [memory_management.py](python/memory_management.py) source files.
