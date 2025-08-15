# TensorRT for RTX API Usage Sample

This sample demonstrates how to use TensorRT for RTX APIs to fine-tune engine
compilation and inference. First please refer to the [Hello World](../helloWorld)
sample that goes over the basic concepts. In addition, this sample covers

- Creating a TensorRT-RTX builder and network definition with dynamic shapes and setting AoT compilation targets using the `setComputeCapability` and associated API.
- Efficiently checking if an engine file is expected to work for the current platform/environment using the Engine Compatibility API.
- Configuring and serializing a runtime cache via `setRuntimeCache` and associated API to store JIT compiled kernels.
- Setting, querying and running inference with dynamic shape information via various dynamic shape APIs.
- Building weightless engines, and subsequently refitting weights on the deployed machines using the refit APIs.
- Running inference for multiple input shapes with the compiled engine.

## Building the Sample

### Prerequisites

- CMake 3.10 or later
- Python 3.9 or later
- CUDA Toolkit
- An installation of TensorRT for RTX

### Build Instructions

On Windows, add the TensorRT for RTX `lib` directory to your `PATH` environment variable:

```powershell
$Env:PATH += ";$Env:PATH_TO_TRT_RTX\lib"
```

On Linux, add the TensorRT for RTX `lib` directory to your `LD_LIBRARY_PATH` environment variable:

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${PATH_TO_TRT_RTX}/lib
```

#### Build for C++

1. Run CMake from the current or the `cpp` directory, pointing it to your TensorRT for RTX installation, to create artifacts in the `build` directory

   ```bash
   cmake -B build -S . -DTRTRTX_INSTALL_DIR=/path/to/tensorrt-rtx
   ```

2. Build the sample:

   ```bash
   cmake --build build
   ```

#### Build for Python

1. Install the `tensorrt_rtx` wheel from your TensorRT for RTX directory:

   ```bash
   python -m pip install /path/to/tensorrt-rtx/python/tensorrt_rtx-${version}-cp${py3-ver}-none-${os-ver}_x86_64.whl
   ```

2. Install `numpy` and `cuda-python` from the `python/requirements.txt` file:

   ```bash
   python -m pip install -r python/requirements.txt
   ```

## Running the Sample

After building, you can run the sample with:

```bash
./apiUsage
```

from the build directory.

For the Python sample, run:

```bash
python api_usage.py
```

The sample will:

1. Create and compile a simple neural network with dynamic shapes.
2. Build a weightless engine on the current device and then refuel its weights.
3. Run inference with different batch sizes and input values.
4. Display the results.

## Code Overview

The sample demonstrates several key concepts related to TensorRT for RTX APIs:

- Network creation and configuration for dynamically-shaped input tensors.
- Selecting deployment targets at AOT.
- Configuring a weightless engine and refueling weights during deployment.
- Using runtime cache to store JIT-compiled kernels.
- Inference execution with changing dynamic shapes.

For detailed comments explaining each step, please refer to the [apiUsage.cpp](cpp/apiUsage.cpp) and [api_usage.py](python/api_usage.py) source files.
