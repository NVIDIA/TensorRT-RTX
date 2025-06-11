# TensorRT for RTX

TensorRT for RTX builds on the proven performance of the NVIDIA TensorRT inference library, and simplifies the deployment of AI models on NVIDIA RTX GPUs across desktops, laptops, and workstations.

TensorRT for RTX is a drop-in replacement for NVIDIA TensorRT in applications targeting NVIDIA RTX GPUs from Turing through Blackwell generations. It introduces a Just-In-Time (JIT) optimizer in the runtime that compiles improved inference engines directly on the end-user's RTX-accelerated PC in under 30 seconds. This eliminates the need for lengthy pre-compilation steps and enables rapid engine generation, improved application portability, and cutting-edge inference performance. To support integration into lightweight applications and deployment in memory-constrained environments, TensorRT for RTX is compact under 200 MB. TensorRT for RTX makes real-time, responsive AI applications for image processing, speech synthesis, and generative AI practical and performant on consumer-grade devices.

For detailed information on TensorRT-RTX features, software enhancements, and release updates, see the [official developer documentation](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/index.html).
To get the latest TensorRT-RTX SDK, visit the [developer download page](http://developer.nvidia.com/tensorrt-rtx) and follow the [installation guide](http://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/installing-tensorrt-rtx/installing.html).

This repository includes open source components that accompany the TensorRT-RTX SDK. If you'd like to contribute, please review our [contribution guide](CONTRIBUTING.md).

# Quickstart Examples

- [Samples](samples/README.md) that illustrate key TensorRT-RTX capabilities and API usage in C++ and Python.
- [Demos](demo/README.md) that highlight practical deployment considerations and reference implementations of popular models.
