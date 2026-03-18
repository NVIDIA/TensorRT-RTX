# TensorRT-RTX Demos Utility Tests

This directory contains utility tests for the TensorRT-RTX Demos.

## Quick Start

We recommend using Python versions between 3.9 and 3.12 inclusive due to supported versions for required dependencies.

1. **Clone and install**

```bash
git clone https://github.com/NVIDIA/TensorRT-RTX.git
cd TensorRT-RTX

# Install TensorRT-RTX
python -m pip install tensorrt-rtx

# Install demo dependencies
python -m pip install -r demo/flux1.dev/requirements.txt

# Install test dependencies
python -m pip install -r demo/tests/requirements-test.txt
```

2. **Run tests**

The tests are located in the `demo/tests` directory.

```bash
python -m pytest demo/tests -v
```
