# TensorRT-RTX Demos

A collection of demos showcasing key [TensorRT-RTX](https://developer.nvidia.com/tensorrt-rtx) features through model pipelines.

| Demo                                       | Workflow                                                                                                    |
| ------------------------------------------ | ----------------------------------------------------------------------------------------------------------- |
| [Flux.1 [dev]](./flux1.dev/README.md)      | ONNX models built into TensorRT-RTX engines                                                                 |
| [FLUX.2 [klein] 4B FP8](./flux2/README.md) | FP8 PyTorch transformer compiled with Torch-TensorRT-RTX (`torch.export` + `torch_tensorrt.dynamo.compile`) |

Each demo has its own `requirements.txt` and README covering installation, parameters, and example commands. All commands are run from the repository root:

```bash
git clone https://github.com/NVIDIA/TensorRT-RTX.git
cd TensorRT-RTX

# Install the dependencies for the FLUX.2 demo
python -m pip install -r demo/flux2/requirements.txt
```

## Running Tests

To configure the test environment and run demo tests, refer to the [test README](./tests/README.md).
