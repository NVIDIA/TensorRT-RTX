# FLUX.2 [klein] 4B FP8 with Torch-TensorRT-RTX

This demo generates images with [FLUX.2 [klein] 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) using the [pre-quantized FP8 transformer](https://huggingface.co/black-forest-labs/FLUX.2-klein-4b-fp8) accelerated by [Torch-TensorRT-RTX](https://pypi.org/project/torch-tensorrt-rtx/). It compiles the PyTorch transformer directly in two explicit steps:

1. `torch.export.export` captures the transformer as an `ExportedProgram`. [ModelOpt](https://github.com/NVIDIA/TensorRT-Model-Optimizer) emits its FP8 quantizers as TensorRT quantize and dequantize operations.
2. `torch_tensorrt.dynamo.compile` converts the exported graph into a single TensorRT-RTX engine that runs the quantized layers as FP8 matrix multiplications.

The compiled transformer is swapped into an otherwise unmodified Diffusers `Flux2KleinPipeline`, so the Qwen3 text encoder, scheduler, and VAE continue to run in PyTorch. FLUX.2 [klein] is step-distilled and generates images in 4 denoising steps.

> **GPU Compatibility**: FP8 requires an Ada or Blackwell GPU. This demo is verified on an NVIDIA GeForce RTX 5090.

## Quick Start

Torch-TensorRT-RTX requires Python 3.10 or later.

```bash
git clone https://github.com/NVIDIA/TensorRT-RTX.git
cd TensorRT-RTX

# Installs PyTorch, Torch-TensorRT-RTX, the matching TensorRT-RTX runtime, and ModelOpt
python -m pip install -r demo/flux2/requirements.txt

# Standalone Python script
python demo/flux2/flux2_demo.py -h

# Interactive Jupyter notebook
jupyter notebook demo/flux2/flux2_demo.ipynb
```

The first run downloads the FP8 transformer (about 4 GB) and the text encoder, tokenizer, scheduler, and VAE from the base model (about 8 GB) into the cache directory. The full-precision base transformer is not downloaded. Generated images are saved as `flux2_<seed>_<index>.png` in the output directory.

> **Tip**: The [Jupyter notebook](./flux2_demo.ipynb) walks through each feature below with timings, memory usage, and generated images.

## Reference Performance

The checked-in notebook produced the following results on an NVIDIA GeForce RTX 5090 (32 GiB) at batch size 1,
1024x1024 resolution, and 4 denoising steps:

| Configuration                   | End-to-end image generation | GPU memory after generation |
| ------------------------------- | --------------------------: | --------------------------: |
| Full transformer weights on GPU |                      0.61 s |                   13.81 GiB |
| 30% weight-streaming budget     |                      0.61 s |                   11.55 GiB |

The 30% budget kept 1.14 GiB of the transformer's 3.79 GiB of streamable weights on the GPU. Generation time
includes text encoding, denoising, VAE decoding, and conversion to a PIL image, but excludes saving the image to
disk. GPU memory is device-wide usage after generation and after releasing unused PyTorch cache blocks; it is not
peak memory and can include allocations from other processes.

The same notebook recorded these compilation times:

| Phase                                                      | Cold runtime cache | Warm runtime cache |
| ---------------------------------------------------------- | -----------------: | -----------------: |
| `torch.export`                                             |             2.18 s |             2.32 s |
| `torch_tensorrt.dynamo.compile`                            |            34.53 s |            34.82 s |
| TensorRT-RTX JIT compilation + first transformer execution |             4.04 s |             0.30 s |

These are single-run reference measurements rather than statistically sampled benchmark results. Performance and
memory usage vary with GPU, software versions, system load, prompt, and image settings.

## Parameters

### Image Generation Parameters

```bash
--prompt "Your text prompt"              # Text prompt for generation
--height 1024                            # Image height, a multiple of 16 (default: 1024)
--width 1024                             # Image width, a multiple of 16 (default: 1024)
--batch-size 1                           # Batch size (default: 1)
--seed 0                                 # Random seed (default: 0)
--num-inference-steps 4                  # Denoising steps (default: 4)
```

### Engine & Performance Options

```bash
--enable-engine-cache                    # Cache built engines and reuse them on later runs
--enable-runtime-cache                   # Cache JIT-compiled kernels and reuse them on later runs
--weight-streaming-budget 0.5            # Keep only this fraction of transformer weights in GPU memory
--verbose                                # Enable verbose logging
```

### Other Options

```bash
--cache-dir ./demo_cache                 # Cache directory (default: ./demo_cache)
--output-dir .                           # Directory for generated images (default: .)
--hf-token YOUR_HF_TOKEN                 # Hugging Face token (optional)
```

## Example Commands

**Default Parameters Image Generation:**

```bash
python demo/flux2/flux2_demo.py
```

**Custom Prompt and Resolution:**

```bash
python demo/flux2/flux2_demo.py --height 768 --width 1344 --prompt "A detailed cityscape at golden hour"
```

**Faster Startup with Runtime and Engine Caching:**

```bash
python demo/flux2/flux2_demo.py --enable-runtime-cache --enable-engine-cache --prompt "A cat meanders down a dimly lit alleyway in a large city."
```

**Low VRAM with Weight Streaming:**

```bash
python demo/flux2/flux2_demo.py --weight-streaming-budget 0.3 --prompt "A serene forest scene"
```

## How It Works

### Loading the FP8 Checkpoint

The FP8 checkpoint stores each quantized linear layer as a float8 weight with a per-tensor weight scale and a static input (activation) scale, using the Black Forest Labs parameter layout. The demo:

1. Renames the parameters to the Diffusers `Flux2Transformer2DModel` layout and dequantizes the FP8 weights to BF16.
2. Inserts ModelOpt FP8 quantizers with `modelopt.torch.quantization.quantize`.
3. Programs each quantizer's range from the checkpoint scales rather than calibrating, so export re-quantizes the weights to exactly the checkpoint's FP8 values.

### Fixed-Shape Compilation

The transformer engine is specialized to one input shape, determined by `--height`, `--width`, and `--batch-size`. To generate at a different shape, compile a new engine. The demo records the exact inputs the pipeline passes to the transformer during a one-step warm-up run and uses them to export and compile the transformer.

`CompiledFlux2Transformer` adapts the compiled module to the call signature the Diffusers pipeline expects. It supports the standard text-to-image path only; image conditioning and attention overrides are rejected.

### Reported Timings

| Phase                           | Description                                                                                                                                                                  |
| ------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `torch.export`                  | Captures the transformer graph.                                                                                                                                              |
| `torch_tensorrt.dynamo.compile` | Converts the graph and builds the TensorRT-RTX engine.                                                                                                                       |
| TensorRT-RTX JIT compilation    | The first execution of the engine, where TensorRT-RTX compiles kernels specialized for the installed GPU. With a warm runtime cache, most kernels are loaded from the cache. |
| Image generation                | End-to-end pipeline time, including text encoding, all denoising steps, and VAE decoding.                                                                                    |

### Runtime Caching

`--enable-runtime-cache` stores the kernels TensorRT-RTX JIT-compiles for your GPU in `<cache-dir>/runtime.cache`. The cache is saved when the engine is released and loaded the next time an engine runs, which shortens JIT compilation on later runs. Without the flag, the demo runs without a runtime cache; note that Torch-TensorRT-RTX otherwise attaches its own runtime cache in the system temp directory by default.

### Engine Caching

`--enable-engine-cache` stores built engines under `<cache-dir>/torchtrt_engines/flux2/fp8/`. Later runs with the same shape still load the checkpoint and export the transformer, but reuse the cached engine rather than building a new one. Cached engines must be refittable so that Torch-TensorRT can load the current weights and FP8 scales into them, so the engine cache disables the immutable-weights optimization.

### Weight Streaming

`--weight-streaming-budget` accepts a fraction in `[0, 1]`, builds a weight-streaming engine, and keeps only the requested fraction of the streamable transformer weights resident in GPU memory; the rest are streamed from host memory during execution. A budget of `0` streams the most weights (maximal offloading); `1` keeps all streamable weights on the GPU. Omitting the option disables weight streaming. Lower budgets reduce VRAM usage at the cost of slower denoising.

## Python API

Choose the image dimensions and batch size when compiling. Every `infer()` call uses that compiled shape.

```python
from demo.flux2.flux2_pipeline import Flux2FP8TorchTensorRTPipeline

pipeline = Flux2FP8TorchTensorRTPipeline(
    cache_dir="./demo_cache", enable_runtime_cache=True, enable_engine_cache=True
)
pipeline.load()
pipeline.compile_transformer(batch_size=1, height=768, width=1344)

# Any prompt can be used with the compiled shape
images, seconds = pipeline.infer(prompt="A lighthouse on a rocky coast", seed=42)
images, seconds = pipeline.infer(prompt="A bowl of ramen, studio lighting", seed=7)
```
