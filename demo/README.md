# TensorRT-RTX Demos

A collection of demos showcasing key [TensorRT-RTX](https://developer.nvidia.com/tensorrt-rtx) features through model pipelines.

## Quick Start

1. **Clone and install**

   We recommend using Python versions between 3.9 and 3.12 inclusive due to supported versions for required dependencies.

   ```bash
   git clone https://github.com/NVIDIA/TensorRT-RTX.git
   cd TensorRT-RTX

   # Install TensorRT-RTX from the wheels located in the downloaded tarfile
   # Visit https://developer.nvidia.com/tensorrt-rtx to download
   # Example below is for Python 3.12 on Linux (customize with your Python version + OS)
   python -m pip install YOUR_TENSORRT_RTX_DOWNLOAD_DIR/python/tensorrt_rtx-1.0.0.20-cp312-none-linux_x86_64.whl

   # Install demo dependencies (example: Flux 1.dev)
   python -m pip install -r demo/flux1.dev/requirements.txt
   ```

2. **Run demo**

   ```bash
   # Standalone Python script
   python demo/flux1.dev/flux_demo.py -h

   # Interactive Jupyter notebook
   jupyter notebook demo/flux1.dev/flux_demo.ipynb
   ```

## Python Script Usage Examples

The standalone script provides extensive configuration options for various use cases. For detailed walkthroughs, interactive exploration, and comprehensive documentation, see the [Flux.1 \[dev\] Demo Notebook](./flux1.dev/flux_demo.ipynb) which offers in-depth coverage of TensorRT-RTX features.

> **GPU Compatibility**: This demo is verified on Ada and Blackwell GPUs. See [Transformer Precision Options](#transformer-precision-options) for more compatibility details.

### Required Parameters

To download model checkpoints for the FLUX.1 \[dev\] pipeline, obtain a `read` access token to the model repository on HuggingFace Hub. See [instructions](https://huggingface.co/docs/hub/security-tokens).

```bash
--hf-token YOUR_HF_TOKEN                # Hugging Face token with read access to the Flux.1 [dev] model
```

### Image Generation Parameters

```bash
--prompt "Your text prompt"              # Text prompt for generation
--height 512                             # Image height (default: 512)
--width 512                              # Image width (default: 512)
--batch-size 1                           # Batch size (default: 1)
--seed 0                                 # Random seed (default: 0)
--num-inference-steps 50                 # Denoising steps (default: 50)
--guidance-scale 3.5                     # Guidance scale (default: 3.5)
```

### Engine & Performance Options

```bash
--precision {bf16,fp8,fp4}               # Transformer precision (default: fp8)
--dynamic-shape                          # Enable dynamic shape engines
--enable-runtime-cache                   # Enable runtime caching
--low-vram                               # Enable low VRAM mode
--verbose                                # Enable verbose logging
```

### Cache Management

```bash
--cache-dir ./demo_cache                 # Cache directory (default: ./demo_cache)
--cache-mode {full,lean}                 # Cache mode (default: full)
```

### Example Commands for Flux.1 \[dev\] Pipeline

**Default Parameters Image Generation:**

```bash
python demo/flux1.dev/flux_demo.py --hf-token YOUR_TOKEN
```

**Large Image Generation (1024x1024):**

```bash
python demo/flux1.dev/flux_demo.py --hf-token YOUR_TOKEN --height 1024 --width 1024 --prompt "A detailed cityscape at golden hour"
```

**Faster JIT Compilation Times with Runtime Caching:**

```bash
python demo/flux1.dev/flux_demo.py --hf-token YOUR_TOKEN --enable-runtime-cache --prompt "A cat meanders down a dimly lit alleyway in a large city."
```

**Dynamic-Shape Engines with Shape-Specialized Kernels:**

```bash
python demo/flux1.dev/flux_demo.py --hf-token YOUR_TOKEN --dynamic-shape --prompt "A dramatic cityscape from a dazzling angle"
```

**Low VRAM + FP4 Quantized (for Blackwell GPUs with memory constraints):**

```bash
python demo/flux1.dev/flux_demo.py --hf-token YOUR_TOKEN --low-vram --precision fp4 --prompt "A serene forest scene"
```

> **Tip**: The [Jupyter notebook](./flux1.dev/flux_demo.ipynb) provides interactive parameter exploration, detailed explanations of each feature, and additional use cases.

## Key Features

- **Smart Caching**: Shared models across pipelines with intelligent cleanup
- **Cross-Platform**: Works on Windows and Linux
- **Flexible Precision**: Configure transformer model precision (bf16, fp8, fp4)
- **Memory Management**: Low-VRAM mode for memory-constrained GPUs
- **Dynamic Shapes**: Support for flexible input dimensions with runtime optimization

## Notable Configuration Options

### Transformer Precision Options

Choose based on your GPU architecture and VRAM requirements:

| Precision | Supported GPU Architecture | VRAM Usage |
| --------- | -------------------------- | ---------- |
| **BF16**  | Ampere, Ada, Blackwell     | Most       |
| **FP8**   | Ada, Blackwell             | Medium     |
| **FP4**   | Blackwell                  | Least      |

```python
# Configure precision when loading engines
pipeline.load_engines(transformer_precision="fp8")  # Default: fp8
```

### Input Shape Modes

```python
# Static shapes (default)
pipeline.load_engines(opt_height=512, opt_width=512, shape_mode="static")

# Dynamic shapes (flexible resolutions without recompilation)
pipeline.load_engines(opt_height=512, opt_width=512, shape_mode="dynamic")
```

### GPU Memory Management

```python
# Default (fastest, more VRAM usage)
pipeline = Pipeline(..., low_vram=False)

# Low VRAM mode (slower, less VRAM usage)
pipeline = Pipeline(..., low_vram=True)
```

### Disk Memory Management

- **`full`** (default): Keep all cached models
- **`lean`**: Auto-cleanup unused models to save disk space

#### Cache Structure

Models and engines are stored in a shared cache by `model_id` and `precision`:

```
demo_cache/
├── shared/
│   ├── onnx/{model_id}/{precision}/           # ONNX models
│   └── engines/{model_id}/{precision}/        # TensorRT engines
├── runtime.cache                              # JIT compilation cache
└── .cache_state.json                          # Usage tracking
```

## Troubleshooting

**Image Quality Issues**

- Ensure the dimensions are multiples of 16
- Try altering the `seed` and `guidance_scale` parameters
- See [Flux.1 \[dev\] Demo Notebook](./flux1.dev/flux_demo.ipynb) for more tips and examples

**GPU Out of Memory**

- Use `low_vram=True` to reduce VRAM usage
- Try lower precision: `fp8` (Ada/Blackwell) or `fp4` (Blackwell only)
- Reduce batch size or image resolution

**Disk Space Issues**

- Use `cache_mode="lean"` to reduce disk usage by automatically cleaning up unused models
- Manually delete demo cache directory

**Build Errors**

- Verify TensorRT-RTX and dependencies are installed (see [Quick Start](#quick-start))
- Ensure the precision being used is supported by the GPU architecture (see [Support Matrix](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/getting-started/support-matrix.html))

## Running Tests

To configure the test environment and run demo tests, refer to the [test README](./tests/README.md).
