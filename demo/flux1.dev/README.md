# Flux.1 [dev]

This demo generates images with [FLUX.1 [dev]](https://huggingface.co/black-forest-labs/FLUX.1-dev) by downloading optimized ONNX models from [FLUX.1-dev-onnx](https://huggingface.co/black-forest-labs/FLUX.1-dev-onnx) and building TensorRT-RTX engines from them.

## Quick Start

1. **Clone and install**

   We recommend using Python versions between 3.9 and 3.12 inclusive due to supported versions for required dependencies.

   ```bash
   git clone https://github.com/NVIDIA/TensorRT-RTX.git
   cd TensorRT-RTX

   # Install TensorRT-RTX
   python -m pip install tensorrt-rtx

   # Install demo dependencies
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

The standalone script provides extensive configuration options for various use cases. For detailed walkthroughs, interactive exploration, and comprehensive documentation, see the [Flux.1 [dev] Demo Notebook](./flux_demo.ipynb) which offers in-depth coverage of TensorRT-RTX features.

> **GPU Compatibility**: This demo is verified on Ada and Blackwell GPUs. See [Transformer Precision Options](#transformer-precision-options) for more compatibility details.

### Required Parameters

To download model checkpoints for the FLUX.1 [dev] pipeline, obtain a `read` access token to the model repository on HuggingFace Hub. See [instructions](https://huggingface.co/docs/hub/security-tokens).

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
--weight-offload MODE                    # Configure weights offloading: none, buffer, or vmm (default: none)
--weight-offload-pinned-host             # Prepare all weights in pinned host RAM at setup
--verbose                                # Enable verbose logging
```

- **`--weight-offload buffer`** copies each engine's weights from a file-backed host buffer into TensorRT-managed GPU memory before execution.
- **`--weight-offload vmm`** uses CUDA virtual memory management to map application-provided GPU memory blocks directly
  into each engine's weight address range, without copying their contents during restore. The demo shares a pool sized
  for the largest engine and copies only missing or overwritten weights from host memory, reusing unchanged cached blocks.
- **`--weight-offload-pinned-host`** keeps all weights in pinned host RAM to avoid repeated file reads during restores.
  Host RAM must accommodate all weights plus runtime allocations.

### Cache Management

```bash
--cache-dir ./demo_cache                 # Cache directory (default: ./demo_cache)
--cache-mode {full,lean}                 # Cache mode (default: full)
```

### Example Commands for Flux.1 [dev] Pipeline

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

**Low-RAM Engine Builds:**

The weight-offloading path builds engines with native TensorRT APIs from Hugging Face configs and safetensors, without ONNX.
These downloads are pinned to Hugging Face commits compatible with the native network definitions.
Weight placeholders avoid loading checkpoint payloads and reduce TensorRT-RTX's internal build-time allocations,
lowering peak CPU memory usage. Actual weights are loaded during refit before inference.

All checkpoint tensors used by the four models are placeholders, including FP8/FP4 scales.
Graph-defined constants remain materialized; unused checkpoint tensors are omitted.

**Weights Offloading:**

```bash
python demo/flux1.dev/flux_demo.py --hf-token YOUR_TOKEN --weight-offload vmm --precision fp8 --prompt "A serene forest scene"
```

Weights offloading reduces GPU memory use across T5, CLIP, transformer, and VAE while retaining execution contexts.
The first run refits the engines and caches their weights beside them in `--cache-dir`; later runs reuse the cache.

This mode supports BF16, FP8 (default), and FP4 transformer weights; other components
remain BF16. FP8 and FP4 require access to [FLUX.1-dev-FP8](https://huggingface.co/black-forest-labs/FLUX.1-dev-FP8)
or [FLUX.1-dev-NVFP4](https://huggingface.co/black-forest-labs/FLUX.1-dev-NVFP4), respectively, in addition to
the original FLUX.1-dev repository. FP8 requires Ada or newer; FP4 requires Blackwell.

Weights are restored as needed between pipeline stages.
Add `--dynamic-shape` to reuse engines across batch sizes 1-4 and image dimensions 256-1024 (multiples of 16),
subject to available GPU memory. Text sequence lengths remain fixed.

### Deployment Options

**Benchmark configuration:** NVIDIA GeForce RTX 5090 (32 GiB), static shapes, batch size 1, 512x512 resolution,
28 denoising steps, guidance scale 3.5, and seed 12345.
Prompt: `A serene lake at sunset with mountains in the background`.

<table>
<thead>
<tr>
<th>Precision / Native weights (GiB)</th>
<th>Configuration</th>
<th>Total engine size (GiB)</th>
<th>Build CPU peak (GiB)</th>
<th>Build GPU peak (GiB)</th>
<th>Inference CPU peak (GiB)</th>
<th>Inference GPU peak (GiB)</th>
<th>E2E latency (s)</th>
</tr>
</thead>
<tbody>
<tr><td rowspan="3"><strong>FP8</strong><br/>20.67</td><td>Full weights</td><td>20.43</td><td>47.08</td><td>&lt; 0.50</td><td>3.47</td><td>21.37</td><td>1.51</td></tr>
<tr><td>Weightless only</td><td rowspan="2">0.13</td><td rowspan="2">1.98</td><td rowspan="2">&lt; 0.50</td><td>3.43</td><td>22.09</td><td>1.67</td></tr>
<tr><td>Weightless + weights offloading</td><td>24.26</td><td>12.90</td><td>2.12</td></tr>
<tr><td rowspan="3"><strong>NVFP4</strong><br/>17.75</td><td>Full weights</td><td>15.61</td><td>27.91</td><td>&lt; 0.50</td><td>3.80</td><td>16.56</td><td>1.07</td></tr>
<tr><td>Weightless only</td><td rowspan="2">0.14</td><td rowspan="2">1.98</td><td rowspan="2">&lt; 0.50</td><td>3.36</td><td>19.16</td><td>1.28</td></tr>
<tr><td>Weightless + weights offloading</td><td>21.32</td><td>10.29</td><td>1.72</td></tr>
<tr><td rowspan="3"><strong>BF16</strong><br/>31.36</td><td>Full weights</td><td>31.42</td><td>68.80</td><td>&lt; 0.50</td><td><strong>GPU OOM</strong></td><td><strong>GPU OOM</strong></td><td><strong>GPU OOM</strong></td></tr>
<tr><td>Weightless only</td><td rowspan="2">0.15</td><td rowspan="2">8.04</td><td rowspan="2">&lt; 0.50</td><td><strong>GPU OOM</strong></td><td><strong>GPU OOM</strong></td><td><strong>GPU OOM</strong></td></tr>
<tr><td>Weightless + weights offloading</td><td>34.96</td><td>23.59</td><td>2.96</td></tr>
</tbody>
</table>

**Table Notes**

- **Offloading:** Results use VMM with pinned host weights; inference CPU peaks include those buffers.
- **Sizes:** Totals cover all four components; only transformer precision varies. Native weight totals refer to
  weightless checkpoints, and stripped engine sizes exclude external weights. Full-weight ONNX exports and native
  checkpoints differ, so the table compares deployment options rather than isolated weight-stripping overhead.
- **Memory:** CPU peaks are process-tree RSS; GPU usage is sampled through NVML at a requested 20 ms interval.
  Build peaks are maxima across component builds, excluding loading, refit, and inference. ONNX measurements include
  the demo and parser processes; native builds use a standalone process without PyTorch. Inference peaks include
  warmup but exclude setup and refit.
- **Latency:** Median after warmup.
  E2E includes weight transfers and postprocessing, but excludes setup, refit, and image-file writes.
  GPU clocks were unlocked.

**Deployment Commands**

These commands use the benchmark generation settings and default prompt.

Buffer offloading:

```bash
python demo/flux1.dev/flux_demo.py --hf-token YOUR_TOKEN --precision fp8 \
    --weight-offload buffer \
    --height 512 --width 512 --batch-size 1 --num-inference-steps 28 --guidance-scale 3.5 --seed 12345
```

VMM offloading with file-backed staging:

```bash
python demo/flux1.dev/flux_demo.py --hf-token YOUR_TOKEN --precision fp8 \
    --weight-offload vmm \
    --height 512 --width 512 --batch-size 1 --num-inference-steps 28 --guidance-scale 3.5 --seed 12345
```

VMM offloading with pinned host weights (table configuration):

```bash
python demo/flux1.dev/flux_demo.py --hf-token YOUR_TOKEN --precision fp8 \
    --weight-offload vmm --weight-offload-pinned-host \
    --height 512 --width 512 --batch-size 1 --num-inference-steps 28 --guidance-scale 3.5 --seed 12345
```

Use `--precision bf16` or `--precision fp4` for the other precisions.

> **Tip**: The [Jupyter notebook](./flux_demo.ipynb) provides interactive parameter exploration, detailed explanations of each feature, and additional use cases.

## Key Features

- **Smart Caching**: Shared models across pipelines with intelligent cleanup
- **Cross-Platform**: Works on Windows and Linux
- **Flexible Precision**: Configure transformer model precision (bf16, fp8, fp4)
- **Memory Management**: Weights offloading for memory-constrained GPUs
- **Dynamic Shapes**: Support for flexible input dimensions with runtime optimization

## Notable Configuration Options

### Transformer Precision Options

See [Deployment Options](#deployment-options) for memory and latency measurements.

- **BF16:** Ampere, Ada, and Blackwell.
- **FP8:** Ada and Blackwell.
- **NVFP4 (`fp4`):** Blackwell.

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

Use `--weight-offload vmm` to reduce GPU memory usage.
See [Deployment Options](#deployment-options) for memory requirements and commands.

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
- See [Flux.1 [dev] Demo Notebook](./flux_demo.ipynb) for more tips and examples

**GPU Out of Memory**

- Use `--weight-offload vmm` to reduce VRAM usage
- Use `enable_runtime_cache=False` or omit the `--enable-runtime-cache` flag
- Try lower precision: `fp8` (Ada/Blackwell) or `fp4` (Blackwell only)
- Reduce batch size or image resolution

**Disk Space Issues**

- Use `cache_mode="lean"` to reduce disk usage by automatically cleaning up unused models
- Manually delete demo cache directory

**Build Errors**

- Verify TensorRT-RTX and dependencies are installed (see [Quick Start](#quick-start))
- Ensure the precision being used is supported by the GPU architecture (see [Support Matrix](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/getting-started/support-matrix.html))
