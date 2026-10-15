# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Define FLUX component networks with the TensorRT Python API.

Supports safetensors weights, weightless placeholders, and refitting.
Constructs CLIP, T5, transformer, and VAE decoder networks without ONNX or PyTorch.
"""

import json
import math
import struct
from pathlib import Path

import ml_dtypes
import numpy as np
import tensorrt_rtx as trt
from safetensors import safe_open

LOGGER = trt.Logger(trt.Logger.WARNING)
DTYPES = {
    "F32": (np.float32, trt.float32),
    "F16": (np.float16, trt.float16),
    "BF16": (ml_dtypes.bfloat16, trt.bfloat16),
    "F8_E4M3": (ml_dtypes.float8_e4m3fn, trt.fp8),
    "U8": (np.uint8, trt.uint8),
}


class CheckpointWeights:
    """Read safetensors metadata and load weight tensors on demand."""

    def __init__(self, directory, weight_file=None):
        self.directory = Path(directory)
        self.entries = {}
        self.storage = []
        self.payload_bytes = 0
        self.metadata_bytes = 0
        indexes = [] if weight_file is not None else list(self.directory.glob("*.safetensors.index.json"))
        if len(indexes) > 1:
            raise ValueError("Select a directory containing one checkpoint")
        weight_map = json.loads(indexes[0].read_text())["weight_map"] if indexes else None
        files = (
            sorted(set(weight_map.values()))
            if weight_map
            else sorted(p.name for p in self.directory.glob("*.safetensors"))
        )
        if weight_file is not None:
            files = [Path(weight_file).name]
        for filename in files:
            relative = Path(filename)
            # Windows rooted and drive-relative paths also escape the component.
            if relative.anchor or ".." in relative.parts:
                raise ValueError("Checkpoint shard paths must stay inside the component directory")
            # HF snapshot entries are symlinks to the cache's shared blob store.
            path = Path(weight_file) if weight_file is not None else self.directory / relative
            with safe_open(path, framework="numpy") as reader:
                names = set(reader.keys())
            with path.open("rb") as source:
                length = struct.unpack("<Q", source.read(8))[0]
                header = json.loads(source.read(length))
            quantization = header.get("__metadata__", {}).get("_quantization_metadata")
            layer_settings = {}
            if quantization is not None:
                quantization = json.loads(quantization)
                if (
                    not isinstance(quantization, dict)
                    or quantization.get("format_version") != "1.0"
                    or not isinstance(quantization.get("layers"), dict)
                    or set(quantization) - {"format_version", "layers"}
                ):
                    raise ValueError("Unsupported checkpoint quantization metadata")
                layer_settings = quantization["layers"]
            for name in names:
                if weight_map and weight_map.get(name) != filename:
                    continue
                if name in self.entries:
                    raise ValueError(f"Duplicate checkpoint tensor: {name}")
                entry = header[name]
                if entry["dtype"] not in DTYPES:
                    raise ValueError(f"Unsupported checkpoint dtype: {name}: {entry['dtype']}")
                self.entries[name] = {**entry, "path": path, "base": length + 8}
                if name.endswith(".weight") and name.removesuffix(".weight") in layer_settings:
                    self.entries[name]["quantization"] = layer_settings[name.removesuffix(".weight")]
        if not self.entries or (weight_map and set(weight_map) != set(self.entries)):
            raise ValueError("Checkpoint is empty or its shard index is incomplete")
        self.bfl = "img_in.weight" in self.entries
        if self.bfl:
            from flux_checkpoint_mapping import map_fused_checkpoint_tensors

            self.entries = map_fused_checkpoint_tensors(
                self.entries, json.loads((self.directory / "config.json").read_text())
            )
        self.packed = set()
        for name, entry in self.entries.items():
            if entry["dtype"] == "U8" and name.endswith(".weight"):
                prefix = name.removesuffix(".weight")
                scale = self.entries.get(prefix + ".weight_scale")
                global_scale = self.entries.get(prefix + ".weight_scale_2")
                if scale is None or global_scale is None or len(entry["shape"]) != 2:
                    raise ValueError(f"Packed weight is missing NVFP4 scales: {name}")
                rows, packed_cols = entry["shape"]
                if packed_cols % 8 or scale["shape"] != [rows, packed_cols // 8]:
                    raise ValueError(f"Unsupported NVFP4 packing or block scales: {name}")
                if scale["dtype"] not in {"F8_E4M3", "U8"} or math.prod(global_scale["shape"]) != 1:
                    raise ValueError(f"Unsupported NVFP4 scale format: {name}")
                self.packed.add(name)
        self.quantization = {}
        metadata = {}
        for name, entry in self.entries.items():
            if "quantization" in entry:
                self._register_quantization_metadata(name.removesuffix(".weight"), entry["quantization"])
            if not name.endswith(".comfy_quant"):
                continue
            start, end = entry["data_offsets"]
            if entry["dtype"] != "U8" or not 0 < end - start <= 4096:
                raise ValueError(f"Invalid quantization metadata: {name}")
            key = (entry["path"], entry["base"] + start, end - start)
            if key not in metadata:
                with entry["path"].open("rb") as source:
                    source.seek(key[1])
                    metadata[key] = json.loads(source.read(key[2]))
                self.metadata_bytes += key[2]
            self._register_quantization_metadata(name.removesuffix(".comfy_quant"), metadata[key])

    def _register_quantization_metadata(self, prefix, settings):
        """Validate and store a layer's checkpoint quantization metadata."""
        name = prefix + ".weight"
        expected = "nvfp4" if name in self.packed else "float8_e4m3fn"
        if (
            name not in self.entries
            or self.dtype(name) not in {trt.fp4, trt.fp8}
            or not isinstance(settings, dict)
            or settings.get("format") != expected
        ):
            raise ValueError(f"Unsupported quantization metadata: {prefix}")
        if set(settings) - {"format", "full_precision_matrix_mult"}:
            raise ValueError(f"Unsupported quantization settings: {prefix}")
        if expected == "nvfp4":
            rows, columns = self.entries[prefix + ".weight_scale"]["shape"]
            if rows % 128 or columns % 4:
                raise ValueError(f"Unsupported tiled NVFP4 scale dimensions: {prefix}")
        if prefix in self.quantization and self.quantization[prefix] != settings:
            raise ValueError(f"Conflicting quantization metadata: {prefix}")
        self.quantization[prefix] = settings

    def shape(self, name):
        shape = tuple(self.entries[name]["shape"])
        return shape[:-1] + (shape[-1] * 2,) if name in self.packed else shape

    def dtype(self, name):
        if name in self.packed:
            return trt.fp4
        if name.endswith(".weight_scale") and name.removesuffix("_scale") in self.packed:
            return trt.fp8
        return DTYPES[self.entries[name]["dtype"]][1]

    def descriptor(self, name, placeholder):
        entry = self.entries[name]
        count = math.prod(self.shape(name))
        dtype = self.dtype(name)
        pointer = 0
        if not placeholder:
            start, end = entry["data_offsets"]
            payload = np.memmap(
                entry["path"], mode="r", dtype=np.uint8, offset=entry["base"] + start, shape=(end - start,)
            )
            prefix = name.rsplit(".", 1)[0]
            if self.quantization.get(prefix, {}).get("format") == "nvfp4":
                if name.endswith(".weight"):
                    # Comfy/BFL stores the even element in the high nibble;
                    # TensorRT expects it in the low nibble.
                    payload = (payload << 4) | (payload >> 4)
                elif name.endswith(".weight_scale"):
                    # Undo the cuBLAS 128-row, four-column scale tiling.
                    rows, columns = entry["shape"]
                    payload = np.ascontiguousarray(
                        payload.reshape(rows // 128, columns // 4, 32, 4, 4).transpose(0, 3, 2, 1, 4)
                    ).reshape(-1)
            self.storage.append(payload)
            self.payload_bytes += end - start
            pointer = payload.ctypes.data
        return trt.Weights(dtype, pointer, count)

    def refit(self, engine, stream=None):
        """Refit one dependency group at a time, without importing PyTorch."""
        from cuda.bindings import runtime as cuda

        def checked(result):
            if int(result[0]) != 0:
                raise RuntimeError(f"CUDA operation failed: {result[0]}")
            return result[1] if len(result) == 2 else None

        owns_stream = stream is None
        if owns_stream:
            stream = checked(cuda.cudaStreamCreate())
        retained_buffers = len(self.storage)
        groups = 0
        try:
            refitter = trt.Refitter(engine, LOGGER)
            remaining = set(refitter.get_all_weights())
            if not remaining or not remaining <= self.entries.keys():
                raise ValueError("Engine refit names do not match this checkpoint")
            while remaining:
                loaded = set()
                pending = [min(remaining)]
                while pending:
                    for name in pending:
                        if name in loaded:
                            raise RuntimeError(f"Refit dependency discovery stalled: {name}")
                        weights = self.descriptor(name, False)
                        prototype = refitter.get_weights_prototype(name)
                        if weights.dtype != prototype.dtype or weights.size != prototype.size:
                            raise ValueError(f"Refit descriptor mismatch: {name}")
                        if not refitter.set_named_weights(name, weights, trt.TensorLocation.HOST):
                            raise RuntimeError(f"Cannot set refit weights: {name}")
                        loaded.add(name)
                        remaining.discard(name)
                    pending = refitter.get_missing_weights()
                if not refitter.refit_cuda_engine_async(stream):
                    raise RuntimeError("Engine refit failed")
                checked(cuda.cudaStreamSynchronize(stream))
                if not refitter.release_refit_resources():
                    raise RuntimeError("Cannot release refit resources")
                for name in loaded:
                    if not refitter.unset_named_weights(name):
                        raise RuntimeError(f"Cannot unset refit weight: {name}")
                del self.storage[retained_buffers:]
                groups += 1
            return groups
        finally:
            if owns_stream:
                checked(cuda.cudaStreamDestroy(stream))


class _Network:
    def __init__(self, builder, weights, weightless, input_profiles=None):
        self.net = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
        self.weights = weights
        self.weightless = weightless
        self.dtype = next(
            (DTYPES[e["dtype"]][1] for e in weights.entries.values() if e["dtype"] in {"F32", "F16", "BF16"}),
            trt.bfloat16,
        )
        self.storage = []
        self.parameters = {}
        self.input_profiles = input_profiles or {}

    @staticmethod
    def out(layer):
        if layer is None:
            raise RuntimeError("TensorRT rejected a layer")
        return layer.get_output(0)

    def input(self, name, shape, dtype=None):
        if self.input_profiles:
            if name not in self.input_profiles:
                raise ValueError(f"Missing input profile for {name}")
            minimum, optimum, maximum = self.input_profiles[name]
            if (
                tuple(optimum) != tuple(shape)
                or any(len(bound) != len(shape) for bound in (minimum, optimum, maximum))
                or any(not 0 < low <= opt <= high for low, opt, high in zip(minimum, optimum, maximum))
            ):
                raise ValueError(f"Invalid input profile for {name}")
            shape = tuple(low if low == high else -1 for low, high in zip(minimum, maximum))
        return self.net.add_input(name, dtype or self.dtype, shape)

    def dimensions(self, value):
        if -1 not in value.shape:
            return tuple(value.shape)
        shape = self.cast(self.out(self.net.add_shape(value)), trt.int32)
        return tuple(
            dim if dim != -1 else self.out(self.net.add_gather(shape, self.constant([i], trt.int32), 0))
            for i, dim in enumerate(value.shape)
        )

    def _shape_tensor(self, dimensions):
        return self.concat(
            [dim if isinstance(dim, trt.ITensor) else self.constant([dim], trt.int32) for dim in dimensions], 0
        )

    def constant(self, value, dtype=None):
        dtype = dtype or self.dtype
        numpy_type = next((n for n, t in DTYPES.values() if t == dtype), None)
        if numpy_type is None:
            numpy_type = np.int32 if dtype == trt.int32 else np.int64
        value = np.asarray(value, dtype=numpy_type)
        if not value.flags.c_contiguous:
            value = np.ascontiguousarray(value)
        self.storage.append(value)
        return self.out(self.net.add_constant(value.shape, trt.Weights(dtype, value.ctypes.data, value.size)))

    def weight(self, name):
        if name not in self.parameters:
            layer = self.net.add_constant(self.weights.shape(name), self.weights.descriptor(name, self.weightless))
            if layer is None or not self.net.set_weights_name(layer.weights, name):
                raise RuntimeError(f"Cannot name parameter: {name}")
            self.parameters[name] = self.out(layer)
        return self.parameters[name]

    def cast(self, value, dtype):
        return value if value.dtype == dtype else self.out(self.net.add_cast(value, dtype))

    def reshape(self, value, shape):
        layer = self.net.add_shuffle(value)
        layer.zero_is_placeholder = False
        if any(isinstance(dim, trt.ITensor) for dim in shape):
            layer.set_input(1, self._shape_tensor(shape))
        else:
            layer.reshape_dims = tuple(shape)
        return self.out(layer)

    def transpose(self, value, permutation):
        layer = self.net.add_shuffle(value)
        layer.first_transpose = permutation
        return self.out(layer)

    def binary(self, left, right, operation):
        if not isinstance(right, trt.ITensor):
            right = self.constant(right, left.dtype)
        rank = max(len(left.shape), len(right.shape))
        left, right = [
            self.reshape(v, (1,) * (rank - len(v.shape)) + self.dimensions(v)) if len(v.shape) < rank else v
            for v in (left, right)
        ]
        return self.out(self.net.add_elementwise(left, right, operation))

    def add(self, left, right):
        return self.binary(left, right, trt.ElementWiseOperation.SUM)

    def mul(self, left, right):
        return self.binary(left, right, trt.ElementWiseOperation.PROD)

    def unary(self, value, operation):
        return self.out(self.net.add_unary(value, operation))

    def concat(self, values, axis):
        layer = self.net.add_concatenation(values)
        layer.axis = axis
        return self.out(layer)

    def slice(self, value, axis, start, size):
        begin, shape = [0] * len(value.shape), list(self.dimensions(value))
        begin[axis], shape[axis] = start, size
        if any(isinstance(dim, trt.ITensor) for dim in begin + shape):
            layer = self.net.add_slice(value, [0] * len(shape), [1] * len(shape), [1] * len(shape))
            layer.set_input(1, self._shape_tensor(begin))
            layer.set_input(2, self._shape_tensor(shape))
            return self.out(layer)
        return self.out(self.net.add_slice(value, begin, shape, [1] * len(shape)))

    def chunks(self, value, count):
        width = value.shape[-1] // count
        return [self.slice(value, len(value.shape) - 1, i * width, width) for i in range(count)]

    def linear(self, value, name):
        if self.weights.dtype(name + ".weight") in {trt.fp8, trt.fp4}:
            return self._quantized_linear(value, name)
        weight = self.cast(self.weight(name + ".weight"), value.dtype)
        if weight.shape[1] != value.shape[-1]:
            raise ValueError(f"Linear input shape mismatch: {name}")
        weight = self.reshape(weight, (1,) * (len(value.shape) - 2) + tuple(weight.shape))
        result = self.out(
            self.net.add_matrix_multiply(value, trt.MatrixOperation.NONE, weight, trt.MatrixOperation.TRANSPOSE)
        )
        bias = name + ".bias"
        return self.add(result, self.cast(self.weight(bias), value.dtype)) if bias in self.weights.entries else result

    def _quantized_linear(self, value, name):
        weight = self.weight(name + ".weight")
        if len(weight.shape) != 2 or weight.shape[1] != value.shape[-1]:
            raise ValueError(f"Quantized linear input shape mismatch: {name}")
        original_shape = self.dimensions(value)
        value = self.reshape(value, (-1, original_shape[-1]))
        for suffix in ("input_scale", "weight_scale" if weight.dtype == trt.fp8 else "weight_scale_2"):
            key = name + "." + suffix
            if key not in self.weights.entries or math.prod(self.weights.shape(key)) != 1:
                raise ValueError(f"Quantized linear requires a scalar {suffix}: {name}")
            if self.weights.dtype(key) not in {trt.float32, trt.float16, trt.bfloat16}:
                raise ValueError(f"Invalid quantization scale dtype: {key}")
        scale = self.weight(name + ".weight_scale")
        input_scale = self.weight(name + ".input_scale")
        input_scale = self.reshape(self.cast(input_scale, trt.float32), ())
        if name + ".pre_quant_scale" in self.weights.entries:
            if math.prod(self.weights.shape(name + ".pre_quant_scale")) != value.shape[-1]:
                raise ValueError(f"Invalid input-channel smoothing scale: {name}")
            smoothing = self.reshape(self.weight(name + ".pre_quant_scale"), (value.shape[-1],))
            value = self.mul(value, self.cast(smoothing, value.dtype))
        if self.weights.quantization.get(name, {}).get("full_precision_matrix_mult", False):
            raise ValueError(f"Weight-only quantized matrix multiplication is not supported: {name}")
        # Keep learned scales outside Q/DQ: RTX 1.7 requires build-time Q/DQ
        # scales, but every checkpoint scale must remain refittable here.
        if weight.dtype == trt.fp8:
            scale = self.reshape(self.cast(scale, trt.float32), ())
            value = self.cast(
                self.binary(self.cast(value, trt.float32), input_scale, trt.ElementWiseOperation.DIV), value.dtype
            )
            unit = self.constant(np.array(1.0, dtype=np.float32), trt.float32)
            quantized = self.out(self.net.add_quantize(value, unit, trt.fp8))
            value = self.out(self.net.add_dequantize(quantized, unit, value.dtype))
            weight = self.out(self.net.add_dequantize(weight, unit, value.dtype))
            output_scale = self.mul(scale, input_scale)
        else:
            global_scale = self.reshape(self.cast(self.weight(name + ".weight_scale_2"), trt.float32), ())
            value = self.cast(
                self.binary(self.cast(value, trt.float32), input_scale, trt.ElementWiseOperation.DIV), value.dtype
            )
            quantized = self.net.add_dynamic_quantize(value, -1, 16, trt.fp4, trt.fp8)
            unit = self.constant(np.array(1.0, dtype=np.float32), trt.float32)
            quantized.set_input(1, unit)
            activation_scale = self.out(self.net.add_dequantize(quantized.get_output(1), unit, value.dtype))
            activation = self.net.add_dequantize(quantized.get_output(0), activation_scale, value.dtype)
            activation.axis = 1
            value = self.out(activation)
            scale = self.out(self.net.add_dequantize(scale, unit, value.dtype))
            dequantized = self.net.add_dequantize(weight, scale, value.dtype)
            dequantized.axis = 1
            weight = self.out(dequantized)
            output_scale = self.mul(global_scale, input_scale)
        result = self.out(
            self.net.add_matrix_multiply(value, trt.MatrixOperation.NONE, weight, trt.MatrixOperation.TRANSPOSE)
        )
        result = self.cast(self.mul(self.cast(result, trt.float32), output_scale), value.dtype)
        result = self.reshape(result, original_shape[:-1] + (weight.shape[0],))
        bias = name + ".bias"
        return self.add(result, self.cast(self.weight(bias), value.dtype)) if bias in self.weights.entries else result

    def silu(self, value):
        return self.mul(value, self.out(self.net.add_activation(value, trt.ActivationType.SIGMOID)))

    def gelu(self, value, quick=False):
        if quick:
            return self.mul(
                value, self.out(self.net.add_activation(self.mul(value, 1.702), trt.ActivationType.SIGMOID))
            )
        cubic = self.mul(self.mul(value, value), value)
        inner = self.mul(self.add(value, self.mul(cubic, 0.044715)), math.sqrt(2 / math.pi))
        return self.mul(
            self.mul(value, 0.5), self.add(self.out(self.net.add_activation(inner, trt.ActivationType.TANH)), 1.0)
        )

    def norm(self, value, epsilon, name=None):
        shape = (1,) * (len(value.shape) - 1) + (value.shape[-1],)
        scale = (
            self.cast(self.weight(name + ".weight"), value.dtype)
            if name
            else self.constant(np.ones(shape), value.dtype)
        )
        bias = (
            self.cast(self.weight(name + ".bias"), value.dtype) if name else self.constant(np.zeros(shape), value.dtype)
        )
        layer = self.net.add_normalization(
            value, self.reshape(scale, shape), self.reshape(bias, shape), 1 << (len(shape) - 1)
        )
        layer.epsilon = epsilon
        return self.out(layer)

    def conv(self, x, name):
        shape = self.weights.shape(name + ".weight")
        kernel = self.weights.descriptor(name + ".weight", self.weightless)
        bias = self.weights.descriptor(name + ".bias", self.weightless)
        if kernel.dtype != x.dtype or bias.dtype != x.dtype:
            raise ValueError(f"Mixed convolution parameter dtype: {name}")
        layer = self.net.add_convolution_nd(x, shape[0], shape[2:], kernel, bias)
        if layer is None:
            raise RuntimeError(f"Cannot create convolution: {name}")
        layer.padding_nd = (shape[2] // 2, shape[3] // 2)
        for suffix, descriptor in (("weight", layer.kernel), ("bias", layer.bias)):
            if not self.net.set_weights_name(descriptor, name + "." + suffix):
                raise RuntimeError(f"Cannot name convolution weights: {name}")
        return self.out(layer)

    def group_norm(self, x, name, groups, epsilon=1e-6):
        batch, channels, height, width = self.dimensions(x)
        if channels % groups:
            raise ValueError("Group normalization channels must divide into groups")
        grouped = self.reshape(self.cast(x, trt.float32), (batch, groups, channels // groups, height, width))
        affine = (1, groups, channels // groups, 1, 1)
        layer = self.net.add_normalization(
            grouped,
            self.reshape(self.cast(self.weight(name + ".weight"), trt.float32), affine),
            self.reshape(self.cast(self.weight(name + ".bias"), trt.float32), affine),
            4 | 8 | 16,
        )
        layer.epsilon = epsilon
        return self.cast(self.reshape(self.out(layer), (batch, channels, height, width)), x.dtype)

    def rms(self, value, name, epsilon):
        data = self.cast(value, trt.float32)
        variance = self.out(
            self.net.add_reduce(self.mul(data, data), trt.ReduceOperation.AVG, 1 << (len(data.shape) - 1), True)
        )
        inverse = self.unary(self.add(variance, epsilon), trt.UnaryOperation.SQRT)
        normalized = self.cast(self.binary(data, inverse, trt.ElementWiseOperation.DIV), value.dtype)
        return self.mul(normalized, self.cast(self.weight(name + ".weight"), value.dtype))

    def heads(self, value, heads):
        batch, length, width = self.dimensions(value)
        if heads <= 0 or width % heads:
            raise ValueError("Attention width must divide into heads")
        return self.transpose(self.reshape(value, (batch, length, heads, width // heads)), (0, 2, 1, 3))

    def attention(self, query, key, value, bias=None, scaled=True):
        scores = self.out(
            self.net.add_matrix_multiply(query, trt.MatrixOperation.NONE, key, trt.MatrixOperation.TRANSPOSE)
        )
        if scaled:
            scores = self.mul(scores, 1 / math.sqrt(query.shape[-1]))
        if bias is not None:
            scores = self.add(scores, self.cast(bias, scores.dtype))
        softmax = self.net.add_softmax(self.cast(scores, trt.float32))
        softmax.axes = 1 << 3
        probabilities = self.cast(self.out(softmax), value.dtype)
        result = self.out(
            self.net.add_matrix_multiply(probabilities, trt.MatrixOperation.NONE, value, trt.MatrixOperation.NONE)
        )
        batch, heads, length, width = self.dimensions(result)
        return self.reshape(self.transpose(result, (0, 2, 1, 3)), (batch, length, heads * width))

    def mark(self, value, name):
        value.name = name
        self.net.mark_output(value)


def clip(n, c, batch, length):
    if c.get("hidden_act", "quick_gelu") != "quick_gelu" or length > c["max_position_embeddings"]:
        raise ValueError("Only FLUX CLIP quick_gelu and configured position range are supported")
    ids = n.input("input_ids", (batch, length), trt.int32)
    embedding = n.weight("text_model.embeddings.token_embedding.weight")
    x = n.out(n.net.add_gather(embedding, ids, 0))
    positions = n.constant(np.arange(length), trt.int32)
    x = n.add(x, n.out(n.net.add_gather(n.weight("text_model.embeddings.position_embedding.weight"), positions, 0)))
    mask = n.constant(np.where(np.triu(np.ones((length, length)), 1), -1e4, 0).reshape(1, 1, length, length), x.dtype)
    for i in range(c["num_hidden_layers"]):
        p = f"text_model.encoder.layers.{i}"
        norm = n.norm(x, c.get("layer_norm_eps", 1e-5), p + ".layer_norm1")
        q, k, v = [
            n.heads(n.linear(norm, p + ".self_attn." + kind + "_proj"), c["num_attention_heads"])
            for kind in ("q", "k", "v")
        ]
        x = n.add(x, n.linear(n.attention(q, k, v, mask), p + ".self_attn.out_proj"))
        norm = n.norm(x, c.get("layer_norm_eps", 1e-5), p + ".layer_norm2")
        x = n.add(x, n.linear(n.gelu(n.linear(norm, p + ".mlp.fc1"), quick=True), p + ".mlp.fc2"))
    x = n.norm(x, c.get("layer_norm_eps", 1e-5), "text_model.final_layer_norm")
    if c.get("eos_token_id", 2) == 2:
        layer = n.net.add_topk(n.cast(ids, trt.float32), trt.TopKOperation.MAX, 1, 2)
        end = n.reshape(layer.get_output(1), (-1,))
    else:
        matches = n.binary(ids, c["eos_token_id"], trt.ElementWiseOperation.EQUAL)
        offsets = n.constant(np.arange(length).reshape(1, length), trt.int32)
        sentinel = n.constant([[length]], trt.int32)
        choices = n.out(n.net.add_select(matches, offsets, sentinel))
        end = n.out(n.net.add_reduce(choices, trt.ReduceOperation.MIN, 2, False))
        valid = n.binary(end, length, trt.ElementWiseOperation.LESS)
        end = n.out(n.net.add_select(valid, end, n.constant([0], trt.int32)))
    if n.input_profiles:
        gather = n.net.add_gather(x, n.reshape(end, (-1, 1)), 1)
        gather.num_elementwise_dims = 1
        pooled = n.reshape(n.out(gather), (-1, x.shape[-1]))
    else:
        index = n.add(end, n.constant(np.arange(batch) * length, trt.int32))
        pooled = n.out(n.net.add_gather(n.reshape(x, (batch * length, x.shape[-1])), index, 0))
    n.mark(x, "text_embeddings")
    n.mark(pooled, "pooled_embeddings")


def t5_relative_position_buckets(length, buckets, distance):
    relative = np.arange(length)[None, :] - np.arange(length)[:, None]
    half = buckets // 2
    result = (relative > 0).astype(np.int32) * half
    position = np.abs(relative)
    exact = half // 2
    large = exact + (np.log(np.maximum(position, exact) / exact) / math.log(distance / exact) * (half - exact)).astype(
        np.int32
    )
    return result + np.where(position < exact, position, np.minimum(large, half - 1)).astype(np.int32)


def t5(n, c, batch, length):
    if c.get("feed_forward_proj", "gated-gelu") != "gated-gelu":
        raise ValueError("Only FLUX T5 gated-gelu is supported")
    ids = n.input("input_ids", (batch, length), trt.int32)
    key = "shared.weight" if "shared.weight" in n.weights.entries else "encoder.embed_tokens.weight"
    x = n.out(n.net.add_gather(n.weight(key), ids, 0))
    buckets = t5_relative_position_buckets(
        length, c.get("relative_attention_num_buckets", 32), c.get("relative_attention_max_distance", 128)
    )
    bias = n.out(
        n.net.add_gather(
            n.weight("encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"),
            n.constant(buckets, trt.int32),
            0,
        )
    )
    bias = n.reshape(n.transpose(bias, (2, 0, 1)), (1, c["num_heads"], length, length))
    for i in range(c["num_layers"]):
        p = f"encoder.block.{i}"
        norm = n.rms(x, p + ".layer.0.layer_norm", c.get("layer_norm_epsilon", 1e-6))
        q, k, v = [
            n.heads(n.linear(norm, p + ".layer.0.SelfAttention." + kind), c["num_heads"]) for kind in ("q", "k", "v")
        ]
        x = n.add(x, n.linear(n.attention(q, k, v, bias, scaled=False), p + ".layer.0.SelfAttention.o"))
        norm = n.rms(x, p + ".layer.1.layer_norm", c.get("layer_norm_epsilon", 1e-6))
        p += ".layer.1.DenseReluDense"
        hidden = n.mul(n.gelu(n.linear(norm, p + ".wi_0")), n.linear(norm, p + ".wi_1"))
        x = n.add(x, n.linear(hidden, p + ".wo"))
    n.mark(n.rms(x, "encoder.final_layer_norm", c.get("layer_norm_epsilon", 1e-6)), "text_embeddings")


def vae(n, c, batch, height, width):
    if c.get("act_fn", "silu") != "silu" or any(t != "UpDecoderBlock2D" for t in c["up_block_types"]):
        raise ValueError("Only the FLUX SiLU UpDecoderBlock2D VAE is supported")
    factor = 2 ** (len(c["block_out_channels"]) - 1)
    if height % factor or width % factor:
        raise ValueError("Image dimensions must divide by the VAE upsampling factor")
    x = n.input("latent", (batch, c["latent_channels"], height // factor, width // factor))
    if c.get("use_post_quant_conv", True):
        x = n.conv(x, "post_quant_conv")
    x = n.conv(x, "decoder.conv_in")

    def residual(value, name):
        hidden = n.conv(n.silu(n.group_norm(value, name + ".norm1", c["norm_num_groups"])), name + ".conv1")
        hidden = n.conv(n.silu(n.group_norm(hidden, name + ".norm2", c["norm_num_groups"])), name + ".conv2")
        if name + ".conv_shortcut.weight" in n.weights.entries:
            value = n.conv(value, name + ".conv_shortcut")
        return n.add(value, hidden)

    x = residual(x, "decoder.mid_block.resnets.0")
    if c.get("mid_block_add_attention", True):
        p = "decoder.mid_block.attentions.0"
        norm = n.group_norm(x, p + ".group_norm", c["norm_num_groups"])
        b, channels, h, w = n.dimensions(x)
        norm = n.transpose(n.reshape(norm, (b, channels, -1)), (0, 2, 1))
        q, k, v = [n.heads(n.linear(norm, p + ".to_" + kind), 1) for kind in ("q", "k", "v")]
        hidden = n.linear(n.attention(q, k, v), p + ".to_out.0")
        x = n.add(x, n.reshape(n.transpose(hidden, (0, 2, 1)), (b, channels, h, w)))
    x = residual(x, "decoder.mid_block.resnets.1")
    for i in range(len(c["block_out_channels"])):
        for j in range(c["layers_per_block"] + 1):
            x = residual(x, f"decoder.up_blocks.{i}.resnets.{j}")
        if i + 1 < len(c["block_out_channels"]):
            layer = n.net.add_resize(x)
            layer.resize_mode = trt.InterpolationMode.NEAREST
            layer.coordinate_transformation = trt.ResizeCoordinateTransformation.ASYMMETRIC
            layer.nearest_rounding = trt.ResizeRoundMode.FLOOR
            layer.scales = (1, 1, 2, 2)
            x = n.conv(n.out(layer), f"decoder.up_blocks.{i}.upsamplers.0.conv")
    x = n.conv(n.silu(n.group_norm(x, "decoder.conv_norm_out", c["norm_num_groups"])), "decoder.conv_out")
    n.mark(x, "images")


def transformer(n, c, batch, image_tokens, text_tokens):
    heads, head_dim = c["num_attention_heads"], c["attention_head_dim"]
    dim = heads * head_dim
    axes = c.get("axes_dims_rope", [16, 56, 56])
    if sum(axes) != head_dim or any(d <= 0 or d % 2 for d in axes) or c.get("patch_size", 1) != 1:
        raise ValueError("Unsupported FLUX rotary dimensions or patch size")
    x = n.input("hidden_states", (batch, image_tokens, c["in_channels"]))
    context = n.input("encoder_hidden_states", (batch, text_tokens, c["joint_attention_dim"]))
    pooled = n.input("pooled_projections", (batch, c["pooled_projection_dim"]))
    timestep = n.input("timestep", (batch,))
    img_ids = n.input("img_ids", (image_tokens, len(axes)), trt.float32)
    txt_ids = n.input("txt_ids", (text_tokens, len(axes)), trt.float32)
    guidance = n.input("guidance", (batch,), trt.float32) if c.get("guidance_embeds", False) else None
    batch = n.dimensions(timestep)[0]
    image_tokens = n.dimensions(img_ids)[0]
    x = n.linear(x, "x_embedder")
    context = n.linear(context, "context_embedder")

    def time_embedding(value, name):
        value = n.cast(n.mul(n.cast(value, n.dtype), 1000.0), trt.float32)
        frequencies = np.exp(-math.log(10000) * np.arange(128, dtype=np.float32) / 128)
        angles = n.mul(n.reshape(value, (batch, 1)), n.constant(frequencies.reshape(1, 128), trt.float32))
        embed = n.concat([n.unary(angles, trt.UnaryOperation.COS), n.unary(angles, trt.UnaryOperation.SIN)], 1)
        return n.linear(n.silu(n.linear(n.cast(embed, n.dtype), name + ".linear_1")), name + ".linear_2")

    temb = time_embedding(timestep, "time_text_embed.timestep_embedder")
    if guidance is not None:
        temb = n.add(temb, time_embedding(guidance, "time_text_embed.guidance_embedder"))
    temb = n.add(
        temb,
        n.linear(
            n.silu(n.linear(pooled, "time_text_embed.text_embedder.linear_1")), "time_text_embed.text_embedder.linear_2"
        ),
    )
    ids = n.concat([txt_ids, img_ids], 0)
    cosines, sines = [], []
    for axis, width in enumerate(axes):
        frequencies = np.power(10000.0, -np.arange(0, width, 2, dtype=np.float64) / width).astype(np.float32)
        frequencies = np.repeat(frequencies, 2).reshape(1, width)
        angles = n.mul(n.slice(ids, 1, axis, 1), n.constant(frequencies, trt.float32))
        cosines.append(n.unary(angles, trt.UnaryOperation.COS))
        sines.append(n.unary(angles, trt.UnaryOperation.SIN))
    cosine = n.reshape(n.concat(cosines, 1), (1, 1, -1, head_dim))
    sine = n.reshape(n.concat(sines, 1), (1, 1, -1, head_dim))

    def rotary(value):
        shape = n.dimensions(value)
        pairs = n.reshape(n.cast(value, trt.float32), shape[:-1] + (head_dim // 2, 2))
        first, second = n.slice(pairs, 4, 0, 1), n.slice(pairs, 4, 1, 1)
        rotated = n.reshape(n.concat([n.unary(second, trt.UnaryOperation.NEG), first], 4), shape)
        return n.cast(n.add(n.mul(n.cast(value, trt.float32), cosine), n.mul(rotated, sine)), value.dtype)

    def projected(value, p, context=False):
        names = ("add_q_proj", "add_k_proj", "add_v_proj") if context else ("to_q", "to_k", "to_v")
        q, k, v = [n.heads(n.linear(value, p + "." + name), heads) for name in names]
        prefix = ".norm_added_" if context else ".norm_"
        return n.rms(q, p + prefix + "q", 1e-6), n.rms(k, p + prefix + "k", 1e-6), v

    def modulate(value, shift, scale):
        return n.add(n.mul(value, n.add(n.reshape(scale, (batch, 1, dim)), 1.0)), n.reshape(shift, (batch, 1, dim)))

    def gated(residual, value, gate):
        return n.add(residual, n.mul(value, n.reshape(gate, (batch, 1, dim))))

    for i in range(c["num_layers"]):
        p = f"transformer_blocks.{i}"
        xs = n.chunks(n.linear(n.silu(temb), p + ".norm1.linear"), 6)
        cs = n.chunks(n.linear(n.silu(temb), p + ".norm1_context.linear"), 6)
        xn = modulate(n.norm(x, 1e-6), xs[0], xs[1])
        cn = modulate(n.norm(context, 1e-6), cs[0], cs[1])
        qkv = projected(xn, p + ".attn")
        cqkv = projected(cn, p + ".attn", True)
        q, k, v = [n.concat([cv, xv], 2) for cv, xv in zip(cqkv, qkv)]
        attention = n.attention(rotary(q), rotary(k), v)
        x = gated(x, n.linear(n.slice(attention, 1, text_tokens, image_tokens), p + ".attn.to_out.0"), xs[2])
        context = gated(context, n.linear(n.slice(attention, 1, 0, text_tokens), p + ".attn.to_add_out"), cs[2])
        for is_context, value, modulation, ff in ((False, x, xs, ".ff"), (True, context, cs, ".ff_context")):
            hidden = modulate(n.norm(value, 1e-6), modulation[3], modulation[4])
            hidden = n.linear(n.gelu(n.linear(hidden, p + ff + ".net.0.proj")), p + ff + ".net.2")
            value = gated(value, hidden, modulation[5])
            if is_context:
                context = value
            else:
                x = value
    x = n.concat([context, x], 1)
    for i in range(c["num_single_layers"]):
        p = f"single_transformer_blocks.{i}"
        shift, scale, gate = n.chunks(n.linear(n.silu(temb), p + ".norm.linear"), 3)
        normalized = modulate(n.norm(x, 1e-6), shift, scale)
        q, k, v = projected(normalized, p + ".attn")
        attention = n.attention(rotary(q), rotary(k), v)
        hidden = n.concat([attention, n.gelu(n.linear(normalized, p + ".proj_mlp"))], 2)
        x = gated(x, n.linear(hidden, p + ".proj_out"), gate)
    x = n.slice(x, 1, text_tokens, image_tokens)
    scale, shift = n.chunks(n.linear(n.silu(temb), "norm_out.linear"), 2)
    if n.weights.bfl:
        shift, scale = scale, shift
    n.mark(n.linear(modulate(n.norm(x, 1e-6), shift, scale), "proj_out"), "latent")


def create_network(
    builder,
    directory,
    role,
    weightless,
    batch=1,
    height=512,
    width=512,
    text_length=None,
    weight_file=None,
    input_profiles=None,
):
    if batch < 1 or height < 1 or width < 1:
        raise ValueError("Batch and image dimensions must be positive")
    if text_length is not None and text_length < 1:
        raise ValueError("Text length must be positive")
    if input_profiles:
        dynamic_axes = {
            "clip": {"input_ids": {0}},
            "t5": {"input_ids": {0}},
            "vae": {"latent": {0, 2, 3}},
            "transformer": {
                "hidden_states": {0, 1},
                "encoder_hidden_states": {0},
                "pooled_projections": {0},
                "timestep": {0},
                "guidance": {0},
                "img_ids": {0},
                "txt_ids": set(),
            },
        }
        for name, bounds in input_profiles.items():
            if role not in dynamic_axes or name not in dynamic_axes[role] or len(bounds) != 3:
                raise ValueError(f"Unsupported input profile for {role}/{name}")
            if any(len(shape) != len(bounds[0]) for shape in bounds):
                raise ValueError(f"Inconsistent input profile ranks for {name}")
            for axis, dimensions in enumerate(zip(*bounds)):
                if axis not in dynamic_axes[role][name] and len(set(dimensions)) != 1:
                    raise ValueError(f"Only batch and image dimensions may vary: {role}/{name}/{axis}")
    config = json.loads((Path(directory) / "config.json").read_text())
    if config.get("quantization_config"):
        raise ValueError(
            "Quantized checkpoints with config-level quantization settings are unsupported; "
            "use the BFL safetensors checkpoint with the original FLUX architecture config"
        )
    weights = CheckpointWeights(directory, weight_file)
    network = _Network(builder, weights, weightless, input_profiles)
    anchors = {
        "clip": "text_model.embeddings.token_embedding.weight",
        "vae": "decoder.conv_in.weight",
        "transformer": "x_embedder.weight",
        "t5": "shared.weight" if "shared.weight" in weights.entries else "encoder.embed_tokens.weight",
    }
    if role not in anchors:
        raise ValueError(f"Unsupported role: {role}")
    network.dtype = weights.dtype(anchors[role])
    if network.dtype in {trt.fp8, trt.fp4}:
        network.dtype = trt.bfloat16
    if role == "clip":
        clip(network, config, batch, text_length or 77)
    elif role == "t5":
        t5(network, config, batch, text_length or 512)
    elif role == "vae":
        vae(network, config, batch, height, width)
    elif role == "transformer":
        if height % 16 or width % 16:
            raise ValueError("FLUX image dimensions must divide by 16")
        transformer(network, config, batch, (height // 16) * (width // 16), text_length or 512)
    else:
        raise ValueError(f"Unsupported role: {role}")
    return network
