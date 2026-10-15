# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Map fused FLUX checkpoint tensors to native network parameter names.

The BF16 checkpoints used by this demo already use the expected Diffusers names.
The FP8/FP4 checkpoints use different names and fused tensors; this module maps
their names and tensor ranges without copying weight data.
"""

import math


def map_fused_checkpoint_tensors(entries, config):
    """Expose Diffusers parameter names without copying or unpacking weight data."""
    aliases = dict(entries)
    dim = config["num_attention_heads"] * config["attention_head_dim"]

    def tensor(source, target, start=None, count=None):
        if source not in entries:
            return
        entry = dict(entries[source])
        if start is not None:
            shape = list(entry["shape"])
            if not shape or start < 0 or start + count > shape[0]:
                raise ValueError(f"Invalid fused checkpoint range: {source}")
            begin, end = entry["data_offsets"]
            row_bytes, remainder = divmod(end - begin, shape[0])
            if remainder:
                raise ValueError(f"Nonintegral checkpoint row size: {source}")
            entry["data_offsets"] = [begin + start * row_bytes, begin + (start + count) * row_bytes]
            shape[0] = count
            entry["shape"] = shape
        aliases[target] = entry

    def linear(source, target, start=None, count=None):
        for suffix in ("weight", "bias"):
            tensor(source + "." + suffix, target + "." + suffix, start, count)
        for suffix in ("weight_scale", "weight_scale_2", "input_scale", "pre_quant_scale", "comfy_quant"):
            key = source + "." + suffix
            if key not in entries:
                continue
            shape = entries[key]["shape"]
            # Tensor-wide scales and input-channel scales are shared by fused outputs.
            split = start is not None and suffix == "weight_scale" and math.prod(shape) > 1
            tensor(key, target + "." + suffix, start if split else None, count if split else None)

    linear("img_in", "x_embedder")
    linear("txt_in", "context_embedder")
    for source, target in (
        ("time_in", "timestep_embedder"),
        ("vector_in", "text_embedder"),
        ("guidance_in", "guidance_embedder"),
    ):
        linear(source + ".in_layer", "time_text_embed." + target + ".linear_1")
        linear(source + ".out_layer", "time_text_embed." + target + ".linear_2")
    for i in range(config["num_layers"]):
        source, target = f"double_blocks.{i}", f"transformer_blocks.{i}"
        for stream, context in (("img", False), ("txt", True)):
            linear(source + f".{stream}_mod.lin", target + (".norm1_context.linear" if context else ".norm1.linear"))
            for j, name in enumerate(
                ("add_q_proj", "add_k_proj", "add_v_proj") if context else ("to_q", "to_k", "to_v")
            ):
                linear(source + f".{stream}_attn.qkv", target + ".attn." + name, j * dim, dim)
            for query, short in (("query", "q"), ("key", "k")):
                tensor(
                    source + f".{stream}_attn.norm.{query}_norm.scale",
                    target + ".attn.norm_" + ("added_" if context else "") + short + ".weight",
                )
            linear(source + f".{stream}_attn.proj", target + (".attn.to_add_out" if context else ".attn.to_out.0"))
            ff = ".ff_context" if context else ".ff"
            linear(source + f".{stream}_mlp.0", target + ff + ".net.0.proj")
            linear(source + f".{stream}_mlp.2", target + ff + ".net.2")
    for i in range(config["num_single_layers"]):
        source, target = f"single_blocks.{i}", f"single_transformer_blocks.{i}"
        linear(source + ".modulation.lin", target + ".norm.linear")
        for j, name in enumerate(("to_q", "to_k", "to_v")):
            linear(source + ".linear1", target + ".attn." + name, j * dim, dim)
        rows = entries[source + ".linear1.weight"]["shape"][0]
        linear(source + ".linear1", target + ".proj_mlp", 3 * dim, rows - 3 * dim)
        linear(source + ".linear2", target + ".proj_out")
        for query, short in (("query", "q"), ("key", "k")):
            tensor(source + f".norm.{query}_norm.scale", target + ".attn.norm_" + short + ".weight")
    linear("final_layer.linear", "proj_out")
    linear("final_layer.adaLN_modulation.1", "norm_out.linear")
    return aliases
