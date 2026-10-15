# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise notebook helper and offloading cells without downloading models."""

import json
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

NOTEBOOK = Path(__file__).parents[1] / "flux1.dev" / "flux_demo.ipynb"


@pytest.fixture
def notebook():
    return json.loads(NOTEBOOK.read_text())


def tagged_cell(notebook, tag):
    cells = [cell for cell in notebook["cells"] if tag in cell.get("metadata", {}).get("tags", [])]
    assert len(cells) == 1, f"Expected one notebook cell tagged {tag}"
    return cells[0]


@pytest.fixture
def helper(notebook):
    pipeline = Mock()
    pipeline.load_engines.return_value = {"transformer": 1.0}
    pipeline.infer.return_value = (None, ["image.png"])
    pipeline.timing_data.to_dict.return_value = {"total_e2e_runtime_ms": 2000.0}
    factory = Mock(return_value=pipeline)
    scope = {
        "dataclass": dataclass,
        "Tuple": tuple,
        "FluxPipeline": factory,
        "DEMO_CACHE_DIR": "test-cache",
        "DEMO_CACHE_MODE": "full",
        "DEVICE": "cuda",
        "HF_TOKEN": "test-token",
        "LOG_LEVEL": "INFO",
        "printmd": Mock(),
        "markdown_bold_green_format": str,
        "display_image_from_path": Mock(),
        "Output": nullcontext,
        "torch": SimpleNamespace(cuda=SimpleNamespace(empty_cache=Mock())),
        "gc": SimpleNamespace(collect=Mock()),
    }
    source = "".join(tagged_cell(notebook, "flux-pipeline-helper")["source"])
    exec(compile(source, str(NOTEBOOK), "exec"), scope)
    return scope, factory, pipeline


@pytest.mark.unit
def test_notebook_code_cells_compile(notebook):
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"{NOTEBOOK}:cell{index}", "exec")


@pytest.mark.unit
@pytest.mark.parametrize("precision", ["bf16", "fp8", "fp4"])
@pytest.mark.parametrize("mode,pinned", [("buffer", False), ("buffer", True), ("vmm", False), ("vmm", True)])
def test_notebook_helper_forwards_offload_options(helper, precision, mode, pinned):
    scope, factory, pipeline = helper
    config = scope["GenerationConfig"]()
    actual, times = scope["prepare_pipeline"](
        config,
        precision=precision,
        weight_offload=mode,
        weight_offload_pinned_host=pinned,
        enable_cudagraphs=True,
        enable_runtime_cache=True,
    )
    options = factory.call_args.kwargs
    assert options["weight_offload"] == mode
    assert options["weight_offload_pinned_host"] is pinned
    assert "low_vram" not in options
    assert options["enable_runtime_cache"] is True
    assert options["cuda_graph_strategy"] == "whole_graph_capture"
    pipeline.load_engines.assert_called_once_with(
        transformer_precision=precision, opt_batch_size=1, opt_height=512, opt_width=512, shape_mode="static"
    )
    pipeline.load_resources.assert_called_once_with(batch_size=1, height=512, width=512)
    assert actual is pipeline
    assert times == {"transformer": 1.0}


@pytest.mark.unit
def test_notebook_helper_preserves_existing_defaults(helper):
    scope, factory, _ = helper
    scope["prepare_pipeline"](scope["GenerationConfig"]())
    options = factory.call_args.kwargs
    assert options["weight_offload"] == "none"
    assert options["weight_offload_pinned_host"] is False
    assert "low_vram" not in options
    assert options["cuda_graph_strategy"] == "disabled"
    assert options["enable_runtime_cache"] is True


@pytest.mark.unit
@pytest.mark.parametrize("fail_inference", [False, True])
def test_notebook_offload_example_cleans_up(notebook, helper, fail_inference):
    scope, factory, pipeline = helper
    previous = Mock()
    scope["pipeline"] = previous
    cells = [tagged_cell(notebook, tag) for tag in ("flux-weight-offload-setup", "flux-weight-offload-inference")]
    assert all(output["output_type"] != "error" for cell in cells for output in cell["outputs"])
    exec(compile("".join(cells[0]["source"]), str(NOTEBOOK), "exec"), scope)
    previous.cleanup.assert_called_once()
    scope["printmd"].assert_not_called()
    assert factory.call_args.kwargs["weight_offload"] == "vmm"
    assert factory.call_args.kwargs["weight_offload_pinned_host"] is True
    if fail_inference:
        pipeline.infer.side_effect = RuntimeError("inference failed")
    with pytest.raises(RuntimeError, match="inference failed") if fail_inference else nullcontext():
        exec(compile("".join(cells[1]["source"]), str(NOTEBOOK), "exec"), scope)
    pipeline.infer.assert_called_once_with(
        prompt=scope["config"].prompt,
        batch_size=1,
        height=512,
        width=512,
        seed=12345,
        num_inference_steps=28,
        guidance_scale=3.5,
        save_path="test-cache",
    )
    pipeline.cleanup.assert_called_once()
    assert "pipeline" not in scope
    scope["torch"].cuda.empty_cache.assert_called_once()
    scope["gc"].collect.assert_called_once()
    if not fail_inference:
        scope["display_image_from_path"].assert_called_once_with("image.png")
