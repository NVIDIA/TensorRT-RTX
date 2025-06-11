# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import base64
import logging
import mimetypes
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests
from IPython.display import HTML, Markdown, display

# Initialize logger for this module
logger = logging.getLogger("rtx_demo.utils.notebook_utils")


def printmd(string: str, newlines: int = 0):
    """Display markdown text in Jupyter notebook."""
    if newlines > 0:
        print("\n" * newlines)

    display(Markdown(string))


def markdown_bold_green_format(string: str) -> str:
    """Format string as bold green markdown."""
    return f"<span style='color: #76B900; font-weight: bold'>{string}</span>"


def print_prompt(prompt: str):
    """Print prompt in markdown."""
    printmd(f"**Prompt**: {prompt}\n")


def display_image_from_path(path: str, width: Optional[int] = None, height: Optional[int] = None):
    """
    Display an image from URL or local path in Jupyter notebook.
    Embeds image data as base64 for HTML export.

    Parameters:
    -----------
    path : str
        URL or local file path of the image
    width : int, optional
        Width in pixels
    height : int, optional
        Height in pixels

    Returns:
    --------
    IPython.display.HTML
        HTML object with embedded image
    """
    try:
        # Determine if path is URL or local file
        parsed = urlparse(str(path))
        is_url = parsed.scheme in ("http", "https")

        if is_url:
            # Handle URL
            response = requests.get(path)
            response.raise_for_status()
            image_data = response.content
            content_type = response.headers.get("content-type", "image/png")
        else:
            # Handle local file
            file_path = Path(path)
            if not file_path.exists():
                raise FileNotFoundError(f"Image file not found: {path}")

            image_data = file_path.read_bytes()
            content_type = mimetypes.guess_type(str(file_path))[0] or "image/png"

        # Encode as base64
        image_base64 = base64.b64encode(image_data).decode("utf-8")

        # Build style
        style_parts = []
        if width:
            style_parts.append(f"width: {width}px")
        if height:
            style_parts.append(f"height: {height}px")
        style = "; ".join(style_parts)

        html = f'<img src="data:{content_type};base64,{image_base64}" style="{style}" />'
        result = HTML(html)

    except Exception as e:
        # Fallback with error message
        error_msg = f"Error loading image: {e}"
        fallback_html = f'<div style="color: red; border: 1px solid red; padding: 10px;">{error_msg}</div>'
        result = HTML(fallback_html)

    display(result)
