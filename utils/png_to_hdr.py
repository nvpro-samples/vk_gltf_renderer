#!/usr/bin/env python3
# Copyright (c) 2023-2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
"""Convert an LDR PNG to a Radiance .hdr, used by comfy_bridge.py when a ComfyUI
HDRI workflow returns PNG bytes for a requested .hdr/.exr output."""

import sys

import numpy as np
from PIL import Image

def write_hdr(filename, img_float):
    """Write a float32 RGB image as Radiance .hdr (RGBE format).

    Vectorized RGBE encoder: per-pixel channel max -> frexp -> shared exponent,
    then scale each channel by mantissa * 256 / max into the [0, 255] byte range.
    Pixels with max < 1e-32 are emitted as (0, 0, 0, 0) per the RGBE convention.
    """
    h, w, _ = img_float.shape
    rgb = np.asarray(img_float, dtype=np.float32)

    mx = rgb.max(axis=2)
    nonzero = mx >= 1e-32

    mantissa, exp = np.frexp(mx)
    scale = np.zeros_like(mx)
    np.divide(mantissa * 256.0, mx, out=scale, where=nonzero)

    rgbe = np.empty((h, w, 4), dtype=np.uint8)
    rgbe[..., 0:3] = np.clip(rgb * scale[..., None], 0.0, 255.0).astype(np.uint8)
    rgbe[..., 3] = np.clip(exp + 128, 0, 255).astype(np.uint8)
    rgbe[~nonzero] = 0

    with open(filename, 'wb') as f:
        header = f"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y {h} +X {w}\n"
        f.write(header.encode('ascii'))
        f.write(rgbe.tobytes())

def convert(input_path, output_path, exposure=3.0):
    img = np.array(Image.open(input_path).convert('RGB')).astype(np.float32) / 255.0

    # Apply inverse tonemapping to expand LDR -> HDR-like range
    # Reinhard inverse: L = l / (1 - l), clamped to avoid infinity
    img = np.clip(img, 0.0, 0.995)
    img = img / (1.0 - img)

    # Apply exposure boost
    img *= exposure

    write_hdr(output_path, img)
    print(f"Wrote {output_path} ({img.shape[1]}x{img.shape[0]})")

if __name__ == "__main__":
    inp = sys.argv[1] if len(sys.argv) > 1 else "input.png"
    out = sys.argv[2] if len(sys.argv) > 2 else "output.hdr"
    exp = float(sys.argv[3]) if len(sys.argv) > 3 else 3.0
    convert(inp, out, exp)