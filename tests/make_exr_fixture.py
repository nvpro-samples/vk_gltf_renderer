#!/usr/bin/env python3
"""Write a small uncompressed OpenEXR, for testing the environment loader.

The repository has no EXR encoder outside the renderer itself, and generating the fixture with the
same library that reads it would test very little. This writes the file format directly instead --
uncompressed scanline RGBA float, the simplest thing a conforming reader must accept -- so a
failure points at our loader rather than at a shared bug.

Deliberately minimal: no compression, no tiles, no deep data, no custom attributes. Enough to be a
valid EXR and nothing more.

    python tests/make_exr_fixture.py out.exr --width 64 --height 32

The default image is a lat-long gradient with distinctly different channels, so a decoder that
swaps or drops one is obvious in the result rather than subtly wrong. EXR stores channels in
alphabetical order (A, B, G, R), which is exactly the kind of detail a hand-rolled reader gets
backwards -- the asymmetric content is what catches it.
"""

import argparse
import struct


def build_exr(width, height, pixels):
    """pixels: flat list of RGBA floats, row-major, length width*height*4. Returns EXR bytes."""
    out = bytearray()
    out += struct.pack('<I', 0x01312F76)      # magic
    out += struct.pack('<I', 2)               # version 2, scanline, single part

    def attr(name, typ, payload):
        return name.encode() + b'\0' + typ.encode() + b'\0' + struct.pack('<i', len(payload)) + payload

    # Channels, alphabetical as the spec requires: A, B, G, R. Each is FLOAT (pixelType 2),
    # pLinear 0, 3 reserved bytes, xSampling 1, ySampling 1.
    chlist = bytearray()
    for name in ('A', 'B', 'G', 'R'):
        chlist += name.encode() + b'\0' + struct.pack('<i', 2) + struct.pack('<B', 0) + b'\0\0\0' \
                  + struct.pack('<ii', 1, 1)
    chlist += b'\0'                            # end of channel list

    out += attr('channels', 'chlist', bytes(chlist))
    out += attr('compression', 'compression', struct.pack('<B', 0))          # NO_COMPRESSION
    out += attr('dataWindow', 'box2i', struct.pack('<iiii', 0, 0, width - 1, height - 1))
    out += attr('displayWindow', 'box2i', struct.pack('<iiii', 0, 0, width - 1, height - 1))
    out += attr('lineOrder', 'lineOrder', struct.pack('<B', 0))              # INCREASING_Y
    out += attr('pixelAspectRatio', 'float', struct.pack('<f', 1.0))
    out += attr('screenWindowCenter', 'v2f', struct.pack('<ff', 0.0, 0.0))
    out += attr('screenWindowWidth', 'float', struct.pack('<f', 1.0))
    out += b'\0'                               # end of header

    # Offset table: one absolute file offset per scanline, so it can only be filled in once the
    # header length is known and each block's size is fixed (true here: no compression).
    offset_table_pos = len(out)
    out += b'\0' * (8 * height)

    row_bytes = width * 4 * 4                  # 4 channels * 4 bytes, one row
    offsets = []
    for y in range(height):
        offsets.append(len(out))
        out += struct.pack('<ii', y, row_bytes)
        for c in (3, 2, 1, 0):                 # A, B, G, R from an RGBA source
            row = bytearray()
            for x in range(width):
                row += struct.pack('<f', pixels[(y * width + x) * 4 + c])
            out += row

    for i, off in enumerate(offsets):
        struct.pack_into('<Q', out, offset_table_pos + 8 * i, off)
    return bytes(out)


def default_pixels(width, height):
    """A lat-long-ish gradient: red rises with u, green with v, blue constant, alpha 1."""
    px = []
    for y in range(height):
        v = y / max(1, height - 1)
        for x in range(width):
            u = x / max(1, width - 1)
            px += [2.0 * u, 1.5 * v, 0.25, 1.0]     # >1 values: it is an HDR format, use the range
    return px


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('output')
    ap.add_argument('--width', type=int, default=64)
    ap.add_argument('--height', type=int, default=32)
    args = ap.parse_args()
    if args.width <= 0 or args.height <= 0:
        ap.error('--width and --height must be positive')

    data = build_exr(args.width, args.height, default_pixels(args.width, args.height))
    with open(args.output, 'wb') as f:
        f.write(data)
    print(f'wrote {args.output}: {args.width}x{args.height} RGBA float, {len(data)} bytes')


if __name__ == '__main__':
    main()
