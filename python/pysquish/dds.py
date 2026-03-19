import struct
from os import fspath

import numpy as np


# Keep these in sync with squish.h.
kDxt1 = (1 << 0)
kDxt3 = (1 << 1)
kDxt5 = (1 << 2)
kBc4 = (1 << 3)
kBc5 = (1 << 4)


def _fourcc(code: str) -> int:
    if len(code) != 4:
        raise ValueError("FourCC code must be 4 characters.")
    return int.from_bytes(code.encode("ascii"), "little", signed=False)


def _method_from_flags(flags: int) -> int:
    method_mask = (kDxt1 | kDxt3 | kDxt5 | kBc4 | kBc5)
    method = flags & method_mask
    # libsquish default: DXT1.
    return method if method != 0 else kDxt1


def _fourcc_and_blocksize(flags: int):
    method = _method_from_flags(flags)
    if method == kDxt1:
        return _fourcc("DXT1"), 8
    if method == kDxt3:
        return _fourcc("DXT3"), 16
    if method == kDxt5:
        return _fourcc("DXT5"), 16
    if method == kBc4:
        return _fourcc("ATI1"), 8
    if method == kBc5:
        return _fourcc("ATI2"), 16
    # Should be unreachable due to _method_from_flags.
    raise ValueError(f"Unsupported compression method flags: {flags}")


def _expected_data_size(width: int, height: int, flags: int) -> int:
    _, bytes_per_block = _fourcc_and_blocksize(flags)
    blockcount = ((width + 3) // 4) * ((height + 3) // 4)
    return blockcount * bytes_per_block


def write_dds(filepath, rgba_to_blocks_bytes: bytes, width: int, height: int, flags: int) -> None:
    """
    Write a minimal DDS file (no mipmaps) for DXT/BC formats produced by libsquish.

    `rgba_to_blocks_bytes` must be the raw compressed block stream in row-major order.
    """
    filepath = fspath(filepath)
    if width <= 0 or height <= 0:
        raise ValueError("width/height must be positive.")
    if not isinstance(rgba_to_blocks_bytes, (bytes, bytearray, memoryview)):
        raise TypeError("Compressed data must be bytes-like.")

    data = bytes(rgba_to_blocks_bytes)
    expected = _expected_data_size(width, height, flags)
    if len(data) != expected:
        raise ValueError(f"Compressed data size mismatch: got {len(data)}, expected {expected}.")

    fourcc, _bytes_per_block = _fourcc_and_blocksize(flags)

    # DDS constants
    DDSD_CAPS = 0x00000001
    DDSD_HEIGHT = 0x00000002
    DDSD_WIDTH = 0x00000004
    DDSD_PIXELFORMAT = 0x00001000
    DDSD_LINEARSIZE = 0x00080000

    DDSCAPS_TEXTURE = 0x00001000

    DDPF_FOURCC = 0x00000004

    dwSize = 124
    dwFlags = DDSD_CAPS | DDSD_HEIGHT | DDSD_WIDTH | DDSD_PIXELFORMAT | DDSD_LINEARSIZE
    dwHeight = int(height)
    dwWidth = int(width)
    dwPitchOrLinearSize = len(data)
    dwDepth = 0
    dwMipMapCount = 0
    dwReserved1 = [0] * 11

    ddspf_dwSize = 32
    ddspf_dwFlags = DDPF_FOURCC
    ddspf_dwFourCC = fourcc
    ddspf_dwRGBBitCount = 0
    ddspf_dwRBitMask = 0
    ddspf_dwGBitMask = 0
    ddspf_dwBBitMask = 0
    ddspf_dwABitMask = 0

    dwCaps1 = DDSCAPS_TEXTURE
    dwCaps2 = 0
    dwCaps3 = 0
    dwCaps4 = 0
    dwReserved2 = 0

    header = struct.pack(
        "<31I",
        dwSize,
        dwFlags,
        dwHeight,
        dwWidth,
        dwPitchOrLinearSize,
        dwDepth,
        dwMipMapCount,
        *dwReserved1,
        ddspf_dwSize,
        ddspf_dwFlags,
        ddspf_dwFourCC,
        ddspf_dwRGBBitCount,
        ddspf_dwRBitMask,
        ddspf_dwGBitMask,
        ddspf_dwBBitMask,
        ddspf_dwABitMask,
        dwCaps1,
        dwCaps2,
        dwCaps3,
        dwCaps4,
        dwReserved2,
    )

    # Magic "DDS "
    with open(filepath, "wb") as f:
        f.write(b"DDS ")
        f.write(header)
        f.write(data)

